"""
modules/latent_space_operations.py
====================================
ContinuousMultiModalDiffuser — always-active continuous diffusion path.

Key design decisions
---------------------
1.  **Flow matching** (Lipman et al. 2022; Liu et al. 2022 — Rectified Flow)
    instead of DDPM. The training loss is simply:
        L = ||v_θ(x_t, t, cond) - (x_1 - x_0)||²
    where x_t = (1-t)*x_0 + t*x_1 is the straight-line interpolant.
    Sampling uses Euler ODE: x_{t+1} = x_t + v_θ * Δt (10-20 steps suffice).

2.  **Always-active path** — not an optional superposition add-on.
    The forward() method always runs. ONI.py routes every token/frame/joint
    vector through the diffusion path and combines with the AR path via a
    learned gating scalar per modality.

3.  **Modal-specific encoders and decoders**:
    - Text:     token embeddings → hidden latent (vocab projection back)
    - Vision:   [B, C, H, W] pixel patches → hidden latent
    - Audio:    [B, T, mel] spectrogram frames → hidden latent
    - Video:    [B, T, C, H, W] frame sequence → hidden latent (temporal + spatial)
    - Robotics: [B, T, joint_dim] joint trajectories → hidden latent (continuous)

4.  **Conditioning**: all modalities can condition on a context vector so the
    diffusion path is aware of the rest of ONI's state.

5.  **Backward-compat alias**: `FatDiffuser = ContinuousMultiModalDiffuser`
    so the existing ONI.py import `from modules.latent_space_operations import FatDiffuser`
    continues to work without modification.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Literal

ModalityType = Literal["text", "vision", "audio", "video", "robotics"]


# ============================================================================
# Shared utilities
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


class SinusoidalTimestepEmbedding(nn.Module):
    """
    Embeds scalar t ∈ [0, 1] into a dense vector using sinusoidal features
    then projects to `out_dim` with two linear layers + SiLU.
    """

    def __init__(self, out_dim: int, base_dim: int = 256):
        super().__init__()
        half = base_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half).float() / half)
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(
            nn.Linear(base_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) → sinusoidal features → projection
        t = t.float().unsqueeze(-1) * self.freqs.unsqueeze(0) * 2 * math.pi
        emb = torch.cat([t.sin(), t.cos()], dim=-1)   # (B, base_dim)
        return self.mlp(emb)


# ============================================================================
# Modal encoders — project raw modality tensors into a common hidden space
# ============================================================================

class TextModalEncoder(nn.Module):
    """Token IDs or embeddings → latent sequence."""

    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.proj  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm  = RMSNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) int token ids  OR  (B, T, hidden_dim) pre-embedded
        if x.dtype in (torch.long, torch.int):
            x = self.embed(x)
        return self.norm(self.proj(x))   # (B, T, H)


class VisionModalEncoder(nn.Module):
    """Image patches → latent sequence (ViT-style patch embedding)."""

    def __init__(self, hidden_dim: int, patch_size: int = 16, in_channels: int = 3):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(
            in_channels, hidden_dim,
            kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.norm = RMSNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.patch_embed(x)           # (B, H, h, w)
        B, H, h, w = x.shape
        x = x.flatten(2).transpose(1, 2) # (B, n_patches, H)
        return self.norm(x)


class AudioModalEncoder(nn.Module):
    """Mel-spectrogram frames → latent sequence."""

    def __init__(self, hidden_dim: int, mel_bins: int = 80):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(mel_bins, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = RMSNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, mel_bins)
        return self.norm(self.proj(x))


class VideoModalEncoder(nn.Module):
    """Video frames → latent sequence using spatial patch embed + temporal pool."""

    def __init__(self, hidden_dim: int, patch_size: int = 16, in_channels: int = 3):
        super().__init__()
        self.spatial_enc = VisionModalEncoder(hidden_dim, patch_size, in_channels)
        # Temporal mixing across frames
        self.temporal_mix = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1,
                                       groups=hidden_dim, bias=False)
        self.norm = RMSNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T_frames, C, H, W)
        B, T_f, C, H, W = x.shape
        # Encode each frame
        x_flat = x.view(B * T_f, C, H, W)
        patches = self.spatial_enc(x_flat)            # (B*T_f, n_patches, H)
        n_p, hd = patches.shape[1], patches.shape[2]
        patches = patches.view(B, T_f * n_p, hd)
        # Temporal mixing over the sequence axis
        patches = self.temporal_mix(patches.transpose(1, 2)).transpose(1, 2)
        return self.norm(patches)


class RoboticsModalEncoder(nn.Module):
    """Joint trajectories → latent sequence (continuous, suitable for diffusion)."""

    def __init__(self, hidden_dim: int, joint_dim: int = 72):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = RMSNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T_steps, joint_dim)
        return self.norm(self.proj(x))


# ============================================================================
# Modal decoders
# ============================================================================

class TextModalDecoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)   # (B, T, vocab_size) logits


class VisionModalDecoder(nn.Module):
    def __init__(self, hidden_dim: int, patch_size: int = 16, in_channels: int = 3,
                 img_size: int = 256):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Linear(hidden_dim, patch_size * patch_size * in_channels)
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.img_size = img_size
        self.n_patches_side = img_size // patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, H = x.shape
        p = self.patch_size
        s = self.n_patches_side
        x = self.proj(x)                            # (B, N, p*p*C)
        x = x.view(B, s, s, p, p, self.in_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, self.in_channels, s * p, s * p)   # (B, C, H, W)
        return x


class AudioModalDecoder(nn.Module):
    def __init__(self, hidden_dim: int, mel_bins: int = 80):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, mel_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)   # (B, T, mel_bins)


class RoboticsModalDecoder(nn.Module):
    def __init__(self, hidden_dim: int, joint_dim: int = 72):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, joint_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)   # (B, T, joint_dim)


# ============================================================================
# Velocity Network — predicts the flow field v_θ(x_t, t, cond)
# ============================================================================

class VelocityBlock(nn.Module):
    """One transformer-style block inside the velocity network."""

    def __init__(self, hidden_dim: int, n_heads: int = 8, ffn_expand: float = 2.67):
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim)
        self.norm2 = RMSNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True, bias=False)
        ffn_dim = int(hidden_dim * ffn_expand)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim, bias=False),
            nn.SiLU(),
            nn.Linear(ffn_dim, hidden_dim, bias=False),
        )
        # Time-conditioning via AdaLN-style scale+shift
        self.t_proj = nn.Linear(hidden_dim, hidden_dim * 2, bias=True)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # t_emb: (B, hidden_dim) → scale + shift per token
        scale, shift = self.t_proj(t_emb).chunk(2, dim=-1)
        scale = scale.unsqueeze(1)   # (B, 1, H)
        shift = shift.unsqueeze(1)

        # Self-attention with AdaLN
        xn = self.norm1(x) * (1 + scale) + shift
        attn_out, _ = self.attn(xn, xn, xn, need_weights=False)
        x = x + attn_out

        # FFN with AdaLN
        xn = self.norm2(x) * (1 + scale) + shift
        x = x + self.ffn(xn)
        return x


class FlowVelocityNet(nn.Module):
    """
    Predicts the velocity field for flow matching:
        v_θ(x_t, t, cond) ≈ x_1 - x_0

    Architecture: DiT (Diffusion Transformer) style — N blocks of
    self-attention + SwiGLU, conditioned on time embedding + optional
    context vector.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_layers: int = 6,
        n_heads: int = 8,
    ):
        super().__init__()
        self.t_embed = SinusoidalTimestepEmbedding(hidden_dim)
        self.cond_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.blocks = nn.ModuleList([
            VelocityBlock(hidden_dim, n_heads) for _ in range(n_layers)
        ])
        self.out_norm = RMSNorm(hidden_dim)

    def forward(
        self,
        x_t: torch.Tensor,          # (B, T, H) noisy latent
        t: torch.Tensor,             # (B,) timestep in [0, 1]
        cond: Optional[torch.Tensor] = None,  # (B, H) context
    ) -> torch.Tensor:
        t_emb = self.t_embed(t)      # (B, H)
        if cond is not None:
            t_emb = t_emb + self.cond_proj(cond)

        x = x_t
        for block in self.blocks:
            x = block(x, t_emb)

        return self.out_norm(x)      # predicted velocity (B, T, H)


# ============================================================================
# Main: ContinuousMultiModalDiffuser
# ============================================================================

class ContinuousMultiModalDiffuser(nn.Module):
    """
    Always-active continuous diffusion path for ONI.

    Usage in ONI.py forward():
        latent, recon = self.imagination_diffuser(
            x=token_embeddings,   # (B, T, H) — already in latent space
            modality="text",
            context=fused_output,
            training=self.training,
        )

    For robotics:
        latent, trajectory = self.imagination_diffuser(
            x=joint_tensor,       # (B, T, joint_dim)
            modality="robotics",
        )

    Returns:
        latent   : (B, T, hidden_dim) — the denoised latent (always)
        recon    : modality-specific reconstructed output (logits / pixels / etc.)
    """

    def __init__(
        self,
        vocab_size: int = 300000,
        hidden_dim: int = 896,
        num_heads: int = 8,
        n_velocity_layers: int = 6,
        n_flow_steps: int = 10,       # ODE steps at inference
        patch_size: int = 16,
        mel_bins: int = 80,
        joint_dim: int = 72,
        timesteps: int = 30,          # kept for API compat; not used in flow matching
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_flow_steps = n_flow_steps

        # --- Modal encoders ---
        self.encoders = nn.ModuleDict({
            "text":     TextModalEncoder(vocab_size, hidden_dim),
            "vision":   VisionModalEncoder(hidden_dim, patch_size),
            "audio":    AudioModalEncoder(hidden_dim, mel_bins),
            "video":    VideoModalEncoder(hidden_dim, patch_size),
            "robotics": RoboticsModalEncoder(hidden_dim, joint_dim),
        })

        # --- Modal decoders ---
        self.decoders = nn.ModuleDict({
            "text":     TextModalDecoder(vocab_size, hidden_dim),
            "vision":   VisionModalDecoder(hidden_dim, patch_size),
            "audio":    AudioModalDecoder(hidden_dim, mel_bins),
            "video":    AudioModalDecoder(hidden_dim, mel_bins),  # reuse audio decoder shape
            "robotics": RoboticsModalDecoder(hidden_dim, joint_dim),
        })

        # --- Velocity network (shared across modalities, conditioned on modality token) ---
        self.velocity_net = FlowVelocityNet(hidden_dim, n_velocity_layers, num_heads)

        # Learned modality embedding added to the velocity network input
        self.modality_embed = nn.Embedding(len(self.encoders), hidden_dim)
        self._modality_ids = {m: i for i, m in enumerate(self.encoders)}

        # --- AR ↔ Diffusion gate (learned per modality) ---
        self.gate = nn.Embedding(len(self.encoders), 1)
        nn.init.constant_(self.gate.weight, 0.0)  # start at 0.5 sigmoid

        # --- Input projection when x is already in latent space (hidden_dim) ---
        self.latent_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.latent_norm = RMSNorm(hidden_dim)

    # ----------------------------------------------------------------
    # Encoding (raw modality → latent)
    # ----------------------------------------------------------------

    def encode(self, x: torch.Tensor, modality: ModalityType) -> torch.Tensor:
        """Encode raw modal tensor into the shared latent space."""
        enc = self.encoders[modality]
        return enc(x)

    # ----------------------------------------------------------------
    # Flow matching training signal
    # ----------------------------------------------------------------

    def flow_loss(
        self,
        x_1: torch.Tensor,            # (B, T, H) clean latent
        cond: Optional[torch.Tensor], # (B, H) context
        modality: ModalityType,
    ) -> torch.Tensor:
        """
        Rectified-flow matching loss.
        Sample t ~ U[0,1], x_t = (1-t)*x_0 + t*x_1 where x_0 ~ N(0, I).
        Target velocity: v* = x_1 - x_0.
        Loss: ||v_θ(x_t, t, cond) - v*||².
        """
        B, T, H = x_1.shape
        device = x_1.device

        x_0 = torch.randn_like(x_1)
        t = torch.rand(B, device=device)
        t_exp = t[:, None, None]
        x_t = (1 - t_exp) * x_0 + t_exp * x_1

        # Add modality conditioning
        mod_id = torch.tensor(self._modality_ids[modality], device=device).expand(B)
        mod_emb = self.modality_embed(mod_id)                  # (B, H)
        cond_full = mod_emb if cond is None else cond + mod_emb

        v_pred = self.velocity_net(x_t, t, cond_full)
        v_target = x_1 - x_0
        return F.mse_loss(v_pred, v_target)

    # ----------------------------------------------------------------
    # Flow sampling (Euler ODE, n_flow_steps steps)
    # ----------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple,                 # (B, T, H)
        cond: Optional[torch.Tensor],
        modality: ModalityType,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample from the flow by integrating the learned velocity field.
        Euler: x_{t+Δt} = x_t + v_θ(x_t, t) * Δt
        """
        n = steps or self.n_flow_steps
        device = next(self.parameters()).device
        B = shape[0]

        x = torch.randn(shape, device=device)
        mod_id = torch.tensor(self._modality_ids[modality], device=device).expand(B)
        mod_emb = self.modality_embed(mod_id)
        cond_full = mod_emb if cond is None else cond + mod_emb

        dt = 1.0 / n
        for i in range(n):
            t = torch.full((B,), i * dt, device=device)
            v = self.velocity_net(x, t, cond_full)
            x = x + v * dt

        return x

    # ----------------------------------------------------------------
    # Forward — always active
    # ----------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        modality: ModalityType = "text",
        context: Optional[torch.Tensor] = None,   # (B, H) or (B, 1, H) global context
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Always-active forward pass.

        During training:
            - Encode x to latent x_1
            - Compute flow loss (backprop separately via .flow_loss())
            - Return denoised latent + reconstruction

        During inference:
            - Encode → run ODE sampling → decode

        Returns
        -------
        latent : (B, T, hidden_dim)
        recon  : modality-dependent decoded output
        """
        device = x.device if isinstance(x, torch.Tensor) else next(self.parameters()).device
        B = x.shape[0]

        # Collapse context to (B, H)
        if context is not None:
            if context.dim() == 3:
                context = context.mean(1)   # pool sequence dim
            context = context.float()

        # Handle the case where x is already a hidden-dim latent (token embeddings)
        if x.dtype in (torch.long, torch.int) or modality != "text":
            x_1 = self.encode(x, modality)
        else:
            # x is already (B, T, H) float — project into flow space
            x_1 = self.latent_norm(self.latent_proj(x.float()))

        if training:
            # During training: just return the clean latent + reconstruction
            # (flow loss is computed separately and summed into ONI's total loss)
            latent = x_1
        else:
            # Inference: denoise from noise → clean latent via flow
            shape = x_1.shape
            latent = self.sample(shape, context, modality)

        # Gating: blend diffusion latent with the incoming representation
        mod_id = torch.tensor(self._modality_ids[modality], device=device).expand(B)
        gate_val = torch.sigmoid(self.gate(mod_id)).unsqueeze(-1)   # (B, 1, 1)
        latent = gate_val * latent + (1 - gate_val) * x_1

        # Decode back to modality space
        dec = self.decoders[modality]
        recon = dec(latent)

        return latent, recon


# Backward-compatible alias — ONI.py does `from modules.latent_space_operations import FatDiffuser`
FatDiffuser = ContinuousMultiModalDiffuser
