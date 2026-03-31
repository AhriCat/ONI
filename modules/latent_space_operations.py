"""
modules/latent_space_operations.py
====================================
MPAD — Multiprocess Active Routed Continuous Diffuser (Any output)

Upgrades ContinuousMultiModalDiffuser with:
  Multiprocess  : each modality has its own independent flow-matching process
                  (encoder + velocity net + decoder), not a shared one
  Active Routed : a learned sparse router fires only top-k modality processes
                  per forward pass — differentiable during training,
                  hard top-k at inference
  Continuous    : always active in the forward pass; not an on-demand module
  Any output    : text, vision, audio, video, robotics — extensible

Flow matching
  L    = ||v_θ(x_t, t, cond) − (x_1 − x_0)||²
  x_t  = (1−t)·x_0 + t·x_1
  sample via Euler ODE: x_{i+1} = x_i + Δt · v_θ(x_i, t_i, cond)

Backward-compat aliases
  FatDiffuser                = MPAD
  ContinuousMultiModalDiffuser = MPAD
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

MODALITIES = ["text", "vision", "audio", "video", "robotics"]

# Input dimensionality for each modality (raw feature space)
_MODALITY_IN_DIM: Dict[str, int] = {
    "text":     896,
    "vision":   512,
    "audio":    128,
    "video":    512,
    "robotics":  64,
}
_LATENT_DIM = 256   # shared latent dimensionality across all modalities


# ─── Shared primitives ───────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class AdaLN(nn.Module):
    """Adaptive layer-norm: scale + shift from a context vector."""

    def __init__(self, dim: int, ctx_dim: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.proj = nn.Linear(ctx_dim, 2 * dim, bias=True)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True
        else:
            squeeze = False
        shift, scale = self.proj(ctx).unsqueeze(1).chunk(2, dim=-1)
        out = self.norm(x) * (1.0 + scale) + shift
        return out.squeeze(1) if squeeze else out


class VelocityBlock(nn.Module):
    """DiT-style transformer block for velocity estimation."""

    def __init__(self, dim: int, ctx_dim: int, num_heads: int = 4):
        super().__init__()
        self.adaln1 = AdaLN(dim, ctx_dim)
        self.attn   = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0.0)
        self.adaln2 = AdaLN(dim, ctx_dim)
        self.ff     = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        nx = self.adaln1(x, ctx)
        x  = x + self.attn(nx, nx, nx)[0]
        x  = x + self.ff(self.adaln2(x, ctx))
        return x.squeeze(1) if x.size(1) == 1 else x


class VelocityNet(nn.Module):
    """Per-modality velocity network (small DiT backbone)."""

    def __init__(self, latent_dim: int, ctx_dim: int, num_layers: int = 3):
        super().__init__()
        n_heads = max(1, latent_dim // 32)
        self.time_emb = nn.Sequential(
            nn.Linear(1, ctx_dim), nn.SiLU(), nn.Linear(ctx_dim, ctx_dim)
        )
        self.blocks = nn.ModuleList(
            [VelocityBlock(latent_dim, ctx_dim, n_heads) for _ in range(num_layers)]
        )
        self.out = nn.Linear(latent_dim, latent_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.time_emb(t.unsqueeze(-1))
        ctx   = cond + t_emb
        for blk in self.blocks:
            x_t = blk(x_t, ctx)
        return self.out(x_t)


# ─── Per-modality flow process ────────────────────────────────────────────────

class ModalityProcess(nn.Module):
    """Independent flow-matching pipeline for a single modality."""

    def __init__(self, modality: str, hidden_dim: int = 896):
        super().__init__()
        in_dim = _MODALITY_IN_DIM[modality]
        self.in_dim     = in_dim
        self.latent_dim = _LATENT_DIM

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, _LATENT_DIM * 2), nn.SiLU(),
            nn.Linear(_LATENT_DIM * 2, _LATENT_DIM),
        )
        self.velocity_net = VelocityNet(
            latent_dim=_LATENT_DIM,
            ctx_dim=hidden_dim,
            num_layers=3,
        )
        self.decoder = nn.Sequential(
            nn.Linear(_LATENT_DIM, _LATENT_DIM * 2), nn.SiLU(),
            nn.Linear(_LATENT_DIM * 2, in_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            B, L, D = x.shape
            return self.encoder(x.view(B * L, D)).view(B, L, _LATENT_DIM)
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 3:
            B, L, D = z.shape
            return self.decoder(z.view(B * L, D)).view(B, L, self.in_dim)
        return self.decoder(z)

    def flow_loss(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        B   = x0.size(0)
        z0  = self.encode(x0)
        z1  = self.encode(x1)
        t   = torch.rand(B, device=x0.device)
        z_t = (1.0 - t.view(B, 1)) * z0 + t.view(B, 1) * z1
        v   = self.velocity_net(z_t, t, cond)
        return F.mse_loss(v, z1 - z0)

    def sample(
        self,
        z0: torch.Tensor,
        cond: torch.Tensor,
        steps: int = 10,
    ) -> torch.Tensor:
        z  = z0
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((z.size(0),), i * dt, device=z.device)
            v = self.velocity_net(z, t, cond)
            z = z + dt * v
        return z


# ─── MPAD Router ─────────────────────────────────────────────────────────────

class MPADRouter(nn.Module):
    """Sparse learned router: selects top-k modality processes per forward pass."""

    def __init__(self, hidden_dim: int, n_modalities: int, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_modalities),
        )

    def forward(
        self,
        x: torch.Tensor,
        training: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = torch.softmax(self.gate(x), dim=-1)
        top_k_vals, top_k_idx = probs.topk(self.top_k, dim=-1)
        if training:
            return probs, top_k_idx
        sparse = torch.zeros_like(probs).scatter(-1, top_k_idx, top_k_vals)
        return sparse, top_k_idx


# ─── MPAD ────────────────────────────────────────────────────────────────────

class MPAD(nn.Module):
    """
    Multiprocess Active Routed Continuous Diffuser — any output.

    Usage
    -----
    mpad = MPAD(hidden_dim=896)

    # Training loss
    loss = mpad.flow_loss(x0, x1, modality='text', context=ctx)

    # Generate a sample
    out = mpad.sample(context=ctx, modality='vision', steps=10)

    # Forward (returns latent + reconstruction)
    latent, recon = mpad(x, modality='audio', context=ctx)

    # Query which modalities would fire
    active = mpad.route(ctx)   # e.g. [['text', 'vision']]
    """

    def __init__(
        self,
        hidden_dim: int = 896,
        modalities: Optional[List[str]] = None,
        top_k_routes: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.modalities = modalities or MODALITIES

        self.processes = nn.ModuleDict(
            {m: ModalityProcess(m, hidden_dim=hidden_dim) for m in self.modalities}
        )

        self.ctx_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.router   = MPADRouter(hidden_dim, len(self.modalities), top_k=top_k_routes)
        self._mod_idx = {m: i for i, m in enumerate(self.modalities)}

    def _process(self, modality: str) -> ModalityProcess:
        if modality not in self.processes:
            raise ValueError(
                f"Unknown modality '{modality}'. Available: {self.modalities}"
            )
        return self.processes[modality]

    def _build_cond(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor],
        modality: str,
    ) -> torch.Tensor:
        if context is not None:
            return self.ctx_encoder(context)
        flat = x.reshape(x.size(0), -1)
        if flat.size(-1) < self.hidden_dim:
            flat = F.pad(flat, (0, self.hidden_dim - flat.size(-1)))
        else:
            flat = flat[..., : self.hidden_dim]
        return self.ctx_encoder(flat)

    def forward(
        self,
        x: torch.Tensor,
        modality: str,
        context: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        proc   = self._process(modality)
        cond   = self._build_cond(x, context, modality)
        latent = proc.encode(x)
        weights, _ = self.router(cond, training=training)

        mod_w = weights[:, self._mod_idx[modality]].view(-1, 1)
        noise = torch.randn_like(latent)
        z0    = (1.0 - mod_w) * noise + mod_w * latent

        recon_z = proc.sample(z0, cond, steps=4 if training else 10)
        recon   = proc.decode(recon_z)

        lat_out = latent.squeeze(1) if (latent.dim() == 3 and latent.size(1) == 1) else latent
        return lat_out, recon

    def flow_loss(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        modality: str,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cond = self._build_cond(x1, context, modality)
        return self._process(modality).flow_loss(x0, x1, cond)

    def sample(
        self,
        context: torch.Tensor,
        modality: str,
        steps: int = 10,
        batch_size: int = 1,
    ) -> torch.Tensor:
        proc = self._process(modality)
        cond = self.ctx_encoder(context)
        z0   = torch.randn(batch_size, proc.latent_dim, device=context.device)
        z1   = proc.sample(z0, cond, steps=steps)
        return proc.decode(z1)

    def route(self, context: torch.Tensor) -> List[List[str]]:
        """Return which modalities would fire for this context (inference)."""
        cond = self.ctx_encoder(context)
        _, top_k_idx = self.router(cond, training=False)
        return [
            [self.modalities[int(i)] for i in row]
            for row in top_k_idx.tolist()
        ]


# Backward-compat aliases
FatDiffuser                  = MPAD
ContinuousMultiModalDiffuser = MPAD
