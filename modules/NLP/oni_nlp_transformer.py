"""
oni_nlp_transformer.py
======================
Griffin-style Hybrid SSM + GRU + Sliding-Window Attention encoder.

Architecture (per the Griffin paper, DeepMind 2024):
  Layers alternate between:
    - MambaSSMBlock  : selective state-space model for long-range dependencies
    - GatedRecurrentBlock : GRU-inspired linear recurrence for local state
    - SlidingWindowAttnBlock : local attention for fine-grained token mixing

  Default pattern (6 layers): [SSM, GRU, Attn, SSM, GRU, Attn]
  Extendable to arbitrary depth.

Why this beats standard transformers for ONI:
  - Mamba SSM: O(n) in sequence length, captures very long context cheaply
  - GRU gating: persistent state across sequence, good for dialogue & memory
  - Local attention: precise token-level interactions within each window
  - RMSNorm + SwiGLU: matches LLaMA3/Gemma recipe for training stability
  - Already coherent with ONI's existing GRU (world_modeler) and Mamba
    (HypergraphNLP) code
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List

from .oni_nlp_core import OniModule
from .oni_nlp_attention import SelfAttentionBlock, RMSNorm
from .oni_nlp_feedforward import FeedForwardBlock


# ---------------------------------------------------------------------------
# Selective SSM Block (Mamba2-style)
# ---------------------------------------------------------------------------

class MambaSSMBlock(nn.Module):
    """
    Selective State Space Model block.

    Implements the core Mamba SSM with:
    - Input-dependent (selective) B and C projections
    - Input-dependent Δ (step size) with softplus activation
    - Depthwise conv1d for local mixing before SSM
    - SiLU gating on the residual branch

    For ONI's hidden_dim=896:
      d_inner  = 896 × 2  = 1792
      d_state  = 16        (SSM state size, controls memory capacity)
      dt_rank  = ceil(896/16) = 56
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        conv_kernel: int = 4,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = dt_rank or math.ceil(d_model / 16)

        self.norm = RMSNorm(d_model)

        # Input projection: splits into x (signal) and z (gate)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Depthwise conv for local context aggregation
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=conv_kernel,
            bias=True,
            padding=conv_kernel - 1,
            groups=self.d_inner,
        )

        # SSM projections (input-dependent — the "selective" part)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # SSM parameters
        # A: negative real, controls state decay (log-parameterized for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))   # skip connection

        # dt bias initialisation (log-uniform between dt_min and dt_max)
        dt_init = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.act = nn.SiLU()

    def _ssm(self, u: torch.Tensor) -> torch.Tensor:
        """
        Simplified SSM scan using cumulative sum approximation.
        For production, swap in a CUDA parallel scan kernel.
        u: (B, T, d_inner)
        """
        B_sz, T, d = u.shape
        A = -torch.exp(self.A_log.float())                     # (d_inner, d_state)
        x_dbl = self.x_proj(u)                                 # (B, T, dt_rank + 2*d_state)
        dt, B_proj, C_proj = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))                      # (B, T, d_inner)

        # Discretise A: Δ × A → Zero-order hold
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B,T,d,ds)

        # dB: input-dependent B × u
        dB = dt.unsqueeze(-1) * B_proj.unsqueeze(2)            # (B, T, d_inner, d_state)
        dB = dB * u.unsqueeze(-1)

        # Iterative scan over T (replace with parallel scan for speed)
        h = torch.zeros(B_sz, d, self.d_state, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(T):
            h = dA[:, t] * h + dB[:, t]                       # (B, d, ds)
            y_t = (h * C_proj[:, t].unsqueeze(1)).sum(-1)     # (B, d)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)                             # (B, T, d_inner)
        return y + u * self.D

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        B, T, _ = x.shape

        # Split into signal and gating branches
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)                         # each (B, T, d_inner)

        # Conv1d (local mixing)
        x_in = self.act(
            self.conv1d(x_in.transpose(1, 2))[:, :, :T].transpose(1, 2)
        )

        # SSM scan
        y = self._ssm(x_in)

        # Gate with z
        y = y * self.act(z)

        return residual + self.out_proj(y)


# ---------------------------------------------------------------------------
# Gated Recurrent Block (Griffin-style)
# ---------------------------------------------------------------------------

class GatedRecurrentBlock(nn.Module):
    """
    Griffin linear recurrent unit (LRU) with forget/input gating.

    Implements a vectorised recurrence:
        h_t = a_t * h_{t-1} + (1 - a_t) * (r_t ⊙ x_t)
        y_t = h_t ⊙ gelu(x_t @ W_y)

    where a_t (forget) and r_t (reset) are input-dependent gates.
    Bridging GRU expressiveness with linear (parallel) training.
    """

    def __init__(self, d_model: int, expand: int = 1):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.norm = RMSNorm(d_model)

        # Input gates
        self.in_proj = nn.Linear(d_model, self.d_inner * 3, bias=False)

        # Recurrence parameter (log-scale for stability, init near 1 = slow decay)
        self.lambda_log = nn.Parameter(torch.full((self.d_inner,), -0.5))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        B, T, _ = x_norm.shape

        # Project to gates + value
        gate_a, gate_r, v = self.in_proj(x_norm).chunk(3, dim=-1)  # (B, T, d_inner)
        a = torch.sigmoid(gate_a)          # forget gate
        r = torch.sigmoid(gate_r)          # reset gate
        v = F.gelu(v)

        # Bounded log-decay parameter (always in (0, 1))
        lam = torch.sigmoid(self.lambda_log)  # (d_inner,)

        # Parallel-friendly recurrence scan (association scan)
        # h_t = a_t * h_{t-1} + (1 - a_t) * (r_t * v_t)
        h = torch.zeros(B, self.d_inner, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(T):
            h = a[:, t] * h + (1 - a[:, t]) * (r[:, t] * v[:, t]) * lam
            outs.append(h)
        y = torch.stack(outs, dim=1)       # (B, T, d_inner)

        return residual + self.out_proj(y)


# ---------------------------------------------------------------------------
# Hybrid layer factory
# ---------------------------------------------------------------------------

_LAYER_TYPES = ("ssm", "gru", "attn")


def _make_layer(layer_type: str, config: dict) -> nn.Module:
    """Build one hybrid layer of the given type."""
    hidden_dim = config.get("hidden_dim", 896)
    if layer_type == "ssm":
        return nn.ModuleList([
            MambaSSMBlock(hidden_dim,
                          d_state=config.get("ssm_d_state", 16),
                          expand=config.get("ssm_expand", 2)),
            FeedForwardBlock(config),
        ])
    elif layer_type == "gru":
        return nn.ModuleList([
            GatedRecurrentBlock(hidden_dim),
            FeedForwardBlock(config),
        ])
    else:  # attn
        return nn.ModuleList([
            SelfAttentionBlock(config),
            FeedForwardBlock(config),
        ])


# ---------------------------------------------------------------------------
# Main hybrid encoder
# ---------------------------------------------------------------------------

class HybridSSMTransformer(nn.Module):
    """
    Griffin-style hybrid encoder.

    layer_pattern: list of layer types, e.g. ["ssm", "gru", "attn"] × 2
    Default for 6 layers: ["ssm", "gru", "attn", "ssm", "gru", "attn"]

    Retains the TransformerEncoder interface so it's a drop-in replacement.
    """

    def __init__(self, config: dict):
        super().__init__()
        hidden_dim = config.get("hidden_dim", 896)
        num_layers = config.get("num_layers", 6)

        # Build pattern: repeat [ssm, gru, attn] to fill num_layers
        base_pattern = ["ssm", "gru", "attn"]
        pattern = []
        while len(pattern) < num_layers:
            pattern.extend(base_pattern)
        pattern = pattern[:num_layers]

        self.layers = nn.ModuleList([
            _make_layer(lt, config) for lt in pattern
        ])
        self.layer_types = pattern
        self.final_norm = RMSNorm(hidden_dim)
        self.initialized = True

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer_pair in self.layers:
            mixer, ffn = layer_pair[0], layer_pair[1]
            # Mixer (SSM / GRU / Attn) — each applies its own pre-norm + residual
            if isinstance(mixer, SelfAttentionBlock):
                x = mixer(x, mask)
            else:
                x = mixer(x)
            # FFN applies its own pre-norm + residual
            x = ffn(x)

        return self.final_norm(x)

    def _get_fallback_output(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x


# ---------------------------------------------------------------------------
# Backward-compatible wrappers (keep old class names so ONI.py doesn't break)
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(OniModule):
    """
    Single hybrid layer — SSM + FFN by default.
    Kept for backward compatibility with any code that instantiates this class.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        hidden_dim = config.get("hidden_dim", 896)
        self._ssm = MambaSSMBlock(hidden_dim)
        self._ffn = FeedForwardBlock(config)
        self.initialized = True

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._ffn(self._ssm(x))

    def _get_fallback_output(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x


class TransformerEncoder(OniModule):
    """
    Multi-layer hybrid encoder.
    Delegates to HybridSSMTransformer for all the real work.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._hybrid = HybridSSMTransformer(config)
        self.initialized = True

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._hybrid(x, mask)

    def _get_fallback_output(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x
