"""
modules/attention/latent_residual_block.py
==========================================
Transformer-style block combining:
  - LatentAttention       : Perceiver-style bottleneck (global context)
  - ResidualAttention     : Local self-attention with KV-cache support
  - MemoryAugmentedAttention : External / persistent memory read
  - Orthogonal layer scales  : independent gamma per path
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Dict, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Primitives
# ──────────────────────────────────────────────────────────────────────────────

class _RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        norm = x.norm(2, dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return self.weight * x / (norm + self.eps)


# ──────────────────────────────────────────────────────────────────────────────
# LatentAttention  (Perceiver-style two-stage bottleneck)
# ──────────────────────────────────────────────────────────────────────────────

class LatentAttention(nn.Module):
    """
    Two-stage cross-attention through a learned latent bottleneck.

    Stage 1 – compress : latents cross-attend to x  →  enriched latents (B, L, D)
    Stage 2 – expand   : x cross-attends to latents →  context-enriched x (B, T, D)
    """

    def __init__(self, d_model: int, n_latents: int = 32,
                 n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads   = n_heads
        self.head_dim  = d_model // n_heads
        self.scale     = self.head_dim ** -0.5

        self.latents   = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)
        self.norm_x    = _RMSNorm(d_model)
        self.norm_l    = _RMSNorm(d_model)

        # compress projections
        self.cq = nn.Linear(d_model, d_model, bias=False)
        self.ck = nn.Linear(d_model, d_model, bias=False)
        self.cv = nn.Linear(d_model, d_model, bias=False)

        # expand projections
        self.eq       = nn.Linear(d_model, d_model, bias=False)
        self.ek       = nn.Linear(d_model, d_model, bias=False)
        self.ev       = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop     = nn.Dropout(dropout)

    def _attend(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        B, Sq, _ = q.shape
        Sk = k.shape[1]

        def split(t, S):
            return t.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split(q, Sq), split(k, Sk), split(v, Sk)
        w = self.drop(torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1))
        out = torch.matmul(w, v)
        return out.transpose(1, 2).reshape(B, Sq, self.n_heads * self.head_dim)

    def forward(self, x: Tensor,
                return_latents: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        B = x.shape[0]
        x_n = self.norm_x(x)
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        l_n = self.norm_l(latents)

        # Stage 1: latents attend to x (compress)
        enriched = self._attend(self.cq(l_n), self.ck(x_n), self.cv(x_n))  # (B, L, D)

        # Stage 2: x attends to enriched latents (expand)
        out = self.out_proj(
            self._attend(self.eq(x_n), self.ek(enriched), self.ev(enriched))
        )  # (B, T, D)

        return (out, enriched) if return_latents else (out, None)


# ──────────────────────────────────────────────────────────────────────────────
# ResidualAttention  (standard causal self-attention + KV-cache)
# ──────────────────────────────────────────────────────────────────────────────

class ResidualAttention(nn.Module):
    """Causal self-attention with optional KV-cache for autoregressive inference."""

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.scale    = self.head_dim ** -0.5

        self.norm = _RMSNorm(d_model)
        self.q    = nn.Linear(d_model, d_model, bias=False)
        self.k    = nn.Linear(d_model, d_model, bias=False)
        self.v    = nn.Linear(d_model, d_model, bias=False)
        self.out  = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        past_kv: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        B, T, D = x.shape
        x_n = self.norm(x)

        def split(t):
            S = t.shape[1]
            return t.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split(self.q(x_n)), split(self.k(x_n)), split(self.v(x_n))

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        present_kv = (k, v) if use_cache else None

        w = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            w = w + attn_mask
        w = self.drop(torch.softmax(w, dim=-1))
        out = torch.matmul(w, v).transpose(1, 2).reshape(B, T, D)
        return self.out(out), present_kv


# ──────────────────────────────────────────────────────────────────────────────
# MemoryAugmentedAttention
# ──────────────────────────────────────────────────────────────────────────────

class MemoryAugmentedAttention(nn.Module):
    """x attends to a persistent learnable memory bank (or supplied external memory)."""

    def __init__(self, d_model: int, n_heads: int = 8,
                 n_memory_slots: int = 64, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.scale    = self.head_dim ** -0.5

        self.memory = nn.Parameter(torch.randn(n_memory_slots, d_model) * 0.02)
        self.norm   = _RMSNorm(d_model)
        self.q      = nn.Linear(d_model, d_model, bias=False)
        self.k      = nn.Linear(d_model, d_model, bias=False)
        self.v      = nn.Linear(d_model, d_model, bias=False)
        self.out    = nn.Linear(d_model, d_model, bias=False)
        self.drop   = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        external_memory: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        B, T, D = x.shape
        x_n = self.norm(x)
        mem = external_memory if external_memory is not None \
              else self.memory.unsqueeze(0).expand(B, -1, -1)

        M = mem.shape[1]

        def split_q(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        def split_m(t):
            return t.view(B, M, self.n_heads, self.head_dim).transpose(1, 2)

        q = split_q(self.q(x_n))
        k = split_m(self.k(mem))
        v = split_m(self.v(mem))

        w = self.drop(torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1
        ))
        out = torch.matmul(w, v).transpose(1, 2).reshape(B, T, D)
        return self.out(out), mem   # return mem unchanged; caller may update it


# ──────────────────────────────────────────────────────────────────────────────
# LatentResidualBlock
# ──────────────────────────────────────────────────────────────────────────────

class LatentResidualBlock(nn.Module):
    """
    Transformer-style block with latent + residual attention
    and orthogonal hyperconnection layer scales.

    Paths (each scaled by its own learnable gamma):
      1. LatentAttention       – global context via Perceiver bottleneck
      2. ResidualAttention     – local precision self-attention (always runs)
      3. MemoryAugmentedAttention – optional persistent memory read
      4. FFN                   – position-wise feed-forward
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_latents: int = 32,
        d_ff: int = None,
        dropout: float = 0.1,
        use_memory: bool = False,
        n_memory_slots: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        d_ff = d_ff or d_model * 4

        self.latent_attn   = LatentAttention(d_model, n_latents=n_latents,
                                             n_heads=n_heads, dropout=dropout)
        self.residual_attn = ResidualAttention(d_model, n_heads=n_heads,
                                               dropout=dropout)

        self.use_memory = use_memory
        if use_memory:
            self.memory_attn = MemoryAugmentedAttention(
                d_model, n_heads=n_heads,
                n_memory_slots=n_memory_slots, dropout=dropout,
            )

        self.ffn = nn.Sequential(
            _RMSNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # Orthogonal layer scales (independent per path)
        self.gamma_latent   = nn.Parameter(torch.ones(d_model) * 0.1)
        self.gamma_residual = nn.Parameter(torch.ones(d_model) * 0.1)
        self.gamma_memory   = nn.Parameter(torch.ones(d_model) * 0.1) if use_memory else None
        self.gamma_ffn      = nn.Parameter(torch.ones(d_model) * 0.1)

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        memory: Optional[Tensor] = None,
        past_kv: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Args:
            x:         (B, T, D) input tensor
            attn_mask: optional additive attention mask for residual_attn
            memory:    optional external memory tensor for memory_attn
            past_kv:   optional KV-cache tuple (k, v) from a previous step
            use_cache: if True, returns present KV-cache in state dict

        Returns:
            output: (B, T, D)
            state:  dict with optional keys
                    'present_kv', 'memory', 'latents', 'attn_out'
        """
        state: Dict[str, Any] = {}

        # ── 1. Global context via latent bottleneck ───────────────────────────
        latent_out, latents = self.latent_attn(x, return_latents=True)
        state['attn_out'] = latent_out
        x = x + self.gamma_latent * latent_out
        if latents is not None:
            state['latents'] = latents

        # ── 2. Local precision: residual self-attention ───────────────────────
        residual_out, present_kv = self.residual_attn(
            x, attn_mask=attn_mask, past_kv=past_kv, use_cache=use_cache
        )
        x = x + self.gamma_residual * (residual_out - x)
        if present_kv is not None:
            state['present_kv'] = present_kv

        # ── 3. Memory augmentation (optional) ────────────────────────────────
        if self.use_memory:
            mem_out, new_memory = self.memory_attn(x, external_memory=memory)
            x = x + self.gamma_memory * (mem_out - x)
            state['memory'] = new_memory

        # ── 4. FFN ────────────────────────────────────────────────────────────
        x = x + self.gamma_ffn * self.ffn(x)

        return x, state
