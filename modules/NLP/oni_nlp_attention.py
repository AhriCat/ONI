"""
oni_nlp_attention.py
====================
Sliding-window local attention with Rotary Position Embeddings (RoPE).

Design choices vs. standard multi-head attention:
  - O(n * w) instead of O(n²) — window_size w defaults to 256
  - RoPE applied inside each window for relative position awareness
  - Grouped-Query Attention (GQA) option: num_kv_heads < num_heads
    reduces KV cache and computation without hurting quality
  - Flash-attention compatible layout (B, H, T, D)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# RMSNorm (faster and more stable than LayerNorm for deep models)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


# ---------------------------------------------------------------------------
# Rotary Position Embeddings
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors. cos/sin: (T, head_dim)."""
    cos = cos.unsqueeze(0).unsqueeze(0)   # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)           # (T, D/2)
        emb = torch.cat([freqs, freqs], dim=-1)          # (T, D)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len * 2)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


# ---------------------------------------------------------------------------
# Sliding-Window Grouped-Query Attention
# ---------------------------------------------------------------------------

class SlidingWindowAttention(nn.Module):
    """
    Local attention restricted to a sliding window of size `window_size`.

    Each token attends to the `window_size` tokens centred on it
    (causal: attends to itself + window_size-1 preceding tokens).

    Uses Grouped-Query Attention: `num_kv_heads` ≤ `num_heads`.
    Every group of `num_heads // num_kv_heads` query heads shares one KV pair.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        window_size: int = 256,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        assert num_heads % self.num_kv_heads == 0
        self.kv_groups = num_heads // self.num_kv_heads
        self.head_dim = hidden_dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        self.rotary = RotaryEmbedding(self.head_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        W = self.window_size

        cos, sin = self.rotary(T)

        # Project
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        Q, K = apply_rope(Q, K, cos, sin)

        # Expand KV groups if using GQA
        if self.kv_groups > 1:
            K = K.repeat_interleave(self.kv_groups, dim=1)
            V = V.repeat_interleave(self.kv_groups, dim=1)

        # Chunked sliding window via unfold — efficient for large T
        # Pad K, V so every window has full width
        pad = W - 1
        K_pad = F.pad(K, (0, 0, pad, 0))   # (B, H, T+pad, D)
        V_pad = F.pad(V, (0, 0, pad, 0))

        # Unfold into windows: (B, H, T, W, D)
        K_win = K_pad.unfold(2, W, 1).transpose(-1, -2)  # (B, H, T, W, D)
        V_win = V_pad.unfold(2, W, 1).transpose(-1, -2)

        # Scores: (B, H, T, 1, D) x (B, H, T, D, W) → (B, H, T, 1, W)
        scores = torch.matmul(
            Q.unsqueeze(3), K_win.transpose(-1, -2)
        ).squeeze(3) * self.scale                          # (B, H, T, W)

        # Causal mask within window (position i can see [i-W+1, i])
        causal_idx = torch.arange(W, device=x.device)
        causal_mask = causal_idx < W                       # all True (causal is implicit via pad)
        scores = scores.masked_fill(~causal_mask.view(1, 1, 1, W), float('-inf'))

        if mask is not None:
            # mask: (B, T) → expand for window dim
            scores = scores.masked_fill(
                mask[:, None, :, None].expand_as(scores) == 0, float('-inf')
            )

        attn = self.attn_drop(F.softmax(scores, dim=-1))  # (B, H, T, W)

        # Output: (B, H, T, W) @ (B, H, T, W, D) → (B, H, T, D)
        out = torch.matmul(attn.unsqueeze(3), V_win).squeeze(3)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Wrapped block (pre-norm + residual) — used by TransformerEncoderLayer
# ---------------------------------------------------------------------------

class SelfAttentionBlock(nn.Module):
    """Pre-RMSNorm + SlidingWindowAttention + residual."""

    def __init__(self, config: dict):
        super().__init__()
        hidden_dim = config.get("hidden_dim", 896)
        num_heads  = config.get("num_heads", 8)
        num_kv_heads = config.get("num_kv_heads", None)
        window_size  = config.get("window_size", 256)
        dropout      = config.get("dropout", 0.0)

        self.norm = RMSNorm(hidden_dim)
        self.attn = SlidingWindowAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            window_size=window_size,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x + self.attn(self.norm(x), mask)
