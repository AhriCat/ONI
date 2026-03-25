"""
oni_nlp_feedforward.py
======================
SwiGLU feed-forward network as used in LLaMA 3, Mistral, and Gemma.

SwiGLU formula:
    FFN(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down

This outperforms standard ReLU/GELU FFNs across a range of tasks (Shazeer 2020).
The hidden expansion is 8/3 × d_model (rather than 4×) to compensate for the
extra gate projection while keeping parameter count comparable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .oni_nlp_attention import RMSNorm


class SwiGLUFFN(nn.Module):
    """
    SwiGLU feed-forward block.

    Parameters
    ----------
    hidden_dim  : model dimension
    expand      : expansion ratio (default 8/3 → ~2.67×, matching LLaMA)
    dropout     : applied after down-projection
    bias        : whether linear layers use bias (usually False for large models)
    """

    def __init__(
        self,
        hidden_dim: int,
        expand: float = 8.0 / 3.0,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        # Round up to multiple of 64 for hardware efficiency
        ffn_dim = int(hidden_dim * expand)
        ffn_dim = (ffn_dim + 63) // 64 * 64

        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=bias)
        self.up_proj   = nn.Linear(hidden_dim, ffn_dim, bias=bias)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU gate × up projection
        return self.drop(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class FeedForwardBlock(nn.Module):
    """Pre-RMSNorm + SwiGLUFFN + residual."""

    def __init__(self, config: dict):
        super().__init__()
        hidden_dim = config.get("hidden_dim", 896)
        expand     = config.get("ffn_expand", 8.0 / 3.0)
        dropout    = config.get("dropout", 0.0)

        self.norm = RMSNorm(hidden_dim)
        self.ffn  = SwiGLUFFN(hidden_dim, expand=expand, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))
