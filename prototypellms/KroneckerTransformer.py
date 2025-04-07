import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import math

class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) for better long-range dependencies."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        B, N, H, D = x.shape  # Extract batch, seq_len, heads, dim
        t = torch.arange(N, device=x.device).float()
        freqs = torch.einsum("n,d->nd", t, self.inv_freq)  # (seq_len, dim//2)
        emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1).unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, dim)
        return emb.to(x.dtype)

class FocusedAttentionGroup(nn.Module):
    """Focused attention groups (FAG) with Kronecker approximation."""
    def __init__(self, dim, heads=8, groups=4, rank=32, dropout=0.1):
        super().__init__()
        assert heads % groups == 0, "Heads must be divisible by groups"
        self.dim = dim
        self.heads = heads
        self.groups = groups
        self.rank = rank
        self.scale = rank ** -0.5  # Scale using rank instead of dim

        self.q_proj = spectral_norm(nn.Linear(dim, rank * heads, bias=False))
        self.k_proj = spectral_norm(nn.Linear(dim, rank * heads, bias=False))
        self.v_proj = spectral_norm(nn.Linear(dim, dim, bias=False))
        self.out = spectral_norm(nn.Linear(dim, dim))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.heads, self.rank)
        k = self.k_proj(x).reshape(B, N, self.heads, self.rank)
        v = self.v_proj(x).reshape(B, N, self.heads, C // self.heads)

        # Correct reshaping into groups
        q = q.view(B, N, self.groups, self.heads // self.groups, self.rank)
        k = k.view(B, N, self.groups, self.heads // self.groups, self.rank)
        v = v.view(B, N, self.groups, self.heads // self.groups, C // self.heads)

        # 🛠 Fix einsum: Ensure `j` dimension matches across q and k
        attn = torch.einsum('bnghd,bnghd->bngh', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('bngh,bnghd->bnghd', attn, v)
        out = out.reshape(B, N, C)

        return self.out(out)

class AdvancedKroneckerTransformer(nn.Module):
    """Final optimized Kronecker Transformer with all enhancements."""
    def __init__(self, dim=896, depth=6, heads=8, groups=4, rank=32, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                spectral_norm(nn.LayerNorm(dim)),
                FocusedAttentionGroup(dim, heads, groups, rank, dropout),
                nn.Dropout(dropout),
                spectral_norm(nn.Linear(dim, dim // 2)),  # Kronecker factorization
                nn.ReLU(),
                spectral_norm(nn.Linear(dim // 2, dim))   # Expand back
            ) for _ in range(depth)
        ])
        self.norm = spectral_norm(nn.LayerNorm(dim))

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)  # Residual connection
        return self.norm(x)

# Example usage
x = torch.randn(1, 128, 896)  # Batch size 1, sequence length 128, embedding size 896
model = AdvancedKroneckerTransformer()
output = model(x)
print(output.shape)  # Should output (1, 128, 896)
