import torch
import torch.nn as nn
import math
#projected attention *2d vector attention*
class EfficientAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, k=1000):
        super(EfficientAttention, self).__init__()
        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k = k  # Scaling factor for differentiable max approximation

    def forward(self, x, mask=None):
        if len(x.size()) == 2:
            x = x.unsqueeze(0)  # Add batch dimension if missing

        batch_size, seq_len, _ = x.size()

        # Project the inputs to Q, K, V
        qkv = self.qkv_proj(x)  # [batch_size, seq_len, hidden_dim * 3]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is [batch_size, num_heads, seq_len, head_dim]

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch_size, num_heads, seq_len, seq_len]

        # Apply causal mask (lower triangular matrix to prevent attending to future tokens)
        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))  # Lower triangular mask

        # Expand mask to match attention weight dimensions: [batch_size, num_heads, seq_len, seq_len]
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        mask = mask.expand(batch_size, self.num_heads, seq_len, seq_len)  # Expand across batch and heads

        # Mask the future tokens by setting large negative value for masked positions
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        # Apply scaled softmax to approximate differentiable max
        attn_probs = torch.softmax(attn_weights * self.k, dim=-1)

        # Weighted sum of values
        attn_output = torch.matmul(attn_probs, v)  # [batch_size, num_heads, seq_len, head_dim]

        # Reshape the output back to the original size
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)

        # Final linear projection to the output dimension
        out = self.output_proj(attn_output)
        return out
