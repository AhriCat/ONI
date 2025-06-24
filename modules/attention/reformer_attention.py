class ReformerAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super(ReformerAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, x, mask=None):
        batch_size, seq_length, dim = x.size()
        assert dim == self.dim, f"Input dimension {dim} does not match layer dimension {self.dim}"

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        if mask is not None:
            attn_mask = mask.unsqueeze(1).repeat(1, seq_length, 1)
        else:
            attn_mask = None

        attn_output, _ = self.multihead_attn(q, k, v, attn_mask=attn_mask)
        attn_output = self.dropout(attn_output)
        output = self.out_proj(attn_output)

        return output + x
