import torch
import torch.nn as nn
import math

class SRNormalization(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        variance = x.var(-1, keepdim=True, unbiased=False)
        sr_inv = self.fast_inv_sqrt(variance + self.eps)
        x = (x - mean) * sr_inv
        return self.weight * x + self.bias

    def fast_inv_sqrt(self, x):
        threehalfs = 1.5
        x2 = x * 0.5
        y = x.clone().detach().float().cpu()
        i = y.view(dtype=torch.int32)
        i = 0x5f3759df - (i >> 1)
        y = i.view(dtype=torch.float32)
        y = y * (threehalfs - (x2 * y * y))
        return y

class QuantumAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.sr_norm = SRNormalization(hidden_size)

    def _split_heads(self, x):
        batch_size, seq_length, hidden_size = x.size()
        new_shape = (batch_size, seq_length, self.num_heads, self.head_dim)
        x = x.view(new_shape).permute(0, 2, 1, 3)
        return x

    def _merge_heads(self, x):
        batch_size, num_heads, seq_length, head_dim = x.size()
        new_shape = (batch_size, seq_length, num_heads * head_dim)
        x = x.permute(0, 2, 1, 3).contiguous().view(new_shape)
        return x

    def _attn(self, query, key, value, attention_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights = attn_weights / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Exponential fractalized growth with condensation
        attn_weights = torch.exp(attn_weights) / (torch.sum(torch.exp(attn_weights), dim=-1, keepdim=True) + 1e-6)
        condensed_weights = attn_weights * torch.sigmoid(attn_weights)

        attn_output = torch.matmul(condensed_weights, value)
        return attn_output, attn_weights

    def forward(self, hidden_states, attention_mask=None):
        query = self._split_heads(self.query(hidden_states))
        key = self._split_heads(self.key(hidden_states))
        value = self._split_heads(self.value(hidden_states))

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        attn_output, _ = self._attn(query, key, value, attention_mask)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        attn_output = self.sr_norm(attn_output + hidden_states)  # Residual connection with SR normalization
        
        return attn_output

class Config:
    def __init__(self, hidden_size, num_attention_heads, attn_pdrop, resid_pdrop):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class QuantumMicroTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size=2048, num_heads=64, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        self.encoder_layers = nn.ModuleList([
            QuantumAttentionLayer(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.transformer_encoder = nn.ModuleList(self.encoder_layers)
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1).transpose(0, 1)
        mask = mask.masked_fill(mask == 1, float('-inf')).to(torch.float32)
        return mask

    def forward(self, src, attention_mask=None):
        src = self.embedding(src) * math.sqrt(self.hidden_size)
        src = self.pos_encoder(src)
        
        if attention_mask is not None:
            src_mask = self.generate_square_subsequent_mask(src.size(0)).to(src.device)
            src_mask = src_mask.masked_fill(attention_mask == 0, float('-inf'))
        else:
            src_mask = None

        for layer in self.transformer_encoder:
            src = layer(src, src_mask)
        output = self.linear(src)
        return output

    def generate(self, input_ids, max_length, tokenizer):
        generated = input_ids
        for _ in range(max_length):
            outputs = self.forward(generated)
            next_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
        return generated

# Example usage
vocab_size = 300000
model = QuantumMicroTransformer(vocab_size)

# Ensure model runs on CPU efficiently
device = torch.device('cpu')
model.to(device)

