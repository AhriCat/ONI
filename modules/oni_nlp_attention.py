"""
Attention mechanisms for ONI NLP
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .oni_nlp_core import OniModule

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                                   mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attn_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear transformation
        output = self.w_o(attn_output)
        return output

class SelfAttentionBlock(OniModule):
    """Self-attention block with residual connections and layer norm"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        self.hidden_dim = config.get("hidden_dim", 896)
        self.num_heads = config.get("num_heads", 8)
        self.dropout = config.get("dropout", 0.1)
        
        self.attention = MultiHeadAttention(self.hidden_dim, self.num_heads, self.dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout_layer = nn.Dropout(self.dropout)
        
        self.initialized = True
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through self-attention block"""
        # Self-attention
        attn_output = self.attention(x, x, x, mask)
        
        # Residual connection and layer norm
        x = self.layer_norm(x + self.dropout_layer(attn_output))
        
        return x
    
    def _get_fallback_output(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Fallback output when forward pass fails"""
        return x  # Return input unchanged as fallback