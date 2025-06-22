"""
Attention mechanisms for ONI NLP
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, List, Tuple
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

class TemporalSparseTrifocusedAttention(nn.Module):
    """
    Temporal Sparse Tri-Focused Hierarchical Attention mechanism.
    
    This attention mechanism operates at three levels of focus:
    1. Global level: Attends to the entire sequence for high-level context
    2. Local level: Attends to nearby tokens for local context
    3. Temporal level: Attends to tokens across different time steps
    
    It uses sparse attention patterns to reduce computational complexity
    and adaptively switches between the three levels based on the input.
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1, 
                 local_window_size: int = 16, sparsity_factor: float = 0.8):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.local_window_size = local_window_size
        self.sparsity_factor = sparsity_factor
        
        # Projection matrices for each focus level
        # Global focus
        self.global_q = nn.Linear(d_model, d_model, bias=False)
        self.global_k = nn.Linear(d_model, d_model, bias=False)
        self.global_v = nn.Linear(d_model, d_model, bias=False)
        
        # Local focus
        self.local_q = nn.Linear(d_model, d_model, bias=False)
        self.local_k = nn.Linear(d_model, d_model, bias=False)
        self.local_v = nn.Linear(d_model, d_model, bias=False)
        
        # Temporal focus
        self.temporal_q = nn.Linear(d_model, d_model, bias=False)
        self.temporal_k = nn.Linear(d_model, d_model, bias=False)
        self.temporal_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.output_projection = nn.Linear(d_model * 3, d_model)
        
        # Focus gate - determines which focus level to use
        self.focus_gate = nn.Sequential(
            nn.Linear(d_model, 3),
            nn.Softmax(dim=-1)
        )
        
        # Sparse attention mask generator
        self.sparse_mask_generator = nn.Linear(d_model, 1)
        
        # Dropouts
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        # Layer normalization for each focus
        self.norm_global = nn.LayerNorm(d_model)
        self.norm_local = nn.LayerNorm(d_model)
        self.norm_temporal = nn.LayerNorm(d_model)
        
        # Positional encoding for temporal attention
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
    def _split_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Split the last dimension into (num_heads, depth)"""
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, depth)
    
    def _merge_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Merge the (num_heads, depth) into d_model"""
        x = x.permute(0, 2, 1, 3)  # (batch_size, seq_len, num_heads, depth)
        return x.reshape(batch_size, -1, self.d_model)
    
    def _generate_sparse_mask(self, x: torch.Tensor, sparsity: float) -> torch.Tensor:
        """Generate a sparse attention mask based on token importance"""
        # Generate token importance scores
        importance = self.sparse_mask_generator(x).squeeze(-1)
        
        # Keep only the top (1-sparsity)% tokens
        k = max(1, int((1 - sparsity) * importance.size(-1)))
        top_k_values, top_k_indices = torch.topk(importance, k, dim=-1)
        
        # Create sparse mask
        mask = torch.zeros_like(importance, dtype=torch.bool)
        mask.scatter_(-1, top_k_indices, True)
        
        # Expand mask for attention
        return mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
    
    def _generate_local_mask(self, seq_len: int, window_size: int, device: torch.device) -> torch.Tensor:
        """Generate a local attention mask with a sliding window"""
        mask = torch.zeros(seq_len, seq_len, device=device)
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = 1
        return mask
    
    def _generate_temporal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate a temporal attention mask that focuses on same positions across time"""
        # For simplicity, we'll use a strided pattern
        mask = torch.zeros(seq_len, seq_len, device=device)
        stride = max(1, seq_len // 8)  # Attend to every stride-th position
        
        for i in range(seq_len):
            for j in range(0, seq_len, stride):
                mask[i, j] = 1
                
        return mask
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the Tri-Focused Attention.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Apply positional encoding for temporal awareness
        x_with_pos = self.pos_encoder(x)
        
        # Determine focus weights using the gate
        focus_weights = self.focus_gate(x.mean(dim=1))  # [batch_size, 3]
        global_weight = focus_weights[:, 0].unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1]
        local_weight = focus_weights[:, 1].unsqueeze(1).unsqueeze(1)   # [batch_size, 1, 1]
        temporal_weight = focus_weights[:, 2].unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1]
        
        # Generate sparse attention mask
        sparse_mask = self._generate_sparse_mask(x, self.sparsity_factor)
        
        # Generate local attention mask
        local_mask = self._generate_local_mask(seq_len, self.local_window_size, device)
        local_mask = local_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]
        
        # Generate temporal attention mask
        temporal_mask = self._generate_temporal_mask(seq_len, device)
        temporal_mask = temporal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]
        
        # Combine with input mask if provided
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            sparse_mask = sparse_mask & mask
            local_mask = local_mask & mask
            temporal_mask = temporal_mask & mask
        
        # Global focus attention
        global_q = self._split_heads(self.global_q(x), batch_size)
        global_k = self._split_heads(self.global_k(x), batch_size)
        global_v = self._split_heads(self.global_v(x), batch_size)
        
        global_scores = torch.matmul(global_q, global_k.transpose(-2, -1)) / math.sqrt(self.d_k)
        global_scores = global_scores.masked_fill(~sparse_mask, -1e9)
        global_attn_weights = F.softmax(global_scores, dim=-1)
        global_attn_weights = self.attn_dropout(global_attn_weights)
        global_output = torch.matmul(global_attn_weights, global_v)
        global_output = self._merge_heads(global_output, batch_size)
        global_output = self.norm_global(global_output)
        
        # Local focus attention
        local_q = self._split_heads(self.local_q(x), batch_size)
        local_k = self._split_heads(self.local_k(x), batch_size)
        local_v = self._split_heads(self.local_v(x), batch_size)
        
        local_scores = torch.matmul(local_q, local_k.transpose(-2, -1)) / math.sqrt(self.d_k)
        local_scores = local_scores.masked_fill(~local_mask, -1e9)
        local_attn_weights = F.softmax(local_scores, dim=-1)
        local_attn_weights = self.attn_dropout(local_attn_weights)
        local_output = torch.matmul(local_attn_weights, local_v)
        local_output = self._merge_heads(local_output, batch_size)
        local_output = self.norm_local(local_output)
        
        # Temporal focus attention
        temporal_q = self._split_heads(self.temporal_q(x_with_pos), batch_size)
        temporal_k = self._split_heads(self.temporal_k(x_with_pos), batch_size)
        temporal_v = self._split_heads(self.temporal_v(x_with_pos), batch_size)
        
        temporal_scores = torch.matmul(temporal_q, temporal_k.transpose(-2, -1)) / math.sqrt(self.d_k)
        temporal_scores = temporal_scores.masked_fill(~temporal_mask, -1e9)
        temporal_attn_weights = F.softmax(temporal_scores, dim=-1)
        temporal_attn_weights = self.attn_dropout(temporal_attn_weights)
        temporal_output = torch.matmul(temporal_attn_weights, temporal_v)
        temporal_output = self._merge_heads(temporal_output, batch_size)
        temporal_output = self.norm_temporal(temporal_output)
        
        # Combine the three focus outputs with adaptive weighting
        combined_output = torch.cat([
            global_output * global_weight,
            local_output * local_weight,
            temporal_output * temporal_weight
        ], dim=-1)
        
        # Project back to original dimension
        output = self.output_projection(combined_output)
        output = self.output_dropout(output)
        
        return output

class PositionalEncoding(nn.Module):
    """Positional encoding for temporal awareness"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class SelfAttentionBlock(OniModule):
    """Self-attention block with residual connections and layer norm"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        self.hidden_dim = config.get("hidden_dim", 896)
        self.num_heads = config.get("num_heads", 8)
        self.dropout = config.get("dropout", 0.1)
        
        # Use the new TemporalSparseTrifocusedAttention instead of MultiHeadAttention
        use_trifocused = config.get("use_trifocused_attention", True)
        if use_trifocused:
            self.attention = TemporalSparseTrifocusedAttention(
                self.hidden_dim, 
                self.num_heads, 
                self.dropout,
                local_window_size=config.get("local_window_size", 16),
                sparsity_factor=config.get("sparsity_factor", 0.8)
            )
        else:
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