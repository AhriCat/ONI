"""
Embedding and positional encoding components for ONI NLP
"""
import torch
import torch.nn as nn
import math
from .oni_nlp_core import OniModule

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models"""
    
    def __init__(self, d_model: int, max_length: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TokenEmbedding(nn.Module):
    """Token embedding with optional scaling"""
    
    def __init__(self, vocab_size: int, d_model: int, padding_idx: int = 0):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)

class EmbeddingModule(OniModule):
    """Combined embedding module with token and positional encodings"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        self.vocab_size = config.get("vocab_size", 300000)
        self.hidden_dim = config.get("hidden_dim", 896)
        self.max_length = config.get("max_length", 4096)
        self.dropout = config.get("dropout", 0.1)
        
        self.token_embedding = TokenEmbedding(
            self.vocab_size, 
            self.hidden_dim,
            padding_idx=config.get("pad_token_id", 0)
        )
        self.positional_encoding = PositionalEncoding(
            self.hidden_dim, 
            self.max_length, 
            self.dropout
        )
        
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout_layer = nn.Dropout(self.dropout)
        
        self.initialized = True
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through embedding layers"""
        # Token embedding
        x = self.token_embedding(input_ids)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Layer norm and dropout
        x = self.layer_norm(x)
        x = self.dropout_layer(x)
        
        return x
    
    def _get_fallback_output(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Fallback output when forward pass fails"""
        batch_size, seq_len = input_ids.shape
        return torch.zeros(batch_size, seq_len, self.hidden_dim, device=self.device)
