"""
Feed-forward network components for ONI NLP
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .oni_nlp_core import OniModule

class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "swish":
            self.activation = lambda x: x * torch.sigmoid(x)
        else:
            self.activation = F.relu
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class FeedForwardBlock(OniModule):
    """Feed-forward block with residual connections and layer norm"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        self.hidden_dim = config.get("hidden_dim", 896)
        self.ff_dim = config.get("ff_dim", self.hidden_dim * 4)
        self.dropout = config.get("dropout", 0.1)
        self.activation = config.get("activation", "relu")
        
        self.feed_forward = FeedForwardNetwork(
            self.hidden_dim, 
            self.ff_dim, 
            self.dropout, 
            self.activation
        )
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout_layer = nn.Dropout(self.dropout)
        
        self.initialized = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feed-forward block"""
        # Feed-forward network
        ff_output = self.feed_forward(x)
        
        # Residual connection and layer norm
        x = self.layer_norm(x + self.dropout_layer(ff_output))
        
        return x
    
    def _get_fallback_output(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Fallback output when forward pass fails"""
        return x  # Return input unchanged as fallback