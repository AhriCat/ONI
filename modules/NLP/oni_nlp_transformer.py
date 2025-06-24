"""
Transformer encoder/decoder components for ONI NLP
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from .oni_nlp_core import OniModule
from .oni_nlp_attention import SelfAttentionBlock
from .oni_nlp_feedforward import FeedForwardBlock

class TransformerEncoderLayer(OniModule):
    """Single transformer encoder layer"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        self.attention_block = SelfAttentionBlock(config)
        self.feedforward_block = FeedForwardBlock(config)
        
        self.initialized = True
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through transformer encoder layer"""
        # Self-attention
        x = self.attention_block(x, mask)
        
        # Feed-forward
        x = self.feedforward_block(x)
        
        return x
    
    def _get_fallback_output(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Fallback output when forward pass fails"""
        return x

class TransformerEncoder(OniModule):
    """Multi-layer transformer encoder"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        self.num_layers = config.get("num_layers", 6)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config) for _ in range(self.num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(config.get("hidden_dim", 896))
        
        self.initialized = True
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through transformer encoder"""
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.layer_norm(x)
        return x
    
    def _get_fallback_output(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Fallback output when forward pass fails"""
        return x
