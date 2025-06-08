"""
Fixed NLP module with proper error handling and structure
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
from .oni_base import OniModule, ModuleNotInitializedError

class OptimizedNLPModule(OniModule):
    """Optimized NLP module with proper error handling"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.vocab_size = config.get("vocab_size", 300000)
        self.hidden_dim = config.get("hidden_dim", 896)
        self.num_heads = config.get("num_heads", 8)
        self.num_layers = config.get("num_layers", 6)
        self.max_length = config.get("max_length", 4096)
        
        self._build_model()
        self.to(self.device)
    
    def _build_model(self):
        """Build the NLP model components"""
        try:
            self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
            self.positional_encoding = nn.Parameter(
                torch.randn(self.max_length, self.hidden_dim) * 0.02
            )
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, self.num_layers)
            
            self.output_projection = nn.Linear(self.hidden_dim, self.vocab_size)
            self.layer_norm = nn.LayerNorm(self.hidden_dim)
            
        except Exception as e:
            self.logger.error(f"Failed to build NLP model: {e}")
            raise ModuleNotInitializedError(f"NLP module initialization failed: {e}")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the NLP model"""
        batch_size, seq_len = input_ids.shape
        
        # Embedding and positional encoding
        x = self.embedding(input_ids)
        if seq_len <= self.max_length:
            x = x + self.positional_encoding[:seq_len].unsqueeze(0)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Transformer processing
        if attention_mask is not None:
            # Convert attention mask to transformer format
            attention_mask = attention_mask.bool()
            attention_mask = ~attention_mask  # Invert for transformer
        
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        
        # Output projection
        logits = self.output_projection(x)
        
        return logits
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 100, 
                temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        """Generate text using the model"""
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                logits = self.forward(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for end token (assuming 5 is EOS)
                if next_token.item() == 5:
                    break
        
        return generated
    
    def _get_fallback_output(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Fallback output when forward pass fails"""
        batch_size, seq_len = input_ids.shape
        return torch.zeros(batch_size, seq_len, self.vocab_size, device=self.device)
    
    def identify_tasks(self, text: str) -> List[str]:
        """Identify tasks from input text"""
        tasks = []
        text_lower = text.lower()
        
        task_patterns = {
            'search': ['/search', 'search for', 'find', 'look up'],
            'draw': ['/draw', 'draw', 'create image', 'generate picture'],
            'animate': ['/animate', 'animate', 'create animation'],
            'calculate': ['calculate', 'compute', 'math', 'solve'],
            'code': ['code', 'program', 'write code', 'debug'],
            'monitor': ['monitor', 'watch', 'observe', 'track']
        }
        
        for task, patterns in task_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                tasks.append(task)
        
        return tasks