"""
Text generation components for ONI NLP
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
from .oni_nlp_core import OniModule

class GenerationHead(nn.Module):
    """Output head for text generation"""
    
    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)

class TextGenerator(OniModule):
    """Text generation module with various sampling strategies"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        self.vocab_size = config.get("vocab_size", 300000)
        self.hidden_dim = config.get("hidden_dim", 896)
        self.max_length = config.get("max_generation_length", 512)
        
        self.generation_head = GenerationHead(self.hidden_dim, self.vocab_size)
        
        # Special token IDs
        self.pad_token_id = config.get("pad_token_id", 0)
        self.eos_token_id = config.get("eos_token_id", 5)
        self.bos_token_id = config.get("bos_token_id", 6)
        
        self.initialized = True
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass to get logits"""
        return self.generation_head(hidden_states)
    
    def generate(self, input_ids: torch.Tensor, model: nn.Module, 
                max_length: Optional[int] = None, temperature: float = 1.0, 
                top_k: int = 50, top_p: float = 0.9, 
                do_sample: bool = True) -> torch.Tensor:
        """Generate text using the model"""
        if max_length is None:
            max_length = self.max_length
        
        model.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get model outputs
                outputs = model(generated)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature
                
                if do_sample:
                    # Apply top-k filtering
                    if top_k > 0:
                        next_token_logits = self._top_k_filtering(next_token_logits, top_k)
                    
                    # Apply top-p (nucleus) filtering
                    if top_p < 1.0:
                        next_token_logits = self._top_p_filtering(next_token_logits, top_p)
                    
                    # Sample from the filtered distribution
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for end of sequence
                if next_token.item() == self.eos_token_id:
                    break
        
        return generated
    
    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Apply top-k filtering to logits"""
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        return logits
    
    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply top-p (nucleus) filtering to logits"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        return logits
    
    def _get_fallback_output(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Fallback output when forward pass fails"""
        batch_size, seq_len, _ = hidden_states.shape
        return torch.zeros(batch_size, seq_len, self.vocab_size, device=self.device)