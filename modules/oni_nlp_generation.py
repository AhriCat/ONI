"""
Text generation components for ONI NLP
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
from .oni_nlp_core import OniModule
from .enhanced_reasoning import EnhancedReasoning

class GenerationHead(nn.Module):
    """Output head for text generation"""
    
    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)

class TextGenerator(OniModule):
    """Text generation module with various sampling strategies and enhanced reasoning"""
    
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
        
        # Initialize enhanced reasoning module
        self.use_enhanced_reasoning = config.get("use_enhanced_reasoning", True)
        if self.use_enhanced_reasoning:
            self.reasoning_module = EnhancedReasoning(self.hidden_dim)
        
        self.initialized = True
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass to get logits"""
        # Apply enhanced reasoning if enabled
        if self.use_enhanced_reasoning:
            reasoning_outputs = self.reasoning_module(hidden_states)
            # Combine original hidden states with reasoning outputs
            hidden_states = hidden_states + reasoning_outputs['output']
            
        return self.generation_head(hidden_states)
    
    def generate(self, input_ids: torch.Tensor, model: nn.Module, 
                max_length: Optional[int] = None, temperature: float = 1.0, 
                top_k: int = 50, top_p: float = 0.9, 
                do_sample: bool = True,
                reasoning_context: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
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
                
                # Apply enhanced reasoning for token selection if enabled
                if self.use_enhanced_reasoning and reasoning_context is not None:
                    # Get token embeddings
                    token_embeddings = model.embedding(next_token).squeeze(1)
                    
                    # Apply reasoning to refine token selection
                    reasoning_outputs = self.reasoning_module(token_embeddings, reasoning_context)
                    refined_embeddings = reasoning_outputs['output']
                    
                    # Convert back to token space
                    refined_logits = torch.matmul(refined_embeddings, model.embedding.weight.t())
                    refined_token = torch.argmax(refined_logits, dim=-1, keepdim=True)
                    
                    # Use refined token with some probability
                    use_refined = (torch.rand(next_token.shape) > 0.5).to(next_token.device)
                    next_token = torch.where(use_refined, refined_token, next_token)
                
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
    
    def generate_with_reasoning(self, input_ids: torch.Tensor, model: nn.Module,
                              reasoning_type: str = 'auto',
                              context: Optional[Dict[str, torch.Tensor]] = None,
                              **kwargs) -> Dict[str, Any]:
        """
        Generate text with explicit reasoning steps.
        
        Args:
            input_ids: Input token IDs
            model: Model to use for generation
            reasoning_type: Type of reasoning to use ('causal', 'analogical', 'counterfactual', 'planning', or 'auto')
            context: Optional context for reasoning
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with generation results and reasoning steps
        """
        if not self.use_enhanced_reasoning:
            # Fall back to regular generation if enhanced reasoning is disabled
            generated = self.generate(input_ids, model, **kwargs)
            return {'generated': generated, 'reasoning_steps': None}
        
        # Prepare reasoning context
        if context is None:
            context = {}
        
        # Get initial hidden states
        with torch.no_grad():
            outputs = model(input_ids)
            if isinstance(outputs, tuple):
                hidden_states = outputs[0][:, -1]
            else:
                hidden_states = outputs[:, -1]
        
        # Determine reasoning type if auto
        if reasoning_type == 'auto':
            reasoning_weights = self.reasoning_module.reasoning_selector(hidden_states)
            reasoning_type_idx = reasoning_weights.argmax(dim=-1).item()
            reasoning_types = ['causal', 'analogical', 'counterfactual', 'planning']
            reasoning_type = reasoning_types[reasoning_type_idx]
        
        # Apply specific reasoning
        reasoning_steps = []
        
        if reasoning_type == 'causal':
            # Causal reasoning
            causal_context = {'intervention_query': context.get('query', hidden_states)}
            causal_output = self.reasoning_module.causal_reasoning(hidden_states, causal_context)
            
            # Record reasoning steps
            reasoning_steps.append({
                'type': 'causal',
                'description': 'Identifying causal relationships',
                'adjacency_matrix': causal_output['adjacency_matrix'].detach().cpu().numpy().tolist()
            })
            
            # Update context
            context['causal_variables'] = causal_output['causal_variables']
            
        elif reasoning_type == 'analogical':
            # Analogical reasoning
            source_domain = context.get('source_domain', input_ids)
            target_domain = context.get('target_domain', input_ids)
            
            # Get embeddings
            with torch.no_grad():
                source_outputs = model(source_domain)
                target_outputs = model(target_domain)
                
                if isinstance(source_outputs, tuple):
                    source_hidden = source_outputs[0]
                    target_hidden = target_outputs[0]
                else:
                    source_hidden = source_outputs
                    target_hidden = target_outputs
            
            analogical_output = self.reasoning_module.analogical_reasoning(
                source_hidden, target_hidden
            )
            
            # Record reasoning steps
            reasoning_steps.append({
                'type': 'analogical',
                'description': 'Mapping between source and target domains',
                'correspondence_scores': analogical_output['correspondence_scores'].mean(dim=0).detach().cpu().numpy().tolist()
            })
            
            # Update context
            context['analogical_mapping'] = analogical_output['correspondence_scores']
            
        elif reasoning_type == 'counterfactual':
            # Counterfactual reasoning
            counterfactual_output = self.reasoning_module.counterfactual_reasoning(
                hidden_states, context.get('query', None)
            )
            
            # Record reasoning steps
            reasoning_steps.append({
                'type': 'counterfactual',
                'description': 'Generating "what if" scenarios',
                'num_scenarios': counterfactual_output['counterfactuals'].shape[1]
            })
            
            # Update context
            context['counterfactuals'] = counterfactual_output['counterfactuals']
            
        elif reasoning_type == 'planning':
            # Multi-step planning
            goal = context.get('goal', hidden_states)
            planning_output = self.reasoning_module.multi_step_planning(hidden_states, goal)
            
            # Record reasoning steps
            reasoning_steps.append({
                'type': 'planning',
                'description': 'Creating a step-by-step plan',
                'num_steps': planning_output['plan'].shape[1]
            })
            
            # Update context
            context['plan'] = planning_output['plan']
        
        # Generate with enhanced reasoning context
        generated = self.generate(input_ids, model, reasoning_context=context, **kwargs)
        
        return {
            'generated': generated,
            'reasoning_type': reasoning_type,
            'reasoning_steps': reasoning_steps
        }
    
    def _get_fallback_output(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Fallback output when forward pass fails"""
        batch_size, seq_len, _ = hidden_states.shape
        return torch.zeros(batch_size, seq_len, self.vocab_size, device=self.device)