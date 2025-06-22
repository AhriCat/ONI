"""
Main ONI NLP module - integrates all NLP components
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple, Union
import logging

# Import all NLP components
from .oni_nlp_core import OniModule, ModuleNotInitializedError
from .oni_nlp_embeddings import EmbeddingModule
from .oni_nlp_transformer import TransformerEncoder
from .oni_nlp_generation import TextGenerator
from .oni_nlp_tasks import TaskIdentifier

# Import enhanced reasoning modules
from .enhanced_reasoning import EnhancedReasoning
from .causal_reasoning import CausalReasoning
from .analogical_reasoning import AnalogicalReasoning
from .counterfactual_reasoning import CounterfactualReasoning
from .multi_step_planning import MultiStepPlanning

logger = logging.getLogger(__name__)

class OptimizedNLPModule(OniModule):
    """Main NLP module integrating all components"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.vocab_size = config.get("vocab_size", 300000)
        self.hidden_dim = config.get("hidden_dim", 896)
        self.num_heads = config.get("num_heads", 8)
        self.num_layers = config.get("num_layers", 6)
        self.max_length = config.get("max_length", 4096)
        
        try:
            self._build_model()
            self.to(self.device)
            self.initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize NLP module: {e}")
            raise ModuleNotInitializedError(f"NLP module initialization failed: {e}")
    
    def _build_model(self):
        """Build the NLP model components"""
        # Core components
        self.embedding = EmbeddingModule(self.config)
        self.transformer = TransformerEncoder(self.config)
        self.generator = TextGenerator(self.config)
        self.task_identifier = TaskIdentifier(self.config)
        
        # Output layers
        self.output_projection = nn.Linear(self.hidden_dim, self.vocab_size)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        
        # Initialize enhanced reasoning modules
        self.use_enhanced_reasoning = self.config.get("use_enhanced_reasoning", True)
        if self.use_enhanced_reasoning:
            self.enhanced_reasoning = EnhancedReasoning(self.hidden_dim)
            
            # Individual reasoning modules for direct access
            self.causal_reasoning = self.enhanced_reasoning.causal_reasoning
            self.analogical_reasoning = self.enhanced_reasoning.analogical_reasoning
            self.counterfactual_reasoning = self.enhanced_reasoning.counterfactual_reasoning
            self.multi_step_planning = self.enhanced_reasoning.multi_step_planning
        
        logger.info("NLP model components built successfully")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                target_ids: Optional[torch.Tensor] = None,
                reasoning_context: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, float]:
        """Forward pass through the NLP model"""
        try:
            batch_size, seq_len = input_ids.shape
            
            # Embedding
            x = self.embedding(input_ids)
            
            # Transformer processing
            if attention_mask is not None:
                # Convert attention mask to transformer format
                attention_mask = attention_mask.bool()
                attention_mask = ~attention_mask  # Invert for transformer
            
            x = self.transformer(x, attention_mask)
            
            # Apply layer normalization
            x = self.layer_norm(x)
            
            # Apply enhanced reasoning if enabled and context provided
            if self.use_enhanced_reasoning and reasoning_context is not None:
                # Convert reasoning context to tensors if needed
                tensor_context = {}
                for key, value in reasoning_context.items():
                    if isinstance(value, torch.Tensor):
                        tensor_context[key] = value
                
                # Apply reasoning to the sequence representation
                reasoning_output = self.enhanced_reasoning(x.mean(dim=1), tensor_context)
                
                # Combine reasoning output with sequence representation
                reasoning_embedding = reasoning_output['output'].unsqueeze(1).expand(-1, seq_len, -1)
                x = x + 0.1 * reasoning_embedding  # Add reasoning with a small weight
            
            # Output projection
            logits = self.output_projection(x)
            
            # Calculate energy (simple norm-based measure)
            energy = torch.norm(x).item() / (batch_size * seq_len)
            
            return logits, energy
            
        except Exception as e:
            logger.error(f"Error in NLP forward pass: {e}")
            return self._get_fallback_output(input_ids), 0.0
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 100, 
                temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9,
                do_sample: bool = True,
                reasoning_type: Optional[str] = None,
                reasoning_context: Optional[Dict[str, Any]] = None) -> Union[torch.Tensor, Dict[str, Any]]:
        """Generate text using the model"""
        try:
            if reasoning_type is not None and self.use_enhanced_reasoning:
                # Generate with explicit reasoning
                return self.generator.generate_with_reasoning(
                    input_ids=input_ids,
                    model=self,
                    reasoning_type=reasoning_type,
                    context=reasoning_context,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample
                )
            else:
                # Standard generation
                return self.generator.generate(
                    input_ids=input_ids,
                    model=self,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                    reasoning_context=reasoning_context
                )
        except Exception as e:
            logger.error(f"Error in text generation: {e}")
            return input_ids  # Return input as fallback
    
    def generate_response(self, text: str, tokenizer=None,
                         reasoning_type: Optional[str] = None,
                         reasoning_context: Optional[Dict[str, Any]] = None) -> str:
        """Generate a response to input text"""
        try:
            if tokenizer is None:
                # Simple fallback response generation
                return f"I understand you said: '{text}'. How can I help you further?"
            
            # Tokenize input
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            input_ids = inputs['input_ids'].to(self.device)
            
            # Generate response
            with torch.no_grad():
                if reasoning_type is not None and self.use_enhanced_reasoning:
                    generation_output = self.generate(
                        input_ids, max_length=50,
                        reasoning_type=reasoning_type,
                        reasoning_context=reasoning_context
                    )
                    generated_ids = generation_output['generated']
                    reasoning_steps = generation_output['reasoning_steps']
                else:
                    generated_ids = self.generate(input_ids, max_length=50)
            
            # Decode response
            response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Add reasoning steps if available
            if reasoning_type is not None and self.use_enhanced_reasoning and 'reasoning_steps' in locals():
                response_with_reasoning = f"Reasoning: {reasoning_steps}\n\nResponse: {response}"
                return response_with_reasoning
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble processing your request right now."
    
    def identify_tasks(self, text: str) -> List[str]:
        """Identify tasks from input text"""
        try:
            return self.task_identifier.identify_tasks(text)
        except Exception as e:
            logger.error(f"Error identifying tasks: {e}")
            return []
    
    def extract_task_parameters(self, text: str, task: str) -> Dict[str, Any]:
        """Extract parameters for a specific task"""
        try:
            return self.task_identifier.extract_task_parameters(text, task)
        except Exception as e:
            logger.error(f"Error extracting task parameters: {e}")
            return {}
    
    def process_raw_input(self, raw_input: Any) -> torch.Tensor:
        """Process raw input and return tensor representation"""
        try:
            if isinstance(raw_input, str):
                # Simple text processing - convert to dummy tensor
                # In practice, this would use a proper tokenizer
                return torch.randn(1, 10, self.hidden_dim, device=self.device)
            elif isinstance(raw_input, torch.Tensor):
                return raw_input.to(self.device)
            else:
                # Convert other types to tensor
                return torch.tensor([0.0], device=self.device).unsqueeze(0).unsqueeze(0)
        except Exception as e:
            logger.error(f"Error processing raw input: {e}")
            return torch.zeros(1, 1, self.hidden_dim, device=self.device)
    
    def _get_fallback_output(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Fallback output when forward pass fails"""
        batch_size, seq_len = input_ids.shape
        return torch.zeros(batch_size, seq_len, self.vocab_size, device=self.device)
    
    def get_module_status(self) -> Dict[str, Any]:
        """Get status of all NLP components"""
        return {
            "initialized": self.initialized,
            "device": str(self.device),
            "vocab_size": self.vocab_size,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "components": {
                "embedding": hasattr(self, 'embedding') and self.embedding.initialized,
                "transformer": hasattr(self, 'transformer') and self.transformer.initialized,
                "generator": hasattr(self, 'generator') and self.generator.initialized,
                "task_identifier": hasattr(self, 'task_identifier') and self.task_identifier.initialized,
                "enhanced_reasoning": hasattr(self, 'enhanced_reasoning')
            },
            "reasoning_modules": {
                "causal_reasoning": hasattr(self, 'causal_reasoning'),
                "analogical_reasoning": hasattr(self, 'analogical_reasoning'),
                "counterfactual_reasoning": hasattr(self, 'counterfactual_reasoning'),
                "multi_step_planning": hasattr(self, 'multi_step_planning')
            }
        }
    
    def apply_causal_reasoning(self, x: torch.Tensor, query: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Apply causal reasoning to input"""
        if not self.use_enhanced_reasoning:
            logger.warning("Enhanced reasoning is disabled")
            return {'output': x}
        
        return self.causal_reasoning(x, query)
    
    def apply_analogical_reasoning(self, source: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply analogical reasoning between source and target domains"""
        if not self.use_enhanced_reasoning:
            logger.warning("Enhanced reasoning is disabled")
            return {'output': target}
        
        return self.analogical_reasoning(source, target)
    
    def apply_counterfactual_reasoning(self, x: torch.Tensor, query: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Apply counterfactual reasoning to input"""
        if not self.use_enhanced_reasoning:
            logger.warning("Enhanced reasoning is disabled")
            return {'output': x}
        
        return self.counterfactual_reasoning(x, query)
    
    def apply_planning(self, state: torch.Tensor, goal: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply multi-step planning to achieve a goal"""
        if not self.use_enhanced_reasoning:
            logger.warning("Enhanced reasoning is disabled")
            return {'output': state}
        
        return self.multi_step_planning(state, goal)

# For backward compatibility
def create_nlp_module(config: Dict[str, Any]) -> OptimizedNLPModule:
    """Factory function to create NLP module"""
    return OptimizedNLPModule(config)