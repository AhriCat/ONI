"""
Core Oni system with improved architecture and error handling
"""
import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

# Import configuration
from config.settings import DEFAULT_MODEL_CONFIG, DEVICE, LOGGING_CONFIG

# Import modules
from modules.oni_base import OniModule, OniError
from modules.oni_nlp_fixed import OptimizedNLPModule
from modules.oni_tokenizer import MultitokenBPETokenizer
from modules.oni_memory import Memory
from modules.oni_emotions import EmotionalEnergyModel

# Setup logging
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class OniCore(nn.Module):
    """
    Core Oni system with modular architecture and proper error handling
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        self.config = config or DEFAULT_MODEL_CONFIG.copy()
        self.device = torch.device(DEVICE)
        self.modules = {}
        self.initialized = False
        
        logger.info(f"Initializing Oni Core on device: {self.device}")
        self._initialize_modules()
    
    def _initialize_modules(self):
        """Initialize all Oni modules with proper error handling"""
        try:
            # Initialize tokenizer
            self.tokenizer = MultitokenBPETokenizer(
                vocab_size=self.config.get("vocab_size", 300000),
                max_merges=self.config.get("max_merges", 30000)
            )
            
            # Initialize NLP module
            self.nlp_module = OptimizedNLPModule(self.config)
            self.modules['nlp'] = self.nlp_module
            
            # Initialize memory system
            self.memory = Memory(self.tokenizer)
            self.modules['memory'] = self.memory
            
            # Initialize emotional system
            self.emotional_system = EmotionalEnergyModel(
                hidden_dim=self.config.get("hidden_dim", 896)
            )
            self.modules['emotions'] = self.emotional_system
            
            self.initialized = True
            logger.info("All modules initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize modules: {e}")
            raise OniError(f"Oni initialization failed: {e}")
    
    def forward(self, text: str, **kwargs) -> Dict[str, Any]:
        """Main forward pass through Oni system"""
        if not self.initialized:
            raise OniError("Oni system not properly initialized")
        
        try:
            # Tokenize input
            tokens = self.tokenizer(text, return_tensors='pt')
            input_ids = tokens['input_ids'].to(self.device)
            attention_mask = tokens['attention_mask'].to(self.device)
            
            # Process through NLP module
            nlp_output = self.nlp_module.safe_forward(input_ids, attention_mask)
            
            # Process through emotional system
            emotional_output = self.emotional_system(text)
            
            # Update memory
            self.memory.update_memory(text)
            
            return {
                'nlp_output': nlp_output,
                'emotional_state': emotional_output,
                'tokens': tokens,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error in Oni forward pass: {e}")
            return {
                'error': str(e),
                'success': False
            }
    
    def generate_response(self, text: str, max_length: int = 100) -> str:
        """Generate a response to input text"""
        try:
            # Process input
            result = self.forward(text)
            
            if not result['success']:
                return f"Error processing input: {result.get('error', 'Unknown error')}"
            
            # Generate response using NLP module
            tokens = self.tokenizer(text, return_tensors='pt')
            input_ids = tokens['input_ids'].to(self.device)
            
            generated = self.nlp_module.generate(
                input_ids, 
                max_length=max_length,
                temperature=0.8
            )
            
            # Decode response
            response = self.tokenizer.decode(generated[0])
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error processing your request."
    
    def safe_execute_task(self, task: str, *args, **kwargs) -> Any:
        """Safely execute a task with error handling"""
        try:
            if hasattr(self, f"_execute_{task}"):
                method = getattr(self, f"_execute_{task}")
                return method(*args, **kwargs)
            else:
                logger.warning(f"Unknown task: {task}")
                return f"Task '{task}' is not supported"
        except Exception as e:
            logger.error(f"Error executing task '{task}': {e}")
            return f"Error executing task: {e}"
    
    def get_module_status(self) -> Dict[str, bool]:
        """Get status of all modules"""
        status = {}
        for name, module in self.modules.items():
            try:
                # Simple test to see if module is working
                if hasattr(module, 'forward'):
                    status[name] = True
                else:
                    status[name] = False
            except Exception:
                status[name] = False
        
        return status
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self.memory, 'cleanup'):
                self.memory.cleanup()
            
            # Clear CUDA cache if using GPU
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                
            logger.info("Oni cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Factory function for easy instantiation
def create_oni(config: Optional[Dict[str, Any]] = None) -> OniCore:
    """Factory function to create Oni instance"""
    return OniCore(config)

# Example usage
if __name__ == "__main__":
    # Create Oni instance
    oni = create_oni()
    
    # Test basic functionality
    response = oni.generate_response("Hello, how are you?")
    print(f"Oni: {response}")
    
    # Check module status
    status = oni.get_module_status()
    print(f"Module status: {status}")
    
    # Cleanup
    oni.cleanup()