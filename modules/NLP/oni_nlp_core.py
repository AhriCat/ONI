"""
Core NLP functionality and base classes
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)

class OniModule(nn.Module):
    """Base class for all ONI modules with error handling"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialized = False
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def safe_forward(self, *args, **kwargs):
        """Safe forward pass with error handling"""
        try:
            return self.forward(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error in {self.__class__.__name__}.forward: {e}")
            return self._get_fallback_output(*args, **kwargs)
    
    def _get_fallback_output(self, *args, **kwargs):
        """Override in subclasses to provide fallback output"""
        return torch.zeros(1, self.config.get("hidden_dim", 896), device=self.device)

class ModuleNotInitializedError(Exception):
    """Raised when a module is used before proper initialization"""
    pass
