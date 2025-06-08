"""
Base classes and utilities for Oni modules
"""
import torch
import torch.nn as nn
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class OniModule(nn.Module, ABC):
    """Base class for all Oni modules"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup module-specific logging"""
        self.logger = logging.getLogger(f"oni.{self.__class__.__name__}")
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass - must be implemented by subclasses"""
        pass
    
    def safe_forward(self, *args, **kwargs):
        """Safe forward pass with error handling"""
        try:
            return self.forward(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error in {self.__class__.__name__}.forward: {e}")
            return self._get_fallback_output(*args, **kwargs)
    
    def _get_fallback_output(self, *args, **kwargs):
        """Fallback output when forward pass fails"""
        return None
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to module device"""
        return tensor.to(self.device)

class OniError(Exception):
    """Base exception for Oni-specific errors"""
    pass

class ModuleNotInitializedError(OniError):
    """Raised when a module is used before initialization"""
    pass

class ConfigurationError(OniError):
    """Raised when there's a configuration issue"""
    pass