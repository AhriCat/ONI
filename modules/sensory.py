# sensory.py
from typing import NamedTuple, Optional
import torch

class SensoryInput(NamedTuple):
    vision: Optional[torch.Tensor] = None
    audio: Optional[torch.Tensor] = None
    text: Optional[torch.Tensor] = None
    proprioception: Optional[torch.Tensor] = None
    timestamp: float = 0.0
    confidence: float = 1.0
