from __future__ import annotations
from typing import Dict, Any, Type

class Node:
    """Base class for effects/filters. Stateless; parameters supplied at apply()."""
    NAME = "base"
    def apply(self, params: Dict[str, Any]) -> str:
        """Return an ffmpeg filtergraph snippet."""
        return ""  # no-op

_REGISTRY: Dict[str, Type[Node]] = {}

def register(node_cls: Type[Node]):
    _REGISTRY[node_cls.NAME] = node_cls
    return node_cls

def get(name: str) -> Type[Node]:
    if name not in _REGISTRY:
        raise KeyError(f"Effect node '{name}' not registered")
    return _REGISTRY[name]

def available() -> Dict[str, Type[Node]]:
    return dict(_REGISTRY)
