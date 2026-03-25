# File: evolution/utils/patch_utils.py
"""Utilities for creating and applying weight patches to ONI variants."""
import torch
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def create_weight_patch(
    old_state: Dict[str, torch.Tensor],
    new_state: Dict[str, torch.Tensor],
    threshold: float = 1e-8
) -> Dict[str, torch.Tensor]:
    """
    Create a weight diff between two model states.
    Only includes parameters that actually changed beyond threshold.
    """
    patch = {}
    for key in new_state:
        if key in old_state:
            diff = new_state[key] - old_state[key]
            if diff.abs().max().item() > threshold:
                patch[key] = diff
    return patch


def apply_weight_patch(
    model: torch.nn.Module,
    patch: Dict[str, torch.Tensor],
    alpha: float = 1.0
) -> None:
    """
    Apply a weight patch to a model in-place.

    Args:
        model: Target model
        patch: Dict of parameter_name -> diff_tensor
        alpha: Scaling factor (1.0 = full patch, 0.5 = half)
    """
    state = model.state_dict()
    for key, diff in patch.items():
        if key in state:
            state[key] = state[key] + alpha * diff
        else:
            logger.warning(f"Patch key {key} not found in model state dict")
    model.load_state_dict(state)


def save_patch(patch: Dict[str, torch.Tensor], path: str) -> None:
    """Save a weight patch to disk."""
    torch.save(patch, path)


def load_patch(path: str) -> Dict[str, torch.Tensor]:
    """Load a weight patch from disk."""
    return torch.load(path, map_location='cpu')
