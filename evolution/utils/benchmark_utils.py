# File: evolution/utils/benchmark_utils.py
"""Benchmark utilities for ONI evaluation."""
import time
import torch
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def timed_inference(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    num_runs: int = 10,
    warmup: int = 3
) -> Dict[str, float]:
    """
    Benchmark inference time for a model.

    Returns dict with mean, std, min, max times in seconds.
    """
    times = []

    with torch.no_grad():
        for _ in range(warmup):
            model(input_tensor)

        for _ in range(num_runs):
            start = time.perf_counter()
            model(input_tensor)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

    mean = sum(times) / len(times)
    variance = sum((t - mean) ** 2 for t in times) / len(times)
    return {
        'mean': mean,
        'std': variance ** 0.5,
        'min': min(times),
        'max': max(times),
        'num_runs': num_runs
    }


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }
