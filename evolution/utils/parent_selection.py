# File: evolution/utils/parent_selection.py
"""
Parent selection algorithms for DGM-style evolution.

[EDITOR] Extracted from ONIArchive.select_parents() for reusability and
testing. The archive calls these internally.
"""
import math
import random
from typing import List, Dict


def sigmoid(x: float, lambda_param: float = 10.0, alpha_0: float = 0.5) -> float:
    """Sigmoid transform matching DGM paper: 1 / (1 + exp(-λ(x - α₀)))"""
    return 1.0 / (1.0 + math.exp(-lambda_param * (x - alpha_0)))


def score_child_proportional(
    candidates: Dict[str, Dict],
    k: int,
    lambda_param: float = 10.0,
    alpha_0: float = 0.5
) -> List[str]:
    """
    Select k parents using score-child proportional selection.

    P(parent_i) = sigmoid(score_i) * 1/(1 + children_i) / Z
    """
    ids = list(candidates.keys())
    if not ids:
        return []

    scores = [sigmoid(candidates[i]['score'], lambda_param, alpha_0) for i in ids]
    child_penalties = [
        1.0 / (1.0 + candidates[i].get('children_count', 0)) for i in ids
    ]

    raw_probs = [s * c for s, c in zip(scores, child_penalties)]
    total = sum(raw_probs)
    if total == 0:
        probs = [1.0 / len(ids)] * len(ids)
    else:
        probs = [p / total for p in raw_probs]

    return random.choices(ids, weights=probs, k=k)


def score_proportional(
    candidates: Dict[str, Dict],
    k: int,
    lambda_param: float = 10.0,
    alpha_0: float = 0.5
) -> List[str]:
    """Select k parents using score-only sigmoid proportional selection."""
    ids = list(candidates.keys())
    if not ids:
        return []

    scores = [sigmoid(candidates[i]['score'], lambda_param, alpha_0) for i in ids]
    total = sum(scores)
    probs = [s / total for s in scores] if total > 0 else [1.0 / len(ids)] * len(ids)
    return random.choices(ids, weights=probs, k=k)
