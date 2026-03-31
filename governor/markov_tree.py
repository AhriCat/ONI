"""
governor/markov_tree.py
========================
MarkovDecisionTree — past decision / outcome registry with Bayesian updating.

Structure
---------
Each node in the tree represents a (state, routing_decision) pair.
The tree branches on routing decisions made from similar states.

State identity
  Hidden states are too high-dimensional to use directly as keys.
  We hash them with sign-LSH: take sign(x @ random_projections) → a
  fixed-length binary string.  Similar states cluster to the same hash
  with high probability (locality-sensitive hashing).

Bayesian updating
  For each (state_hash, path_index) pair we maintain a Beta(α, β)
  distribution over success probability.
  - Prior: α=1, β=1 (uniform — no assumption)
  - On success (reward > 0): α += reward_magnitude
  - On failure (reward ≤ 0): β += |reward_magnitude|
  Thompson Sampling: draw p ~ Beta(α, β) for each path; use log(p) as
  an additive logit adjustment biasing future routing.

Live policy adjusting
  call observe_outcome(reward) after each step to update the tree.
  No gradient required — purely count-based Bayesian update.
"""

import hashlib
import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class _Node:
    """Single tree node: Beta params for one (state_hash, path_idx) pair."""
    __slots__ = ("alpha", "beta", "visit_count")

    def __init__(self):
        self.alpha = 1.0        # Beta prior: successes + 1
        self.beta  = 1.0        # Beta prior: failures  + 1
        self.visit_count = 0

    def update(self, reward: float):
        clipped = max(-10.0, min(10.0, reward))
        if clipped > 0:
            self.alpha += clipped
        else:
            self.beta  += abs(clipped)
        self.visit_count += 1

    def thompson_sample(self) -> float:
        """Draw one sample from Beta(α, β) — approximated via ratio of Gammas."""
        # Use the mean as a deterministic approximation when counts are low
        # and sampling otherwise (no scipy dependency)
        if self.visit_count < 3:
            return self.alpha / (self.alpha + self.beta)
        # Approximate Beta sample: u^(1/α) / (u^(1/α) + v^(1/β))
        import random
        u = random.random() ** (1.0 / self.alpha)
        v = random.random() ** (1.0 / self.beta)
        denom = u + v
        return u / denom if denom > 0 else 0.5

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


class MarkovDecisionTree(nn.Module):
    """
    Hash-backed Markov decision registry with Thompson-Sampling Bayesian priors.

    Parameters
    ----------
    n_paths      : number of routing paths (must match FiringRouter)
    lsh_dim      : number of random projections for state hashing
    capacity     : maximum number of distinct state hashes to retain
    history_len  : rolling window of recent decisions for live updates
    """

    def __init__(
        self,
        n_paths:     int = 6,
        hidden_dim:  int = 896,
        lsh_dim:     int = 64,
        capacity:    int = 8192,
        history_len: int = 128,
    ):
        super().__init__()
        self.n_paths    = n_paths
        self.lsh_dim    = lsh_dim
        self.capacity   = capacity

        # Random projection matrix for LSH hashing (not trained)
        self.register_buffer(
            "lsh_proj",
            torch.randn(hidden_dim, lsh_dim) / math.sqrt(lsh_dim),
        )

        # Registry: state_hash_str → list of _Node, one per path
        self._registry: Dict[str, List[_Node]] = {}

        # Ring buffer: recent (hash, fired_path_indices) for live updating
        self._history: deque = deque(maxlen=history_len)

        # Aggregate path priors across all states (smoothed)
        # Shape (n_paths,) — updated with each observe_outcome call
        self.register_buffer("path_priors", torch.ones(n_paths) * 0.5)

    # -----------------------------------------------------------------------
    # State hashing
    # -----------------------------------------------------------------------

    def _hash(self, state: torch.Tensor) -> str:
        """
        LSH hash of a hidden state vector.
        state: (H,) or (B, H) — if batched, hash the mean.
        Returns a hex string.
        """
        with torch.no_grad():
            if state.dim() > 1:
                state = state.float().mean(0)
            proj = (state.float() @ self.lsh_proj.float())   # (lsh_dim,)
            bits = (proj > 0).cpu().numpy().tobytes()
        return hashlib.md5(bits).hexdigest()[:16]

    def _get_nodes(self, state_hash: str) -> List[_Node]:
        """Retrieve or create nodes for a state hash."""
        if state_hash not in self._registry:
            if len(self._registry) >= self.capacity:
                # Evict the least-visited state
                victim = min(
                    self._registry,
                    key=lambda k: sum(n.visit_count for n in self._registry[k]),
                )
                del self._registry[victim]
            self._registry[state_hash] = [_Node() for _ in range(self.n_paths)]
        return self._registry[state_hash]

    # -----------------------------------------------------------------------
    # Record / update
    # -----------------------------------------------------------------------

    def record(self, state: torch.Tensor, fired_indices: List[int]) -> str:
        """Record a routing decision before the outcome is known."""
        h = self._hash(state)
        self._get_nodes(h)                          # ensure nodes exist
        self._history.append((h, fired_indices))
        return h

    def observe_outcome(self, reward: float) -> None:
        """
        Live policy update: push the outcome reward back onto the most recent
        decision recorded via record().
        """
        if not self._history:
            return
        state_hash, fired_indices = self._history[-1]
        nodes = self._get_nodes(state_hash)
        for idx in fired_indices:
            if 0 <= idx < self.n_paths:
                nodes[idx].update(reward)
        # Refresh aggregate priors
        self._refresh_priors()

    def _refresh_priors(self):
        """Recompute aggregate path_priors from all nodes (mean of Beta means)."""
        if not self._registry:
            return
        totals = [0.0] * self.n_paths
        counts = [0]   * self.n_paths
        for nodes in self._registry.values():
            for i, node in enumerate(nodes):
                totals[i] += node.mean
                counts[i] += 1
        with torch.no_grad():
            for i in range(self.n_paths):
                if counts[i] > 0:
                    self.path_priors[i] = totals[i] / counts[i]

    # -----------------------------------------------------------------------
    # Bayesian logit adjustment (Thompson Sampling)
    # -----------------------------------------------------------------------

    def bayesian_logit_adjustment(self, state: torch.Tensor) -> torch.Tensor:
        """
        Returns (n_paths,) additive logit adjustments based on Thompson
        Sampling from the Beta distribution of each path at the current state.

        Values are log-probability scale so they can be directly added to
        gate logits before softmax.
        """
        h = self._hash(state)
        nodes = self._get_nodes(h)

        samples = torch.tensor(
            [node.thompson_sample() for node in nodes],
            dtype=torch.float32,
            device=self.path_priors.device,
        )
        # log-prob: log(p) so high-success paths get positive adjustment
        adj = torch.log(samples.clamp(min=1e-6))
        # Normalise to zero-mean so we don't shift the total activation level
        adj = adj - adj.mean()
        return adj

    # -----------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------

    def path_stats(self) -> Dict[str, Dict]:
        """Return per-path aggregate statistics across all states."""
        stats: Dict[str, Dict] = {}
        for path_idx in range(self.n_paths):
            nodes = [
                self._registry[h][path_idx]
                for h in self._registry
                if path_idx < len(self._registry[h])
            ]
            if not nodes:
                stats[f"path_{path_idx}"] = {"mean": 0.5, "visits": 0}
                continue
            mean     = sum(n.mean          for n in nodes) / len(nodes)
            visits   = sum(n.visit_count   for n in nodes)
            stats[f"path_{path_idx}"] = {"mean": mean, "visits": visits}
        return stats

    def __len__(self) -> int:
        return len(self._registry)
