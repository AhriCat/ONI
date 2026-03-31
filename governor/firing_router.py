"""
governor/firing_router.py
==========================
StochasticFiringRouter — the ONI Governor.

A multi-AI orthogonal network with a stochastic firing routing mechanism.
Integrates:
  MarkovDecisionTree  : past decision / outcome registry with Bayesian
                        (Thompson Sampling) logit adjustment
  Gumbel-Softmax      : differentiable stochastic path sampling during
                        training; hard top-k at inference
  Live policy update  : observe_outcome(reward) propagates reward back
                        to the Markov tree without requiring gradients

Design properties
-----------------
  Sparse activation   : only top-k paths fire per forward pass
  Firing threshold    : paths below the threshold are suppressed entirely
  Stochastic training : Gumbel noise → differentiable discrete choices
  Bayesian priors     : Thompson Sampling biases routing toward
                        historically successful paths for the current state
  Orthogonality loss  : diversity_loss() encourages path specialisation
  Residual blend      : learned α blends router output with passthrough
  Live updating       : no gradient required — count-based Bayesian update

Backward-compat alias
  FiringRouter = StochasticFiringRouter
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from governor.markov_tree import MarkovDecisionTree


class PathGate(nn.Module):
    """Compact gate: hidden_dim → n_paths firing logits."""

    def __init__(self, hidden_dim: int, n_paths: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_paths),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (B, n_paths) — caller applies softmax/gumbel."""
        return self.proj(x)


class StochasticFiringRouter(nn.Module):
    """
    Multi-AI orthogonal network router (ONI Governor) — stochastic variant.

    Parameters
    ----------
    hidden_dim        : shared hidden dimension for all paths
    paths             : dict of name → nn.Module; each path must accept
                        (B, hidden_dim) and return (B, hidden_dim)
    top_k             : maximum number of paths that can fire simultaneously
    firing_threshold  : minimum gate probability for a path to fire
    gumbel_tau        : initial Gumbel-Softmax temperature (anneals toward 0)
    tau_min           : minimum temperature floor
    markov_lsh_dim    : LSH dimension for MarkovDecisionTree state hashing
    markov_capacity   : capacity of the Markov registry
    markov_history    : rolling window length for live updating

    Forward returns
    ---------------
    output      : (B, hidden_dim) — weighted sum of fired paths + residual
    gate_weights: (B, n_paths)   — routing probabilities (soft or hard)
    fired_names : List[str]      — names of paths that fired this step
    """

    def __init__(
        self,
        hidden_dim: int,
        paths: Dict[str, nn.Module],
        top_k: int = 2,
        firing_threshold: float = 0.1,
        gumbel_tau: float = 1.0,
        tau_min: float = 0.1,
        markov_lsh_dim: int = 64,
        markov_capacity: int = 8192,
        markov_history: int = 128,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.firing_threshold = firing_threshold
        self.tau = gumbel_tau
        self.tau_min = tau_min
        self.path_names = list(paths.keys())
        n_paths = len(paths)

        self.paths = nn.ModuleDict(paths)

        self.gate = PathGate(hidden_dim, n_paths)

        # Per-path output projection back to hidden_dim
        self.out_projs = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, hidden_dim, bias=False) for name in paths}
        )

        # Learned blend coefficient α: output = α*routed + (1-α)*passthrough
        self.blend = nn.Parameter(torch.tensor(0.0))   # sigmoid(0) = 0.5

        # Markov decision tree with Bayesian Thompson Sampling
        self.markov = MarkovDecisionTree(
            n_paths=n_paths,
            hidden_dim=hidden_dim,
            lsh_dim=markov_lsh_dim,
            capacity=markov_capacity,
            history_len=markov_history,
        )

        # Last recorded state hash (for deferred observe_outcome calls)
        self._last_hash: Optional[str] = None
        self._last_fired_indices: List[int] = []

    # -----------------------------------------------------------------------
    # Temperature annealing (call from training loop)
    # -----------------------------------------------------------------------

    def anneal_temperature(self, factor: float = 0.995) -> float:
        """Multiply tau by factor, clamp to tau_min. Returns new tau."""
        self.tau = max(self.tau_min, self.tau * factor)
        return self.tau

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        x       : (B, hidden_dim)
        training: if True use Gumbel-Softmax; if False hard top-k
        Returns : (output, gate_weights, fired_names)
        """
        B, H = x.shape
        device = x.device

        # Raw gate logits
        logits = self.gate(x)                                       # (B, n_paths)

        # Bayesian adjustment from Markov tree (additive in log-prob space)
        # Use the mean hidden state as the state signature
        state_sig = x.mean(0) if B > 1 else x.squeeze(0)
        bayes_adj = self.markov.bayesian_logit_adjustment(state_sig)  # (n_paths,)
        logits = logits + bayes_adj.unsqueeze(0)                    # broadcast (B, n_paths)

        if training:
            # Gumbel-Softmax: differentiable stochastic top-k
            gate_weights = F.gumbel_softmax(logits, tau=self.tau, hard=False)
        else:
            gate_weights = torch.softmax(logits, dim=-1)

        k = min(self.top_k, len(self.path_names))
        top_k_vals, top_k_indices = gate_weights.topk(k, dim=-1)   # (B, k)

        combined      = torch.zeros(B, H, device=device)
        total_weight  = torch.zeros(B, device=device)
        fired_set: set = set()
        fired_indices: List[int] = []

        for k_pos in range(k):
            path_indices = top_k_indices[:, k_pos]     # (B,)
            path_weights  = top_k_vals[:, k_pos]       # (B,)
            above_thresh  = path_weights > self.firing_threshold

            for path_idx in path_indices[above_thresh].unique().tolist():
                path_idx = int(path_idx)
                path_name = self.path_names[path_idx]
                fired_set.add(path_name)
                if path_idx not in fired_indices:
                    fired_indices.append(path_idx)

                batch_mask = above_thresh & (path_indices == path_idx)
                if not batch_mask.any():
                    continue

                path_out = self.paths[path_name](x[batch_mask])         # (m, H)
                proj_out = self.out_projs[path_name](path_out)          # (m, H)
                w = gate_weights[batch_mask, path_idx].unsqueeze(-1)
                combined[batch_mask]     += w * proj_out
                total_weight[batch_mask] += gate_weights[batch_mask, path_idx]

        # Normalise fired outputs; passthrough where nothing fired
        fired_mask = total_weight > 0
        if fired_mask.any():
            combined[fired_mask] = (
                combined[fired_mask] / total_weight[fired_mask].unsqueeze(-1)
            )
        combined[~fired_mask] = x[~fired_mask]

        # Residual blend
        alpha  = torch.sigmoid(self.blend)
        output = alpha * combined + (1.0 - alpha) * x

        # Record decision in Markov tree for subsequent observe_outcome()
        self._last_hash = self.markov.record(state_sig, fired_indices)
        self._last_fired_indices = fired_indices

        return output, gate_weights, sorted(fired_set)

    # -----------------------------------------------------------------------
    # Live policy updating (no gradient required)
    # -----------------------------------------------------------------------

    def observe_outcome(self, reward: float) -> None:
        """
        Push the scalar reward back to the Markov tree for the most recent
        routing decision.  Call this after evaluating the quality of the
        step that used the output of the last forward() call.

        reward > 0 : success  → increases alpha (favours those paths)
        reward ≤ 0 : failure  → increases beta  (disfavours those paths)
        """
        self.markov.observe_outcome(reward)

    # -----------------------------------------------------------------------
    # Diversity / orthogonality loss
    # -----------------------------------------------------------------------

    def diversity_loss(self, gate_weights: torch.Tensor) -> torch.Tensor:
        """
        Encourages paths to specialise by penalising correlated routing.
        L = ||W^T W − I||_F  (Frobenius norm of off-diagonal correlations)
        """
        W = gate_weights - gate_weights.mean(0, keepdim=True)
        W = F.normalize(W, dim=0)           # (B, n_paths) column-normalised
        gram = W.t() @ W                    # (n_paths, n_paths)
        I = torch.eye(gram.size(0), device=gram.device)
        return (gram - I).pow(2).mean()

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def routing_stats(self, gate_weights: torch.Tensor) -> Dict[str, float]:
        """Average gate probability per path (for logging)."""
        avg = gate_weights.mean(0)
        return {name: float(avg[i]) for i, name in enumerate(self.path_names)}

    def markov_stats(self) -> Dict[str, Dict]:
        """Pass-through to MarkovDecisionTree.path_stats()."""
        return self.markov.path_stats()

    def governor_summary(self, gate_weights: torch.Tensor) -> Dict:
        """Combined routing + Markov stats for a single logging call."""
        return {
            "routing": self.routing_stats(gate_weights),
            "markov":  self.markov_stats(),
            "tau":     self.tau,
            "blend":   float(torch.sigmoid(self.blend)),
            "n_states": len(self.markov),
        }


# Backward-compat alias
FiringRouter = StochasticFiringRouter
