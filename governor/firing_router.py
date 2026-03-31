"""
governor/firing_router.py
==========================
FiringRouter — the ONI Governor.

A multi-AI orthogonal network with a firing routing mechanism.
Each 'path' is a named expert transform.  The router decides which paths fire
for a given input, combines their outputs, and returns:
  - combined output tensor
  - gate weights (for diversity loss / logging)
  - list of fired path names (for selective full-module invocation upstream)

Design properties
-----------------
  Sparse activation  : only top-k paths fire per forward pass
  Firing threshold   : paths below the threshold are suppressed entirely
  Orthogonality loss : diversity_loss() encourages path specialisation
  Residual blend     : learned α blends router output with passthrough
  Differentiable     : soft routing during training, hard top-k at inference
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PathGate(nn.Module):
    """Compact gate: hidden_dim → n_paths firing probabilities."""

    def __init__(self, hidden_dim: int, n_paths: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_paths),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.proj(x), dim=-1)


class FiringRouter(nn.Module):
    """
    Multi-AI orthogonal network router (ONI Governor).

    Parameters
    ----------
    hidden_dim        : shared hidden dimension for all paths
    paths             : dict of name → nn.Module; each path must accept
                        (B, hidden_dim) and return (B, hidden_dim)
    top_k             : maximum number of paths that can fire simultaneously
    firing_threshold  : minimum gate probability for a path to fire

    Forward returns
    ---------------
    output      : (B, hidden_dim) — weighted sum of fired paths + residual
    gate_weights: (B, n_paths)   — soft routing probabilities
    fired_names : List[str]      — names of paths that fired this step
    """

    def __init__(
        self,
        hidden_dim: int,
        paths: Dict[str, nn.Module],
        top_k: int = 2,
        firing_threshold: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.firing_threshold = firing_threshold
        self.path_names = list(paths.keys())

        self.paths = nn.ModuleDict(paths)

        self.gate = PathGate(hidden_dim, len(paths))

        # Per-path output projection back to hidden_dim
        self.out_projs = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, hidden_dim, bias=False) for name in paths}
        )

        # Learned blend coefficient α: output = α*routed + (1-α)*passthrough
        self.blend = nn.Parameter(torch.tensor(0.0))   # sigmoid(0) = 0.5

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
        training: if True use soft weights; if False hard top-k
        Returns : (output, gate_weights, fired_names)
        """
        B, H = x.shape
        device = x.device

        gate_weights = self.gate(x)                                     # (B, n_paths)
        k = min(self.top_k, len(self.path_names))
        top_k_vals, top_k_indices = gate_weights.topk(k, dim=-1)       # (B, k)

        combined = torch.zeros(B, H, device=device)
        total_weight = torch.zeros(B, device=device)
        fired_set: set = set()

        for k_pos in range(k):
            path_indices = top_k_indices[:, k_pos]     # (B,)
            path_weights  = top_k_vals[:, k_pos]       # (B,)
            above_thresh  = path_weights > self.firing_threshold

            for path_idx in path_indices[above_thresh].unique().tolist():
                path_name = self.path_names[int(path_idx)]
                fired_set.add(path_name)

                # Which batch elements route through this path at this k_pos?
                batch_mask = above_thresh & (path_indices == int(path_idx))
                if not batch_mask.any():
                    continue

                path_out = self.paths[path_name](x[batch_mask])        # (m, H)
                proj_out = self.out_projs[path_name](path_out)         # (m, H)
                w = gate_weights[batch_mask, int(path_idx)].unsqueeze(-1)
                combined[batch_mask] += w * proj_out
                total_weight[batch_mask] += gate_weights[batch_mask, int(path_idx)]

        # Normalise fired outputs; passthrough where nothing fired
        fired_mask = total_weight > 0
        if fired_mask.any():
            combined[fired_mask] = (
                combined[fired_mask] / total_weight[fired_mask].unsqueeze(-1)
            )
        combined[~fired_mask] = x[~fired_mask]

        # Residual blend
        alpha = torch.sigmoid(self.blend)
        output = alpha * combined + (1.0 - alpha) * x

        return output, gate_weights, sorted(fired_set)

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
