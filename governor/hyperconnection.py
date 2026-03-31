"""
governor/hyperconnection.py
============================
HyperConnectionLayer — Multi-Head Cross-Attention residual hyperconnections.

Each module output attends to all other module outputs, enriching its
representation with cross-module information while preserving identity
through a residual path.

Orthogonality
  A diversity loss identical in spirit to FiringRouter.diversity_loss()
  is applied to the cross-attention weight matrices, encouraging each
  module to attend to *different* aspects of the module pool — preventing
  collapse where all modules attend to the same source.

Architecture
  For N modules with hidden_dim H:
    Q_i = W_Q_i · x_i            (B, H)
    K_j = W_K · x_j  ∀ j        (N, B, H) — shared key projection
    V_j = W_V · x_j  ∀ j        (N, B, H) — shared value projection
    a_i = softmax(Q_i · K^T / √H_head)   (B, N)
    c_i = Σ_j a_i_j · V_j               (B, H)
    y_i = x_i + gate_i · c_i            residual blend

  gate_i is a learned scalar per module (sigmoid-initialised to ~0.1 so
  the hyperconnection starts weak and grows as training proceeds).

Usage
-----
    hc = HyperConnectionLayer(module_names=["nlp","vision","audio"], hidden_dim=896)

    # In the forward pass after all module outputs are computed:
    enriched = hc({"nlp": h_nlp, "vision": h_vision, "audio": h_audio})
    # enriched["nlp"], enriched["vision"], ... each is (B, H)

    # Orthogonality loss (add to total loss during training):
    loss += hc.diversity_loss(enriched)
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperConnectionLayer(nn.Module):
    """
    Multi-Head Cross-Attention residual hyperconnection layer.

    Parameters
    ----------
    module_names : ordered list of module names that participate
    hidden_dim   : shared hidden dimension H
    num_heads    : number of attention heads (must divide hidden_dim)
    gate_init    : initial logit for the residual blend gate
                   (sigmoid(gate_init) ≈ 0.12 by default — weak start)
    """

    def __init__(
        self,
        module_names: List[str],
        hidden_dim: int = 896,
        num_heads: int = 8,
        gate_init: float = -2.0,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.module_names = list(module_names)
        self.hidden_dim   = hidden_dim
        self.num_heads    = num_heads
        self.head_dim     = hidden_dim // num_heads
        self.n            = len(module_names)
        self._name_to_idx = {name: i for i, name in enumerate(module_names)}

        # Per-module query projections (so each module attends differently)
        self.W_Q = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, hidden_dim, bias=False) for name in module_names}
        )

        # Shared key / value projections (pool keys from all modules jointly)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Per-module output projection (after attention aggregation)
        self.W_O = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, hidden_dim, bias=False) for name in module_names}
        )

        # Per-module residual gate: y_i = x_i + sigmoid(gate_i) * c_i
        self.gates = nn.ParameterDict(
            {name: nn.Parameter(torch.tensor(gate_init)) for name in module_names}
        )

        # LayerNorm per module (post-residual)
        self.norms = nn.ModuleDict(
            {name: nn.LayerNorm(hidden_dim) for name in module_names}
        )

        self.scale = self.head_dim ** -0.5

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        module_outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        module_outputs : dict name → (B, H) tensors
                         (missing modules are skipped gracefully)
        Returns        : dict name → (B, H) enriched tensors
        """
        # Only process names we know; skip unknowns
        active = [n for n in self.module_names if n in module_outputs]
        if not active:
            return module_outputs

        # Stack all active outputs: (N_active, B, H)
        stacked = torch.stack([module_outputs[n] for n in active], dim=0)  # (N, B, H)
        N, B, H = stacked.shape

        # Shared K, V across all active modules: (N, B, H)
        K_all = self.W_K(stacked)   # (N, B, H)
        V_all = self.W_V(stacked)   # (N, B, H)

        # Reshape for multi-head: (N, B, n_heads, head_dim)
        def to_heads(t: torch.Tensor) -> torch.Tensor:
            # t: (N, B, H) → (N, B, n_heads, head_dim)
            return t.view(N, B, self.num_heads, self.head_dim)

        K_h = to_heads(K_all)   # (N, B, n_heads, head_dim)
        V_h = to_heads(V_all)   # (N, B, n_heads, head_dim)

        enriched: Dict[str, torch.Tensor] = {}

        for name in active:
            x_i = module_outputs[name]  # (B, H)

            # Per-module query
            q = self.W_Q[name](x_i)    # (B, H)
            q = q.view(B, self.num_heads, self.head_dim)  # (B, n_heads, head_dim)

            # Attention scores: q × K^T
            # q: (B, n_heads, head_dim) × K_h: (N, B, n_heads, head_dim)
            # → (B, n_heads, N)
            scores = torch.einsum("bnh,nbnh->bnn2", q, K_h)
            # Simpler explicit einsum: (B, n_heads, N)
            scores = torch.einsum("bnh,ibnh->bin", q, K_h) * self.scale   # (B, n_heads, N)
            # Rearrange: (B, n_heads, N)
            attn = torch.softmax(scores, dim=-1)   # (B, n_heads, N)

            # Weighted sum of values: (B, n_heads, head_dim)
            # V_h: (N, B, n_heads, head_dim) → need (B, n_heads, N, head_dim)
            V_perm = V_h.permute(1, 2, 0, 3)       # (B, n_heads, N, head_dim)
            c = (attn.unsqueeze(-1) * V_perm).sum(dim=2)  # (B, n_heads, head_dim)
            c = c.reshape(B, H)                     # (B, H)

            # Output projection + residual gate
            c_proj  = self.W_O[name](c)             # (B, H)
            gate    = torch.sigmoid(self.gates[name])
            y       = x_i + gate * c_proj           # (B, H)
            enriched[name] = self.norms[name](y)    # (B, H)

        # Pass through any names not in our module list unchanged
        for name, val in module_outputs.items():
            if name not in enriched:
                enriched[name] = val

        return enriched

    # -----------------------------------------------------------------------
    # Diversity / orthogonality loss
    # -----------------------------------------------------------------------

    def diversity_loss(
        self,
        module_outputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Penalise correlated cross-module representations.

        Stacks all active module outputs, column-normalises, computes the
        Gram matrix, and returns ||G − I||_F.  Encourages each module's
        output to be orthogonal to every other module's output.
        """
        active = [n for n in self.module_names if n in module_outputs]
        if len(active) < 2:
            device = next(iter(module_outputs.values())).device
            return torch.tensor(0.0, device=device)

        # Each x: (B, H) → mean over batch → (H,)
        vecs = torch.stack(
            [module_outputs[n].mean(0) for n in active], dim=0
        )  # (N, H)
        vecs = F.normalize(vecs, dim=-1)    # (N, H)
        gram = vecs @ vecs.t()              # (N, N)
        I    = torch.eye(gram.size(0), device=gram.device)
        return (gram - I).pow(2).mean()

    # -----------------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------------

    def gate_stats(self) -> Dict[str, float]:
        """Current blend gate magnitudes per module (for logging)."""
        return {
            name: float(torch.sigmoid(self.gates[name]))
            for name in self.module_names
        }
