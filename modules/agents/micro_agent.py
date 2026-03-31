"""
modules/agents/micro_agent.py
==============================
MicroAgent — lightweight task executor with a state-based LM backbone.

The default LM is the HybridSSMTransformer (Mamba2+GRU+SlidingAttention).
It is passed in from the parent ONI model so weights are shared, not duplicated.

MicroAgent instances are spawned by the governor per-task and hold only:
  - a reference to the shared state LM
  - a module subset relevant to the task
  - a compressed local_state slice from MemoryEnsemble
  - an optional rule callable and goal
"""

from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn


class MicroAgent(nn.Module):
    """
    Ephemeral task executor.

    Parameters
    ----------
    hidden_dim  : shared hidden dimension
    lm          : shared state-based LM (HybridSSMTransformer); if None a
                  small default SSM is constructed (not recommended for
                  production — pass the parent model's LM)
    modules     : dict of name → nn.Module for task-specific tools
    local_state : compressed context dict from MemoryEnsemble
    rule        : callable(modules, local_state, goal) → delta dict
    goal        : goal embedding (Tensor) or goal string
    """

    def __init__(
        self,
        hidden_dim: int = 896,
        lm: Optional[nn.Module] = None,
        modules: Optional[Dict[str, nn.Module]] = None,
        local_state: Optional[Dict[str, Any]] = None,
        rule: Optional[Callable] = None,
        goal: Any = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.local_state = local_state or {}
        self.rule = rule
        self.goal = goal
        self.ext_modules = modules or {}

        # --- State-based LM backbone ---
        if lm is not None:
            # Shared weights — don't re-register as a submodule to avoid
            # duplicating parameters in the parent model's parameter list.
            self._lm = lm
        else:
            # Fallback: small standalone SSM (6 layers)
            from modules.NLP.oni_nlp_transformer import HybridSSMTransformer
            self._lm = HybridSSMTransformer(
                vocab_size=32000,
                hidden_dim=hidden_dim,
                num_layers=6,
                num_heads=8,
                num_kv_heads=2,
                window_size=256,
                d_state=64,
                d_inner=hidden_dim * 2,
                dt_rank=64,
            )

    @property
    def lm(self) -> nn.Module:
        return self._lm

    # -----------------------------------------------------------------------
    # Goal encoding
    # -----------------------------------------------------------------------

    def encode_goal(self, goal: Any) -> torch.Tensor:
        """Return a (1, 1, H) hidden state for the goal using the state LM."""
        if isinstance(goal, torch.Tensor):
            x = goal
            if x.dim() == 1:
                x = x.unsqueeze(0).unsqueeze(0)
            elif x.dim() == 2:
                x = x.unsqueeze(0)
            return self._lm(x)
        # String or unknown — return zeros; caller should tokenise
        return torch.zeros(1, 1, self.hidden_dim)

    # -----------------------------------------------------------------------
    # Execution
    # -----------------------------------------------------------------------

    def run(self) -> Dict:
        """Execute the assigned rule, falling back to default_rule."""
        if self.rule is not None:
            return self.rule(self.ext_modules, self.local_state, self.goal)
        return self.default_rule()

    def default_rule(self) -> Dict:
        """Encode the goal via the state LM and return the hidden state."""
        if self.goal is not None:
            state = self.encode_goal(self.goal)
            return {"lm_state": state, "task": "default_encode"}
        return {"task": "no_goal"}

    # -----------------------------------------------------------------------
    # nn.Module forward (for direct training use)
    # -----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._lm(x)
