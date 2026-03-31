"""
modules/planner/gru_planner.py
===============================
GRU-based task planner with goal encoding and autoregressive decomposition.

Replaces the LLM-based Planner.  No LLM required at planning time — a small
learned GRU encodes goals and emits typed subtask embeddings.

Architecture
------------
  goal_encoder  : 2-layer GRU    (goal_emb_seq → hidden state)
  decomp_gru    : 1-layer GRU    (autoregressive subtask generation)
  n_head        : Linear         (predicts how many subtasks, 1..max_subtasks)
  type_head     : Linear         (classifies each subtask into a manifest type)
  stop_head     : Linear → σ     (per-step probability of stopping early)
  sos           : nn.Parameter   (learned start-of-sequence token)
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

TASK_TYPES = [
    "nlp_step",
    "vision_step",
    "audio_step",
    "memory_write",
    "homeostasis_step",
    "planning_step",
    "action_step",
    "default_step",
]


class GRUPlanner(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 896,
        max_subtasks: int = 8,
        n_task_types: int = len(TASK_TYPES),
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_subtasks = max_subtasks
        self.n_task_types = n_task_types

        # Goal encoder: collapses a variable-length goal embedding into a
        # single hidden state used to seed decomposition.
        self.goal_encoder = nn.GRU(
            hidden_dim, hidden_dim, num_layers=2, batch_first=True
        )

        # Autoregressive decomposition GRU
        self.decomp_gru = nn.GRU(
            hidden_dim, hidden_dim, num_layers=1, batch_first=True
        )

        # Predict number of subtasks (1-indexed output of argmax + 1)
        self.n_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, max_subtasks),
        )

        # Per-subtask type classifier
        self.type_head = nn.Linear(hidden_dim, n_task_types)

        # Soft stop: when > 0.85 the planner stops early
        self.stop_head = nn.Linear(hidden_dim, 1)

        # Start-of-sequence token for the decomp GRU
        self.sos = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.sos, std=0.02)

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _encode_goal(self, goal_emb: torch.Tensor) -> torch.Tensor:
        """
        goal_emb: (B, seq, H) or (B, H)
        Returns : (B, H) — final hidden state of goal encoder
        """
        if goal_emb.dim() == 2:
            goal_emb = goal_emb.unsqueeze(1)
        _, h = self.goal_encoder(goal_emb)   # h: (n_layers, B, H)
        return h[-1]                          # (B, H)

    def _decompose(self, goal_state: torch.Tensor) -> List[Dict]:
        """
        Autoregressively emit subtask dicts from goal_state.
        goal_state: (B, H)
        Returns: list of {'embedding', 'type', 'stop_prob'}
        """
        B, H = goal_state.shape
        device = goal_state.device

        n_subtasks = int(self.n_head(goal_state).argmax(-1).float().mean().item()) + 1
        n_subtasks = max(1, min(n_subtasks, self.max_subtasks))

        h = goal_state.unsqueeze(0)                          # (1, B, H) for GRU
        x = self.sos.expand(B, 1, H).to(device)

        subtasks: List[Dict] = []
        for _ in range(n_subtasks):
            out, h = self.decomp_gru(x, h)                  # out: (B, 1, H)
            emb = out.squeeze(1)                             # (B, H)
            type_idx = int(self.type_head(emb).argmax(-1).float().mean().item())
            stop_prob = float(torch.sigmoid(self.stop_head(emb)).mean().item())
            subtasks.append(
                {
                    "embedding": emb.detach(),
                    "type": TASK_TYPES[type_idx],
                    "stop_prob": stop_prob,
                }
            )
            if stop_prob > 0.85:
                break
            x = emb.unsqueeze(1)

        return subtasks

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def plan(self, goal_emb: torch.Tensor) -> List[Dict]:
        """Full pipeline: encode goal → decompose. Returns list of subtask dicts."""
        goal_state = self._encode_goal(goal_emb)
        return self._decompose(goal_state)

    def forward(
        self, goal_emb: torch.Tensor
    ) -> Tuple[List[Dict], torch.Tensor]:
        """Returns (subtasks, goal_state) — goal_state usable for training losses."""
        goal_state = self._encode_goal(goal_emb)
        subtasks = self._decompose(goal_state)
        return subtasks, goal_state
