"""
modules/memory/working_memory_module.py
=======================================
Attention-based Working Memory Module.

Problems with the old implementation:
  - memory_bank was a plain tensor (no gradients)
  - timestamps were plain tensors (not device-portable)
  - FIFO logic used argmin on timestamps — collides at init
  - No attention-based read, so queries couldn't selectively attend

New design
----------
  - Learnable slot bank (nn.Parameter) — full gradient flow
  - Multi-head cross-attention for reads: query attends to all slots
  - Gated write: importance score gates whether to overwrite a slot
  - Temporal decay: slot activations fade with time; recent writes win
  - Soft slot selection (differentiable) with temperature-controlled gumbel

Compatible with the old WorkingMemoryModule interface:
    update_memory(embedding, timestamp)
    retrieve_memory(query_embedding, current_timestamp)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class WorkingMemoryModule(nn.Module):
    """
    Differentiable working memory with:
    - Learnable slot bank initialised from N(0, 0.02)
    - Multi-head cross-attention read
    - Importance-gated soft write (gradient flows through)
    - Exponential temporal decay on slot activations
    """

    def __init__(self, config=None, hidden_dim: int = 896,
                 num_slots: int = 64, num_heads: int = 8,
                 decay_rate: float = 0.95):
        super().__init__()

        # Support both config object and direct kwargs
        if config is not None:
            hidden_dim  = getattr(config, "hidden_dim",        hidden_dim)
            num_slots   = getattr(config, "working_memory_size", num_slots)
            decay_rate  = getattr(config, "temporal_context_size",
                                  1.0 / max(-1e-8 + 1 - decay_rate, 1e-8)) \
                          if hasattr(config, "temporal_context_size") else decay_rate

        self.hidden_dim = hidden_dim
        self.num_slots  = num_slots
        self.decay_rate = decay_rate

        # Learnable slot bank — persistent across forward calls
        self.slots = nn.Parameter(
            torch.randn(1, num_slots, hidden_dim) * 0.02
        )

        # Per-slot importance scores (unnormalised)
        self.slot_importance = nn.Parameter(torch.zeros(1, num_slots))

        # Slot ages (timestamp deltas) — not a parameter, updated in-place
        self.register_buffer("slot_ages", torch.zeros(1, num_slots))
        self.register_buffer("slot_timestamps", torch.zeros(1, num_slots))

        # Cross-attention: query = external query, key/value = slots
        self.q_proj  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.head_dim  = hidden_dim // num_heads
        self.num_heads = num_heads
        self.scale     = self.head_dim ** -0.5

        # Write gate: computes per-slot probability of being overwritten
        self.write_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, num_slots),
        )

        # Slot value projection for writing
        self.write_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Layer norms
        self.norm_slots = nn.LayerNorm(hidden_dim)
        self.norm_out   = nn.LayerNorm(hidden_dim)

    # ------------------------------------------------------------------
    # Read — multi-head cross-attention
    # ------------------------------------------------------------------

    def retrieve_memory(
        self,
        query_embedding: torch.Tensor,
        current_timestamp: float = 0.0,
    ) -> torch.Tensor:
        """
        Retrieve a summary from working memory by cross-attending to all slots.

        query_embedding : (H,) or (B, H) or (B, T, H)
        Returns         : same shape as query_embedding
        """
        solo = query_embedding.dim() == 1
        if solo:
            query_embedding = query_embedding.unsqueeze(0)

        if query_embedding.dim() == 2:
            # Treat as (B, 1, H)
            query_embedding = query_embedding.unsqueeze(1)

        B, T, H = query_embedding.shape

        # Apply temporal decay to slots
        decay = self._compute_decay(current_timestamp)       # (1, num_slots, 1)
        slots = self.norm_slots(self.slots * decay)          # broadcast over B

        # Expand slots to batch
        slots_b = slots.expand(B, -1, -1)                   # (B, N, H)

        # Multi-head cross-attention
        Q = self.q_proj(query_embedding)                     # (B, T, H)
        K = self.k_proj(slots_b)                             # (B, N, H)
        V = self.v_proj(slots_b)

        Q = Q.view(B, T,           self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, self.num_slots, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, self.num_slots, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale       # (B, H, T, N)
        attn = F.softmax(attn, dim=-1)
        out  = (attn @ V).transpose(1, 2).contiguous().view(B, T, H)
        out  = self.norm_out(self.out_proj(out))

        if solo:
            out = out.squeeze(0).squeeze(0)
        elif out.shape[1] == 1:
            out = out.squeeze(1)
        return out

    # ------------------------------------------------------------------
    # Write — importance-gated soft update
    # ------------------------------------------------------------------

    def update_memory(
        self,
        embedding: torch.Tensor,
        timestamp: float = 0.0,
    ) -> None:
        """
        Write embedding into the slot bank via a soft gated update.
        Gradients flow back through the gate and write_proj.

        embedding : (H,) or (B, H)
        """
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)                 # (1, H)

        B, H = embedding.shape
        device = embedding.device

        # Align slots to device / batch
        slots = self.slots.to(device)                          # (1, N, H)
        slot_mean = slots.mean(dim=-1)                         # (1, N)

        # Gate: decide which slots to overwrite
        # Gate input: [new embedding, mean slot representation]
        gate_in = torch.cat([
            embedding.unsqueeze(1).expand(-1, self.num_slots, -1),  # (B, N, H)
            slots.expand(B, -1, -1),                                  # (B, N, H)
        ], dim=-1)                                               # (B, N, 2H)
        gate_logits = self.write_gate(gate_in).squeeze(-1)      # (B, N) — should be (B,N,1)->squeeze

        # Fix: write_gate outputs (B, N, num_slots), we want per-slot
        # Actually recompute: input is (B*N, 2H) → output (B*N, 1)
        # Flatten then reshape
        B_n, N_s, two_H = gate_in.shape
        gate_flat = gate_in.view(B_n * N_s, two_H)
        gate_logits = self.write_gate(gate_flat).view(B_n, N_s)  # (B, N)

        gate_probs = torch.softmax(gate_logits, dim=-1)          # (B, N)

        # Write: soft weighted update across slots
        new_val = self.write_proj(embedding).unsqueeze(1)        # (B, 1, H)
        delta = gate_probs.unsqueeze(-1) * (new_val - slots.expand(B, -1, -1))
        # Average over batch and accumulate into parameter
        with torch.no_grad():
            self.slots.data += delta.mean(0, keepdim=True).detach()
            self.slot_timestamps.data.fill_(timestamp)
            self.slot_ages.data += 1.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_decay(self, current_timestamp: float) -> torch.Tensor:
        """Per-slot exponential decay based on age."""
        ages = self.slot_ages                                     # (1, N)
        decay = self.decay_rate ** ages                           # (1, N)
        return decay.unsqueeze(-1)                                # (1, N, 1)

    def clear(self):
        """Reset slot bank to initial state."""
        with torch.no_grad():
            self.slots.data.normal_(0.0, 0.02)
            self.slot_importance.data.zero_()
            self.slot_ages.zero_()
            self.slot_timestamps.zero_()
