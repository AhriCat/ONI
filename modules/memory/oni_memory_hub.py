"""
modules/memory/oni_memory_hub.py
================================
ONIMemoryHub — coherent three-tier memory system.

Tier 1 — Working Memory  (fast, volatile, attention-based)
    • ~64 slots, full gradient flow
    • Stores the last N activations / conversation turns

Tier 2 — Episodic Memory  (medium, indexed by content + recency)
    • Fixed capacity ring-buffer with importance scoring
    • Retrieved via cosine similarity + recency weighting
    • Consolidation: high-importance items migrate to Semantic

Tier 3 — Semantic Memory  (slow, abstract, persistent)
    • Key-value store keyed by a learned embedding
    • Background consolidation via MemoryConsolidator-style pruning
    • Supports batch cosine nearest-neighbor lookup

MemoryRouter:
    - Decides how much each tier contributes to the final read
    - Learned gating: query → softmax([w_work, w_episodic, w_semantic])
    - Weighted sum returned to ONI main model

Design goals:
    - Zero hard-coded file paths (was a problem in oni_memory.py)
    - No pygame or external rendering deps
    - All operations differentiable (gradients flow through reads)
    - Consolidation is a method call, not a background thread
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict


# ============================================================================
# Tier 1: Working Memory (from working_memory_module — re-exported here)
# ============================================================================

from .working_memory_module import WorkingMemoryModule


# ============================================================================
# Tier 2: Episodic Memory
# ============================================================================

class EpisodicMemory(nn.Module):
    """
    Content-addressable episodic store with:
    - Ring-buffer storage (constant size, O(1) write)
    - Cosine-similarity retrieval + recency exponential discount
    - Importance score: access_count + emotional_salience (if provided)
    - Consolidation signal: returns items above importance threshold
    """

    def __init__(
        self,
        hidden_dim: int,
        capacity: int = 4096,
        top_k: int = 8,
        recency_decay: float = 0.99,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.capacity   = capacity
        self.top_k      = top_k
        self.recency_decay = recency_decay

        # Storage
        self.register_buffer("store",       torch.zeros(capacity, hidden_dim))
        self.register_buffer("importance",  torch.zeros(capacity))
        self.register_buffer("timestamps",  torch.zeros(capacity))
        self.register_buffer("ptr",         torch.zeros(1, dtype=torch.long))
        self.register_buffer("size",        torch.zeros(1, dtype=torch.long))

        # Key projection (for a learned query space different from raw hidden)
        self.key_proj   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj   = nn.Linear(hidden_dim, hidden_dim, bias=False)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(
        self,
        embedding: torch.Tensor,       # (H,) or (B, H)
        importance: float = 1.0,
        timestamp: float = 0.0,
    ) -> None:
        """Write embedding(s) into the ring buffer."""
        if embedding.dim() == 2:
            for i in range(embedding.shape[0]):
                self.write(embedding[i], importance, timestamp)
            return

        idx = self.ptr[0].item()
        with torch.no_grad():
            self.store[idx]      = embedding.detach()
            self.importance[idx] = importance
            self.timestamps[idx] = timestamp
            self.ptr[0]          = (idx + 1) % self.capacity
            self.size[0]         = min(self.size[0].item() + 1, self.capacity)

    # ------------------------------------------------------------------
    # Read (differentiable top-k soft retrieval)
    # ------------------------------------------------------------------

    def read(
        self,
        query: torch.Tensor,             # (H,) or (B, H) or (B, T, H)
        current_timestamp: float = 0.0,
    ) -> torch.Tensor:
        """
        Retrieve a weighted sum of the top-k episodic entries.
        Returns same shape as query.
        """
        squeeze_T = query.dim() == 3
        squeeze_B = query.dim() == 1

        if squeeze_B:
            query = query.unsqueeze(0)
        if squeeze_T:
            B, T, H = query.shape
            query = query.view(B * T, H)

        B, H = query.shape
        n = self.size[0].item()
        if n == 0:
            out = torch.zeros_like(query)
        else:
            # Keys from store
            store = self.store[:n]                              # (n, H)
            keys  = self.key_proj(store)                        # (n, H)
            vals  = self.value_proj(store)                      # (n, H)

            # Recency weights
            ages   = current_timestamp - self.timestamps[:n]   # (n,)
            recency = torch.exp(-ages.abs() * (1 - self.recency_decay))
            imp     = self.importance[:n]
            weights = recency * (imp + 1.0)                    # (n,)
            weights = weights / (weights.sum() + 1e-8)

            # Cosine similarity scores
            q_norm = F.normalize(query, dim=-1)                 # (B, H)
            k_norm = F.normalize(keys,  dim=-1)                 # (n, H)
            sim    = q_norm @ k_norm.t()                        # (B, n)

            # Weighted similarity → attention
            sim    = sim * weights.unsqueeze(0)
            k_ret  = min(self.top_k, n)
            topk_sim, topk_idx = sim.topk(k_ret, dim=-1)       # (B, k)
            attn = F.softmax(topk_sim, dim=-1)                  # (B, k)
            topk_vals = vals[topk_idx]                          # (B, k, H)
            out = (attn.unsqueeze(-1) * topk_vals).sum(1)       # (B, H)
            out = self.out_proj(out)

        if squeeze_T:
            out = out.view(B // (T if squeeze_T else 1), T if squeeze_T else 1, H)
        if squeeze_B:
            out = out.squeeze(0)
        return out

    def get_consolidation_candidates(
        self, threshold: float = 5.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return high-importance items for migration to semantic memory."""
        n = self.size[0].item()
        if n == 0:
            return torch.empty(0), torch.empty(0)
        mask = self.importance[:n] > threshold
        return self.store[:n][mask], self.importance[:n][mask]


# ============================================================================
# Tier 3: Semantic Memory
# ============================================================================

class SemanticMemory(nn.Module):
    """
    Key-value semantic store with nearest-neighbour lookup.

    Keys are compressed representations of concepts.
    Values are the associated rich representations.
    Supports gradual capacity growth via consolidation writes.
    """

    def __init__(
        self,
        hidden_dim: int,
        capacity: int = 16384,
        top_k: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.capacity   = capacity
        self.top_k      = top_k

        self.register_buffer("keys",    torch.zeros(capacity, hidden_dim))
        self.register_buffer("values",  torch.zeros(capacity, hidden_dim))
        self.register_buffer("filled",  torch.zeros(capacity, dtype=torch.bool))
        self.register_buffer("ptr",     torch.zeros(1, dtype=torch.long))

        # Learned key/query projections
        self.q_proj   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def write(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Write a single (key, value) pair."""
        if key.dim() == 2:
            for i in range(key.shape[0]):
                self.write(key[i], value[i] if value.dim() == 2 else value)
            return
        idx = self.ptr[0].item() % self.capacity
        with torch.no_grad():
            self.keys[idx]   = key.detach()
            self.values[idx] = value.detach()
            self.filled[idx] = True
            self.ptr[0]      = idx + 1

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """Retrieve a blended value for the given query."""
        squeeze = query.dim() == 1
        if squeeze:
            query = query.unsqueeze(0)

        B, H = query.shape
        n_filled = self.filled.sum().item()
        if n_filled == 0:
            out = torch.zeros_like(query)
            return out.squeeze(0) if squeeze else out

        filled_keys   = self.keys[self.filled]    # (n, H)
        filled_values = self.values[self.filled]

        q   = self.q_proj(query)                          # (B, H)
        k   = self.k_proj(filled_keys)                    # (n, H)

        sim = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).t()  # (B, n)
        k_ret = min(self.top_k, int(n_filled))
        topk_sim, topk_idx = sim.topk(k_ret, dim=-1)
        attn = F.softmax(topk_sim, dim=-1)                 # (B, k)
        topk_v = filled_values[topk_idx]                  # (B, k, H)
        out = (attn.unsqueeze(-1) * topk_v).sum(1)        # (B, H)
        out = self.out_proj(out)

        return out.squeeze(0) if squeeze else out


# ============================================================================
# Memory Router
# ============================================================================

class MemoryRouter(nn.Module):
    """
    Given a query vector, routes to working / episodic / semantic tiers.
    Returns a gated blend of all three tier outputs.
    """

    def __init__(self, hidden_dim: int, temperature: float = 1.0):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )
        self.temperature = temperature
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        query: torch.Tensor,           # (B, H) or (H,)
        working_out: torch.Tensor,
        episodic_out: torch.Tensor,
        semantic_out: torch.Tensor,
    ) -> torch.Tensor:
        q = query if query.dim() == 2 else query.unsqueeze(0)
        weights = F.softmax(self.gate(q) / self.temperature, dim=-1)  # (B, 3)
        w_w = weights[:, 0:1]
        w_e = weights[:, 1:2]
        w_s = weights[:, 2:3]

        # Ensure all outputs are (B, H)
        def _ensure(t):
            if t.dim() == 1:
                t = t.unsqueeze(0)
            return t

        blended = (
            w_w * _ensure(working_out) +
            w_e * _ensure(episodic_out) +
            w_s * _ensure(semantic_out)
        )
        out = self.out_proj(blended)
        return out.squeeze(0) if query.dim() == 1 else out


# ============================================================================
# ONIMemoryHub — top-level three-tier system
# ============================================================================

class ONIMemoryHub(nn.Module):
    """
    Three-tier differentiable memory system for ONI.

    Usage
    -----
    hub = ONIMemoryHub(hidden_dim=896)

    # Write a new experience
    hub.write(embedding, importance=emotion_salience, timestamp=step)

    # Read a context-aware memory retrieval
    memory_ctx = hub.read(query, timestamp=step)

    # Periodic consolidation (call e.g. every 100 steps)
    hub.consolidate()

    Note: no file I/O, no pygame, no hard-coded paths.
    """

    def __init__(
        self,
        hidden_dim: int = 896,
        working_slots: int = 64,
        episodic_capacity: int = 4096,
        semantic_capacity: int = 16384,
        episodic_top_k: int = 8,
        semantic_top_k: int = 4,
        consolidation_threshold: float = 5.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.consolidation_threshold = consolidation_threshold

        self.working  = WorkingMemoryModule(
            hidden_dim=hidden_dim,
            num_slots=working_slots,
        )
        self.episodic = EpisodicMemory(
            hidden_dim=hidden_dim,
            capacity=episodic_capacity,
            top_k=episodic_top_k,
        )
        self.semantic = SemanticMemory(
            hidden_dim=hidden_dim,
            capacity=semantic_capacity,
            top_k=semantic_top_k,
        )
        self.router = MemoryRouter(hidden_dim)

        # Input projection for consistent query space
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(
        self,
        embedding: torch.Tensor,
        importance: float = 1.0,
        timestamp: float = 0.0,
    ) -> None:
        """Write to working + episodic tiers."""
        self.working.update_memory(embedding, timestamp)
        self.episodic.write(embedding, importance, timestamp)

    # ------------------------------------------------------------------
    # Read (differentiable)
    # ------------------------------------------------------------------

    def read(
        self,
        query: torch.Tensor,
        timestamp: float = 0.0,
    ) -> torch.Tensor:
        """
        Read a blended context vector from all three tiers.

        query: (H,) or (B, H) or (B, T, H)
        Returns: same shape as query
        """
        # Flatten T dim for routing
        squeeze_T = query.dim() == 3
        if squeeze_T:
            B, T, H = query.shape
            query_flat = query.view(B * T, H)
        else:
            query_flat = query

        q = self.query_proj(query_flat)

        w_out = self.working.retrieve_memory(q, timestamp)
        e_out = self.episodic.read(q, timestamp)
        s_out = self.semantic.read(q)

        out = self.router(q, w_out, e_out, s_out)
        out = self.norm(out)

        if squeeze_T:
            out = out.view(B, T, H)
        return out

    # ------------------------------------------------------------------
    # Consolidation — migrate high-importance episodic → semantic
    # ------------------------------------------------------------------

    def consolidate(self) -> int:
        """
        Migrate high-importance episodic items into semantic memory.
        Returns the number of items migrated.
        """
        keys, imps = self.episodic.get_consolidation_candidates(
            self.consolidation_threshold
        )
        if keys.shape[0] == 0:
            return 0

        # Use keys as both key and value (abstract representation)
        self.semantic.write(keys, keys)

        # Decay those items' importance to avoid re-migration
        n = self.episodic.size[0].item()
        mask = self.episodic.importance[:n] > self.consolidation_threshold
        with torch.no_grad():
            self.episodic.importance[:n][mask] *= 0.1

        return keys.shape[0]

    # ------------------------------------------------------------------
    # Convenience wrappers that match old Memory interface
    # ------------------------------------------------------------------

    def update_context(self, embedding: torch.Tensor, timestamp: float = 0.0):
        self.write(embedding, timestamp=timestamp)

    def get_context(self, query: torch.Tensor, timestamp: float = 0.0) -> torch.Tensor:
        return self.read(query, timestamp)
