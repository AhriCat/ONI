"""
modules/memory/memory_ensemble.py
==================================
MemoryEnsemble — unified interface over all ONI memory tiers.

Replaces MemoryManager. Composes:
  - ONIMemoryHub  (working + episodic + semantic + router, all differentiable)
  - ModernContinuousHopfieldNetwork  (optional associative recall)
  - SnapshotMemorySystem             (optional circular parameter snapshot)
  - FadingMemorySystem               (optional exponential decay stream)

Adds meta-save / meta-load: serialises all weights + buffer state to disk.
Periodic consolidation runs automatically on write (every N writes).
"""

import json
import os
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn


class MemoryEnsemble(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 896,
        working_slots: int = 64,
        episodic_capacity: int = 4096,
        semantic_capacity: int = 16384,
        snapshot_size: int = 8192,
        use_hopfield: bool = True,
        use_fading: bool = True,
        use_snapshot: bool = True,
        consolidate_every: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self._consolidate_every = consolidate_every
        self._write_count = 0

        from modules.memory.oni_memory_hub import ONIMemoryHub
        self.hub = ONIMemoryHub(
            hidden_dim=hidden_dim,
            working_slots=working_slots,
            episodic_capacity=episodic_capacity,
            semantic_capacity=semantic_capacity,
        )

        if use_hopfield:
            from modules.memory.hopfield import ModernContinuousHopfieldNetwork
            self.hopfield: Optional[nn.Module] = ModernContinuousHopfieldNetwork(hidden_dim=hidden_dim)
        else:
            self.hopfield = None

        if use_snapshot:
            from modules.memory.snapshot_memory import SnapshotMemorySystem
            self.snapshot: Optional[nn.Module] = SnapshotMemorySystem(
                hidden_dim=hidden_dim, memory_size=snapshot_size
            )
        else:
            self.snapshot = None

        if use_fading:
            from modules.memory.fading_memory import FadingMemorySystem
            self.fading: Optional[nn.Module] = FadingMemorySystem(
                hidden_dim=hidden_dim, decay_rate=0.05
            )
        else:
            self.fading = None

    # -----------------------------------------------------------------------
    # Write
    # -----------------------------------------------------------------------

    def write(
        self,
        value: torch.Tensor,
        importance: float = 1.0,
        timestamp: float = 0.0,
    ) -> None:
        """Write a value into all active tiers."""
        self.hub.write(value, importance=importance, timestamp=timestamp)
        if self.fading is not None:
            self.fading(value)
        if self.snapshot is not None and value.dim() >= 2:
            self.snapshot.update(value.unsqueeze(0) if value.dim() == 2 else value)
        self._write_count += 1
        if self._write_count % self._consolidate_every == 0:
            self.consolidate()

    # -----------------------------------------------------------------------
    # Read (differentiable)
    # -----------------------------------------------------------------------

    def read(
        self,
        query: torch.Tensor,
        timestamp: float = 0.0,
    ) -> torch.Tensor:
        """Retrieve a blended context vector from all active tiers."""
        out = self.hub.read(query, timestamp=timestamp)
        if self.hopfield is not None:
            n_stored = self.hub.episodic.size[0].item()
            if n_stored > 0:
                h_out = self.hopfield(
                    query if query.dim() == 2 else query.unsqueeze(0)
                )
                if query.dim() == 1:
                    h_out = h_out.squeeze(0)
                out = 0.85 * out + 0.15 * h_out
        return out

    # -----------------------------------------------------------------------
    # Consolidation
    # -----------------------------------------------------------------------

    def consolidate(self) -> int:
        """Migrate high-importance episodic items to semantic + Hopfield."""
        n_migrated = self.hub.consolidate()
        if self.hopfield is not None and n_migrated > 0:
            keys, _ = self.hub.episodic.get_consolidation_candidates(threshold=3.0)
            if keys.shape[0] > 0:
                self.hopfield.store(keys)
        return n_migrated

    # -----------------------------------------------------------------------
    # Meta-save / Meta-load
    # -----------------------------------------------------------------------

    def save_state(self, path: str) -> None:
        """Serialise all weights and buffer state to disk."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "ensemble_weights.pt"))

        # Episodic ring-buffer
        ep = self.hub.episodic
        torch.save(
            {
                "store": ep.store,
                "importance": ep.importance,
                "timestamps": ep.timestamps,
                "ptr": ep.ptr,
                "size": ep.size,
            },
            os.path.join(path, "episodic_state.pt"),
        )

        # Semantic KV store
        sem = self.hub.semantic
        torch.save(
            {"keys": sem.keys, "values": sem.values, "filled": sem.filled, "ptr": sem.ptr},
            os.path.join(path, "semantic_state.pt"),
        )

        meta = {
            "write_count": self._write_count,
            "hidden_dim": self.hidden_dim,
            "saved_at": time.time(),
        }
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f)

    def load_state(self, path: str) -> None:
        """Restore all weights and buffer state from disk."""
        weights_path = os.path.join(path, "ensemble_weights.pt")
        if os.path.exists(weights_path):
            self.load_state_dict(
                torch.load(weights_path, map_location="cpu"), strict=False
            )

        ep_path = os.path.join(path, "episodic_state.pt")
        if os.path.exists(ep_path):
            ep = self.hub.episodic
            d = torch.load(ep_path, map_location="cpu")
            with torch.no_grad():
                ep.store.copy_(d["store"])
                ep.importance.copy_(d["importance"])
                ep.timestamps.copy_(d["timestamps"])
                ep.ptr.copy_(d["ptr"])
                ep.size.copy_(d["size"])

        sem_path = os.path.join(path, "semantic_state.pt")
        if os.path.exists(sem_path):
            sem = self.hub.semantic
            d = torch.load(sem_path, map_location="cpu")
            with torch.no_grad():
                sem.keys.copy_(d["keys"])
                sem.values.copy_(d["values"])
                sem.filled.copy_(d["filled"])
                sem.ptr.copy_(d["ptr"])

        meta_path = os.path.join(path, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            self._write_count = meta.get("write_count", 0)

    # -----------------------------------------------------------------------
    # Convenience aliases matching ONIMemoryHub interface
    # -----------------------------------------------------------------------

    def update_context(self, embedding: torch.Tensor, timestamp: float = 0.0):
        self.write(embedding, timestamp=timestamp)

    def get_context(self, query: torch.Tensor, timestamp: float = 0.0) -> torch.Tensor:
        return self.read(query, timestamp=timestamp)
