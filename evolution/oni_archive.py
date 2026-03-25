# File: evolution/oni_archive.py
"""
Archive management for ONI variant population.

Adapted from DGM's archive pattern (DGM_outer.py) with ONI-specific extensions.

[EDITOR] The original DGM uses flat JSON metadata files per variant. We preserve
that pattern for compatibility.
"""
import os
import json
import math
import random
import shutil
import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import asdict
import logging

from .improvement_proposal import ONIVariant, EvaluationLog

logger = logging.getLogger(__name__)


class ONIArchive:
    """
    Manages the archive of ONI variants for open-ended exploration.

    DGM's key insight: Open-ended exploration prevents getting stuck in
    local optima. Maintaining an archive of all variants allows
    backtracking and exploring alternative improvement paths.
    """

    def __init__(
        self,
        archive_dir: str,
        initial_variant_path: str,
        lambda_param: float = 10.0,
        alpha_0: float = 0.5
    ):
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.lambda_param = lambda_param
        self.alpha_0 = alpha_0

        # Archive state
        self.variants: Dict[str, ONIVariant] = {}
        self.generation = 0

        # Initialize with base variant
        self._initialize_archive(initial_variant_path)

    def _initialize_archive(self, initial_path: str):
        """Initialize archive with the initial ONI variant."""
        initial_id = "initial"
        variant_dir = self.archive_dir / initial_id
        if not variant_dir.exists():
            if os.path.exists(initial_path):
                shutil.copytree(initial_path, str(variant_dir))
            else:
                variant_dir.mkdir(parents=True, exist_ok=True)
                logger.warning(
                    f"Initial variant path {initial_path} does not exist. "
                    f"Created empty directory."
                )

        self.variants[initial_id] = ONIVariant(
            variant_id=initial_id,
            parent_id=None,
            generation=0,
            patch_file="",
            overall_score=0.0,
            is_compiled=True,
            is_coherent=True,
            timestamp=datetime.datetime.now().isoformat()
        )
        self._save_state()

    def select_parents(
        self,
        k: int = 2,
        method: str = 'score_child_prop'
    ) -> List[str]:
        """
        Select parent variants for the next generation.

        Implements DGM-style parent selection:
            P(parent_i) = sigmoid(score_i) * (1 / (1 + children_count_i)) / Z

        [EDITOR] Matches the actual DGM_outer.py choose_selfimproves() logic.
        """
        # Filter to compiled-only candidates
        candidates = {
            vid: v for vid, v in self.variants.items()
            if v.is_compiled
        }

        if not candidates:
            logger.warning("No compiled variants in archive, falling back to initial")
            return ["initial"] * k

        variant_ids = list(candidates.keys())

        if method == 'score_child_prop':
            scores = [candidates[vid].overall_score for vid in variant_ids]
            sigmoid_scores = [
                1.0 / (1.0 + math.exp(-self.lambda_param * (s - self.alpha_0)))
                for s in scores
            ]
            child_penalties = [
                1.0 / (1.0 + candidates[vid].children_count)
                for vid in variant_ids
            ]
            raw_probs = [s * c for s, c in zip(sigmoid_scores, child_penalties)]
            total = sum(raw_probs)
            if total == 0:
                probs = [1.0 / len(variant_ids)] * len(variant_ids)
            else:
                probs = [p / total for p in raw_probs]
            selected = random.choices(variant_ids, weights=probs, k=k)

        elif method == 'score_prop':
            scores = [candidates[vid].overall_score for vid in variant_ids]
            sigmoid_scores = [
                1.0 / (1.0 + math.exp(-self.lambda_param * (s - self.alpha_0)))
                for s in scores
            ]
            total = sum(sigmoid_scores)
            probs = ([s / total for s in sigmoid_scores] if total > 0
                     else [1.0 / len(variant_ids)] * len(variant_ids))
            selected = random.choices(variant_ids, weights=probs, k=k)

        elif method == 'best':
            sorted_ids = sorted(
                variant_ids, key=lambda v: candidates[v].overall_score, reverse=True
            )
            selected = sorted_ids[:min(k, len(sorted_ids))]
            while len(selected) < k:
                selected.append(random.choice(sorted_ids[:max(1, len(sorted_ids) // 2)]))

        else:  # random
            selected = random.choices(variant_ids, k=k)

        # Update children counts
        for vid in selected:
            if vid in self.variants:
                self.variants[vid].children_count += 1

        return selected

    def add_variant(self, variant: ONIVariant) -> bool:
        """
        Add a new variant to the archive.

        [EDITOR] Following DGM pattern: archive accepts all compiled variants
        by default (method='keep_all'). Use update_method='keep_better' to
        filter by score.
        """
        if not variant.is_compiled:
            logger.info(f"Variant {variant.variant_id} did not compile, skipping")
            return False

        variant.timestamp = datetime.datetime.now().isoformat()
        self.variants[variant.variant_id] = variant

        # Save variant metadata
        variant_dir = self.archive_dir / variant.variant_id
        variant_dir.mkdir(parents=True, exist_ok=True)
        meta_path = variant_dir / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(asdict(variant), f, indent=2, default=str)

        self._save_state()
        logger.info(
            f"Added variant {variant.variant_id} "
            f"(score={variant.overall_score:.4f}, gen={variant.generation})"
        )
        return True

    def get_best_variant(self) -> Optional[ONIVariant]:
        """Return the highest-scoring compiled variant."""
        compiled = [v for v in self.variants.values() if v.is_compiled]
        if not compiled:
            return None
        return max(compiled, key=lambda v: v.overall_score)

    def _save_state(self):
        """Append current state to the archive state log."""
        best = self.get_best_variant()
        state = {
            'generation': self.generation,
            'num_variants': len(self.variants),
            'best_variant': best.variant_id if best else None,
            'best_score': best.overall_score if best else 0.0,
            'archive_ids': list(self.variants.keys()),
            'timestamp': datetime.datetime.now().isoformat()
        }
        state_file = self.archive_dir / "archive_state.jsonl"
        with open(state_file, 'a') as f:
            f.write(json.dumps(state) + "\n")

    def load_state(self) -> Optional[Dict]:
        """Load the latest archive state."""
        state_file = self.archive_dir / "archive_state.jsonl"
        if not state_file.exists():
            return None
        with open(state_file) as f:
            lines = f.readlines()
        if lines:
            return json.loads(lines[-1])
        return None
