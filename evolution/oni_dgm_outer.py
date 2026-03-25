# File: evolution/oni_dgm_outer.py
"""
Main orchestrator for ONI-DGM evolution.

Combines:
- DGM archive and parent selection (adapted from Meta's DGM_outer.py)
- ONI self-diagnosis (using its own metacognition — virtuous cycle)
- Superintelligence Oven for GRPO training
- Multi-modal evaluation

[EDITOR] Modelled after dgm-main/DGM_outer.py, but replaces:
  - External o1 diagnosis      → ONI self-diagnosis (metacognition)
  - SWE-bench evaluation       → ONI multi-modal evaluation suite
  - Git-based code patching    → Module weight patching (torch.save diff)
  - Docker sandboxed execution → Oven-managed training subprocess
"""
import datetime
import json
import logging
import os
from typing import Dict, List, Optional

import torch

from .oni_archive import ONIArchive
from .oni_self_diagnosis import ONISelfDiagnosis
from .oni_oven_integration import ONIOvenIntegration
from .oni_evaluation import ONIEvaluator
from .improvement_proposal import ModuleType, ImprovementProposal, ONIVariant

logger = logging.getLogger(__name__)


class ONIDarwinGodelMachine:
    """
    Main orchestrator for ONI evolution.

    Evolution loop per generation:
    1. Select parent(s) from archive using DGM sigmoid-child-proportional selection
    2. Evaluate parent → baseline score + per-task data
    3. Self-diagnosis on failures → ImprovementProposals (ranked by priority)
    4. Apply top proposal via Oven GRPO training
    5. Re-evaluate new variant
    6. Add to archive if compiled (no runtime errors) & coherent (score > 0)
    """

    def __init__(
        self,
        oni_model: torch.nn.Module,
        tokenizer,
        archive_dir: str,
        initial_variant_path: str,
        config: Dict = None
    ):
        self.oni_model = oni_model
        self.tokenizer = tokenizer
        self.config = config or self._default_config()

        # Archive
        self.archive = ONIArchive(
            archive_dir=archive_dir,
            initial_variant_path=initial_variant_path,
            lambda_param=self.config['archive']['lambda_param'],
            alpha_0=self.config['archive']['alpha_0']
        )

        # Self-diagnosis — grab the real metacognition or fall back to mock
        metacog = getattr(oni_model, 'metacognition_module', None)
        if metacog is None:
            logger.warning(
                "ONI model has no metacognition_module attribute. "
                "Self-diagnosis will use a mock."
            )
            metacog = self._build_mock_metacog()

        self.self_diagnosis = ONISelfDiagnosis(
            metacognition_module=metacog,
            epistemic_threshold=self.config['diagnosis']['epistemic_threshold'],
            conflict_threshold=self.config['diagnosis']['conflict_threshold'],
            min_proposal_priority=self.config['diagnosis']['min_proposal_priority']
        )

        # Oven integration
        self.oven = ONIOvenIntegration(
            oni_model=oni_model,
            tokenizer=tokenizer,
            self_diagnosis=self.self_diagnosis,
            training_config=self.config.get('training')
        )

        # Evaluator
        self.evaluator = ONIEvaluator(
            oni_model=oni_model,
            benchmark_configs=self.config.get('evaluation')
        )

        # Populate with synthetic tasks
        hidden_dim = getattr(metacog, 'hidden_dim', 512)
        self.evaluator.generate_synthetic_tasks(
            hidden_dim, count_per_module=10
        )

        logger.info(
            f"ONI-DGM initialized | archive={archive_dir} | "
            f"hidden_dim={hidden_dim}"
        )

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _default_config(self) -> Dict:
        from .config import DEFAULT_CONFIG
        return DEFAULT_CONFIG

    @staticmethod
    def _build_mock_metacog():
        """Minimal mock metacognition for when ONI.metacognition_module is absent."""
        import torch

        class _MockMeta:
            hidden_dim = 512

            def diagnose_self(self, task_input, task_output,
                              expected_output=None, task_id="unknown"):
                return {
                    'explanation': {
                        'reasoning_strategy': {'name': 'Mock'},
                        'confidence': {'score': 0.5},
                        'uncertainty': {'epistemic': 0.5, 'aleatoric': 0.5},
                        'conflicts': {'detected': []}
                    },
                    'error_signal': {},
                    'weakness_indicators': [('low_confidence', 0.5)],
                    'task_id': task_id
                }

            class abductive_reasoning:
                @staticmethod
                def __call__(x):
                    return x, {'best_score': torch.tensor(0.3)}

        return _MockMeta()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, max_generations: int = None) -> Optional[ONIVariant]:
        """Run the full evolution loop."""
        max_gen = max_generations or self.config['evolution']['max_generations']
        logger.info(
            f"Starting ONI-DGM evolution | max_generations={max_gen} | "
            f"archive size={len(self.archive.variants)}"
        )

        for gen in range(max_gen):
            new_ids = self.run_generation(gen)
            best = self.archive.get_best_variant()

            logger.info(
                f"Generation {gen} complete. "
                f"New variants: {new_ids}. "
                f"Archive size: {len(self.archive.variants)}. "
                f"Best: {best.variant_id if best else 'none'} "
                f"(score={best.overall_score:.4f if best else 0:.4f})"
            )

            # Persist generation metadata (mirrors DGM_outer.py pattern)
            state_file = os.path.join(
                str(self.archive.archive_dir), "dgm_metadata.jsonl"
            )
            with open(state_file, "a") as f:
                f.write(json.dumps({
                    "generation": gen,
                    "new_variants": new_ids,
                    "archive": list(self.archive.variants.keys()),
                    "best_variant": best.variant_id if best else None,
                    "best_score": best.overall_score if best else 0.0,
                    "timestamp": datetime.datetime.now().isoformat()
                }) + "\n")

        logger.info("Evolution complete.")
        best = self.archive.get_best_variant()
        if best:
            logger.info(
                f"Final best: {best.variant_id} (score={best.overall_score:.4f})"
            )
        return best

    def run_generation(self, generation_num: int) -> List[str]:
        """
        Run a single generation.

        Returns list of new variant IDs added to archive.
        """
        logger.info(f"=== Generation {generation_num} ===")
        new_variant_ids: List[str] = []

        num_parents = self.config['evolution']['parents_per_generation']
        method = self.config['evolution']['parent_selection_method']
        parent_ids = self.archive.select_parents(k=num_parents, method=method)
        logger.info(f"Selected parents: {parent_ids}")

        for parent_id in parent_ids:
            try:
                variant_id = self._evolve_from_parent(parent_id, generation_num)
                if variant_id:
                    new_variant_ids.append(variant_id)
            except Exception as e:
                import traceback
                logger.error(
                    f"Evolution from parent {parent_id} failed: {e}\n"
                    f"{traceback.format_exc()}"
                )

        self.archive.generation = generation_num
        return new_variant_ids

    def _evolve_from_parent(
        self, parent_id: str, generation_num: int
    ) -> Optional[str]:
        """Produce one child variant from a parent."""

        # --- Step 2: Evaluate parent ---
        logger.info(f"Evaluating parent {parent_id}...")
        parent_score, eval_logs = self.evaluator.evaluate_variant()

        # Collect task data for self-diagnosis
        task_data = self._collect_task_data(eval_logs)

        passed = sum(1 for l in eval_logs if l.success)
        logger.info(
            f"Parent {parent_id} score={parent_score:.4f} "
            f"({passed}/{len(eval_logs)} passed)"
        )

        # --- Step 3: Self-diagnosis ---
        proposals = self.self_diagnosis.batch_diagnose(eval_logs, task_data)
        logger.info(f"Generated {len(proposals)} improvement proposals")

        if not proposals:
            logger.info("No proposals generated, skipping parent")
            return None

        top_proposal = proposals[0]
        logger.info(
            f"Top proposal: {top_proposal.proposal_type.value} "
            f"(priority={top_proposal.priority_score:.2f}) "
            f"→ {top_proposal.target_module.value}"
        )

        # --- Step 4: Train via Oven ---
        pre_train_state = {
            k: v.clone() for k, v in self.oni_model.state_dict().items()
        }

        train_result = self.oven.train_on_proposal(
            proposal=top_proposal,
            training_data=None,
            num_steps=self.config['training'].get('steps_per_proposal', 500)
        )
        mock_flag = " [MOCK]" if train_result.get('mock') else ""
        logger.info(
            f"Training done{mock_flag}: "
            f"loss={train_result.get('loss', 'N/A'):.4f}, "
            f"reward={train_result.get('reward_mean', 0.0):.4f}"
        )

        # --- Step 5: Re-evaluate ---
        new_score, new_eval_logs = self.evaluator.evaluate_variant()
        logger.info(f"Post-training score: {new_score:.4f}")

        # --- Step 6: Create variant & add to archive ---
        ts = datetime.datetime.now().strftime('%H%M%S')
        variant_id = f"gen{generation_num}_{parent_id}_{ts}"

        # Save weight diff as "patch"
        patch_dir = os.path.join(str(self.archive.archive_dir), variant_id)
        os.makedirs(patch_dir, exist_ok=True)
        patch_file = os.path.join(patch_dir, "weight_patch.pt")

        weight_diff = {}
        current_state = self.oni_model.state_dict()
        for k in current_state:
            if k in pre_train_state:
                diff = current_state[k] - pre_train_state[k]
                if diff.abs().max().item() > 1e-8:
                    weight_diff[k] = diff
        torch.save(weight_diff, patch_file)

        is_compiled = all(l.error_message is None for l in new_eval_logs)

        variant = ONIVariant(
            variant_id=variant_id,
            parent_id=parent_id,
            generation=generation_num,
            patch_file=patch_file,
            overall_score=new_score,
            module_scores={
                mt.value: (
                    sum(l.score for l in new_eval_logs if l.module == mt) /
                    max(sum(1 for l in new_eval_logs if l.module == mt), 1)
                )
                for mt in ModuleType
            },
            is_compiled=is_compiled,
            is_coherent=new_score > 0.0,
            evaluation_logs=new_eval_logs,
            proposals_applied=[top_proposal.proposal_type.value]
        )

        added = self.archive.add_variant(variant)

        if not added:
            # Rollback if rejected
            self.oni_model.load_state_dict(pre_train_state)
            logger.info(f"Variant {variant_id} rejected — model rolled back")
            return None

        return variant_id

    def _collect_task_data(
        self, eval_logs
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Reconstruct input/output tensors for tasks in eval_logs."""
        task_data: Dict[str, Dict[str, torch.Tensor]] = {}
        for log in eval_logs:
            for task in self.evaluator.task_registry.get(log.module, []):
                if task.task_id == log.task_id:
                    try:
                        inp = task.get_input()
                        with torch.no_grad():
                            raw = self.oni_model(inp)
                        out = raw[0] if isinstance(raw, tuple) else raw
                        task_data[log.task_id] = {
                            'input': inp,
                            'output': out,
                            'expected': task.get_expected_output()
                        }
                    except Exception:
                        pass
                    break
        return task_data
