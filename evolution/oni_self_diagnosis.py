# File: evolution/oni_self_diagnosis.py
"""
Self-referential diagnosis using ONI's own metacognition module.

KEY INNOVATION: Unlike DGM which uses an external o1 model for diagnosis,
we use ONI's own reasoning capabilities. As ONI improves, its ability to
diagnose improvements ALSO improves — creating a virtuous cycle.

[EDITOR] This module depends on Fix 1C (diagnose_self method) being applied
to modules/NLP/oni_metacognition.py first.
"""
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
import logging

from .improvement_proposal import (
    ImprovementProposal,
    ProposalType,
    ModuleType,
    EvaluationLog
)

logger = logging.getLogger(__name__)


class ONISelfDiagnosis:
    """
    Uses ONI's own metacognition module to identify weaknesses and propose
    improvements. This is the CORE DIFFERENTIATOR from vanilla DGM.
    """

    def __init__(
        self,
        metacognition_module,
        epistemic_threshold: float = 0.7,
        conflict_threshold: float = 0.5,
        min_proposal_priority: float = 0.3
    ):
        self.metacognition = metacognition_module
        self.epistemic_threshold = epistemic_threshold
        self.conflict_threshold = conflict_threshold
        self.min_proposal_priority = min_proposal_priority

        # Track diagnosis history for meta-learning
        self.diagnosis_history: List[Dict] = []

    def diagnose(
        self,
        task_input: torch.Tensor,
        task_output: torch.Tensor,
        expected_output: Optional[torch.Tensor],
        eval_log: EvaluationLog
    ) -> List[ImprovementProposal]:
        """
        Run self-diagnosis on a failed or low-scoring task.

        Returns a list of ImprovementProposals sorted by priority.
        """
        proposals = []

        # Use ONI's own metacognition to diagnose
        try:
            diagnosis = self.metacognition.diagnose_self(
                task_input=task_input,
                task_output=task_output,
                expected_output=expected_output,
                task_id=eval_log.task_id
            )
        except Exception as e:
            logger.error(f"Self-diagnosis failed on task {eval_log.task_id}: {e}")
            proposals.append(ImprovementProposal(
                proposal_type=ProposalType.MODULE_WEAKNESS,
                target_module=eval_log.module,
                description=f"Task {eval_log.task_id} failed and self-diagnosis raised: {e}",
                suggested_action="Inspect module for runtime errors; review forward() signature.",
                priority_score=0.8,
                evidence={'error': str(e)},
                source_task_id=eval_log.task_id
            ))
            return proposals

        explanation = diagnosis['explanation']
        weakness_indicators = diagnosis['weakness_indicators']
        error_signal = diagnosis['error_signal']

        # --- Generate proposals from weakness indicators ---

        for indicator_name, severity in weakness_indicators:
            if indicator_name == 'high_epistemic_uncertainty':
                proposals.append(ImprovementProposal(
                    proposal_type=ProposalType.KNOWLEDGE_GAP,
                    target_module=eval_log.module,
                    description=(
                        f"High epistemic uncertainty ({severity:.2f}) on task "
                        f"{eval_log.task_id} indicates the model lacks knowledge "
                        f"in this domain."
                    ),
                    suggested_action=(
                        "Expand training data for this task type, or increase "
                        "model capacity in the target module."
                    ),
                    priority_score=min(severity, 1.0),
                    evidence={
                        'epistemic_uncertainty': severity,
                        'task_id': eval_log.task_id,
                        'error_signal': error_signal
                    },
                    source_task_id=eval_log.task_id,
                    epistemic_uncertainty=severity
                ))

            elif indicator_name == 'principle_conflict':
                proposals.append(ImprovementProposal(
                    proposal_type=ProposalType.PRINCIPLE_CONFLICT,
                    target_module=eval_log.module,
                    description=(
                        f"Principle conflict (score={severity:.2f}) detected during "
                        f"task {eval_log.task_id}."
                    ),
                    suggested_action=(
                        "Review conflicting principles and either reconcile them "
                        "or add context-dependent priority weighting."
                    ),
                    priority_score=min(severity * 0.9, 1.0),
                    evidence={
                        'conflict_score': severity,
                        'conflicts': explanation['conflicts']['detected']
                    },
                    source_task_id=eval_log.task_id
                ))

            elif indicator_name == 'low_confidence':
                proposals.append(ImprovementProposal(
                    proposal_type=ProposalType.STRATEGY_MISMATCH,
                    target_module=eval_log.module,
                    description=(
                        f"Low confidence ({1.0 - severity:.2f}) on task "
                        f"{eval_log.task_id} suggests the chosen reasoning "
                        f"strategy was suboptimal."
                    ),
                    suggested_action=(
                        "Retrain the strategy selector or expand the set of "
                        "available reasoning strategies."
                    ),
                    priority_score=min(severity * 0.85, 1.0),
                    evidence={
                        'confidence': 1.0 - severity,
                        'strategy_used': explanation['reasoning_strategy']['name']
                    },
                    source_task_id=eval_log.task_id
                ))

            elif indicator_name == 'high_aleatoric_uncertainty':
                proposals.append(ImprovementProposal(
                    proposal_type=ProposalType.MODULE_WEAKNESS,
                    target_module=eval_log.module,
                    description=(
                        f"High aleatoric uncertainty ({severity:.2f}) on task "
                        f"{eval_log.task_id} — the data itself is noisy or "
                        f"ambiguous for this module."
                    ),
                    suggested_action=(
                        "Improve input preprocessing or add data augmentation "
                        "to handle ambiguous inputs."
                    ),
                    priority_score=min(severity * 0.7, 1.0),
                    evidence={'aleatoric_uncertainty': severity},
                    source_task_id=eval_log.task_id,
                    aleatoric_uncertainty=severity
                ))

        # --- Use abductive reasoning for hypothesis generation ---
        try:
            abductive = self.metacognition.abductive_reasoning
            inp = (task_input.mean(dim=0, keepdim=True)
                   if task_input.dim() > 1 else task_input.unsqueeze(0))
            hyp_output, hyp_meta = abductive(inp)
            best_score = hyp_meta.get('best_score', torch.tensor(0.0))
            if isinstance(best_score, torch.Tensor):
                best_score = best_score.item()

            if best_score > 0.6:
                proposals.append(ImprovementProposal(
                    proposal_type=ProposalType.ABDUCTIVE_HYPOTHESIS,
                    target_module=eval_log.module,
                    description=(
                        f"Abductive reasoning generated a hypothesis "
                        f"(score={best_score:.2f}) for why task "
                        f"{eval_log.task_id} failed."
                    ),
                    suggested_action=(
                        "Test the generated hypothesis by modifying the module "
                        "and re-evaluating on similar tasks."
                    ),
                    priority_score=min(best_score * 0.8, 1.0),
                    evidence={'hypothesis_score': best_score},
                    source_task_id=eval_log.task_id
                ))
        except Exception as e:
            logger.debug(f"Abductive reasoning skipped: {e}")

        # Filter by minimum priority
        proposals = [p for p in proposals if p.priority_score >= self.min_proposal_priority]

        # Sort by priority descending
        proposals.sort(key=lambda p: p.priority_score, reverse=True)

        # Record in history
        self.diagnosis_history.append({
            'task_id': eval_log.task_id,
            'num_proposals': len(proposals),
            'top_proposal': proposals[0].proposal_type.value if proposals else None,
            'top_priority': proposals[0].priority_score if proposals else 0.0
        })

        return proposals

    def batch_diagnose(
        self,
        eval_logs: List[EvaluationLog],
        task_data: Dict[str, Dict[str, torch.Tensor]]
    ) -> List[ImprovementProposal]:
        """
        Diagnose across multiple evaluation logs and deduplicate proposals.

        Args:
            eval_logs: List of evaluation results
            task_data: Dict mapping task_id -> {'input': tensor, 'output': tensor,
                       'expected': tensor_or_None}
        """
        all_proposals = []

        # Only diagnose failed or low-scoring tasks
        for log in eval_logs:
            if log.success and log.score > 0.8:
                continue

            data = task_data.get(log.task_id)
            if data is None:
                logger.warning(f"No task data for {log.task_id}, skipping diagnosis")
                continue

            proposals = self.diagnose(
                task_input=data['input'],
                task_output=data['output'],
                expected_output=data.get('expected'),
                eval_log=log
            )
            all_proposals.extend(proposals)

        # Deduplicate: group by (proposal_type, target_module), keep highest priority
        seen = {}
        for p in all_proposals:
            key = (p.proposal_type, p.target_module)
            if key not in seen or p.priority_score > seen[key].priority_score:
                seen[key] = p

        deduped = sorted(seen.values(), key=lambda p: p.priority_score, reverse=True)
        return deduped
