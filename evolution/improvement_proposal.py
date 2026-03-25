# File: evolution/improvement_proposal.py
"""
Data structures for the ONI-DGM evolution system.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class ProposalType(Enum):
    KNOWLEDGE_GAP = "knowledge_gap"
    PRINCIPLE_CONFLICT = "principle_conflict"
    STRATEGY_MISMATCH = "strategy_mismatch"
    ABDUCTIVE_HYPOTHESIS = "abductive_hypothesis"
    MODULE_WEAKNESS = "module_weakness"
    CROSS_MODAL_INCOHERENCE = "cross_modal_incoherence"


class ModuleType(Enum):
    NLP = "nlp"
    VISION = "vision"
    AUDIO = "audio"
    MEMORY = "memory"
    EMOTION = "emotion"
    ROBOTICS = "robotics"
    WORLD_MODEL = "world_model"
    METACOGNITION = "metacognition"


@dataclass
class ImprovementProposal:
    """Represents a proposed improvement to ONI."""
    proposal_type: ProposalType
    target_module: ModuleType
    description: str
    suggested_action: str
    priority_score: float  # 0.0 to 1.0

    evidence: Dict[str, Any] = field(default_factory=dict)
    source_task_id: Optional[str] = None
    epistemic_uncertainty: float = 0.0
    aleatoric_uncertainty: float = 0.0

    def to_problem_statement(self) -> str:
        """Convert to a problem statement for self-modification."""
        return (
            f"## Improvement Proposal\n"
            f"**Type:** {self.proposal_type.value}\n"
            f"**Target Module:** {self.target_module.value}\n"
            f"**Priority:** {self.priority_score:.2f}\n\n"
            f"### Description\n{self.description}\n\n"
            f"### Suggested Action\n{self.suggested_action}\n\n"
            f"### Evidence\n{self.evidence}\n"
        )

    def to_oven_prompt(self) -> str:
        """Convert to a training prompt for the Superintelligence Oven."""
        return (
            f"Train the {self.target_module.value} module to address: {self.description}\n"
            f"Priority: {self.priority_score:.2f} | Type: {self.proposal_type.value}\n"
            f"Action: {self.suggested_action}"
        )


@dataclass
class EvaluationLog:
    """Log entry for a single evaluation task."""
    task_id: str
    module: ModuleType
    success: bool
    score: float
    time_taken: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ONIVariant:
    """Represents a variant in the archive."""
    variant_id: str
    parent_id: Optional[str]
    generation: int
    patch_file: str  # Path to weight diff saved by torch.save
    overall_score: float = 0.0
    module_scores: Dict[str, float] = field(default_factory=dict)
    is_compiled: bool = False
    is_coherent: bool = False
    children_count: int = 0
    evaluation_logs: List[EvaluationLog] = field(default_factory=list)
    proposals_applied: List[str] = field(default_factory=list)
    timestamp: str = ""
