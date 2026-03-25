# File: evolution/__init__.py
"""ONI-DGM Evolution System."""

from .config import DEFAULT_CONFIG
from .improvement_proposal import (
    ProposalType, ModuleType, ImprovementProposal,
    EvaluationLog, ONIVariant
)
from .oni_self_diagnosis import ONISelfDiagnosis
from .oni_archive import ONIArchive
from .oni_oven_integration import ONIOvenIntegration
from .oni_evaluation import ONIEvaluator
from .oni_dgm_outer import ONIDarwinGodelMachine

__all__ = [
    'DEFAULT_CONFIG',
    'ProposalType', 'ModuleType', 'ImprovementProposal',
    'EvaluationLog', 'ONIVariant',
    'ONISelfDiagnosis', 'ONIArchive',
    'ONIOvenIntegration', 'ONIEvaluator',
    'ONIDarwinGodelMachine',
]
