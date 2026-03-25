# File: evolution/tests/test_self_diagnosis.py
import pytest
import torch
from evolution.oni_self_diagnosis import ONISelfDiagnosis
from evolution.improvement_proposal import EvaluationLog, ModuleType, ProposalType


class MockMetacognition:
    """Mock metacognition module for testing — no GPU required."""

    def __init__(self, hidden_dim=256):
        self.hidden_dim = hidden_dim

    def diagnose_self(self, task_input, task_output, expected_output=None,
                      task_id="test"):
        return {
            'explanation': {
                'reasoning_strategy': {'name': 'Deductive'},
                'confidence': {'score': 0.3},
                'uncertainty': {'epistemic': 0.8, 'aleatoric': 0.2},
                'conflicts': {'detected': []}
            },
            'error_signal': (
                {'mse': 0.5, 'cosine_similarity': 0.3}
                if expected_output is not None else {}
            ),
            'weakness_indicators': [
                ('high_epistemic_uncertainty', 0.8),
                ('low_confidence', 0.7)
            ],
            'task_id': task_id
        }

    class abductive_reasoning:
        @staticmethod
        def __call__(x):
            return x, {'best_score': torch.tensor(0.7)}


class TestONISelfDiagnosis:
    def setup_method(self):
        self.metacog = MockMetacognition()
        self.diagnosis = ONISelfDiagnosis(
            metacognition_module=self.metacog,
            epistemic_threshold=0.7,
            conflict_threshold=0.5,
            min_proposal_priority=0.3
        )

    def test_diagnose_generates_proposals(self):
        task_input = torch.randn(1, 256)
        task_output = torch.randn(1, 256)
        expected = torch.randn(1, 256)
        log = EvaluationLog(
            task_id="test_task_1",
            module=ModuleType.NLP,
            success=False,
            score=0.2,
            time_taken=0.1
        )

        proposals = self.diagnosis.diagnose(task_input, task_output, expected, log)
        assert len(proposals) > 0
        assert all(p.priority_score >= 0.3 for p in proposals)
        # Sorted by priority descending
        assert proposals[0].priority_score >= proposals[-1].priority_score

    def test_diagnose_records_history(self):
        task_input = torch.randn(1, 256)
        task_output = torch.randn(1, 256)
        log = EvaluationLog(
            task_id="test_task_2",
            module=ModuleType.VISION,
            success=False,
            score=0.1,
            time_taken=0.05
        )

        self.diagnosis.diagnose(task_input, task_output, None, log)
        assert len(self.diagnosis.diagnosis_history) == 1
        assert self.diagnosis.diagnosis_history[0]['task_id'] == "test_task_2"

    def test_batch_diagnose_deduplicates(self):
        logs = [
            EvaluationLog("task_a", ModuleType.NLP, False, 0.1, 0.05),
            EvaluationLog("task_b", ModuleType.NLP, False, 0.2, 0.06),
        ]
        task_data = {
            "task_a": {
                'input': torch.randn(1, 256),
                'output': torch.randn(1, 256),
                'expected': torch.randn(1, 256)
            },
            "task_b": {
                'input': torch.randn(1, 256),
                'output': torch.randn(1, 256),
                'expected': torch.randn(1, 256)
            },
        }
        proposals = self.diagnosis.batch_diagnose(logs, task_data)
        # Should be deduplicated by (type, module)
        types_modules = [(p.proposal_type, p.target_module) for p in proposals]
        assert len(types_modules) == len(set(types_modules))

    def test_high_scoring_tasks_skipped(self):
        """Tasks with score > 0.8 and success=True should be skipped."""
        logs = [
            EvaluationLog("pass_task", ModuleType.NLP, True, 0.95, 0.05),
        ]
        task_data = {
            "pass_task": {
                'input': torch.randn(1, 256),
                'output': torch.randn(1, 256),
                'expected': torch.randn(1, 256)
            }
        }
        proposals = self.diagnosis.batch_diagnose(logs, task_data)
        assert len(proposals) == 0

    def test_proposals_have_correct_types(self):
        task_input = torch.randn(1, 256)
        task_output = torch.randn(1, 256)
        log = EvaluationLog("t1", ModuleType.MEMORY, False, 0.1, 0.05)
        proposals = self.diagnosis.diagnose(task_input, task_output, None, log)

        valid_types = set(ProposalType)
        for p in proposals:
            assert p.proposal_type in valid_types
            assert p.target_module == ModuleType.MEMORY
