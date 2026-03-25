# File: evolution/oni_oven_integration.py
"""
Integrates ONI with the Superintelligence Oven for multi-agent GRPO training.

Uses the real superintelligence_oven.bake() API:
    from superintelligence_oven import bake, OvenConfig, TeacherConfig, AgentConfig
    oven = bake(model=..., name=..., modality=..., ...)
    oven.run_sync()

[EDITOR] The oven is mock-first: when superintelligence_oven is unavailable,
a fallback SGD step keeps the evolution loop runnable for testing.
When the oven IS available (pip install -e ../super_intelligence_oven), only
the OVEN_AVAILABLE flag flips — no other code changes needed.

Key design: ONI's metacognition replaces the oven's curriculum agent as the
prompt/task generator. ONI decides what to improve; the oven executes the GRPO.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import logging

# Import the real oven API
try:
    from superintelligence_oven import (
        bake,
        OvenConfig,
        TeacherConfig,
        AgentConfig,
        ModelDescription,
        PromptSource,
    )
    OVEN_AVAILABLE = True
    logging.getLogger(__name__).info(
        "superintelligence_oven loaded — full GRPO training enabled"
    )
except ImportError:
    OVEN_AVAILABLE = False
    logging.getLogger(__name__).info(
        "superintelligence_oven not installed — using mock SGD fallback. "
        "Install with: pip install -e ../super_intelligence_oven"
    )

from .oni_self_diagnosis import ONISelfDiagnosis
from .improvement_proposal import ModuleType, ImprovementProposal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-to-domain mapping for oven prompts
# ---------------------------------------------------------------------------
_MODULE_DOMAIN_MAP = {
    ModuleType.NLP:          ("text",       "nlp",       "Natural language understanding and generation"),
    ModuleType.VISION:       ("multimodal", "vision",    "Visual perception and scene understanding"),
    ModuleType.AUDIO:        ("multimodal", "audio",     "Speech recognition and audio processing"),
    ModuleType.MEMORY:       ("text",       "memory",    "Episodic and semantic memory retrieval"),
    ModuleType.EMOTION:      ("text",       "emotion",   "Affective state modeling and empathic response"),
    ModuleType.ROBOTICS:     ("motion",     "robotics",  "Joint trajectory and motor control"),
    ModuleType.WORLD_MODEL:  ("multimodal", "world_model", "World-state prediction and planning"),
    ModuleType.METACOGNITION:("text",       "metacognition", "Self-reflection and reasoning strategy selection"),
}

# Oven teacher to use per module type (maps to TeacherConfig names)
_MODULE_TEACHER_MAP = {
    ModuleType.NLP:          "small_qwen3_4b",
    ModuleType.VISION:       "small_qwen3_4b",
    ModuleType.AUDIO:        "small_qwen3_4b",
    ModuleType.MEMORY:       "small_qwen3_4b",
    ModuleType.EMOTION:      "small_qwen3_4b",
    ModuleType.ROBOTICS:     "diffusion_motion",   # falls back to small if unavailable
    ModuleType.WORLD_MODEL:  "big_qwen3_8b_q4",
    ModuleType.METACOGNITION:"big_qwen3_8b_q4",
}


class ONIOvenIntegration:
    """
    Wraps the Superintelligence Oven for GRPO training of ONI variants.

    When the oven is available, it drives full multi-agent GRPO with:
      - Local teacher KL anchor (hot-swappable)
      - Remote agent swarm (critic / adversary / specialist / style / curriculum)
      - QwenEmbedVerifier semantic reward (text modalities)
      - Frontier calibration via DeepSeek / Grok logprobs

    When unavailable, it falls back to a 10-step SGD mock that keeps the
    evolution loop testable without any external dependencies.
    """

    def __init__(
        self,
        oni_model: nn.Module,
        tokenizer: Any,
        self_diagnosis: ONISelfDiagnosis,
        training_config: Dict = None
    ):
        self.oni_model = oni_model
        self.tokenizer = tokenizer
        self.self_diagnosis = self_diagnosis
        self.config = training_config or {}

        # Cache of active OvenWrapper instances keyed by module type
        # so we don't reload teacher models every proposal
        self._oven_cache: Dict[ModuleType, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_on_proposal(
        self,
        proposal: ImprovementProposal,
        training_data: Any = None,
        num_steps: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train ONI on a specific improvement proposal using GRPO (or mock SGD).

        The oven receives:
          - A ModelDescription derived from the proposal's target module
          - Prompts generated from the proposal's to_oven_prompt()
          - The appropriate teacher for the module type

        Returns training metrics dict.
        """
        steps = num_steps or self.config.get('steps_per_proposal', 500)

        if OVEN_AVAILABLE:
            return self._train_with_oven(proposal, training_data, steps)
        else:
            return self._train_mock(proposal, training_data, steps)

    def swap_agent(self, agent_name: str, **kwargs):
        """Hot-swap an oven agent model at runtime."""
        for oven_wrapper in self._oven_cache.values():
            try:
                oven_wrapper.swap_agent(agent_name, **kwargs)
            except Exception as e:
                logger.warning(f"Could not swap agent {agent_name}: {e}")

    def swap_all_agents(self, provider: str, model: str, **kwargs):
        """Swap all agents to a new provider/model."""
        for oven_wrapper in self._oven_cache.values():
            try:
                oven_wrapper.swap_all_agents(provider, model, **kwargs)
            except Exception as e:
                logger.warning(f"Could not swap agents: {e}")

    # ------------------------------------------------------------------
    # Internal: real oven
    # ------------------------------------------------------------------

    def _build_oven_for_proposal(
        self,
        proposal: ImprovementProposal,
        steps: int
    ) -> Any:
        """
        Build (or reuse cached) OvenWrapper for the proposal's target module.
        """
        module_type = proposal.target_module
        if module_type in self._oven_cache:
            return self._oven_cache[module_type]

        modality, domain, description = _MODULE_DOMAIN_MAP.get(
            module_type, ("text", "general", "General module improvement")
        )
        teacher_name = _MODULE_TEACHER_MAP.get(module_type, "small_qwen3_4b")

        # For robotics / motion, disable text verifier
        use_verifier = (modality == "text")

        # Build extra teachers list — always include diffusion for robotics
        extra_teachers = []
        if module_type == ModuleType.ROBOTICS:
            try:
                extra_teachers.append(TeacherConfig(
                    name="diffusion_motion",
                    kind="diffusion",
                    model_name="d-teacher",
                    joint_dim=72,
                    intent_dim=512,
                    timesteps=100,
                    hidden_dim=4096,
                ))
            except Exception:
                teacher_name = "small_qwen3_4b"  # fallback

        oven_wrapper = bake(
            model=self.oni_model,
            name=f"oni-{module_type.value}",
            modality=modality,
            domain=domain,
            description=description,
            output_format="text" if modality == "text" else module_type.value,
            scoring_guidance=proposal.suggested_action,
            use_verifier=use_verifier,
            tokenizer=self.tokenizer,
            teacher=teacher_name,
            teachers=extra_teachers if extra_teachers else None,
            total_steps=steps,
            group_size=self.config.get('group_size', 8),
            batch_size=4,
            lr=self.config.get('learning_rate', 1e-6),
            kl_beta=self.config.get('kl_beta', 0.1),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        self._oven_cache[module_type] = oven_wrapper
        return oven_wrapper

    def _train_with_oven(
        self,
        proposal: ImprovementProposal,
        training_data: Any,
        num_steps: int
    ) -> Dict[str, Any]:
        """Train using the real Superintelligence Oven."""
        logger.info(
            f"[OVEN] Training on proposal: {proposal.description[:80]}... "
            f"({num_steps} steps)"
        )

        oven_wrapper = self._build_oven_for_proposal(proposal, num_steps)

        # Generate prompts from the proposal — ONI decides what to train on
        seed_prompts = self._generate_training_prompts(proposal, training_data)

        metrics_history = oven_wrapper.run_sync(prompts=seed_prompts)

        # Summarise
        if metrics_history:
            last = metrics_history[-1]
            return {
                'loss': last.get('total_loss', 0.0),
                'reward_mean': last.get('reward', 0.0),
                'kl_divergence': last.get('kl_loss', 0.0),
                'steps_completed': len(metrics_history),
                'teacher': oven_wrapper.status().get('teacher', 'unknown'),
                'proposal_type': proposal.proposal_type.value,
                'mock': False,
            }
        return {
            'loss': 0.0, 'reward_mean': 0.0, 'kl_divergence': 0.0,
            'steps_completed': 0, 'proposal_type': proposal.proposal_type.value,
            'mock': False,
        }

    def _generate_training_prompts(
        self,
        proposal: ImprovementProposal,
        training_data: Any
    ) -> List[str]:
        """
        Generate training prompts from a proposal.

        ONI's metacognition (via self_diagnosis history) informs what to train on.
        """
        base_prompt = proposal.to_oven_prompt()
        prompts = [base_prompt]

        # Add task-specific variants from diagnosis history
        for entry in self.self_diagnosis.diagnosis_history[-10:]:
            task_id = entry.get('task_id', '')
            if task_id:
                prompts.append(
                    f"Improve performance on task '{task_id}': {base_prompt}"
                )

        # Add user-supplied training data if available
        if isinstance(training_data, list):
            prompts.extend([str(d) for d in training_data[:20]])
        elif isinstance(training_data, dict) and 'prompts' in training_data:
            prompts.extend(training_data['prompts'][:20])

        return prompts[:50]  # Cap at 50 seed prompts

    # ------------------------------------------------------------------
    # Internal: mock fallback
    # ------------------------------------------------------------------

    def _train_mock(
        self,
        proposal: ImprovementProposal,
        training_data: Any,
        num_steps: int
    ) -> Dict[str, Any]:
        """
        Mock SGD training when oven is unavailable.

        [EDITOR] Runs at most 10 gradient steps so the evolution loop can
        complete end-to-end for testing. Not a substitute for GRPO.
        """
        logger.info(
            f"[MOCK] Training on proposal: {proposal.description[:80]}..."
        )

        optimizer = torch.optim.AdamW(
            self.oni_model.parameters(),
            lr=self.config.get('learning_rate', 1e-5)
        )

        total_loss = 0.0
        actual_steps = min(num_steps, 10)

        hidden_dim = getattr(
            getattr(self.oni_model, 'metacognition_module', None),
            'hidden_dim', 512
        )

        for step in range(actual_steps):
            optimizer.zero_grad()
            dummy_input = torch.randn(1, hidden_dim)
            try:
                output = self.oni_model(dummy_input)
                if isinstance(output, tuple):
                    output = output[0]
                loss = output.mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            except Exception as e:
                logger.debug(f"[MOCK] Step {step} failed: {e}")
                break

        return {
            'loss': total_loss / max(actual_steps, 1),
            'reward_mean': 0.0,
            'kl_divergence': 0.0,
            'steps_completed': actual_steps,
            'teacher': 'mock',
            'proposal_type': proposal.proposal_type.value,
            'mock': True,
        }
