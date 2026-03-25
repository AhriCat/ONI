# File: evolution/oni_robotics_trainer.py
"""
Specialized training pipeline for ONI's robotics capabilities.

Uses:
- Diffusion-based trajectory generation
- ROS integration for environment feedback (optional)
- Haptic/tactile modules for proprioceptive learning

Works with the Superintelligence Oven's SmallCausalDiffusionTeacher when
modality="motion" is passed to bake().
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import logging
import numpy as np

from .improvement_proposal import ModuleType

logger = logging.getLogger(__name__)


class ONIRoboticsTrainer:
    """Robotics-specific training pipeline for ONI evolution."""

    def __init__(
        self,
        oni_robotics_module: nn.Module,
        oni_haptics_module: Optional[nn.Module] = None,
        joint_dim: int = 72,
        intent_dim: int = 512,
        use_ros: bool = False
    ):
        self.robotics = oni_robotics_module
        self.haptics = oni_haptics_module
        self.joint_dim = joint_dim
        self.intent_dim = intent_dim
        self.use_ros = use_ros
        self.ros_available = False

        if use_ros:
            self._init_ros()

    def _init_ros(self):
        """Initialize ROS integration if available."""
        try:
            import rospy
            rospy.init_node('oni_robotics_trainer', anonymous=True)
            self.ros_available = True
            logger.info("ROS integration initialized")
        except ImportError:
            logger.warning("ROS not available — using simulated environment")
            self.ros_available = False

    def generate_trajectory(
        self,
        intent: torch.Tensor,
        num_steps: int = 50,
        noise_scale: float = 0.1
    ) -> torch.Tensor:
        """
        Generate a trajectory from intent using diffusion denoising.

        Args:
            intent: Intent vector [batch, intent_dim]
            num_steps: Number of trajectory steps
            noise_scale: Initial noise scale

        Returns:
            trajectory: [batch, num_steps, joint_dim]
        """
        batch_size = intent.size(0)
        trajectory = torch.randn(batch_size, num_steps, self.joint_dim) * noise_scale

        try:
            with torch.no_grad():
                for t in range(num_steps - 1, -1, -1):
                    time_embed = torch.full((batch_size, 1), t / num_steps)
                    step_input = torch.cat([
                        trajectory[:, t, :],
                        intent,
                        time_embed
                    ], dim=-1)

                    if hasattr(self.robotics, 'denoise_step'):
                        trajectory[:, t, :] = self.robotics.denoise_step(step_input)
        except Exception as e:
            logger.debug(f"Diffusion denoising not supported: {e}")

        return trajectory

    def evaluate_trajectory(
        self,
        trajectory: torch.Tensor,
        target_pose: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Evaluate trajectory quality across multiple metrics."""
        metrics: Dict[str, float] = {}

        # Smoothness: lower jerk = smoother motion
        if trajectory.size(1) >= 3:
            velocity = trajectory[:, 1:, :] - trajectory[:, :-1, :]
            acceleration = velocity[:, 1:, :] - velocity[:, :-1, :]
            if acceleration.size(1) >= 2:
                jerk = acceleration[:, 1:, :] - acceleration[:, :-1, :]
                metrics['smoothness'] = 1.0 / (1.0 + jerk.norm(dim=-1).mean().item())
            else:
                metrics['smoothness'] = 0.5
        else:
            metrics['smoothness'] = 0.5

        # Target reaching accuracy
        if target_pose is not None:
            final_pose = trajectory[:, -1, :target_pose.size(-1)]
            distance = (final_pose - target_pose).norm(dim=-1).mean().item()
            metrics['target_accuracy'] = 1.0 / (1.0 + distance)
        else:
            metrics['target_accuracy'] = 0.5

        # Joint limit compliance (assumes normalized joints in [-1, 1])
        out_of_range = (
            (trajectory.abs() > 1.0).float().sum() / trajectory.numel()
        ).item()
        metrics['joint_compliance'] = 1.0 - out_of_range

        # Haptic consistency: if haptics module available, check tactile coherence
        if self.haptics is not None:
            try:
                with torch.no_grad():
                    haptic_input = trajectory[:, -1, :]
                    haptic_score = self.haptics(haptic_input)
                    if isinstance(haptic_score, tuple):
                        haptic_score = haptic_score[0]
                    metrics['haptic_consistency'] = torch.sigmoid(
                        haptic_score.mean()
                    ).item()
            except Exception as e:
                logger.debug(f"Haptic evaluation skipped: {e}")
                metrics['haptic_consistency'] = 0.5

        metrics['overall'] = float(np.mean(list(metrics.values())))
        return metrics

    def train_step(
        self,
        intent_batch: torch.Tensor,
        target_trajectories: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, Any]:
        """Single supervised training step for the robotics module."""
        optimizer.zero_grad()

        predicted = self.generate_trajectory(
            intent_batch, num_steps=target_trajectories.size(1)
        )
        loss = nn.functional.mse_loss(predicted, target_trajectories)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.robotics.parameters(), max_norm=1.0)
        optimizer.step()

        return {
            'loss': loss.item(),
            'metrics': self.evaluate_trajectory(
                predicted.detach(), target_trajectories[:, -1, :]
            )
        }

    def build_oven_prompts_for_robotics(
        self,
        num_prompts: int = 20
    ) -> List[str]:
        """
        Generate motion-domain prompts for the Superintelligence Oven.

        These are passed to bake() when modality="motion" and the
        SmallCausalDiffusionTeacher is active.
        """
        actions = [
            "wave hello", "reach forward", "grasp object", "release grip",
            "walk forward", "turn left", "sit down", "stand up",
            "nod yes", "shake head no", "point at target", "push button",
            "open door", "pick up box", "place on shelf", "dance excitedly",
            "scratch head", "fold arms", "stretch arms up", "bend forward",
        ]
        return actions[:num_prompts]
