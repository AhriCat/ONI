import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from latent_space_operations import LatentSpaceOperations, LatentSpaceConfig
from snapshot_memory import SnapshotMemorySystem
from working_memory_module import WorkingMemoryModule
from modules.memory.gaussian_splat_memory import GaussianSplatMemory

class OniAutoregressiveWorldModel(nn.Module):
    def __init__(
        self,
        latent_dim,
        vision_dim,
        audio_dim,
        motor_dim,
        hidden_dim,
        config: LatentSpaceConfig,
        use_scene_memory: bool = True,
        n_gaussians: int = 2048,
        max_scenes: int = 64,
    ):
        super().__init__()
        self.config = config
        self.latent_ops = LatentSpaceOperations(config)

        self.input_proj = nn.Linear(vision_dim + audio_dim + motor_dim, latent_dim)
        self.autoregressive = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, vision_dim + audio_dim + motor_dim)

        # Minimal memory systems needed for Genie-2-level modeling
        self.snapshot_memory = SnapshotMemorySystem(hidden_dim)
        self.working_memory = WorkingMemoryModule(config)

        # 3D Gaussian Splat scene memory — encodes spatial scene context as
        # a latent that is fused into the diffusion-based latent refinement step
        self.scene_memory: Optional[GaussianSplatMemory] = (
            GaussianSplatMemory(
                hidden_dim=hidden_dim,
                vision_dim=vision_dim,
                n_gaussians=n_gaussians,
                max_scenes=max_scenes,
            )
            if use_scene_memory else None
        )

        # Project scene context (hidden_dim) into the memory_context slot expected
        # by LatentSpaceOperations (config.hidden_dim). A linear keeps it compatible
        # even if the two hidden dims differ.
        if use_scene_memory:
            self.scene_ctx_proj = nn.Linear(hidden_dim, config.hidden_dim)

        # Temporal consistency loss module
        self.temporal_consistency_weight = 0.1

        # Reward predictor
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        vision,
        audio,
        motor,
        hidden=None,
        memory_context=None,
        scene_id=None,
    ):
        """
        Args:
            vision, audio, motor: modality tensors (B, T, dim)
            hidden:               GRU hidden state or None
            memory_context:       optional external context (B, 1, hidden_dim)
            scene_id:             if set and scene_memory is available, the stored
                                  Gaussian scene for this ID is encoded and fused
                                  into memory_context (overrides explicit memory_context
                                  only when memory_context is None)
        """
        # Optionally enrich memory_context with the current 3DGS scene
        if scene_id is not None and self.scene_memory is not None and memory_context is None:
            scene_ctx = self.scene_memory.encode_scene(scene_id)   # (1, D) or None
            if scene_ctx is not None:
                memory_context = self.scene_ctx_proj(scene_ctx).unsqueeze(1)  # (1, 1, D)

        # Combine inputs into single latent representation
        combined_input = torch.cat([vision, audio, motor], dim=-1)
        latent_input = F.relu(self.input_proj(combined_input))

        # Autoregressive modeling
        output_seq, hidden = self.autoregressive(latent_input, hidden)

        # Diffusion-based latent refinement
        refined_output = self.latent_ops.encode_to_latent(output_seq, memory_context)

        # Decode predicted states (vision, audio, motor)
        predicted_output = self.output_proj(refined_output)
        predicted_vision, predicted_audio, predicted_motor = torch.split(
            predicted_output, [vision.shape[-1], audio.shape[-1], motor.shape[-1]], dim=-1
        )

        # Reward prediction
        predicted_reward = self.reward_head(refined_output)

        return predicted_vision, predicted_audio, predicted_motor, predicted_reward, hidden

    def observe_scene(
        self,
        scene_id,
        vision_features: torch.Tensor,
        xyz: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Build or update the Gaussian Splat scene for scene_id from vision observations.

        Call this whenever ONI receives a new visual frame so the scene memory
        stays current before the next forward() / act() call.

        Args:
            scene_id:        hashable key (room coord, episode ID, …)
            vision_features: (B, T, vision_dim)
            xyz:             (B, T, 3) optional world-space positions
        """
        if self.scene_memory is None:
            return
        self.scene_memory.update_scene(scene_id, vision_features, xyz)

    def temporal_consistency_loss(self, latents):
        # Penalize large changes in latent space over time
        diffs = latents[:, 1:] - latents[:, :-1]
        loss = torch.mean(torch.norm(diffs, dim=-1))
        return self.temporal_consistency_weight * loss

    def visualize(self, initial_states, steps_ahead=10, memory_context=None):
        vision, audio, motor = initial_states
        predictions = []
        hidden = None

        for _ in range(steps_ahead):
            vision, audio, motor, _, hidden = self.forward(
                vision.unsqueeze(1), audio.unsqueeze(1), motor.unsqueeze(1), hidden, memory_context
            )
            predictions.append((vision.squeeze(1), audio.squeeze(1), motor.squeeze(1)))

        return predictions

    def act(
        self,
        current_state,
        use_memory: bool = True,
        memory_context=None,
        scene_id=None,
    ):
        """
        Predict the next motor action.

        Args:
            current_state: (vision, audio, motor) tuple
            use_memory:    whether to retrieve working-memory context
            memory_context: explicit context override
            scene_id:       if set, also fuse the Gaussian scene context
        """
        vision, audio, motor = current_state

        if use_memory:
            timestamp = torch.tensor([self.config.max_seq_len], dtype=torch.float32)
            query = torch.cat([vision, audio, motor], dim=-1)
            memory_context = self.working_memory.retrieve_memory(query, timestamp).unsqueeze(0).unsqueeze(0)

        _, _, predicted_motor, _, _ = self.forward(
            vision.unsqueeze(1), audio.unsqueeze(1), motor.unsqueeze(1),
            memory_context=memory_context,
            scene_id=scene_id,
        )

        return predicted_motor.squeeze(1)
