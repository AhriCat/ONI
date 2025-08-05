import torch
import torch.nn as nn
import torch.nn.functional as F
from latent_space_operations import LatentSpaceOperations, LatentSpaceConfig
from snapshot_memory import SnapshotMemorySystem
from working_memory_module import WorkingMemoryModule

class OniAutoregressiveWorldModel(nn.Module):
    def __init__(self, latent_dim, vision_dim, audio_dim, motor_dim, hidden_dim, config: LatentSpaceConfig):
        super().__init__()
        self.config = config
        self.latent_ops = LatentSpaceOperations(config)

        self.input_proj = nn.Linear(vision_dim + audio_dim + motor_dim, latent_dim)
        self.autoregressive = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, vision_dim + audio_dim + motor_dim)

        # Minimal memory systems needed for Genie-2-level modeling
        self.snapshot_memory = SnapshotMemorySystem(hidden_dim)
        self.working_memory = WorkingMemoryModule(config)

        # Temporal consistency loss module
        self.temporal_consistency_weight = 0.1

        # Reward predictor
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, vision, audio, motor, hidden=None, memory_context=None):
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

    def act(self, current_state, use_memory=True, memory_context=None):
        vision, audio, motor = current_state

        if use_memory:
            # Use working memory only for fast context
            timestamp = torch.tensor([self.config.max_seq_len], dtype=torch.float32)
            query = torch.cat([vision, audio, motor], dim=-1)
            memory_context = self.working_memory.retrieve_memory(query, timestamp).unsqueeze(0).unsqueeze(0)

        _, _, predicted_motor, _, _ = self.forward(
            vision.unsqueeze(1), audio.unsqueeze(1), motor.unsqueeze(1), memory_context=memory_context
        )

        return predicted_motor.squeeze(1)
