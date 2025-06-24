import torch.nn as nn
import torch 

class DynamicLayer(nn.Module):
    def __init__(self, 
                 d_model, 
                 d_ff, 
                 radius=1.0, 
                 num_heads=8, 
                 noise_factor=0.1):
        super().__init__()
        self.radius = radius
        self.noise_factor = noise_factor

        # Placeholder for EfficientAttention - you'll need to implement this
        self.attention = EfficientAttention(
            hidden_dim= 896, 
            num_heads=num_heads
        )

        self.ffn = HypersphericalFFN(d_model, d_ff, radius, timesteps=10)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # More robust field representations
        self.vector_field = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 3)
        )
        self.scalar_field = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x, mask=None, time_step=None):
        # Prepare the input dimensions
        batch_size, seq_len, d_model = x.size()

        # Handle optional mask
        if mask is not None:
            # Ensure mask is [batch_size, seq_len]
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            else:
                raise ValueError("Invalid mask dimensions. Expected 2D or 3D tensor.")

            # Ensure compatibility with attention module
            mask = mask.expand(batch_size, self.attention.num_heads, seq_len, seq_len)

        # Perform attention (with or without mask)
        attn_output = self.attention(x, mask)  # EfficientAttention expects an explicit mask
        x = self.norm1(x + attn_output)

        # Apply Hyperspherical FFN with optional time step
        if time_step is None:
            time_step = 0  # Default to the first timestep
        ffn_output = self.ffn(x, time_step)
        x = self.norm2(x + ffn_output)

        # Compute fields
        vector_field = self.vector_field(x)
        scalar_field = self.scalar_field(x)

        # Compute energy and scale the output
        field_energy = self._compute_field_energy(vector_field, scalar_field)
        energy_scale = torch.sigmoid(field_energy / self.radius).unsqueeze(-1)
        x = x * energy_scale

        # Apply optional noise and efficiency mode
        if torch.mean(field_energy) < self.radius * 0.5:
            x = self._apply_adaptive_noise(x, field_energy)
            x = self._efficiency_mode(x)

        return x, field_energy

    
    def _compute_field_energy(self, vector_field, scalar_field):
        """More robust energy computation."""
        vector_energy = torch.norm(vector_field, dim=-1)
        scalar_energy = torch.abs(scalar_field.squeeze())
        return torch.mean(vector_energy + scalar_energy)

    def _apply_adaptive_noise(self, x, energy):
        """Adaptive noise injection with energy-based scaling."""
        noise_strength = self.noise_factor * torch.exp(-energy / self.radius)
        noise = noise_strength * torch.randn_like(x)
        return x + noise

    def _efficiency_mode(self, x):
        """Low-energy efficiency mode with soft approximation."""
        return x + F.relu(torch.mean(x, dim=-1, keepdim=True))
