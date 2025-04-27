
class HypersphericalFFN(nn.Module):
    def __init__(self, d_model, d_ff, radius, timesteps):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()  # Consider GELU over ReLU for smoother gradients
        self.linear2 = nn.Linear(d_ff, d_model)
        self.radius = radius

        # Improved time embedding initialization
        self.time_embedding = nn.Parameter(
            torch.randn(timesteps, d_model) * 0.02,  # Smaller scale initialization
            requires_grad=True
        )
    def forward(self, x, t):
        # Ensure t is a tensor
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.long, device=self.time_embedding.device)

        # Safer time embedding handling
        t = torch.clamp(t, 0, self.time_embedding.size(0) - 1)
        time_embed = self.time_embedding[t].unsqueeze(0)
        
        x = x + time_embed
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        # More numerically stable hyperspherical projection
        norm = torch.norm(x, dim=-1, keepdim=True)
        x = torch.where(
            norm > 0, 
            self.radius * x / (norm + 1e-8), 
            x
        )
        return x
