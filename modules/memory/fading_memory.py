class FadingMemorySystem(nn.Module):
    def __init__(self, hidden_dim: int, decay_rate: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.decay_rate = decay_rate
        self.memory_state = None
        
    def forward(self, x):
        if self.memory_state is None:
            self.memory_state = torch.zeros_like(x)
            
        # Apply exponential decay to existing memory
        self.memory_state = self.memory_state * math.exp(-self.decay_rate)
        
        # Update memory with new information
        self.memory_state = self.memory_state + x
        
        return self.memory_state
