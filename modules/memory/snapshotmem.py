class SnapshotMemorySystem(nn.Module):
    def __init__(self, hidden_dim: int, memory_size: int = 8192):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.current_index = 0
        
        # Initialize memory bank as a Parameter instead of a buffer
        # This ensures proper handling of gradients
        self.register_parameter(
            'memory_bank',
            nn.Parameter(torch.zeros(memory_size, hidden_dim), requires_grad=True)
        )
        
    def update(self, snapshot: torch.Tensor):
        """
        Update memory bank with new snapshots
        Args:
            snapshot: Tensor of shape (batch_size, seq_len, hidden_dim)
        """
        # Get mean representation across sequence length
        snapshot = snapshot.mean(dim=1)  # (batch_size, hidden_dim)
        batch_size = snapshot.size(0)
        
        # Calculate indices for the batch
        indices = torch.arange(
            self.current_index,
            self.current_index + batch_size
        ) % self.memory_size
        
        # Use scatter_ instead of direct assignment
        # Create a temporary tensor of the same size as memory_bank
        new_memory = self.memory_bank.clone()
        new_memory.data[indices] = snapshot.detach().clone()
        
        # Update the parameter
        self.memory_bank = nn.Parameter(new_memory)
        
        # Update current index
        self.current_index = (self.current_index + batch_size) % self.memory_size
        
    def get_snapshots(self, num_snapshots: Optional[int] = None) -> torch.Tensor:
        """
        Retrieve snapshots from memory
        Args:
            num_snapshots: Number of most recent snapshots to retrieve
        Returns:
            Tensor of shape (num_snapshots, hidden_dim)
        """
        if num_snapshots is None:
            num_snapshots = self.memory_size
            
        num_snapshots = min(num_snapshots, self.memory_size)
        
        # Calculate indices of most recent snapshots
        indices = torch.arange(
            self.current_index - num_snapshots,
            self.current_index
        ) % self.memory_size
        
        return self.memory_bank[indices]
