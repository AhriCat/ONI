from modules.attention.efficient_attention import EfficientAttention 
class ChainOfThought(nn.Module):
    def __init__(self, input_dim, memory_size):
        super().__init__()
        self.input_dim = input_dim
        self.memory_size = memory_size
        self.memory = nn.Parameter(torch.randn(memory_size, input_dim) / math.sqrt(input_dim))  # Proper initialization
        self.attention = EfficientAttention(input_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(input_dim)
        self.current_size = 0
        self.pointer = 0

    def forward(self, x):
        x = x.float()
        batch_size = x.size(0)
        
        # Update memory with new information
        x_flat = x.view(-1, self.input_dim)
        num_new_items = x_flat.size(0)
        indices = torch.arange(num_new_items) % self.memory_size
        self.memory.data[indices] = x_flat.to(self.memory.dtype)
        
        # Use attention to access memory
        memory_batch = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        attended_memory, _ = self.attention(x, memory_batch, memory_batch)
        
        # Combine with input through residual connection
        output = self.norm(x + attended_memory)
        
        self.current_size = min(self.current_size + num_new_items, self.memory_size)
        self.pointer = (self.pointer + num_new_items) % self.memory_size
        
        return output
