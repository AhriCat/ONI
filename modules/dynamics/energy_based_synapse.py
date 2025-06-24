import torch
import torch.nn as nn
import torch.nn.functional as F

class EnergyBasedSynapse(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EnergyBasedSynapse, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim))
        self.dynamic_param = nn.Parameter(torch.zeros(output_dim))
        self.history = nn.Parameter(torch.zeros(output_dim), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        # Ensure x is 2D: (batch_size, input_dim)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() == 3:
            x = x.squeeze(0)

        dynamic_weight = self.weight * torch.sigmoid(self.dynamic_param + self.history)
        
        with torch.no_grad():
            self.history.data = 0.9 * self.history + 0.1 * torch.tanh(F.linear(x, self.weight)).mean(dim=0)
        
        output = F.linear(x, dynamic_weight, self.bias)
        energy = torch.mean(output**2, dim=1, keepdim=True)
        
        return output, energy

# Example usage
input_dim = 896 
output_dim = 896
batch_size = 1  # Example batch size
seq_len = 896  # Example sequence length

# Create an instance of EnergyBasedSynapse
dynamic_layer = EnergyBasedSynapse(input_dim, output_dim)

# Example input tensor of shape (batch_size, seq_len, input_dim)
x = torch.randn(batch_size, seq_len, input_dim)

# Reshape x to (batch_size * seq_len, input_dim) for linear layer compatibility
x = x.view(-1, input_dim)

# Forward pass through the dynamic layer
output, energy = dynamic_layer(x)
print(output.shape, energy.shape)
