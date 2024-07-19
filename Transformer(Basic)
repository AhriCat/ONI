import torch
import torch.nn as nn
import math

class GRUCellTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GRUCellTransformer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Input embedding
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # GRU layers
        self.gru_cells = nn.ModuleList([nn.GRUCell(hidden_dim, hidden_dim) for _ in range(num_layers)])

        # Self-attention layer
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads=4)

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        # x shape: (seq_len, batch_size, input_dim)
        seq_len, batch_size, _ = x.size()

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = [torch.zeros(batch_size, self.hidden_dim).to(x.device) for _ in range(self.num_layers)]

        # Embed input
        embedded = self.embedding(x)

        outputs = []
        for t in range(seq_len):
            # Process through GRU cells
            h = embedded[t]
            for i, gru_cell in enumerate(self.gru_cells):
                h = gru_cell(h, hidden[i])
                hidden[i] = h

            # Self-attention
            h_unsqueezed = h.unsqueeze(0)
            attn_output, _ = self.self_attention(h_unsqueezed, h_unsqueezed, h_unsqueezed)
            h = attn_output.squeeze(0)

            # Output layer
            output = self.output_layer(h)
            outputs.append(output)

        outputs = torch.stack(outputs)
        return outputs, hidden

# Example usage
input_dim = 10
hidden_dim = 20
output_dim = 5
num_layers = 2
seq_len = 15
batch_size = 32

model = GRUCellTransformer(input_dim, hidden_dim, output_dim, num_layers)
input_data = torch.randn(seq_len, batch_size, input_dim)
output, hidden = model(input_data)

print(f"Output shape: {output.shape}")
print(f"Hidden state shape: {len(hidden)}, {hidden[0].shape}")
