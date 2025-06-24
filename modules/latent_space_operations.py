import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import os
import numpy as np

 

class FatDiffuser(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        hidden_dim: int, 
        num_heads: int = 16, 
        timesteps: int = 30, 
        max_seq_len: int = 512, 
        embedding: nn.Embedding = None  # Option to pass custom embedding
    ):
        super(FatDiffuser, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.timesteps = timesteps
        self.max_seq_len = max_seq_len

        # Use provided embedding or initialize a new one
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
            nn.init.xavier_uniform_(self.embedding.weight)

        # Positional and time embeddings
        self.positional_encoding = nn.Parameter(torch.randn(max_seq_len, hidden_dim))
        nn.init.xavier_uniform_(self.positional_encoding)
        self.time_embedding = nn.Parameter(torch.randn(timesteps, hidden_dim))
        nn.init.xavier_uniform_(self.time_embedding)

        # Diffusion layers
        self.diffusion_layers = nn.ModuleList([
            ReformerAttention(dim=self.hidden_dim, num_heads=self.num_heads)
            for _ in range(self.timesteps)
        ])

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

        # Noise schedule with clipping for gradient stability
        self.register_buffer('alpha_schedule', torch.clamp(torch.linspace(0.1, 1.0, timesteps), min=0.1, max=0.9))

        # Early stopping parameters
        self.early_stop_threshold = 1e-4  # Stop if the change in logits is below this
        self.max_early_stop_checks = 5   # Maximum steps to check for early stopping

    def forward(self, input_ids):
        device = input_ids.device
        batch_size, seq_len = input_ids.size()

        # Embed input tokens
        h = self.embedding(input_ids)

        # Add positional encoding
        pos_encoding = self.positional_encoding[:seq_len, :].unsqueeze(0).to(device)
        h = h + pos_encoding

        # Precompute time encodings and noise
        time_encodings = self.time_embedding.unsqueeze(1).unsqueeze(1).to(device)
        alphas = self.alpha_schedule.view(self.timesteps, 1, 1, 1).to(device)
        noise = torch.randn(self.timesteps, batch_size, seq_len, self.hidden_dim, device=device)

        logits_list = []  # Keep track of logits for early stopping
        for t in range(self.timesteps):
            # Add time encoding
            h = h + time_encodings[t]

            # Apply attention and normalization
            attn_output = self.diffusion_layers[t](h)
            h = h + attn_output
            h = self.layer_norm(h)

            # Add noise scaled by alpha
            h = h + alphas[t] * noise[t]

            # Calculate logits for current timestep
            logits = self.output_projection(h)
            logits_list.append(logits)

            # Early stopping: Check if logits change significantly
            if t > 0 and self._check_early_stopping(logits_list, t):
                break

        return logits

    def _check_early_stopping(self, logits_list, t):
        """
        Compares the change in logits to decide whether to stop early.
        """
        delta_logits = torch.abs(logits_list[t] - logits_list[t-1]).mean()
        if delta_logits < self.early_stop_threshold:
            self.max_early_stop_checks -= 1
        else:
            self.max_early_stop_checks = 5  # Reset counter if change is significant
        return self.max_early_stop_checks <= 0
