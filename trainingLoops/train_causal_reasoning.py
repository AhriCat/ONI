import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import numpy as np

from modules.NLP.causal_reasoning import CausalReasoning
from modules.oni_loss_fn import UpgradedDynamicAdvancedNLPLoss

# === Configuration ===
class Config:
    input_dim = 512
    hidden_dim = 256
    num_variables = 20
    batch_size = 16
    lr = 1e-4
    epochs = 25
    ignore_index = -100
    accumulation_steps = 12
    intervention_prob = 0.5
    checkpoint_path = "./checkpoints/causal_reasoning.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Dummy Dataset with Intervention Target ===
class DummyCausalDataset(Dataset):
    def __init__(self, n_samples=1000, input_dim=Config.input_dim):
        self.data = torch.randn(n_samples, input_dim)
        self.targets = self.data + torch.randn_like(self.data) * 0.05  # noisy copy

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# === Generate synthetic intervention queries ===
def generate_intervention_query(batch_x: torch.Tensor, intervention_prob: float = 0.5) -> torch.Tensor:
    batch_size, input_dim = batch_x.shape
    num_variables = Config.num_variables

    intervention_logits = torch.randn(batch_size, num_variables * 2).to(batch_x.device)

    # Randomly mask variables (probabilistic intervention)
    intervention_mask = (torch.rand(batch_size, num_variables) < intervention_prob).float()
    intervention_values = torch.randn_like(intervention_mask)

    intervention_logits[:, :num_variables] = intervention_mask
    intervention_logits[:, num_variables:] = intervention_values

    return intervention_logits


# === Training Function ===
def train():
    cfg = Config()
    model = CausalReasoning(cfg.input_dim, cfg.hidden_dim, cfg.num_variables).to(cfg.device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = UpgradedDynamicAdvancedNLPLoss(ignore_index=cfg.ignore_index).to(cfg.device)
    dataloader = DataLoader(DummyCausalDataset(), batch_size=cfg.batch_size, shuffle=True)

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0

        for step, (inputs, targets) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.epochs}")):
            inputs = inputs.to(cfg.device)
            targets = targets.to(cfg.device)

            # Generate intervention query
            intervention_query = generate_intervention_query(inputs, cfg.intervention_prob)

            output_dict = model(inputs, intervention_query=intervention_query)
            predictions = output_dict["output"]

            logits = predictions.view(-1, predictions.size(-1))
            targets_flat = targets.view(-1, targets.size(-1))
            target_indices = torch.argmax(targets_flat, dim=-1)

            loss = loss_fn(logits, target_indices, epoch=epoch, max_epochs=cfg.epochs, model=model)
            total_loss += loss.item()

            if loss_fn.accumulated_gradients == 0:
                optimizer.step()
                optimizer.zero_grad()

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}")

        # Optional: visualize learned causal graph
        adj = output_dict["adjacency_matrix"]
        print("Adjacency (mean abs):", adj.abs().mean().item())

        # Save model
        os.makedirs(os.path.dirname(cfg.checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), cfg.checkpoint_path)
        print(f"Saved model checkpoint to {cfg.checkpoint_path}")


if __name__ == "__main__":
    train()
