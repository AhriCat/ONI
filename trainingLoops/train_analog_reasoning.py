import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm

from modules.NLP.analogical_reasoning import AnalogicalReasoning
from modules.oni_loss_fn import UpgradedDynamicAdvancedNLPLoss

# === Configuration ===
class Config:
    input_dim = 512
    hidden_dim = 256
    num_heads = 4
    batch_size = 16
    lr = 5e-4
    epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "./checkpoints/analogical_reasoning.pt"
    accumulation_steps = 12  # Gradient accumulation from your loss
    ignore_index = -100      # Placeholder; replace with actual tokenizer.pad_token_id
    eval_interval = 1


# === Dummy Dataset (to be replaced with real structured analogy dataset) ===
class DummyAnalogyDataset(Dataset):
    def __init__(self, num_samples: int = 100):
        self.num_samples = num_samples
        self.seq_len = 10
        self.input_dim = Config.input_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        source = torch.randn(self.seq_len, self.input_dim)
        target = torch.randn(self.seq_len, self.input_dim)
        source_mask = torch.ones(self.seq_len).bool()
        target_mask = torch.ones(self.seq_len).bool()
        target_labels = torch.randint(0, 10, (self.seq_len,))  # Dummy labels; replace later
        return source, target, source_mask, target_mask, target_labels


# === Training Function ===
def train():
    cfg = Config()

    model = AnalogicalReasoning(cfg.input_dim, cfg.hidden_dim, cfg.num_heads).to(cfg.device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)

    # Loss function (tokenizer=None for now; replace when available)
    loss_fn = UpgradedDynamicAdvancedNLPLoss(ignore_index=cfg.ignore_index)
    loss_fn.to(cfg.device)

    dataset = DummyAnalogyDataset()
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")):
            source, target, source_mask, target_mask, labels = [x.to(cfg.device) for x in batch]

            outputs = model(source, target, source_mask, target_mask)

            # This assumes you'll later replace outputs['correspondence_scores'] with usable logits
            # and labels with appropriate targets (e.g., entity indices or classes)
            logits = outputs["correspondence_scores"]  # shape: [B, S, T]
            # Dummy reshape for matching target format expected by loss
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)

            # Compute and backprop loss (accumulated inside loss module)
            loss = loss_fn(logits, labels, epoch=epoch, max_epochs=cfg.epochs, model=model)
            total_loss += loss.item()

            # Optimizer step handled when gradients accumulate
            if loss_fn.accumulated_gradients == 0:
                optimizer.step()
                optimizer.zero_grad()

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch + 1}] Avg Loss: {avg_loss:.4f}")

        # Save model
        if (epoch + 1) % cfg.eval_interval == 0:
            os.makedirs(os.path.dirname(cfg.checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), cfg.checkpoint_path)
            print(f"Checkpoint saved to {cfg.checkpoint_path}")


if __name__ == "__main__":
    train()
