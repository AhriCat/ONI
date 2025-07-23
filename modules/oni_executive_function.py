import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class TimeAwareExecutiveNet(nn.Module):
    def __init__(self, input_channels, output_dim, seq_len=49):
        super(TimeAwareExecutiveNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = ResidualBlock(64, 128)
        self.layer2 = ResidualBlock(128, 256)
        self.layer3 = ResidualBlock(256, 512)

        # Self-attention and time embedding
        self.attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.time_embedding = nn.Linear(1, 512)

        # Temporal urgency projection
        self.urgency_projection = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.Sigmoid()  # output urgency weighting
        )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * seq_len, 1024)
        self.fc2 = nn.Linear(1024, output_dim)

    def forward(self, x, time_remaining):
        batch_size = x.size(0)

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Flatten spatial to sequence for attention
        x_seq = x.flatten(2).transpose(1, 2)  # shape (B, Seq, 512)

        # Time embedding and urgency modulation
        time_embed = self.time_embedding(time_remaining.unsqueeze(-1))  # shape (B, 512)
        time_embed = time_embed.unsqueeze(1).expand(-1, x_seq.size(1), -1)
        x_seq += time_embed  # add positional time context

        attn_out, _ = self.attn(x_seq, x_seq, x_seq)
        urgency_weight = self.urgency_projection(time_remaining.unsqueeze(-1))
        attn_out = attn_out * urgency_weight.unsqueeze(1)  # time-dependent focusing

        x = attn_out.flatten(1)  # (B, 512 * seq_len)
        x = F.elu(self.fc1(x))
        x = self.fc2(x)
        return x

# Example usage
input_channels = 3
output_dim = 896
model = TimeAwareExecutiveNet(input_channels, output_dim)

# Dummy inputs
x = torch.randn(4, 3, 224, 224)
time_remaining = torch.tensor([0.5, 1.0, 1.5, 2.0])  # normalized remaining time (in seconds)

out = model(x, time_remaining)
print(out.shape)  # Expected: (4, 896)
