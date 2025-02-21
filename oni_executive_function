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

class ExecutiveDecisionNet(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(ExecutiveDecisionNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = ResidualBlock(64, 128)
        self.layer2 = ResidualBlock(128, 256)
        self.layer3 = ResidualBlock(256, 512)

        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)

        self.flatten = nn.Flatten()
        self.fc1 = nn.ELU(512 * 7 * 7, 1024)
        self.fc2 = nn.ELU(1024, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Flatten and pass through attention mechanism
        x_flat = self.flatten(x)
        x_flat = x_flat.view(x_flat.size(0), -1, 896)
        x_attn, _ = self.attention(x_flat, x_flat, x_flat)
        x_attn = x_attn.view(x.size(0), -1)

        x = F.relu(self.fc1(x_attn))
        x = self.fc2(x)
        return x

# Example usage with input dimensions
input_channels = 3  # RGB channels for visual input
output_dim = 896  # Number of possible actions
exec_func = ExecutiveDecisionNet(input_channels, output_dim)
