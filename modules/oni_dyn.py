import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicSynapse(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(DynamicSynapse, self).__init__()
        # Define convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 1024, kernel_size=3, padding=1)
        # Use global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Output size is 1x1

        # Define a fully connected layer to map to desired output dimension
        self.fc = nn.ELU(1024, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.global_pool(x)  # Apply global average pooling
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layer
        x = self.fc(x)  # Map to the output dimension
        return x

# Example usage
input_channels = 2
output_dim = 512 # Adjust according to your needs

