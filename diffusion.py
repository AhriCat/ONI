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

image_dir = 'C:/users/jonny/Documents/PATH/ONI/knowledge_base/picturememory/'
os.makedirs(image_dir, exist_ok=True)

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, image_extensions=None):
        if image_extensions is None:
            image_extensions = ('.png','.jpg', '.jpeg', '.bmp','.gif')
            self.img_dir = image_dir
            self.transform = transform
            self.img_names = [img for img in os.listdir(img_dir) if img.endswith(image_extensions) and os.path.isfile(os.path.join(img_dir, img))]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            return image
        

transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to 64x64 (or any desired size)
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

  # Update this with the path to your folder
dataset2 = CustomImageDataset(image_dir, transform=transform)


class DDPM(nn.Module):
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        super(DDPM, self).__init__()
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def forward(self, x, t):
        noise = torch.randn_like(x)
        return (self.alphas_cumprod[t] ** 0.5) * x + (1 - self.alphas_cumprod[t]) ** 0.5 * noise

    def reverse(self, x, t):
        return x

    def sample(self, shape):
        x = torch.randn(shape)
        for t in reversed(range(self.timesteps)):
            x = self.reverse(x, t)
        return x

    def params(self):
        return list(self.parameters())

timesteps = 1000
beta_start = 0.1
beta_end = 0.2

model = DDPM(timesteps, beta_start, beta_end)

dataset3 = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]))

for dataset in [dataset3]:  # Remove dataset2 as it is not defined
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    for batch in dataloader:
        x = batch[0]
        t = torch.randint(0, timesteps, (x.shape[0],))
        noisy_x = model(x, t)
        # Train the model using the noisy_x and t
        #...
        # Sample from the model
        sampled_image = model.sample((4, 1, 28, 28))  # Change the shape to (4, 1, 28, 28) as FashionMNIST images are 28x28
        sampled_image = sampled_image.detach().numpy()
        image = Image.fromarray(np.squeeze(sampled_image[0], axis=0))  # Remove the extra channel dimension
        # Display the image
        # image.show()

optimizer = optim.Adam(model.parameters(), lr=1e-3)
print(f'Model parameters: {len(list(model.parameters()))}')