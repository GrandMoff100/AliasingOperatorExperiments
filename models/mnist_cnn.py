import torch
import torch.nn as nn

BASIS_FUNCTIONS = 200


class Normalize(nn.Module):
    """Normalize input images using fixed mean and std."""

    def __init__(self, mean=0.1307, std=0.3081):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))

    def forward(self, x):
        return (x - self.mean) / self.std


class MnistConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            Normalize(),
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, BASIS_FUNCTIONS),
            nn.ReLU(),
            nn.Linear(BASIS_FUNCTIONS, 10),
        )

    def forward(self, x) -> torch.Tensor:
        return self.net(x)

    def embed(self, x) -> torch.Tensor:
        """Extract features before the final classification layer."""
        for layer_index in range(len(self.net) - 1):
            x = self.net[layer_index](x)
        return x
