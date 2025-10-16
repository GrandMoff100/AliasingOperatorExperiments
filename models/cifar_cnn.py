import torch
import torch.nn as nn

BASIS_FUNCTIONS = 200


class Normalize(nn.Module):
    """Normalize input images using fixed mean and std."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        return (x - self.mean) / self.std


class CifarConvNet(nn.Module):
    def __init__(
        self,
        mean: torch.Tensor = torch.tensor([0.4914, 0.4822, 0.4465]).reshape(
            1, -1, 1, 1
        ),
        std: torch.Tensor = torch.tensor([0.2470, 0.2435, 0.2616]).reshape(1, -1, 1, 1),
    ):
        super().__init__()
        self.net = nn.Sequential(
            Normalize(mean, std),
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, BASIS_FUNCTIONS),
            nn.BatchNorm1d(BASIS_FUNCTIONS),
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
