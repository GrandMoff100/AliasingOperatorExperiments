import torch
import torch.nn as nn

BASIS_FUNCTIONS = 200


class Normalize(nn.Module):
    """Normalize input images using fixed mean and std."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean.reshape(1, -1, 1, 1))
        self.register_buffer("std", std.reshape(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class CifarConvNet(nn.Module):
    def __init__(
        self,
        mean: torch.Tensor = torch.tensor([0.4914, 0.4822, 0.4465]),
        std: torch.Tensor = torch.tensor([0.2470, 0.2435, 0.2616]),
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


class BiggerCifarConvNet(nn.Module):
    def __init__(
        self,
        mean: torch.Tensor = torch.tensor([0.4914, 0.4822, 0.4465]),
        std: torch.Tensor = torch.tensor([0.2470, 0.2435, 0.2616]),
        num_classes: int = 10,
    ):
        super().__init__()

        self.net = nn.Sequential(
            # Normalize input
            Normalize(mean, std),

            # --- Block 1 ---
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
            nn.Dropout(0.25),

            # --- Block 2 ---
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
            nn.Dropout(0.25),

            # --- Block 3 ---
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8 -> 4x4
            nn.Dropout(0.4),

            nn.Flatten(),

            # --- Fully Connected Head ---
            nn.Linear(256 * 4 * 4, BASIS_FUNCTIONS),
            nn.BatchNorm1d(BASIS_FUNCTIONS),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(BASIS_FUNCTIONS, num_classes),
        )

    def forward(self, x) -> torch.Tensor:
        return self.net(x)

    def embed(self, x) -> torch.Tensor:
        """Extract features before the final classification layer."""
        for layer_index in range(len(self.net) - 1):
            x = self.net[layer_index](x)
        return x
