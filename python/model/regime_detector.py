import torch
import torch.nn as nn


class RegimeDetector(nn.Module):
    """
    Module detecting market regime (trending/ranging/volatile).
    Allows the model to adapt its strategy to current conditions.
    Optional — placeholder to be filled in after experiments.
    """

    def __init__(self, input_dim, num_regimes=3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_regimes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.classifier(x)
