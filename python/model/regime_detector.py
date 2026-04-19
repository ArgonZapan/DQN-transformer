import torch
import torch.nn as nn


class RegimeDetector(nn.Module):
    """
    Moduł wykrywający reżim rynkowy (trend/boczny/volatilny).
    Pozwala modelowi dostosować strategię do aktualnych warunków.
    Opcjonalny — placeholder do uzupełnienia po eksperymentach.
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
