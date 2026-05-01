import torch
import torch.nn as nn
import torch.nn.functional as F


class GRN(nn.Module):
    """Gated Residual Network — core TFT building block.

    a → ELU(W2·a) → W1 → GLU → LayerNorm(a + gated)
    Residual projection applied when input_dim != output_dim.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = None, dropout: float = 0.0):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        self.W2       = nn.Linear(input_dim,  hidden_dim)
        self.W1       = nn.Linear(hidden_dim, output_dim * 2)  # split for GLU
        self.residual = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.norm     = nn.LayerNorm(output_dim)
        self.drop     = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.zeros_(self.W2.bias)
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.zeros_(self.W1.bias)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        eta2 = F.elu(self.W2(a))
        eta1 = self.drop(self.W1(eta2))
        h, gate = eta1.chunk(2, dim=-1)
        gated = h * torch.sigmoid(gate)
        return self.norm(self.residual(a) + gated)
