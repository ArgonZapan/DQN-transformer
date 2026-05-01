import torch
import torch.nn as nn
import torch.nn.functional as F
from .grn import GRN


class VSN(nn.Module):
    """Variable Selection Network.

    Learns which input features are most relevant at each timestep.
    Input: [batch, seq_len, n_vars, d_model]  (each var already projected to d_model)
    Output: [batch, seq_len, d_model]          (weighted sum of processed vars)
    """

    def __init__(self, n_vars: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model

        # Per-variable GRN
        self.var_grns = nn.ModuleList([
            GRN(d_model, d_model, dropout=dropout) for _ in range(n_vars)
        ])

        # Weight GRN: flatten all vars → softmax weights
        self.weight_grn = GRN(n_vars * d_model, d_model, output_dim=n_vars, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, n_vars, d_model]
        b, s, n, d = x.shape
        assert n == self.n_vars

        # Per-variable processing: [batch * seq_len, d_model] each
        xi = x.view(b * s, n, d)
        processed = torch.stack([self.var_grns[i](xi[:, i, :]) for i in range(n)], dim=1)
        # processed: [batch * seq_len, n_vars, d_model]

        # Softmax weights over variables
        flat = xi.reshape(b * s, n * d)
        weights = F.softmax(self.weight_grn(flat), dim=-1)  # [batch * seq_len, n_vars]
        weights = weights.unsqueeze(-1)                      # [batch * seq_len, n_vars, 1]

        # Weighted sum
        out = (weights * processed).sum(dim=1)               # [batch * seq_len, d_model]
        return out.view(b, s, d)
