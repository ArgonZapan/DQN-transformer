import torch
import torch.nn as nn


class QuantileHead(nn.Module):
    """Predicts price threshold at fixed quantile levels.

    Output shape: [batch, n_horizons, n_quantiles, n_directions]
    Values are % price movements (e.g. 0.02 = price moves 2%).
    """

    def __init__(self, d_model: int, n_horizons: int, n_quantiles: int, n_directions: int):
        super().__init__()
        out_dim = n_horizons * n_quantiles * n_directions
        self.n_horizons    = n_horizons
        self.n_quantiles   = n_quantiles
        self.n_directions  = n_directions
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, d_model]
        out = self.fc(x)
        return out.view(-1, self.n_horizons, self.n_quantiles, self.n_directions)


class ThresholdHead(nn.Module):
    """Predicts logits for exceeding fixed price thresholds.

    Output shape: [batch, n_horizons, n_thresholds, n_directions]
    Values are RAW LOGITS — sigmoid is applied at inference time
    (`run_inference` in backtest.py / live loop), and the loss uses
    BCEWithLogitsLoss for numerical stability.
    """

    def __init__(self, d_model: int, n_horizons: int, n_thresholds: int, n_directions: int):
        super().__init__()
        out_dim = n_horizons * n_thresholds * n_directions
        self.n_horizons   = n_horizons
        self.n_thresholds = n_thresholds
        self.n_directions = n_directions
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, d_model] → logits [batch, n_horizons, n_thresholds, n_directions]
        out = self.fc(x)
        return out.view(-1, self.n_horizons, self.n_thresholds, self.n_directions)
