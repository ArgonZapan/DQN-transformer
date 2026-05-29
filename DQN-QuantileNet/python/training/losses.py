import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def pinball_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantiles: torch.Tensor,
    horizon_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Quantile (pinball) loss averaged over all quantiles.

    pred:      [batch, n_horizons, n_quantiles, n_directions]
    target:    [batch, n_horizons, 1, n_directions]  — broadcast over quantiles
    quantiles: [n_quantiles] — tau values in (0, 1)
    horizon_weights: optional [n_horizons] tensor; defaults to uniform.
    """
    diff = target - pred
    q    = quantiles.view(1, 1, -1, 1)
    loss = torch.where(diff >= 0, q * diff, (q - 1) * diff)
    # loss: [batch, n_horizons, n_quantiles, n_directions]

    if horizon_weights is not None:
        w = horizon_weights.view(1, -1, 1, 1)
        loss = loss * w
        # Mean still divides by full count to keep scale stable across configs.
    return loss.mean()


def noncrossing_penalty(pred_q: torch.Tensor) -> torch.Tensor:
    """Soft penalty on quantile crossings.

    pred_q: [batch, n_horizons, n_quantiles, n_directions], with quantiles
    sorted ascending. Returns mean of relu(q_k - q_{k+1}) — non-zero only
    when the model violates monotonicity.
    """
    if pred_q.shape[2] < 2:
        return torch.zeros((), device=pred_q.device, dtype=pred_q.dtype)
    diff = pred_q[:, :, :-1, :] - pred_q[:, :, 1:, :]
    return F.relu(diff).mean()


def threshold_bce_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """BCE-with-logits for threshold exceedance prediction.

    pred:   [batch, n_horizons, n_thresholds, n_directions]  — RAW LOGITS
    target: [batch, n_horizons, n_thresholds, n_directions]  — binary labels
    pos_weight: optional [n_horizons, n_thresholds, n_directions] for class imbalance.
    label_smoothing: epsilon for soft labels (0 → no smoothing).
    """
    if label_smoothing > 0:
        target = target * (1.0 - 2.0 * label_smoothing) + label_smoothing

    if pos_weight is not None:
        # F.binary_cross_entropy_with_logits broadcasts pos_weight over batch.
        return F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)
    return F.binary_cross_entropy_with_logits(pred, target)


class LossModule(nn.Module):
    """Multi-task loss with optional uncertainty weighting and regularizers.

    Owns:
      - quantiles buffer (tau values)
      - per-horizon weights (optional)
      - pos_weight buffer (filled from training data)
      - log_var parameters for uncertainty weighting (Kendall et al. 2018)

    Behaviour switches via config (`[loss]` section):
      - uncertainty_weighting: learned task weights vs. fixed pinball_scale
      - pinball_scale:         multiplicative factor on quantile loss (used when
                               uncertainty_weighting=False) to bring its scale
                               (~0.001 of % returns) up to BCE's (~0.6 nats).
      - noncrossing_weight:    soft penalty on q_k > q_{k+1} crossings.
      - bce_label_smoothing:   smooth BCE targets toward 0.5.
      - bce_use_pos_weight:    enable pos_weight in BCE.
    """

    def __init__(self, config: dict, n_horizons: int, n_thresholds: int, n_directions: int):
        super().__init__()
        loss_cfg = config.get('loss', {})
        pred_cfg = config['prediction']

        self.uncertainty_weighting = bool(loss_cfg.get('uncertainty_weighting', False))
        self.pinball_scale         = float(loss_cfg.get('pinball_scale', 100.0))
        self.noncrossing_weight    = float(loss_cfg.get('noncrossing_weight', 0.1))
        self.bce_label_smoothing   = float(loss_cfg.get('bce_label_smoothing', 0.05))
        self.bce_use_pos_weight    = bool(loss_cfg.get('bce_use_pos_weight', True))

        # Uncertainty-weighting parameters (only registered when used, but always
        # created so state_dict is shape-stable across config toggles).
        self.log_var_q = nn.Parameter(torch.zeros(()), requires_grad=self.uncertainty_weighting)
        self.log_var_t = nn.Parameter(torch.zeros(()), requires_grad=self.uncertainty_weighting)

        quantiles = torch.tensor(pred_cfg.get('quantiles', []), dtype=torch.float32)
        self.register_buffer('quantiles', quantiles)

        # pos_weight is populated lazily from train data via `set_pos_weight`.
        self.register_buffer(
            'pos_weight',
            torch.ones(n_horizons, n_thresholds, n_directions, dtype=torch.float32),
        )
        self._pos_weight_set = False

        # Manual loss-mix weights (fallback when uncertainty_weighting=False).
        self.q_weight = float(pred_cfg.get('quantile_loss_weight',  0.5))
        self.t_weight = float(pred_cfg.get('threshold_loss_weight', 0.5))

    @torch.no_grad()
    def set_pos_weight(self, pos_freq: torch.Tensor, eps: float = 1e-3, cap: float = 50.0):
        """Set pos_weight from per-cell positive-class frequency.

        pos_freq: [n_horizons, n_thresholds, n_directions], in [0, 1].
        Uses pos_weight = (1 - p) / p, clipped to [1/cap, cap] to avoid blow-up
        on cells with no/all positives.
        """
        p = pos_freq.clamp(min=eps, max=1.0 - eps)
        w = (1.0 - p) / p
        w = w.clamp(min=1.0 / cap, max=cap)
        self.pos_weight.copy_(w.to(self.pos_weight.device, self.pos_weight.dtype))
        self._pos_weight_set = True

    def forward(
        self,
        outputs: dict,
        quantile_labels:  torch.Tensor | None,
        threshold_labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict]:
        # NOTE: components values are kept as detached GPU tensors here; the
        # caller is expected to defer .item() until epoch end. Per-batch
        # .item() forces cudaStreamSynchronize and stalls the pipeline.
        components: dict[str, torch.Tensor] = {}
        device = next(iter(outputs.values())).device
        total = torch.zeros((), device=device)

        q_loss_val = None
        t_loss_val = None

        if 'quantile' in outputs and quantile_labels is not None:
            pred_q = outputs['quantile']
            q_loss = pinball_loss(pred_q, quantile_labels, self.quantiles)
            components['loss/quantile_raw'] = q_loss.detach()

            if self.noncrossing_weight > 0:
                nc = noncrossing_penalty(pred_q)
                components['loss/noncrossing'] = nc.detach()
                q_loss = q_loss + self.noncrossing_weight * nc

            q_loss_val = q_loss

        if 'threshold' in outputs and threshold_labels is not None:
            pos_w = self.pos_weight if self.bce_use_pos_weight and self._pos_weight_set else None
            t_loss = threshold_bce_loss(
                outputs['threshold'], threshold_labels,
                pos_weight=pos_w, label_smoothing=self.bce_label_smoothing,
            )
            components['loss/threshold_raw'] = t_loss.detach()
            t_loss_val = t_loss

        # ── Combine ──────────────────────────────────────────────────────────
        if self.uncertainty_weighting:
            # Kendall et al.: total = sum_i  exp(-s_i) * L_i + s_i, where s = log(sigma^2).
            # Halving (the 1/2 in regression) is folded into pinball/BCE conventions.
            if q_loss_val is not None:
                total = total + torch.exp(-self.log_var_q) * q_loss_val + self.log_var_q
                components['loss/log_var_q'] = self.log_var_q.detach()
            if t_loss_val is not None:
                total = total + torch.exp(-self.log_var_t) * t_loss_val + self.log_var_t
                components['loss/log_var_t'] = self.log_var_t.detach()
        else:
            if q_loss_val is not None:
                total = total + self.q_weight * self.pinball_scale * q_loss_val
            if t_loss_val is not None:
                total = total + self.t_weight * t_loss_val

        components['loss/total'] = total.detach()
        return total, components


# ── Helpers ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_pos_freq(loader, device: torch.device, max_batches: int | None = None) -> torch.Tensor | None:
    """Estimate positive-class frequency for the threshold head over a DataLoader.

    Returns tensor [n_horizons, n_thresholds, n_directions] or None if labels
    are absent. Uses up to `max_batches` batches (None = all).
    """
    sum_pos = None
    count   = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        t = batch.get('threshold_label')
        if t is None:
            return None
        t = t.to(device, non_blocking=True)
        if sum_pos is None:
            sum_pos = torch.zeros_like(t.sum(dim=0))
        sum_pos = sum_pos + t.sum(dim=0)
        count  += t.shape[0]

    if sum_pos is None or count == 0:
        return None
    return sum_pos / float(count)
