import logging

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def compute_volatility_weights(windows: list, n_bins: int) -> tuple:
    """Per-window sampling weights based on future realised range.

    Vol proxy = (future_high_max - future_low_min) / current_close at the
    longest horizon — i.e. the total price excursion the model is asked to
    predict for that window. Windows are bucketed into `n_bins` equal-frequency
    quantile bins; each window's weight is 1/(n_bins * count_in_its_bin) so
    that, in expectation, every bin contributes the same number of samples
    per epoch. Effect: extreme-vol regimes (which dominate tail calibration)
    are upsampled, calm chop is downsampled.

    Returns (weights[N] float32, counts[n_bins] int).
    """
    cc      = np.array([w['current_close']      for w in windows], dtype=np.float64)
    fh_last = np.array([w['future_highs'][-1]   for w in windows], dtype=np.float64)
    fl_last = np.array([w['future_lows'][-1]    for w in windows], dtype=np.float64)

    # Symmetric realised range over the longest horizon.
    vol = (fh_last - fl_last) / np.maximum(cc, 1e-12)

    n_bins = max(1, int(n_bins))
    if n_bins == 1 or len(vol) < n_bins:
        weights = np.full(len(vol), 1.0 / max(len(vol), 1), dtype=np.float32)
        return weights, np.array([len(vol)], dtype=np.int64)

    # Equal-frequency bins via interior quantile edges.
    qs    = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.quantile(vol, qs)
    bin_idx = np.searchsorted(edges, vol, side='right')      # values in [0, n_bins-1]
    counts  = np.bincount(bin_idx, minlength=n_bins).clip(min=1).astype(np.float64)
    weights = (1.0 / (n_bins * counts[bin_idx])).astype(np.float32)
    return weights, counts.astype(np.int64)


class QuantileDataset(Dataset):
    """Dataset built from pre-loaded windows.

    Each window: {
        tf_data:       dict tf → np.ndarray [seq_len, 11]
        future_closes: np.ndarray [n_horizons]
        current_close: float
    }
    """

    def __init__(self, windows: list, config: dict, active_tfs: list, tf_seq_lens: dict,
                 train_augment: bool = False):
        self.windows     = windows
        self.active_tfs  = active_tfs
        self.tf_seq_lens = tf_seq_lens
        pred_cfg         = config['prediction']
        aug_cfg          = config.get('augmentation', {})
        sampling_cfg     = config.get('sampling', {})

        self.mode       = pred_cfg['mode']
        self.direction  = pred_cfg['direction']
        self.n_horizons = len(pred_cfg['horizons_hours'])
        self.quantiles  = pred_cfg.get('quantiles', [])
        self.thresholds = np.array(pred_cfg.get('thresholds_pct', []), dtype=np.float32) / 100.0

        # Feature-space Gaussian noise (only applied when train_augment=True).
        # Acts as a tiny regularizer; price-space jitter would require redoing
        # feature engineering at __getitem__ time, so we perturb the already-
        # normalized features instead.
        self.train_augment    = bool(train_augment)
        self.feature_noise_std = float(aug_cfg.get('feature_noise_std', 0.0))

        # Volatility-stratified sampling weights (training only — eval loaders
        # ignore these by passing shuffle=False without a sampler).
        self.sample_weights: np.ndarray | None = None
        if train_augment and bool(sampling_cfg.get('volatility_stratified', False)):
            n_bins = int(sampling_cfg.get('n_bins', 5))
            self.sample_weights, counts = compute_volatility_weights(windows, n_bins)
            logger.info(
                f'[QuantileDataset] Vol-stratified sampling: n_bins={n_bins} '
                f'counts={counts.tolist()}'
            )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int):
        w             = self.windows[idx]
        current_close = w['current_close']
        future_highs  = w['future_highs']   # [n_horizons]
        future_lows   = w['future_lows']    # [n_horizons]

        # Max favorable excursion upward and downward over each horizon window.
        # Both are always positive: mfe_up = how far price rose, mfe_down = how far it fell.
        mfe_up   = (future_highs - current_close) / current_close  # [n_horizons]
        mfe_down = (current_close - future_lows)  / current_close  # [n_horizons]

        if self.train_augment and self.feature_noise_std > 0:
            std = self.feature_noise_std
            tf_tensors = {}
            for tf in self.active_tfs:
                arr = w['tf_data'][tf]
                noise = np.random.normal(0.0, std, size=arr.shape).astype(arr.dtype)
                tf_tensors[tf] = torch.from_numpy(arr + noise)
            result = {'tf': tf_tensors}
        else:
            result = {
                'tf': {tf: torch.from_numpy(w['tf_data'][tf]) for tf in self.active_tfs}
            }

        if self.mode in ('quantile', 'both'):
            # dir 0 = up excursion (long TP / short SL)
            # dir 1 = down excursion (short TP / long SL)
            q_label = torch.tensor(
                np.stack([mfe_up, mfe_down], axis=-1), dtype=torch.float32
            ).unsqueeze(1)  # [n_horizons, 1, 2]
            result['quantile_label'] = q_label

        if self.mode in ('threshold', 'both'):
            thr_t      = torch.tensor(self.thresholds, dtype=torch.float32)  # [n_thresholds]
            up_t       = torch.tensor(mfe_up,   dtype=torch.float32)          # [n_horizons]
            down_t     = torch.tensor(mfe_down, dtype=torch.float32)          # [n_horizons]
            up_labels   = (up_t.unsqueeze(-1)   >= thr_t).float()  # [n_horizons, n_thr]
            down_labels = (down_t.unsqueeze(-1) >= thr_t).float()  # [n_horizons, n_thr]
            result['threshold_label'] = torch.stack([up_labels, down_labels], dim=-1)

        return result


class GPUWindowDataset:
    """In-VRAM dataset: stacks all windows once into GPU tensors and yields
    batches by index — no host↔device copies, no DataLoader workers.

    Duck-typed as a DataLoader for the training loop (supports `len()` and
    iteration yielding the same dict layout as `collate_fn`). Labels
    (mfe_up/mfe_down → quantile/threshold targets) are precomputed once on
    GPU since they're deterministic; feature noise is applied per-iteration.
    """

    def __init__(self, windows: list, config: dict, active_tfs: list, tf_seq_lens: dict,
                 device: torch.device, batch_size: int, shuffle: bool,
                 train_augment: bool = False):
        self.device      = device
        self.batch_size  = int(batch_size)
        self.shuffle     = bool(shuffle)
        self.active_tfs  = active_tfs
        self.n           = len(windows)

        pred_cfg     = config['prediction']
        aug_cfg      = config.get('augmentation', {})
        sampling_cfg = config.get('sampling', {})
        self.mode              = pred_cfg['mode']
        self.train_augment     = bool(train_augment)
        self.feature_noise_std = float(aug_cfg.get('feature_noise_std', 0.0))

        # Volatility-stratified sampling — only meaningful when shuffling
        # (i.e. training pass). Eval loaders pass shuffle=False, which forces
        # a deterministic full sweep and skips the multinomial draw below.
        self._strat_enabled = (
            train_augment and shuffle
            and bool(sampling_cfg.get('volatility_stratified', False))
        )
        if self._strat_enabled:
            n_bins = int(sampling_cfg.get('n_bins', 5))
            w_np, counts = compute_volatility_weights(windows, n_bins)
            self.sample_weights = torch.from_numpy(w_np).to(device)
            logger.info(
                f'[GPUWindowDataset] Vol-stratified sampling: n_bins={n_bins} '
                f'counts={counts.tolist()}'
            )
        else:
            self.sample_weights = None

        # Per-TF features stacked into a single GPU tensor [N, seq_len, F].
        self.tf_data = {
            tf: torch.from_numpy(
                np.stack([w['tf_data'][tf] for w in windows], axis=0)
            ).to(device, non_blocking=True)
            for tf in active_tfs
        }

        cc = torch.tensor(
            [w['current_close'] for w in windows], dtype=torch.float32, device=device
        ).unsqueeze(1)  # [N, 1]
        fh = torch.from_numpy(
            np.stack([w['future_highs'] for w in windows], axis=0)
        ).to(device, dtype=torch.float32)
        fl = torch.from_numpy(
            np.stack([w['future_lows'] for w in windows], axis=0)
        ).to(device, dtype=torch.float32)
        mfe_up   = (fh - cc) / cc   # [N, n_horizons]
        mfe_down = (cc - fl) / cc

        if self.mode in ('quantile', 'both'):
            # [N, n_horizons, 1, 2] — matches per-sample shape from QuantileDataset.
            self.quantile_label = torch.stack([mfe_up, mfe_down], dim=-1).unsqueeze(2).contiguous()
        else:
            self.quantile_label = None

        if self.mode in ('threshold', 'both'):
            thr = torch.tensor(
                pred_cfg.get('thresholds_pct', []), dtype=torch.float32, device=device
            ) / 100.0
            up_lab   = (mfe_up.unsqueeze(-1)   >= thr).float()  # [N, H, n_thr]
            down_lab = (mfe_down.unsqueeze(-1) >= thr).float()
            # [N, H, n_thr, 2]
            self.threshold_label = torch.stack([up_lab, down_lab], dim=-1).contiguous()
        else:
            self.threshold_label = None

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            if self.sample_weights is not None:
                # With replacement: keeps total samples/epoch = N, but every
                # vol bin contributes ~equally instead of being dominated by
                # the modal (low-vol) regime.
                idx = torch.multinomial(self.sample_weights, self.n, replacement=True)
            else:
                idx = torch.randperm(self.n, device=self.device)
        else:
            idx = torch.arange(self.n, device=self.device)

        bs  = self.batch_size
        std = self.feature_noise_std if self.train_augment else 0.0

        for start in range(0, self.n, bs):
            sel = idx[start:start + bs]
            tf_batch = {tf: t.index_select(0, sel) for tf, t in self.tf_data.items()}
            if std > 0:
                tf_batch = {tf: x + torch.randn_like(x) * std for tf, x in tf_batch.items()}
            batch = {'tf': tf_batch}
            if self.quantile_label is not None:
                batch['quantile_label'] = self.quantile_label.index_select(0, sel)
            if self.threshold_label is not None:
                batch['threshold_label'] = self.threshold_label.index_select(0, sel)
            yield batch


def collate_fn(batch: list) -> dict:
    active_tfs = list(batch[0]['tf'].keys())
    out = {
        'tf': {tf: torch.stack([b['tf'][tf] for b in batch]) for tf in active_tfs}
    }
    if 'quantile_label' in batch[0]:
        out['quantile_label'] = torch.stack([b['quantile_label'] for b in batch])
    if 'threshold_label' in batch[0]:
        out['threshold_label'] = torch.stack([b['threshold_label'] for b in batch])
    return out
