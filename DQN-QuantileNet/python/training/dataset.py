import numpy as np
import torch
from torch.utils.data import Dataset


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
