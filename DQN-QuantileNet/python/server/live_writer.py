import os
import json
import logging
import threading
import time
from datetime import datetime, timezone

import numpy as np
import torch

logger = logging.getLogger(__name__)

UTC = timezone.utc

# Path relative to this file: python/server/ -> python/ -> DQN-QuantileNet/
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LIVE_JSON_PATH = os.path.join(_PROJECT_ROOT, 'python', 'live_predictions.json')


def _build_live_windows(symbol: str, config: dict) -> list[dict]:
    """Return the single most-recent window for a symbol (no future closes needed)."""
    from ..data.loader import (
        _active_tfs, _base_tf, _load_csv, _align_tf,
        TF_MINUTES, TF_CFG_KEY,
    )
    from ..data.indicators import compute_features

    data_path  = os.path.join(_PROJECT_ROOT, 'node', 'data', 'historical')
    active_tfs = _active_tfs(config)
    base_tf    = _base_tf(config)
    norm_win   = config['features']['normalization_window']

    raw:      dict = {}
    features: dict = {}
    for tf in active_tfs:
        raw[tf]      = _load_csv(symbol, tf, data_path)
        features[tf] = compute_features(
            raw[tf]['open'], raw[tf]['high'], raw[tf]['low'],
            raw[tf]['close'], raw[tf]['volume'], norm_win,
        )

    base_closes      = raw[base_tf]['close']
    base_close_times = raw[base_tf]['close_time']
    n_base           = len(base_closes)

    alignments: dict = {}
    for tf in active_tfs:
        if tf != base_tf:
            alignments[tf] = _align_tf(base_close_times, raw[tf]['close_time'])

    seq_lens = {tf: config['timeframes'][TF_CFG_KEY[tf]] for tf in active_tfs}

    # Use the very last candle as the current position
    i = n_base - 1
    tf_data = {}
    for tf in active_tfs:
        seq_len = seq_lens[tf]
        end     = i + 1 if tf == base_tf else int(alignments[tf][i])
        start   = max(0, end - seq_len)
        chunk   = features[tf][start:end].astype(np.float32)
        if len(chunk) < seq_len:
            pad   = np.zeros((seq_len - len(chunk), chunk.shape[1] if chunk.ndim > 1 else 11), dtype=np.float32)
            chunk = np.vstack([pad, chunk])
        tf_data[tf] = chunk

    return [{
        'symbol':             symbol,
        'tf_data':            tf_data,
        'future_closes':      np.zeros(len(config['prediction']['horizons_hours']), dtype=np.float32),
        'future_close_times': np.zeros(len(config['prediction']['horizons_hours']), dtype=np.int64),
        'current_close':      float(base_closes[i]),
        'current_close_time': int(base_close_times[i]),
    }]


def write_live_predictions(model, config: dict, checkpoint_path: str, device: torch.device):
    """Run inference on the latest window for each symbol and write live_predictions.json."""
    from ..backtest import run_inference

    symbols    = [a['symbol'] for a in config['actors']]
    pred_cfg   = config['prediction']
    horizons   = pred_cfg['horizons_hours']
    quantiles  = pred_cfg.get('quantiles', [])
    thresholds = pred_cfg.get('thresholds_pct', [])

    now = datetime.now(UTC)
    out = {
        'generated_at':      now.isoformat(),
        'model_checkpoint':  os.path.basename(checkpoint_path),
        'predictions':       [],
    }

    model.eval()
    with torch.no_grad():
        for symbol in symbols:
            try:
                windows = _build_live_windows(symbol, config)
            except Exception as e:
                logger.warning(f'[LiveWriter] {symbol}: failed to build window — {e}')
                continue

            results = run_inference(model, windows, config, device)

            pred_q = results.get('pred_quantile')   # [1, H, Q, D]
            pred_t = results.get('pred_threshold')  # [1, H, T, D]

            current_price = float(windows[0]['current_close'])
            current_time  = int(windows[0]['current_close_time'])

            horizons_out = []
            for hi, h in enumerate(horizons):
                target_ts  = current_time / 1000 + h * 3600
                target_iso = datetime.fromtimestamp(target_ts, tz=UTC).isoformat()

                entry: dict = {
                    'horizon_h':   h,
                    'target_time': target_iso,
                    'long':  {},
                    'short': {},
                }

                if pred_q is not None:
                    for di, dir_key in enumerate(['long', 'short']):
                        entry[dir_key]['quantiles'] = {
                            f'p{int(q * 100)}': round(float(pred_q[0, hi, qi, di]) * 100, 3)
                            for qi, q in enumerate(quantiles)
                        }

                if pred_t is not None:
                    for di, dir_key in enumerate(['long', 'short']):
                        entry[dir_key]['threshold_probs'] = {
                            f'{t}pct': round(float(pred_t[0, hi, ti, di]), 4)
                            for ti, t in enumerate(thresholds)
                        }

                horizons_out.append(entry)

            out['predictions'].append({
                'symbol':        symbol,
                'current_price': current_price,
                'current_time':  datetime.fromtimestamp(current_time / 1000, tz=UTC).isoformat(),
                'horizons':      horizons_out,
            })
            logger.info(f'[LiveWriter] {symbol}: live prediction written.')

    os.makedirs(os.path.dirname(LIVE_JSON_PATH), exist_ok=True)
    tmp = LIVE_JSON_PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, LIVE_JSON_PATH)   # atomic write
    logger.info(f'[LiveWriter] Written → {LIVE_JSON_PATH}  ({len(out["predictions"])} symbols)')
    return out


class LivePredictionLoop:
    """Background thread that writes live_predictions.json every `interval_sec` seconds.

    Usage:
        loop = LivePredictionLoop(model, config, checkpoint_path, device, interval_sec=900)
        loop.start()
        ...
        loop.stop()
    """

    def __init__(self, model, config: dict, checkpoint_path: str,
                 device: torch.device, interval_sec: int = 900):
        self.model           = model
        self.config          = config
        self.checkpoint_path = checkpoint_path
        self.device          = device
        self.interval_sec    = interval_sec
        self._stop           = threading.Event()
        self._thread         = None

    def start(self):
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name='live-predictor'
        )
        self._thread.start()
        logger.info(f'[LiveWriter] Background loop started (interval={self.interval_sec}s)')

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def trigger(self):
        """Force an immediate write (e.g. after a new checkpoint is saved)."""
        try:
            write_live_predictions(self.model, self.config, self.checkpoint_path, self.device)
        except Exception as e:
            logger.error(f'[LiveWriter] trigger() failed: {e}')

    def _loop(self):
        while not self._stop.is_set():
            try:
                write_live_predictions(self.model, self.config, self.checkpoint_path, self.device)
            except Exception as e:
                logger.error(f'[LiveWriter] loop error: {e}')
            self._stop.wait(self.interval_sec)
