import os
import logging
import numpy as np
from .indicators import compute_features

logger = logging.getLogger(__name__)

TF_MINUTES = {'1m': 1, '15m': 15, '1h': 60, '4h': 240, '1d': 1440, '1w': 10080}
TF_ORDER   = ['1m', '15m', '1h', '4h', '1d', '1w']
TF_CFG_KEY = {
    '1m': 'candles_1m', '15m': 'candles_15m', '1h': 'candles_1h',
    '4h': 'candles_4h', '1d': 'candles_1d',  '1w': 'candles_1w',
}


def _active_tfs(config: dict) -> list[str]:
    return [tf for tf in TF_ORDER if config['timeframes'].get(TF_CFG_KEY[tf], 0) > 0]


def _base_tf(config: dict) -> str:
    return _active_tfs(config)[0]


def _load_csv(symbol: str, tf: str, data_path: str) -> dict[str, np.ndarray]:
    """Load a single CSV file. Returns dict of OHLCV arrays."""
    filename = f'{symbol}_{tf}.csv'
    filepath = os.path.join(data_path, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'Historical data not found: {filepath}')

    data = np.genfromtxt(filepath, delimiter=',', skip_header=1,
                         usecols=(0, 1, 2, 3, 4, 5, 6),
                         dtype=np.float64)
    return {
        'timestamp':  data[:, 0].astype(np.int64),
        'open':       data[:, 1],
        'high':       data[:, 2],
        'low':        data[:, 3],
        'close':      data[:, 4],
        'volume':     data[:, 5],
        'close_time': data[:, 6].astype(np.int64),
    }


def _align_tf(base_close_times: np.ndarray, tf_close_times: np.ndarray) -> np.ndarray:
    """For each base-TF candle, find aligned end-index in the coarser TF."""
    align = np.searchsorted(tf_close_times, base_close_times, side='right')
    return align.astype(np.int32)


def build_windows(symbol: str, config: dict) -> list[dict]:
    """Load CSVs for all active TFs, compute features, and slide windows.

    Returns list of dicts:
        tf_data:       dict tf → np.ndarray [seq_len, 11]
        future_closes: np.ndarray [n_horizons]
        current_close: float

    Behaviour:
      - Windows whose coarser-TF history is shorter than `seq_len` are
        SKIPPED (not zero-padded). Zero-padding silently fabricates feature
        values (RSI=0 → "deeply oversold", normalizedClose=0 → "at MA")
        that the model never sees in production — `window_builder.py`
        already returns None in that case, so this keeps train/serve
        in sync.
      - Optional `[training].window_stride` slides the cursor by N base-TF
        bars instead of 1. Default 1 keeps current behaviour. Setting it
        to `min(horizons_hours)` (in base-TF units) gives non-overlapping
        labels — useful when raw co-bar samples produce too much
        autocorrelation.
      - Windows with `current_close <= 0` are dropped defensively
        (corrupt CSV row would otherwise produce inf/NaN labels).
    """
    data_path   = os.path.join(os.path.dirname(__file__), '..', '..', 'node', 'data', 'historical')
    active_tfs  = _active_tfs(config)
    base_tf     = _base_tf(config)
    norm_window = config['features']['normalization_window']
    stride      = max(1, int(config.get('training', {}).get('window_stride', 1)))

    # Load and compute features for each TF
    raw: dict[str, dict] = {}
    features: dict[str, np.ndarray] = {}
    for tf in active_tfs:
        raw[tf] = _load_csv(symbol, tf, data_path)
        features[tf] = compute_features(
            raw[tf]['open'], raw[tf]['high'], raw[tf]['low'],
            raw[tf]['close'], raw[tf]['volume'], norm_window,
        )
        logger.info(f'[{symbol}] {tf}: {len(raw[tf]["close"])} candles, features computed.')

    base_close_times = raw[base_tf]['close_time']
    base_closes      = raw[base_tf]['close']
    n_base           = len(base_closes)

    # Alignments: for each base-TF index → end-index in coarser TF
    alignments: dict[str, np.ndarray] = {}
    for tf in active_tfs:
        if tf == base_tf:
            continue
        alignments[tf] = _align_tf(base_close_times, raw[tf]['close_time'])

    # Horizon steps in base-TF units
    base_min      = TF_MINUTES[base_tf]
    horizon_steps = [round(h * 60 / base_min) for h in config['prediction']['horizons_hours']]
    max_horizon   = max(horizon_steps)

    # Warmup: largest input window expressed in base-TF candles
    max_seq = max(
        round(config['timeframes'][TF_CFG_KEY[tf]] * TF_MINUTES[tf] / base_min)
        for tf in active_tfs
    )

    # Walk-forward splitting is handled externally — load all available windows.
    end_idx   = n_base - max_horizon
    start_idx = max_seq

    logger.info(f'[{symbol}] Windows: {start_idx} -> {end_idx} '
                f'(base={base_tf}, ~{end_idx - start_idx} samples)')

    windows = []
    skipped_short  = 0
    skipped_badpx  = 0
    seq_lens = {tf: config['timeframes'][TF_CFG_KEY[tf]] for tf in active_tfs}

    base_highs = raw[base_tf]['high']
    base_lows  = raw[base_tf]['low']

    for i in range(start_idx, end_idx, stride):
        cur_close = float(base_closes[i])
        if not np.isfinite(cur_close) or cur_close <= 0:
            skipped_badpx += 1
            continue

        # Future closes (endpoint returns)
        future_closes = np.array([
            base_closes[i + s] if i + s < n_base else np.nan
            for s in horizon_steps
        ], dtype=np.float32)
        future_close_times = np.array([
            base_close_times[i + s] if i + s < n_base else -1
            for s in horizon_steps
        ], dtype=np.int64)
        if np.any(np.isnan(future_closes)):
            continue

        # Max high and min low over each horizon window (for MFE/MAE labels)
        future_highs = np.array([
            np.max(base_highs[i + 1 : i + s + 1]) if i + s < n_base else np.nan
            for s in horizon_steps
        ], dtype=np.float32)
        future_lows = np.array([
            np.min(base_lows[i + 1 : i + s + 1]) if i + s < n_base else np.nan
            for s in horizon_steps
        ], dtype=np.float32)

        # Build multi-TF window — skip if any TF lacks full history (no zero pad).
        tf_data = {}
        short = False
        for tf in active_tfs:
            seq_len = seq_lens[tf]
            if tf == base_tf:
                end = i + 1
            else:
                end = int(alignments[tf][i])
            start = max(0, end - seq_len)
            chunk = features[tf][start:end]
            if len(chunk) < seq_len:
                short = True
                break
            tf_data[tf] = chunk.astype(np.float32)
        if short:
            skipped_short += 1
            continue

        windows.append({
            'symbol':             symbol,
            'tf_data':            tf_data,
            'future_closes':      future_closes,
            'future_close_times': future_close_times,
            'future_highs':       future_highs,
            'future_lows':        future_lows,
            'current_close':      cur_close,
            'current_close_time': int(base_close_times[i]),
        })

    logger.info(
        f'[{symbol}] Built {len(windows)} windows '
        f'(stride={stride}; skipped {skipped_short} short, {skipped_badpx} bad-price).'
    )
    return windows
