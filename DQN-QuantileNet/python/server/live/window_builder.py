"""Fetch raw klines and build inference windows (current + historical anchors).

A window is a per-timeframe `(seq_len, n_features)` tensor anchored at a given
close_time. `cutoff_ms` lets us reconstruct the exact market state at any past
moment to verify model behaviour against actual price action.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from . import binance_client
from .csv_appender import append_new_candles

logger = logging.getLogger(__name__)

# Coverage of historical anchors. Anchors are placed every `min(horizons)` hours
# so the frontend can stride them per selected horizon.
HIST_ANCHOR_DAYS_BACK = 60

# Extra candles beyond seq_len + norm_window per TF to absorb warmup.
WARMUP_BUFFER_CANDLES = 100


def _compute_fetch_limit(config: dict, tf: str, days_back: int) -> int:
    from ...data.loader import TF_MINUTES, TF_CFG_KEY
    seq_len  = int(config['timeframes'].get(TF_CFG_KEY[tf], 0))
    norm_win = int(config['features']['normalization_window'])
    candles_per_day = (24 * 60) // TF_MINUTES[tf] or 1
    return candles_per_day * days_back + seq_len + norm_win + WARMUP_BUFFER_CANDLES


def fetch_all_raw(symbol: str, config: dict,
                  days_back: int = HIST_ANCHOR_DAYS_BACK,
                  update_csv: bool = True) -> dict:
    """Fetch candles for every active TF in one pass; optionally extend CSVs."""
    from ...data.loader import _active_tfs

    raw = {}
    for tf in _active_tfs(config):
        limit = _compute_fetch_limit(config, tf, days_back)
        logger.info('[Window] %s fetching %s (%d candles)…', symbol, tf, limit)
        raw[tf] = binance_client.fetch_klines(symbol, tf, limit=limit)
        if update_csv:
            try:
                append_new_candles(symbol, tf, raw[tf])
            except Exception as e:
                logger.warning('[Window] %s %s: CSV append failed — %s', symbol, tf, e)
    return raw


def build_window_at(symbol: str, raw: dict, config: dict,
                    cutoff_ms: Optional[int] = None) -> Optional[dict]:
    """Build a single inference window. Returns None when there isn't enough data.

    Returning None (instead of `[]`) makes the "not enough data" branch explicit
    for callers — the previous list-based contract conflated empty results with
    real errors.
    """
    from ...data.loader import _active_tfs, _base_tf, _align_tf, TF_CFG_KEY
    from ...data.indicators import compute_features

    active_tfs = _active_tfs(config)
    base_tf    = _base_tf(config)
    norm_win   = config['features']['normalization_window']

    cut: dict      = {}
    features: dict = {}
    for tf in active_tfs:
        np_fields = {k: v for k, v in raw[tf].items() if isinstance(v, np.ndarray)}
        if cutoff_ms is not None:
            mask = np_fields['close_time'] <= cutoff_ms
            cut[tf] = {k: v[mask] for k, v in np_fields.items()}
        else:
            cut[tf] = np_fields
        if len(cut[tf]['close']) == 0:
            return None
        features[tf] = compute_features(
            cut[tf]['open'], cut[tf]['high'], cut[tf]['low'],
            cut[tf]['close'], cut[tf]['volume'], norm_win,
        )

    base_closes      = cut[base_tf]['close']
    base_close_times = cut[base_tf]['close_time']
    n_base           = len(base_closes)
    if n_base == 0:
        return None

    alignments = {tf: _align_tf(base_close_times, cut[tf]['close_time'])
                  for tf in active_tfs if tf != base_tf}
    seq_lens   = {tf: config['timeframes'][TF_CFG_KEY[tf]] for tf in active_tfs}

    i = n_base - 1
    tf_data = {}
    for tf in active_tfs:
        seq_len = seq_lens[tf]
        end     = i + 1 if tf == base_tf else int(alignments[tf][i])
        start   = max(0, end - seq_len)
        chunk   = features[tf][start:end].astype(np.float32)
        if len(chunk) < seq_len:
            # Insufficient feature history — refuse to fabricate a padded window.
            return None
        tf_data[tf] = chunk

    n_horizons = len(config['prediction']['horizons_hours'])
    return {
        'symbol':             symbol,
        'tf_data':            tf_data,
        'future_closes':      np.zeros(n_horizons, dtype=np.float32),
        'future_close_times': np.zeros(n_horizons, dtype=np.int64),
        'future_highs':       np.zeros(n_horizons, dtype=np.float32),
        'future_lows':        np.zeros(n_horizons, dtype=np.float32),
        'current_close':      float(base_closes[i]),
        'current_close_time': int(base_close_times[i]),
    }


def build_all_windows_vectorized(symbol: str, raw: dict, config: dict) -> list[dict]:
    """Build a window for EVERY base-TF candle, vectorised.

    Computes per-TF features once on the full array, then slices per candle.
    Replaces N calls to build_window_at (each recomputing features) — the loop
    is now O(N) instead of O(N²). Used by /api/live/predict-range startup.
    """
    from ...data.loader import _active_tfs, _base_tf, _align_tf, TF_CFG_KEY
    from ...data.indicators import compute_features

    active_tfs = _active_tfs(config)
    base_tf    = _base_tf(config)
    norm_win   = config['features']['normalization_window']
    seq_lens   = {tf: config['timeframes'][TF_CFG_KEY[tf]] for tf in active_tfs}
    n_horizons = len(config['prediction']['horizons_hours'])

    # Compute features ONCE for the full per-TF arrays.
    features = {}
    for tf in active_tfs:
        d = raw[tf]
        features[tf] = compute_features(d['open'], d['high'], d['low'],
                                         d['close'], d['volume'], norm_win)

    base_closes      = raw[base_tf]['close']
    base_close_times = raw[base_tf]['close_time']
    n_base           = len(base_close_times)
    if n_base == 0:
        return []

    alignments = {tf: _align_tf(base_close_times, raw[tf]['close_time'])
                  for tf in active_tfs if tf != base_tf}

    windows: list[dict] = []
    for i in range(n_base):
        tf_data = {}
        ok = True
        for tf in active_tfs:
            seq_len = seq_lens[tf]
            end     = i + 1 if tf == base_tf else int(alignments[tf][i])
            start   = max(0, end - seq_len)
            chunk   = features[tf][start:end].astype(np.float32)
            if len(chunk) < seq_len:
                ok = False
                break
            tf_data[tf] = chunk
        if not ok:
            continue
        windows.append({
            'symbol':             symbol,
            'tf_data':            tf_data,
            'future_closes':      np.zeros(n_horizons, dtype=np.float32),
            'future_close_times': np.zeros(n_horizons, dtype=np.int64),
            'future_highs':       np.zeros(n_horizons, dtype=np.float32),
            'future_lows':        np.zeros(n_horizons, dtype=np.float32),
            'current_close':      float(base_closes[i]),
            'current_close_time': int(base_close_times[i]),
        })
    return windows


def build_all_windows(symbol: str, raw: dict, config: dict,
                      hist_days_back: int = HIST_ANCHOR_DAYS_BACK
                      ) -> tuple[list[dict], int]:
    """Build [current, hist_-1, hist_-2, …] windows. Returns (windows, anchor_step_h).

    Stops appending historical anchors as soon as one fails to build (insufficient
    history) — caller still receives every successfully-built anchor up to that
    point, instead of losing the whole symbol.
    """
    horizons = config['prediction']['horizons_hours']
    min_h    = min(horizons)
    interval_ms = min_h * 3600 * 1000

    cur = build_window_at(symbol, raw, config)
    if cur is None:
        return [], min_h

    windows = [cur]
    current_time_ms = cur['current_close_time']
    anchor_count    = (hist_days_back * 24) // min_h

    for a in range(1, anchor_count + 1):
        cutoff_ms = current_time_ms - a * interval_ms
        try:
            hist = build_window_at(symbol, raw, config, cutoff_ms=cutoff_ms)
        except Exception as e:
            logger.warning('[Window] %s anchor -%dh build error — %s', symbol, a * min_h, e)
            break
        if hist is None:
            logger.debug('[Window] %s anchor -%dh: not enough data, stopping', symbol, a * min_h)
            break
        windows.append(hist)

    return windows, min_h
