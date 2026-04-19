"""
Standalone backtesting script for DQN-OPUS.
Loads a trained model checkpoint and evaluates it on the OOS (last 20%) portion
of historical CSV data. No Node.js or ZMQ required.

Usage:
    python backtest.py [--config ../config.toml] [--model path/to/checkpoint.pt] [--device cpu]
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime, date, timezone

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model.network import TradingDQN
from config import load_config
from utils.metrics import (
    sharpe_ratio, max_drawdown, win_rate,
    profit_factor, avg_win_loss, max_consecutive_losses,
)

ACTION_LONG, ACTION_SHORT, ACTION_HOLD, ACTION_CLOSE = 0, 1, 2, 3


# ── CSV Loading ────────────────────────────────────────────────────────────────

def load_csv(path: str) -> list:
    """Load candles from CSV (12 columns, last is 'ignore'). Parses by index."""
    candles = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) < 7:
                continue
            try:
                candles.append({
                    'timestamp':  int(row[0]),
                    'open':       float(row[1]),
                    'high':       float(row[2]),
                    'low':        float(row[3]),
                    'close':      float(row[4]),
                    'volume':     float(row[5]),
                    'close_time': int(row[6]),
                })
            except (ValueError, IndexError):
                continue
    return candles


# ── Pre-computed indicators (numpy, O(N) once per TF) ────────────────────────

def _rolling_mean_np(arr: np.ndarray, window: int) -> np.ndarray:
    """Partial-window rolling mean matching JS rollingMean."""
    out = np.empty(len(arr), dtype=np.float64)
    cumsum = np.cumsum(arr)
    for i in range(len(arr)):
        w = min(i + 1, window)
        out[i] = (cumsum[i] - (cumsum[i - w] if i >= w else 0.0)) / w
    return out


def _rolling_std_np(arr: np.ndarray, window: int) -> np.ndarray:
    """Population std, sliding window O(N). Matches JS rollingStd (Welford)."""
    n = len(arr)
    out = np.zeros(n, dtype=np.float64)
    sum_x = sum_x2 = 0.0
    for i in range(n):
        sum_x  += arr[i]
        sum_x2 += arr[i] * arr[i]
        if i >= window:
            sum_x  -= arr[i - window]
            sum_x2 -= arr[i - window] * arr[i - window]
        w = min(i + 1, window)
        if w < 2:
            continue
        mean = sum_x / w
        var = sum_x2 / w - mean * mean
        out[i] = var ** 0.5 if var > 0 else 0.0
    return out


def _ema_np(arr: np.ndarray, span: int) -> np.ndarray:
    k = 2.0 / (span + 1)
    out = np.empty(len(arr), dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = (arr[i] - out[i - 1]) * k + out[i - 1]
    return out


def _rsi_np(closes: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(closes)
    out = np.zeros(n, dtype=np.float64)
    if n < period + 1:
        return out
    avg_gain = avg_loss = 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        if d > 0:
            avg_gain += d
        else:
            avg_loss -= d
    avg_gain /= period
    avg_loss /= period
    out[period] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for i in range(period + 1, n):
        d = closes[i] - closes[i - 1]
        g = d if d > 0 else 0.0
        l = -d if d < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
        out[i] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return out


def precompute_features(candles: list, norm_window: int) -> np.ndarray:
    """
    Pre-compute all 8 features for the full candle array once.
    Returns float32 array of shape [N, 8].
    Matches build_features() output exactly.
    """
    n = len(candles)
    closes  = np.array([c['close']  for c in candles], dtype=np.float64)
    opens   = np.array([c['open']   for c in candles], dtype=np.float64)
    highs   = np.array([c['high']   for c in candles], dtype=np.float64)
    lows    = np.array([c['low']    for c in candles], dtype=np.float64)
    volumes = np.array([c['volume'] for c in candles], dtype=np.float64)

    mean_c   = _rolling_mean_np(closes, norm_window)
    std_c    = _rolling_std_np(closes, norm_window)
    rsi      = _rsi_np(closes)
    ema12    = _ema_np(closes, 12)
    ema26    = _ema_np(closes, 26)
    macd     = ema12 - ema26
    sma20    = _rolling_mean_np(closes, 20)
    mean_vol = _rolling_mean_np(volumes, norm_window)

    prev_closes = np.empty(n, dtype=np.float64)
    prev_closes[0] = closes[0]
    prev_closes[1:] = closes[:-1]

    out = np.zeros((n, 8), dtype=np.float32)

    # np.where avoids computing division where denominator is 0 (no RuntimeWarning)
    out[:, 0] = np.where(std_c != 0,        (closes - mean_c) / np.where(std_c != 0,        std_c,    1.0), 0.0)
    out[:, 1] = np.where(closes != 0,       (highs - lows)    / np.where(closes != 0,        closes,   1.0), 0.0)
    out[:, 2] = np.where(closes != 0,       (closes - opens)  / np.where(closes != 0,        closes,   1.0), 0.0)
    out[:, 3] = np.where(mean_vol != 0,     volumes           / np.where(mean_vol != 0,      mean_vol, 1.0), 0.0)
    out[:, 4] = rsi / 100.0
    out[:, 5] = np.where(closes != 0,       macd              / np.where(closes != 0,        closes,   1.0), 0.0)
    out[:, 6] = np.where(prev_closes != 0,  (closes - prev_closes) / np.where(prev_closes != 0, prev_closes, 1.0), 0.0)
    out[:, 7] = (closes > sma20).astype(np.float32)

    return out


# ── Wskaźniki v2 (niezaktywowane) ────────────────────────────────────────────
#
# 8 cech (num_features = 8, bez zmian architektury):
#   0: normalizedClose   — z-score, okno 60 (bez zmian)
#   1: relativeRange     — (H-L)/close (bez zmian)
#   2: candleDirection   — (close-open)/close (bez zmian)
#   3: volumeClipped     — min(vol/meanVol, 3.0) — clip na 3x mean
#   4: stochasticK       — (close-min14)/(max14-min14)
#   5: macdHistNorm      — (macdLine-signalLine)/close
#   6: bollingerWidth    — 4*std20/sma20
#   7: smaDistance       — (close-sma20)/close
#
# Opcjonalne (num_features = 9):
#   8: atrNorm           — ATR(14)/close


def _atr_np(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(closes)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = highs[0] - lows[0]
    prev = closes[:-1]
    tr[1:] = np.maximum(highs[1:] - lows[1:],
              np.maximum(np.abs(highs[1:] - prev), np.abs(lows[1:] - prev)))
    return _ema_np(tr, period)


def _stochastic_k_np(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(closes)
    out = np.full(n, 0.5, dtype=np.float64)
    for i in range(n):
        s = max(0, i - period + 1)
        lo = lows[s:i + 1].min()
        hi = highs[s:i + 1].max()
        rng = hi - lo
        out[i] = (closes[i] - lo) / rng if rng != 0 else 0.5
    return out


def precompute_features_v2(candles: list, norm_window: int) -> np.ndarray:
    """
    Pre-compute all 8 v2 features. Returns float32 array of shape [N, 8].
    Drop-in replacement for precompute_features() — matches precomputeIndicatorsV2() in state.js.
    NIEZAKTYWOWANE — użyj zamiast precompute_features() gdy gotowe do testu.
    """
    n = len(candles)
    closes  = np.array([c['close']  for c in candles], dtype=np.float64)
    opens   = np.array([c['open']   for c in candles], dtype=np.float64)
    highs   = np.array([c['high']   for c in candles], dtype=np.float64)
    lows    = np.array([c['low']    for c in candles], dtype=np.float64)
    volumes = np.array([c['volume'] for c in candles], dtype=np.float64)

    mean_c   = _rolling_mean_np(closes, norm_window)
    std_c    = _rolling_std_np(closes, norm_window)
    ema12    = _ema_np(closes, 12)
    ema26    = _ema_np(closes, 26)
    macd_line = ema12 - ema26
    signal   = _ema_np(macd_line, 9)
    sma20    = _rolling_mean_np(closes, 20)
    std20    = _rolling_std_np(closes, 20)
    mean_vol = _rolling_mean_np(volumes, norm_window)
    stoch_k  = _stochastic_k_np(highs, lows, closes)

    out = np.zeros((n, 8), dtype=np.float32)
    out[:, 0] = np.where(std_c != 0,    (closes - mean_c) / np.where(std_c != 0,    std_c,   1.0), 0.0)
    out[:, 1] = np.where(closes != 0,   (highs - lows)    / np.where(closes != 0,   closes,  1.0), 0.0)
    out[:, 2] = np.where(closes != 0,   (closes - opens)  / np.where(closes != 0,   closes,  1.0), 0.0)
    out[:, 3] = np.where(mean_vol != 0, np.minimum(volumes / np.where(mean_vol != 0, mean_vol, 1.0), 3.0), 0.0)
    out[:, 4] = stoch_k.astype(np.float32)
    out[:, 5] = np.where(closes != 0,   (macd_line - signal) / np.where(closes != 0, closes, 1.0), 0.0)
    out[:, 6] = np.where(sma20 != 0,    (4 * std20) / np.where(sma20 != 0, sma20, 1.0), 0.0)
    out[:, 7] = np.where(closes != 0,   (closes - sma20) / np.where(closes != 0, closes, 1.0), 0.0)

    return out


def build_alignment_map(candles_1m: list, candles_tf: list) -> np.ndarray:
    """
    For each 1m candle index, find how many TF candles have close_time <= that 1m close_time.
    Returns int32 array of shape [N_1m] — endIdx for slicing precomputed TF features.
    Binary search O(N_1m * log(N_tf)).
    """
    n1m = len(candles_1m)
    ntf = len(candles_tf)
    tf_times = np.array([c['close_time'] for c in candles_tf], dtype=np.int64)
    alignment = np.empty(n1m, dtype=np.int32)
    for i in range(n1m):
        t = candles_1m[i]['close_time']
        # searchsorted: find rightmost position where tf_times <= t
        alignment[i] = int(np.searchsorted(tf_times, t, side='right'))
    return alignment


def trim_candles(all_candles: dict, oos_start_1m: int, config: dict) -> tuple:
    """
    Trim all TF candle arrays to OOS + lookback only.
    Lookback = state_window + indicator_warmup (RSI 14, MACD 35, SMA 20, rolling norm_w).
    Returns (trimmed_candles, new_oos_start) where new_oos_start is relative to trimmed 1m.
    """
    norm_w = config['data']['normalization_window']
    indicator_warmup = max(norm_w, 35, 20, 14)  # largest indicator window
    tf_map = {'candles_1m': '1m', 'candles_15m': '15m',
              'candles_1h': '1h', 'candles_1d': '1d', 'candles_1w': '1w'}

    candles_1m = all_candles.get('1m', [])
    state_window_1m = config['timeframes'].get('candles_1m', 0)
    lookback_1m = state_window_1m + indicator_warmup

    trim_start_1m = max(0, oos_start_1m - lookback_1m)
    trim_time_ms = candles_1m[trim_start_1m]['timestamp']

    trimmed = {}
    for cfg_key, tf_name in tf_map.items():
        candles = all_candles.get(tf_name)
        if not candles:
            continue
        if tf_name == '1m':
            trimmed[tf_name] = candles[trim_start_1m:]
        else:
            state_window_tf = config['timeframes'].get(cfg_key, 0)
            lookback_tf = state_window_tf + indicator_warmup
            # binary search: first candle with timestamp >= trim_time_ms
            lo, hi = 0, len(candles)
            while lo < hi:
                mid = (lo + hi) // 2
                if candles[mid]['timestamp'] < trim_time_ms:
                    lo = mid + 1
                else:
                    hi = mid
            trim_idx = max(0, lo - lookback_tf)
            trimmed[tf_name] = candles[trim_idx:]
            kept = len(candles) - trim_idx
            print(f"[Backtest]   Trimmed {tf_name}: {len(candles)} → {kept} candles")

    new_oos_start = oos_start_1m - trim_start_1m
    kept_1m = len(candles_1m) - trim_start_1m
    print(f"[Backtest]   Trimmed 1m: {len(candles_1m)} → {kept_1m} candles (OOS starts at idx {new_oos_start})")
    return trimmed, new_oos_start


def precompute_all(all_candles: dict, config: dict) -> dict:
    """Pre-compute features and alignment maps for all TFs. Called once per symbol."""
    norm_w = config['data']['normalization_window']
    tf_map = {'candles_1m': '1m', 'candles_15m': '15m',
              'candles_1h': '1h', 'candles_1d': '1d', 'candles_1w': '1w'}

    candles_1m = all_candles.get('1m', [])
    result = {'features': {}, 'alignment': {}}

    for cfg_key, tf_name in tf_map.items():
        if config['timeframes'].get(cfg_key, 0) <= 0:
            continue
        candles = all_candles.get(tf_name, [])
        if not candles:
            continue
        print(f"[Backtest] Pre-computing features for {tf_name} ({len(candles)} candles)...")
        result['features'][tf_name] = precompute_features(candles, norm_w)
        if tf_name != '1m':
            result['alignment'][tf_name] = build_alignment_map(candles_1m, candles)

    return result


def build_state_fast(precomp: dict, idx_1m: int, config: dict) -> dict:
    """
    Build state from pre-computed arrays. O(num_candles) slice instead of O(N) rescan.
    idx_1m = current 1m candle index in the FULL array (not OOS-relative).
    """
    tf_map = {'candles_1m': '1m', 'candles_15m': '15m',
              'candles_1h': '1h', 'candles_1d': '1d', 'candles_1w': '1w'}
    state = {}
    for cfg_key, tf_name in tf_map.items():
        num = config['timeframes'].get(cfg_key, 0)
        if num <= 0:
            continue
        feats = precomp['features'].get(tf_name)
        if feats is None:
            continue

        if tf_name == '1m':
            end_idx = idx_1m + 1
        else:
            end_idx = int(precomp['alignment'][tf_name][idx_1m])

        start_idx = max(0, end_idx - num)
        actual = feats[start_idx:end_idx]  # shape [K, 8]

        pad = num - len(actual)
        if pad > 0:
            actual = np.concatenate([np.zeros((pad, 8), dtype=np.float32), actual], axis=0)

        state[tf_name] = actual
    return state


# ── Model ──────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, config: dict, device: str) -> TradingDQN:
    """Load TradingDQN from checkpoint without instantiating Trainer."""
    model = TradingDQN(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"[Backtest] Loaded: step={ckpt.get('step', '?')}, epsilon={ckpt.get('epsilon', '?'):.4f}")
    return model


@torch.no_grad()
def predict(model: TradingDQN, state_dict: dict, action_mask: list,
            config: dict, device: str, position_features: list | None = None) -> int:
    """
    Greedy prediction. Tensor order matches Trainer.predict():
    sorted active config keys → ['candles_15m', 'candles_1d', 'candles_1h', 'candles_1m']
    position_features: [is_long, is_short, unrealized_pnl, bars_norm] or None
    """
    tf_keys = sorted(k for k, v in config['timeframes'].items() if v > 0)
    tensors = []
    for key in tf_keys:
        tf_name = key.replace('candles_', '')
        num = config['timeframes'][key]
        arr = state_dict.get(tf_name, np.zeros((num, 8), dtype=np.float32))
        tensors.append(torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(device))

    mask = torch.tensor([action_mask], dtype=torch.float32).to(device)
    pos_tensor = torch.tensor([position_features], dtype=torch.float32).to(device) if position_features is not None else None
    q = model(tensors, action_mask=mask, position_features=pos_tensor)
    return q.argmax(dim=1).item()


# ── Position Tracking ──────────────────────────────────────────────────────────

class Position:
    def __init__(self, side: str, open_price: float, open_time_ms: int):
        self.side = side
        self.open_price = open_price
        self.open_time_ms = open_time_ms
        self.close_price: float | None = None
        self.close_time_ms: int | None = None

    def close(self, price: float, time_ms: int):
        self.close_price = price
        self.close_time_ms = time_ms


def net_pnl(pos: Position, config: dict) -> float:
    """Gross PnL minus exchange fee (open + close). No training penalties."""
    gross = ((pos.close_price - pos.open_price) / pos.open_price
             if pos.side == 'LONG'
             else (pos.open_price - pos.close_price) / pos.open_price)
    fee = config['backtesting']['fee']
    return gross - fee * 2


def action_mask_for(position) -> list:
    return [1, 1, 1, 0] if position is None else [0, 0, 1, 1]


# ── Per-Symbol Backtest ────────────────────────────────────────────────────────

# Module-level cache: symbol → {precomp, oos_start, candles_1m}
# Keyed by (symbol, data_path, validation_days, max_oos_days) to survive config changes.
_DATA_CACHE: dict = {}


def _cache_key(symbol: str, config: dict) -> tuple:
    vw = config['training'].get('validation_weeks', 0)
    vd = vw * 7 if vw else config['training']['validation_days']
    return (symbol, config['data']['path'],
            vd, config['backtesting'].get('max_oos_days', 60))


def preload_backtest_data(symbols: list, config: dict) -> None:
    """Load, trim, and precompute backtest data for all symbols once.
    Call this once at startup; run_symbol_backtest will use the cache."""
    for symbol in symbols:
        key = _cache_key(symbol, config)
        if key in _DATA_CACHE:
            continue
        all_candles = load_all_candles(symbol, config)
        candles_1m = all_candles.get('1m', [])
        if not candles_1m:
            _DATA_CACHE[key] = None
            continue
        validation_weeks = config['training'].get('validation_weeks', 0)
        validation_days = validation_weeks * 7 if validation_weeks else config['training']['validation_days']
        split = max(0, len(candles_1m) - validation_days * 24 * 60)
        max_oos_candles = config['backtesting'].get('max_oos_days', 60) * 24 * 60
        oos_start = max(split, len(candles_1m) - max_oos_candles)
        trimmed, oos_start = trim_candles(all_candles, oos_start, config)
        precomp = precompute_all(trimmed, config)
        _DATA_CACHE[key] = {
            'candles_1m': trimmed['1m'],
            'oos_start': oos_start,
            'precomp': precomp,
        }
        print(f"[Backtest] Preloaded {symbol}: {len(trimmed['1m'])} candles cached")


def load_all_candles(symbol: str, config: dict) -> dict:
    data_path = config['data']['path']
    tf_map = {
        'candles_1m': '1m', 'candles_15m': '15m',
        'candles_1h': '1h', 'candles_1d': '1d', 'candles_1w': '1w',
    }
    result = {}
    for cfg_key, tf_name in tf_map.items():
        if config['timeframes'].get(cfg_key, 0) <= 0:
            continue
        fpath = os.path.join(data_path, f'{symbol}_{tf_name}.csv')
        if not os.path.exists(fpath):
            print(f"[Backtest] WARNING: {fpath} not found — using empty")
            result[tf_name] = []
        else:
            candles = load_csv(fpath)
            result[tf_name] = candles
            print(f"[Backtest]   {symbol} {tf_name}: {len(candles)} candles")
    return result


def _fmt_time(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')


def run_symbol_backtest(symbol: str, config: dict, model: TradingDQN, device: str,
                        last_n: int | None = None) -> dict:
    """Sequential OOS backtest for one symbol. Uses _DATA_CACHE when available."""
    key = _cache_key(symbol, config)
    cached = _DATA_CACHE.get(key)

    if cached is not None:
        candles_1m = cached['candles_1m']
        oos_start  = cached['oos_start']
        precomp    = cached['precomp']
        print(f"[Backtest] {symbol}: using cached data ({len(candles_1m)} candles, OOS from {oos_start})")
    else:
        # Cache miss — load, trim, precompute (and populate cache for next call)
        all_candles = load_all_candles(symbol, config)
        candles_1m = all_candles.get('1m', [])

        if not candles_1m:
            print(f"[Backtest] {symbol}: no 1m data — skipping")
            return {'trades': [], 'equity_curve': [0.0], 'daily_pnl': {}, 'total_steps': 0}

        validation_weeks = config['training'].get('validation_weeks', 0)
        validation_days = validation_weeks * 7 if validation_weeks else config['training']['validation_days']
        split = max(0, len(candles_1m) - validation_days * 24 * 60)
        max_oos_candles = config['backtesting'].get('max_oos_days', 60) * 24 * 60
        oos_start = max(split, len(candles_1m) - max_oos_candles)

        all_candles, oos_start = trim_candles(all_candles, oos_start, config)
        candles_1m = all_candles['1m']
        precomp = precompute_all(all_candles, config)

        _DATA_CACHE[key] = {'candles_1m': candles_1m, 'oos_start': oos_start, 'precomp': precomp}

    split_total = len(candles_1m)
    if last_n is not None and last_n < (split_total - oos_start):
        oos_start = split_total - last_n
        print(f"[Backtest] {symbol}: OOS = last {last_n} candles")
    else:
        print(f"[Backtest] {symbol}: OOS = {split_total - oos_start} steps "
              f"(last {(split_total - oos_start) // 1440} days)")

    trades, equity_curve, daily_pnl = [], [0.0], {}
    cumulative_pnl = 0.0
    position = None
    hold_minutes = 0
    bars_in_trade = 0

    step_interval = config['backtesting'].get('step_interval', 1)

    for abs_idx in range(oos_start, len(candles_1m)):
        if (abs_idx - oos_start) % step_interval != 0:
            if position is not None:
                bars_in_trade += 1
            hold_minutes += 1
            continue

        candle = candles_1m[abs_idx]
        t_ms = candle['close_time']
        price = candle['close']

        # Cechy pozycji: [is_long, is_short, unrealized_pnl, bars_norm]
        if position is not None:
            is_long  = 1 if position.side == 'LONG'  else 0
            is_short = 1 if position.side == 'SHORT' else 0
            upnl = ((price - position.open_price) / position.open_price if is_long
                    else (position.open_price - price) / position.open_price)
            pos_feat = [is_long, is_short, upnl, min(bars_in_trade / 200, 1.0)]
        else:
            pos_feat = [0, 0, 0, 0]

        state = build_state_fast(precomp, abs_idx, config)
        mask = action_mask_for(position)
        action = predict(model, state, mask, config, device, position_features=pos_feat)

        if action == ACTION_LONG and position is None:
            if hold_minutes:
                print(f"  [{symbol}] HOLD {hold_minutes} min")
                hold_minutes = 0
            position = Position('LONG', price, t_ms)
            bars_in_trade = 0
            print(f"  [{symbol}] OPEN LONG  @ {price:.4f}  {_fmt_time(t_ms)}")

        elif action == ACTION_SHORT and position is None:
            if hold_minutes:
                print(f"  [{symbol}] HOLD {hold_minutes} min")
                hold_minutes = 0
            position = Position('SHORT', price, t_ms)
            bars_in_trade = 0
            print(f"  [{symbol}] OPEN SHORT @ {price:.4f}  {_fmt_time(t_ms)}")

        elif action == ACTION_CLOSE and position is not None:
            position.close(price, t_ms)
            pnl = net_pnl(position, config)
            held = round((t_ms - position.open_time_ms) / 60000)
            cumulative_pnl += pnl
            equity_curve.append(cumulative_pnl)
            d = str(date.fromtimestamp(t_ms / 1000))
            daily_pnl[d] = daily_pnl.get(d, 0.0) + pnl
            trades.append({'pnl': pnl, 'side': position.side,
                           'open_time': position.open_time_ms, 'close_time': t_ms})
            pnl_sign = '+' if pnl >= 0 else ''
            print(f"  [{symbol}] CLOSE {position.side:<5} @ {price:.4f}  "
                  f"PnL: {pnl_sign}{pnl:.4f}  held {held} min  "
                  f"cumPnL: {cumulative_pnl:.4f}  {_fmt_time(t_ms)}")
            position = None
            bars_in_trade = 0
            hold_minutes = 0

        else:
            if position is not None:
                bars_in_trade += 1
            hold_minutes += 1

    if hold_minutes:
        print(f"  [{symbol}] HOLD {hold_minutes} min")

    # Force-close any open position at end of OOS
    bars_in_trade = 0
    if position is not None and len(candles_1m) > oos_start:
        last = candles_1m[-1]
        position.close(last['close'], last['close_time'])
        pnl = net_pnl(position, config)
        held = round((last['close_time'] - position.open_time_ms) / 60000)
        cumulative_pnl += pnl
        equity_curve.append(cumulative_pnl)
        d = str(date.fromtimestamp(last['close_time'] / 1000))
        daily_pnl[d] = daily_pnl.get(d, 0.0) + pnl
        trades.append({'pnl': pnl, 'side': position.side,
                       'open_time': position.open_time_ms, 'close_time': last['close_time'],
                       'force_closed': True})
        pnl_sign = '+' if pnl >= 0 else ''
        print(f"  [{symbol}] FORCE-CLOSE {position.side:<5} @ {last['close']:.4f}  "
              f"PnL: {pnl_sign}{pnl:.4f}  held {held} min  {_fmt_time(last['close_time'])}")

    print(f"[Backtest] {symbol}: {len(trades)} trades, cumulative PnL = {cumulative_pnl:.4f}")
    return {'trades': trades, 'equity_curve': equity_curve,
            'daily_pnl': daily_pnl, 'total_steps': len(candles_1m) - oos_start}


# ── Metrics ────────────────────────────────────────────────────────────────────

def calc_metrics(trades: list, equity_curve: list, daily_pnl: dict) -> dict:
    """Full metrics suite. Reuses python/utils/metrics.py."""
    if not trades:
        return {'total_pnl': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0,
                'win_rate': 0.0, 'profit_factor': 0.0, 'avg_win': 0.0,
                'avg_loss': 0.0, 'total_trades': 0, 'max_consecutive_losses': 0}

    import math
    daily_returns = list(daily_pnl.values())
    if len(daily_returns) >= 2:
        ann_sharpe = sharpe_ratio(daily_returns) * math.sqrt(252)
    else:
        ann_sharpe = sharpe_ratio([t['pnl'] for t in trades])

    wl = avg_win_loss(trades)
    return {
        'total_pnl':               round(sum(t['pnl'] for t in trades), 6),
        'sharpe_ratio':            round(ann_sharpe, 4),
        'max_drawdown':            round(max_drawdown(equity_curve), 4),
        'win_rate':                round(win_rate(trades), 4),
        'profit_factor':           round(profit_factor(trades), 4),
        'avg_win':                 round(wl['avg_win'], 6),
        'avg_loss':                round(wl['avg_loss'], 6),
        'total_trades':            len(trades),
        'max_consecutive_losses':  max_consecutive_losses(trades),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='DQN-OPUS Backtester')
    parser.add_argument('--config', default=None,  help='Path to config.toml')
    parser.add_argument('--model',  default=None,  help='Override checkpoint path')
    parser.add_argument('--device', default='cpu', help='cpu or cuda (default: cpu)')
    args = parser.parse_args()

    cfg_path = args.config or os.path.join(os.path.dirname(__file__), '..', 'config.toml')
    config = load_config(os.path.abspath(cfg_path))
    config['learner']['device'] = args.device  # override — batch_size=1, CPU is faster

    bt = config['backtesting']
    model_path = args.model or bt.get('model_path', '')
    results_dir = bt.get('results_dir', 'results')

    if not model_path or not os.path.exists(model_path):
        # Auto-find latest checkpoint
        ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
        if os.path.isdir(ckpt_dir):
            pts = [f for f in os.listdir(ckpt_dir) if f.endswith('.pt')]
            if pts:
                pts.sort(key=lambda f: os.path.getmtime(os.path.join(ckpt_dir, f)))
                model_path = os.path.join(ckpt_dir, pts[-1])
                print(f"[Backtest] Auto-selected latest checkpoint: {model_path}")
        if not model_path or not os.path.exists(model_path):
            print(f"[Backtest] ERROR: no checkpoint found in '{ckpt_dir}'")
            print("[Backtest] Set [backtesting] model_path in config.toml or pass --model <path>")
            sys.exit(1)

    print(f"[Backtest] Model: {model_path}")
    model = load_model(model_path, config, args.device)

    try:
        raw = input("\nIle ostatnich swiec 1m przetestowac? (Enter = wszystkie OOS): ").strip()
        last_n = int(raw) if raw else None
    except (ValueError, EOFError):
        last_n = None
    if last_n is not None:
        print(f"[Backtest] Testing last {last_n} candles per symbol")
    else:
        print("[Backtest] Testing full OOS set")

    symbols = [a['symbol'] for a in config['actors']]
    per_symbol, all_trades, all_equity, all_daily = {}, [], [0.0], {}

    for symbol in symbols:
        print(f"\n[Backtest] ── {symbol} ──")
        res = run_symbol_backtest(symbol, config, model, args.device, last_n=last_n)
        m = calc_metrics(res['trades'], res['equity_curve'], res['daily_pnl'])
        per_symbol[symbol] = {**m, 'equity_curve': res['equity_curve']}
        all_trades.extend(res['trades'])
        for t in res['trades']:
            all_equity.append(all_equity[-1] + t['pnl'])
        for d, v in res['daily_pnl'].items():
            all_daily[d] = all_daily.get(d, 0.0) + v

    combined = calc_metrics(all_trades, all_equity, all_daily)

    output = {
        'timestamp':  datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        'model_path': model_path,
        'combined':   combined,
        'per_symbol': per_symbol,
    }

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    # ── Summary table ──
    W = 62
    print(f"\n{'='*W}")
    print(f"{'BACKTEST RESULTS':^{W}}")
    print(f"{'='*W}")
    print(f"{'Symbol':<12} {'Trades':>6} {'PnL':>10} {'Sharpe':>8} {'WinRate':>8} {'MaxDD':>8}")
    print(f"{'-'*W}")
    for sym, m in per_symbol.items():
        print(f"{sym:<12} {m['total_trades']:>6} {m['total_pnl']:>10.4f} "
              f"{m['sharpe_ratio']:>8.3f} {m['win_rate']:>8.2%} {m['max_drawdown']:>8.4f}")
    print(f"{'-'*W}")
    cm = combined
    print(f"{'COMBINED':<12} {cm['total_trades']:>6} {cm['total_pnl']:>10.4f} "
          f"{cm['sharpe_ratio']:>8.3f} {cm['win_rate']:>8.2%} {cm['max_drawdown']:>8.4f}")
    print(f"{'='*W}")
    print(f"\nResults saved → {out_path}")


if __name__ == '__main__':
    main()
