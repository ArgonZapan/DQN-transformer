"""
Backtest for QuantileNet.

Two evaluation modes:
  1. Calibration  — statistical accuracy of probability and quantile predictions
                    (results also saved to calibration.json in the checkpoint run folder)
  2. Trading      — brute-force TP/SL combinations driven by threshold probabilities,
                    top-20 by SQN printed to console

Usage (from DQN-QuantileNet/):
    ..\\..\\venv_cuda\\Scripts\\python.exe python/backtest.py [--checkpoint path/to/best.pt]
"""

import os
import sys
import argparse
import json
import logging
import glob
import hashlib
import pickle
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, UTC
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from python.config                import load_config, get_device
from python.data.loader           import build_windows, _load_csv, _base_tf, TF_MINUTES
from python.model.tft             import QuantileNet
from python.training.dataset      import QuantileDataset, collate_fn
from python.training.walk_forward import build_walk_forward_folds

logger = logging.getLogger(__name__)

COMMISSION = 0.00075  # Binance taker fee per side
MIN_TRADES_VALID = 20  # flag trades below this as statistically low


# ── EV-Based Strategy Helpers ──────────────────────────────────────────────────

def calculate_ev(p_win: float, p_loss: float, return_win: float, return_loss: float) -> float:
    """Expected Value = P(win) * return_win + P(loss) * return_loss (in %)."""
    return (p_win * return_win) + (p_loss * return_loss)


def calculate_kelly(p_win: float, avg_win: float, avg_loss: float) -> float:
    """Kelly Criterion: f = (p*b - q) / b, where p=win%, b=ratio, q=loss%.

    Returns fraction of capital to risk [0, 1]. Cap at 0.25 for safety.
    """
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0
    b = avg_win / avg_loss  # odds ratio
    q = 1 - p_win
    kelly_f = (p_win * b - q) / b if b > 0 else 0.0
    return min(max(kelly_f, 0.0), 0.25)  # clamp [0, 0.25]


def get_quantile_tp_sl(quantiles: np.ndarray, q_idx_tp: int, q_idx_sl: int,
                       direction: str) -> tuple[float, float]:
    """Extract TP and SL percentages from predicted return quantiles.

    Args:
        quantiles: [n_quantiles] predicted RETURNS at quantile levels (e.g. 0.05 = 5%)
        q_idx_tp: index for TP quantile (e.g., 4 for p90)
        q_idx_sl: index for SL quantile (e.g., 0 for p10)
        direction: 'long' or 'short'

    Returns:
        (tp_pct, sl_pct) as positive percentages
    """
    if direction == 'long':
        # For long: TP = upper quantile return, SL = magnitude of lower quantile return
        tp_pct = float(quantiles[q_idx_tp]) * 100.0   # e.g. 0.08 -> 8%
        sl_pct = -float(quantiles[q_idx_sl]) * 100.0  # e.g. -0.03 -> 3%
    else:  # short
        # For short: TP = magnitude of lower quantile (price drops = profit), SL = upper quantile
        tp_pct = -float(quantiles[q_idx_sl]) * 100.0  # e.g. -0.08 -> 8%
        sl_pct = float(quantiles[q_idx_tp]) * 100.0   # e.g. 0.03 -> 3%
    return tp_pct, sl_pct


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_model(config: dict, device: torch.device, checkpoint: str) -> QuantileNet:
    model = QuantileNet(config).to(device)
    ckpt  = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    logger.info(f'Loaded checkpoint: {checkpoint}  (epoch {ckpt.get("epoch", "?")},'
                f' val_loss={ckpt.get("val_loss", float("nan")):.5f})')
    return model


def find_best_checkpoint(ckpt_dir: str) -> str:
    # Search all run subfolders for best checkpoints
    pattern_best = os.path.join(ckpt_dir, '*', '*_best_*.pt')
    files = glob.glob(pattern_best)
    if not files:
        files = glob.glob(os.path.join(ckpt_dir, '*', '*.pt'))
    if not files:
        # Fallback: flat checkpoints directory
        files = glob.glob(os.path.join(ckpt_dir, '*.pt'))
    if not files:
        raise FileNotFoundError(f'No checkpoint found in {ckpt_dir}')
    return max(files, key=os.path.getmtime)


def build_test_windows(config: dict) -> list[dict]:
    """Rebuild the explicit out-of-sample test slice via walk-forward split."""
    symbols = [a['symbol'] for a in config['actors']]
    all_windows = []
    for symbol in symbols:
        windows = build_windows(symbol, config)
        logger.info(f'[{symbol}] Full-history windows: {len(windows)}')
        all_windows.extend(windows)

    train_cfg = config['training']
    _, test_windows = build_walk_forward_folds(
        all_windows,
        train_months=train_cfg['train_months'],
        val_months=train_cfg['val_months'],
        test_months=train_cfg['test_months'],
    )
    logger.info(f'[Backtest] Test windows: {len(test_windows)}')
    return test_windows


def _windows_cache_key(config: dict) -> str:
    """Hash that captures config + CSV mtime so cache invalidates on data change."""
    data_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'node', 'data', 'historical')
    )
    h = hashlib.md5()
    # Config fields that affect window contents
    for key in ('timeframes', 'prediction', 'features', 'training'):
        h.update(json.dumps(config.get(key, {}), sort_keys=True).encode())
    for actor in config.get('actors', []):
        symbol = actor['symbol']
        for tf_key in config.get('timeframes', {}):
            if not config['timeframes'].get(tf_key, 0):
                continue
            tf = tf_key.replace('candles_', '')
            fpath = os.path.join(data_path, f'{symbol}_{tf}.csv')
            if os.path.exists(fpath):
                h.update(str(os.path.getmtime(fpath)).encode())
    return h.hexdigest()


def load_test_windows_cached(config: dict) -> list[dict]:
    """Load test windows, using a file cache to skip recomputation."""
    cache_dir = os.path.join(os.path.dirname(__file__), '..', 'node', 'data', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    key   = _windows_cache_key(config)
    fpath = os.path.join(cache_dir, f'test_windows_{key}.pkl')

    if os.path.exists(fpath):
        logger.info(f'[Backtest] Loading test windows from cache: {os.path.basename(fpath)}')
        with open(fpath, 'rb') as f:
            return pickle.load(f)

    windows = build_test_windows(config)
    with open(fpath, 'wb') as f:
        pickle.dump(windows, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f'[Backtest] Test windows cached to: {os.path.basename(fpath)}')
    return windows


def load_ohlcv(config: dict) -> tuple[dict[str, dict], int]:
    """Load raw OHLCV for the base TF for all symbols as sorted NumPy arrays.

    Returns (ohlcv_by_symbol, base_tf_min) where:
      ohlcv_by_symbol: symbol -> {'times': int64[N], 'highs': f64[N], 'lows': f64[N]}
      times are sorted ascending, enabling np.searchsorted lookups.
    """
    data_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'node', 'data', 'historical')
    )
    symbols  = [a['symbol'] for a in config['actors']]
    base_tf  = _base_tf(config)
    base_min = TF_MINUTES[base_tf]

    ohlcv_by_symbol: dict[str, dict] = {}
    for symbol in symbols:
        raw = _load_csv(symbol, base_tf, data_path)
        times = np.asarray(raw['close_time'], dtype=np.int64)
        highs = np.asarray(raw['high'],       dtype=np.float64)
        lows  = np.asarray(raw['low'],        dtype=np.float64)
        order = np.argsort(times, kind='stable')
        ohlcv_by_symbol[symbol] = {
            'times': np.ascontiguousarray(times[order]),
            'highs': np.ascontiguousarray(highs[order]),
            'lows':  np.ascontiguousarray(lows[order]),
        }
        logger.info(f'[{symbol}] OHLCV index: {len(times)} candles ({base_tf})')

    return ohlcv_by_symbol, base_min


class StageTimer:
    """Lightweight stage profiler for backtest logs."""

    def __init__(self, label: str):
        self.label = label
        self.started = time.perf_counter()

    def checkpoint(self, name: str):
        now = time.perf_counter()
        elapsed = now - self.started
        logger.info(f'[{self.label}] {name}: {elapsed:.2f}s')
        self.started = now


@torch.no_grad()
def run_inference(model: QuantileNet, windows: list, config: dict,
                  device: torch.device) -> dict[str, np.ndarray]:
    """Run model on all windows. Returns dict of raw predictions and labels."""
    active_tfs  = model.active_tfs
    tf_seq_lens = model.tf_seq_lens
    dataset     = QuantileDataset(windows, config, active_tfs, tf_seq_lens)
    train_cfg   = config.get('training', {})
    loader_workers = int(train_cfg.get('backtest_inference_workers', 0))
    if loader_workers <= 0:
        if device.type == 'cuda':
            loader_workers = min(8, os.cpu_count() or 1)
        else:
            loader_workers = max(1, min(4, os.cpu_count() or 1))
    loader      = DataLoader(dataset, batch_size=256, shuffle=False,
                             collate_fn=collate_fn,
                             num_workers=loader_workers,
                             pin_memory=device.type == 'cuda',
                             persistent_workers=loader_workers > 0)

    logger.info(
        f'[Backtest] Inference loader: batch_size=256 num_workers={loader_workers} '
        f'device={device.type}'
    )

    pred_q_list, pred_t_list = [], []
    label_q_list, label_t_list = [], []

    for batch in loader:
        tf_inputs = {tf: t.to(device) for tf, t in batch['tf'].items()}
        outputs   = model(tf_inputs)

        if 'quantile' in outputs:
            pred_q_list.append(outputs['quantile'].cpu().numpy())
        if 'threshold' in outputs:
            pred_t_list.append(outputs['threshold'].cpu().numpy())
        if 'quantile_label' in batch:
            label_q_list.append(batch['quantile_label'].numpy())
        if 'threshold_label' in batch:
            label_t_list.append(batch['threshold_label'].numpy())

    results = {}
    if pred_q_list:
        results['pred_quantile']   = np.concatenate(pred_q_list,  axis=0)
    if pred_t_list:
        results['pred_threshold']  = np.concatenate(pred_t_list,  axis=0)
        pred_t_hash = int(hashlib.md5(results['pred_threshold'].tobytes()).hexdigest()[:16], 16)
        logger.info(f'[Backtest] run_inference computed pred_threshold hash: {pred_t_hash:016x}')
    if label_q_list:
        results['label_quantile']  = np.concatenate(label_q_list, axis=0)
    if label_t_list:
        results['label_threshold'] = np.concatenate(label_t_list, axis=0)

    return results


# ── Calibration ────────────────────────────────────────────────────────────────

def calibration_report(results: dict, config: dict, save_path: str | None = None):
    """Print calibration report and optionally save all data to JSON."""
    pred_cfg   = config['prediction']
    horizons   = pred_cfg['horizons_hours']
    thresholds = pred_cfg.get('thresholds_pct', [])
    quantiles  = pred_cfg.get('quantiles', [])
    direction  = pred_cfg['direction']
    dir_labels = ['long', 'short'] if direction == 'both' else [direction]

    json_data: dict = {
        'generated_at': datetime.now(UTC).isoformat(),
        'threshold_calibration': [],
        'quantile_coverage': [],
    }

    print('\n' + '=' * 70)
    print('CALIBRATION REPORT')
    print('=' * 70)

    # ── Threshold calibration ──────────────────────────────────────────────
    if 'pred_threshold' in results and 'label_threshold' in results:
        pred  = results['pred_threshold']   # [N, n_horizons, n_thr, n_dir]
        label = results['label_threshold']  # [N, n_horizons, n_thr, n_dir]

        print('\nThreshold calibration (predicted prob vs actual hit rate):')
        print(f'{"Horizon":>8} {"Direction":>10} {"Threshold%":>12} '
              f'{"Mean Pred%":>12} {"Actual%":>10} {"Error":>8} {"N":>8}')
        print('-' * 70)

        for hi, h in enumerate(horizons):
            for di, d in enumerate(dir_labels):
                for ti, thr in enumerate(thresholds):
                    p = pred[:, hi, ti, di]
                    l = label[:, hi, ti, di]
                    mean_pred = float(p.mean())
                    actual    = float(l.mean())
                    error     = mean_pred - actual
                    print(f'{h:>7}h {d:>10} {thr:>11.1f}% '
                          f'{mean_pred * 100:>11.1f}% {actual * 100:>9.1f}% '
                          f'{error * 100:>+7.1f}% {len(p):>8}')
                    json_data['threshold_calibration'].append({
                        'horizon_h':     h,
                        'direction':     d,
                        'threshold_pct': thr,
                        'mean_pred_pct': round(mean_pred * 100, 4),
                        'actual_pct':    round(actual * 100, 4),
                        'error_pct':     round(error * 100, 4),
                        'n':             int(len(p)),
                    })

    # ── Quantile coverage ──────────────────────────────────────────────────
    if 'pred_quantile' in results and 'label_quantile' in results:
        pred  = results['pred_quantile']    # [N, n_horizons, n_q, n_dir]
        label = results['label_quantile']   # [N, n_horizons, 1, n_dir]

        print('\nQuantile coverage (fraction of actuals below predicted quantile):')
        print(f'{"Horizon":>8} {"Direction":>10} {"Quantile":>10} '
              f'{"Expected%":>11} {"Actual%":>10} {"Error":>8}')
        print('-' * 70)

        for hi, h in enumerate(horizons):
            for di, d in enumerate(dir_labels):
                for qi, q in enumerate(quantiles):
                    p_q      = pred[:, hi, qi, di]
                    actual_r = label[:, hi, 0, di]
                    coverage = float((actual_r <= p_q).mean())
                    error    = coverage - q
                    print(f'{h:>7}h {d:>10} {q:>10.2f} '
                          f'{q * 100:>10.1f}% {coverage * 100:>9.1f}% '
                          f'{error * 100:>+7.1f}%')
                    json_data['quantile_coverage'].append({
                        'horizon_h':    h,
                        'direction':    d,
                        'quantile':     q,
                        'expected_pct': round(q * 100, 4),
                        'actual_pct':   round(coverage * 100, 4),
                        'error_pct':    round(error * 100, 4),
                        'n':            int(len(p_q)),
                    })

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
        logger.info(f'[Calibration] Saved to {save_path}')


# ── TP/SL simulation ───────────────────────────────────────────────────────────

def _format_ts(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=UTC).strftime('%Y-%m-%d %H:%M')


def _portfolio_equity(symbol_equity: dict[str, float]) -> float:
    return float(np.mean(list(symbol_equity.values()))) if symbol_equity else 1.0


def _timeline_stats(trades: list[dict]) -> tuple[float, float, float, float, float, float, float, float, float]:
    if not trades:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    trade_returns = np.array([t['return'] for t in trades], dtype=np.float64)
    wins   = float((trade_returns > 0).mean())
    avg    = float(trade_returns.mean())
    tp_rate = float(sum(1 for t in trades if t.get('exit_type') == 'tp') / len(trades))
    sl_rate = float(sum(1 for t in trades if t.get('exit_type') == 'sl') / len(trades))
    hz_rate = float(sum(1 for t in trades if t.get('exit_type') == 'hz') / len(trades))
    t_std  = float(trade_returns.std())
    sqn    = float(np.sqrt(len(trade_returns)) * avg / t_std) if len(trade_returns) >= 2 and t_std > 0 else 0.0

    active_symbols = {t['symbol'] for t in trades}
    symbol_equity = {s: 1.0 for s in active_symbols}
    trades_by_exit: dict[int, list[dict]] = defaultdict(list)
    for trade in trades:
        trades_by_exit[int(trade['exit_time'])].append(trade)

    event_returns  = []
    event_times    = []
    portfolio_curve = []
    for exit_time in sorted(trades_by_exit):
        old = _portfolio_equity(symbol_equity)
        for trade in trades_by_exit[exit_time]:
            symbol_equity[trade['symbol']] *= 1.0 + trade['return']
        new = _portfolio_equity(symbol_equity)
        event_returns.append(new / old - 1.0)
        event_times.append(exit_time)
        portfolio_curve.append(new)

    total    = _portfolio_equity(symbol_equity) - 1.0
    curve    = np.array(portfolio_curve, dtype=np.float64)
    peak     = np.maximum.accumulate(curve)
    max_dd   = float(((curve / peak) - 1.0).min()) if len(curve) else 0.0
    ev_ret   = np.array(event_returns, dtype=np.float64)
    if len(ev_ret) < 2 or ev_ret.std() < 1e-10:
        sharpe = 0.0
    else:
        ms_per_year  = 365.25 * 24 * 60 * 60 * 1000
        span_years   = max((event_times[-1] - event_times[0]) / ms_per_year, 1e-9)
        events_yr    = len(ev_ret) / span_years
        sharpe = float(ev_ret.mean() / ev_ret.std() * np.sqrt(events_yr))

    return wins, avg, float(total), sharpe, sqn, max_dd, tp_rate, sl_rate, hz_rate


# Module-level globals — set once per worker via initializer, shared by all jobs in that worker.
# _W_SYM_DATA[symbol] = dict with precomputed NumPy arrays:
#   current_times:  int64[N]   window entry times (sorted)
#   entry_prices:   f64[N]
#   horizon_times:  int64[N, H]
#   horizon_closes: f64[N, H]
#   pred:           f64[N, H, T, D]
#   ohlcv_times:    int64[M]   base-TF candle close times (sorted)
#   ohlcv_highs:    f64[M]
#   ohlcv_lows:     f64[M]
#   entry_idx:      int64[N]   precomputed: searchsorted(ohlcv_times, current_times + step_ms)
_W_SYM_DATA: dict = {}


def _worker_init(sym_data: dict):
    global _W_SYM_DATA
    _W_SYM_DATA = sym_data


def _combo_chunk_worker(args: tuple):
    """Process a coarse chunk of combinations inside one worker."""
    (symbol, hi, ti, direction, horizon_hours, threshold_pct,
     entry_probs, tp_sl_pairs) = args
    sd = _W_SYM_DATA[symbol]
    rows = []
    for entry_prob in entry_probs:
        for tp_pct, sl_pct in tp_sl_pairs:
            row = _simulate_combo(
                sd=sd,
                symbol=symbol,
                hi=hi,
                ti=ti,
                entry_prob=entry_prob,
                direction=direction,
                horizon_hours=horizon_hours,
                threshold_pct=threshold_pct,
                tp_pct=tp_pct,
                sl_pct=sl_pct,
            )
            if row is not None:
                rows.append(row)
    return symbol, rows


def _collect_ev_based_trades(
    sd: dict,
    hi: int,
    ti: int,
    entry_prob: float,
    direction: str,
    q_idx_tp: int,
    q_idx_sl: int,
    pred_quantile: np.ndarray,
    pred_threshold: np.ndarray,
) -> list[dict]:
    """EV-based trading with dynamic TP/SL from quantiles.

    Strategy:
      1. Entry: threshold probability >= entry_prob
      2. TP/SL: dynamic from quantiles (p75=TP, p25=SL)
      3. Kelly: position sizing based on EV
      4. Parallel entries: no sequential lock
    """
    pred_t         = pred_threshold    # [N, H, T, D]
    pred_q         = pred_quantile      # [N, H, n_quantiles, D]
    current_times  = sd['current_times'] # [N]
    entry_prices   = sd['entry_prices']  # [N]
    horizon_times  = sd['horizon_times'] # [N, H]
    horizon_closes = sd['horizon_closes']# [N, H]
    entry_idx_all  = sd['entry_idx']     # [N]
    o_times        = sd['ohlcv_times']
    o_highs        = sd['ohlcv_highs']
    o_lows         = sd['ohlcv_lows']

    # 1) Signal selection (entry if threshold >= entry_prob)
    if direction == 'both':
        p_long  = pred_t[:, hi, ti, 0]
        p_short = pred_t[:, hi, ti, 1]
        long_mask  = (p_long  >= entry_prob) & (p_long  > p_short)
        short_mask = (p_short >= entry_prob) & (p_short > p_long)
        signal_mask = long_mask | short_mask
        dirs_all = np.where(long_mask, 1, np.where(short_mask, -1, 0)).astype(np.int8)
    else:
        p_dir = pred_t[:, hi, ti, 0]
        signal_mask = p_dir >= entry_prob
        dirs_all = np.where(signal_mask, 1 if direction == 'long' else -1, 0).astype(np.int8)

    sig_indices = np.nonzero(signal_mask)[0]
    if sig_indices.size == 0:
        return []

    cost = 2.0 * COMMISSION
    trades: list[dict] = []
    next_entry_time = -1  # sequential lock: no new entry before previous trade closes

    for idx in sig_indices:
        ct = current_times[idx]
        if ct < next_entry_time:
            continue  # previous trade still open, skip this signal
        trade_dir = int(dirs_all[idx])
        entry_px  = entry_prices[idx]
        h_time    = horizon_times[idx, hi]
        h_close   = horizon_closes[idx, hi]

        # Get dynamic TP/SL from quantiles (predicted returns, not prices)
        # For long: use quantiles[:, 0], for short: use quantiles[:, 1]
        dir_idx = 0 if trade_dir > 0 else 1
        quantiles = pred_q[idx, hi, :, dir_idx]  # [n_quantiles] — predicted returns
        tp_pct, sl_pct = get_quantile_tp_sl(quantiles, q_idx_tp, q_idx_sl,
                                            'long' if trade_dir > 0 else 'short')

        # Safety: ensure positive SL, cap TP at horizon
        sl_pct = max(sl_pct, 0.5)  # minimum 0.5% SL
        tp_pct = min(tp_pct, 50.0)  # cap TP at 50%
        sl_pct = min(sl_pct, 50.0)  # cap SL at 50%

        tp_mult = tp_pct / 100.0
        sl_mult = sl_pct / 100.0

        start = int(entry_idx_all[idx])
        end = int(np.searchsorted(o_times, h_time, side='right'))

        if end > start:
            highs = o_highs[start:end]
            lows = o_lows[start:end]
            if trade_dir > 0:  # long
                sl_lvl = entry_px * (1.0 - sl_mult)
                tp_lvl = entry_px * (1.0 + tp_mult)
                sl_hits = lows <= sl_lvl
                tp_hits = highs >= tp_lvl
            else:  # short
                sl_lvl = entry_px * (1.0 + sl_mult)
                tp_lvl = entry_px * (1.0 - tp_mult)
                sl_hits = highs >= sl_lvl
                tp_hits = lows <= tp_lvl

            sl_any = sl_hits.any()
            tp_any = tp_hits.any()
            sl_i = int(sl_hits.argmax()) if sl_any else -1
            tp_i = int(tp_hits.argmax()) if tp_any else -1

            if sl_any and (not tp_any or sl_i <= tp_i):
                exit_type = 'sl'
                exit_t = int(o_times[start + sl_i])
                ret = -sl_mult - cost
            elif tp_any:
                exit_type = 'tp'
                exit_t = int(o_times[start + tp_i])
                ret = tp_mult - cost
            else:
                exit_type = 'hz'
                exit_t = int(h_time)
                ret = ((h_close - entry_px) / entry_px if trade_dir > 0
                       else (entry_px - h_close) / entry_px) - cost
        else:
            exit_type = 'hz'
            exit_t = int(h_time)
            ret = ((h_close - entry_px) / entry_px if trade_dir > 0
                   else (entry_px - h_close) / entry_px) - cost

        trades.append({
            'symbol':     sd['symbol'],
            'entry_time': int(ct),
            'exit_time':  exit_t,
            'return':     float(ret),
            'exit_type':  exit_type,
            'tp_pct':     float(tp_pct),
            'sl_pct':     float(sl_pct),
        })
        next_entry_time = exit_t  # block new entries until this trade closes

    return trades


def _simulate_combo_ev(
    sd: dict,
    symbol: str,
    hi: int,
    ti: int,
    entry_prob: float,
    direction: str,
    horizon_hours: float,
    threshold_pct: float,
    pred_quantile: np.ndarray,
    pred_threshold: np.ndarray,
    q_idx_tp: int = 4,  # p90
    q_idx_sl: int = 0,  # p10
) -> dict | None:
    """Run EV-based combo: dynamic TP/SL from quantiles, rank by EV."""
    trades = _collect_ev_based_trades(sd, hi, ti, entry_prob, direction,
                                       q_idx_tp, q_idx_sl, pred_quantile, pred_threshold)
    if not trades:
        return None

    # Calculate EV and Kelly
    returns = np.array([t['return'] for t in trades])
    wins = (returns > 0).sum()
    p_win = wins / len(trades) if len(trades) > 0 else 0.0
    p_loss = 1.0 - p_win
    avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0.01
    avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 0.01

    ev = calculate_ev(p_win, p_loss, avg_win * 100, -avg_loss * 100)
    kelly = calculate_kelly(p_win, avg_win * 100, avg_loss * 100)

    # Calculate extended metrics (for printing, compatibility with old format)
    total_return = returns.sum()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0.0

    # SQN: Sequential probability ratio test statistic (for ranking)
    # SQN = sqrt(N) * avg / std
    if returns.std() > 0:
        sqn = np.sqrt(len(trades)) * returns.mean() / returns.std()
    else:
        sqn = 0.0

    # Compute TP/SL/Hz rates
    tp_count = sum(1 for t in trades if t['exit_type'] == 'tp')
    sl_count = sum(1 for t in trades if t['exit_type'] == 'sl')
    hz_count = sum(1 for t in trades if t['exit_type'] == 'hz')
    n = len(trades)
    tp_rate = tp_count / n if n > 0 else 0.0
    sl_rate = sl_count / n if n > 0 else 0.0
    hz_rate = hz_count / n if n > 0 else 0.0

    # Max drawdown
    if len(trades) > 0:
        cumul_returns = np.cumsum(returns)
        peak = np.maximum.accumulate(cumul_returns)
        drawdown = (cumul_returns - peak) / peak
        max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0
    else:
        max_dd = 0.0

    # Get average TP/SL percentages from trades
    if trades:
        tp_pcts = [t['tp_pct'] for t in trades]
        sl_pcts = [t['sl_pct'] for t in trades]
        tp_pct_sample = np.mean(tp_pcts) if tp_pcts else 0.0
        sl_pct_sample = np.mean(sl_pcts) if sl_pcts else 0.0
    else:
        tp_pct_sample = sl_pct_sample = 0.0

    return {
        'symbol':     symbol,
        'horizon':    horizon_hours,
        'threshold':  threshold_pct,
        'direction':  direction,
        'entry_prob': entry_prob,
        'tp_pct':     float(tp_pct_sample),
        'sl_pct':     float(sl_pct_sample),
        'n_trades':   len(trades),
        'p_win':      float(p_win),
        'avg_win':    float(avg_win * 100),
        'avg_loss':   float(avg_loss * 100),
        'total_return': float(total_return * 100),
        'total':      float(total_return),
        'sharpe':     float(sharpe),
        'sqn':        float(sqn),
        'max_dd':     float(max_dd),
        'tp_rate':    float(tp_rate),
        'sl_rate':    float(sl_rate),
        'hz_rate':    float(hz_rate),
        'avg':        float(returns.mean()),
        'ev':         float(ev),
        'kelly':      float(kelly),
        'trades':     trades,
    }


def _print_top100_ev(rows: list[dict], title: str) -> list[dict]:
    """Print top 100 by SQN across all symbols."""
    any_valid = any(r['n_trades'] >= MIN_TRADES_VALID for r in rows)
    display   = [r for r in rows if r['n_trades'] >= MIN_TRADES_VALID] if any_valid else rows
    top100    = sorted(display, key=lambda r: r['sqn'], reverse=True)[:100]

    print(f'\n{"=" * 122}')
    print(f'{title}')
    print(f'{"=" * 122}')
    print(f'{"#":>3} {"Symbol":>8} {"H":>5} {"Thr%":>5} {"Entry%":>8} {"N":>6} {"P(Win)":>7} {"Avg+%":>7} {"Avg-%":>7} '
          f'{"Total%":>8} {"Sharpe":>7} {"SQN":>7} {"EV%":>7} {"Kelly":>7} {"TP%":>5} {"SL%":>5}')
    print('-' * 122)

    for rank, row in enumerate(top100, 1):
        low_flag = ' *' if row['n_trades'] < MIN_TRADES_VALID else '  '
        entry_str = f'{row["entry_prob"]:.0%}'
        print(
            f'{rank:>3} {row["symbol"]:>8} {row["horizon"]:>4}h {row["threshold"]:>4.1f}% '
            f'{entry_str:>8} {row["n_trades"]:>5}{low_flag} '
            f'{row["p_win"] * 100:>6.1f}% {row["avg_win"]:>6.2f}% {row["avg_loss"]:>6.2f}% '
            f'{row["total_return"]:>7.2f}% {row["sharpe"]:>7.2f} {row["sqn"]:>7.2f} '
            f'{row["ev"]:>6.2f}% {row["kelly"]:>6.3f} '
            f'{row["tp_pct"]:>4.1f}% {row["sl_pct"]:>4.1f}%'
        )
    return top100


def _print_top20(rows: list[dict], title: str,
                 sqn_key: str, n_key: str,
                 tp_key: str, sl_key: str, hz_key: str | None,
                 avg_key: str, total_key: str, mdd_key: str, sharpe_key: str):
    any_valid = any(r[n_key] >= MIN_TRADES_VALID for r in rows)
    display   = [r for r in rows if r[n_key] >= MIN_TRADES_VALID] if any_valid else rows
    top20     = sorted(display, key=lambda r: r[sqn_key], reverse=True)[:20]
    has_hz    = hz_key is not None

    print(f'\n{"=" * 70}')
    print(f'{title} — TOP 20 BY SQN')
    print(f'{"=" * 70}')
    hz_col = f' {"Hz%":>6}' if has_hz else ''
    width  = 115 if has_hz else 107
    print(f'{"#":>3} {"H":>5} {"Thr%":>5} {"Entry(s)":>13} {"TP%":>5} {"SL%":>5} {"RR":>5} '
          f'{"N":>6} {"TP%":>6} {"SL%":>6}{hz_col} {"AvgR%":>7} {"Total%":>8} {"MDD%":>7} {"Sharpe":>7} {"SQN":>7}')
    print('-' * width)

    for rank, row in enumerate(top20, 1):
        low_flag = ' *' if row[n_key] < MIN_TRADES_VALID else '  '
        hz_val   = f' {row[hz_key] * 100:>5.1f}%' if has_hz else ''
        # Show all entry_probs that produced this combo, comma-separated
        entry_strs = [f'{ep:.0%}' for ep in row.get('entry_probs_list', [row['entry_prob']])]
        entry_str = ','.join(entry_strs)
        print(
            f'{rank:>3} {row["horizon"]:>4}h {row["threshold"]:>4.1f}% '
            f'{entry_str:>13} {row["tp_pct"]:>4.1f}% {row["sl_pct"]:>4.1f}% {row["rr"]:>5.1f} '
            f'{row[n_key]:>5}{low_flag} '
            f'{row[tp_key] * 100:>5.1f}% {row[sl_key] * 100:>5.1f}%{hz_val} '
            f'{row[avg_key] * 100:>6.3f}% {row[total_key] * 100:>7.1f}% '
            f'{row[mdd_key] * 100:>6.1f}% {row[sharpe_key]:>7.2f} {row[sqn_key]:>7.2f}'
        )
    return top20


def _print_top20_fixed(top20: list[dict], title: str):
    """Print Hz-excluded stats for a pre-ordered top20 list (same order as ALL EXITS)."""
    print(f'\n{"=" * 70}')
    print(f'{title}')
    print(f'{"=" * 70}')
    print(f'{"#":>3} {"H":>5} {"Thr%":>5} {"Entry(s)":>13} {"TP%":>5} {"SL%":>5} {"RR":>5} '
          f'{"N":>6} {"TP%":>6} {"SL%":>6} {"AvgR%":>7} {"Total%":>8} {"MDD%":>7} {"Sharpe":>7} {"SQN":>7}')
    print('-' * 107)
    for rank, row in enumerate(top20, 1):
        n = row['n_no_hz']
        low_flag = ' *' if n < MIN_TRADES_VALID else '  '
        # Show all entry_probs that produced this combo, comma-separated
        entry_strs = [f'{ep:.0%}' for ep in row.get('entry_probs_list', [row['entry_prob']])]
        entry_str = ','.join(entry_strs)
        print(
            f'{rank:>3} {row["horizon"]:>4}h {row["threshold"]:>4.1f}% '
            f'{entry_str:>13} {row["tp_pct"]:>4.1f}% {row["sl_pct"]:>4.1f}% {row["rr"]:>5.1f} '
            f'{n:>5}{low_flag} '
            f'{row["tp_rate_nh"] * 100:>5.1f}% {row["sl_rate_nh"] * 100:>5.1f}% '
            f'{row["avg_nh"] * 100:>6.3f}% {row["total_nh"] * 100:>7.1f}% '
            f'{row["max_dd_nh"] * 100:>6.1f}% {row["sharpe_nh"]:>7.2f} {row["sqn_nh"]:>7.2f}'
        )


def _build_row(trades: list, symbol: str, h: float, thr: float, direction: str,
               entry_prob: float, tp_pct: float, sl_pct: float) -> dict | None:
    n = len(trades)
    if n == 0:
        return None
    wins, avg, total, sharpe, sqn, max_dd, tp_rate, sl_rate, hz_rate = _timeline_stats(trades)
    trades_no_hz = [t for t in trades if t.get('exit_type') != 'hz']
    if trades_no_hz:
        wins_nh, avg_nh, total_nh, sharpe_nh, sqn_nh, max_dd_nh, tp_rate_nh, sl_rate_nh, _ = \
            _timeline_stats(trades_no_hz)
    else:
        wins_nh = avg_nh = total_nh = sharpe_nh = sqn_nh = max_dd_nh = tp_rate_nh = sl_rate_nh = 0.0
    return {
        'symbol':     symbol,
        'horizon':    h,
        'threshold':  thr,
        'direction':  direction,
        'entry_prob': entry_prob,
        'tp_pct':     tp_pct,
        'sl_pct':     sl_pct,
        'rr':         round(tp_pct / sl_pct, 2),
        'n_trades':   n,
        'wins':       wins,
        'tp_rate':    tp_rate,
        'sl_rate':    sl_rate,
        'hz_rate':    hz_rate,
        'avg':        avg,
        'total':      total,
        'sharpe':     sharpe,
        'sqn':        sqn,
        'max_dd':     max_dd,
        'n_no_hz':    len(trades_no_hz),
        'wins_nh':    wins_nh,
        'tp_rate_nh': tp_rate_nh,
        'sl_rate_nh': sl_rate_nh,
        'avg_nh':     avg_nh,
        'total_nh':   total_nh,
        'sharpe_nh':  sharpe_nh,
        'sqn_nh':     sqn_nh,
        'max_dd_nh':  max_dd_nh,
    }



def brute_force_tp_sl_top20(
    results: dict,
    windows: list,
    config: dict,
    ohlcv_by_symbol: dict[str, dict],
    base_tf_min: int,
):
    """Brute-force all (horizon × threshold × direction × entry_prob × TP × SL) combinations
    where RR > 1. Prints top-20 by SQN per symbol.

    STRATEGY: Parallel entries — each signal creates a trade immediately.
    Multiple concurrent positions are allowed (no sequential locking).
    entry_prob now has real impact on number of trades.
    """
    if 'pred_threshold' not in results:
        print('\n[Backtest] Threshold predictions not available — skipping TP/SL simulation.')
        return

    timer      = StageTimer('Backtest')
    pred_cfg   = config['prediction']
    train_cfg  = config.get('training', {})
    horizons   = pred_cfg['horizons_hours']
    thresholds = pred_cfg.get('thresholds_pct', [])
    direction  = pred_cfg['direction']
    symbols    = sorted({w['symbol'] for w in windows})
    test_start = min(int(w['current_close_time']) for w in windows)
    test_end   = max(int(w['future_close_times'][-1]) for w in windows)
    # Backtest is pure CPU work (unlike training) — use every logical thread.
    # Can be overridden via config [training].backtest_workers.
    num_workers = int(train_cfg.get('backtest_workers', 0)) or (os.cpu_count() or 1)

    pred = results['pred_threshold']   # [N, n_horizons, n_thr, n_dir]

    print(f'\nPred threshold stats: min={pred.min():.4f}  max={pred.max():.4f}'
          f'  mean={pred.mean():.4f}  p50={np.median(pred):.4f}'
          f'  p90={np.percentile(pred, 90):.4f}  p95={np.percentile(pred, 95):.4f}')

    # Full percentile sweep — exposes saturation / bimodal distributions.
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_vals = [float(np.percentile(pred, p)) for p in pcts]
    print('Pred percentiles ' + '  '.join(f'p{p}={v:.4f}' for p, v in zip(pcts, pct_vals)))
    flat = pred.ravel()
    bin_edges = np.linspace(0.0, 1.0, 11)
    hist, _ = np.histogram(flat, bins=bin_edges)
    total = max(int(hist.sum()), 1)
    print('Pred histogram (bin -> %): ' + '  '.join(
        f'[{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}]={hist[i] / total * 100:.1f}%'
        for i in range(len(hist))
    ))

    p50 = float(np.median(pred))
    p90 = float(np.percentile(pred, 90))
    entry_probs = sorted(set([
        round(p50 + 0.05, 2),
        round(p50 + 0.10, 2),
        round(p90, 2),
        round(p90 + 0.05, 2),
        round(min(p90 + 0.10, 0.99), 2),
    ]))

    # Calculate total combinations for EV-based strategy
    total_combinations = len(horizons) * len(thresholds) * len(entry_probs)

    # Build per-symbol data once (precomputed NumPy arrays for the inner loop).
    # Windows are sorted by current_close_time so the sequential next_entry_time
    # lock in the worker reflects the true chronological order.
    step_ms = base_tf_min * 60_000
    sym_data: dict[str, dict] = {}
    for sym in symbols:
        sym_idx = [i for i, w in enumerate(windows) if w['symbol'] == sym]
        sym_windows = [windows[i] for i in sym_idx]
        order = sorted(range(len(sym_windows)),
                       key=lambda k: int(sym_windows[k]['current_close_time']))
        sym_windows = [sym_windows[k] for k in order]
        sym_pred    = pred[np.asarray(sym_idx, dtype=np.int64)][np.asarray(order, dtype=np.int64)]

        current_times  = np.asarray([int(w['current_close_time']) for w in sym_windows], dtype=np.int64)
        entry_prices   = np.asarray([float(w['current_close'])    for w in sym_windows], dtype=np.float64)
        horizon_times  = np.asarray([w['future_close_times'] for w in sym_windows], dtype=np.int64)
        horizon_closes = np.asarray([w['future_closes']      for w in sym_windows], dtype=np.float64)

        ohlcv = ohlcv_by_symbol[sym]
        o_times = ohlcv['times']
        # entry_idx[i] = first OHLCV index strictly after current_time (t = ct + step_ms)
        entry_idx = np.searchsorted(o_times, current_times + step_ms, side='left').astype(np.int64)

        sym_data[sym] = {
            'symbol':         sym,
            'current_times':  current_times,
            'entry_prices':   entry_prices,
            'horizon_times':  horizon_times,
            'horizon_closes': horizon_closes,
            'pred':           np.ascontiguousarray(sym_pred),
            'ohlcv_times':    o_times,
            'ohlcv_highs':    ohlcv['highs'],
            'ohlcv_lows':     ohlcv['lows'],
            'entry_idx':      entry_idx,
        }
    timer.checkpoint('prepared per-symbol arrays')

    # EV-based strategy: extract quantile predictions and compute combos
    pred_q = results.get('pred_quantile')  # [N, n_horizons, n_quantiles, n_dir]
    if pred_q is None:
        logger.error('[Backtest] Quantile predictions not available for EV-based strategy.')
        return

    symbol_rows: dict[str, list[dict]] = {sym: [] for sym in symbols}
    total_combos = 0

    print('\n' + '=' * 70)
    print('EV-BASED ADAPTIVE STRATEGY — TOP 20 BY EXPECTED VALUE')
    print('=' * 70)
    print(f'Test range: {_format_ts(test_start)} -> {_format_ts(test_end)} UTC')
    print(f'Symbols: {", ".join(symbols)}')
    print(f'Entry prob grid: {entry_probs}')
    print(f'Strategy: Dynamic TP/SL from quantiles, Kelly Criterion sizing')

    for sym in symbols:
        sym_idx = [i for i, w in enumerate(windows) if w['symbol'] == sym]
        sym_pred_q = pred_q[np.asarray(sym_idx, dtype=np.int64)]

        for hi, horizon_h in enumerate(horizons):
            for ti, threshold_t in enumerate(thresholds):
                for entry_prob in entry_probs:
                    total_combos += 1
                    row = _simulate_combo_ev(
                        sym_data[sym], sym, hi, ti, entry_prob, direction,
                        horizon_h, threshold_t, sym_pred_q, pred[:, :, :, :],
                        q_idx_tp=4, q_idx_sl=0  # p90 TP, p10 SL
                    )
                    if row:
                        symbol_rows[sym].append(row)

    timer.checkpoint(f'computed {total_combos} EV-based combos')

    all_rows = [row for sym in symbols for row in symbol_rows[sym]]
    n_with_trades = len(all_rows)
    print(f'\nCombinations tested: {len(symbols) * total_combinations} total | '
          f'{n_with_trades} with trades | {len(symbols) * total_combinations - n_with_trades} empty (no signals)')

    _print_top100_ev(all_rows, title='ALL SYMBOLS — EV-BASED STRATEGY TOP 100 BY SQN')

    timer.checkpoint('aggregated and printed results')
    print(f'\n  * = fewer than {MIN_TRADES_VALID} trades (statistically low)')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='QuantileNet Backtest')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Path to .pt checkpoint (default: best in checkpoints/)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    config = load_config()
    device = get_device(config)
    main_timer = StageTimer('BacktestMain')

    ckpt_dir   = os.path.join(os.path.dirname(__file__), 'checkpoints')
    checkpoint = args.checkpoint or find_best_checkpoint(ckpt_dir)
    run_dir    = os.path.dirname(os.path.abspath(checkpoint))

    model = load_model(config, device, checkpoint)

    logger.info('Loading test windows...')
    test_windows = load_test_windows_cached(config)
    logger.info(f'Total test windows: {len(test_windows)}')
    main_timer.checkpoint('loaded test windows')

    if not test_windows:
        logger.error('No test windows — check walk_forward config (test_months).')
        sys.exit(1)

    logger.info('Running inference...')
    results = run_inference(model, test_windows, config, device)
    main_timer.checkpoint('completed inference')

    calibration_json = os.path.join(run_dir, 'calibration.json')
    calibration_report(results, config, save_path=calibration_json)
    main_timer.checkpoint('generated calibration report')

    logger.info('Loading OHLCV data for TP/SL simulation...')
    ohlcv_by_symbol, base_tf_min = load_ohlcv(config)
    main_timer.checkpoint('loaded OHLCV')

    brute_force_tp_sl_top20(results, test_windows, config, ohlcv_by_symbol, base_tf_min)
    main_timer.checkpoint('completed brute-force simulation')

    print('\n[Backtest] Done.')


if __name__ == '__main__':
    main()
