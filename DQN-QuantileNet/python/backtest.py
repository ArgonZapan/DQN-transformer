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


# ── Backtest config ────────────────────────────────────────────────────────────

# Defaults — overridden by [backtest] section of config.toml.
_BACKTEST_DEFAULTS = {
    'strategy':           'ev_threshold',
    'asymmetry_ratio':    1.5,
    'commission':         0.00075,
    'slippage_bps':       1.0,
    'min_trades_valid':   20,
    'sequential_lock':    True,
    'same_bar_priority':  'sl',
    'tp_quantile':        0.90,
    'sl_quantile':        0.10,
    'sl_min_pct':         0.5,
    'tp_max_pct':         50.0,
    'sl_max_pct':         50.0,
}


def _backtest_cfg(config: dict) -> dict:
    """Merge [backtest] section over defaults."""
    cfg = dict(_BACKTEST_DEFAULTS)
    cfg.update(config.get('backtest', {}) or {})
    return cfg


# ── EV-Based Strategy Helpers ──────────────────────────────────────────────────

def calculate_ev(p_win: float, p_loss: float, return_win: float, return_loss: float) -> float:
    """Expected Value = P(win) * return_win + P(loss) * return_loss (in %)."""
    return (p_win * return_win) + (p_loss * return_loss)


def calculate_kelly(p_win: float, avg_win: float, avg_loss: float) -> float:
    """Kelly Criterion: f = (p*b - q) / b, where p=win%, b=ratio, q=loss%.

    Returns fraction of capital to risk [0, 0.25] (capped for safety).
    Returns 0.0 when win or loss legs are missing — Kelly is undefined
    without at least one of each.
    """
    if avg_loss <= 0 or avg_win <= 0 or p_win <= 0 or p_win >= 1:
        return 0.0
    b = avg_win / avg_loss  # odds ratio
    q = 1 - p_win
    kelly_f = (p_win * b - q) / b
    return min(max(kelly_f, 0.0), 0.25)  # clamp [0, 0.25]


def resolve_quantile_indices(quantiles: list[float], tp_q: float, sl_q: float) -> tuple[int, int]:
    """Map TP/SL quantile values to indices in config['prediction'].quantiles.

    Raises ValueError if a target quantile is not present.
    """
    def _find(target: float) -> int:
        for i, q in enumerate(quantiles):
            if abs(q - target) < 1e-9:
                return i
        raise ValueError(
            f'Quantile {target} not in config quantiles {quantiles}. '
            f'Adjust [backtest] tp_quantile/sl_quantile or [prediction] quantiles.'
        )
    return _find(tp_q), _find(sl_q)


def get_quantile_tp_sl(quantiles_up: np.ndarray, quantiles_down: np.ndarray,
                       q_idx_tp: int, q_idx_sl: int,
                       direction: str) -> tuple[float, float]:
    """Extract TP and SL from MFE/MAE quantile predictions.

    Args:
        quantiles_up:   [n_quantiles] up-excursion quantiles (dir 0), always positive
        quantiles_down: [n_quantiles] down-excursion quantiles (dir 1), always positive
        q_idx_tp: optimistic quantile index (e.g. 4 for p90)
        q_idx_sl: pessimistic quantile index (e.g. 0 for p10)
        direction: 'long' or 'short'

    Returns:
        (tp_pct, sl_pct) as positive percentages
    """
    if direction == 'long':
        tp_pct = float(quantiles_up[q_idx_tp])   * 100.0  # how far up price can go
        sl_pct = float(quantiles_down[q_idx_sl]) * 100.0  # how far down price can go
    else:  # short
        tp_pct = float(quantiles_down[q_idx_tp]) * 100.0  # how far down price can go
        sl_pct = float(quantiles_up[q_idx_sl])   * 100.0  # how far up price can go
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
                  device: torch.device,
                  num_workers: int | None = None) -> dict[str, np.ndarray]:
    """Run model on all windows. Returns dict of raw predictions and labels.

    num_workers: override DataLoader workers (e.g. 0 for small live batches
    where worker spawn overhead dominates).
    """
    active_tfs  = model.active_tfs
    tf_seq_lens = model.tf_seq_lens
    dataset     = QuantileDataset(windows, config, active_tfs, tf_seq_lens)
    train_cfg   = config.get('training', {})
    if num_workers is not None:
        loader_workers = max(0, int(num_workers))
    else:
        loader_workers = int(train_cfg.get('backtest_inference_workers', 0))
        if loader_workers <= 0:
            if device.type == 'cuda':
                loader_workers = min(8, os.cpu_count() or 1)
            else:
                loader_workers = max(1, min(4, os.cpu_count() or 1))
    # `persistent_workers=False` — backtest iterates the loader exactly once,
    # so persistent workers add no benefit and on Windows (spawn) leave 8
    # processes pinning a fork-copy of `windows` until GC eventually triggers.
    # When called repeatedly from per-epoch backtest, that pile-up was a real
    # RAM peak.
    infer_bs = int(train_cfg.get('inference_batch_size', 256))
    loader   = DataLoader(dataset, batch_size=infer_bs, shuffle=False,
                             collate_fn=collate_fn,
                             num_workers=loader_workers,
                             pin_memory=device.type == 'cuda',
                             persistent_workers=False)

    logger.info(
        f'[Backtest] Inference loader: batch_size={infer_bs} num_workers={loader_workers} '
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
            # ThresholdHead emits raw logits; convert to probabilities here so
            # downstream code (calibration_report, brute_force_tp_sl_top20,
            # live_writer) keeps the previous (0, 1) contract.
            pred_t_list.append(torch.sigmoid(outputs['threshold']).cpu().numpy())
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
    dir_labels = ['up', 'down']

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


def compute_q_score(rows: list[dict], test_start_ms: int, test_end_ms: int) -> dict:
    """Compute Q-score for a family of strategy parameterizations.

    Q = clip(Y³ × min(√freq, 10) × tanh(mean_Sharpe), -10, 10)

    Y          — fraction of combos with total_return > 0
    freq       — mean_N_tx × 3 / period_months  (saturates: √100 = 10)
    mean_Sharpe — mean Sharpe across all combos (tanh-dampened)
    """
    if not rows:
        return {'q_score': 0.0, 'Y': 0.0, 'freq': 0.0, 'mean_sharpe': 0.0,
                'mean_n_trades': 0.0, 'period_months': 0.0, 'n_combos': 0}

    ms_per_month  = 30.4375 * 24 * 3600 * 1000
    period_months = max((test_end_ms - test_start_ms) / ms_per_month, 1e-6)

    Y            = float(sum(1 for r in rows if r['total'] > 0) / len(rows))
    mean_n_tx    = float(np.mean([r['n_trades'] for r in rows]))
    freq         = mean_n_tx * 3.0 / period_months
    mean_sharpe  = float(np.mean([r['sharpe'] for r in rows]))

    freq_factor  = min(float(np.sqrt(freq)), 10.0)
    sharpe_factor = float(np.tanh(mean_sharpe))
    raw          = (Y ** 3) * freq_factor * sharpe_factor
    q_score      = float(np.clip(raw, -10.0, 10.0))

    return {
        'q_score':      q_score,
        'Y':            Y,
        'freq':         freq,
        'freq_factor':  freq_factor,
        'mean_sharpe':  mean_sharpe,
        'mean_n_trades': mean_n_tx,
        'period_months': period_months,
        'n_combos':     len(rows),
    }


def print_q_score_report(qs: dict, label: str = ''):
    """Print a formatted Q-score summary."""
    tag = f' [{label}]' if label else ''
    bar_len = max(0, min(40, int(round((qs['q_score'] + 10) / 20 * 40))))
    bar = '#' * bar_len + '.' * (40 - bar_len)
    print(f'\n{"=" * 70}')
    print(f'Q-SCORE{tag}')
    print(f'{"=" * 70}')
    print(f'  Q = {qs["q_score"]:+.3f}  [{bar}]  (range -10 … +10)')
    print(f'  Y (profitable combos) : {qs["Y"] * 100:.1f}%  of {qs["n_combos"]} combos  →  Y³ = {qs["Y"]**3:.4f}')
    print(f'  freq                  : {qs["freq"]:.1f} tx/3mo  →  √freq factor = {qs["freq_factor"]:.2f} (cap 10)')
    print(f'  mean Sharpe           : {qs["mean_sharpe"]:.3f}  →  tanh = {np.tanh(qs["mean_sharpe"]):.4f}')
    print(f'  period                : {qs["period_months"]:.1f} months  |  mean trades/combo = {qs["mean_n_trades"]:.0f}')
    print(f'{"=" * 70}')


def _simulate_exit(
    trade_dir: int,
    entry_px: float,
    tp_pct: float,
    sl_pct: float,
    h_time: int,
    h_close: float,
    start: int,
    o_times: np.ndarray,
    o_highs: np.ndarray,
    o_lows: np.ndarray,
    cost: float,
    sl_first_on_tie: bool,
) -> tuple[str, int, float]:
    """Walk forward bar-by-bar from `start` until h_time, deciding TP/SL/horizon exit.

    Returns (exit_type, exit_time_ms, return_after_cost).
    """
    tp_mult = tp_pct / 100.0
    sl_mult = sl_pct / 100.0
    end = int(np.searchsorted(o_times, h_time, side='right'))

    if end > start:
        highs = o_highs[start:end]
        lows  = o_lows[start:end]
        if trade_dir > 0:  # long
            sl_lvl = entry_px * (1.0 - sl_mult)
            tp_lvl = entry_px * (1.0 + tp_mult)
            sl_hits = lows  <= sl_lvl
            tp_hits = highs >= tp_lvl
        else:  # short
            sl_lvl = entry_px * (1.0 + sl_mult)
            tp_lvl = entry_px * (1.0 - tp_mult)
            sl_hits = highs >= sl_lvl
            tp_hits = lows  <= tp_lvl

        sl_any = sl_hits.any()
        tp_any = tp_hits.any()
        sl_i = int(sl_hits.argmax()) if sl_any else -1
        tp_i = int(tp_hits.argmax()) if tp_any else -1
        sl_wins = sl_any and (not tp_any
                              or (sl_i < tp_i)
                              or (sl_i == tp_i and sl_first_on_tie))
        if sl_wins:
            return 'sl', int(o_times[start + sl_i]), -sl_mult - cost
        if tp_any:
            return 'tp', int(o_times[start + tp_i]), tp_mult - cost

    ret = ((h_close - entry_px) / entry_px if trade_dir > 0
           else (entry_px - h_close) / entry_px) - cost
    return 'hz', int(h_time), ret


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
    bt_cfg: dict,
) -> list[dict]:
    """EV-based trading with dynamic TP/SL from quantiles.

    Strategy:
      1. Entry: threshold probability >= entry_prob
      2. TP/SL: dynamic from quantiles (tp_quantile / sl_quantile from bt_cfg)
      3. Cost: 2 x commission + 2 x slippage applied to every fill
      4. Sequential lock (configurable): blocks new entries until previous closes
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

    commission       = float(bt_cfg['commission'])
    slippage         = float(bt_cfg['slippage_bps']) / 10000.0
    cost             = 2.0 * (commission + slippage)
    sl_min           = float(bt_cfg['sl_min_pct'])
    tp_max           = float(bt_cfg['tp_max_pct'])
    sl_max           = float(bt_cfg['sl_max_pct'])
    sequential_lock  = bool(bt_cfg['sequential_lock'])
    same_bar_pri     = str(bt_cfg['same_bar_priority']).lower()
    sl_first_on_tie  = same_bar_pri != 'tp'  # default = pessimistic (sl wins ties)

    trades: list[dict] = []
    next_entry_time = -1  # sequential lock: no new entry before previous trade closes

    for idx in sig_indices:
        ct = current_times[idx]
        if sequential_lock and ct < next_entry_time:
            continue
        trade_dir = int(dirs_all[idx])
        entry_px  = entry_prices[idx]
        h_time    = horizon_times[idx, hi]
        h_close   = horizon_closes[idx, hi]

        # dir 0 = up excursion (MFE), dir 1 = down excursion (MAE) — both always positive
        quantiles_up   = pred_q[idx, hi, :, 0]
        quantiles_down = pred_q[idx, hi, :, 1]
        tp_pct, sl_pct = get_quantile_tp_sl(quantiles_up, quantiles_down, q_idx_tp, q_idx_sl,
                                            'long' if trade_dir > 0 else 'short')

        sl_pct = min(max(sl_pct, sl_min), sl_max)
        tp_pct = min(tp_pct, tp_max)

        start = int(entry_idx_all[idx])
        exit_type, exit_t, ret = _simulate_exit(
            trade_dir, entry_px, tp_pct, sl_pct, h_time, h_close,
            start, o_times, o_highs, o_lows, cost, sl_first_on_tie,
        )

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


def _compute_combo_metrics(trades: list[dict]) -> dict | None:
    """Single source of truth for combo metrics.

    Uses `_timeline_stats` for compound-equity-based total/sharpe/MDD,
    so dashboard numbers match what `_print_top100_ev` prints.

    Notes:
      - p_win is computed AFTER cost (a 0% gross trade with cost > 0 counts as loss).
      - Kelly returns 0 when win or loss legs are missing.
    """
    if not trades:
        return None

    # Compound-based timeline stats (the correct ones).
    wins, avg, total, sharpe, sqn, max_dd, tp_rate, sl_rate, hz_rate = _timeline_stats(trades)

    returns = np.asarray([t['return'] for t in trades], dtype=np.float64)
    p_win = float(wins)
    p_loss = 1.0 - p_win

    win_returns  = returns[returns > 0]
    loss_returns = returns[returns < 0]
    avg_win_pct  = float(win_returns.mean()  * 100) if win_returns.size  else 0.0
    avg_loss_pct = float(abs(loss_returns.mean()) * 100) if loss_returns.size else 0.0

    if win_returns.size and loss_returns.size:
        ev    = float(calculate_ev(p_win, p_loss, avg_win_pct, -avg_loss_pct))
        kelly = float(calculate_kelly(p_win, avg_win_pct, avg_loss_pct))
    else:
        ev    = float(avg * 100)  # all wins or all losses — EV = mean return per trade
        kelly = 0.0

    tp_pct_sample = float(np.mean([t['tp_pct'] for t in trades]))
    sl_pct_sample = float(np.mean([t['sl_pct'] for t in trades]))

    return {
        'tp_pct':       tp_pct_sample,
        'sl_pct':       sl_pct_sample,
        'n_trades':     len(trades),
        'p_win':        p_win,
        'avg_win':      avg_win_pct,
        'avg_loss':     avg_loss_pct,
        'total_return': float(total * 100),
        'total':        float(total),
        'sharpe':       float(sharpe),
        'sqn':          float(sqn),
        'max_dd':       float(max_dd),
        'tp_rate':      float(tp_rate),
        'sl_rate':      float(sl_rate),
        'hz_rate':      float(hz_rate),
        'avg':          float(avg),
        'ev':           ev,
        'kelly':        kelly,
    }


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
    q_idx_tp: int,
    q_idx_sl: int,
    bt_cfg: dict,
) -> dict | None:
    """Run EV-based combo: dynamic TP/SL from quantiles, rank by SQN."""
    trades = _collect_ev_based_trades(sd, hi, ti, entry_prob, direction,
                                       q_idx_tp, q_idx_sl, pred_quantile, pred_threshold,
                                       bt_cfg)
    metrics = _compute_combo_metrics(trades)
    if metrics is None:
        return None

    return {
        'symbol':     symbol,
        'horizon':    horizon_hours,
        'threshold':  threshold_pct,
        'direction':  direction,
        'entry_prob': entry_prob,
        **metrics,
        'trades':     trades,
    }


def _collect_quantile_asymmetry_trades(
    sd: dict,
    hi: int,
    q_idx: int,
    asymmetry_ratio: float,
    direction: str,
    pred_quantile: np.ndarray,
    bt_cfg: dict,
) -> list[dict]:
    """Quantile-asymmetry strategy.

    For the given quantile q (q_idx), compare predicted up-excursion (MFE) and
    down-excursion (MAE):
      - if up >= ratio * down → LONG: TP=up (farther), SL=down (nearer)
      - if down >= ratio * up → SHORT: TP=down (farther), SL=up (nearer)
      - else                  → no trade
    `direction` filter: 'long' / 'short' / 'both' restricts which side may fire.
    """
    pred_q          = pred_quantile      # [N, H, n_quantiles, D]
    current_times   = sd['current_times']
    entry_prices    = sd['entry_prices']
    horizon_times   = sd['horizon_times']
    horizon_closes  = sd['horizon_closes']
    entry_idx_all   = sd['entry_idx']
    o_times         = sd['ohlcv_times']
    o_highs         = sd['ohlcv_highs']
    o_lows          = sd['ohlcv_lows']

    up_q   = pred_q[:, hi, q_idx, 0]   # MFE (up), positive %
    down_q = pred_q[:, hi, q_idx, 1]   # MAE (down), positive %

    # ratio >= 1 → mutually exclusive by construction; require both sides to
    # be non-degenerate so we don't fire on (0, 0) noise.
    valid = np.maximum(up_q, down_q) > 1e-9
    long_mask  = valid & (up_q   >= asymmetry_ratio * down_q) & (up_q   > down_q)
    short_mask = valid & (down_q >= asymmetry_ratio * up_q  ) & (down_q > up_q)

    if direction == 'long':
        short_mask[:] = False
    elif direction == 'short':
        long_mask[:] = False

    signal_mask = long_mask | short_mask
    dirs_all = np.where(long_mask, 1, np.where(short_mask, -1, 0)).astype(np.int8)

    sig_indices = np.nonzero(signal_mask)[0]
    if sig_indices.size == 0:
        return []

    commission       = float(bt_cfg['commission'])
    slippage         = float(bt_cfg['slippage_bps']) / 10000.0
    cost             = 2.0 * (commission + slippage)
    sl_min           = float(bt_cfg['sl_min_pct'])
    tp_max           = float(bt_cfg['tp_max_pct'])
    sl_max           = float(bt_cfg['sl_max_pct'])
    sequential_lock  = bool(bt_cfg['sequential_lock'])
    same_bar_pri     = str(bt_cfg['same_bar_priority']).lower()
    sl_first_on_tie  = same_bar_pri != 'tp'

    trades: list[dict] = []
    next_entry_time = -1

    for idx in sig_indices:
        ct = current_times[idx]
        if sequential_lock and ct < next_entry_time:
            continue
        trade_dir = int(dirs_all[idx])
        entry_px  = entry_prices[idx]
        h_time    = horizon_times[idx, hi]
        h_close   = horizon_closes[idx, hi]

        u = float(up_q[idx])   * 100.0
        d = float(down_q[idx]) * 100.0
        if trade_dir > 0:   # long: TP=up (farther), SL=down (nearer)
            tp_pct, sl_pct = u, d
        else:               # short: TP=down (farther), SL=up (nearer)
            tp_pct, sl_pct = d, u

        sl_pct = min(max(sl_pct, sl_min), sl_max)
        tp_pct = min(tp_pct, tp_max)

        start = int(entry_idx_all[idx])
        exit_type, exit_t, ret = _simulate_exit(
            trade_dir, entry_px, tp_pct, sl_pct, h_time, h_close,
            start, o_times, o_highs, o_lows, cost, sl_first_on_tie,
        )

        trades.append({
            'symbol':     sd['symbol'],
            'entry_time': int(ct),
            'exit_time':  exit_t,
            'return':     float(ret),
            'exit_type':  exit_type,
            'tp_pct':     float(tp_pct),
            'sl_pct':     float(sl_pct),
        })
        next_entry_time = exit_t

    return trades


def _simulate_combo_quantile_asymmetry(
    sd: dict,
    symbol: str,
    hi: int,
    q_idx: int,
    quantile_value: float,
    asymmetry_ratio: float,
    direction: str,
    horizon_hours: float,
    pred_quantile: np.ndarray,
    bt_cfg: dict,
) -> dict | None:
    trades = _collect_quantile_asymmetry_trades(
        sd, hi, q_idx, asymmetry_ratio, direction, pred_quantile, bt_cfg,
    )
    metrics = _compute_combo_metrics(trades)
    if metrics is None:
        return None

    return {
        'symbol':          symbol,
        'horizon':         horizon_hours,
        'quantile':        quantile_value,
        'asymmetry_ratio': asymmetry_ratio,
        'direction':       direction,
        **metrics,
        'trades':          trades,
    }


def _print_top100_quantile_asymmetry(rows: list[dict], title: str, min_trades_valid: int) -> list[dict]:
    any_valid = any(r['n_trades'] >= min_trades_valid for r in rows)
    display   = [r for r in rows if r['n_trades'] >= min_trades_valid] if any_valid else rows
    top100    = sorted(display, key=lambda r: r['sqn'], reverse=True)[:100]

    print(f'\n{"=" * 122}')
    print(f'{title}')
    print(f'{"=" * 122}')
    print(f'{"#":>3} {"Symbol":>8} {"H":>5} {"Q":>5} {"Dir":>5} {"N":>6} {"P(Win)":>7} {"Avg+%":>7} {"Avg-%":>7} '
          f'{"Total%":>8} {"Sharpe":>7} {"SQN":>7} {"EV%":>7} {"Kelly":>7} {"TP%":>5} {"SL%":>5}')
    print('-' * 122)

    for rank, row in enumerate(top100, 1):
        low_flag = ' *' if row['n_trades'] < min_trades_valid else '  '
        print(
            f'{rank:>3} {row["symbol"]:>8} {row["horizon"]:>4}h {row["quantile"]:>4.2f} '
            f'{row["direction"]:>5} {row["n_trades"]:>5}{low_flag} '
            f'{row["p_win"] * 100:>6.1f}% {row["avg_win"]:>6.2f}% {row["avg_loss"]:>6.2f}% '
            f'{row["total_return"]:>7.2f}% {row["sharpe"]:>7.2f} {row["sqn"]:>7.2f} '
            f'{row["ev"]:>6.2f}% {row["kelly"]:>6.3f} '
            f'{row["tp_pct"]:>4.1f}% {row["sl_pct"]:>4.1f}%'
        )
    return top100


def _print_top100_ev(rows: list[dict], title: str, min_trades_valid: int) -> list[dict]:
    """Print top 100 by SQN across all symbols."""
    any_valid = any(r['n_trades'] >= min_trades_valid for r in rows)
    display   = [r for r in rows if r['n_trades'] >= min_trades_valid] if any_valid else rows
    top100    = sorted(display, key=lambda r: r['sqn'], reverse=True)[:100]

    print(f'\n{"=" * 122}')
    print(f'{title}')
    print(f'{"=" * 122}')
    print(f'{"#":>3} {"Symbol":>8} {"H":>5} {"Thr%":>5} {"Entry%":>8} {"N":>6} {"P(Win)":>7} {"Avg+%":>7} {"Avg-%":>7} '
          f'{"Total%":>8} {"Sharpe":>7} {"SQN":>7} {"EV%":>7} {"Kelly":>7} {"TP%":>5} {"SL%":>5}')
    print('-' * 122)

    for rank, row in enumerate(top100, 1):
        low_flag = ' *' if row['n_trades'] < min_trades_valid else '  '
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


def _build_sym_data(
    symbols: list[str],
    windows: list,
    pred_threshold: np.ndarray | None,
    pred_quantile: np.ndarray | None,
    ohlcv_by_symbol: dict[str, dict],
    base_tf_min: int,
) -> dict[str, dict]:
    """Build per-symbol precomputed arrays. Windows sorted by entry time so
    the sequential lock in the inner loop sees true chronological order."""
    step_ms = base_tf_min * 60_000
    sym_data: dict[str, dict] = {}
    for sym in symbols:
        sym_idx = [i for i, w in enumerate(windows) if w['symbol'] == sym]
        sym_windows = [windows[i] for i in sym_idx]
        order = sorted(range(len(sym_windows)),
                       key=lambda k: int(sym_windows[k]['current_close_time']))
        sym_windows = [sym_windows[k] for k in order]
        order_arr = np.asarray(order, dtype=np.int64)
        idx_arr   = np.asarray(sym_idx, dtype=np.int64)

        sym_pred   = pred_threshold[idx_arr][order_arr] if pred_threshold is not None else None
        sym_pred_q = pred_quantile [idx_arr][order_arr] if pred_quantile  is not None else None

        current_times  = np.asarray([int(w['current_close_time']) for w in sym_windows], dtype=np.int64)
        entry_prices   = np.asarray([float(w['current_close'])    for w in sym_windows], dtype=np.float64)
        horizon_times  = np.asarray([w['future_close_times'] for w in sym_windows], dtype=np.int64)
        horizon_closes = np.asarray([w['future_closes']      for w in sym_windows], dtype=np.float64)

        ohlcv   = ohlcv_by_symbol[sym]
        o_times = ohlcv['times']
        entry_idx = np.searchsorted(o_times, current_times + step_ms, side='left').astype(np.int64)

        sym_data[sym] = {
            'symbol':         sym,
            'current_times':  current_times,
            'entry_prices':   entry_prices,
            'horizon_times':  horizon_times,
            'horizon_closes': horizon_closes,
            'pred':           np.ascontiguousarray(sym_pred)   if sym_pred   is not None else None,
            'pred_q':         np.ascontiguousarray(sym_pred_q) if sym_pred_q is not None else None,
            'ohlcv_times':    o_times,
            'ohlcv_highs':    ohlcv['highs'],
            'ohlcv_lows':     ohlcv['lows'],
            'entry_idx':      entry_idx,
        }
    return sym_data


def _run_ev_threshold_strategy(
    results: dict, windows: list, config: dict,
    ohlcv_by_symbol: dict, base_tf_min: int, bt_cfg: dict, timer: StageTimer,
):
    if 'pred_threshold' not in results:
        print('\n[Backtest] Threshold predictions not available — ev_threshold strategy needs them.')
        return None
    pred_q = results.get('pred_quantile')
    if pred_q is None:
        logger.error('[Backtest] Quantile predictions not available for ev_threshold strategy.')
        return None

    pred_cfg   = config['prediction']
    horizons   = pred_cfg['horizons_hours']
    thresholds = pred_cfg.get('thresholds_pct', [])
    quantiles  = pred_cfg.get('quantiles', [])
    direction  = pred_cfg['direction']
    symbols    = sorted({w['symbol'] for w in windows})
    test_start = min(int(w['current_close_time']) for w in windows)
    test_end   = max(int(w['future_close_times'][-1]) for w in windows)

    q_idx_tp, q_idx_sl = resolve_quantile_indices(
        quantiles, float(bt_cfg['tp_quantile']), float(bt_cfg['sl_quantile'])
    )

    pred = results['pred_threshold']

    print(f'\nPred threshold stats: min={pred.min():.4f}  max={pred.max():.4f}'
          f'  mean={pred.mean():.4f}  p50={np.median(pred):.4f}'
          f'  p90={np.percentile(pred, 90):.4f}  p95={np.percentile(pred, 95):.4f}')

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

    total_combinations = len(horizons) * len(thresholds) * len(entry_probs)

    sym_data = _build_sym_data(symbols, windows, pred, pred_q, ohlcv_by_symbol, base_tf_min)
    timer.checkpoint('prepared per-symbol arrays')

    symbol_rows: dict[str, list[dict]] = {sym: [] for sym in symbols}
    total_combos = 0

    print('\n' + '=' * 70)
    print('EV-BASED ADAPTIVE STRATEGY — TOP 100 BY SQN')
    print('=' * 70)
    print(f'Test range: {_format_ts(test_start)} -> {_format_ts(test_end)} UTC')
    print(f'Symbols: {", ".join(symbols)}')
    print(f'Entry prob grid: {entry_probs}')
    print(f'TP quantile: {bt_cfg["tp_quantile"]}  SL quantile: {bt_cfg["sl_quantile"]}  '
          f'commission: {bt_cfg["commission"]}  slippage: {bt_cfg["slippage_bps"]}bps')
    print(f'sequential_lock={bt_cfg["sequential_lock"]}  same_bar_priority={bt_cfg["same_bar_priority"]}')

    for sym in symbols:
        sd = sym_data[sym]
        for hi, horizon_h in enumerate(horizons):
            for ti, threshold_t in enumerate(thresholds):
                for entry_prob in entry_probs:
                    total_combos += 1
                    row = _simulate_combo_ev(
                        sd, sym, hi, ti, entry_prob, direction,
                        horizon_h, threshold_t,
                        sd['pred_q'], sd['pred'],
                        q_idx_tp, q_idx_sl, bt_cfg,
                    )
                    if row:
                        symbol_rows[sym].append(row)

    timer.checkpoint(f'computed {total_combos} EV-based combos')

    all_rows = [row for sym in symbols for row in symbol_rows[sym]]
    n_with_trades = len(all_rows)
    print(f'\nCombinations tested: {len(symbols) * total_combinations} total | '
          f'{n_with_trades} with trades | {len(symbols) * total_combinations - n_with_trades} empty (no signals)')

    min_trades_valid = int(bt_cfg['min_trades_valid'])
    _print_top100_ev(all_rows, 'ALL SYMBOLS — EV-BASED STRATEGY TOP 100 BY SQN', min_trades_valid)

    timer.checkpoint('aggregated and printed results')
    print(f'\n  * = fewer than {min_trades_valid} trades (statistically low)')

    qs = compute_q_score(all_rows, test_start, test_end)
    print_q_score_report(qs, label='ALL SYMBOLS · ev_threshold')
    for sym in symbols:
        sym_rows = symbol_rows[sym]
        if sym_rows:
            qs_sym = compute_q_score(sym_rows, test_start, test_end)
            print_q_score_report(qs_sym, label=sym)

    return all_rows, {
        'strategy':         'ev_threshold',
        'test_start':       test_start,
        'test_end':         test_end,
        'symbols':          list(symbols),
        'entry_probs':      entry_probs,
        'min_trades_valid': min_trades_valid,
        'q_score':          qs,
    }


def _run_quantile_asymmetry_strategy(
    results: dict, windows: list, config: dict,
    ohlcv_by_symbol: dict, base_tf_min: int, bt_cfg: dict, timer: StageTimer,
):
    pred_q = results.get('pred_quantile')
    if pred_q is None:
        logger.error('[Backtest] Quantile predictions not available — quantile_asymmetry strategy needs them.')
        return None

    pred_cfg   = config['prediction']
    horizons   = pred_cfg['horizons_hours']
    quantiles  = pred_cfg.get('quantiles', [])
    direction  = pred_cfg['direction']
    symbols    = sorted({w['symbol'] for w in windows})
    test_start = min(int(w['current_close_time']) for w in windows)
    test_end   = max(int(w['future_close_times'][-1]) for w in windows)
    asymmetry_ratio = float(bt_cfg['asymmetry_ratio'])

    # Brute-force only quantiles >= 0.25; the lower ones are too tight to
    # produce a meaningful asymmetry signal (and likely always fire).
    q_grid = [(qi, float(q)) for qi, q in enumerate(quantiles) if float(q) >= 0.25]
    if not q_grid:
        logger.error('[Backtest] No quantiles >= 0.25 in config.prediction.quantiles.')
        return None

    sym_data = _build_sym_data(symbols, windows, None, pred_q, ohlcv_by_symbol, base_tf_min)
    timer.checkpoint('prepared per-symbol arrays')

    symbol_rows: dict[str, list[dict]] = {sym: [] for sym in symbols}
    total_combinations = len(horizons) * len(q_grid)
    total_combos = 0

    print('\n' + '=' * 70)
    print('QUANTILE-ASYMMETRY STRATEGY — TOP 100 BY SQN')
    print('=' * 70)
    print(f'Test range: {_format_ts(test_start)} -> {_format_ts(test_end)} UTC')
    print(f'Symbols: {", ".join(symbols)}')
    print(f'Quantile grid: {[q for _, q in q_grid]}')
    print(f'Asymmetry ratio: {asymmetry_ratio}  direction: {direction}  '
          f'commission: {bt_cfg["commission"]}  slippage: {bt_cfg["slippage_bps"]}bps')
    print(f'sequential_lock={bt_cfg["sequential_lock"]}  same_bar_priority={bt_cfg["same_bar_priority"]}')

    for sym in symbols:
        sd = sym_data[sym]
        for hi, horizon_h in enumerate(horizons):
            for q_idx, q_val in q_grid:
                total_combos += 1
                row = _simulate_combo_quantile_asymmetry(
                    sd, sym, hi, q_idx, q_val, asymmetry_ratio, direction,
                    horizon_h, sd['pred_q'], bt_cfg,
                )
                if row:
                    symbol_rows[sym].append(row)

    timer.checkpoint(f'computed {total_combos} quantile-asymmetry combos')

    all_rows = [row for sym in symbols for row in symbol_rows[sym]]
    n_with_trades = len(all_rows)
    print(f'\nCombinations tested: {len(symbols) * total_combinations} total | '
          f'{n_with_trades} with trades | {len(symbols) * total_combinations - n_with_trades} empty (no signals)')

    min_trades_valid = int(bt_cfg['min_trades_valid'])
    _print_top100_quantile_asymmetry(
        all_rows, 'ALL SYMBOLS — QUANTILE-ASYMMETRY STRATEGY TOP 100 BY SQN', min_trades_valid,
    )

    timer.checkpoint('aggregated and printed results')
    print(f'\n  * = fewer than {min_trades_valid} trades (statistically low)')

    qs = compute_q_score(all_rows, test_start, test_end)
    print_q_score_report(qs, label='ALL SYMBOLS · quantile_asymmetry')
    for sym in symbols:
        sym_rows = symbol_rows[sym]
        if sym_rows:
            qs_sym = compute_q_score(sym_rows, test_start, test_end)
            print_q_score_report(qs_sym, label=sym)

    return all_rows, {
        'strategy':         'quantile_asymmetry',
        'test_start':       test_start,
        'test_end':         test_end,
        'symbols':          list(symbols),
        'asymmetry_ratio':  asymmetry_ratio,
        'quantile_grid':    [q for _, q in q_grid],
        'min_trades_valid': min_trades_valid,
        'q_score':          qs,
    }


def brute_force_tp_sl_top20(
    results: dict,
    windows: list,
    config: dict,
    ohlcv_by_symbol: dict[str, dict],
    base_tf_min: int,
):
    """Run the strategy selected via [backtest].strategy.

    Dispatches to:
      - ev_threshold       — entry by threshold prob, TP/SL from MFE/MAE quantiles
      - quantile_asymmetry — entry when one side's quantile is >= ratio x the other
    """
    bt_cfg   = _backtest_cfg(config)
    strategy = str(bt_cfg.get('strategy', 'ev_threshold')).lower()
    timer    = StageTimer('Backtest')

    if strategy == 'ev_threshold':
        return _run_ev_threshold_strategy(
            results, windows, config, ohlcv_by_symbol, base_tf_min, bt_cfg, timer,
        )
    if strategy == 'quantile_asymmetry':
        return _run_quantile_asymmetry_strategy(
            results, windows, config, ohlcv_by_symbol, base_tf_min, bt_cfg, timer,
        )
    raise ValueError(
        f'Unknown [backtest].strategy = {strategy!r}. '
        f'Expected "ev_threshold" or "quantile_asymmetry".'
    )


def export_strategy_top100(all_rows: list, meta: dict, save_path: str):
    """Serialize top-100 strategy combos (by SQN) to JSON for the dashboard."""
    min_trades_valid = int(meta.get('min_trades_valid', 20))
    sorted_rows = sorted(
        [r for r in all_rows if r['n_trades'] >= min_trades_valid],
        key=lambda r: r['sqn'], reverse=True
    )[:100]

    strategy = meta.get('strategy', 'ev_threshold')
    out = {
        'generated_at': datetime.now(UTC).isoformat(),
        'strategy':     strategy,
        'test_range': {
            'start': _format_ts(meta['test_start']),
            'end':   _format_ts(meta['test_end']),
        },
        'symbols':  meta['symbols'],
        'q_score':  meta.get('q_score'),
        'combos':   [],
    }
    if strategy == 'ev_threshold':
        out['entry_prob_grid'] = meta.get('entry_probs', [])
    elif strategy == 'quantile_asymmetry':
        out['quantile_grid']   = meta.get('quantile_grid', [])
        out['asymmetry_ratio'] = meta.get('asymmetry_ratio')

    for rank, row in enumerate(sorted_rows, start=1):
        combo = {
            'rank':         rank,
            'symbol':       row['symbol'],
            'horizon':      row['horizon'],
            'direction':    row['direction'],
            'threshold':    row.get('threshold'),
            'entry_prob':   row.get('entry_prob'),
            'quantile':     row.get('quantile'),
            'asymmetry_ratio': row.get('asymmetry_ratio'),
            'tp_pct':       row['tp_pct'],
            'sl_pct':       row['sl_pct'],
            'n_trades':     row['n_trades'],
            'p_win':        row['p_win'],
            'avg_win':      row['avg_win'],
            'avg_loss':     row['avg_loss'],
            'total_return': row['total_return'],
            'sharpe':       row['sharpe'],
            'sqn':          row['sqn'],
            'ev':           row['ev'],
            'kelly':        row['kelly'],
            'tp_rate':      row['tp_rate'],
            'sl_rate':      row['sl_rate'],
            'hz_rate':      row['hz_rate'],
            'max_dd':       row['max_dd'],
            'trades': [
                {
                    'symbol':     t['symbol'],
                    'entry_time': int(t['entry_time']),
                    'exit_time':  int(t['exit_time']),
                    'return':     round(float(t['return']), 6),
                    'exit_type':  t['exit_type'],
                    'tp_pct':     t['tp_pct'],
                    'sl_pct':     t['sl_pct'],
                }
                for t in row.get('trades', [])
            ],
        }
        out['combos'].append(combo)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    logger.info(f'[Strategy] Saved top-{len(out["combos"])} combos to {save_path}')


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

    bt_result = brute_force_tp_sl_top20(results, test_windows, config, ohlcv_by_symbol, base_tf_min)
    if bt_result is not None:
        all_rows, meta = bt_result
        strategy_json = os.path.join(run_dir, 'strategy_top100.json')
        export_strategy_top100(all_rows, meta, strategy_json)
    main_timer.checkpoint('completed brute-force simulation')

    print('\n[Backtest] Done.')


if __name__ == '__main__':
    main()
