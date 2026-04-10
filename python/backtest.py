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
import math
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


# ── Indicators — exact port from node/data/indicators.js ─────────────────────

def rolling_mean(values: list, window: int) -> list:
    """Partial window at start (identical to calculateSMA / rollingMean in JS)."""
    result = []
    for i in range(len(values)):
        sl = values[max(0, i - window + 1):i + 1]
        result.append(sum(sl) / len(sl))
    return result


def rolling_std(values: list, window: int) -> list:
    """Population std — divides by n, NOT n-1. Matches rollingStd in JS."""
    result = []
    for i in range(len(values)):
        sl = values[max(0, i - window + 1):i + 1]
        n = len(sl)
        if n < 2:
            result.append(0.0)
            continue
        mean = sum(sl) / n
        variance = sum((v - mean) ** 2 for v in sl) / n
        result.append(math.sqrt(variance))
    return result


def calculate_ema(values: list, span: int) -> list:
    """Seed = values[0]. Matches calculateEMA in JS."""
    if not values:
        return []
    k = 2.0 / (span + 1)
    ema = [values[0]]
    for v in values[1:]:
        ema.append((v - ema[-1]) * k + ema[-1])
    return ema


def calculate_rsi(closes: list, period: int = 14) -> list:
    """Wilder's RSI. Fills with 0.0 before index period+1. Matches calculateRSI in JS."""
    rsi = [0.0] * len(closes)
    if len(closes) < period + 1:
        return rsi

    avg_gain = avg_loss = 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        if d > 0:
            avg_gain += d
        else:
            avg_loss += abs(d)
    avg_gain /= period
    avg_loss /= period

    rsi[period] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        g = d if d > 0 else 0.0
        l = abs(d) if d < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
        rsi[i] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    return rsi


def calculate_macd_line(closes: list, fast: int = 12, slow: int = 26) -> list:
    """Returns MACD line (ema_fast - ema_slow). Matches calculateMACD in JS."""
    ema_f = calculate_ema(closes, fast)
    ema_s = calculate_ema(closes, slow)
    return [f - s for f, s in zip(ema_f, ema_s)]


# ── Feature Engineering — exact port of buildFeatures() from state.js ────────

def build_features(candles: list, norm_window: int = 20) -> list:
    """8 features per candle. Identical output to state.js buildFeatures()."""
    if not candles:
        return []

    closes  = [c['close']  for c in candles]
    opens   = [c['open']   for c in candles]
    highs   = [c['high']   for c in candles]
    lows    = [c['low']    for c in candles]
    volumes = [c['volume'] for c in candles]

    mean_c    = rolling_mean(closes, norm_window)
    std_c     = rolling_std(closes, norm_window)
    rsi       = calculate_rsi(closes)
    macd      = calculate_macd_line(closes)
    sma20     = rolling_mean(closes, 20)
    mean_vol  = rolling_mean(volumes, norm_window)

    features = []
    for i, c in enumerate(candles):
        cl, op, hi, lo, vo = closes[i], opens[i], highs[i], lows[i], volumes[i]
        prev = closes[i - 1] if i > 0 else cl

        features.append([
            (cl - mean_c[i]) / std_c[i]  if std_c[i]   != 0 else 0.0,  # normalizedClose
            (hi - lo) / cl               if cl          != 0 else 0.0,  # relativeRange
            (cl - op) / cl               if cl          != 0 else 0.0,  # candleDirection
            vo / mean_vol[i]             if mean_vol[i] != 0 else 0.0,  # normalizedVolume
            rsi[i] / 100.0,                                              # rsiNorm
            macd[i] / cl                 if cl          != 0 else 0.0,  # macdNorm
            (cl - prev) / prev           if prev        != 0 else 0.0,  # pctChange
            1.0 if cl > sma20[i] else 0.0,                              # aboveSma
        ])
    return features


# ── State Building — port of buildState() / getAlignedCandles() from state.js ─

def get_aligned_candles(all_candles: list, current_time_ms: int, num_candles: int) -> list:
    if num_candles <= 0:
        return []
    filtered = [c for c in all_candles if c['close_time'] <= current_time_ms]
    return filtered[-num_candles:]


def build_state(all_candles_per_tf: dict, current_time_ms: int, config: dict) -> dict:
    """
    Port of buildState() from state.js.
    Zeros at FRONT, real features at END (matching JS padding order).
    Returns {tf_name: np.ndarray([N, 8], float32)}.
    """
    tf_config = config['timeframes']
    norm_w = config['data']['normalization_window']
    tf_map = {
        'candles_1m': '1m', 'candles_15m': '15m',
        'candles_1h': '1h', 'candles_1d': '1d', 'candles_1w': '1w',
    }
    state = {}
    for cfg_key, tf_name in tf_map.items():
        num = tf_config.get(cfg_key, 0)
        if num <= 0:
            continue
        aligned = get_aligned_candles(all_candles_per_tf.get(tf_name, []), current_time_ms, num)
        feats = build_features(aligned, norm_w)
        pad = num - len(feats)
        combined = [[0.0] * 8] * pad + feats  # zeros first
        state[tf_name] = np.array(combined, dtype=np.float32)
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
            config: dict, device: str) -> int:
    """
    Greedy prediction. Tensor order matches Trainer.predict():
    sorted active config keys → ['candles_15m', 'candles_1d', 'candles_1h', 'candles_1m']
    """
    tf_keys = sorted(k for k, v in config['timeframes'].items() if v > 0)
    tensors = []
    for key in tf_keys:
        tf_name = key.replace('candles_', '')
        num = config['timeframes'][key]
        arr = state_dict.get(tf_name, np.zeros((num, 8), dtype=np.float32))
        tensors.append(torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(device))

    mask = torch.tensor([action_mask], dtype=torch.float32).to(device)
    q = model(tensors, action_mask=mask)
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
    """Gross PnL minus commissions and trade_penalty. Matches calculateReward() in reward.js."""
    gross = ((pos.close_price - pos.open_price) / pos.open_price
             if pos.side == 'LONG'
             else (pos.open_price - pos.close_price) / pos.open_price)
    r = config['reward']
    return gross - r['commission_open'] - r['commission_close'] - r['trade_penalty']


def action_mask_for(position) -> list:
    return [1, 1, 1, 0] if position is None else [0, 0, 1, 1]


# ── Per-Symbol Backtest ────────────────────────────────────────────────────────

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
    """Sequential OOS backtest for one symbol."""
    all_candles = load_all_candles(symbol, config)
    candles_1m = all_candles.get('1m', [])

    if not candles_1m:
        print(f"[Backtest] {symbol}: no 1m data — skipping")
        return {'trades': [], 'equity_curve': [0.0], 'daily_pnl': {}, 'total_steps': 0}

    split = int(len(candles_1m) * config['training']['train_data_fraction'])
    oos = candles_1m[split:]

    if last_n is not None and last_n < len(oos):
        oos = oos[-last_n:]
        print(f"[Backtest] {symbol}: OOS = last {len(oos)} candles (of {split} OOS total)")
    else:
        print(f"[Backtest] {symbol}: OOS = {len(oos)} steps (split at {split}/{len(candles_1m)})")

    trades, equity_curve, daily_pnl = [], [0.0], {}
    cumulative_pnl = 0.0
    position = None
    hold_minutes = 0

    for candle in oos:
        t_ms = candle['close_time']
        price = candle['close']

        state = build_state(all_candles, t_ms, config)
        mask = action_mask_for(position)
        action = predict(model, state, mask, config, device)

        if action == ACTION_LONG and position is None:
            if hold_minutes:
                print(f"  [{symbol}] HOLD {hold_minutes} min")
                hold_minutes = 0
            position = Position('LONG', price, t_ms)
            print(f"  [{symbol}] OPEN LONG  @ {price:.4f}  {_fmt_time(t_ms)}")

        elif action == ACTION_SHORT and position is None:
            if hold_minutes:
                print(f"  [{symbol}] HOLD {hold_minutes} min")
                hold_minutes = 0
            position = Position('SHORT', price, t_ms)
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
            hold_minutes = 0

        else:
            hold_minutes += 1

    if hold_minutes:
        print(f"  [{symbol}] HOLD {hold_minutes} min")

    # Force-close any open position at end of OOS
    if position is not None and oos:
        last = oos[-1]
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
            'daily_pnl': daily_pnl, 'total_steps': len(oos)}


# ── Metrics ────────────────────────────────────────────────────────────────────

def calc_metrics(trades: list, equity_curve: list, daily_pnl: dict) -> dict:
    """Full metrics suite. Reuses python/utils/metrics.py."""
    if not trades:
        return {'total_pnl': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0,
                'win_rate': 0.0, 'profit_factor': 0.0, 'avg_win': 0.0,
                'avg_loss': 0.0, 'total_trades': 0, 'max_consecutive_losses': 0}

    daily_returns = list(daily_pnl.values())
    if len(daily_returns) >= 2:
        # Annualized daily Sharpe — industry standard
        ann_sharpe = sharpe_ratio(daily_returns) * math.sqrt(252)
    else:
        # Fallback: per-trade (no annualization)
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
