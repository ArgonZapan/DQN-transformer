"""Kronos foundation-model wrapper for the QuantileNet dashboard.

Lazy-loads `Kronos-mini` + `Kronos-Tokenizer-base` from HuggingFace on first use,
then exposes:

  * `generate_continuation(history_df, pred_len, T, top_p, seed)`
      → DataFrame with `pred_len` future OHLCV candles.
  * `bruteforce_strategies(history_df, pred_len, n_simulations, ...)`
      → ranked list of (direction, horizon, threshold) combos by SQN/EV
        across Monte-Carlo paths sampled from Kronos. Each path is a single
        TP/SL trade — bar-by-bar fill simulation matching python/backtest.py.

The wrapper assumes uniformly-spaced candles (e.g. 1h, 4h). Timestamps for the
prediction window are extrapolated from the last two history rows.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch

from .kronos import Kronos, KronosTokenizer, KronosPredictor

logger = logging.getLogger(__name__)

# HuggingFace model registry. Each entry: (model_repo, tokenizer_repo, max_context).
# Note tokenizer differs: Kronos-mini uses the 2k-context tokenizer, the rest use -base.
KRONOS_MODELS: dict[str, tuple[str, str, int]] = {
    "mini":  ("NeoQuasar/Kronos-mini",  "NeoQuasar/Kronos-Tokenizer-2k",   2048),
    "small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base", 512),
    "base":  ("NeoQuasar/Kronos-base",  "NeoQuasar/Kronos-Tokenizer-base", 512),
}
DEFAULT_MODEL = "mini"

_predictors: dict[str, KronosPredictor] = {}
_load_lock = threading.Lock()
# Serializes all model invocations: torch.manual_seed mutates a global RNG,
# so concurrent generate/bruteforce calls would race on determinism.
_inference_lock = threading.Lock()


def get_predictor(name: str = DEFAULT_MODEL) -> KronosPredictor:
    """Lazy per-model singleton. First call for a given name downloads weights from HF."""
    if name not in KRONOS_MODELS:
        raise ValueError(f"unknown Kronos model {name!r}; choose one of {list(KRONOS_MODELS)}")
    if name in _predictors:
        return _predictors[name]
    with _load_lock:
        if name in _predictors:
            return _predictors[name]
        model_repo, tok_repo, max_ctx = KRONOS_MODELS[name]
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info("[Kronos] loading %s + %s on %s …", model_repo, tok_repo, device)
        tokenizer = KronosTokenizer.from_pretrained(tok_repo)
        model     = Kronos.from_pretrained(model_repo)
        _predictors[name] = KronosPredictor(model, tokenizer, device=device, max_context=max_ctx)
        logger.info("[Kronos] %s ready", name)
    return _predictors[name]


# ── Helpers ─────────────────────────────────────────────────────────────────

def _infer_step(timestamps: pd.Series) -> pd.Timedelta:
    """Median delta between consecutive candle open-times. Must be positive."""
    diffs = timestamps.diff().dropna()
    if diffs.empty:
        raise ValueError("history must contain at least 2 candles to infer step")
    step = diffs.median()
    if step <= pd.Timedelta(0):
        raise ValueError("history has zero/negative timestep (duplicate or unsorted candles)")
    return step


def _future_timestamps(last_ts: pd.Timestamp, step: pd.Timedelta, n: int) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([last_ts + step * (i + 1) for i in range(n)])


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Public API ──────────────────────────────────────────────────────────────

def generate_continuation(
    history: pd.DataFrame,
    pred_len: int,
    T: float = 1.0,
    top_p: float = 0.9,
    seed: Optional[int] = None,
    model_name: str = DEFAULT_MODEL,
) -> pd.DataFrame:
    """Generate `pred_len` future candles given a history DataFrame.

    `history` columns required: timestamp (datetime), open, high, low, close, volume.
    Returns DataFrame indexed by future timestamp with OHLCV columns.
    """
    if not {"timestamp", "open", "high", "low", "close"}.issubset(history.columns):
        raise ValueError("history needs columns: timestamp, open, high, low, close (+ volume)")
    if len(history) < 2:
        raise ValueError("history too short")

    pred = get_predictor(model_name)
    ts = pd.to_datetime(history["timestamp"])
    step = _infer_step(ts)

    # Trim to model context. Kronos tokenizes ~2 tokens per candle, so we cap
    # the input at half of max_context to stay safely below the limit.
    lookback = min(len(history), max(64, pred.max_context // 2 - 32))
    h = history.iloc[-lookback:].reset_index(drop=True)

    x_df = pd.DataFrame({
        "open":   h["open"].astype(float).values,
        "high":   h["high"].astype(float).values,
        "low":    h["low"].astype(float).values,
        "close":  h["close"].astype(float).values,
        "volume": h["volume"].astype(float).values if "volume" in h.columns else 0.0,
    })

    x_timestamp = pd.to_datetime(h["timestamp"]).reset_index(drop=True)
    last_ts = pd.Timestamp(x_timestamp.iloc[-1])
    y_timestamp = pd.Series(_future_timestamps(last_ts, step, pred_len))

    with _inference_lock:
        _set_seed(seed)
        pred_df = pred.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=float(T),
            top_p=float(top_p),
            sample_count=1,
            verbose=False,
        )
    # De-normalised volumes can come out slightly negative — clamp.
    if "volume" in pred_df.columns:
        pred_df["volume"] = pred_df["volume"].clip(lower=0.0)
    pred_df.index.name = "timestamp"
    return pred_df


# ── Bruteforce strategy search (mirrors python/backtest.py TP/SL logic) ────

@dataclass
class Combo:
    direction: str          # "long" | "short"
    horizon_bars: int
    threshold_pct: float    # used as TP target; SL is symmetric (1:1 R:R)
    n_trades: int           # = n_simulations (one MC path = one trade)
    p_win: float
    avg_win: float
    avg_loss: float
    mean_return: float      # mean PnL across paths (replaces misleading total_return)
    ev: float
    sqn: float
    kelly: float
    fire_rate: float        # fraction of paths that hit TP (informational)
    tp_rate: float
    sl_rate: float
    hz_rate: float          # neither TP nor SL hit — exit at horizon close


def _sqn(returns: np.ndarray) -> float:
    if returns.size < 2:
        return 0.0
    mu = float(np.mean(returns))
    sd = float(np.std(returns, ddof=1))
    if sd == 0:
        return 0.0
    return mu / sd * math.sqrt(returns.size)


def _kelly(p: float, win: float, loss: float) -> float:
    if win <= 0 or loss <= 0:
        return 0.0
    b = win / loss
    return p - (1 - p) / b


def bruteforce_strategies(
    history: pd.DataFrame,
    pred_len: int,
    n_simulations: int = 100,
    T: float = 1.0,
    top_p: float = 0.9,
    horizons_bars: Optional[list[int]] = None,
    thresholds_pct: Optional[list[float]] = None,
    commission: float = 0.00075,
    slippage_bps: float = 1.0,
    base_seed: int = 0,
    progress_cb=None,
    model_name: str = DEFAULT_MODEL,
) -> dict:
    """Monte-Carlo strategy search over Kronos-sampled futures with TP/SL fills.

    For each simulation we generate a single OHLC path. For each
    (direction, horizon_bars, threshold_pct) cell we walk that path bar-by-bar
    using high/low to determine whether TP or SL hit first (pessimistic on
    same-bar ties — SL wins). Threshold doubles as TP and SL (1:1 R:R).
    If neither hit, the trade exits at the horizon close.

    Round-trip cost = 2 × (commission + slippage) is subtracted from every
    trade, matching python/backtest.py defaults (0.00075 + 1bp ≈ 0.17%/trade).

    Combos that produce identical effective statistics (e.g. 18h horizon at
    2/3/5% thresholds where no path moves enough to fire either side) are
    collapsed: only the smallest threshold + horizon variant is kept.

    n_trades equals n_simulations: each MC path is one trade scenario.

    Returns dict with `combos` (sorted by SQN desc) and `meta`.
    """
    if horizons_bars is None:
        horizons_bars = sorted({max(1, pred_len // 4), max(1, pred_len // 2), pred_len})
    if thresholds_pct is None:
        thresholds_pct = [0.5, 1.0, 2.0, 3.0, 5.0]
    cost_pct = 2.0 * (float(commission) + float(slippage_bps) / 10000.0) * 100.0

    last_close = float(history["close"].iloc[-1])

    # Collect full OHLC paths up to pred_len bars.
    paths_h = np.zeros((n_simulations, pred_len), dtype=np.float64)
    paths_l = np.zeros((n_simulations, pred_len), dtype=np.float64)
    paths_c = np.zeros((n_simulations, pred_len), dtype=np.float64)
    for i in range(n_simulations):
        if progress_cb is not None:
            progress_cb(i, n_simulations)
        path = generate_continuation(
            history=history, pred_len=pred_len,
            T=T, top_p=top_p, seed=base_seed + i,
            model_name=model_name,
        )
        paths_h[i] = path["high"].values[:pred_len]
        paths_l[i] = path["low"].values[:pred_len]
        paths_c[i] = path["close"].values[:pred_len]

    if progress_cb is not None:
        progress_cb(n_simulations, n_simulations)

    combos: list[Combo] = []
    # (direction, stat-fingerprint) → already-seen. We iterate hb/thr ascending,
    # so the first hit is the smallest variant — others are pure duplicates.
    seen_fingerprints: set[tuple] = set()
    for hb in sorted(set(horizons_bars)):
        end = min(hb, pred_len)
        for thr in sorted(set(thresholds_pct)):
            mult = thr / 100.0
            for direction in ("long", "short"):
                if direction == "long":
                    tp_lvl = last_close * (1.0 + mult)
                    sl_lvl = last_close * (1.0 - mult)
                else:
                    tp_lvl = last_close * (1.0 - mult)
                    sl_lvl = last_close * (1.0 + mult)

                pnls = np.empty(n_simulations, dtype=np.float64)
                tp_count = sl_count = hz_count = 0

                for i in range(n_simulations):
                    highs = paths_h[i, :end]
                    lows  = paths_l[i, :end]
                    if direction == "long":
                        tp_hits = highs >= tp_lvl
                        sl_hits = lows  <= sl_lvl
                    else:
                        tp_hits = lows  <= tp_lvl
                        sl_hits = highs >= sl_lvl

                    tp_any = bool(tp_hits.any())
                    sl_any = bool(sl_hits.any())
                    tp_i = int(tp_hits.argmax()) if tp_any else end
                    sl_i = int(sl_hits.argmax()) if sl_any else end

                    if sl_any and (not tp_any or sl_i <= tp_i):
                        pnls[i] = -thr - cost_pct
                        sl_count += 1
                    elif tp_any:
                        pnls[i] = +thr - cost_pct
                        tp_count += 1
                    else:
                        close_h = paths_c[i, end - 1]
                        ret_pct = (close_h - last_close) / last_close * 100.0
                        if direction == "short":
                            ret_pct = -ret_pct
                        pnls[i] = ret_pct - cost_pct
                        hz_count += 1

                n = n_simulations
                wins   = pnls[pnls > 0]
                losses = pnls[pnls < 0]
                p_win   = float(wins.size / n) if n else 0.0
                avg_win  = float(wins.mean())   if wins.size   else 0.0
                avg_loss = float(-losses.mean()) if losses.size else 0.0
                ev = p_win * avg_win - (1 - p_win) * avg_loss

                # Dedup: when no path crosses thr (TP=SL=0) or all resolve before
                # hb, raising thr/hb produces an identical row. Fingerprint by
                # exit distribution + per-bucket means (rounded for FP noise).
                fp = (
                    direction, tp_count, sl_count, hz_count,
                    round(float(pnls.mean()), 6),
                    round(avg_win, 6), round(avg_loss, 6),
                )
                if fp in seen_fingerprints:
                    continue
                seen_fingerprints.add(fp)

                combos.append(Combo(
                    direction=direction, horizon_bars=hb, threshold_pct=thr,
                    n_trades=n,
                    p_win=round(p_win, 4),
                    avg_win=round(avg_win, 4),
                    avg_loss=round(avg_loss, 4),
                    mean_return=round(float(pnls.mean()), 4),
                    ev=round(ev, 4),
                    sqn=round(_sqn(pnls), 3),
                    kelly=round(_kelly(p_win, avg_win, avg_loss), 4),
                    fire_rate=round(tp_count / n, 4),
                    tp_rate=round(tp_count / n, 4),
                    sl_rate=round(sl_count / n, 4),
                    hz_rate=round(hz_count / n, 4),
                ))

    combos.sort(key=lambda c: c.sqn, reverse=True)
    return {
        "combos": [c.__dict__ for c in combos],
        "meta": {
            "n_simulations": n_simulations,
            "pred_len": pred_len,
            "horizons_bars": horizons_bars,
            "thresholds_pct": thresholds_pct,
            "T": T, "top_p": top_p,
            "last_close": last_close,
            "model": model_name,
        },
    }
