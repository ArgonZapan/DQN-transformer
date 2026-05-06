"""Preroll QuantileNet predictions for DQN-OPUS training.

Builds historical windows, runs QuantileNet inference, and writes a Parquet
file per symbol with quantile features indexed by close_time_ms.

Output layout (per symbol):
    <output_dir>/<symbol>.parquet
    columns: close_time_ms (int64),
             q_h{H}_q{QQ}_{up|down} for H in horizons, QQ = int(quantile*100)

The DQN-OPUS loader keys lookups by close_time_ms — make sure the actor
sends the close_time of the *current* (last closed) base-TF candle.

No-lookahead: build_windows() builds windows whose current_close_time is the
last closed bar at decision time, so a row at t describes predictions made
using only data up to and including t.

Usage (from DQN-QuantileNet/):
    python python/preroll_for_dqn.py --output-dir ../python/data/quantile_predictions
        [--horizons 24 48 72] [--checkpoint path.pt]

Leakage guard:
    The default mode writes only the held-out walk-forward test block.
    Generating full-history predictions with one checkpoint is in-sample for
    much of the history and can leak future training information into DQN
    states, so it requires --allow-in-sample.

    For full causal coverage, pass --checkpoint-manifest manifest.json:
        [{"checkpoint": "path/to/fold1.pt", "start_ms": 0, "end_ms": 1710000000000}]
    Each timestamp is predicted by the checkpoint assigned to its time range.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from python.config import load_config, get_device
from python.data.loader import build_windows
from python.backtest import find_best_checkpoint, load_model, run_inference
from python.training.walk_forward import build_walk_forward_folds

logger = logging.getLogger(__name__)


def _resolve_horizon_indices(cfg_horizons: list[int], wanted: list[int]) -> list[int]:
    idx = []
    for h in wanted:
        if h not in cfg_horizons:
            raise ValueError(
                f"Horizon {h}h not in QuantileNet config horizons {cfg_horizons}. "
                f"Either retrain QN with this horizon or pick a different --horizons."
            )
        idx.append(cfg_horizons.index(h))
    return idx


def _build_columns(horizons: list[int], quantiles: list[float]) -> list[str]:
    cols = []
    for h in horizons:
        for q in quantiles:
            qq = int(round(q * 100))
            cols.append(f"q_h{h}_q{qq:02d}_up")
            cols.append(f"q_h{h}_q{qq:02d}_down")
    return cols


def preroll_symbol(
    symbol: str,
    config: dict,
    model,
    device,
    horizons: list[int],
    horizon_idx: list[int],
    quantiles: list[float],
    output_dir: Path,
    windows: list[dict] | None = None,
) -> int:
    if windows is None:
        logger.info(f"[Preroll] {symbol}: building windows...")
        windows = build_windows(symbol, config)
    if not windows:
        logger.warning(f"[Preroll] {symbol}: no windows produced — skipping")
        return 0

    logger.info(f"[Preroll] {symbol}: {len(windows)} windows; running inference...")
    results = run_inference(model, windows, config, device)
    pred_q = results.get("pred_quantile")
    if pred_q is None:
        logger.error(f"[Preroll] {symbol}: model produced no quantile output — "
                     f"check QN config [prediction] mode (must be 'quantile' or 'both').")
        return 0

    # pred_q shape: [N, n_horizons_total, n_quantiles, 2]
    # Slice to selected horizons.
    selected = pred_q[:, horizon_idx, :, :]  # [N, len(horizons), n_quantiles, 2]
    n = selected.shape[0]
    n_q = selected.shape[2]
    assert n_q == len(quantiles), f"quantile count mismatch: {n_q} vs {len(quantiles)}"

    # Flatten to (N, 30) in column order matching _build_columns:
    # for h: for q: [up, down]
    flat = np.empty((n, len(horizons) * n_q * 2), dtype=np.float32)
    col = 0
    for hi in range(len(horizons)):
        for qi in range(n_q):
            flat[:, col]     = selected[:, hi, qi, 0]
            flat[:, col + 1] = selected[:, hi, qi, 1]
            col += 2

    close_times_ms = np.asarray(
        [int(w["current_close_time"]) for w in windows], dtype=np.int64
    )
    # Sort by timestamp for predictable lookups (build_windows isn't guaranteed sorted).
    order = np.argsort(close_times_ms, kind="stable")
    close_times_ms = close_times_ms[order]
    flat = flat[order]

    # Drop duplicate timestamps if any (keep last).
    _, unique_idx = np.unique(close_times_ms[::-1], return_index=True)
    keep = (len(close_times_ms) - 1 - unique_idx)
    keep.sort()
    close_times_ms = close_times_ms[keep]
    flat = flat[keep]

    cols = _build_columns(horizons, quantiles)
    df = pd.DataFrame(flat, columns=cols)
    df.insert(0, "close_time_ms", close_times_ms)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{symbol}.parquet"
    df.to_parquet(out_path, index=False, compression="zstd")
    logger.info(f"[Preroll] {symbol}: wrote {len(df)} rows -> {out_path}")
    return len(df)


def _label_end(w: dict) -> int:
    fct = w.get("future_close_times")
    if fct is not None and len(fct):
        return int(max(fct))
    return int(w["current_close_time"])


def _split_by_symbol(windows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for w in windows:
        grouped.setdefault(str(w["symbol"]), []).append(w)
    return grouped


def _build_oos_windows(config: dict) -> dict[str, list[dict]]:
    all_windows = []
    for symbol in [a["symbol"] for a in config["actors"]]:
        wins = build_windows(symbol, config)
        logger.info(f"[Preroll] {symbol}: full windows={len(wins)}")
        all_windows.extend(wins)

    train_cfg = config["training"]
    _, test_windows = build_walk_forward_folds(
        all_windows,
        train_months=train_cfg["train_months"],
        val_months=train_cfg["val_months"],
        test_months=train_cfg["test_months"],
    )
    logger.info(f"[Preroll] OOS/test windows selected: {len(test_windows)}")
    return _split_by_symbol(test_windows)


def _load_manifest(path: str, config: dict, device) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    if not isinstance(entries, list) or not entries:
        raise ValueError("checkpoint manifest must be a non-empty JSON list")

    loaded = []
    for i, entry in enumerate(entries):
        ckpt = entry.get("checkpoint")
        if not ckpt:
            raise ValueError(f"manifest entry {i} is missing 'checkpoint'")
        start_ms = int(entry.get("start_ms", -2**63))
        end_ms = int(entry.get("end_ms", 2**63 - 1))
        if end_ms <= start_ms:
            raise ValueError(f"manifest entry {i} has end_ms <= start_ms")
        loaded.append({
            "start_ms": start_ms,
            "end_ms": end_ms,
            "checkpoint": ckpt,
            "model": load_model(config, device, ckpt),
        })
    return sorted(loaded, key=lambda e: e["start_ms"])


def _windows_for_entry(windows: list[dict], entry: dict) -> list[dict]:
    return [
        w for w in windows
        if entry["start_ms"] <= _label_end(w) <= entry["end_ms"]
    ]


def _write_manifest_preroll(
    config: dict,
    device,
    manifest: list[dict],
    horizons: list[int],
    horizon_idx: list[int],
    quantiles: list[float],
    output_dir: Path,
) -> int:
    total = 0
    cols = _build_columns(horizons, quantiles)
    for symbol in [a["symbol"] for a in config["actors"]]:
        windows = build_windows(symbol, config)
        chunks = []
        for entry in manifest:
            part = _windows_for_entry(windows, entry)
            if not part:
                continue
            logger.info(
                f"[Preroll] {symbol}: {len(part)} rows from {Path(entry['checkpoint']).name}"
            )
            results = run_inference(entry["model"], part, config, device)
            pred_q = results.get("pred_quantile")
            if pred_q is None:
                raise RuntimeError("model produced no quantile output")
            selected = pred_q[:, horizon_idx, :, :]
            flat = np.empty((selected.shape[0], len(cols)), dtype=np.float32)
            col = 0
            for hi in range(len(horizons)):
                for qi in range(len(quantiles)):
                    flat[:, col] = selected[:, hi, qi, 0]
                    flat[:, col + 1] = selected[:, hi, qi, 1]
                    col += 2
            df = pd.DataFrame(flat, columns=cols)
            df.insert(0, "close_time_ms", [int(w["current_close_time"]) for w in part])
            chunks.append(df)

        if not chunks:
            logger.warning(f"[Preroll] {symbol}: manifest produced no rows")
            continue

        out = pd.concat(chunks, ignore_index=True)
        out = out.sort_values("close_time_ms", kind="stable")
        out = out.drop_duplicates("close_time_ms", keep="last")
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{symbol}.parquet"
        out.to_parquet(out_path, index=False, compression="zstd")
        logger.info(f"[Preroll] {symbol}: wrote {len(out)} OOF rows -> {out_path}")
        total += len(out)
    return total


def main():
    ap = argparse.ArgumentParser(description="Preroll QuantileNet predictions for DQN-OPUS")
    ap.add_argument("--output-dir", type=str, required=True,
                    help="Directory for <symbol>.parquet files")
    ap.add_argument("--horizons", type=int, nargs="+", default=[24, 48, 72],
                    help="Horizons (hours) to extract; must exist in QN config (default: 24 48 72)")
    ap.add_argument("--checkpoint", type=str, default="",
                    help="QN checkpoint path (default: best in checkpoints/)")
    ap.add_argument("--checkpoint-manifest", type=str, default="",
                    help="JSON manifest mapping timestamp ranges to checkpoints for OOF preroll")
    ap.add_argument("--allow-in-sample", action="store_true",
                    help="Allow full-history preroll with one checkpoint (can leak future info)")
    ap.add_argument("--symbols", type=str, nargs="*", default=None,
                    help="Override symbols (default: from QN config [[actors]])")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    config = load_config()
    device = get_device(config)

    pred_cfg = config["prediction"]
    cfg_horizons = list(pred_cfg["horizons_hours"])
    quantiles = list(pred_cfg.get("quantiles", []))
    if not quantiles:
        raise SystemExit("QN config has no [prediction].quantiles — quantile mode required.")
    horizon_idx = _resolve_horizon_indices(cfg_horizons, args.horizons)

    symbols = args.symbols or [a["symbol"] for a in config["actors"]]
    if args.symbols:
        config["actors"] = [{"symbol": s, "exchange": "binance"} for s in symbols]
    logger.info(f"[Preroll] Symbols: {symbols}")
    logger.info(f"[Preroll] Horizons: {args.horizons} (indices {horizon_idx} in QN config)")
    logger.info(f"[Preroll] Quantiles: {quantiles}")
    logger.info(f"[Preroll] Output: {args.output_dir}")

    output_dir = Path(args.output_dir).resolve()
    if args.checkpoint_manifest:
        manifest = _load_manifest(args.checkpoint_manifest, config, device)
        total = _write_manifest_preroll(
            config, device, manifest, args.horizons, horizon_idx, quantiles, output_dir
        )
    else:
        ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
        checkpoint = args.checkpoint or find_best_checkpoint(ckpt_dir)
        logger.info(f"[Preroll] Using checkpoint: {checkpoint}")
        model = load_model(config, device, checkpoint)

        if args.allow_in_sample:
            logger.warning("[Preroll] --allow-in-sample set: writing full-history predictions.")
            windows_by_symbol = {symbol: build_windows(symbol, config) for symbol in symbols}
        else:
            logger.info("[Preroll] Leakage guard active: writing held-out OOS/test predictions only.")
            windows_by_symbol = _build_oos_windows(config)

        total = 0
        for symbol in symbols:
            try:
                total += preroll_symbol(
                    symbol, config, model, device,
                    args.horizons, horizon_idx, quantiles, output_dir,
                    windows=windows_by_symbol.get(symbol, []),
                )
            except Exception as e:
                logger.exception(f"[Preroll] {symbol} failed: {e}")

    logger.info(f"[Preroll] Done. Total rows across all symbols: {total}")


if __name__ == "__main__":
    main()
