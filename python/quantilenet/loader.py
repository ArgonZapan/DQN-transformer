"""Loader for prerolled QuantileNet predictions consumed by the DQN learner.

Reads <symbol>.parquet files written by DQN-QuantileNet/python/preroll_for_dqn.py
and answers O(1) lookups keyed by (symbol, close_time_ms).

Output of lookup() is a fixed-length vector (QUANTILE_FEATURE_DIM = 30 by
default, controlled by config), normalized via /scale + clip([-clip, +clip]).
Exact timestamp matches are preferred. If no exact row exists, the loader can
fall back to the latest prediction whose close_time_ms is <= the requested DQN
timestamp, which lets a 1m actor consume QuantileNet rows produced on a coarser
base timeframe without looking into the future.

Column order mirrors preroll_for_dqn.py._build_columns:
    for h in horizons:
        for q in quantiles:
            <q_h{h}_q{QQ}_up>, <q_h{h}_q{QQ}_down>
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _build_columns(horizons: list[int], quantiles: list[float]) -> list[str]:
    cols = []
    for h in horizons:
        for q in quantiles:
            qq = int(round(q * 100))
            cols.append(f"q_h{h}_q{qq:02d}_up")
            cols.append(f"q_h{h}_q{qq:02d}_down")
    return cols


def _expected_dim(horizons: list[int], quantiles: list[float]) -> int:
    return len(horizons) * len(quantiles) * 2


# Default — recomputed in QuantileFeatureLoader.__init__ from config.
QUANTILE_FEATURE_DIM = 3 * 5 * 2  # 30


def quantile_feature_dim(config: dict) -> int:
    """Pure-config dim derivation; used by buffers/network without instantiating the loader."""
    qn_cfg = config.get("quantilenet", {}) or {}
    horizons = list(qn_cfg.get("horizons_hours", [24, 48, 72]))
    quantiles = list(qn_cfg.get("quantiles", [0.10, 0.25, 0.50, 0.75, 0.90]))
    return _expected_dim(horizons, quantiles)


def extended_position_dim(config: dict) -> int:
    """Total position-vector dim fed to the network: 10 base + quantile features."""
    return 10 + quantile_feature_dim(config)


class QuantileFeatureLoader:
    """Per-symbol cache of prerolled quantile predictions.

    Internal layout per symbol:
      _times[symbol]:    int64 array [N], sorted ascending
      _features[symbol]: float32 array [N, dim], normalized & clipped
      _ts_to_row[symbol]: dict[int -> int]   (close_time_ms -> row index)
    """

    def __init__(self, config: dict):
        qn_cfg = config.get("quantilenet", {}) or {}
        self.enabled = bool(qn_cfg.get("enabled", False))
        self.horizons = list(qn_cfg.get("horizons_hours", [24, 48, 72]))
        self.quantiles = list(qn_cfg.get("quantiles", [0.10, 0.25, 0.50, 0.75, 0.90]))
        self.scale = float(qn_cfg.get("feature_scale", 0.10))
        self.clip = float(qn_cfg.get("feature_clip", 3.0))
        self.lookup_mode = str(qn_cfg.get("lookup_mode", "previous")).lower()
        self.max_staleness_ms = int(qn_cfg.get("max_staleness_ms", 15 * 60 * 1000))
        self._miss_warn_limit = int(qn_cfg.get("miss_warn_limit", 5))

        self.dim = _expected_dim(self.horizons, self.quantiles)
        # Update module-level constant so the rest of the codebase can rely on it.
        global QUANTILE_FEATURE_DIM
        QUANTILE_FEATURE_DIM = self.dim

        self._zero = np.zeros(self.dim, dtype=np.float32)
        self._times: dict[str, np.ndarray] = {}
        self._features: dict[str, np.ndarray] = {}
        self._ts_to_row: dict[str, dict] = {}
        self._miss_count: dict[str, int] = {}

        if not self.enabled:
            logger.info("[QuantileLoader] disabled — emitting zeros for 30 quantile features")
            return

        symbols = [a["symbol"] for a in config.get("actors", []) if a.get("symbol")]
        predictions_dir = qn_cfg.get("predictions_dir", "python/data/quantile_predictions")
        # Resolve relative to the DQN-OPUS root (parent of python/).
        root = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        pred_dir = (root / predictions_dir) if not os.path.isabs(predictions_dir) else Path(predictions_dir)

        if not pred_dir.is_dir():
            raise FileNotFoundError(
                f"[QuantileLoader] predictions_dir not found: {pred_dir}. "
                f"Run DQN-QuantileNet/python/preroll_for_dqn.py first, or set "
                f"[quantilenet].enabled = false in config.toml."
            )

        cols = _build_columns(self.horizons, self.quantiles)
        for symbol in symbols:
            self._load_symbol(symbol, pred_dir, cols)

        loaded = sum(1 for s in symbols if s in self._features)
        logger.info(
            f"[QuantileLoader] enabled — loaded {loaded}/{len(symbols)} symbols, "
            f"dim={self.dim}, scale={self.scale}, clip={self.clip}, "
            f"lookup={self.lookup_mode}, max_staleness_ms={self.max_staleness_ms}"
        )

    def _load_symbol(self, symbol: str, pred_dir: Path, cols: list[str]) -> None:
        path = pred_dir / f"{symbol}.parquet"
        if not path.is_file():
            raise FileNotFoundError(
                f"[QuantileLoader] missing parquet for {symbol}: {path}. "
                f"Re-run preroll for this symbol, or remove it from [[actors]]."
            )

        # Lazy import — pandas is heavy and only needed when enabled.
        import pandas as pd

        df = pd.read_parquet(path)
        if "close_time_ms" not in df.columns:
            raise ValueError(f"[QuantileLoader] {path}: missing close_time_ms column")
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"[QuantileLoader] {path}: missing columns {missing[:3]}... — "
                f"horizons/quantiles in DQN-OPUS config don't match what was prerolled."
            )

        df = df.sort_values("close_time_ms", kind="stable").reset_index(drop=True)
        times = df["close_time_ms"].to_numpy(dtype=np.int64)
        feats = df[cols].to_numpy(dtype=np.float32)

        # Normalize: divide by scale, then clip.
        feats = feats / self.scale
        np.clip(feats, -self.clip, self.clip, out=feats)

        self._times[symbol] = times
        self._features[symbol] = feats
        self._ts_to_row[symbol] = {int(t): i for i, t in enumerate(times)}
        self._miss_count[symbol] = 0
        logger.info(f"[QuantileLoader] {symbol}: {len(df):,} prerolled rows from {path.name}")

    def lookup(self, symbol: str | None, ts_ms: int | None) -> np.ndarray:
        """Return the 30-D feature vector for (symbol, close_time_ms).

        Always returns a freshly allocated array safe for the caller to keep —
        we don't share memory with the underlying parquet-backed buffer.
        Misses fall back to zeros.
        """
        if not self.enabled:
            return self._zero.copy()
        if symbol is None or ts_ms is None:
            return self._zero.copy()
        feats_by_ts = self._ts_to_row.get(symbol)
        if feats_by_ts is None:
            return self._zero.copy()
        ts = int(ts_ms)
        row = feats_by_ts.get(ts)
        if row is None and self.lookup_mode in ("previous", "prev", "floor"):
            times = self._times.get(symbol)
            if times is not None and len(times):
                pos = int(np.searchsorted(times, ts, side="right") - 1)
                if pos >= 0:
                    pred_ts = int(times[pos])
                    if self.max_staleness_ms <= 0 or ts - pred_ts <= self.max_staleness_ms:
                        row = pos
        if row is None:
            n = self._miss_count.get(symbol, 0)
            if n < self._miss_warn_limit:
                logger.warning(
                    f"[QuantileLoader] {symbol}: timestamp {ts_ms} not in preroll "
                    f"({n + 1}/{self._miss_warn_limit} warnings)"
                )
                self._miss_count[symbol] = n + 1
            return self._zero.copy()
        return self._features[symbol][row].copy()
