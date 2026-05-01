"""Orchestrates live prediction generation: fetch → window → inference → write.

`LivePredictionLoop` runs the orchestrator on a timer in a daemon thread.
Coordination with the dashboard server is via the live_predictions.json
file (the server polls its mtime) — writer and server run as separate
processes (see start_dashboard.bat).
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import torch

from .window_builder    import fetch_all_raw, build_all_windows, HIST_ANCHOR_DAYS_BACK
from .inference_runner  import format_symbol_predictions
from .json_writer       import write_atomic, LIVE_JSON_PATH

logger = logging.getLogger(__name__)


def write_live_predictions(model, config: dict, checkpoint_path: str,
                           device: torch.device) -> dict:
    """Run inference for all configured symbols and write live_predictions.json."""
    from ...backtest import run_inference

    symbols    = [a['symbol'] for a in config['actors']]
    pred_cfg   = config['prediction']
    horizons   = pred_cfg['horizons_hours']
    quantiles  = pred_cfg.get('quantiles', [])
    thresholds = pred_cfg.get('thresholds_pct', [])

    predictions: list[dict] = []
    model.eval()
    with torch.no_grad():
        for symbol in symbols:
            try:
                raw = fetch_all_raw(symbol, config, days_back=HIST_ANCHOR_DAYS_BACK)
            except Exception as e:
                logger.warning('[Loop] %s: fetch failed — %s', symbol, e)
                continue

            try:
                windows, step_h = build_all_windows(symbol, raw, config, HIST_ANCHOR_DAYS_BACK)
            except Exception as e:
                logger.warning('[Loop] %s: window build failed — %s', symbol, e)
                continue
            if not windows:
                logger.warning('[Loop] %s: no usable windows, skipping', symbol)
                continue

            results = run_inference(model, windows, config, device, num_workers=0)
            entry = format_symbol_predictions(results, windows, horizons, quantiles,
                                              thresholds, hist_anchor_step_h=step_h)
            if entry is None:
                continue
            predictions.append(entry)
            logger.info('[Loop] %s: %d historical anchors', symbol, len(entry['historical']))

    return write_atomic(predictions, checkpoint_path)


class LivePredictionLoop:
    """Background thread: writes live_predictions.json every `interval_sec` seconds."""

    def __init__(self, model, config: dict, checkpoint_path: str,
                 device: torch.device, interval_sec: int = 900):
        self.model           = model
        self.config          = config
        self.checkpoint_path = checkpoint_path
        self.device          = device
        self.interval_sec    = interval_sec
        self._stop           = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True, name='live-predictor')
        self._thread.start()
        logger.info('[Loop] background started (interval=%ds)', self.interval_sec)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def trigger(self) -> None:
        """Run one iteration synchronously (manual refresh)."""
        try:
            write_live_predictions(
                self.model, self.config, self.checkpoint_path, self.device,
            )
        except Exception as e:
            logger.error('[Loop] trigger failed: %s', e)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                write_live_predictions(
                    self.model, self.config, self.checkpoint_path, self.device,
                )
            except Exception as e:
                logger.error('[Loop] iteration failed: %s', e)
            self._stop.wait(self.interval_sec)


__all__ = ['LivePredictionLoop', 'write_live_predictions', 'LIVE_JSON_PATH']
