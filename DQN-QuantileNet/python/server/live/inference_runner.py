"""Run batched inference on a list of windows and format outputs.

`format_horizons_batched` is the vectorised replacement for the per-sample
`_format_horizons` loop in the original live_writer — it builds the full list
of (current + historical) entries in one walk over the result tensors.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

UTC = timezone.utc


def _iso(ts_ms: float) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).isoformat()


def _format_one(pred_q, pred_t, sample_idx: int, horizons: list,
                quantiles: list, thresholds: list, current_time_ms: int) -> list:
    out = []
    for hi, h in enumerate(horizons):
        target_iso = _iso(current_time_ms + h * 3600 * 1000)
        entry = {
            'horizon_h':   h,
            'target_time': target_iso,
            'long':        {'quantiles': {}, 'threshold_probs': {}},
            'short':       {'quantiles': {}, 'threshold_probs': {}},
        }
        if pred_q is not None:
            for di, dir_key in enumerate(('long', 'short')):
                entry[dir_key]['quantiles'] = {
                    f'p{int(q * 100)}': round(float(pred_q[sample_idx, hi, qi, di]) * 100, 3)
                    for qi, q in enumerate(quantiles)
                }
        if pred_t is not None:
            for di, dir_key in enumerate(('long', 'short')):
                entry[dir_key]['threshold_probs'] = {
                    f'{t}pct': round(float(pred_t[sample_idx, hi, ti, di]), 4)
                    for ti, t in enumerate(thresholds)
                }
        out.append(entry)
    return out


def format_symbol_predictions(results: dict, windows: list, horizons: list,
                              quantiles: list, thresholds: list,
                              hist_anchor_step_h: int) -> Optional[dict]:
    """Build a SymbolPrediction-shaped dict from batched inference results.

    windows[0] is the current sample; windows[1:] are historical anchors.
    """
    if not windows:
        return None

    pred_q = results.get('pred_quantile')
    pred_t = results.get('pred_threshold')

    cur_w = windows[0]
    cur_time_ms = int(cur_w['current_close_time'])
    cur_price   = float(cur_w['current_close'])

    cur_horizons = _format_one(pred_q, pred_t, 0, horizons, quantiles, thresholds, cur_time_ms)

    historical = []
    for i in range(1, len(windows)):
        w = windows[i]
        ts_ms = int(w['current_close_time'])
        historical.append({
            'anchor_time':  _iso(ts_ms),
            'anchor_price': float(w['current_close']),
            'horizons':     _format_one(pred_q, pred_t, i, horizons, quantiles, thresholds, ts_ms),
        })

    return {
        'symbol':              cur_w['symbol'],
        'current_price':       cur_price,
        'current_time':        _iso(cur_time_ms),
        'horizons':            cur_horizons,
        'historical':          historical,
        'hist_anchor_step_h':  hist_anchor_step_h,
    }
