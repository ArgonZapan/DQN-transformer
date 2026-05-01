"""Append fresh closed candles to historical CSVs (only when gap is large enough)."""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
CSV_DIR       = os.path.join(_PROJECT_ROOT, 'node', 'data', 'historical')

# Only append when the gap exceeds this many candles — avoids per-call IO churn.
GAP_THRESHOLD = 500


def _read_last_close_time(path: str) -> Optional[int]:
    """Read close_time (column 6) of the last row. Returns None on any error/empty."""
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 4096))
            tail = f.read().decode('utf-8', errors='replace')
        lines = [ln for ln in tail.strip().split('\n') if ln and not ln.startswith('timestamp')]
        if not lines:
            return None
        return int(lines[-1].split(',')[6])
    except Exception:
        return None


def append_new_candles(symbol: str, tf: str, raw: dict,
                       threshold: int = GAP_THRESHOLD) -> int:
    """Append new closed candles to existing CSV if gap >= threshold.

    Returns number of rows appended (0 if skipped or file missing).
    Never creates new CSVs.
    """
    path = os.path.join(CSV_DIR, f'{symbol}_{tf}.csv')
    if not os.path.exists(path):
        logger.warning('[CSV] %s %s: file missing at %s — append skipped', symbol, tf, path)
        return 0

    last_ct = _read_last_close_time(path)
    if last_ct is None:
        return 0

    new_idx = np.where(raw['close_time'] > last_ct)[0]
    if new_idx.size < threshold:
        return 0

    raw_rows = raw.get('raw_rows') or []
    if not raw_rows:
        return 0

    lines = []
    for i in new_idx:
        row = raw_rows[int(i)]
        if len(row) < 12:
            continue
        lines.append(','.join(str(v) for v in row[:12]))
    if not lines:
        return 0

    # Defend against a previous crash that left the file without a trailing
    # newline — without this, the first appended line would glue onto the
    # last existing line and corrupt that row.
    with open(path, 'rb') as f:
        f.seek(-1, 2)
        needs_lead_nl = f.read(1) != b'\n'

    with open(path, 'ab') as f:
        prefix = b'\n' if needs_lead_nl else b''
        f.write(prefix + ('\n'.join(lines) + '\n').encode('utf-8'))

    logger.info('[CSV] %s %s: appended %d new candles (gap=%d)',
                symbol, tf, len(lines), new_idx.size)
    return len(lines)
