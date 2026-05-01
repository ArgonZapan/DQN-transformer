"""Single point of contact with Binance public REST API.

Responsibilities:
  - Page klines beyond the 1000-row API limit
  - Retry transient errors (429 / 5xx) with exponential backoff and Retry-After
  - Decode the 12-column kline rows into typed numpy arrays
  - Reused by:
      - python/server/live/window_builder.py  (live inference fetch)
      - python/server/app.py /api/klines proxy (frontend chart)
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

KLINES_URL = 'https://api.binance.com/api/v3/klines'
TF_TO_BINANCE = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1w'}
MAX_LIMIT = 1000

_DEFAULT_RETRIES = 4
_DEFAULT_BACKOFF = 0.5  # seconds, doubled each retry


def _request(url: str, timeout: float = 15.0,
             retries: int = _DEFAULT_RETRIES,
             backoff: float = _DEFAULT_BACKOFF) -> list:
    """GET with retry on 429 / 5xx. Honours Retry-After header when present."""
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 429 or 500 <= e.code < 600:
                wait = float(e.headers.get('Retry-After', 0)) or backoff * (2 ** attempt)
                logger.warning('[Binance] %s on %s — retry in %.1fs (%d/%d)',
                               e.code, url, wait, attempt + 1, retries)
                time.sleep(wait)
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            wait = backoff * (2 ** attempt)
            logger.warning('[Binance] network error on %s — retry in %.1fs (%d/%d): %s',
                           url, wait, attempt + 1, retries, e)
            time.sleep(wait)
    raise RuntimeError(f'Binance request failed after {retries} retries: {last_err}')


def fetch_klines_raw(symbol: str, interval: str, limit: int = MAX_LIMIT,
                     end_time_ms: Optional[int] = None) -> list:
    """Single Binance request (≤1000 candles). Returns raw 12-col rows."""
    if interval not in TF_TO_BINANCE:
        raise ValueError(f'Unsupported interval: {interval}')
    url = f'{KLINES_URL}?symbol={symbol}&interval={TF_TO_BINANCE[interval]}&limit={limit}'
    if end_time_ms is not None:
        url += f'&endTime={end_time_ms}'
    return _request(url)


def fetch_klines_paginated(symbol: str, interval: str, limit: int,
                           end_time_ms: Optional[int] = None) -> list:
    """Paginated fetch (limit may exceed MAX_LIMIT). Returns raw rows oldest-first."""
    rows: list = []
    remaining = limit
    cursor    = end_time_ms
    while remaining > 0:
        page_limit = min(MAX_LIMIT, remaining)
        page = fetch_klines_raw(symbol, interval, page_limit, cursor)
        if not page:
            break
        rows = page + rows                           # prepend: each page is older
        remaining -= len(page)
        if len(page) < page_limit:
            break                                    # exchange has no more history
        cursor = int(page[0][0]) - 1
    return rows


def decode_klines(rows: list) -> dict:
    """Convert raw 12-col rows into numpy-typed dict. Keeps `raw_rows` for CSV append."""
    if not rows:
        raise ValueError('empty kline response')
    arr = np.array([[float(k[i]) for i in range(7)] for k in rows], dtype=np.float64)
    return {
        'timestamp':  arr[:, 0].astype(np.int64),
        'open':       arr[:, 1],
        'high':       arr[:, 2],
        'low':        arr[:, 3],
        'close':      arr[:, 4],
        'volume':     arr[:, 5],
        'close_time': arr[:, 6].astype(np.int64),
        'raw_rows':   rows,
    }


def fetch_klines(symbol: str, interval: str, limit: int = MAX_LIMIT,
                 end_time_ms: Optional[int] = None) -> dict:
    """High-level wrapper: paginate + decode."""
    rows = fetch_klines_paginated(symbol, interval, limit, end_time_ms)
    return decode_klines(rows)
