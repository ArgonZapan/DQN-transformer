"""FastAPI dashboard server.

Endpoints:
  GET /                          → dashboard SPA (dist or public)
  GET /api/config                → symbols, horizons, quantiles, thresholds, direction
  GET /api/runs                  → list of run_ids in checkpoints/
  GET /api/calibration[/{run}]   → calibration JSON (latest run if omitted)
  GET /api/strategy[/{run}]      → strategy_top100 JSON (latest run if omitted)
  GET /api/klines                → CORS-free proxy to Binance klines
  GET /api/live                  → SSE stream of validated live_predictions.json

Run:
    python -m python.server.app [--port 8080] [--host 127.0.0.1]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import time as _time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse

from .live.binance_client import fetch_klines_paginated, TF_TO_BINANCE
from .schemas             import AppConfig, LivePredictions
from ..kronos.predictor   import (
    KRONOS_MODELS, DEFAULT_MODEL, _predictors,
    bruteforce_strategies, generate_continuation,
)

logger = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parents[2]
CHECKPOINTS = ROOT / 'python' / 'checkpoints'
LIVE_JSON   = ROOT / 'python' / 'live_predictions.json'
DIST_DIR    = ROOT / 'dashboard' / 'dist'
PUBLIC_DIR  = ROOT / 'dashboard' / 'public'
CONFIG_TOML = ROOT / 'config.toml'

# Hard limits for the public-facing klines proxy.
KLINES_HARD_LIMIT      = 50_000
SYMBOL_RE              = re.compile(r'^[A-Z0-9]{4,16}$')
MAX_SSE_CLIENTS        = 64
WATCHER_INTERVAL_SEC   = 2.0


# ── App + state ─────────────────────────────────────────────────────────────

def _load_config() -> dict:
    try:
        import tomllib
    except ImportError:                          # Python <3.11
        import tomli as tomllib                  # type: ignore
    with open(CONFIG_TOML, 'rb') as f:
        return tomllib.load(f)


def _precompute_sync(config: dict, model, device) -> dict:
    """Build all prediction windows and run batched inference for every symbol.

    Runs entirely in a worker thread (called via asyncio.to_thread).
    Returns {symbol: {close_time_s: {horizon_h: {dir: {q_key: pct}}}}}
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from .live.window_builder import fetch_all_raw, build_all_windows_vectorized
    from ..data.loader import _base_tf
    from ..training.dataset import QuantileDataset, collate_fn

    pred_cfg   = config['prediction']
    quantiles  = list(pred_cfg.get('quantiles', []))
    horizons   = list(pred_cfg['horizons_hours'])
    train_cfg  = config.get('training', {})
    batch_size = int(train_cfg.get('inference_batch_size', 4096))
    days_back  = 30

    all_preds: dict = {}
    symbols = [a['symbol'] for a in config['actors']]

    for symbol in symbols:
        logger.info('[Precompute] %s — fetching %d days of klines…', symbol, days_back)
        try:
            raw = fetch_all_raw(symbol, config, days_back, update_csv=False)
        except Exception as e:
            logger.warning('[Precompute] %s — klines fetch failed: %s', symbol, e)
            continue

        base_tf   = _base_tf(config)
        n_candles = len(raw[base_tf]['close_time'])
        logger.info('[Precompute] %s — building windows for %d candles (vectorised)…',
                    symbol, n_candles)
        t0 = _time.monotonic()
        windows = build_all_windows_vectorized(symbol, raw, config)
        logger.info('[Precompute] %s — %d valid windows ready in %.2fs',
                    symbol, len(windows), _time.monotonic() - t0)

        if not windows:
            continue

        # Build DataLoader manually so we can log per-batch progress.
        active_tfs  = model.active_tfs
        tf_seq_lens = model.tf_seq_lens
        dataset     = QuantileDataset(windows, config, active_tfs, tf_seq_lens)
        loader      = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=collate_fn, num_workers=0,
                                 pin_memory=(device.type == 'cuda'))
        n_passes    = len(loader)
        logger.info('[Precompute] %s — inference: %d windows, batch=%d → %d pass(es) on %s',
                    symbol, len(windows), batch_size, n_passes, device)

        pred_q_parts: list = []
        with torch.no_grad():
            for pass_idx, batch in enumerate(loader):
                tf_inputs = {tf: t.to(device) for tf, t in batch['tf'].items()}
                out = model(tf_inputs)
                if 'quantile' in out:
                    pred_q_parts.append(out['quantile'].cpu().numpy())
                done = min((pass_idx + 1) * batch_size, len(windows))
                logger.info('[Precompute] %s — pass %d / %d done  (%d / %d windows)',
                            symbol, pass_idx + 1, n_passes, done, len(windows))

        if not pred_q_parts:
            logger.warning('[Precompute] %s — model produced no quantile output', symbol)
            continue

        pred_q = np.concatenate(pred_q_parts, axis=0)   # [N, H, Q, D]

        sym_preds: dict = {}
        for i, w in enumerate(windows):
            t_s = str(int(w['current_close_time']) // 1000)
            row: dict = {}
            for hi, h in enumerate(horizons):
                row[h] = {}
                for di, dk in enumerate(('long', 'short')):
                    row[h][dk] = {
                        f'p{int(q * 100)}': round(float(pred_q[i, hi, qi, di]) * 100, 3)
                        for qi, q in enumerate(quantiles)
                    }
            sym_preds[t_s] = row

        all_preds[symbol] = sym_preds
        logger.info('[Precompute] %s — stored %d predictions', symbol, len(sym_preds))

    return all_preds


async def _precompute_predictions(app: FastAPI) -> None:
    """Background task: precompute all predictions and store in app.state."""
    logger.info('[Precompute] starting background precomputation…')
    try:
        result = await asyncio.to_thread(
            _precompute_sync, app.state.config, app.state.qnet_model, app.state.qnet_device
        )
        app.state.all_predictions = result
        total = sum(len(v) for v in result.values())
        logger.info('[Precompute] done — %d total predictions across %d symbols',
                    total, len(result))
    except Exception as e:
        logger.error('[Precompute] failed: %s', e)
    finally:
        app.state.precompute_done = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Read mtime BEFORE payload — otherwise a write between the two reads
    # would leave us with stale payload paired with the new mtime.
    app.state.last_mtime      = LIVE_JSON.stat().st_mtime if LIVE_JSON.is_file() else 0.0
    app.state.last_payload    = await _read_live_json()
    app.state.config          = _load_config()
    app.state.clients         = set()
    app.state.watcher_stop    = asyncio.Event()
    app.state.watcher_task    = asyncio.create_task(_watch_live_json(app))
    app.state.all_predictions = {}
    app.state.precompute_done = False

    try:
        import torch
        from ..backtest import load_model, find_best_checkpoint
        from ..config   import get_device
        _ckpt   = find_best_checkpoint(str(CHECKPOINTS))
        _device = get_device(app.state.config)
        _model  = load_model(app.state.config, _device, _ckpt)
        _model.eval()
        app.state.qnet_model  = _model
        app.state.qnet_device = _device
        logger.info('[App] QuantileNet loaded (%s): %s', _device, _ckpt)
        asyncio.create_task(_precompute_predictions(app))
    except Exception as _e:
        app.state.qnet_model      = None
        app.state.qnet_device     = None
        app.state.precompute_done = True   # no model → skip warmup gate
        logger.info('[App] QuantileNet not loaded — continuous mode disabled: %s', _e)

    logger.info('[App] startup complete (live=%s)', LIVE_JSON.is_file())
    try:
        yield
    finally:
        app.state.watcher_stop.set()
        if app.state.watcher_task:
            try:
                await asyncio.wait_for(app.state.watcher_task, timeout=5)
            except asyncio.TimeoutError:
                logger.warning('[App] watcher did not stop within 5s')


app = FastAPI(title='QuantileNet Dashboard',
              default_response_class=JSONResponse,
              lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1024)


# ── live_predictions.json watcher ───────────────────────────────────────────

async def _read_live_json() -> Optional[dict]:
    if not LIVE_JSON.is_file():
        return None
    try:
        text = await asyncio.to_thread(LIVE_JSON.read_text, encoding='utf-8')
        return json.loads(text)
    except Exception as e:
        logger.warning('[App] failed to read live_predictions.json: %s', e)
        return None


async def _broadcast(payload: dict) -> None:
    """Push payload to all SSE clients.

    Slow clients have their oldest queued item dropped instead of being
    disconnected — a 4-deep queue covers normal back-pressure, and dropping
    the oldest preserves liveness without permanent bans.
    """
    app.state.last_payload = payload
    for q in list(app.state.clients):
        if q.full():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            # Should be unreachable after the drain above, but keep the client.
            logger.debug('[App] broadcast: queue still full after drain')


async def _watch_live_json(app: FastAPI, interval: float = WATCHER_INTERVAL_SEC) -> None:
    """Polls mtime; on change reads + validates + broadcasts to SSE clients."""
    while not app.state.watcher_stop.is_set():
        try:
            if LIVE_JSON.is_file():
                mtime = LIVE_JSON.stat().st_mtime
                if mtime != app.state.last_mtime:
                    app.state.last_mtime = mtime
                    payload = await _read_live_json()
                    if payload is not None:
                        try:
                            LivePredictions.model_validate(payload)
                        except Exception as e:
                            logger.warning('[App] invalid live_predictions.json — skip: %s', e)
                        else:
                            await _broadcast(payload)
        except Exception as e:
            logger.warning('[App] watcher error: %s', e)
        try:
            await asyncio.wait_for(app.state.watcher_stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


# ── /api/config ─────────────────────────────────────────────────────────────

@app.get('/api/config', response_model=AppConfig)
async def api_config() -> AppConfig:
    cfg = app.state.config
    return AppConfig(
        symbols        = [a['symbol'] for a in cfg.get('actors', [])],
        horizons_hours = list(cfg['prediction']['horizons_hours']),
        quantiles      = list(cfg['prediction'].get('quantiles', [])),
        thresholds_pct = list(cfg['prediction'].get('thresholds_pct', [])),
        direction      = cfg['prediction'].get('direction', 'both'),
    )


# ── /api/runs, /api/calibration, /api/strategy ──────────────────────────────

def _list_runs() -> list[str]:
    if not CHECKPOINTS.is_dir():
        return []
    runs = [p for p in CHECKPOINTS.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime)  # oldest -> newest
    return [p.name for p in runs]


def _latest_json(run_id: str, prefix: str) -> Optional[Path]:
    run_dir = CHECKPOINTS / run_id
    if not run_dir.is_dir():
        return None
    exact = run_dir / f'{prefix}.json'
    if exact.is_file():
        return exact
    candidates = []
    for p in run_dir.glob(f'{prefix}_epoch*.json'):
        try:
            candidates.append((int(p.stem.removeprefix(f'{prefix}_epoch')), p))
        except ValueError:
            continue
    return max(candidates)[1] if candidates else None


def _serve_run_json(run: Optional[str], prefix: str) -> FileResponse:
    runs = _list_runs()
    if not runs:
        raise HTTPException(404, 'no runs available')
    if run is None:
        rid = runs[-1]               # newest by mtime
    else:
        # Whitelist: reject anything not in _list_runs() to block path traversal.
        if run not in runs:
            raise HTTPException(404, f'run {run!r} not found')
        rid = run
    path = _latest_json(rid, prefix)
    if path is None:
        raise HTTPException(404, f'{prefix} not found for {rid}')
    return FileResponse(path, media_type='application/json')


@app.get('/api/runs')
async def api_runs() -> list[str]:
    return _list_runs()


@app.get('/api/calibration')
async def api_calibration_latest():
    return _serve_run_json(None, 'calibration')


@app.get('/api/calibration/{run}')
async def api_calibration(run: str):
    return _serve_run_json(run, 'calibration')


@app.get('/api/strategy')
async def api_strategy_latest():
    return _serve_run_json(None, 'strategy_top100')


@app.get('/api/strategy/{run}')
async def api_strategy(run: str):
    return _serve_run_json(run, 'strategy_top100')


# ── /api/klines (CORS-free Binance proxy) ───────────────────────────────────

@app.get('/api/klines')
async def api_klines(
    symbol:   str = Query(..., min_length=4, max_length=16, pattern=r'^[A-Z0-9]+$'),
    interval: str = Query(...),
    limit:    int = Query(1000, ge=1, le=KLINES_HARD_LIMIT),
    end_time: Optional[int] = Query(None, alias='endTime', ge=0),
):
    """Server-side Binance klines proxy. Validates input to prevent
    abuse as an open Binance forwarder. `limit` capped at KLINES_HARD_LIMIT."""
    if not SYMBOL_RE.match(symbol):
        raise HTTPException(400, 'invalid symbol')
    if interval not in TF_TO_BINANCE:
        raise HTTPException(400, f'unsupported interval (allowed: {sorted(TF_TO_BINANCE)})')
    try:
        rows = await asyncio.to_thread(
            fetch_klines_paginated, symbol, interval, limit, end_time,
        )
    except Exception as e:
        raise HTTPException(502, f'Binance: {e}') from e
    return rows


# ── /api/kronos/* (foundation-model continuation + bruteforce) ──────────────

KRONOS_MAX_PRED_LEN     = 1024     # safety cap to prevent runaway requests
KRONOS_MAX_SIMULATIONS  = 500      # cap bruteforce loop


def _candles_to_df(candles: list[list[float]]) -> pd.DataFrame:
    """Binance-style 12-col rows → minimal OHLCV DataFrame with timestamp."""
    if not candles:
        raise HTTPException(400, 'empty history')
    rows = []
    for k in candles:
        rows.append({
            'timestamp': pd.to_datetime(int(k[0]), unit='ms'),
            'open':   float(k[1]),
            'high':   float(k[2]),
            'low':    float(k[3]),
            'close':  float(k[4]),
            'volume': float(k[5]),
        })
    return pd.DataFrame(rows)


def _df_to_kline_rows(df: pd.DataFrame, step_ms: int) -> list[list]:
    """Predicted DataFrame (indexed by timestamp) → Binance-shaped rows for the chart."""
    out = []
    for ts, row in df.iterrows():
        t_ms = int(pd.Timestamp(ts).timestamp() * 1000)
        o, h, l, c = float(row['open']), float(row['high']), float(row['low']), float(row['close'])
        v = max(0.0, float(row.get('volume', 0.0)))
        # Mimic Binance 12-col layout (only first 6 read by the chart).
        out.append([t_ms, o, h, l, c, v, t_ms + step_ms - 1, 0.0, 0, 0.0, 0.0, '0'])
    return out


def _validate_model(name: Optional[str]) -> str:
    if not name:
        return DEFAULT_MODEL
    if name not in KRONOS_MODELS:
        raise HTTPException(400, f'unknown model {name!r}; choose: {list(KRONOS_MODELS)}')
    return name


def _validate_sampling(T: float, top_p: float) -> None:
    if not (T > 0.0 and T <= 5.0):
        raise HTTPException(400, 'T must be in (0, 5]')
    if not (0.0 < top_p <= 1.0):
        raise HTTPException(400, 'top_p must be in (0, 1]')


@app.get('/api/kronos/models')
async def api_kronos_models():
    """List available Kronos model variants and which ones are already cached locally."""
    return {
        'models': [
            {'name': name, 'repo': repo, 'tokenizer': tok, 'max_context': ctx,
             'loaded': name in _predictors}
            for name, (repo, tok, ctx) in KRONOS_MODELS.items()
        ],
    }


@app.post('/api/kronos/generate')
async def api_kronos_generate(payload: dict = Body(...)):
    """Generate `pred_len` future candles given a history list.

    Body: {history: [[ts_ms, o, h, l, c, v, ...], ...], pred_len, T?, top_p?, seed?}
    Returns: {candles: [...binance-shape rows...], step_ms}
    """
    history = payload.get('history') or []
    pred_len = int(payload.get('pred_len') or 0)
    if pred_len <= 0 or pred_len > KRONOS_MAX_PRED_LEN:
        raise HTTPException(400, f'pred_len must be 1..{KRONOS_MAX_PRED_LEN}')
    T     = float(payload.get('T', 1.0))
    top_p = float(payload.get('top_p', 0.9))
    _validate_sampling(T, top_p)
    seed  = payload.get('seed')
    seed  = int(seed) if seed is not None else None
    model_name = _validate_model(payload.get('model'))

    df = _candles_to_df(history)
    if len(df) < 32:
        raise HTTPException(400, 'history too short (need ≥32 candles)')

    step_ms = int((df['timestamp'].iloc[-1] - df['timestamp'].iloc[-2]).total_seconds() * 1000)
    if step_ms <= 0:
        raise HTTPException(400, 'history has zero/negative timestep')

    def _run():
        pred_df = generate_continuation(
            history=df, pred_len=pred_len, T=T, top_p=top_p, seed=seed,
            model_name=model_name,
        )
        return _df_to_kline_rows(pred_df, step_ms)

    try:
        candles = await asyncio.to_thread(_run)
    except Exception as e:
        logger.exception('[Kronos] generate failed: %s', e)
        raise HTTPException(500, f'kronos: {e}') from e
    return {'candles': candles, 'step_ms': step_ms}


@app.post('/api/kronos/bruteforce')
async def api_kronos_bruteforce(payload: dict = Body(...)):
    """Run N stochastic Kronos rollouts and rank (direction × horizon × threshold) combos.

    Streams newline-delimited JSON: {"progress": i, "total": n} lines while
    simulating, then a final {"result": {...}} line. Frontend can parse
    incrementally to show a progress bar.

    Body: {history, pred_len, n_simulations?, T?, top_p?,
           horizons_bars?, thresholds_pct?}
    """
    history = payload.get('history') or []
    pred_len = int(payload.get('pred_len') or 0)
    if pred_len <= 0 or pred_len > KRONOS_MAX_PRED_LEN:
        raise HTTPException(400, f'pred_len must be 1..{KRONOS_MAX_PRED_LEN}')
    n_sims = int(payload.get('n_simulations') or 100)
    if n_sims <= 0 or n_sims > KRONOS_MAX_SIMULATIONS:
        raise HTTPException(400, f'n_simulations must be 1..{KRONOS_MAX_SIMULATIONS}')
    T     = float(payload.get('T', 1.0))
    top_p = float(payload.get('top_p', 0.9))
    _validate_sampling(T, top_p)
    model_name = _validate_model(payload.get('model'))

    df = _candles_to_df(history)
    if len(df) < 32:
        raise HTTPException(400, 'history too short (need ≥32 candles)')

    horizons_bars   = payload.get('horizons_bars')   or None
    thresholds_pct  = payload.get('thresholds_pct')  or None

    progress_q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _on_progress(i: int, n: int) -> None:
        loop.call_soon_threadsafe(progress_q.put_nowait, {'progress': i, 'total': n})

    def _run():
        return bruteforce_strategies(
            history=df, pred_len=pred_len, n_simulations=n_sims,
            T=T, top_p=top_p,
            horizons_bars=horizons_bars,
            thresholds_pct=thresholds_pct,
            progress_cb=_on_progress,
            model_name=model_name,
        )

    async def _stream():
        task = asyncio.create_task(asyncio.to_thread(_run))
        try:
            while not task.done():
                try:
                    msg = await asyncio.wait_for(progress_q.get(), timeout=1.0)
                    yield json.dumps(msg) + '\n'
                except asyncio.TimeoutError:
                    yield ''  # heartbeat keeps the connection alive
            # Drain any remaining progress events.
            while not progress_q.empty():
                yield json.dumps(progress_q.get_nowait()) + '\n'
            result = await task
            yield json.dumps({'result': result}) + '\n'
        except Exception as e:
            logger.exception('[Kronos] bruteforce failed: %s', e)
            yield json.dumps({'error': f'kronos: {e}'}) + '\n'

    return StreamingResponse(_stream(), media_type='application/x-ndjson')


# ── /api/live/predict-range (per-candle continuous predictions) ─────────────

_TF_MS = {'1m': 60_000, '5m': 300_000, '15m': 900_000,
           '1h': 3_600_000, '4h': 14_400_000, '1d': 86_400_000}


@app.get('/api/live/predict-range')
async def api_predict_range(
    symbol:    str = Query(..., min_length=4, max_length=16, pattern=r'^[A-Z0-9]+$'),
    from_ms:   int = Query(...),
    to_ms:     int = Query(...),
    horizon_h: int = Query(8),
    tf:        str = Query('15m'),
) -> dict:
    """Serve precomputed per-candle predictions from memory.

    from_ms / to_ms are open_time_ms (chart visible range).
    Response: {close_time_s: {long: {p10: float, …}, short: {…}}}
    """
    if not app.state.precompute_done:
        raise HTTPException(503, 'warming up — precomputation in progress')

    sym_preds = app.state.all_predictions.get(symbol)
    if sym_preds is None:
        raise HTTPException(404, f'no predictions for {symbol}')

    horizons  = list(app.state.config['prediction']['horizons_hours'])
    if horizon_h not in horizons:
        horizon_h = horizons[0]

    # Convert visible open_time range → close_time range so it matches stored keys.
    tf_ms          = _TF_MS.get(tf, 900_000)
    close_from_s   = (from_ms + tf_ms - 1) // 1000
    close_to_s     = (to_ms   + tf_ms - 1) // 1000

    out: dict[str, dict] = {}
    for t_s, horizons_data in sym_preds.items():
        t = int(t_s)
        if close_from_s <= t <= close_to_s:
            h_data = horizons_data.get(horizon_h)
            if h_data:
                out[t_s] = h_data
    return out


# ── /api/live (SSE) ─────────────────────────────────────────────────────────

@app.get('/api/live')
async def api_live(request: Request):
    if len(app.state.clients) >= MAX_SSE_CLIENTS:
        raise HTTPException(503, 'too many SSE clients')
    queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=4)
    # Snapshot BEFORE registering: avoids the race where a broadcast lands
    # between register and snapshot, causing the same payload to be sent twice.
    snapshot = app.state.last_payload
    app.state.clients.add(queue)

    async def event_gen():
        try:
            if snapshot is not None:
                yield {'data': json.dumps(snapshot, separators=(',', ':'))}
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=15)
                    yield {'data': json.dumps(payload, separators=(',', ':'))}
                except asyncio.TimeoutError:
                    yield {'event': 'ping', 'data': ''}
        finally:
            app.state.clients.discard(queue)

    return EventSourceResponse(event_gen())


# ── Static / SPA ────────────────────────────────────────────────────────────

_MIME = {
    '.html': 'text/html; charset=utf-8',
    '.js':   'application/javascript',
    '.mjs':  'application/javascript',
    '.css':  'text/css',
    '.json': 'application/json',
    '.png':  'image/png',
    '.svg':  'image/svg+xml',
    '.ico':  'image/x-icon',
    '.woff': 'font/woff',
    '.woff2':'font/woff2',
    '.map':  'application/json',
}


def _resolve_asset(path: str) -> Optional[Path]:
    rel = path.lstrip('/')
    for base in (DIST_DIR, PUBLIC_DIR):
        base_resolved = base.resolve()
        full = (base / rel).resolve()
        # Accept only files that live under base_resolved (handles the
        # `full == base_resolved` and the symlink-escape edge cases).
        if full.is_file() and base_resolved in (full, *full.parents):
            return full
    return None


@app.get('/{full_path:path}', include_in_schema=False)
async def spa(full_path: str):
    if full_path.startswith('api/') or full_path == 'api':
        raise HTTPException(404, 'unknown api endpoint')
    path = '/' + full_path
    asset = _resolve_asset(path if path != '/' else '/index.html')
    if asset is None:
        # SPA fallback — let the React router handle the route
        asset = _resolve_asset('/index.html')
        if asset is None:
            raise HTTPException(404, 'dashboard not built — run `npm run build`')
    mime = _MIME.get(asset.suffix.lower(), 'application/octet-stream')
    return FileResponse(asset, media_type=mime)


# ── Entrypoint ──────────────────────────────────────────────────────────────

def run(port: int = 8080, host: str = '127.0.0.1') -> None:
    import uvicorn
    if not DIST_DIR.is_dir():
        logger.warning('[App] dashboard/dist missing — build with `cd dashboard && npm run build`')
    uvicorn.run(app, host=host, port=port, log_level='info')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--host', type=str, default='127.0.0.1',
                    help='Bind host (default 127.0.0.1; pass 0.0.0.0 for LAN access)')
    args = ap.parse_args()
    run(args.port, args.host)
