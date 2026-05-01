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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

from .live.binance_client import fetch_klines_paginated, TF_TO_BINANCE
from .schemas             import AppConfig, LivePredictions

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Read mtime BEFORE payload — otherwise a write between the two reads
    # would leave us with stale payload paired with the new mtime.
    app.state.last_mtime   = LIVE_JSON.stat().st_mtime if LIVE_JSON.is_file() else 0.0
    app.state.last_payload = await _read_live_json()
    app.state.config       = _load_config()
    app.state.clients      = set()           # set[asyncio.Queue]
    app.state.watcher_stop = asyncio.Event()
    app.state.watcher_task = asyncio.create_task(_watch_live_json(app))
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
