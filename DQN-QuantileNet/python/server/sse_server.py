"""Minimal SSE + static-file server for the dashboard.

Serves:
  GET /                          -> dashboard/index.html
  GET /static/*                  -> dashboard/dist/* or dashboard/public/*
  GET /api/live                  -> SSE stream of live_predictions.json
  GET /api/calibration/<run_id>  -> python/checkpoints/<run_id>/calibration.json
  GET /api/strategy/<run_id>     -> python/checkpoints/<run_id>/strategy_top100.json
  GET /api/runs                  -> list of available run_ids

Run with:
    python -m python.server.sse_server          # default port 8080
    python -m python.server.sse_server --port 9000
"""

import argparse
import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CHECKPOINTS    = os.path.join(_PROJECT_ROOT, 'python', 'checkpoints')
LIVE_JSON      = os.path.join(_PROJECT_ROOT, 'python', 'live_predictions.json')
DASHBOARD_DIR  = os.path.join(_PROJECT_ROOT, 'dashboard')
DIST_DIR       = os.path.join(DASHBOARD_DIR, 'dist')    # after `npm run build`
PUBLIC_DIR     = os.path.join(DASHBOARD_DIR, 'public')

MIME = {
    '.html': 'text/html; charset=utf-8',
    '.js':   'application/javascript',
    '.jsx':  'application/javascript',
    '.css':  'text/css',
    '.json': 'application/json',
    '.png':  'image/png',
    '.svg':  'image/svg+xml',
    '.ico':  'image/x-icon',
}

_sse_clients: list = []
_sse_lock = threading.Lock()


def _notify_sse_clients(data: str):
    payload = f'data: {data}\n\n'.encode('utf-8')
    with _sse_lock:
        dead = []
        for wfile in _sse_clients:
            try:
                wfile.write(payload)
                wfile.flush()
            except Exception:
                dead.append(wfile)
        for d in dead:
            _sse_clients.remove(d)


def _watch_live_json(interval: float = 5.0):
    """Background thread: push SSE event when live_predictions.json changes."""
    last_mtime = 0.0
    while True:
        try:
            mtime = os.path.getmtime(LIVE_JSON)
            if mtime != last_mtime:
                last_mtime = mtime
                with open(LIVE_JSON, encoding='utf-8') as f:
                    raw = f.read().replace('\n', '')
                _notify_sse_clients(raw)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f'[SSE] watcher error: {e}')
        time.sleep(interval)


def _list_runs() -> list[str]:
    if not os.path.isdir(CHECKPOINTS):
        return []
    return sorted(
        d for d in os.listdir(CHECKPOINTS)
        if os.path.isdir(os.path.join(CHECKPOINTS, d))
    )


def _serve_static(path: str) -> tuple[int, bytes, str]:
    """Try dist/ then public/ for a static asset. Returns (status, body, mime)."""
    rel  = path.lstrip('/')
    ext  = os.path.splitext(rel)[1].lower()
    mime = MIME.get(ext, 'application/octet-stream')

    for base in (DIST_DIR, PUBLIC_DIR, DASHBOARD_DIR):
        full = os.path.join(base, rel)
        if os.path.isfile(full):
            with open(full, 'rb') as f:
                return 200, f.read(), mime

    return 404, b'Not Found', 'text/plain'


class DashboardHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        logger.debug(f'[HTTP] {self.address_string()} - {fmt % args}')

    def _send(self, status: int, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _json_file(self, path: str):
        if not os.path.isfile(path):
            self._send(404, b'{"error":"not found"}', 'application/json')
            return
        with open(path, 'rb') as f:
            self._send(200, f.read(), 'application/json')

    def do_GET(self):
        parsed = urlparse(self.path)
        p      = parsed.path.rstrip('/')

        # ── SSE stream ────────────────────────────────────────────────────────
        if p == '/api/live':
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            # Send current snapshot immediately
            if os.path.isfile(LIVE_JSON):
                with open(LIVE_JSON, encoding='utf-8') as f:
                    raw = f.read().replace('\n', '')
                self.wfile.write(f'data: {raw}\n\n'.encode('utf-8'))
                self.wfile.flush()

            with _sse_lock:
                _sse_clients.append(self.wfile)

            # Hold connection open until client disconnects
            try:
                while True:
                    time.sleep(30)
                    self.wfile.write(b': keepalive\n\n')
                    self.wfile.flush()
            except Exception:
                with _sse_lock:
                    if self.wfile in _sse_clients:
                        _sse_clients.remove(self.wfile)
            return

        # ── List runs ─────────────────────────────────────────────────────────
        if p == '/api/runs':
            runs = _list_runs()
            body = json.dumps(runs).encode()
            self._send(200, body, 'application/json')
            return

        # ── Calibration JSON ──────────────────────────────────────────────────
        if p.startswith('/api/calibration/'):
            run_id = p[len('/api/calibration/'):]
            self._json_file(os.path.join(CHECKPOINTS, run_id, 'calibration.json'))
            return

        if p == '/api/calibration':
            runs = _list_runs()
            path = os.path.join(CHECKPOINTS, runs[-1], 'calibration.json') if runs else ''
            self._json_file(path)
            return

        # ── Strategy top-100 JSON ─────────────────────────────────────────────
        if p.startswith('/api/strategy/'):
            run_id = p[len('/api/strategy/'):]
            self._json_file(os.path.join(CHECKPOINTS, run_id, 'strategy_top100.json'))
            return

        if p == '/api/strategy':
            runs = _list_runs()
            path = os.path.join(CHECKPOINTS, runs[-1], 'strategy_top100.json') if runs else ''
            self._json_file(path)
            return

        # ── Static files / SPA fallback ───────────────────────────────────────
        asset_path = parsed.path if parsed.path != '/' else '/index.html'
        status, body, mime = _serve_static(asset_path)
        if status == 404 and not parsed.path.startswith('/api'):
            # SPA fallback: return index.html for all non-API routes
            status, body, mime = _serve_static('/index.html')
        self._send(status, body, mime)


def run(port: int = 8080):
    watcher = threading.Thread(target=_watch_live_json, daemon=True, name='sse-watcher')
    watcher.start()

    server = HTTPServer(('0.0.0.0', port), DashboardHandler)
    logger.info(f'[SSE] Dashboard server running at http://localhost:{port}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info('[SSE] Server stopped.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8080)
    args = ap.parse_args()
    run(args.port)
