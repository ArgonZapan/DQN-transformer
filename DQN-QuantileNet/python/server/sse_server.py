"""Deprecated entrypoint — kept so existing scripts keep working.

The server now lives in `python.server.app` (FastAPI). Run it with:
    python -m python.server.app [--port 8080]
"""

from .app import run


if __name__ == '__main__':
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    logging.getLogger(__name__).warning(
        'python.server.sse_server is deprecated — use `python -m python.server.app` instead.'
    )
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8080)
    run(ap.parse_args().port)
