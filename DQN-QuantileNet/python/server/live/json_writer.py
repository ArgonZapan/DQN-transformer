"""Validate against schema and atomically replace live_predictions.json.

Validation step rejects malformed payloads BEFORE the rename, so SSE clients
never see a partial / out-of-spec snapshot.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from ..schemas import LivePredictions, SCHEMA_VERSION

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
LIVE_JSON_PATH = os.path.join(_PROJECT_ROOT, 'python', 'live_predictions.json')


def write_atomic(predictions: list[dict], checkpoint_path: str,
                 path: str = LIVE_JSON_PATH) -> dict:
    """Validate and atomically write live_predictions.json.

    Raises pydantic.ValidationError on schema violation — caller decides how to
    surface that (current loop logs the error and keeps the previous file).
    """
    payload = {
        'schema_version':   SCHEMA_VERSION,
        'generated_at':     datetime.now(timezone.utc).isoformat(),
        'model_checkpoint': os.path.basename(checkpoint_path),
        'predictions':      predictions,
    }

    LivePredictions.model_validate(payload)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    # Compact separators — payload is hot-pathed through SSE on every refresh,
    # indent=2 inflated it 2-3x. Fsync before rename so a crash can't leave a
    # tmp with buffered-only data that os.replace then promotes to the real path.
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, separators=(',', ':'))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    logger.info('[Writer] → %s  (%d symbols)', path, len(predictions))
    return payload
