"""
MetricLogger — zapisuje metryki treningowe do JSON Lines co N kroków.
Format: jedna linia JSON per rekord, append-only.
"""

import json
import math
import os
import time
from collections import deque


class MetricLogger:
    def __init__(self, path: str, flush_every: int = 100):
        self.path = path
        self.flush_every = flush_every
        self._buffer = []
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fh = open(path, 'a', encoding='utf-8', buffering=1)

    def log(self, step: int, metrics: dict) -> None:
        record = {'step': step, 'ts': time.time()}
        for k, v in metrics.items():
            if isinstance(v, float):
                if math.isnan(v):
                    record[k] = 'nan'
                elif math.isinf(v):
                    record[k] = 'inf' if v > 0 else '-inf'
                else:
                    record[k] = v
            else:
                record[k] = v
        self._buffer.append(record)
        if len(self._buffer) >= self.flush_every:
            self._flush()

    def _flush(self) -> None:
        for record in self._buffer:
            self._fh.write(json.dumps(record, ensure_ascii=False) + '\n')
        self._buffer.clear()

    def read_window(self, last_n: int = 1000) -> list:
        """Wczytuje ostatnie last_n rekordów z pliku."""
        self._flush()
        window = deque(maxlen=last_n)
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        window.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        except FileNotFoundError:
            return []
        return list(window)

    def close(self) -> None:
        self._flush()
        self._fh.close()
