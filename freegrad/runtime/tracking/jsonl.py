"""JSONL metrics logging."""

from __future__ import annotations

import json
from pathlib import Path

from freegrad.runtime.tracking.base import MetricLogger

class JSONLLogger(MetricLogger):
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def log_metrics(self, step: int, metrics: dict):
        payload = {"step": step, **metrics}
        self._handle.write(json.dumps(payload) + "\n")
        self._handle.flush()

    def close(self):
        self._handle.close()