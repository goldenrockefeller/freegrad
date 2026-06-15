"""Tracking interface definitions."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class MetricLogger(Protocol):
    def log_metrics(self, step: int, metrics: dict) -> None:
        ...

    def close(self) -> None:
        ...