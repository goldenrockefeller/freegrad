"""Metric aggregation helpers."""

from __future__ import annotations

from collections.abc import Iterable


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)