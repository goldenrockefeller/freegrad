"""Normalization helpers."""

from __future__ import annotations

import numpy as np


def normalize_with_stats(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (values - mean) / std