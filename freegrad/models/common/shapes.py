"""Shape helpers."""

from __future__ import annotations


def num_features(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total