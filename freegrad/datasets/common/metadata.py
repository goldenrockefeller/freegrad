"""Dataset metadata helpers."""

from __future__ import annotations


def make_metadata(name: str, num_classes: int, input_shape: tuple[int, ...]) -> dict[str, object]:
    return {
        "name": name,
        "num_classes": num_classes,
        "input_shape": input_shape,
    }