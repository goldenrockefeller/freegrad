"""Array container helpers."""

from __future__ import annotations

from typing import Any


def ensure_array_dict(**arrays: Any) -> dict[str, Any]:
    return dict(arrays)