"""Simple pickle checkpoint helpers."""

from __future__ import annotations

import pickle
from pathlib import Path


CHECKPOINT_SCHEMA_VERSION = 2


def save_checkpoint(path: Path, state, metadata: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "state": state,
                "metadata": metadata or {},
            },
            handle,
        )
    return path


def load_checkpoint(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    schema_version = payload.get("schema_version")
    if schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported checkpoint schema version: {schema_version!r}. "
            f"Expected {CHECKPOINT_SCHEMA_VERSION}."
        )
    return payload["state"], payload["metadata"]


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.exists():
        return None
    candidates = sorted(checkpoint_dir.glob("step_*.pkl"))
    if not candidates:
        return None
    return candidates[-1]