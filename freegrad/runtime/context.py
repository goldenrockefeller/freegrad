"""Runtime context objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunContext:
    output_dir: Path
    checkpoint_dir: Path
    metrics_path: Path
    status_path: Path
    run_path: Path