"""Status file writing."""

from __future__ import annotations

import json
from pathlib import Path


VALID_STATUSES = {"PENDING", "RUNNING", "COMPLETED", "FAILED"}


def write_status(output_dir: Path, status: str, message: str | None = None, step: int | None = None) -> None:
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status: {status!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"status": status, "message": message, "step": step}
    (output_dir / "status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")