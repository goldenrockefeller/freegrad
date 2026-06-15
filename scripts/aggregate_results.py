"""Aggregate run outputs into summary.csv."""

from __future__ import annotations

import sys
from pathlib import Path

from freegrad.runtime.reporting.aggregate import aggregate_study


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("Usage: python scripts/aggregate_results.py results/<study_name>")
        return 1
    study_dir = Path(argv[0])
    if not study_dir.exists():
        print(f"Study path does not exist: {study_dir}")
        return 1
    aggregate_study(study_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())