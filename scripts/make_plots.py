"""Generate plots for a study directory."""

from __future__ import annotations

import sys
from pathlib import Path

from freegrad.runtime.reporting.plots import make_study_plots


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("Usage: python scripts/make_plots.py results/<study_name>")
        return 1
    study_dir = Path(argv[0])
    if not study_dir.exists():
        print(f"Study path does not exist: {study_dir}")
        return 1
    make_study_plots(study_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())