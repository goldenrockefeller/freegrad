"""Run a condition from an import path."""

from __future__ import annotations

from datetime import datetime
import sys

from freegrad.runtime.backends.local import LocalBackend
from freegrad.utils.imports import import_from_string


def report_progress(message: str) -> None:
    timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("Usage: python -m scripts.run_condition module.submodule:factory")
        return 1
    try:
        factory = import_from_string(argv[0])
        condition = factory()
    except Exception as exc:
        print(f"Failed to load condition factory: {exc}")
        return 1
    backend = LocalBackend(progress_callback=report_progress)
    results = backend.run_condition(condition)
    failed = [result for result in results if result.status != "COMPLETED"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
