"""Run a study from an import path."""

from __future__ import annotations

import sys

from freegrad.runtime.backends.base import StudyConfirmationHandler
from freegrad.runtime.backends.local import LocalBackend
from freegrad.runtime.git import StudyExecutionAborted
from freegrad.utils.imports import import_from_string


class CLIStudyConfirmationHandler(StudyConfirmationHandler):
    def confirm_study_execution(self, message: str) -> bool:
        if not sys.stdin or not sys.stdin.isatty():
            raise StudyExecutionAborted(f"{message}\nConfirmation required, but no interactive terminal is available.")

        print(message)
        while True:
            response = input("Continue study execution? [y/N]: ").strip().lower()
            if response in {"", "n", "no"}:
                return False
            if response in {"y", "yes"}:
                return True
            print("Please answer 'y' or 'n'.")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("Usage: python scripts/run_study.py module.submodule:factory")
        return 1
    try:
        factory = import_from_string(argv[0])
        study = factory()
    except Exception as exc:
        print(f"Failed to load study factory: {exc}")
        return 1
    backend = LocalBackend(confirmation_handler=CLIStudyConfirmationHandler())
    try:
        results = backend.run_study(study)
    except StudyExecutionAborted as exc:
        print(str(exc))
        return 1
    failed = [result for result in results if result.status != "COMPLETED"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())