"""Cross-platform terminal dashboard for freegrad run statuses."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class StatusRow:
    condition: str
    run: str
    status: str
    step: str
    message: str
    updated: str


def collect_statuses(root: Path) -> list[StatusRow]:
    rows = []
    for path in sorted(root.rglob("status.json")) if root.exists() else []:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        condition, run = _condition_and_run(path)
        modified = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
        rows.append(
            StatusRow(
                condition=condition,
                run=run,
                status=str(payload.get("status", "UNKNOWN")),
                step=str(payload.get("step", "-")),
                message=str(payload.get("message") or ""),
                updated=modified.strftime("%H:%M:%S"),
            )
        )
    return rows


def render_dashboard(root: Path, rows: list[StatusRow]) -> str:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [f"freegrad status | {timestamp}", f"root: {root.resolve()}"]
    if not rows:
        lines.append("No status.json files found. Waiting for runs to start...")
        return "\n".join(lines)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    summary = " ".join(f"{status.lower()}={count}" for status, count in sorted(counts.items()))
    lines.append(f"runs={len(rows)} {summary}")
    lines.append("")

    headers = ("Condition", "Run", "Status", "Step", "Updated", "Message")
    values = [
        (row.condition, row.run, row.status, row.step, row.updated, _truncate(row.message, 60))
        for row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in values))
        for index in range(len(headers))
    ]
    lines.append(_format_row(headers, widths))
    lines.append(_format_row(tuple("-" * width for width in widths), widths))
    lines.extend(_format_row(row, widths) for row in values)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Watch freegrad run status files.")
    parser.add_argument(
        "path",
        nargs="?",
        default="results",
        type=Path,
        help="Study or results directory to watch (default: results).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Refresh interval in seconds (default: 2).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print one snapshot and exit.",
    )
    args = parser.parse_args(argv)
    if args.interval <= 0:
        parser.error("--interval must be positive")

    try:
        while True:
            dashboard = render_dashboard(args.path, collect_statuses(args.path))
            if not args.once and sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            print(dashboard, flush=True)
            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped status dashboard.")
        return 0


def _condition_and_run(status_path: Path) -> tuple[str, str]:
    parts = status_path.parts
    try:
        condition_index = parts.index("conditions") + 1
        run_index = parts.index("runs", condition_index) + 1
        return parts[condition_index], parts[run_index]
    except (ValueError, IndexError):
        return status_path.parent.parent.name, status_path.parent.name


def _format_row(values: tuple[str, ...], widths: list[int]) -> str:
    return "  ".join(value.ljust(width) for value, width in zip(values, widths))


def _truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"


if __name__ == "__main__":
    raise SystemExit(main())
