"""Metrics aggregation for runs, conditions, and studies."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


def load_metrics_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def aggregate_run_final_metrics(run_dirs: list[Path]) -> list[dict]:
    rows = []
    for run_dir in run_dirs:
        run_path = run_dir / "run.json"
        if not run_path.exists():
            continue
        payload = json.loads(run_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "condition_name": payload["condition_name"],
                "run_name": payload["name"],
                "seed": payload["seed"],
                **payload.get("final_metrics", {}),
            }
        )
    return rows


def aggregate_condition(condition_dir: Path) -> dict:
    run_dirs = sorted((condition_dir / "runs").glob("seed_*"))
    rows = aggregate_run_final_metrics(run_dirs)
    accuracies = [row.get("eval/accuracy", 0.0) for row in rows]
    losses = [row.get("eval/loss", 0.0) for row in rows]
    return {
        "condition_name": condition_dir.name,
        "num_runs": len(rows),
        "mean_final_accuracy": _safe_mean(accuracies),
        "std_final_accuracy": _safe_std(accuracies),
        "mean_final_loss": _safe_mean(losses),
        "std_final_loss": _safe_std(losses),
        "best_final_accuracy": max(accuracies) if accuracies else 0.0,
    }


def aggregate_study(study_dir: Path) -> list[dict]:
    conditions_root = study_dir / "conditions"
    summaries = [aggregate_condition(condition_dir) for condition_dir in sorted(conditions_root.iterdir()) if condition_dir.is_dir()]
    if summaries:
        summary_path = study_dir / "summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
    return summaries


def _safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0