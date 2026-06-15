"""Plot generation for study outputs."""

from __future__ import annotations

import json
from pathlib import Path

# TODO, make plots as training it happening, have a separate n_chunk_per_plot variable
def make_study_plots(study_dir: Path) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to generate plots.") from exc

    plots_dir = study_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    first_metrics = _find_first_metrics_file(study_dir)
    if first_metrics is not None:
        records = [json.loads(line) for line in first_metrics.read_text(encoding="utf-8").splitlines() if line.strip()]
        if records:
            steps = [row["step"] for row in records]
            train_losses = [row.get("train/loss") for row in records if "train/loss" in row]
            train_accuracies = [row.get("train/accuracy") for row in records if "train/accuracy" in row]
            eval_accuracies = [row.get("eval/accuracy") for row in records if "eval/accuracy" in row]

            if train_losses:
                plt.figure()
                plt.plot(steps[: len(train_losses)], train_losses)
                plt.xlabel("step")
                plt.ylabel("train/loss")
                plt.tight_layout()
                plt.savefig(plots_dir / "train_loss_vs_step.png")
                plt.close()

            if train_accuracies:
                plt.figure()
                plt.plot(steps[: len(train_accuracies)], train_accuracies)
                plt.xlabel("step")
                plt.ylabel("train/accuracy")
                plt.tight_layout()
                plt.savefig(plots_dir / "train_accuracy_vs_step.png")
                plt.close()

            if eval_accuracies:
                eval_steps = [row["step"] for row in records if "eval/accuracy" in row]
                plt.figure()
                plt.plot(eval_steps, eval_accuracies)
                plt.xlabel("step")
                plt.ylabel("eval/accuracy")
                plt.tight_layout()
                plt.savefig(plots_dir / "eval_accuracy_vs_step.png")
                plt.close()

    summary_rows = _load_summary_rows(study_dir / "summary.csv")
    if summary_rows:
        plt.figure()
        plt.bar([row["condition_name"] for row in summary_rows], [float(row["mean_final_accuracy"]) for row in summary_rows])
        plt.ylabel("mean_final_accuracy")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(plots_dir / "condition_final_accuracy_bar.png")
        plt.close()

    return plots_dir


def _find_first_metrics_file(study_dir: Path) -> Path | None:
    for path in sorted(study_dir.glob("conditions/*/runs/seed_*/metrics.jsonl")):
        return path
    return None


def _load_summary_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return []
    header = lines[0].split(",")
    rows = []
    for line in lines[1:]:
        values = line.split(",")
        rows.append(dict(zip(header, values)))
    return rows