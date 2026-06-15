"""Path helpers for runtime outputs."""

from __future__ import annotations

from pathlib import Path


def condition_dir(root_output_dir: Path, condition_name: str) -> Path:
    return root_output_dir / "conditions" / condition_name


def run_dir(root_output_dir: Path, condition_name: str, seed: int) -> Path:
    return condition_dir(root_output_dir, condition_name) / "runs" / f"seed_{seed}"


def checkpoints_dir(output_dir: Path) -> Path:
    return output_dir / "checkpoints"


def study_manifest_path(study_output_dir: Path) -> Path:
    return study_output_dir / "study.json"