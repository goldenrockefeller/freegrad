"""Study manifest helpers for run discovery and resume."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from freegrad.runtime.git import GitProvenance
from freegrad.runtime.checkpointing.pickle_fallback import latest_checkpoint
from freegrad.runtime.paths import checkpoints_dir, study_manifest_path
from freegrad.runtime.run import RunResult, RunSpec


@dataclass(frozen=True)
class StudyManifest:
    name: str
    output_dir: str
    status: str
    num_runs: int
    git_provenance: GitProvenance | None
    runs: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "output_dir": self.output_dir,
            "status": self.status,
            "num_runs": self.num_runs,
            "runs": self.runs,
        }
        if self.git_provenance is not None:
            payload["git"] = self.git_provenance.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StudyManifest:
        return cls(
            name=str(payload["name"]),
            output_dir=str(payload["output_dir"]),
            status=str(payload["status"]),
            num_runs=int(payload["num_runs"]),
            git_provenance=GitProvenance.from_dict(payload.get("git")),
            runs=list(payload.get("runs", [])),
        )


def snapshot_run(run_spec: RunSpec) -> dict[str, Any]:
    run_payload = _load_json(run_spec.output_dir / "run.json")
    status_payload = _load_json(run_spec.output_dir / "status.json")
    checkpoint_path = latest_checkpoint(checkpoints_dir(run_spec.output_dir))
    status = str(run_payload.get("status") or status_payload.get("status") or "PENDING")
    return {
        "name": run_spec.name,
        "condition_name": run_spec.condition_name,
        "study_name": run_spec.study_name,
        "seed": run_spec.seed,
        "output_dir": str(run_spec.output_dir),
        "status": status,
        "step": status_payload.get("step"),
        "final_metrics": run_payload.get("final_metrics", {}),
        "error": run_payload.get("error") or status_payload.get("message"),
        "has_checkpoint": checkpoint_path is not None,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
    }


def load_study_manifest(study_output_dir: Path) -> StudyManifest | None:
    path = study_manifest_path(study_output_dir)
    if not path.exists():
        return None
    return StudyManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_study_git_provenance(study_output_dir: Path) -> GitProvenance | None:
    manifest = load_study_manifest(study_output_dir)
    if manifest is None:
        return None
    return manifest.git_provenance


def write_study_manifest(
    study_name: str,
    study_output_dir: Path,
    run_specs: list[RunSpec],
    *,
    git_provenance: GitProvenance,
) -> Path:
    manifest = StudyManifest(
        name=study_name,
        output_dir=str(study_output_dir),
        status=_study_status(run_specs),
        num_runs=len(run_specs),
        git_provenance=git_provenance,
        runs=[snapshot_run(run_spec) for run_spec in run_specs],
    )
    path = study_manifest_path(study_output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return path


def is_completed_run(run_spec: RunSpec) -> bool:
    return snapshot_run(run_spec)["status"] == "COMPLETED"


def should_resume_run(run_spec: RunSpec) -> bool:
    snapshot = snapshot_run(run_spec)
    return snapshot["status"] != "COMPLETED" and bool(snapshot["has_checkpoint"])


def skipped_run_result(run_spec: RunSpec) -> RunResult:
    snapshot = snapshot_run(run_spec)
    return RunResult(
        name=run_spec.name,
        status=str(snapshot["status"]),
        output_dir=run_spec.output_dir,
        final_metrics=dict(snapshot["final_metrics"]),
    )


def _study_status(run_specs: list[RunSpec]) -> str:
    statuses = [snapshot_run(run_spec)["status"] for run_spec in run_specs]
    if statuses and all(status == "COMPLETED" for status in statuses):
        return "COMPLETED"
    if any(status == "FAILED" for status in statuses):
        return "FAILED"
    if any(status == "RUNNING" for status in statuses):
        return "RUNNING"
    return "PENDING"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))