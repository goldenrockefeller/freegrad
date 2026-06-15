from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

from freegrad.losses.classification import build_cross_entropy_loss
from freegrad.metrics.classification import build_classification_metrics
from freegrad.models.mlp import SimpleMLP
from freegrad.optimizers.adam import Adam
from freegrad.runtime.backends.base import StudyConfirmationHandler
from freegrad.runtime.backends.local import LocalBackend
from freegrad.runtime.condition import ConditionSpec
from freegrad.runtime.checkpointing.pickle_fallback import latest_checkpoint, load_checkpoint
from freegrad.runtime.git import GitProvenance, StudyExecutionAborted
from freegrad.runtime.paths import study_manifest_path
from freegrad.runtime.study import StudyExecutionPolicy, StudySpec


def _dataset_factory(seed: int):
    def factory():
        rng = np.random.default_rng(seed)
        return {
            "train_images": rng.normal(size=(24, 28, 28, 1)).astype(np.float32),
            "train_labels": rng.integers(0, 10, size=(24,), dtype=np.int32),
            "test_images": rng.normal(size=(8, 28, 28, 1)).astype(np.float32),
            "test_labels": rng.integers(0, 10, size=(8,), dtype=np.int32),
        }

    return factory


def _prep_factory():
    return lambda raw_data, validation_size=8: {
        "train_images": raw_data["train_images"][:-validation_size],
        "train_labels": raw_data["train_labels"][:-validation_size],
        "validation_images": raw_data["train_images"][-validation_size:],
        "validation_labels": raw_data["train_labels"][-validation_size:],
        "test_images": raw_data["test_images"],
        "test_labels": raw_data["test_labels"],
    }


def _make_condition(name: str, *, max_steps: int, dataset_seed: int) -> ConditionSpec:
    return ConditionSpec(
        name=name,
        seeds=[0],
        dataset_loader=_dataset_factory(dataset_seed),
        data_preparer_builder=_prep_factory,
        model_builder=lambda: SimpleMLP(),
        optimizer_builder=lambda: Adam(learning_rate=1e-3),
        loss_builder=build_cross_entropy_loss,
        metrics_builder=build_classification_metrics,
        training_config={
            "mini_batch_size": 4,
            "macro_batch_size": 8,
            "eval_mini_batch_size": 4,
            "eval_macro_batch_size": 8,
            "max_steps": max_steps,
            "train_chunk_size": 2,
            "eval_every": 1,
            "validation_size": 8,
            "n_chunks_per_checkpoint": 1,
            "resume": False,
        },
    )


class StubConfirmationHandler(StudyConfirmationHandler):
    def __init__(self, response: bool):
        self.response = response
        self.messages: list[str] = []

    def confirm_study_execution(self, message: str) -> bool:
        self.messages.append(message)
        return self.response


def test_checkpoint_resume_reaches_expected_step(tmp_path):
    condition = _make_condition("resume_demo", max_steps=2, dataset_seed=1)
    backend = LocalBackend(root_output_dir=tmp_path)
    backend.run_condition(condition)

    resumed = ConditionSpec(
        name=condition.name,
        seeds=condition.seeds,
        dataset_loader=condition.dataset_loader,
        data_preparer_builder=condition.data_preparer_builder,
        model_builder=condition.model_builder,
        optimizer_builder=condition.optimizer_builder,
        loss_builder=condition.loss_builder,
        metrics_builder=condition.metrics_builder,
        training_config={**condition.training_config, "max_steps": 4, "resume": True},
    )
    backend.run_condition(resumed)

    run_dir = tmp_path / "resume_demo" / "conditions" / "resume_demo" / "runs" / "seed_0"
    checkpoint_path = latest_checkpoint(run_dir / "checkpoints")
    state, _ = load_checkpoint(checkpoint_path)

    assert int(np.asarray(state.step)) == 4


def test_study_resume_skips_completed_runs_and_resumes_incomplete_runs(tmp_path):
    backend = LocalBackend(root_output_dir=tmp_path)
    study_name = "resume_study"

    initial_git = GitProvenance(branch="main", commit_sha="abc123", is_dirty=False)
    backend._current_git_provenance = lambda: initial_git

    completed_condition = _make_condition("completed", max_steps=4, dataset_seed=10)
    incomplete_condition = _make_condition("incomplete", max_steps=2, dataset_seed=20)
    initial_study = StudySpec(name=study_name, conditions=[completed_condition, incomplete_condition])
    backend.run_study(initial_study)

    study_root = tmp_path / study_name
    completed_run_dir = study_root / "conditions" / "completed" / "runs" / "seed_0"
    incomplete_run_dir = study_root / "conditions" / "incomplete" / "runs" / "seed_0"
    completed_metrics_before = (completed_run_dir / "metrics.jsonl").read_text(encoding="utf-8")

    run_payload = json.loads((incomplete_run_dir / "run.json").read_text(encoding="utf-8"))
    run_payload["status"] = "FAILED"
    run_payload["final_metrics"] = {}
    (incomplete_run_dir / "run.json").write_text(json.dumps(run_payload, indent=2), encoding="utf-8")

    status_payload = json.loads((incomplete_run_dir / "status.json").read_text(encoding="utf-8"))
    status_payload["status"] = "FAILED"
    status_payload["message"] = "interrupted"
    (incomplete_run_dir / "status.json").write_text(json.dumps(status_payload, indent=2), encoding="utf-8")

    resumed_study = StudySpec(
        name=study_name,
        conditions=[
            _make_condition("completed", max_steps=4, dataset_seed=10),
            _make_condition("incomplete", max_steps=4, dataset_seed=20),
        ],
    )

    resumed_git = GitProvenance(branch="experiment", commit_sha="def456", is_dirty=False)
    confirmation_handler = StubConfirmationHandler(response=True)
    backend._current_git_provenance = lambda: resumed_git
    backend.confirmation_handler = confirmation_handler
    results = backend.run_study(resumed_study)

    assert [result.status for result in results] == ["COMPLETED", "COMPLETED"]
    assert len(confirmation_handler.messages) == 1
    assert "different git provenance" in confirmation_handler.messages[0]
    assert (completed_run_dir / "metrics.jsonl").read_text(encoding="utf-8") == completed_metrics_before

    checkpoint_path = latest_checkpoint(incomplete_run_dir / "checkpoints")
    state, metadata = load_checkpoint(checkpoint_path)
    assert int(np.asarray(state.step)) == 4
    assert metadata["study_name"] == study_name

    resumed_run_payload = json.loads((incomplete_run_dir / "run.json").read_text(encoding="utf-8"))
    assert resumed_run_payload["study_name"] == study_name
    assert resumed_run_payload["status"] == "COMPLETED"

    manifest_payload = json.loads(study_manifest_path(study_root).read_text(encoding="utf-8"))
    assert manifest_payload["name"] == study_name
    assert manifest_payload["status"] == "COMPLETED"
    assert manifest_payload["git"] == resumed_git.to_dict()
    assert {run["name"]: run["status"] for run in manifest_payload["runs"]} == {
        "completed_seed_0": "COMPLETED",
        "incomplete_seed_0": "COMPLETED",
    }


def test_new_study_on_dirty_branch_requires_confirmation(tmp_path):
    backend = LocalBackend(root_output_dir=tmp_path, confirmation_handler=StubConfirmationHandler(response=False))
    backend._current_git_provenance = lambda: GitProvenance(branch="main", commit_sha="abc123", is_dirty=True)

    study = StudySpec(name="dirty_study", conditions=[_make_condition("dirty", max_steps=2, dataset_seed=30)])

    with pytest.raises(StudyExecutionAborted):
        backend.run_study(study)

    assert not study_manifest_path(tmp_path / study.name).exists()


def test_resume_mismatch_aborts_without_interactive_terminal(tmp_path):
    backend = LocalBackend(root_output_dir=tmp_path)
    backend._current_git_provenance = lambda: GitProvenance(branch="main", commit_sha="abc123", is_dirty=False)

    initial_study = StudySpec(name="noninteractive_resume", conditions=[_make_condition("resume", max_steps=2, dataset_seed=40)])
    backend.run_study(initial_study)

    study_root = tmp_path / initial_study.name
    run_dir = study_root / "conditions" / "resume" / "runs" / "seed_0"
    run_payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    run_payload["status"] = "FAILED"
    run_payload["final_metrics"] = {}
    (run_dir / "run.json").write_text(json.dumps(run_payload, indent=2), encoding="utf-8")

    status_payload = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    status_payload["status"] = "FAILED"
    status_payload["message"] = "interrupted"
    (run_dir / "status.json").write_text(json.dumps(status_payload, indent=2), encoding="utf-8")

    backend._current_git_provenance = lambda: GitProvenance(branch="other", commit_sha="def456", is_dirty=False)

    resumed_study = StudySpec(name="noninteractive_resume", conditions=[_make_condition("resume", max_steps=4, dataset_seed=40)])

    with pytest.raises(StudyExecutionAborted):
        backend.run_study(resumed_study)


def test_study_policy_can_disable_dirty_branch_confirmation(tmp_path):
    backend = LocalBackend(root_output_dir=tmp_path)
    backend._current_git_provenance = lambda: GitProvenance(branch="main", commit_sha="abc123", is_dirty=True)

    study = StudySpec(
        name="policy_override",
        conditions=[_make_condition("dirty", max_steps=2, dataset_seed=50)],
        execution_policy=StudyExecutionPolicy(require_clean_git_for_new_study=False),
    )

    results = backend.run_study(study)

    assert [result.status for result in results] == ["COMPLETED"]
    manifest_payload = json.loads(study_manifest_path(tmp_path / study.name).read_text(encoding="utf-8"))
    assert manifest_payload["git"]["is_dirty"] is True


def test_legacy_checkpoint_schema_is_rejected(tmp_path):
    checkpoint_path = tmp_path / "legacy.pkl"
    with checkpoint_path.open("wb") as handle:
        pickle.dump({"state": object(), "metadata": {}}, handle)

    with pytest.raises(ValueError, match="Unsupported checkpoint schema version"):
        load_checkpoint(checkpoint_path)