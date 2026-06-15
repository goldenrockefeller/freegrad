from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from freegrad.losses.classification import build_cross_entropy_loss
from freegrad.metrics.classification import build_classification_metrics
from freegrad.models.mlp import SimpleMLP
from freegrad.optimizers.adam import Adam
from freegrad.runtime.condition import ConditionSpec
from freegrad.runtime.backends.local import LocalBackend


def test_smoke_condition_runs_and_writes_artifacts(tmp_path):
    rng = np.random.default_rng(0)

    def dataset_factory():
        return {
            "train_images": rng.normal(size=(32, 28, 28, 1)).astype(np.float32),
            "train_labels": rng.integers(0, 10, size=(32,), dtype=np.int32),
            "test_images": rng.normal(size=(16, 28, 28, 1)).astype(np.float32),
            "test_labels": rng.integers(0, 10, size=(16,), dtype=np.int32),
        }

    condition = ConditionSpec(
        name="smoke_mnist_mlp_adam",
        seeds=[0],
        dataset_loader=dataset_factory,
        data_preparer_builder=lambda: (lambda raw_data, validation_size=8: {
            "train_images": raw_data["train_images"][:-validation_size],
            "train_labels": raw_data["train_labels"][:-validation_size],
            "validation_images": raw_data["train_images"][-validation_size:],
            "validation_labels": raw_data["train_labels"][-validation_size:],
            "test_images": raw_data["test_images"],
            "test_labels": raw_data["test_labels"],
        }),
        model_builder=lambda: SimpleMLP(),
        optimizer_builder=lambda: Adam(learning_rate=1e-3),
        loss_builder=build_cross_entropy_loss,
        metrics_builder=build_classification_metrics,
        training_config={
            "mini_batch_size": 4,
            "macro_batch_size": 8,
            "eval_mini_batch_size": 4,
            "eval_macro_batch_size": 8,
            "max_steps": 4,
            "train_chunk_size": 2,
            "eval_every": 1,
            "validation_size": 8,
            "n_chunks_per_checkpoint": 1,
            "resume": False,
        },
    )

    backend = LocalBackend(root_output_dir=tmp_path)
    results = backend.run_condition(condition)
    run_dir = tmp_path / "smoke_mnist_mlp_adam" / "conditions" / "smoke_mnist_mlp_adam" / "runs" / "seed_0"

    assert results[0].status == "COMPLETED"
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "status.json").exists()
    assert (run_dir / "run.json").exists()
    assert any((run_dir / "checkpoints").glob("step_*.pkl"))

    status_payload = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status_payload["status"] == "COMPLETED"

    metric_rows = _read_metrics(run_dir / "metrics.jsonl")
    assert len(metric_rows) == 4
    assert [row["step"] for row in metric_rows] == [2, 2, 4, 4]
    assert ["train/loss" in row for row in metric_rows] == [True, False, True, False]
    assert ["eval/loss" in row for row in metric_rows] == [False, True, False, True]


def test_legacy_training_config_keys_raise_clear_errors():
    from freegrad.runtime.run import _parse_training_config

    with pytest.raises(ValueError, match="batch_size -> mini_batch_size"):
        _parse_training_config({"batch_size": 4})


def test_training_config_rejects_unknown_keys():
    from freegrad.runtime.run import _parse_training_config

    with pytest.raises(ValueError, match="unsupported keys: mystery"):
        _parse_training_config(
            {
                "mini_batch_size": 4,
                "macro_batch_size": 8,
                "eval_mini_batch_size": 4,
                "eval_macro_batch_size": 8,
                "max_steps": 4,
                "train_chunk_size": 2,
                "eval_every": 1,
                "n_chunks_per_checkpoint": 1,
                "resume": False,
                "mystery": 1,
            }
        )


def test_training_config_requires_explicit_keys():
    from freegrad.runtime.run import _parse_training_config

    with pytest.raises(ValueError, match="missing required keys"):
        _parse_training_config({"mini_batch_size": 4})


def _read_metrics(path: Path) -> list[dict[str, float]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]