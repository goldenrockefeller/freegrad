from __future__ import annotations

from pathlib import Path

from freegrad.learning import sgd
from freegrad.losses.classification import CrossEntropyLoss
from freegrad.metrics.classification import ClassificationMetrics
from freegrad.runtime.condition import ConditionSpec
from freegrad.runtime.run import RunSpec
from freegrad.runtime.study import StudySpec


def _dummy_condition(name: str, seeds: list[int]) -> ConditionSpec:
    return ConditionSpec(
        name=name,
        seeds=seeds,
        dataset_loader=lambda: None,
        data_preparer_builder=lambda: (lambda raw_data, validation_size=5000: raw_data),
        model_builder=lambda: None,
        learning_stack_builder=lambda: sgd(0.1),
        loss_builder=lambda: CrossEntropyLoss(),
        metrics_builder=lambda: ClassificationMetrics(),
        training_config={"max_steps": 1},
    )


def test_condition_expands_runs_deterministically():
    condition = _dummy_condition("demo", [0, 1, 2])

    runs = condition.expand_runs(Path("results") / "demo")

    assert len(runs) == 3
    assert [run.name for run in runs] == ["demo_seed_0", "demo_seed_1", "demo_seed_2"]
    assert runs[0].study_name is None
    assert str(runs[0].output_dir).endswith("results\\demo\\conditions\\demo\\runs\\seed_0")


def test_study_expands_all_runs_with_unique_names():
    study = StudySpec(name="study", conditions=[_dummy_condition("a", [0, 1, 2]), _dummy_condition("b", [0, 1, 2])])

    runs = study.expand_runs(Path("results") / "study")

    assert len(runs) == 6
    assert len({run.name for run in runs}) == 6
    assert {run.study_name for run in runs} == {"study"}


def test_runspec_defaults_study_name_for_direct_construction():
    run_spec = RunSpec(
        name="demo_seed_0",
        condition_name="demo",
        seed=0,
        output_dir=Path("results") / "demo",
        dataset_loader=lambda: None,
        data_preparer_builder=lambda: (lambda raw_data, validation_size=5000: raw_data),
        model_builder=lambda: None,
        learning_stack_builder=lambda: sgd(0.1),
        loss_builder=lambda: CrossEntropyLoss(),
        metrics_builder=lambda: ClassificationMetrics(),
        training_config={"max_steps": 1},
    )

    assert run_spec.study_name is None
