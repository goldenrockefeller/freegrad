"""Condition specification and run expansion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from freegrad.runtime.interfaces import DataPreparerBuilder, DatasetLoader, LossBuilder, MetricsBuilder, ModelBuilder, OptimizerBuilder
from freegrad.runtime.paths import run_dir
from freegrad.runtime.run import RunSpec


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    seeds: list[int]
    dataset_loader: DatasetLoader
    data_preparer_builder: DataPreparerBuilder
    model_builder: ModelBuilder
    optimizer_builder: OptimizerBuilder
    loss_builder: LossBuilder
    metrics_builder: MetricsBuilder
    training_config: dict

    def expand_runs(self, root_output_dir: Path, study_name: str | None = None) -> list[RunSpec]:
        runs = []
        for seed in self.seeds:
            runs.append(
                RunSpec(
                    name=f"{self.name}_seed_{seed}",
                    condition_name=self.name,
                    study_name=study_name,
                    seed=seed,
                    output_dir=run_dir(root_output_dir, self.name, seed),
                    dataset_loader=self.dataset_loader,
                    data_preparer_builder=self.data_preparer_builder,
                    model_builder=self.model_builder,
                    optimizer_builder=self.optimizer_builder,
                    loss_builder=self.loss_builder,
                    metrics_builder=self.metrics_builder,
                    training_config=dict(self.training_config),
                )
            )
        return runs