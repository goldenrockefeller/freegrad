"""Training-layer loop owners used by the runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from freegrad.data_prep.batching import ArrayDataLoader
from freegrad.execution.base import ExecutionDriver
from freegrad.models.common.base import ModelMode
from freegrad.training.state import TrainState


@dataclass(frozen=True)
class TrainChunkResult:
    state: TrainState
    metrics: dict[str, Any]
    chunk_steps: int


class TrainChunkRunner:
    def __init__(
        self,
        *,
        arrays: dict[str, np.ndarray],
        macro_batch_size: int,
        train_chunk_size: int,
        max_steps: int,
        seed: int,
        train_update_chunk_step,
    ):
        if macro_batch_size <= 0:
            raise ValueError("macro_batch_size must be positive.")
        if train_chunk_size <= 0:
            raise ValueError("train_chunk_size must be positive.")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if int(arrays["x"].shape[0]) < macro_batch_size:
            raise ValueError("train set must contain at least one full macro batch.")

        self.arrays = arrays
        self.macro_batch_size = macro_batch_size
        self.train_chunk_size = train_chunk_size
        self.max_steps = max_steps
        self.seed = seed
        self.train_update_chunk_step = train_update_chunk_step
        self._epoch = 0
        self._loader = iter(self._make_loader())

    def has_next_chunk(self, state: TrainState) -> bool:
        return int(np.asarray(state.step)) < self.max_steps

    def run_next_chunk(self, state: TrainState) -> TrainChunkResult:
        current_step = int(np.asarray(state.step))
        remaining_steps = self.max_steps - current_step
        if remaining_steps <= 0:
            raise ValueError("No train chunk remains to execute.")

        chunk_steps = min(self.train_chunk_size, remaining_steps)
        macro_batches = [self._next_macro_batch() for _ in range(chunk_steps)]
        next_state, metrics = self.train_update_chunk_step(state, _stack_batches(macro_batches))
        return TrainChunkResult(state=next_state, metrics=metrics, chunk_steps=chunk_steps)

    def _next_macro_batch(self) -> dict[str, np.ndarray]:
        while True:
            try:
                return next(self._loader)
            except StopIteration:
                self._epoch += 1
                self._loader = iter(self._make_loader())

    def _make_loader(self):
        return ArrayDataLoader(
            arrays=self.arrays,
            batch_size=self.macro_batch_size,
            shuffle=True,
            seed=self.seed + self._epoch,
            drop_last=True,
        )


class ValidationRunner:
    def __init__(
        self,
        *,
        arrays: dict[str, np.ndarray],
        macro_batch_size: int,
        eval_step=None,
        execution_driver: ExecutionDriver | None = None,
    ):
        if macro_batch_size <= 0:
            raise ValueError("macro_batch_size must be positive.")
        self.arrays = arrays
        self.macro_batch_size = macro_batch_size
        self.eval_step = eval_step
        self.execution_driver = execution_driver

    def evaluate(self, variables) -> dict[str, Any]:
        loader = ArrayDataLoader(
            arrays=self.arrays,
            batch_size=self.macro_batch_size,
            shuffle=False,
            seed=0,
            drop_last=False,
        )
        if self.execution_driver is not None and getattr(self.execution_driver, "metric", None) is not None:
            total_contribution = None
            metric = getattr(self.execution_driver, "metric")
            model = getattr(self.execution_driver, "model")
            for batch in loader:
                contribution = self.execution_driver.apply_batch(
                    batch,
                    apply=lambda micro_batch: metric.apply(
                        model=model,
                        variables=variables,
                        batch=micro_batch,
                        mode=ModelMode.EVAL,
                    ),
                    reduce=metric.reduce,
                )
                total_contribution = (
                    contribution
                    if total_contribution is None
                    else metric.reduce(total_contribution, contribution)
                )
            if total_contribution is None:
                return {"eval/loss": 0.0}
            return {f"eval/{key}": value for key, value in metric.finalize(total_contribution).items()}

        if self.eval_step is None:
            raise ValueError("ValidationRunner requires eval_step or execution_driver.")

        total_examples = 0
        weighted_totals: dict[str, float] = {}
        for batch in loader:
            batch_metrics = self.eval_step(variables, batch)
            batch_size_actual = int(np.asarray(batch["y"]).shape[0])
            total_examples += batch_size_actual
            for key, value in batch_metrics.items():
                weighted_totals[key] = weighted_totals.get(key, 0.0) + (
                    float(np.asarray(value)) * batch_size_actual
                )

        if total_examples == 0:
            return {"eval/loss": 0.0}

        return {key: total / total_examples for key, total in weighted_totals.items()}


def take_steps(train_loader, train_step, state, max_steps: int):
    metrics_history = []
    for batch in train_loader:
        state, metrics = train_step(state, batch)
        metrics_history.append(metrics)
        if state.step >= max_steps:
            break
    return state, metrics_history


def _stack_batches(batches: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        key: np.stack([batch[key] for batch in batches], axis=0)
        for key in batches[0]
    }
