"""Factory for evaluation steps."""

from __future__ import annotations

from freegrad.execution.base import ExecutionDriver
from freegrad.models.common.base import ModelVariables


def make_eval_step(driver: ExecutionDriver):
    def eval_step(variables: ModelVariables, batch):
        metrics = driver.metrics(variables, batch)
        return {f"eval/{name}": value for name, value in metrics.items()}

    return driver.compile(eval_step)
