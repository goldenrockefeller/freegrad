"""Factory for evaluation steps."""

from __future__ import annotations

import jax

from freegrad.execution.base import ExecutionDriver
from freegrad.models.common.base import ModelVariables


def make_eval_step(driver: ExecutionDriver):
    @jax.jit
    def eval_step(variables: ModelVariables, batch):
        metrics = driver.metrics(variables, batch)
        return {f"eval/{name}": value for name, value in metrics.items()}

    return eval_step
