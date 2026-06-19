from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from freegrad.execution import SingleCPUExecutionDriver
from freegrad.learning import adam, sgd
from freegrad.learning.base import RepairResult, SearchMemory, UpdateSignal
from freegrad.learning.conditioning import AdamConditioner
from freegrad.learning.policies import NoOpRepair
from freegrad.losses.classification import CrossEntropyLoss
from freegrad.metrics.classification import ClassificationMetrics
from freegrad.models.mlp import SimpleMLP


def _batch():
    return {
        "x": jnp.ones((4, 28, 28, 1), dtype=jnp.float32),
        "y": jnp.array([0, 1, 2, 3], dtype=jnp.int32),
    }


def _driver_and_variables():
    model = SimpleMLP()
    variables = model.init(jax.random.PRNGKey(0), input_shape=(28, 28, 1), num_classes=10)
    driver = SingleCPUExecutionDriver(
        model=model,
        loss=CrossEntropyLoss(),
        metric=ClassificationMetrics(),
        micro_batch_size=2,
    )
    return driver, variables


def test_generic_learning_stack_expresses_sgd():
    driver, variables = _driver_and_variables()
    stack = sgd(learning_rate=1e-2)
    state = stack.init(variables)

    next_variables, next_state, metrics = stack.step(
        variables=variables,
        state=state,
        driver=driver,
        batch=_batch(),
    )

    assert isinstance(next_state.search_memory, SearchMemory)
    assert "step_size/value" in metrics
    assert any(
        not jnp.array_equal(before, after)
        for before, after in zip(jax.tree_util.tree_leaves(variables.params), jax.tree_util.tree_leaves(next_variables.params))
    )


def test_generic_learning_stack_expresses_adam_like_conditioning():
    driver, variables = _driver_and_variables()
    stack = adam(learning_rate=1e-3)
    state = stack.init(variables)

    _, next_state, metrics = stack.step(variables=variables, state=state, driver=driver, batch=_batch())

    assert isinstance(stack.conditioner, AdamConditioner)
    assert int(next_state.conditioner_state.count) == 1
    assert "conditioning/adam_count" in metrics


@dataclass(frozen=True)
class RecordingRepair(NoOpRepair):
    def repair(
        self,
        *,
        variables,
        search_memory,
        raw_gradient,
        conditioned_gradient,
        update_signal: UpdateSignal,
        objective,
        conditioning_info: Any,
    ) -> RepairResult:
        assert update_signal.updates is not None
        assert jax.tree_util.tree_structure(raw_gradient) == jax.tree_util.tree_structure(conditioned_gradient)
        return RepairResult(
            variables=variables,
            search_memory=SearchMemory(data={"repaired": jnp.asarray(1)}),
            metrics={"repair/saw_step": jnp.asarray(1)},
        )


def test_step_size_runs_before_repair_and_repair_does_not_mutate_gradient():
    driver, variables = _driver_and_variables()
    stack = sgd(learning_rate=1e-2)
    stack = stack.__class__(
        name="sgd_with_repair",
        probe_point=stack.probe_point,
        conditioner=stack.conditioner,
        step_size=stack.step_size,
        repair=RecordingRepair(),
        update=stack.update,
    )
    state = stack.init(variables)
    _, next_state, metrics = stack.step(variables=variables, state=state, driver=driver, batch=_batch())

    assert int(next_state.search_memory.data["repaired"]) == 1
    assert int(metrics["repair/saw_step"]) == 1
