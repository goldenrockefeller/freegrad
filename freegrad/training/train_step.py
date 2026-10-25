"""Factories for train-chunk execution."""

from __future__ import annotations

import jax

from freegrad.execution.base import ExecutionDriver, TrainChunkSpec
from freegrad.learning.base import LearningStack
from freegrad.training.state import TrainState
from freegrad.utils.pytrees import tree_l2_norm


def _train_step_impl(state: TrainState, batch, driver: ExecutionDriver, learning_stack: LearningStack):
    next_rng, _ = jax.random.split(state.rng_key)
    next_variables, next_learning_state, learning_metrics = learning_stack.step(
        variables=state.variables,
        state=state.learning_state,
        driver=driver,
        batch=batch,
        rng_key=state.rng_key,
    )
    grads = driver.grad(state.variables, batch, rng_key=state.rng_key)
    updates = jax.tree_util.tree_map(lambda new, old: new - old, next_variables.params, state.variables.params)
    metrics = {
        "grad_norm": tree_l2_norm(grads),
        "update_norm": tree_l2_norm(updates),
        "param_norm": tree_l2_norm(next_variables.params),
        **learning_metrics,
    }
    new_state = state.replace(
        step=state.step + 1,
        variables=next_variables,
        learning_state=next_learning_state,
        rng_key=next_rng,
    )
    return new_state, metrics


def make_train_update_chunk_step(
    driver: ExecutionDriver,
    learning_stack: LearningStack,
    chunk_spec: TrainChunkSpec | None = None,
):
    if chunk_spec is None:
        chunk_spec = TrainChunkSpec(jittable=learning_stack.jittable)

    def update_step(state: TrainState, batch):
        return _train_step_impl(state, batch, driver, learning_stack)

    return driver.compile_train_chunk(update_step, chunk_spec)
