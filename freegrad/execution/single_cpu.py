"""Eager single-device execution driver.

The driver is intentionally named for a physical execution strategy. Algorithm
semantics live in ``freegrad.learning``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from freegrad.execution.base import (
    BatchApplyFn,
    BatchFinalizeFn,
    BatchReduceFn,
    ObjectiveEvaluation,
    TrainChunkSpec,
    TrainUpdateFn,
)
from freegrad.losses.base import ReducibleLoss
from freegrad.metrics.base import ReducibleMetric
from freegrad.models.common.base import Model, ModelMode, ModelVariables


@dataclass(frozen=True)
class SingleCPUExecutionDriver:
    model: Model
    loss: ReducibleLoss
    metric: ReducibleMetric | None = None
    micro_batch_size: int | None = None

    @property
    def device(self) -> jax.Device:
        return jax.devices("cpu")[0]

    def value(
        self,
        variables: ModelVariables,
        batch: dict[str, jax.Array],
        *,
        rng_key: jax.Array | None = None,
    ) -> jax.Array:
        value, _ = self._effective_loss_and_state(variables, batch, ModelMode.TRAIN, rng_key)
        return value

    def grad(
        self,
        variables: ModelVariables,
        batch: dict[str, jax.Array],
        *,
        rng_key: jax.Array | None = None,
    ) -> Any:
        variables = _put_on_device(variables, self.device)
        batch = _put_on_device(batch, self.device)
        rng_key = _put_on_device(rng_key, self.device) if rng_key is not None else None
        return jax.grad(
            lambda params: self.value(variables.replace(params=params), batch, rng_key=rng_key)
        )(variables.params)

    def value_and_grad(
        self,
        variables: ModelVariables,
        batch: dict[str, jax.Array],
        *,
        rng_key: jax.Array | None = None,
    ) -> ObjectiveEvaluation:
        variables = _put_on_device(variables, self.device)
        batch = _put_on_device(batch, self.device)
        rng_key = _put_on_device(rng_key, self.device) if rng_key is not None else None

        def loss_with_state(params):
            return self._effective_loss_and_state(
                variables.replace(params=params),
                batch,
                ModelMode.TRAIN,
                rng_key,
            )

        (value, model_state), grad = jax.value_and_grad(loss_with_state, has_aux=True)(variables.params)
        metric_values = self.metrics(variables, batch, rng_key=rng_key)
        return ObjectiveEvaluation(value=value, grad=grad, metrics=metric_values, model_state=model_state)

    def hvp(
        self,
        variables: ModelVariables,
        batch: dict[str, jax.Array],
        vector: Any,
        *,
        rng_key: jax.Array | None = None,
    ) -> tuple[jax.Array, Any]:
        variables = _put_on_device(variables, self.device)
        batch = _put_on_device(batch, self.device)
        vector = _put_on_device(vector, self.device)
        rng_key = _put_on_device(rng_key, self.device) if rng_key is not None else None

        def loss_fn(candidate):
            return self.value(variables.replace(params=candidate), batch, rng_key=rng_key)

        _, pullback = jax.linearize(jax.grad(loss_fn), variables.params)
        return self.value(variables, batch, rng_key=rng_key), pullback(vector)

    def metrics(
        self,
        variables: ModelVariables,
        batch: dict[str, jax.Array],
        *,
        rng_key: jax.Array | None = None,
    ) -> dict[str, Any]:
        if self.metric is None:
            return {}
        variables = _put_on_device(variables, self.device)
        batch = _put_on_device(batch, self.device)
        rng_key = _put_on_device(rng_key, self.device) if rng_key is not None else None
        return self.apply_batch(
            batch,
            apply=lambda micro_batch: self.metric.apply(
                model=self.model,
                variables=variables,
                batch=micro_batch,
                mode=ModelMode.EVAL,
                rng_key=rng_key,
            ),
            reduce=self.metric.reduce,
            finalize=lambda contribution: dict(self.metric.finalize(contribution)),
        )

    def apply_batch(
        self,
        batch: dict[str, jax.Array],
        *,
        apply: BatchApplyFn,
        reduce: BatchReduceFn,
        finalize: BatchFinalizeFn | None = None,
    ) -> Any:
        batch = _put_on_device(batch, self.device)
        total_examples = _leading_axis_size(batch)
        micro_size = self.micro_batch_size or total_examples
        if micro_size <= 0:
            raise ValueError("micro_batch_size must be positive.")
        if total_examples == 0:
            raise ValueError("apply_batch requires at least one example.")
        if total_examples <= micro_size:
            contribution = apply(batch)
            return finalize(contribution) if finalize is not None else contribution

        contributions: Any | None = None
        start = 0
        while start < total_examples:
            micro_batch = _slice_batch(batch, start, min(start + micro_size, total_examples))
            local = apply(micro_batch)
            contributions = local if contributions is None else reduce(contributions, local)
            start += micro_size
        if contributions is None:
            raise ValueError("apply_batch produced no contributions.")
        return finalize(contributions) if finalize is not None else contributions

    def compile(self, function, *, jittable: bool = True):
        executable = jax.jit(function) if jittable else function

        def execute_on_cpu(*args, **kwargs):
            placed_args = _put_on_device(args, self.device)
            placed_kwargs = _put_on_device(kwargs, self.device)
            return executable(*placed_args, **placed_kwargs)

        return execute_on_cpu

    def compile_train_chunk(
        self,
        update_step: TrainUpdateFn,
        chunk_spec: TrainChunkSpec,
    ):
        def train_chunk(state, update_batches):
            leaves = jax.tree_util.tree_leaves(update_batches)
            if not leaves:
                raise ValueError("update_batches must contain at least one array.")
            actual_updates = int(leaves[0].shape[0])
            if chunk_spec.num_updates is not None and actual_updates > chunk_spec.num_updates:
                raise ValueError(
                    f"Received {actual_updates} updates, exceeding chunk limit {chunk_spec.num_updates}."
                )
            next_state, metrics = jax.lax.scan(update_step, state, update_batches)
            last_metrics = jax.tree_util.tree_map(lambda value: value[-1], metrics)
            return next_state, last_metrics

        return self.compile(train_chunk, jittable=chunk_spec.jittable)

    def _effective_loss_and_state(
        self,
        variables: ModelVariables,
        batch: dict[str, jax.Array],
        mode: ModelMode,
        rng_key: jax.Array | None,
    ) -> tuple[jax.Array, Any]:
        variables = _put_on_device(variables, self.device)
        batch = _put_on_device(batch, self.device)
        rng_key = _put_on_device(rng_key, self.device) if rng_key is not None else None
        total_examples = _leading_axis_size(batch)
        micro_size = self.micro_batch_size or total_examples
        if micro_size <= 0:
            raise ValueError("micro_batch_size must be positive.")
        if total_examples == 0:
            raise ValueError("Loss evaluation requires at least one example.")

        contribution = None
        current_variables = variables
        micro_index = 0
        for start in range(0, total_examples, micro_size):
            micro_batch = _slice_batch(batch, start, min(start + micro_size, total_examples))
            micro_key = jax.random.fold_in(rng_key, micro_index) if rng_key is not None else None
            local = self.loss.apply(
                model=self.model,
                variables=current_variables,
                batch=micro_batch,
                mode=mode,
                rng_key=micro_key,
            )
            next_model_state = getattr(local, "model_state", None)
            if next_model_state is None:
                next_model_state = current_variables.state
            current_variables = current_variables.replace(state=next_model_state)
            contribution = local if contribution is None else self.loss.reduce(contribution, local)
            micro_index += 1

        if contribution is None:
            raise ValueError("Loss evaluation produced no contribution.")
        return self.loss.finalize(contribution), current_variables.state


def _leading_axis_size(batch: dict[str, jax.Array]) -> int:
    leaves = jax.tree_util.tree_leaves(batch)
    if not leaves:
        raise ValueError("Expected a non-empty batch.")
    return int(jnp.asarray(leaves[0]).shape[0])


def _slice_batch(batch: dict[str, jax.Array], start: int, stop: int) -> dict[str, jax.Array]:
    return jax.tree_util.tree_map(lambda leaf: jnp.asarray(leaf)[start:stop], batch)


def _put_on_device(value: Any, device: jax.Device) -> Any:
    return jax.device_put(value, device)
