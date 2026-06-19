"""Eager single-device execution driver.

The driver is intentionally named for a physical execution strategy. Algorithm
semantics live in ``freegrad.learning``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from freegrad.execution.base import BatchApplyFn, BatchFinalizeFn, BatchReduceFn, ObjectiveEvaluation
from freegrad.losses.base import ReducibleLoss
from freegrad.metrics.base import ReducibleMetric
from freegrad.models.common.base import Model, ModelMode, ModelVariables


@dataclass(frozen=True)
class SingleCPUExecutionDriver:
    model: Model
    loss: ReducibleLoss
    metric: ReducibleMetric | None = None
    micro_batch_size: int | None = None

    def value(
        self,
        variables: ModelVariables,
        batch: dict[str, jax.Array],
        *,
        rng_key: jax.Array | None = None,
    ) -> jax.Array:
        return self._effective_loss(variables, batch, ModelMode.TRAIN, rng_key)

    def grad(
        self,
        variables: ModelVariables,
        batch: dict[str, jax.Array],
        *,
        rng_key: jax.Array | None = None,
    ) -> Any:
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
        value, grad = jax.value_and_grad(
            lambda params: self.value(variables.replace(params=params), batch, rng_key=rng_key)
        )(variables.params)
        metric_values = self.metrics(variables, batch, rng_key=rng_key)
        return ObjectiveEvaluation(value=value, grad=grad, metrics=metric_values, model_state=variables.state)

    def hvp(
        self,
        variables: ModelVariables,
        batch: dict[str, jax.Array],
        vector: Any,
        *,
        rng_key: jax.Array | None = None,
    ) -> tuple[jax.Array, Any]:
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
        del rng_key
        if self.metric is None:
            return {}
        return self.apply_batch(
            batch,
            apply=lambda micro_batch: self.metric.apply(
                model=self.model,
                variables=variables,
                batch=micro_batch,
                mode=ModelMode.EVAL,
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

    def _effective_loss(
        self,
        variables: ModelVariables,
        batch: dict[str, jax.Array],
        mode: ModelMode,
        rng_key: jax.Array | None,
    ) -> jax.Array:
        del rng_key
        return self.apply_batch(
            batch,
            apply=lambda micro_batch: self.loss.apply(
                model=self.model,
                variables=variables,
                batch=micro_batch,
                mode=mode,
            ),
            reduce=self.loss.reduce,
            finalize=self.loss.finalize,
        )


def _leading_axis_size(batch: dict[str, jax.Array]) -> int:
    leaves = jax.tree_util.tree_leaves(batch)
    if not leaves:
        raise ValueError("Expected a non-empty batch.")
    return int(jnp.asarray(leaves[0]).shape[0])


def _slice_batch(batch: dict[str, jax.Array], start: int, stop: int) -> dict[str, jax.Array]:
    return jax.tree_util.tree_map(lambda leaf: jnp.asarray(leaf)[start:stop], batch)
