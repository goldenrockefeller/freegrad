"""Classification metrics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from freegrad.losses.classification import cross_entropy_from_logits, cross_entropy_sum_from_logits
from freegrad.metrics.base import MeanMetricReduction, MetricContribution
from freegrad.models.common.base import Model, ModelMode, ModelVariables


def accuracy_from_logits(logits: jax.Array, labels: jax.Array) -> jax.Array:
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == labels)


@dataclass(frozen=True)
class ClassificationMetrics(MeanMetricReduction):
    def apply(
        self,
        *,
        model: Model,
        variables: ModelVariables,
        batch: dict[str, jax.Array],
        mode: ModelMode,
        rng_key: jax.Array | None = None,
    ) -> MetricContribution:
        result = model.apply(variables, batch["x"], mode=mode, rng_key=rng_key)
        labels = batch["y"]
        predictions = jnp.argmax(result.output, axis=-1)
        count = jnp.asarray(labels.shape[0], dtype=jnp.float32)
        return MetricContribution(
            totals={
                "loss": cross_entropy_sum_from_logits(result.output, labels),
                "accuracy": jnp.sum(predictions == labels),
            },
            count=count,
        )


def build_accuracy_metric(model_apply: Callable[..., jax.Array]):
    def metric_fn(variables: Any, inputs: jax.Array, targets: jax.Array) -> jax.Array:
        result = model_apply(variables, inputs)
        logits = getattr(result, "output", result)
        return accuracy_from_logits(logits, targets)

    return metric_fn


def build_classification_metrics(model_apply: Callable[..., jax.Array]):
    def metrics_fn(variables: Any, inputs: jax.Array, targets: jax.Array) -> dict[str, jax.Array]:
        result = model_apply(variables, inputs)
        logits = getattr(result, "output", result)
        return {
            "loss": cross_entropy_from_logits(logits, targets),
            "accuracy": accuracy_from_logits(logits, targets),
        }

    return metrics_fn
