"""Classification metrics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from freegrad.losses.classification import cross_entropy_from_logits


def accuracy_from_logits(logits: jax.Array, labels: jax.Array) -> jax.Array:
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == labels)


def build_accuracy_metric(model_apply: Callable[..., jax.Array]):
    def metric_fn(params: Any, inputs: jax.Array, targets: jax.Array) -> jax.Array:
        logits = model_apply(params, inputs)
        return accuracy_from_logits(logits, targets)

    return metric_fn


def build_classification_metrics(model_apply: Callable[..., jax.Array]):
    def metrics_fn(params: Any, inputs: jax.Array, targets: jax.Array) -> dict[str, jax.Array]:
        logits = model_apply(params, inputs)
        return {
            "loss": cross_entropy_from_logits(logits, targets),
            "accuracy": accuracy_from_logits(logits, targets),
        }

    return metrics_fn