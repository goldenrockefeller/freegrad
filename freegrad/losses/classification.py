"""Classification losses."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from freegrad.losses.base import LossContribution, MeanLossReduction
from freegrad.models.common.base import Model, ModelMode, ModelVariables


def cross_entropy_from_logits(logits: jax.Array, labels: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.mean(log_probs[jnp.arange(labels.shape[0]), labels])


def cross_entropy_sum_from_logits(logits: jax.Array, labels: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.sum(log_probs[jnp.arange(labels.shape[0]), labels])


@dataclass(frozen=True)
class CrossEntropyLoss(MeanLossReduction):
    def apply(
        self,
        *,
        model: Model,
        variables: ModelVariables,
        batch: dict[str, jax.Array],
        mode: ModelMode,
        rng_key: jax.Array | None = None,
    ) -> LossContribution:
        result = model.apply(variables, batch["x"], mode=mode, rng_key=rng_key)
        labels = batch["y"]
        return LossContribution(
            total=cross_entropy_sum_from_logits(result.output, labels),
            count=jnp.asarray(labels.shape[0], dtype=jnp.float32),
        )


def build_cross_entropy_loss(model_apply: Callable[..., jax.Array]):
    def loss_fn(variables: Any, inputs: jax.Array, targets: jax.Array) -> jax.Array:
        result = model_apply(variables, inputs)
        logits = getattr(result, "output", result)
        return cross_entropy_from_logits(logits, targets)

    return loss_fn
