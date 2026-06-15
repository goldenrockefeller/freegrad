"""Classification losses."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp


def cross_entropy_from_logits(logits: jax.Array, labels: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.mean(log_probs[jnp.arange(labels.shape[0]), labels])


def build_cross_entropy_loss(model_apply: Callable[..., jax.Array]):
    def loss_fn(params: Any, inputs: jax.Array, targets: jax.Array) -> jax.Array:
        logits = model_apply(params, inputs)
        return cross_entropy_from_logits(logits, targets)

    return loss_fn