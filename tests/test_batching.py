from __future__ import annotations

import jax.numpy as jnp

from freegrad.data_prep.batching import batched_loss, batched_metrics


def test_batched_loss_matches_full_batch_mean_for_partial_tail():
    def loss_fn(_params, inputs, targets):
        return jnp.mean(inputs + targets)

    wrapped = batched_loss(loss_fn, batch_size=4)
    inputs = jnp.arange(10, dtype=jnp.float32)
    targets = jnp.arange(10, dtype=jnp.float32) * 2.0

    assert jnp.allclose(wrapped(None, inputs, targets), loss_fn(None, inputs, targets))


def test_batched_metrics_matches_full_batch_mean_for_partial_tail():
    def metrics_fn(_params, inputs, targets):
        errors = inputs - targets
        return {
            "mae": jnp.mean(jnp.abs(errors)),
            "bias": jnp.mean(errors),
        }

    wrapped = batched_metrics(metrics_fn, batch_size=3)
    inputs = jnp.arange(8, dtype=jnp.float32)
    targets = jnp.array([0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 5.0, 8.0], dtype=jnp.float32)
    actual = wrapped(None, inputs, targets)
    expected = metrics_fn(None, inputs, targets)

    assert jnp.allclose(actual["mae"], expected["mae"])
    assert jnp.allclose(actual["bias"], expected["bias"])