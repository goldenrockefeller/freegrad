"""Parameter initialization helpers."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def glorot_uniform(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    fan_in, fan_out = _fan_in_out(shape)
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit, dtype=jnp.float32)


def kaiming_uniform(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    fan_in, _ = _fan_in_out(shape)
    limit = math.sqrt(6.0 / fan_in)
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit, dtype=jnp.float32)


def zeros(shape: tuple[int, ...]) -> jax.Array:
    return jnp.zeros(shape, dtype=jnp.float32)


def _fan_in_out(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) < 2:
        raise ValueError(f"Expected at least 2 dimensions, got shape={shape!r}")
    if len(shape) == 2:
        return shape[0], shape[1]
    receptive_field = math.prod(shape[:-2])
    return receptive_field * shape[-2], receptive_field * shape[-1]