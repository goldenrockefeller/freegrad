"""Random key helpers."""

from __future__ import annotations

import jax


def make_rng(seed: int) -> jax.Array:
    return jax.random.PRNGKey(seed)


def split_rng(key: jax.Array, num: int = 2) -> tuple[jax.Array, ...]:
    return tuple(jax.random.split(key, num))