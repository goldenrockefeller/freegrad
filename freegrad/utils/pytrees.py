"""Helpers for common pytree operations."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def tree_zeros_like(tree: Any) -> Any:
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def tree_add(a: Any, b: Any) -> Any:
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def tree_scalar_mul(scalar: float | jax.Array, tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: scalar * x, tree)


def tree_l2_norm(tree: Any) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.array(0.0, dtype=jnp.float32)
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))