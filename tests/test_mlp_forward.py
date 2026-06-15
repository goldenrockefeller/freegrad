from __future__ import annotations

import jax
import jax.numpy as jnp

from freegrad.models.mlp import SimpleMLP


def test_simple_mlp_forward_shape():
    model = SimpleMLP()
    x = jnp.ones((4, 28, 28, 1), dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(0), input_shape=(28, 28, 1), num_classes=10)

    logits = model.apply(params, x)

    assert logits.shape == (4, 10)
    leaves = jax.tree_util.tree_leaves(params)
    assert leaves
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)