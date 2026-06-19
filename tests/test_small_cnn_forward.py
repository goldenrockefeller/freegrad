from __future__ import annotations

import jax
import jax.numpy as jnp

from freegrad.models.small_cnn import ModernSmallCNN


def test_small_cnn_forward_shape():
    model = ModernSmallCNN()
    x = jnp.ones((4, 28, 28, 1), dtype=jnp.float32)
    variables = model.init(jax.random.PRNGKey(0), input_shape=(28, 28, 1), num_classes=10)

    result = model.apply(variables, x)

    assert result.output.shape == (4, 10)
    assert result.model_state == {}
    leaves = jax.tree_util.tree_leaves(variables.params)
    assert leaves
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
