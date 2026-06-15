"""Simple multilayer perceptron for MNIST."""

from __future__ import annotations

import jax

from freegrad.models.common.base import Model
from freegrad.models.common.init import glorot_uniform, zeros
from freegrad.models.common.layers import flatten, linear, relu


class SimpleMLP(Model):
    def init(self, key: jax.Array, input_shape: tuple[int, ...], num_classes: int) -> dict[str, dict[str, jax.Array]]:
        flat_dim = input_shape[0] * input_shape[1] * input_shape[2]
        keys = jax.random.split(key, 3)
        return {
            "dense1": {
                "w": glorot_uniform(keys[0], (flat_dim, 256)),
                "b": zeros((256,)),
            },
            "dense2": {
                "w": glorot_uniform(keys[1], (256, 128)),
                "b": zeros((128,)),
            },
            "dense3": {
                "w": glorot_uniform(keys[2], (128, num_classes)),
                "b": zeros((num_classes,)),
            },
        }

    def apply(self, params: dict[str, dict[str, jax.Array]], x: jax.Array, training: bool = False, rng_key: jax.Array | None = None) -> jax.Array:
        del training, rng_key
        x = flatten(x)
        x = relu(linear(params["dense1"], x))
        x = relu(linear(params["dense2"], x))
        return linear(params["dense3"], x)