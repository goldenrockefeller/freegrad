"""Small convolutional network for MNIST."""

from __future__ import annotations

import math

import jax

from freegrad.models.common.base import Model
from freegrad.models.common.init import glorot_uniform, kaiming_uniform, zeros
from freegrad.models.common.layers import conv2d, flatten, linear, max_pool_2x2, relu


class ModernSmallCNN(Model):
    def init(self, key: jax.Array, input_shape: tuple[int, ...], num_classes: int) -> dict[str, dict[str, jax.Array]]:
        height, width, channels = input_shape
        pooled_height = math.ceil(height / 2)
        pooled_width = math.ceil(width / 2)
        pooled_height = math.ceil(pooled_height / 2)
        pooled_width = math.ceil(pooled_width / 2)
        flat_dim = pooled_height * pooled_width * 128
        keys = jax.random.split(key, 5)
        return {
            "conv1": {
                "w": kaiming_uniform(keys[0], (3, 3, channels, 32)),
                "b": zeros((32,)),
            },
            "conv2": {
                "w": kaiming_uniform(keys[1], (3, 3, 32, 64)),
                "b": zeros((64,)),
            },
            "conv3": {
                "w": kaiming_uniform(keys[2], (3, 3, 64, 128)),
                "b": zeros((128,)),
            },
            "dense1": {
                "w": glorot_uniform(keys[3], (flat_dim, 256)),
                "b": zeros((256,)),
            },
            "dense2": {
                "w": glorot_uniform(keys[4], (256, num_classes)),
                "b": zeros((num_classes,)),
            },
        }

    def apply(self, params: dict[str, dict[str, jax.Array]], x: jax.Array, training: bool = False, rng_key: jax.Array | None = None) -> jax.Array:
        del training, rng_key
        x = relu(conv2d(params["conv1"], x, padding="SAME"))
        x = relu(conv2d(params["conv2"], x, padding="SAME"))
        x = max_pool_2x2(x)
        x = relu(conv2d(params["conv3"], x, padding="SAME"))
        x = max_pool_2x2(x)
        x = flatten(x)
        x = relu(linear(params["dense1"], x))
        return linear(params["dense2"], x)