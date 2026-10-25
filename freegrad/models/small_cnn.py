"""Keras-reference-sized convolutional network for MNIST."""

from __future__ import annotations

import jax

from freegrad.models.common.base import ApplyResult, Model, ModelMode, ModelVariables, ParamLayout
from freegrad.models.common.init import glorot_uniform, kaiming_uniform, zeros
from freegrad.models.common.layers import conv2d, flatten, linear, max_pool_2x2, relu
from freegrad.models.common.layout import infer_param_layout


class ModernSmallCNN(Model):
    def init(self, key: jax.Array, input_shape: tuple[int, ...], num_classes: int) -> ModelVariables:
        height, width, channels = input_shape
        # Two 3x3 VALID convolutions and two 2x2 VALID pools map MNIST's
        # 28x28 input to 5x5x64, matching the Keras MNIST convnet recipe.
        feature_height = ((height - 2) // 2 - 2) // 2
        feature_width = ((width - 2) // 2 - 2) // 2
        flat_dim = feature_height * feature_width * 64
        keys = jax.random.split(key, 3)
        params = {
            "conv1": {
                "w": kaiming_uniform(keys[0], (3, 3, channels, 32)),
                "b": zeros((32,)),
            },
            "conv2": {
                "w": kaiming_uniform(keys[1], (3, 3, 32, 64)),
                "b": zeros((64,)),
            },
            "classifier": {
                "w": glorot_uniform(keys[2], (flat_dim, num_classes)),
                "b": zeros((num_classes,)),
            },
        }
        return ModelVariables(params=params, state={})

    def apply(
        self,
        variables: ModelVariables,
        x: jax.Array,
        mode: ModelMode = ModelMode.EVAL,
        rng_key: jax.Array | None = None,
    ) -> ApplyResult:
        del mode, rng_key
        params = variables.params
        x = relu(conv2d(params["conv1"], x, padding="VALID"))
        x = max_pool_2x2(x, padding="VALID")
        x = relu(conv2d(params["conv2"], x, padding="VALID"))
        x = max_pool_2x2(x, padding="VALID")
        x = flatten(x)
        return ApplyResult(output=linear(params["classifier"], x), model_state=variables.state)

    def param_layout(self, variables: ModelVariables) -> ParamLayout:
        return infer_param_layout(variables.params)
