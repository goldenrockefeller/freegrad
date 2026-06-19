"""Simple multilayer perceptron for MNIST."""

from __future__ import annotations

import jax

from freegrad.models.common.base import ApplyResult, Model, ModelMode, ModelVariables, ParamLayout
from freegrad.models.common.init import glorot_uniform, zeros
from freegrad.models.common.layers import flatten, linear, relu
from freegrad.models.common.layout import infer_param_layout


class SimpleMLP(Model):
    def init(self, key: jax.Array, input_shape: tuple[int, ...], num_classes: int) -> ModelVariables:
        flat_dim = input_shape[0] * input_shape[1] * input_shape[2]
        keys = jax.random.split(key, 3)
        params = {
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
        x = flatten(x)
        x = relu(linear(params["dense1"], x))
        x = relu(linear(params["dense2"], x))
        return ApplyResult(output=linear(params["dense3"], x), model_state=variables.state)

    def param_layout(self, variables: ModelVariables) -> ParamLayout:
        return infer_param_layout(variables.params)
