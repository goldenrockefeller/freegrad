"""Pure JAX layers used by simple models."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def linear(params: dict[str, jax.Array], x: jax.Array) -> jax.Array:
    return x @ params["w"] + params["b"]


def relu(x: jax.Array) -> jax.Array:
    return jax.nn.relu(x)


def flatten(x: jax.Array) -> jax.Array:
    return x.reshape((x.shape[0], -1))


def conv2d(
    params: dict[str, jax.Array],
    x: jax.Array,
    stride: int | tuple[int, int] = 1,
    padding: str = "SAME",
) -> jax.Array:
    strides = (stride, stride) if isinstance(stride, int) else stride
    y = jax.lax.conv_general_dilated(
        lhs=x,
        rhs=params["w"],
        window_strides=strides,
        padding=padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return y + params["b"]


def max_pool_2x2(x: jax.Array) -> jax.Array:
    return jax.lax.reduce_window(
        operand=x,
        init_value=-jnp.inf,
        computation=jax.lax.max,
        window_dimensions=(1, 2, 2, 1),
        window_strides=(1, 2, 2, 1),
        padding="SAME",
    )