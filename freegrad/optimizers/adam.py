"""Adam optimizer implemented with JAX pytrees."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from freegrad.optimizers.common.base import GradFn, GradScalerFn, LossFn, Optimizer
from freegrad.utils.pytrees import tree_add


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AdamState:
    count: jax.Array
    mu: Any
    nu: Any
    grad_scaler_state: Any

    def tree_flatten(self):
        children = (self.count, self.mu, self.nu, self.grad_scaler_state)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


@dataclass(frozen=True)
class Adam(Optimizer):
    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def init(self, params: Any) -> AdamState:
        zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
        return AdamState(count=jnp.array(0, dtype=jnp.int32), mu=zeros, nu=zeros, grad_scaler_state=None)

    def step(
        self,
        *,
        params: Any,
        state: AdamState,
        grad_scaler_state: Any,
        loss_fn: LossFn,
        grad_fn: GradFn,
        grad_scaler_fn: GradScalerFn,
    ) -> tuple[Any, AdamState, dict[str, jax.Array]]:
        y_t = params
        grads_at_y_t = grad_fn(y_t)
        scaled_grads, grad_scaler_state, grad_scaler_metrics = grad_scaler_fn(
            point=y_t,
            grads=grads_at_y_t,
            state=grad_scaler_state,
            loss_fn=loss_fn,
            grad_fn=grad_fn,
        )
        count = state.count + jnp.array(1, dtype=jnp.int32)
        mu = jax.tree_util.tree_map(
            lambda m, g: self.beta1 * m + (1.0 - self.beta1) * g,
            state.mu,
            scaled_grads,
        )
        nu = jax.tree_util.tree_map(
            lambda v, g: self.beta2 * v + (1.0 - self.beta2) * jnp.square(g),
            state.nu,
            scaled_grads,
        )
        beta1_correction = 1.0 - jnp.power(self.beta1, count)
        beta2_correction = 1.0 - jnp.power(self.beta2, count)
        updates = jax.tree_util.tree_map(
            lambda m, v: -self.learning_rate * (m / beta1_correction) / (jnp.sqrt(v / beta2_correction) + self.eps),
            mu,
            nu,
        )
        next_params = tree_add(params, updates)
        metrics = {
            "optimizer/learning_rate": jnp.asarray(self.learning_rate),
            **dict(grad_scaler_metrics),
        }
        next_state = AdamState(count=count, mu=mu, nu=nu, grad_scaler_state=grad_scaler_state)
        return next_params, next_state, metrics