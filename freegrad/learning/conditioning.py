"""Conditioning policies for raw gradients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from freegrad.execution.base import ExecutionDriver, ObjectiveEvaluation
from freegrad.learning.base import ConditioningResult
from freegrad.models.common.base import ModelVariables
from freegrad.utils.pytrees import tree_zeros_like


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class EmptyContribution:
    def tree_flatten(self):
        return (), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data, children
        return cls()


@dataclass(frozen=True)
class IdentityConditioner:
    name: str = "identity_conditioner"
    jittable: bool = True

    def init(self, params: Any) -> None:
        del params
        return None

    def apply(
        self,
        *,
        driver: ExecutionDriver,
        variables: ModelVariables,
        batch: dict[str, jax.Array],
        raw_gradient: Any,
        objective: ObjectiveEvaluation,
    ) -> EmptyContribution:
        del driver, variables, batch, raw_gradient, objective
        return EmptyContribution()

    def reduce(self, left: EmptyContribution, right: EmptyContribution) -> EmptyContribution:
        del left, right
        return EmptyContribution()

    def finalize(
        self,
        *,
        state: None,
        contribution: EmptyContribution,
        raw_gradient: Any,
        objective: ObjectiveEvaluation,
    ) -> ConditioningResult:
        del state, contribution, objective
        return ConditioningResult(
            gradient=raw_gradient,
            state=None,
            metrics={"conditioning/type": jnp.asarray(0, dtype=jnp.int32)},
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AdamConditionerState:
    count: jax.Array
    mu: Any
    nu: Any

    def tree_flatten(self):
        return (self.count, self.mu, self.nu), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        count, mu, nu = children
        return cls(count=count, mu=mu, nu=nu)


@dataclass(frozen=True)
class AdamConditioner:
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    name: str = "adam_conditioner"
    jittable: bool = True

    def init(self, params: Any) -> AdamConditionerState:
        zeros = tree_zeros_like(params)
        return AdamConditionerState(count=jnp.asarray(0, dtype=jnp.int32), mu=zeros, nu=zeros)

    def apply(
        self,
        *,
        driver: ExecutionDriver,
        variables: ModelVariables,
        batch: dict[str, jax.Array],
        raw_gradient: Any,
        objective: ObjectiveEvaluation,
    ) -> EmptyContribution:
        del driver, variables, batch, raw_gradient, objective
        return EmptyContribution()

    def reduce(self, left: EmptyContribution, right: EmptyContribution) -> EmptyContribution:
        del left, right
        return EmptyContribution()

    def finalize(
        self,
        *,
        state: AdamConditionerState,
        contribution: EmptyContribution,
        raw_gradient: Any,
        objective: ObjectiveEvaluation,
    ) -> ConditioningResult:
        del contribution, objective
        count = state.count + jnp.asarray(1, dtype=jnp.int32)
        mu = jax.tree_util.tree_map(
            lambda m, g: self.beta1 * m + (1.0 - self.beta1) * g,
            state.mu,
            raw_gradient,
        )
        nu = jax.tree_util.tree_map(
            lambda v, g: self.beta2 * v + (1.0 - self.beta2) * jnp.square(g),
            state.nu,
            raw_gradient,
        )
        beta1_correction = 1.0 - jnp.power(self.beta1, count)
        beta2_correction = 1.0 - jnp.power(self.beta2, count)
        conditioned = jax.tree_util.tree_map(
            lambda m, v: (m / beta1_correction) / (jnp.sqrt(v / beta2_correction) + self.eps),
            mu,
            nu,
        )
        return ConditioningResult(
            gradient=conditioned,
            state=AdamConditionerState(count=count, mu=mu, nu=nu),
            metrics={"conditioning/adam_count": count},
        )
