"""Reducer-friendly loss contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp

from freegrad.models.common.base import Model, ModelMode, ModelVariables


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LossContribution:
    total: jax.Array
    count: jax.Array

    def tree_flatten(self):
        return (self.total, self.count), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        total, count = children
        return cls(total=total, count=count)


@runtime_checkable
class ReducibleLoss(Protocol):
    def apply(
        self,
        *,
        model: Model,
        variables: ModelVariables,
        batch: dict[str, jax.Array],
        mode: ModelMode,
        rng_key: jax.Array | None = None,
    ) -> LossContribution:
        ...

    def reduce(self, left: LossContribution, right: LossContribution) -> LossContribution:
        ...

    def finalize(self, contribution: LossContribution) -> jax.Array:
        ...


class MeanLossReduction:
    def reduce(self, left: LossContribution, right: LossContribution) -> LossContribution:
        return LossContribution(total=left.total + right.total, count=left.count + right.count)

    def finalize(self, contribution: LossContribution) -> jax.Array:
        return contribution.total / jnp.maximum(contribution.count, jnp.asarray(1, contribution.count.dtype))
