"""Reducer-friendly metric contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

import jax
import jax.numpy as jnp

from freegrad.models.common.base import Model, ModelMode, ModelVariables


MetricReport = Mapping[str, Any]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MetricContribution:
    totals: dict[str, jax.Array]
    count: jax.Array

    def tree_flatten(self):
        return (self.totals, self.count), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        totals, count = children
        return cls(totals=totals, count=count)


@runtime_checkable
class ReducibleMetric(Protocol):
    def apply(
        self,
        *,
        model: Model,
        variables: ModelVariables,
        batch: dict[str, jax.Array],
        mode: ModelMode,
        rng_key: jax.Array | None = None,
    ) -> MetricContribution:
        ...

    def reduce(self, left: MetricContribution, right: MetricContribution) -> MetricContribution:
        ...

    def finalize(self, contribution: MetricContribution) -> MetricReport:
        ...


class MeanMetricReduction:
    def reduce(self, left: MetricContribution, right: MetricContribution) -> MetricContribution:
        return MetricContribution(
            totals={key: left.totals[key] + right.totals[key] for key in left.totals},
            count=left.count + right.count,
        )

    def finalize(self, contribution: MetricContribution) -> MetricReport:
        denom = jnp.maximum(contribution.count, jnp.asarray(1, contribution.count.dtype))
        return {key: total / denom for key, total in contribution.totals.items()}
