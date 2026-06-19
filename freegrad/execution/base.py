"""Execution-driver contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

import jax

from freegrad.models.common.base import ModelVariables


Contribution = TypeVar("Contribution")
FinalValue = TypeVar("FinalValue")
BatchApplyFn = Callable[[dict[str, jax.Array]], Contribution]
BatchReduceFn = Callable[[Contribution, Contribution], Contribution]
BatchFinalizeFn = Callable[[Contribution], FinalValue]


@dataclass(frozen=True)
class ExecutionHints:
    static_axes: dict[str, Any] | None = None
    sharding: dict[str, Any] | None = None
    donate_argnames: tuple[str, ...] = ()
    state_policy: dict[str, Any] | None = None


@dataclass(frozen=True)
class ObjectiveEvaluation:
    value: jax.Array
    grad: Any
    metrics: dict[str, Any]
    model_state: Any


@dataclass(frozen=True)
class TrainChunkSpec:
    num_updates: int
    jittable: bool = True
    compile_policies: bool = True


@runtime_checkable
class ExecutionDriver(Protocol):
    micro_batch_size: int | None

    def value(
        self,
        variables: ModelVariables,
        batch: dict[str, jax.Array],
        *,
        rng_key: jax.Array | None = None,
    ) -> jax.Array:
        ...

    def grad(
        self,
        variables: ModelVariables,
        batch: dict[str, jax.Array],
        *,
        rng_key: jax.Array | None = None,
    ) -> Any:
        ...

    def value_and_grad(
        self,
        variables: ModelVariables,
        batch: dict[str, jax.Array],
        *,
        rng_key: jax.Array | None = None,
    ) -> ObjectiveEvaluation:
        ...

    def hvp(
        self,
        variables: ModelVariables,
        batch: dict[str, jax.Array],
        vector: Any,
        *,
        rng_key: jax.Array | None = None,
    ) -> tuple[jax.Array, Any]:
        ...

    def metrics(
        self,
        variables: ModelVariables,
        batch: dict[str, jax.Array],
        *,
        rng_key: jax.Array | None = None,
    ) -> dict[str, Any]:
        ...

    def apply_batch(
        self,
        batch: dict[str, jax.Array],
        *,
        apply: BatchApplyFn,
        reduce: BatchReduceFn,
        finalize: BatchFinalizeFn | None = None,
    ) -> Contribution | FinalValue:
        ...
