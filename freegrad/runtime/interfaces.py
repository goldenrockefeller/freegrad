"""Callable interface definitions used by runtime specs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import jax

from freegrad.grad_scalers.common.base import GradScaler
from freegrad.models.common.base import Model
from freegrad.optimizers.common.base import Optimizer


ModelApply = Callable[..., jax.Array]


@runtime_checkable
class DatasetLoader(Protocol):
    def __call__(self) -> dict[str, Any]:
        ...


@runtime_checkable
class DataPreparer(Protocol):
    def __call__(self, raw_data: dict[str, Any], validation_size: int = 5000) -> dict[str, Any]:
        ...


@runtime_checkable
class DataPreparerBuilder(Protocol):
    def __call__(self) -> DataPreparer:
        ...


@runtime_checkable
class ModelBuilder(Protocol):
    def __call__(self) -> Model:
        ...


@runtime_checkable
class GradScalerBuilder(Protocol):
    def __call__(self) -> GradScaler:
        ...


@runtime_checkable
class OptimizerBuilder(Protocol):
    def __call__(self) -> Optimizer:
        ...


@runtime_checkable
class LossFn(Protocol):
    def __call__(self, params: Any, inputs: Any, targets: Any) -> jax.Array:
        ...


@runtime_checkable
class LossBuilder(Protocol):
    def __call__(self, model_apply: ModelApply) -> LossFn:
        ...


@runtime_checkable
class MetricsFn(Protocol):
    def __call__(self, params: Any, inputs: Any, targets: Any) -> dict[str, jax.Array]:
        ...


@runtime_checkable
class MetricsBuilder(Protocol):
    def __call__(self, model_apply: ModelApply) -> MetricsFn:
        ...