"""Callable interface definitions used by runtime specs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import jax

from freegrad.learning.base import LearningStack
from freegrad.models.common.base import Model


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
@runtime_checkable
class LearningStackBuilder(Protocol):
    def __call__(self) -> LearningStack:
        ...


@runtime_checkable
class LossFn(Protocol):
    def __call__(self, params: Any, inputs: Any, targets: Any) -> jax.Array:
        ...


@runtime_checkable
class LossBuilder(Protocol):
    def __call__(self, model_apply: ModelApply | None = None) -> Any:
        ...


@runtime_checkable
class MetricsFn(Protocol):
    def __call__(self, params: Any, inputs: Any, targets: Any) -> dict[str, jax.Array]:
        ...


@runtime_checkable
class MetricsBuilder(Protocol):
    def __call__(self, model_apply: ModelApply | None = None) -> Any:
        ...
