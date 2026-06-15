"""Optimizer interfaces."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol, runtime_checkable

LossFn = Callable[[Any], Any]
GradFn = Callable[[Any], Any]

# TODO, Move Grad scaler to it own sub directory
class GradScalerFn(Protocol):
    def __call__(
        self,
        *,
        point: Any,
        grads: Any,
        state: Any,
        loss_fn: LossFn,
        grad_fn: GradFn,
    ) -> tuple[Any, Any, Mapping[str, Any]]:
        ...


@runtime_checkable
class Optimizer(Protocol):
    def init(self, params: Any) -> Any:
        ...

    def step(
        self,
        *,
        params: Any,
        state: Any,
        grad_scaler_state: Any,
        loss_fn: LossFn,
        grad_fn: GradFn,
        grad_scaler_fn: GradScalerFn,
    ) -> tuple[Any, Any, Mapping[str, Any]]:
        ...