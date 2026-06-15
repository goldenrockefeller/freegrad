"""Grad-scaler interfaces."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from freegrad.optimizers.common.base import GradFn, LossFn


@runtime_checkable
class GradScaler(Protocol):
    def init(self, params: Any) -> Any:
        ...

    def scale(
        self,
        *,
        point: Any,
        grads: Any,
        state: Any,
        loss_fn: LossFn,
        grad_fn: GradFn,
    ) -> tuple[Any, Any, Mapping[str, Any]]:
        ...