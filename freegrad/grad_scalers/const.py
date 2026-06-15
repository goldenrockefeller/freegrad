"""Constant grad scaler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from freegrad.grad_scalers.common.base import GradScaler
from freegrad.optimizers.common.base import GradFn, LossFn


@dataclass(frozen=True)
class ConstGradScaler(GradScaler):
    constant: float = 1.0

    def init(self, params: Any) -> None:
        del params
        return None

    def scale(
        self,
        *,
        point: Any,
        grads: Any,
        state: None,
        loss_fn: LossFn,
        grad_fn: GradFn,
    ) -> tuple[Any, None, dict[str, jax.Array]]:
        del point, loss_fn, grad_fn, state
        scaled_grads = jax.tree_util.tree_map(lambda g: self.constant * g, grads)
        metrics = {
            "grad_scaler/constant": jnp.asarray(self.constant),
        }
        return scaled_grads, None, metrics