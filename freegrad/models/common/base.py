"""Model interface definitions."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import jax


@runtime_checkable
class Model(Protocol):
    def init(self, key: jax.Array, input_shape: tuple[int, ...], num_classes: int) -> Any:
        ...

    def apply(
        self,
        params: Any,
        x: jax.Array,
        training: bool = False,
        rng_key: jax.Array | None = None,
    ) -> jax.Array:
        ...