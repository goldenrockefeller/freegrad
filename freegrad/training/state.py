"""Training state definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import jax

Params = Any
Batch = Mapping[str, jax.Array]
Metrics = Mapping[str, float]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class TrainState:
    step: int
    params: Params
    optimizer_state: Any
    model_state: dict[str, Any] | None
    rng_key: jax.Array

    def tree_flatten(self):
        children = (self.step, self.params, self.optimizer_state, self.model_state, self.rng_key)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)

    def replace(self, **kwargs: Any) -> "TrainState":
        values = {
            "step": self.step,
            "params": self.params,
            "optimizer_state": self.optimizer_state,
            "model_state": self.model_state,
            "rng_key": self.rng_key,
        }
        values.update(kwargs)
        return TrainState(**values)