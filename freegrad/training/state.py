"""Training state definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import jax

from freegrad.learning.base import LearningStackState
from freegrad.models.common.base import ModelVariables

Params = Any
Batch = Mapping[str, jax.Array]
Metrics = Mapping[str, float]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class TrainState:
    step: int
    variables: ModelVariables
    learning_state: LearningStackState
    rng_key: jax.Array

    def tree_flatten(self):
        children = (self.step, self.variables, self.learning_state, self.rng_key)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)

    def replace(self, **kwargs: Any) -> "TrainState":
        values = {
            "step": self.step,
            "variables": self.variables,
            "learning_state": self.learning_state,
            "rng_key": self.rng_key,
        }
        values.update(kwargs)
        return TrainState(**values)

    @property
    def params(self) -> Params:
        return self.variables.params

    @property
    def model_state(self) -> dict[str, Any] | None:
        return dict(self.variables.state)
