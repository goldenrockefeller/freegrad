"""Model interface definitions.

Models own parameters, model-owned state, and forward computation. They do not
own execution policy such as micro-batch size, sharding, learning state, or
compiled train chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable

import jax


Params = Any
ModelState = Mapping[str, Any]


class ModelMode(str, Enum):
    TRAIN = "train"
    EVAL = "eval"


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ModelVariables:
    """Parameters plus explicit model-owned state such as batch-norm stats."""

    params: Params
    state: ModelState

    def tree_flatten(self):
        return (self.params, self.state), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        params, state = children
        return cls(params=params, state=state)

    def replace(self, **kwargs: Any) -> "ModelVariables":
        values = {"params": self.params, "state": self.state}
        values.update(kwargs)
        return ModelVariables(**values)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ApplyResult:
    """Result of a model application.

    ``output`` is usually logits or predictions. ``model_state`` is the next
    explicit model state produced by the forward pass.
    """

    output: Any
    model_state: ModelState
    metrics: Mapping[str, Any] | None = None

    def tree_flatten(self):
        return (self.output, self.model_state, self.metrics), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        output, model_state, metrics = children
        return cls(output=output, model_state=model_state, metrics=metrics)


@dataclass(frozen=True)
class ParamGroup:
    path: tuple[str, ...]
    role: str | None = None
    shape: tuple[int, ...] | None = None
    dtype: Any | None = None


@dataclass(frozen=True)
class ParamLayout:
    """Semantic parameter structure for algorithmic policies.

    Execution drivers should consume only narrow execution hints, not this
    semantic grouping object.
    """

    groups: tuple[ParamGroup, ...] = ()


@runtime_checkable
class Model(Protocol):
    def init(self, key: jax.Array, input_shape: tuple[int, ...], num_classes: int) -> ModelVariables:
        ...

    def apply(
        self,
        variables: ModelVariables,
        x: jax.Array,
        mode: ModelMode = ModelMode.EVAL,
        rng_key: jax.Array | None = None,
    ) -> ApplyResult:
        ...

    def param_layout(self, variables: ModelVariables) -> ParamLayout:
        ...
