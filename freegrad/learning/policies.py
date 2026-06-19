"""Probe, step-size, repair, and update policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from freegrad.execution.base import ExecutionDriver, ObjectiveEvaluation
from freegrad.learning.base import (
    ProbeResult,
    RepairResult,
    SearchMemory,
    StepSizeResult,
    UpdateSignal,
    apply_updates,
    negative_step,
)
from freegrad.models.common.base import ModelVariables
from freegrad.utils.pytrees import tree_add, tree_scalar_mul, tree_zeros_like


@dataclass(frozen=True)
class CurrentParamsProbe:
    name: str = "current_params_probe"
    jittable: bool = True

    def select(self, *, variables: ModelVariables, search_memory: SearchMemory) -> ProbeResult:
        del search_memory
        return ProbeResult(variables=variables, metrics={})


@dataclass(frozen=True)
class LookaheadProbe:
    coefficient: float
    name: str = "lookahead_probe"
    jittable: bool = True

    def select(self, *, variables: ModelVariables, search_memory: SearchMemory) -> ProbeResult:
        velocity = search_memory.data["velocity"]
        return ProbeResult(
            variables=variables.replace(params=tree_add(variables.params, tree_scalar_mul(self.coefficient, velocity))),
            metrics={"probe/lookahead": jnp.asarray(self.coefficient)},
        )


@dataclass(frozen=True)
class ConstantStepSize:
    value: float
    name: str = "constant_step_size"
    jittable: bool = True

    def init(self) -> None:
        return None

    def choose(
        self,
        *,
        driver: ExecutionDriver,
        variables: ModelVariables,
        batch: dict[str, jax.Array],
        raw_gradient: Any,
        conditioned_gradient: Any,
        objective: ObjectiveEvaluation,
        state: None,
    ) -> StepSizeResult:
        del driver, variables, batch, raw_gradient, conditioned_gradient, objective, state
        step = jnp.asarray(self.value, dtype=jnp.float32)
        return StepSizeResult(step_size=step, state=None, metrics={"step_size/value": step})


@dataclass(frozen=True)
class NoOpRepair:
    name: str = "no_op_repair"
    jittable: bool = True

    def repair(
        self,
        *,
        variables: ModelVariables,
        search_memory: SearchMemory,
        raw_gradient: Any,
        conditioned_gradient: Any,
        update_signal: UpdateSignal,
        objective: ObjectiveEvaluation,
        conditioning_info: Any,
    ) -> RepairResult:
        del raw_gradient, conditioned_gradient, update_signal, objective, conditioning_info
        return RepairResult(variables=variables, search_memory=search_memory, metrics={})


@dataclass(frozen=True)
class GradientDescentUpdate:
    name: str = "gradient_descent_update"
    jittable: bool = True

    def init(self, params: Any) -> SearchMemory:
        del params
        return SearchMemory(data=None)

    def build_update_signal(
        self,
        *,
        search_memory: SearchMemory,
        conditioned_gradient: Any,
        step_size: Any,
    ) -> UpdateSignal:
        del search_memory
        updates = negative_step(step_size, conditioned_gradient)
        return UpdateSignal(direction=conditioned_gradient, step_size=step_size, updates=updates)

    def commit(
        self,
        *,
        variables: ModelVariables,
        search_memory: SearchMemory,
        update_signal: UpdateSignal,
    ) -> tuple[ModelVariables, SearchMemory, dict[str, Any]]:
        return apply_updates(variables, update_signal.updates), search_memory, {}


@dataclass(frozen=True)
class NesterovUpdate:
    momentum: float
    name: str = "nesterov_update"
    jittable: bool = True

    def init(self, params: Any) -> SearchMemory:
        return SearchMemory(data={"velocity": tree_zeros_like(params)})

    def build_update_signal(
        self,
        *,
        search_memory: SearchMemory,
        conditioned_gradient: Any,
        step_size: Any,
    ) -> UpdateSignal:
        velocity = search_memory.data["velocity"]
        next_velocity = jax.tree_util.tree_map(
            lambda v, g: self.momentum * v - step_size * g,
            velocity,
            conditioned_gradient,
        )
        return UpdateSignal(direction=conditioned_gradient, step_size=step_size, updates=next_velocity)

    def commit(
        self,
        *,
        variables: ModelVariables,
        search_memory: SearchMemory,
        update_signal: UpdateSignal,
    ) -> tuple[ModelVariables, SearchMemory, dict[str, Any]]:
        del search_memory
        return (
            apply_updates(variables, update_signal.updates),
            SearchMemory(data={"velocity": update_signal.updates}),
            {"update/momentum": jnp.asarray(self.momentum)},
        )
