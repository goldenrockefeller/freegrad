"""Generic learning-stack orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

import jax

from freegrad.execution.base import ExecutionDriver, ObjectiveEvaluation
from freegrad.models.common.base import ModelVariables
from freegrad.utils.pytrees import tree_add, tree_scalar_mul


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SearchMemory:
    """General algorithm memory: moments, velocities, histories, masks, etc."""

    data: Any = None

    def tree_flatten(self):
        return (self.data,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        (data,) = children
        return cls(data=data)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LearningStackState:
    search_memory: SearchMemory
    conditioner_state: Any
    step_size_state: Any

    def tree_flatten(self):
        return (self.search_memory, self.conditioner_state, self.step_size_state), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        search_memory, conditioner_state, step_size_state = children
        return cls(search_memory=search_memory, conditioner_state=conditioner_state, step_size_state=step_size_state)

    def replace(self, **kwargs: Any) -> "LearningStackState":
        values = {
            "search_memory": self.search_memory,
            "conditioner_state": self.conditioner_state,
            "step_size_state": self.step_size_state,
        }
        values.update(kwargs)
        return LearningStackState(**values)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class UpdateSignal:
    direction: Any
    step_size: Any
    updates: Any

    def tree_flatten(self):
        return (self.direction, self.step_size, self.updates), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        direction, step_size, updates = children
        return cls(direction=direction, step_size=step_size, updates=updates)


@dataclass(frozen=True)
class ProbeResult:
    variables: ModelVariables
    metrics: Mapping[str, Any]


@dataclass(frozen=True)
class ConditioningResult:
    gradient: Any
    state: Any
    metrics: Mapping[str, Any]
    info: Any = None


@dataclass(frozen=True)
class StepSizeResult:
    step_size: Any
    state: Any
    metrics: Mapping[str, Any]


@dataclass(frozen=True)
class RepairResult:
    variables: ModelVariables
    search_memory: SearchMemory
    metrics: Mapping[str, Any]


@runtime_checkable
class ProbePointPolicy(Protocol):
    name: str
    jittable: bool

    def select(self, *, variables: ModelVariables, search_memory: SearchMemory) -> ProbeResult:
        ...


@runtime_checkable
class ConditioningPolicy(Protocol):
    name: str
    jittable: bool

    def init(self, params: Any) -> Any:
        ...

    def apply(
        self,
        *,
        driver: ExecutionDriver,
        variables: ModelVariables,
        batch: dict[str, jax.Array],
        raw_gradient: Any,
        objective: ObjectiveEvaluation,
    ) -> Any:
        ...

    def reduce(self, left: Any, right: Any) -> Any:
        ...

    def finalize(
        self,
        *,
        state: Any,
        contribution: Any,
        raw_gradient: Any,
        objective: ObjectiveEvaluation,
    ) -> ConditioningResult:
        ...


@runtime_checkable
class StepSizePolicy(Protocol):
    name: str
    jittable: bool

    def init(self) -> Any:
        ...

    def choose(
        self,
        *,
        driver: ExecutionDriver,
        variables: ModelVariables,
        batch: dict[str, jax.Array],
        raw_gradient: Any,
        conditioned_gradient: Any,
        objective: ObjectiveEvaluation,
        state: Any,
    ) -> StepSizeResult:
        ...


@runtime_checkable
class RepairPolicy(Protocol):
    name: str
    jittable: bool

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
        ...


@runtime_checkable
class UpdatePolicy(Protocol):
    name: str
    jittable: bool

    def init(self, params: Any) -> SearchMemory:
        ...

    def build_update_signal(
        self,
        *,
        search_memory: SearchMemory,
        conditioned_gradient: Any,
        step_size: Any,
    ) -> UpdateSignal:
        ...

    def commit(
        self,
        *,
        variables: ModelVariables,
        search_memory: SearchMemory,
        update_signal: UpdateSignal,
    ) -> tuple[ModelVariables, SearchMemory, Mapping[str, Any]]:
        ...


@dataclass(frozen=True)
class LearningStack:
    probe_point: ProbePointPolicy
    conditioner: ConditioningPolicy
    step_size: StepSizePolicy
    repair: RepairPolicy
    update: UpdatePolicy
    name: str = "learning_stack"

    @property
    def jittable(self) -> bool:
        return all(
            (
                self.probe_point.jittable,
                self.conditioner.jittable,
                self.step_size.jittable,
                self.repair.jittable,
                self.update.jittable,
            )
        )

    def init(self, variables: ModelVariables) -> LearningStackState:
        return LearningStackState(
            search_memory=self.update.init(variables.params),
            conditioner_state=self.conditioner.init(variables.params),
            step_size_state=self.step_size.init(),
        )

    def step(
        self,
        *,
        variables: ModelVariables,
        state: LearningStackState,
        driver: ExecutionDriver,
        batch: dict[str, jax.Array],
        rng_key: jax.Array | None = None,
    ) -> tuple[ModelVariables, LearningStackState, dict[str, Any]]:
        probe = self.probe_point.select(variables=variables, search_memory=state.search_memory)
        objective = driver.value_and_grad(probe.variables, batch, rng_key=rng_key)
        contribution = driver.apply_batch(
            batch,
            apply=lambda micro_batch: self.conditioner.apply(
                driver=driver,
                variables=probe.variables,
                batch=micro_batch,
                raw_gradient=objective.grad,
                objective=objective,
            ),
            reduce=self.conditioner.reduce,
        )
        conditioned = self.conditioner.finalize(
            state=state.conditioner_state,
            contribution=contribution,
            raw_gradient=objective.grad,
            objective=objective,
        )
        step_size = self.step_size.choose(
            driver=driver,
            variables=probe.variables,
            batch=batch,
            raw_gradient=objective.grad,
            conditioned_gradient=conditioned.gradient,
            objective=objective,
            state=state.step_size_state,
        )
        update_signal = self.update.build_update_signal(
            search_memory=state.search_memory,
            conditioned_gradient=conditioned.gradient,
            step_size=step_size.step_size,
        )
        current_variables = variables.replace(state=objective.model_state)
        repaired = self.repair.repair(
            variables=current_variables,
            search_memory=state.search_memory,
            raw_gradient=objective.grad,
            conditioned_gradient=conditioned.gradient,
            update_signal=update_signal,
            objective=objective,
            conditioning_info=conditioned.info,
        )
        next_variables, next_search_memory, update_metrics = self.update.commit(
            variables=repaired.variables,
            search_memory=repaired.search_memory,
            update_signal=update_signal,
        )
        next_state = state.replace(
            search_memory=next_search_memory,
            conditioner_state=conditioned.state,
            step_size_state=step_size.state,
        )
        metrics = {
            **{f"train/{key}": value for key, value in objective.metrics.items()},
            "train/loss": objective.value,
            **dict(probe.metrics),
            **dict(conditioned.metrics),
            **dict(step_size.metrics),
            **dict(repaired.metrics),
            **dict(update_metrics),
        }
        return next_variables, next_state, metrics


def negative_step(step_size: Any, direction: Any) -> Any:
    return tree_scalar_mul(-step_size, direction)


def apply_updates(variables: ModelVariables, updates: Any) -> ModelVariables:
    return variables.replace(params=tree_add(variables.params, updates))
