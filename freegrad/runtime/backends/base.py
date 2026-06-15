"""Backend interface definitions."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from freegrad.runtime.condition import ConditionSpec
from freegrad.runtime.run import RunResult, RunSpec
from freegrad.runtime.study import StudySpec

# TODO why does this exist?
@runtime_checkable
class StudyConfirmationHandler(Protocol):
    def confirm_study_execution(self, message: str) -> bool:
        ...


@runtime_checkable
class Backend(Protocol):
    def run(self, run_spec: RunSpec) -> RunResult:
        ...

    def run_condition(self, condition: ConditionSpec) -> list[RunResult]:
        ...

    def run_study(self, study: StudySpec) -> list[RunResult]:
        ...