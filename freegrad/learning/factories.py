"""Concrete learning-stack factories."""

from __future__ import annotations

from freegrad.learning.base import LearningStack
from freegrad.learning.conditioning import AdamConditioner, IdentityConditioner
from freegrad.learning.policies import (
    ConstantStepSize,
    CurrentParamsProbe,
    GradientDescentUpdate,
    LookaheadProbe,
    NesterovUpdate,
    NoOpRepair,
)


def sgd(learning_rate: float) -> LearningStack:
    return LearningStack(
        name="sgd",
        probe_point=CurrentParamsProbe(),
        conditioner=IdentityConditioner(),
        step_size=ConstantStepSize(learning_rate),
        repair=NoOpRepair(),
        update=GradientDescentUpdate(),
    )


def adam(learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> LearningStack:
    return LearningStack(
        name="adam",
        probe_point=CurrentParamsProbe(),
        conditioner=AdamConditioner(beta1=beta1, beta2=beta2, eps=eps),
        step_size=ConstantStepSize(learning_rate),
        repair=NoOpRepair(),
        update=GradientDescentUpdate(),
    )


def nesterov(learning_rate: float, momentum: float = 0.9) -> LearningStack:
    return LearningStack(
        name="nesterov",
        probe_point=LookaheadProbe(coefficient=momentum),
        conditioner=IdentityConditioner(),
        step_size=ConstantStepSize(learning_rate),
        repair=NoOpRepair(),
        update=NesterovUpdate(momentum=momentum),
    )
