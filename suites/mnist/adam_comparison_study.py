"""Study comparing MLP and CNN on MNIST with Adam."""

from __future__ import annotations

from freegrad.runtime.study import StudySpec
from suites.mnist.conditions.mlp_adam import make_condition as make_mlp_condition
from suites.mnist.conditions.small_cnn_adam import make_condition as make_small_cnn_condition


def make_study() -> StudySpec:
    return StudySpec(
        name="mnist_adam_model_comparison",
        conditions=[
            make_mlp_condition(),
            make_small_cnn_condition(),
        ],
    )