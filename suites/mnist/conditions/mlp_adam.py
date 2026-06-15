"""MNIST + SimpleMLP + Adam condition."""

from __future__ import annotations

from freegrad.data_prep.mnist import prepare_mnist_arrays
from freegrad.datasets.mnist import load_mnist
from freegrad.losses.classification import build_cross_entropy_loss
from freegrad.metrics.classification import build_classification_metrics
from freegrad.models.mlp import SimpleMLP
from freegrad.optimizers.adam import Adam
from freegrad.runtime.condition import ConditionSpec


def make_condition() -> ConditionSpec:
    return ConditionSpec(
        name="mnist_mlp_adam",
        seeds=[0],
        dataset_loader=lambda: load_mnist(),
        data_preparer_builder=lambda: prepare_mnist_arrays,
        model_builder=lambda: SimpleMLP(),
        optimizer_builder=lambda: Adam(learning_rate=1e-3),
        loss_builder=build_cross_entropy_loss,
        metrics_builder=build_classification_metrics,
        training_config={
            "mini_batch_size": 128,
            "macro_batch_size": 128,
            "eval_mini_batch_size": 128,
            "eval_macro_batch_size": 128,
            "max_steps": 1000,
            "train_chunk_size": 1,
            "eval_every": 100,
            "validation_size": 5000,
            "n_chunks_per_checkpoint": 250,
            "resume": False,
        },
    )