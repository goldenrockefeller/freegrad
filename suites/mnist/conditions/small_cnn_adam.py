"""MNIST + ModernSmallCNN + Adam condition."""

from __future__ import annotations

from freegrad.data_prep.mnist import prepare_mnist_arrays
from freegrad.datasets.mnist import load_mnist
from freegrad.learning import adam
from freegrad.losses.classification import CrossEntropyLoss
from freegrad.metrics.classification import ClassificationMetrics
from freegrad.models.small_cnn import ModernSmallCNN
from freegrad.runtime.condition import ConditionSpec


def make_condition() -> ConditionSpec:
    return ConditionSpec(
        name="mnist_small_cnn_adam",
        seeds=[0],
        dataset_loader=lambda: load_mnist(),
        data_preparer_builder=lambda: prepare_mnist_arrays,
        model_builder=lambda: ModernSmallCNN(),
        learning_stack_builder=lambda: adam(learning_rate=1e-3),
        loss_builder=lambda: CrossEntropyLoss(),
        metrics_builder=lambda: ClassificationMetrics(),
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
