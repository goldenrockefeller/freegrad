"""MNIST preprocessing."""

from __future__ import annotations

import numpy as np

from freegrad.data_prep.common.normalize import normalize_with_stats
from freegrad.data_prep.common.splits import split_train_validation


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def normalize_mnist(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images, dtype=np.float32) / 255.0
    return normalize_with_stats(images, MNIST_MEAN, MNIST_STD).astype(np.float32)


def prepare_mnist_arrays(raw_data: dict[str, np.ndarray], validation_size: int = 5000) -> dict[str, np.ndarray]:
    arrays = split_train_validation(raw_data, validation_size=validation_size)
    return {
        "train_images": normalize_mnist(arrays["train_images"]),
        "train_labels": np.asarray(arrays["train_labels"], dtype=np.int32),
        "validation_images": normalize_mnist(arrays["validation_images"]),
        "validation_labels": np.asarray(arrays["validation_labels"], dtype=np.int32),
        "test_images": normalize_mnist(arrays["test_images"]),
        "test_labels": np.asarray(arrays["test_labels"], dtype=np.int32),
    }