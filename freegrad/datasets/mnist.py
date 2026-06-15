"""MNIST loading via TensorFlow Datasets."""

from __future__ import annotations

import numpy as np


def load_mnist(data_dir: str | None = None) -> dict[str, np.ndarray]:
    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise ImportError("tensorflow-datasets is required to load MNIST.") from exc

    train_ds = tfds.load(
        "mnist",
        split="train",
        batch_size=-1,
        as_supervised=True,
        data_dir=data_dir,
    )
    test_ds = tfds.load(
        "mnist",
        split="test",
        batch_size=-1,
        as_supervised=True,
        data_dir=data_dir,
    )
    train_images, train_labels = tfds.as_numpy(train_ds)
    test_images, test_labels = tfds.as_numpy(test_ds)
    return {
        "train_images": _prepare_images(train_images),
        "train_labels": np.asarray(train_labels, dtype=np.int32),
        "test_images": _prepare_images(test_images),
        "test_labels": np.asarray(test_labels, dtype=np.int32),
    }


def _prepare_images(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images, dtype=np.float32)
    if images.ndim == 3:
        images = images[..., None]
    return images