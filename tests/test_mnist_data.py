from __future__ import annotations

import numpy as np
import pytest

from freegrad.data_prep.batching import ArrayDataLoader
from freegrad.data_prep.mnist import prepare_mnist_arrays


def test_prepare_mnist_arrays_and_loader_are_deterministic():
    raw = {
        "train_images": np.arange(20 * 28 * 28, dtype=np.float32).reshape(20, 28, 28, 1),
        "train_labels": np.arange(20, dtype=np.int32) % 10,
        "test_images": np.arange(8 * 28 * 28, dtype=np.float32).reshape(8, 28, 28, 1),
        "test_labels": np.arange(8, dtype=np.int32) % 10,
    }

    arrays = prepare_mnist_arrays(raw, validation_size=4)
    loader_a = ArrayDataLoader(
        arrays={"x": arrays["train_images"], "y": arrays["train_labels"]},
        batch_size=4,
        shuffle=True,
        seed=123,
        drop_last=True,
    )
    loader_b = ArrayDataLoader(
        arrays={"x": arrays["train_images"], "y": arrays["train_labels"]},
        batch_size=4,
        shuffle=True,
        seed=123,
        drop_last=True,
    )

    batch_a = next(iter(loader_a))
    batch_b = next(iter(loader_b))

    assert arrays["train_images"].dtype == np.float32
    assert arrays["train_labels"].dtype == np.int32
    assert arrays["train_images"].shape[1:] == (28, 28, 1)
    assert batch_a["x"].shape == (4, 28, 28, 1)
    assert batch_a["y"].shape == (4,)
    assert np.array_equal(batch_a["x"], batch_b["x"])
    assert np.array_equal(batch_a["y"], batch_b["y"])


@pytest.mark.slow
def test_load_mnist_returns_expected_arrays():
    tfds = pytest.importorskip("tensorflow_datasets")
    del tfds
    from freegrad.datasets.mnist import load_mnist

    arrays = load_mnist()

    assert arrays["train_images"].dtype == np.float32
    assert arrays["train_labels"].dtype == np.int32
    assert arrays["train_images"].shape[1:] == (28, 28, 1)
    assert arrays["test_images"].shape[1:] == (28, 28, 1)