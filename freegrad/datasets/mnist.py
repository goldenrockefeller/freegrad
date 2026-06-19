"""MNIST loading via TensorFlow Datasets."""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np


def load_mnist(data_dir: str | None = None) -> dict[str, np.ndarray]:
    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise ImportError("tensorflow-datasets is required to load MNIST.") from exc

    try:
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
    except Exception:
        train_images, train_labels, test_images, test_labels = _load_mnist_idx_fallback(data_dir)

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


_MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def _load_mnist_idx_fallback(data_dir: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_root = Path(data_dir) if data_dir is not None else Path.home() / ".freegrad" / "mnist"
    cache_root.mkdir(parents=True, exist_ok=True)

    paths = {
        "train_images": _resolve_or_download(cache_root, "train-images-idx3-ubyte.gz", "*train-images-idx3-ubyte*.gz"),
        "train_labels": _resolve_or_download(cache_root, "train-labels-idx1-ubyte.gz", "*train-labels-idx1-ubyte*.gz"),
        "test_images": _resolve_or_download(cache_root, "t10k-images-idx3-ubyte.gz", "*t10k-images-idx3-ubyte*.gz"),
        "test_labels": _resolve_or_download(cache_root, "t10k-labels-idx1-ubyte.gz", "*t10k-labels-idx1-ubyte*.gz"),
    }

    train_images = _read_idx_images(paths["train_images"])
    train_labels = _read_idx_labels(paths["train_labels"])
    test_images = _read_idx_images(paths["test_images"])
    test_labels = _read_idx_labels(paths["test_labels"])
    return train_images, train_labels, test_images, test_labels


def _resolve_or_download(cache_root: Path, canonical_name: str, glob_pattern: str) -> Path:
    direct = cache_root / canonical_name
    if direct.exists():
        return direct

    match = next(cache_root.rglob(glob_pattern), None)
    if match is not None:
        return match

    url_key = canonical_name.replace("-idx3-ubyte.gz", "").replace("-idx1-ubyte.gz", "")
    if url_key == "train-images":
        url = _MNIST_URLS["train_images"]
    elif url_key == "train-labels":
        url = _MNIST_URLS["train_labels"]
    elif url_key == "t10k-images":
        url = _MNIST_URLS["test_images"]
    else:
        url = _MNIST_URLS["test_labels"]

    urllib.request.urlretrieve(url, direct)
    return direct


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        header = handle.read(16)
        magic, count, rows, cols = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(f"Invalid image IDX magic number in {path}: {magic}")
        data = np.frombuffer(handle.read(count * rows * cols), dtype=np.uint8)
    return data.reshape(count, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        header = handle.read(8)
        magic, count = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(f"Invalid label IDX magic number in {path}: {magic}")
        data = np.frombuffer(handle.read(count), dtype=np.uint8)
    return data