"""Dataset split helpers."""

from __future__ import annotations

from typing import Mapping


def split_train_validation(arrays: Mapping[str, object], validation_size: int) -> dict[str, object]:
    train_images = arrays["train_images"]
    train_labels = arrays["train_labels"]
    if validation_size <= 0:
        return {
            "train_images": train_images,
            "train_labels": train_labels,
            "validation_images": train_images[:0],
            "validation_labels": train_labels[:0],
            "test_images": arrays["test_images"],
            "test_labels": arrays["test_labels"],
        }
    return {
        "train_images": train_images[:-validation_size],
        "train_labels": train_labels[:-validation_size],
        "validation_images": train_images[-validation_size:],
        "validation_labels": train_labels[-validation_size:],
        "test_images": arrays["test_images"],
        "test_labels": arrays["test_labels"],
    }