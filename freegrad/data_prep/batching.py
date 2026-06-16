"""Deterministic array-backed batching."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class ArrayDataLoader:
    arrays: dict[str, np.ndarray]
    batch_size: int
    shuffle: bool
    seed: int
    drop_last: bool = True

    def __iter__(self):
        num_examples = int(self.arrays["x"].shape[0])
        indices = np.arange(num_examples)
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)

        if self.drop_last:
            stop = num_examples - (num_examples % self.batch_size)
        else:
            stop = num_examples

        for start in range(0, stop, self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            if batch_indices.shape[0] < self.batch_size and self.drop_last:
                continue
            yield {
                "x": self.arrays["x"][batch_indices],
                "y": self.arrays["y"][batch_indices],
            }


def batched_output(function: Callable[[Any, Any, Any], Any], batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    def wrapped(params: Any, inputs: Any, targets: Any):
        total_examples = _leading_axis_size(targets)
        if total_examples == 0:
            raise ValueError("batched_output requires at least one example.")
        # Small inputs can be processed in one call, so skip the batching path.
        if total_examples <= batch_size:
            return function(params, inputs, targets)

        # Split the leading axis into full batches and a possible remainder.
        full_batch_count = total_examples // batch_size
        remainder = total_examples % batch_size
        totals = None

        if full_batch_count > 0:
            # Reshape each leaf to [num_batches, batch_size, ...] for scanning.
            full_inputs = _reshape_full_batches(inputs, full_batch_count, batch_size)
            full_targets = _reshape_full_batches(targets, full_batch_count, batch_size)
            # Use one batch to infer the output tree shape and initialize zeros.
            sample_value = function(
                params,
                jax.tree_util.tree_map(lambda leaf: leaf[0], full_inputs),
                jax.tree_util.tree_map(lambda leaf: leaf[0], full_targets),
            )
            zero_totals = jax.tree_util.tree_map(jnp.zeros_like, sample_value)

            def scan_step(carry, batch):
                batch_inputs, batch_targets = batch
                values = function(params, batch_inputs, batch_targets)
                # Weight each batch result by its example count before accumulating.
                weighted_values = jax.tree_util.tree_map(
                    lambda value: value * jnp.asarray(batch_size, dtype=value.dtype),
                    values,
                )
                next_totals = jax.tree_util.tree_map(lambda left, right: left + right, carry, weighted_values)
                return next_totals, None

            totals, _ = jax.lax.scan(scan_step, zero_totals, (full_inputs, full_targets))

        if remainder > 0:
            # Handle the tail separately so it contributes with its true size.
            tail_inputs = _slice_tree(inputs, full_batch_count * batch_size, total_examples)
            tail_targets = _slice_tree(targets, full_batch_count * batch_size, total_examples)
            tail_values = function(params, tail_inputs, tail_targets)
            weighted_tail = jax.tree_util.tree_map(
                lambda value: value * jnp.asarray(remainder, dtype=value.dtype),
                tail_values,
            )
            totals = weighted_tail if totals is None else jax.tree_util.tree_map(
                lambda left, right: left + right,
                totals,
                weighted_tail,
            )

        return jax.tree_util.tree_map(
            lambda value: value / jnp.asarray(total_examples, dtype=value.dtype),
            totals,
        )

    return wrapped


def batched_loss(loss_fn: Callable[[Any, Any, Any], jax.Array], batch_size: int):
    return batched_output(loss_fn, batch_size)


def batched_metrics(metrics_fn: Callable[[Any, Any, Any], dict[str, jax.Array]], batch_size: int):
    return batched_output(metrics_fn, batch_size)


def _leading_axis_size(tree: Any) -> int:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        raise ValueError("Expected at least one array leaf.")
    return int(jnp.asarray(leaves[0]).shape[0])


def _reshape_full_batches(tree: Any, full_batch_count: int, batch_size: int):
    # Truncate to whole batches, then add the batch dimension back in.
    return jax.tree_util.tree_map(
        lambda leaf: jnp.asarray(leaf[: full_batch_count * batch_size]).reshape(
            (full_batch_count, batch_size, *jnp.asarray(leaf).shape[1:])
        ),
        tree,
    )


def _slice_tree(tree: Any, start: int, stop: int):
    # Slice every leaf consistently along the leading axis.
    return jax.tree_util.tree_map(lambda leaf: jnp.asarray(leaf[start:stop]), tree)