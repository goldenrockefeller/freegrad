from __future__ import annotations

import jax
import jax.numpy as jnp

from freegrad.data_prep.batching import batched_loss, batched_metrics
from freegrad.losses.classification import build_cross_entropy_loss
from freegrad.metrics.classification import build_classification_metrics
from freegrad.models.mlp import SimpleMLP
from freegrad.optimizers.adam import Adam
from freegrad.training.eval_step import make_eval_step
from freegrad.training.loops import TrainChunkRunner, ValidationRunner
from freegrad.training.state import TrainState
from freegrad.training.train_step import make_train_update_chunk_step
from freegrad.utils.rng import make_rng


def test_train_and_eval_steps_behave_as_expected():
    model = SimpleMLP()
    params = model.init(jax.random.PRNGKey(0), input_shape=(28, 28, 1), num_classes=10)
    optimizer = Adam(learning_rate=1e-3)
    state = TrainState(
        step=0,
        params=params,
        optimizer_state=optimizer.init(params),
        model_state=None,
        rng_key=make_rng(0),
    )
    batch = {
        "x": jnp.ones((4, 28, 28, 1), dtype=jnp.float32),
        "y": jnp.array([0, 1, 2, 3], dtype=jnp.int32),
    }
    loss_fn = build_cross_entropy_loss(model.apply)
    metrics_fn = build_classification_metrics(model.apply)

    train_step = make_train_update_chunk_step(loss_fn, metrics_fn, optimizer)
    eval_step = make_eval_step(metrics_fn)

    new_state, metrics = train_step(state, {"x": batch["x"][None, ...], "y": batch["y"][None, ...]})
    eval_metrics = eval_step(state.params, batch)

    assert int(new_state.step) == 1
    assert jax.tree_util.tree_leaves(metrics)
    assert "train/loss" in metrics
    assert "grad_norm" in metrics
    assert jnp.isfinite(metrics["train/loss"])
    assert any(
        not jnp.array_equal(before, after)
        for before, after in zip(jax.tree_util.tree_leaves(state.params), jax.tree_util.tree_leaves(new_state.params))
    )
    assert "eval/loss" in eval_metrics
    assert "eval/accuracy" in eval_metrics
    assert all(
        jnp.array_equal(before, after)
        for before, after in zip(jax.tree_util.tree_leaves(state.params), jax.tree_util.tree_leaves(params))
    )


def test_train_update_chunk_step_advances_multiple_batches():
    model = SimpleMLP()
    params = model.init(jax.random.PRNGKey(1), input_shape=(28, 28, 1), num_classes=10)
    optimizer = Adam(learning_rate=1e-3)
    state = TrainState(
        step=0,
        params=params,
        optimizer_state=optimizer.init(params),
        model_state=None,
        rng_key=make_rng(1),
    )
    update_batches = {
        "x": jnp.ones((3, 4, 28, 28, 1), dtype=jnp.float32),
        "y": jnp.array(
            [
                [0, 1, 2, 3],
                [3, 2, 1, 0],
                [4, 5, 6, 7],
            ],
            dtype=jnp.int32,
        ),
    }

    loss_fn = build_cross_entropy_loss(model.apply)
    metrics_fn = build_classification_metrics(model.apply)

    train_chunk_step = make_train_update_chunk_step(loss_fn, metrics_fn, optimizer)

    new_state, metrics = train_chunk_step(state, update_batches)

    assert int(new_state.step) == 3
    assert "train/loss" in metrics
    assert "grad_norm" in metrics
    assert jnp.isfinite(metrics["train/loss"])
    assert any(
        not jnp.array_equal(before, after)
        for before, after in zip(jax.tree_util.tree_leaves(state.params), jax.tree_util.tree_leaves(new_state.params))
    )


def test_train_update_chunk_step_uses_batched_loss_within_one_update():
    model = SimpleMLP()
    params = model.init(jax.random.PRNGKey(2), input_shape=(28, 28, 1), num_classes=10)
    optimizer = Adam(learning_rate=1e-3)
    state = TrainState(
        step=0,
        params=params,
        optimizer_state=optimizer.init(params),
        model_state=None,
        rng_key=make_rng(2),
    )
    batch = {
        "x": jnp.ones((8, 28, 28, 1), dtype=jnp.float32),
        "y": jnp.array([0, 1, 2, 3, 3, 2, 1, 0], dtype=jnp.int32),
    }

    base_loss_fn = build_cross_entropy_loss(model.apply)
    base_metrics_fn = build_classification_metrics(model.apply)
    loss_fn = batched_loss(base_loss_fn, batch_size=4)
    metrics_fn = batched_metrics(base_metrics_fn, batch_size=4)

    train_step = make_train_update_chunk_step(loss_fn, metrics_fn, optimizer)

    new_state, metrics = train_step(state, {"x": batch["x"][None, ...], "y": batch["y"][None, ...]})

    assert int(new_state.step) == 1
    assert "train/loss" in metrics
    assert "train/accuracy" in metrics
    assert jnp.isfinite(metrics["train/loss"])
    assert jnp.isfinite(metrics["train/accuracy"])


def test_train_chunk_runner_owns_macro_batch_iteration():
    model = SimpleMLP()
    params = model.init(jax.random.PRNGKey(3), input_shape=(28, 28, 1), num_classes=10)
    optimizer = Adam(learning_rate=1e-3)
    state = TrainState(
        step=0,
        params=params,
        optimizer_state=optimizer.init(params),
        model_state=None,
        rng_key=make_rng(3),
    )
    train_arrays = {
        "x": jnp.ones((24, 28, 28, 1), dtype=jnp.float32),
        "y": jnp.arange(24, dtype=jnp.int32) % 10,
    }

    base_loss_fn = build_cross_entropy_loss(model.apply)
    base_metrics_fn = build_classification_metrics(model.apply)
    loss_fn = batched_loss(base_loss_fn, batch_size=4)
    metrics_fn = batched_metrics(base_metrics_fn, batch_size=4)
    train_chunk_step = make_train_update_chunk_step(loss_fn, metrics_fn, optimizer)
    runner = TrainChunkRunner(
        arrays=train_arrays,
        macro_batch_size=8,
        train_chunk_size=2,
        max_steps=3,
        seed=3,
        train_update_chunk_step=train_chunk_step,
    )

    first_chunk = runner.run_next_chunk(state)
    second_chunk = runner.run_next_chunk(first_chunk.state)

    assert first_chunk.chunk_steps == 2
    assert int(first_chunk.state.step) == 2
    assert second_chunk.chunk_steps == 1
    assert int(second_chunk.state.step) == 3
    assert not runner.has_next_chunk(second_chunk.state)


def test_validation_runner_aggregates_across_macro_batches():
    model = SimpleMLP()
    params = model.init(jax.random.PRNGKey(4), input_shape=(28, 28, 1), num_classes=10)
    metrics_fn = batched_metrics(build_classification_metrics(model.apply), batch_size=2)
    eval_step = make_eval_step(metrics_fn)
    runner = ValidationRunner(
        arrays={
            "x": jnp.ones((5, 28, 28, 1), dtype=jnp.float32),
            "y": jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32),
        },
        macro_batch_size=3,
        eval_step=eval_step,
    )

    metrics = runner.evaluate(params)

    assert "eval/loss" in metrics
    assert "eval/accuracy" in metrics
    assert jnp.isfinite(metrics["eval/loss"])
    assert jnp.isfinite(metrics["eval/accuracy"])