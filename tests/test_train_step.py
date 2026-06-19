from __future__ import annotations

import jax
import jax.numpy as jnp

from freegrad.execution import SingleCPUExecutionDriver
from freegrad.learning import adam, sgd
from freegrad.losses.classification import CrossEntropyLoss
from freegrad.metrics.classification import ClassificationMetrics
from freegrad.models.mlp import SimpleMLP
from freegrad.training.eval_step import make_eval_step
from freegrad.training.loops import TrainChunkRunner, ValidationRunner
from freegrad.training.state import TrainState
from freegrad.training.train_step import make_train_update_chunk_step
from freegrad.utils.rng import make_rng


def _make_state_and_driver(*, seed: int, micro_batch_size: int = 4, stack=None):
    model = SimpleMLP()
    variables = model.init(jax.random.PRNGKey(seed), input_shape=(28, 28, 1), num_classes=10)
    learning_stack = stack or adam(learning_rate=1e-3)
    driver = SingleCPUExecutionDriver(
        model=model,
        loss=CrossEntropyLoss(),
        metric=ClassificationMetrics(),
        micro_batch_size=micro_batch_size,
    )
    state = TrainState(
        step=0,
        variables=variables,
        learning_state=learning_stack.init(variables),
        rng_key=make_rng(seed),
    )
    return state, driver, learning_stack


def test_train_and_eval_steps_behave_as_expected():
    state, driver, learning_stack = _make_state_and_driver(seed=0)
    batch = {
        "x": jnp.ones((4, 28, 28, 1), dtype=jnp.float32),
        "y": jnp.array([0, 1, 2, 3], dtype=jnp.int32),
    }

    train_step = make_train_update_chunk_step(driver, learning_stack)
    eval_step = make_eval_step(driver)

    new_state, metrics = train_step(state, {"x": batch["x"][None, ...], "y": batch["y"][None, ...]})
    eval_metrics = eval_step(state.variables, batch)

    assert int(new_state.step) == 1
    assert "train/loss" in metrics
    assert "grad_norm" in metrics
    assert jnp.isfinite(metrics["train/loss"])
    assert any(
        not jnp.array_equal(before, after)
        for before, after in zip(jax.tree_util.tree_leaves(state.params), jax.tree_util.tree_leaves(new_state.params))
    )
    assert "eval/loss" in eval_metrics
    assert "eval/accuracy" in eval_metrics
    assert state.model_state == {}


def test_train_update_chunk_step_advances_multiple_batches():
    state, driver, learning_stack = _make_state_and_driver(seed=1)
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

    train_chunk_step = make_train_update_chunk_step(driver, learning_stack)

    new_state, metrics = train_chunk_step(state, update_batches)

    assert int(new_state.step) == 3
    assert "train/loss" in metrics
    assert "grad_norm" in metrics
    assert jnp.isfinite(metrics["train/loss"])


def test_execution_micro_batches_effective_batch_without_history():
    state, driver, _ = _make_state_and_driver(seed=2, micro_batch_size=2, stack=sgd(learning_rate=1e-2))
    full_driver = SingleCPUExecutionDriver(
        model=driver.model,
        loss=driver.loss,
        metric=driver.metric,
        micro_batch_size=None,
    )
    batch = {
        "x": jnp.ones((8, 28, 28, 1), dtype=jnp.float32),
        "y": jnp.array([0, 1, 2, 3, 3, 2, 1, 0], dtype=jnp.int32),
    }

    micro_value = driver.value(state.variables, batch)
    full_value = full_driver.value(state.variables, batch)
    contribution = driver.apply_batch(
        batch,
        apply=lambda micro_batch: jnp.asarray(micro_batch["y"].shape[0]),
        reduce=lambda left, right: left + right,
    )

    assert jnp.allclose(micro_value, full_value, atol=1e-6)
    assert int(contribution) == 8


def test_train_chunk_runner_owns_macro_batch_iteration():
    state, driver, learning_stack = _make_state_and_driver(seed=3, micro_batch_size=4)
    train_arrays = {
        "x": jnp.ones((24, 28, 28, 1), dtype=jnp.float32),
        "y": jnp.arange(24, dtype=jnp.int32) % 10,
    }

    train_chunk_step = make_train_update_chunk_step(driver, learning_stack)
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


def test_validation_runner_reduces_metrics_across_batches():
    state, driver, _ = _make_state_and_driver(seed=4, micro_batch_size=2)
    eval_step = make_eval_step(driver)
    runner = ValidationRunner(
        arrays={
            "x": jnp.ones((5, 28, 28, 1), dtype=jnp.float32),
            "y": jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32),
        },
        macro_batch_size=3,
        eval_step=eval_step,
        execution_driver=driver,
    )

    metrics = runner.evaluate(state.variables)

    assert "eval/loss" in metrics
    assert "eval/accuracy" in metrics
    assert jnp.isfinite(metrics["eval/loss"])
    assert jnp.isfinite(metrics["eval/accuracy"])
