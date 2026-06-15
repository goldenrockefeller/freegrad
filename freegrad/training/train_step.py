"""Factory for JIT-compiled training steps."""

from __future__ import annotations

import jax

from freegrad.grad_scalers.const import ConstGradScaler
from freegrad.training.state import TrainState
from freegrad.utils.pytrees import tree_l2_norm


def _train_step_impl(state: TrainState, batch, loss_fn, metrics_fn, optimizer, grad_scaler_fn):
    next_rng, _ = jax.random.split(state.rng_key)

    def objective(params):
        return loss_fn(params, batch["x"], batch["y"])

    grad_fn = jax.grad(objective)
    grads = grad_fn(state.params)
    params, optimizer_state, optimizer_metrics = optimizer.step(
        params=state.params,
        state=state.optimizer_state,
        grad_scaler_state=getattr(state.optimizer_state, "grad_scaler_state", None),
        loss_fn=objective,
        grad_fn=grad_fn,
        grad_scaler_fn=grad_scaler_fn,
    )
    updates = jax.tree_util.tree_map(lambda new, old: new - old, params, state.params)
    train_metrics = metrics_fn(state.params, batch["x"], batch["y"])
    metrics = {
        **{f"train/{name}": value for name, value in train_metrics.items()},
        "grad_norm": tree_l2_norm(grads),
        "update_norm": tree_l2_norm(updates),
        "param_norm": tree_l2_norm(params),
        **optimizer_metrics,
    }
    new_state = state.replace(
        step=state.step + 1,
        params=params,
        optimizer_state=optimizer_state,
        rng_key=next_rng,
    )
    return new_state, metrics


def make_train_update_chunk_step(loss_fn, metrics_fn, optimizer, grad_scaler_fn=None):
    if grad_scaler_fn is None:
        grad_scaler_fn = ConstGradScaler().scale

    @jax.jit
    def train_update_chunk_step(state: TrainState, update_batches):
        def scan_step(carry, batch):
            return _train_step_impl(carry, batch, loss_fn, metrics_fn, optimizer, grad_scaler_fn)

        state, metrics = jax.lax.scan(scan_step, state, update_batches)
        last_metrics = jax.tree_util.tree_map(lambda value: value[-1], metrics)
        return state, last_metrics

    return train_update_chunk_step