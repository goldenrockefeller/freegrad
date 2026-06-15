from __future__ import annotations

import jax.numpy as jnp

from freegrad.grad_scalers.const import ConstGradScaler
from freegrad.optimizers.adam import Adam


def quadratic_loss(params):
    return jnp.sum(jnp.square(params["x"]))


def quadratic_grad(params):
    return {"x": 2.0 * params["x"]}


def test_adam_init_matches_param_structure():
    params = {"w": jnp.ones((2, 3)), "b": jnp.ones((3,))}
    optimizer = Adam(learning_rate=0.1)

    state = optimizer.init(params)

    assert state.mu.keys() == params.keys()
    assert state.nu.keys() == params.keys()
    assert state.grad_scaler_state is None
    assert state.mu["w"].shape == params["w"].shape
    assert state.nu["b"].shape == params["b"].shape


def test_adam_update_matches_param_structure():
    params = {"w": jnp.ones((2, 3)), "b": jnp.ones((3,))}
    optimizer = Adam(learning_rate=0.1)

    state = optimizer.init(params)
    new_params, new_state, metrics = optimizer.step(
        params=params,
        state=state,
        grad_scaler_state=state.grad_scaler_state,
        loss_fn=quadratic_loss,
        grad_fn=quadratic_grad,
        grad_scaler_fn=ConstGradScaler().scale,
    )

    assert new_params.keys() == params.keys()
    assert new_params["w"].shape == params["w"].shape
    assert new_state.mu["b"].shape == params["b"].shape
    assert "optimizer/learning_rate" in metrics


def test_adam_step_moves_x_toward_zero():
    params = {"x": jnp.array([2.0], dtype=jnp.float32)}
    optimizer = Adam(learning_rate=0.1)

    state = optimizer.init(params)
    new_params, _, _ = optimizer.step(
        params=params,
        state=state,
        grad_scaler_state=state.grad_scaler_state,
        loss_fn=quadratic_loss,
        grad_fn=quadratic_grad,
        grad_scaler_fn=ConstGradScaler().scale,
    )

    assert jnp.abs(new_params["x"][0]) < jnp.abs(params["x"][0])


def test_adam_respects_const_grad_scaler():
    params = {"x": jnp.array([2.0], dtype=jnp.float32)}
    optimizer = Adam(learning_rate=0.1)

    state = optimizer.init(params)
    new_params, _, metrics = optimizer.step(
        params=params,
        state=state,
        grad_scaler_state=state.grad_scaler_state,
        loss_fn=quadratic_loss,
        grad_fn=quadratic_grad,
        grad_scaler_fn=ConstGradScaler(constant=0.5).scale,
    )

    assert "grad_scaler/constant" in metrics
    assert jnp.abs(new_params["x"][0]) < jnp.abs(params["x"][0])