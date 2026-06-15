"""Factory for JIT-compiled evaluation steps."""

from __future__ import annotations

import jax


def make_eval_step(metrics_fn):
    @jax.jit
    def eval_step(params, batch):
        metrics = metrics_fn(params, batch["x"], batch["y"])
        return {f"eval/{name}": value for name, value in metrics.items()}

    return eval_step