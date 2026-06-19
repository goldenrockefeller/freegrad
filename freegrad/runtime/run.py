"""Single-run execution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np

from freegrad.execution.single_cpu import SingleCPUExecutionDriver
from freegrad.losses.base import ReducibleLoss
from freegrad.losses.classification import CrossEntropyLoss
from freegrad.metrics.base import ReducibleMetric
from freegrad.metrics.classification import ClassificationMetrics
from freegrad.models.common.base import Model
from freegrad.runtime.checkpointing.pickle_fallback import latest_checkpoint, load_checkpoint, save_checkpoint
from freegrad.runtime.context import RunContext
from freegrad.runtime.interfaces import (
    DataPreparerBuilder,
    DatasetLoader,
    LearningStackBuilder,
    LossBuilder,
    MetricsBuilder,
    ModelBuilder,
)
from freegrad.runtime.paths import checkpoints_dir
from freegrad.runtime.status import write_status
from freegrad.runtime.tracking.base import MetricLogger
from freegrad.runtime.tracking.jsonl import JSONLLogger
from freegrad.training.eval_step import make_eval_step
from freegrad.training.loops import TrainChunkRunner, ValidationRunner
from freegrad.training.state import TrainState
from freegrad.training.train_step import make_train_update_chunk_step
from freegrad.utils.rng import make_rng


@dataclass(frozen=True)
class RunSpec:
    name: str
    condition_name: str
    seed: int
    output_dir: Path
    dataset_loader: DatasetLoader
    data_preparer_builder: DataPreparerBuilder
    model_builder: ModelBuilder
    learning_stack_builder: LearningStackBuilder
    loss_builder: LossBuilder
    metrics_builder: MetricsBuilder
    training_config: dict[str, Any]
    study_name: str | None = None


@dataclass(frozen=True)
class RunResult:
    name: str
    status: str
    output_dir: Path
    final_metrics: dict[str, Any]

# TODO add n_chunks_per_plot
@dataclass(frozen=True)
class TrainingLoopConfig:
    mini_batch_size: int
    macro_batch_size: int
    max_steps: int
    train_chunk_size: int
    eval_every: int
    n_chunks_per_checkpoint: int
    resume: bool


@dataclass(frozen=True)
class ValidationStepConfig:
    mini_batch_size: int
    macro_batch_size: int


def checkpoint_metadata(run_spec: RunSpec, *, final: bool = False) -> dict[str, Any]:
    metadata = {
        "run_name": run_spec.name,
        "condition_name": run_spec.condition_name,
    }
    if run_spec.study_name is not None:
        metadata["study_name"] = run_spec.study_name
    if final:
        metadata["final"] = True
    return metadata


def run_one(run_spec: RunSpec) -> RunResult:
    context = _make_context(run_spec.output_dir)
    context.output_dir.mkdir(parents=True, exist_ok=True)
    _write_run_metadata(run_spec, context.run_path)
    write_status(context.output_dir, "PENDING", step=0)
    logger: MetricLogger = JSONLLogger(context.metrics_path)

    try:
        write_status(context.output_dir, "RUNNING", step=0)
        config = dict(run_spec.training_config)
        loop_config = _parse_training_config(config)
        raw_data = run_spec.dataset_loader()
        prepare_arrays = run_spec.data_preparer_builder()
        validation_size = int(config.get("validation_size", 5000))
        try:
            arrays = prepare_arrays(raw_data, validation_size=validation_size)
        except TypeError:
            arrays = prepare_arrays(raw_data)

        learning_stack = run_spec.learning_stack_builder()
        model = run_spec.model_builder()
        loss = _build_loss(run_spec.loss_builder)
        metric = _build_metric(run_spec.metrics_builder)
        validation_config = _parse_validation_config(config)

        train_driver = SingleCPUExecutionDriver(
            model=model,
            loss=loss,
            metric=metric,
            micro_batch_size=loop_config.mini_batch_size,
        )
        eval_driver = SingleCPUExecutionDriver(
            model=model,
            loss=loss,
            metric=metric,
            micro_batch_size=validation_config.mini_batch_size,
        )

        train_update_chunk_step = make_train_update_chunk_step(train_driver, learning_stack)
        eval_step = make_eval_step(eval_driver)
        train_runner = TrainChunkRunner(
            arrays={"x": arrays["train_images"], "y": arrays["train_labels"]},
            macro_batch_size=loop_config.macro_batch_size,
            train_chunk_size=loop_config.train_chunk_size,
            max_steps=loop_config.max_steps,
            seed=run_spec.seed,
            train_update_chunk_step=train_update_chunk_step,
        )
        validation_runner = _make_validation_runner(arrays, validation_config, eval_step, eval_driver)

        state = _initialize_or_resume_state(
            run_spec=run_spec,
            arrays=arrays,
            model=model,
            learning_stack=learning_stack,
            resume=loop_config.resume,
            checkpoint_dir=context.checkpoint_dir,
        )

        last_train_metrics: dict[str, float] = {}
        last_eval_metrics: dict[str, float] = {}
        last_eval_step: int | None = None
        completed_train_chunks = int(np.asarray(state.step)) // loop_config.train_chunk_size
        while train_runner.has_next_chunk(state):
            chunk_result = train_runner.run_next_chunk(state)
            state = chunk_result.state
            current_step = int(np.asarray(state.step))
            completed_train_chunks += 1
            last_train_metrics = _to_python_dict(chunk_result.metrics)
            _record_chunk_artifacts(
                context=context,
                logger=logger,
                run_spec=run_spec,
                n_chunks_per_checkpoint=loop_config.n_chunks_per_checkpoint,
                completed_train_chunks=completed_train_chunks,
                state=state,
                train_metrics=last_train_metrics,
            )
            should_validate = (
                loop_config.eval_every > 0
                and completed_train_chunks % loop_config.eval_every == 0
            ) or _is_checkpoint_chunk(loop_config.n_chunks_per_checkpoint, completed_train_chunks)
            if should_validate:
                last_eval_metrics = _to_python_dict(validation_runner.evaluate(state.variables))
                logger.log_metrics(current_step, last_eval_metrics)
                last_eval_step = current_step

        final_step = int(np.asarray(state.step))
        if last_eval_step != final_step:
            last_eval_metrics = _to_python_dict(validation_runner.evaluate(state.variables))
            logger.log_metrics(final_step, last_eval_metrics)
            last_eval_step = final_step

        if loop_config.n_chunks_per_checkpoint > 0:
            save_checkpoint(
                context.checkpoint_dir / f"step_{final_step:08d}.pkl",
                state,
                metadata=checkpoint_metadata(run_spec, final=True),
            )

        final_metrics = {**last_train_metrics, **last_eval_metrics}
        _update_run_result(context.run_path, final_metrics, status="COMPLETED")
        write_status(context.output_dir, "COMPLETED", step=final_step)
        return RunResult(name=run_spec.name, status="COMPLETED", output_dir=context.output_dir, final_metrics=final_metrics)
    except Exception as exc:
        _update_run_result(context.run_path, final_metrics={}, status="FAILED", error=str(exc))
        write_status(context.output_dir, "FAILED", message=str(exc))
        return RunResult(name=run_spec.name, status="FAILED", output_dir=context.output_dir, final_metrics={"error": str(exc)})
    finally:
        logger.close()


def _make_context(output_dir: Path) -> RunContext:
    checkpoint_dir = checkpoints_dir(output_dir)
    return RunContext(
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        metrics_path=output_dir / "metrics.jsonl",
        status_path=output_dir / "status.json",
        run_path=output_dir / "run.json",
    )


def _initialize_or_resume_state(
    run_spec: RunSpec,
    arrays: dict[str, Any],
    model: Model,
    learning_stack,
    resume: bool,
    checkpoint_dir: Path,
) -> TrainState:
    if resume:
        checkpoint_path = latest_checkpoint(checkpoint_dir)
        if checkpoint_path is not None:
            state, _ = load_checkpoint(checkpoint_path)
            return state

    rng_key = make_rng(run_spec.seed)
    model_key, train_key = jax.random.split(rng_key)
    input_shape = tuple(arrays["train_images"].shape[1:])
    num_classes = int(np.max(arrays["train_labels"])) + 1
    variables = model.init(model_key, input_shape=input_shape, num_classes=num_classes)
    learning_state = learning_stack.init(variables)
    return TrainState(step=0, variables=variables, learning_state=learning_state, rng_key=train_key)


def _parse_training_config(config: dict[str, Any]) -> TrainingLoopConfig:
    _reject_unknown_training_config_keys(config)

    required_keys = {
        "mini_batch_size",
        "macro_batch_size",
        "max_steps",
        "train_chunk_size",
        "eval_every",
        "n_chunks_per_checkpoint",
        "resume",
    }
    missing = sorted(key for key in required_keys if key not in config)
    if missing:
        raise ValueError("training_config is missing required keys: " + ", ".join(missing) + ".")

    mini_batch_size = int(config["mini_batch_size"])
    macro_batch_size = int(config["macro_batch_size"])
    if mini_batch_size <= 0:
        raise ValueError("mini_batch_size must be positive.")
    if macro_batch_size < mini_batch_size:
        raise ValueError("macro_batch_size must be at least mini_batch_size.")
    if macro_batch_size % mini_batch_size != 0:
        raise ValueError("macro_batch_size must be divisible by mini_batch_size.")

    train_chunk_size = int(config["train_chunk_size"])
    eval_every = int(config["eval_every"])
    max_steps = int(config["max_steps"])
    if train_chunk_size <= 0:
        raise ValueError("train_chunk_size must be positive.")
    if eval_every <= 0:
        raise ValueError("eval_every must be positive.")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")

    return TrainingLoopConfig(
        mini_batch_size=mini_batch_size,
        macro_batch_size=macro_batch_size,
        max_steps=max_steps,
        train_chunk_size=train_chunk_size,
        eval_every=eval_every,
        n_chunks_per_checkpoint=int(config["n_chunks_per_checkpoint"]),
        resume=bool(config["resume"]),
    )


def _parse_validation_config(config: dict[str, Any]) -> ValidationStepConfig:
    required_keys = {"eval_mini_batch_size", "eval_macro_batch_size"}
    missing = sorted(key for key in required_keys if key not in config)
    if missing:
        raise ValueError("training_config is missing required keys: " + ", ".join(missing) + ".")

    mini_batch_size = int(config["eval_mini_batch_size"])
    macro_batch_size = int(config["eval_macro_batch_size"])
    if mini_batch_size <= 0:
        raise ValueError("eval_mini_batch_size must be positive.")
    if macro_batch_size < mini_batch_size:
        raise ValueError("eval_macro_batch_size must be at least eval_mini_batch_size.")
    if macro_batch_size % mini_batch_size != 0:
        raise ValueError("eval_macro_batch_size must be divisible by eval_mini_batch_size.")

    return ValidationStepConfig(mini_batch_size=mini_batch_size, macro_batch_size=macro_batch_size)


def _reject_unknown_training_config_keys(config: dict[str, Any]) -> None:
    allowed_keys = {
        "mini_batch_size",
        "macro_batch_size",
        "eval_mini_batch_size",
        "eval_macro_batch_size",
        "max_steps",
        "train_chunk_size",
        "eval_every",
        "n_chunks_per_checkpoint",
        "resume",
        "validation_size",
    }
    unknown = sorted(key for key in config if key not in allowed_keys)
    if unknown:
        raise ValueError("training_config has unsupported keys: " + ", ".join(unknown) + ".")


def _record_chunk_artifacts(
    *,
    context: RunContext,
    logger: MetricLogger,
    run_spec: RunSpec,
    n_chunks_per_checkpoint: int,
    completed_train_chunks: int,
    state: TrainState,
    train_metrics: dict[str, float],
) -> None:
    current_step = int(np.asarray(state.step))
    logger.log_metrics(current_step, train_metrics)
    write_status(context.output_dir, "RUNNING", step=current_step)
    if _is_checkpoint_chunk(n_chunks_per_checkpoint, completed_train_chunks):
        save_checkpoint(
            context.checkpoint_dir / f"step_{current_step:08d}.pkl",
            state,
            metadata=checkpoint_metadata(run_spec),
        )


def _is_checkpoint_chunk(n_chunks_per_checkpoint: int, completed_train_chunks: int) -> bool:
    return (
        n_chunks_per_checkpoint > 0
        and completed_train_chunks > 0
        and completed_train_chunks % n_chunks_per_checkpoint == 0
    )


def _make_validation_runner(
    arrays: dict[str, Any],
    validation_config: ValidationStepConfig,
    eval_step,
    eval_driver,
) -> ValidationRunner:
    if arrays["validation_images"].shape[0] > 0:
        eval_arrays = {"x": arrays["validation_images"], "y": arrays["validation_labels"]}
    else:
        eval_arrays = {"x": arrays["test_images"], "y": arrays["test_labels"]}
    return ValidationRunner(
        arrays=eval_arrays,
        macro_batch_size=validation_config.macro_batch_size,
        eval_step=eval_step,
        execution_driver=eval_driver,
    )


def _build_loss(loss_builder: LossBuilder) -> ReducibleLoss:
    try:
        loss = loss_builder()
    except TypeError:
        loss = loss_builder(None)
    if hasattr(loss, "apply") and hasattr(loss, "reduce") and hasattr(loss, "finalize"):
        return loss
    return CrossEntropyLoss()


def _build_metric(metrics_builder: MetricsBuilder) -> ReducibleMetric:
    try:
        metric = metrics_builder()
    except TypeError:
        metric = metrics_builder(None)
    if hasattr(metric, "apply") and hasattr(metric, "reduce") and hasattr(metric, "finalize"):
        return metric
    return ClassificationMetrics()


def _write_run_metadata(run_spec: RunSpec, path: Path) -> None:
    payload = {
        "name": run_spec.name,
        "condition_name": run_spec.condition_name,
        "study_name": run_spec.study_name,
        "seed": run_spec.seed,
        "output_dir": str(run_spec.output_dir),
        "training_config": run_spec.training_config,
        "status": "PENDING",
        "final_metrics": {},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _update_run_result(path: Path, final_metrics: dict, status: str, error: str | None = None) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = status
    payload["final_metrics"] = final_metrics
    if error is not None:
        payload["error"] = error
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _to_python_dict(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: _to_python_float(value) for key, value in metrics.items()}


def _to_python_float(value: Any) -> float:
    array = np.asarray(jax.device_get(value))
    return float(array.reshape(()))
