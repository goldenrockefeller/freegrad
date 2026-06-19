from __future__ import annotations

from pathlib import Path

from freegrad.execution import ExecutionDriver, SingleCPUExecutionDriver
from freegrad.learning import LearningStack, adam
from freegrad.losses.classification import CrossEntropyLoss
from freegrad.metrics.classification import ClassificationMetrics
from freegrad.models.common.base import Model
from freegrad.models.mlp import SimpleMLP
from freegrad.models.small_cnn import ModernSmallCNN
from freegrad.runtime.backends.base import Backend
from freegrad.runtime.backends.local import LocalBackend
from freegrad.runtime.tracking.base import MetricLogger
from freegrad.runtime.tracking.jsonl import JSONLLogger


def test_concrete_classes_implement_explicit_protocols(tmp_path):
    assert isinstance(SimpleMLP(), Model)
    assert isinstance(ModernSmallCNN(), Model)
    assert isinstance(adam(learning_rate=1e-3), LearningStack)
    assert isinstance(
        SingleCPUExecutionDriver(
            model=SimpleMLP(),
            loss=CrossEntropyLoss(),
            metric=ClassificationMetrics(),
            micro_batch_size=4,
        ),
        ExecutionDriver,
    )
    assert isinstance(LocalBackend(), Backend)
    assert isinstance(JSONLLogger(tmp_path / "metrics.jsonl"), MetricLogger)


def test_backend_and_logger_protocol_construction(tmp_path):
    backend = LocalBackend(root_output_dir=tmp_path)
    logger = JSONLLogger(Path(tmp_path) / "metrics.jsonl")
    logger.close()

    assert isinstance(backend, Backend)
