from __future__ import annotations

from pathlib import Path

from freegrad.models.common.base import Model
from freegrad.models.mlp import SimpleMLP
from freegrad.models.small_cnn import ModernSmallCNN
from freegrad.grad_scalers.common.base import GradScaler
from freegrad.grad_scalers.const import ConstGradScaler
from freegrad.optimizers.adam import Adam
from freegrad.optimizers.common.base import Optimizer
from freegrad.runtime.backends.base import Backend
from freegrad.runtime.backends.local import LocalBackend
from freegrad.runtime.tracking.base import MetricLogger
from freegrad.runtime.tracking.jsonl import JSONLLogger


def test_concrete_classes_implement_explicit_protocols(tmp_path):
    assert isinstance(SimpleMLP(), Model)
    assert isinstance(ModernSmallCNN(), Model)
    assert isinstance(ConstGradScaler(constant=1.0), GradScaler)
    assert isinstance(Adam(learning_rate=1e-3), Optimizer)
    assert isinstance(LocalBackend(), Backend)
    assert isinstance(JSONLLogger(tmp_path / "metrics.jsonl"), MetricLogger)


def test_backend_and_logger_protocol_construction(tmp_path):
    backend = LocalBackend(root_output_dir=tmp_path)
    logger = JSONLLogger(Path(tmp_path) / "metrics.jsonl")
    logger.close()

    assert isinstance(backend, Backend)