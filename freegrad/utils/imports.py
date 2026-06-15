"""Import helpers for script entry points."""

from __future__ import annotations

import importlib


def import_from_string(path: str):
    try:
        module_name, attr_name = path.split(":", 1)
    except ValueError as exc:
        raise ValueError(f"Import path must look like 'module.submodule:attribute', got: {path!r}") from exc

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise ImportError(f"Could not import module {module_name!r} from {path!r}") from exc

    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise ImportError(f"Module {module_name!r} does not define attribute {attr_name!r}") from exc