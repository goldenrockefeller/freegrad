"""Helpers for semantic parameter layouts."""

from __future__ import annotations

from typing import Any

import jax

from freegrad.models.common.base import ParamGroup, ParamLayout


def infer_param_layout(params: Any) -> ParamLayout:
    """Infer a simple path/shape/dtype layout from a pytree."""

    groups: list[ParamGroup] = []
    leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(params)
    for path, leaf in leaves_with_paths:
        names = tuple(_path_entry_to_string(entry) for entry in path)
        groups.append(
            ParamGroup(
                path=names,
                role=_infer_role(names),
                shape=tuple(getattr(leaf, "shape", ())),
                dtype=getattr(leaf, "dtype", None),
            )
        )
    return ParamLayout(groups=tuple(groups))


def _path_entry_to_string(entry: Any) -> str:
    key = getattr(entry, "key", None)
    if key is not None:
        return str(key)
    idx = getattr(entry, "idx", None)
    if idx is not None:
        return str(idx)
    return str(entry)


def _infer_role(path: tuple[str, ...]) -> str | None:
    if not path:
        return None
    name = path[-1].lower()
    if name in {"b", "bias"}:
        return "bias"
    if name in {"w", "weight", "kernel"}:
        return "weight"
    if "norm" in ".".join(path).lower():
        return "normalization"
    return None
