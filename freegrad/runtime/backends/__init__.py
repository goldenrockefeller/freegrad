"""Execution backends."""

from freegrad.runtime.backends.base import Backend
from freegrad.runtime.backends.local import LocalBackend

__all__ = ["Backend", "LocalBackend"]