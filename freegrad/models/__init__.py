"""Model definitions."""

from freegrad.models.common.base import Model
from freegrad.models.mlp import SimpleMLP
from freegrad.models.small_cnn import ModernSmallCNN

__all__ = ["Model", "SimpleMLP", "ModernSmallCNN"]