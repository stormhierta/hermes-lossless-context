"""Hermes Lossless Context addon.

This module can be installed as a user plugin under ``~/.hermes/plugins`` or
loaded as a context engine package. It intentionally does not monkeypatch Hermes.
"""

from .engine import LosslessContextEngine
from .plugin import register

__all__ = ["LosslessContextEngine", "register"]
__version__ = "0.1.5"
