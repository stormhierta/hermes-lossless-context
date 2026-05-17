"""User-plugin shim for Hermes.

Install by copying this directory to `~/.hermes/plugins/lossless_context/`.
"""
from lossless_context.plugin import register

__all__ = ["register"]
