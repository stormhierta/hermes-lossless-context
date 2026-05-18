from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .engine import LosslessContextEngine

_tool_engine: LosslessContextEngine | None = None


def get_engine() -> LosslessContextEngine:
    """Return the passive recall-tool engine.

    Active context-engine registrations receive fresh instances in register()
    so gateway sessions do not share mutable per-session state. The passive
    tool engine is safe to reuse because LosslessStore uses thread-local SQLite
    connections.
    """
    global _tool_engine
    if _tool_engine is None:
        _tool_engine = LosslessContextEngine()
    return _tool_engine


def _requirements_available() -> bool:
    try:
        engine = get_engine()
        engine.db_path.parent.mkdir(parents=True, exist_ok=True)
        return os.access(engine.db_path.parent, os.W_OK)
    except Exception:
        return False


def _tool_handler(name: str):
    def handler(args: dict[str, Any], **kwargs: Any) -> str:
        return get_engine().handle_tool_call(name, args, **kwargs)
    return handler


def register(ctx: Any) -> None:
    """Hermes plugin entrypoint.

    Registers the opt-in ContextEngine and also exposes passive recall tools when
    the plugin is enabled as a normal Hermes plugin. Full engine mode still
    requires `context.engine: lossless`.
    """
    if hasattr(ctx, "register_context_engine"):
        ctx.register_context_engine(LosslessContextEngine())
    if hasattr(ctx, "register_tool"):
        engine = get_engine()
        for schema in engine.get_tool_schemas():
            name = schema["name"]
            ctx.register_tool(
                name=name,
                toolset="lossless_context",
                schema=schema,
                handler=_tool_handler(name),
                check_fn=_requirements_available,
                emoji="🧠",
            )
