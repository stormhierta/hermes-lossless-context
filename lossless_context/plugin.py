from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .engine import LosslessContextEngine

_engine: LosslessContextEngine | None = None


def get_engine() -> LosslessContextEngine:
    global _engine
    if _engine is None:
        _engine = LosslessContextEngine()
    return _engine


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
    engine = get_engine()
    if hasattr(ctx, "register_context_engine"):
        ctx.register_context_engine(engine)
    if hasattr(ctx, "register_tool"):
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
