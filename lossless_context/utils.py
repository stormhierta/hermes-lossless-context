from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

CHARS_PER_TOKEN = 4


def estimate_tokens(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return max(1, len(value) // CHARS_PER_TOKEN) if value else 0
    if isinstance(value, list):
        return sum(estimate_tokens(v.get("text", v) if isinstance(v, dict) else v) for v in value)
    try:
        return max(1, len(json.dumps(value, ensure_ascii=False)) // CHARS_PER_TOKEN)
    except Exception:
        return max(1, len(str(value)) // CHARS_PER_TOKEN)


def content_to_text(content: Any, *, max_chars: int | None = None) -> str:
    if content is None:
        text = ""
    elif isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item.get("type") in {"image", "image_url", "input_image"}:
                    parts.append("[image]")
                else:
                    try:
                        parts.append(json.dumps(item, ensure_ascii=False)[:1000])
                    except Exception:
                        parts.append(str(item)[:1000])
        text = "\n".join(p for p in parts if p)
    else:
        try:
            text = json.dumps(content, ensure_ascii=False)
        except Exception:
            text = str(content)
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars] + "…[truncated]"
    return text


def stable_hash(*parts: Any) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(content_to_text(part).encode("utf-8", errors="replace"))
        h.update(b"\0")
    return h.hexdigest()


def safe_snippet(text: str, query: str = "", limit: int = 500) -> str:
    text = text or ""
    if not query:
        return text[:limit] + ("…" if len(text) > limit else "")
    idx = text.lower().find(query.lower())
    if idx < 0:
        return text[:limit] + ("…" if len(text) > limit else "")
    start = max(0, idx - limit // 3)
    end = min(len(text), start + limit)
    return ("…" if start else "") + text[start:end] + ("…" if end < len(text) else "")


def sanitize_fts_query(query: str) -> str:
    # SQLite FTS MATCH is parameter-bound, but its query language still has
    # operators that can raise or alter query semantics. Keep non-operator word
    # tokens and quoted phrases only; drop punctuation and FTS boolean operators.
    blocked = {"AND", "OR", "NOT", "NEAR"}
    tokens = []
    for token in re.findall(r'"[^"]+"|[\w]+', query or ""):
        if token.startswith('"') and token.endswith('"'):
            words = [w for w in re.findall(r"\w+", token[1:-1]) if w.upper() not in blocked]
            if words:
                tokens.append('"' + " ".join(words[:8]) + '"')
        else:
            cleaned = re.sub(r"\W+", "", token)
            if cleaned and cleaned.upper() not in blocked:
                tokens.append(cleaned)
    return " ".join(tokens[:32]) or '""'


def escape_like_pattern(pattern: str) -> str:
    return (pattern or "").replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def default_state_dir() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home() / "plugins" / "lossless-context"
    except Exception:
        return Path.home() / ".hermes" / "plugins" / "lossless-context"
