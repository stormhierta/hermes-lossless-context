import json

from lossless_context.engine import LosslessContextEngine
from lossless_context.store import LosslessStore
from lossless_context.utils import sanitize_fts_query


def test_store_ingests_searches_and_expands(tmp_path):
    store = LosslessStore(tmp_path / "lcm.db")
    cid = store.get_or_create_conversation("s1")
    inserted = store.ingest_messages(cid, [
        {"role": "user", "content": "remember the alpha deployment key was rotated"},
        {"role": "assistant", "content": "noted alpha deployment rotation"},
    ])
    assert inserted == 2
    results = store.grep("alpha deployment", conversation_id=cid)
    assert results
    chunk = store.unsummarized_messages(cid, protect_last_n=0, limit_tokens=1000)
    sid = store.create_leaf_summary(cid, chunk, "alpha deployment summary")
    assert sid.startswith("sum_")
    sources = store.source_messages_for_summary(sid)
    assert [m.seq for m in sources] == [1, 2]
    assert store.integrity_report()["ok"] is True


def test_fts_sanitizer_drops_operator_punctuation():
    sanitized = sanitize_fts_query("@foo:bar -baz AND qux OR NOT NEAR")
    assert "@" not in sanitized
    assert ":" not in sanitized
    assert "-" not in sanitized
    assert "AND" not in sanitized
    assert "OR" not in sanitized
    assert "NOT" not in sanitized
    assert "NEAR" not in sanitized


def test_grep_full_text_special_chars_do_not_crash(tmp_path):
    store = LosslessStore(tmp_path / "lcm.db")
    cid = store.get_or_create_conversation("s1")
    store.ingest_messages(cid, [{"role": "user", "content": "foo bar baz"}])
    assert isinstance(store.grep("@foo:bar -baz", conversation_id=cid), list)


def test_engine_compress_preserves_source_and_returns_valid_messages(tmp_path):
    engine = LosslessContextEngine(db_path=tmp_path / "lcm.db", context_length=200)
    engine.on_session_start("session-a")
    messages = [{"role": "system", "content": "sys"}]
    for i in range(12):
        messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"message {i} about beta incident details"})
    out = engine.compress(messages, current_tokens=500)
    assert all("role" in m and "content" in m for m in out)
    assert len(out) < len(messages)
    assert any(m["role"] == "assistant" and "untrusted reference data" in m["content"] for m in out)
    status = engine.get_status()
    assert status["compression_count"] >= 1
    grep = json.loads(engine.handle_tool_call("lcm_grep", {"pattern": "beta incident"}))
    assert grep["results"]
    summary_id = next(r["summaryId"] for r in grep["results"] if r["type"] == "summary")
    expanded = json.loads(engine.handle_tool_call("lcm_expand", {"id": summary_id}))
    assert expanded["messages"]


def test_plugin_registers_engine_and_tools():
    from lossless_context.plugin import register

    class Ctx:
        def __init__(self):
            self.engines = []
            self.tools = []
        def register_context_engine(self, engine):
            self.engines.append(engine)
        def register_tool(self, **kwargs):
            self.tools.append(kwargs)

    ctx = Ctx()
    register(ctx)
    assert ctx.engines and ctx.engines[0].name == "lossless"
    assert {t["name"] for t in ctx.tools} >= {"lcm_grep", "lcm_describe", "lcm_expand", "lcm_status"}


def test_user_plugin_shim_contains_manifest():
    from pathlib import Path
    import yaml

    manifest_path = Path(__file__).resolve().parents[1] / "plugin_entry" / "plugin.yaml"
    assert manifest_path.exists()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["name"] == "lossless_context"
    assert manifest["module"] == "lossless_context"


def test_package_exposes_hermes_plugin_entry_point():
    import tomllib
    from pathlib import Path

    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["entry-points"]["hermes_agent.plugins"]["lossless_context"] == "lossless_context"


def test_get_summary_maps_fields_correctly(tmp_path):
    store = LosslessStore(tmp_path / "lcm.db")
    cid = store.get_or_create_conversation("s-summary")
    store.ingest_messages(cid, [
        {"role": "user", "content": "first summary source"},
        {"role": "assistant", "content": "second summary source"},
    ])
    sid = store.create_leaf_summary(cid, store.unsummarized_messages(cid, protect_last_n=0, limit_tokens=1000), "summary body", model="test-model")
    summary = store.get_summary(sid)
    assert summary is not None
    assert summary.summary_id == sid
    assert summary.conversation_id == cid
    assert summary.earliest_seq == 1
    assert summary.latest_seq == 2
    assert summary.created_at > 0


def test_sanitizer_preserves_unicode_terms():
    sanitized = sanitize_fts_query("猫 café мир AND")
    assert "猫" in sanitized
    assert "café" in sanitized
    assert "мир" in sanitized
    assert "AND" not in sanitized


def test_engine_rejects_env_db_path_outside_state_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_LCM_DB", str(tmp_path / "outside.db"))
    try:
        LosslessContextEngine()
    except ValueError as exc:
        assert "must stay under" in str(exc)
    else:
        raise AssertionError("outside HERMES_LCM_DB path should be rejected")


def test_lcm_expand_truncated_only_when_more_sources_exist(tmp_path):
    engine = LosslessContextEngine(db_path=tmp_path / "lcm.db", context_length=200)
    engine.on_session_start("session-expand")
    messages = []
    for i in range(6):
        messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"expand source {i}"})
    engine.store.ingest_messages(engine.conversation_id or 0, messages)
    chunk = engine.store.unsummarized_messages(engine.conversation_id or 0, protect_last_n=0, limit_tokens=1000)
    sid = engine.store.create_leaf_summary(engine.conversation_id or 0, chunk, "expand summary")
    exact = json.loads(engine.handle_tool_call("lcm_expand", {"id": sid, "maxMessages": len(chunk)}))
    assert exact["truncated"] is False
    limited = json.loads(engine.handle_tool_call("lcm_expand", {"id": sid, "maxMessages": 2}))
    assert limited["truncated"] is True
    assert len(limited["messages"]) == 2


def test_compressed_summary_reference_is_not_system_role(tmp_path):
    engine = LosslessContextEngine(db_path=tmp_path / "lcm.db", context_length=200)
    engine.on_session_start("session-role")
    messages = [{"role": "system", "content": "real system prompt"}]
    for i in range(12):
        messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"role check incident {i}"})
    out = engine.compress(messages, current_tokens=500)
    reference_msgs = [m for m in out if "lossless_context_summary" in m.get("content", "")]
    assert reference_msgs
    assert all(m["role"] == "assistant" for m in reference_msgs)
