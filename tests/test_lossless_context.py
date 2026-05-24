import concurrent.futures
import json
import threading

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


def test_lcm_grep_searches_indexed_history_by_default(tmp_path):
    engine = LosslessContextEngine(db_path=tmp_path / "lcm.db", context_length=200)
    engine.on_session_start("historical-session")
    engine.store.ingest_messages(engine.conversation_id or 0, [
        {"role": "user", "content": "historic recall marker alpha"},
    ])
    engine.conversation_id = None
    engine.on_session_start("current-session")
    global_results = json.loads(engine.handle_tool_call("lcm_grep", {"pattern": "historic recall marker", "mode": "like"}))
    assert global_results["results"]
    current_results = json.loads(engine.handle_tool_call("lcm_grep", {"pattern": "historic recall marker", "mode": "like", "currentOnly": True}))
    assert current_results["results"] == []


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


def test_plugin_registers_fresh_context_engine_instances():
    from lossless_context.plugin import register

    class Ctx:
        def __init__(self):
            self.engines = []
        def register_context_engine(self, engine):
            self.engines.append(engine)
        def register_tool(self, **kwargs):
            pass

    first = Ctx()
    second = Ctx()
    register(first)
    register(second)
    assert first.engines and second.engines
    assert first.engines[0] is not second.engines[0]


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


def test_store_uses_thread_local_sqlite_connections(tmp_path):
    store = LosslessStore(tmp_path / "lcm.db")
    main_conn_id = id(store.conn)
    worker_conn_ids = []

    def touch_store():
        cid = store.get_or_create_conversation("thread-local")
        store.ingest_messages(cid, [{"role": "user", "content": "worker thread message"}])
        worker_conn_ids.append(id(store.conn))

    thread = threading.Thread(target=touch_store)
    thread.start()
    thread.join()

    assert worker_conn_ids
    assert worker_conn_ids[0] != main_conn_id


def test_store_supports_concurrent_threadpool_access(tmp_path):
    store = LosslessStore(tmp_path / "lcm.db")
    cid = store.get_or_create_conversation("concurrent-store")

    def ingest(i: int):
        store.ingest_messages(cid, [{"role": "user", "content": f"concurrent message {i}"}])

    def read():
        store.grep("concurrent", conversation_id=cid)
        store.unsummarized_messages(cid, protect_last_n=2, limit_tokens=1000)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(ingest, i) for i in range(40)]
        futures.extend(executor.submit(read) for _ in range(40))
        for future in futures:
            future.result()

    assert store.grep("concurrent", conversation_id=cid)


def test_engine_supports_concurrent_tool_calls(tmp_path):
    engine = LosslessContextEngine(db_path=tmp_path / "lcm.db", context_length=200)
    engine.on_session_start("concurrent-engine")
    engine.store.ingest_messages(
        engine.conversation_id or 0,
        [{"role": "user", "content": f"threaded tool message {i}"} for i in range(10)],
    )

    def call_tool():
        result = json.loads(engine.handle_tool_call("lcm_grep", {"pattern": "threaded tool"}))
        assert "error" not in result
        assert result["results"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(call_tool) for _ in range(30)]
        for future in futures:
            future.result()


def test_passive_tool_engine_singleton_persists(monkeypatch, tmp_path):
    import lossless_context.plugin as plugin

    from lossless_context import engine as engine_module

    plugin.get_engine().close()
    monkeypatch.setattr(plugin, "_tool_engine", None)
    monkeypatch.setattr(engine_module, "default_state_dir", lambda: tmp_path)
    monkeypatch.setenv("HERMES_LCM_DB", str(tmp_path / "lossless-context.db"))

    first = plugin.get_engine()
    second = plugin.get_engine()

    assert plugin._tool_engine is first
    assert second is first


def test_lcm_describe_nonexistent_returns_error_json(tmp_path):
    engine = LosslessContextEngine(db_path=tmp_path / "lcm.db")
    engine.on_session_start("missing-summary")

    result = json.loads(engine.handle_tool_call("lcm_describe", {"id": "sum_missing"}))

    assert result == {"error": "summary not found: sum_missing"}


def test_lcm_status_returns_expected_keys(tmp_path):
    engine = LosslessContextEngine(db_path=tmp_path / "lcm.db", context_length=1000)
    engine.on_session_start("status")

    status = json.loads(engine.handle_tool_call("lcm_status", {}))

    assert status["engine"] == "lossless"
    assert status["db_path"].endswith("lcm.db")
    assert status["threshold_tokens"] == 750
    assert status["context_length"] == 1000
    assert status["compression_count"] == 0
    assert status["integrity"]["ok"] is True
    assert isinstance(status["integrity"]["issues"], list)


def test_engine_should_compress_preflight_respects_threshold(tmp_path):
    engine = LosslessContextEngine(db_path=tmp_path / "lcm.db", context_length=100)

    assert engine.should_compress_preflight([{"role": "user", "content": "tiny"}]) is False
    assert engine.should_compress_preflight([{"role": "user", "content": "word " * 400}]) is True


def test_leaf_summary_descendant_count_matches_source_links(tmp_path):
    store = LosslessStore(tmp_path / "lcm.db")
    cid = store.get_or_create_conversation("descendants")
    store.ingest_messages(cid, [{"role": "user", "content": f"source {i}"} for i in range(5)])
    chunk = store.unsummarized_messages(cid, protect_last_n=0, limit_tokens=1000)

    sid = store.create_leaf_summary(cid, chunk, "descendant summary")
    row = store.conn.execute(
        "SELECT descendant_count, (SELECT COUNT(*) FROM message_summaries WHERE summary_id=?) AS link_count "
        "FROM summaries WHERE summary_id=?",
        (sid, sid),
    ).fetchone()

    assert row["descendant_count"] == len(chunk)
    assert row["link_count"] == len(chunk)
    assert store.integrity_report()["ok"] is True


def test_store_preserves_repeated_identical_messages(tmp_path):
    store = LosslessStore(tmp_path / "lcm.db")
    cid = store.get_or_create_conversation("repeat")
    inserted = store.ingest_messages(cid, [
        {"role": "user", "content": "same"},
        {"role": "user", "content": "same"},
    ])
    assert inserted == 2
    rows = store.conn.execute("SELECT seq, role, content_text FROM messages WHERE conversation_id=? ORDER BY seq", (cid,)).fetchall()
    assert len(rows) == 2
    assert rows[0]["seq"] != rows[1]["seq"]
    assert rows[0]["content_text"] == rows[1]["content_text"]


def test_store_replay_is_idempotent(tmp_path):
    store = LosslessStore(tmp_path / "lcm.db")
    cid = store.get_or_create_conversation("replay")
    transcript = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    assert store.ingest_messages(cid, transcript) == 2
    assert store.ingest_messages(cid, transcript) == 0
    rows = store.conn.execute("SELECT seq, role, content_text FROM messages WHERE conversation_id=? ORDER BY seq", (cid,)).fetchall()
    assert len(rows) == 2
    assert [r["seq"] for r in rows] == [1, 2]


def test_schema_migration_v2_adds_metadata_and_removes_identity_uniqueness(tmp_path):
    import sqlite3, time
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
    CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
    INSERT INTO meta(key,value) VALUES('schema_version','2');
    CREATE TABLE conversations(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL UNIQUE, session_key TEXT NOT NULL, title TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL);
    CREATE TABLE messages(id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id INTEGER NOT NULL, seq INTEGER NOT NULL, role TEXT NOT NULL, content_json TEXT, content_text TEXT NOT NULL, token_count INTEGER NOT NULL, identity_hash TEXT NOT NULL, created_at REAL NOT NULL, UNIQUE(conversation_id, seq), UNIQUE(conversation_id, identity_hash));
    CREATE TABLE summaries(summary_id TEXT PRIMARY KEY, conversation_id INTEGER NOT NULL, kind TEXT NOT NULL, depth INTEGER NOT NULL, content TEXT NOT NULL, token_count INTEGER NOT NULL, earliest_seq INTEGER, latest_seq INTEGER, descendant_count INTEGER NOT NULL DEFAULT 0, source_message_token_count INTEGER NOT NULL DEFAULT 0, model TEXT, created_at REAL NOT NULL);
    CREATE TABLE message_summaries(message_id INTEGER NOT NULL, summary_id TEXT NOT NULL, PRIMARY KEY(message_id, summary_id));
    CREATE TABLE summary_edges(parent_summary_id TEXT NOT NULL, child_summary_id TEXT NOT NULL, PRIMARY KEY(parent_summary_id, child_summary_id));
    CREATE TABLE context_items(conversation_id INTEGER NOT NULL, ordinal INTEGER NOT NULL, item_type TEXT NOT NULL, message_id INTEGER, summary_id TEXT, created_at REAL NOT NULL, PRIMARY KEY(conversation_id, ordinal));
    """)
    now = time.time()
    conn.execute("INSERT INTO conversations(id, session_id, session_key, created_at, updated_at) VALUES(1,'s','s',?,?)", (now, now))
    conn.execute("INSERT INTO messages(conversation_id, seq, role, content_json, content_text, token_count, identity_hash, created_at) VALUES(1,1,'user','\"same\"','same',1,'oldhash',?)", (now,))
    conn.commit(); conn.close()

    store = LosslessStore(db)
    assert store.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
    cols = {r[1] for r in store.conn.execute("PRAGMA table_info(messages)").fetchall()}
    assert {"source_type", "source_identifier"} <= cols
    assert store.maintenance_stats()["schema_version"] == 3
    cid = store.get_or_create_conversation("s")
    assert store.ingest_messages(cid, [{"role": "user", "content": "same"}, {"role": "user", "content": "same"}]) >= 1


def test_parent_summary_dag_expands_to_leaf_messages(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_LCM_PARENT_SUMMARY_FANOUT", "2")
    engine = LosslessContextEngine(db_path=tmp_path / "lcm.db", context_length=200)
    engine.on_session_start("dag")
    cid = engine.conversation_id or 0
    for batch in range(2):
        msgs = [{"role": "user", "content": f"message {batch}-{i} " * 20} for i in range(5)]
        engine.store.ingest_messages(cid, msgs)
        chunk = engine.store.unsummarized_messages(cid, protect_last_n=0, limit_tokens=10000)
        sid = engine.store.create_leaf_summary(cid, chunk, f"leaf {batch}")
        assert sid
    engine._maybe_create_parent_summary(cid)
    parents = engine.store.conn.execute("SELECT summary_id FROM summaries WHERE kind='parent'").fetchall()
    assert parents
    sources = engine.store.source_messages_for_summary(parents[0]["summary_id"])
    assert len(sources) >= 5


def test_budget_aware_assembly_records_budget_and_limits_summaries(tmp_path):
    engine = LosslessContextEngine(db_path=tmp_path / "lcm.db", context_length=80)
    engine.on_session_start("budget")
    cid = engine.conversation_id or 0
    engine.store.ingest_messages(cid, [{"role": "user", "content": "x " * 100}])
    msg = engine.store.unsummarized_messages(cid, protect_last_n=0, limit_tokens=10000)
    for i in range(3):
        engine.store.create_leaf_summary(cid, msg, "summary " + ("y " * 200) + str(i))
    assembled = engine._assemble_messages(cid, [{"role": "system", "content": "s"}, {"role": "user", "content": "tail"}])
    assert assembled[0]["role"] == "system"
    assert "budget_tokens" in engine.last_assembly
    assert engine.last_assembly["summary_count"] <= 3


def test_on_turn_end_and_session_end_same_transcript_do_not_duplicate(tmp_path):
    engine = LosslessContextEngine(db_path=tmp_path / "lcm.db")
    engine.on_session_start("hooks")
    msgs = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    engine.on_turn_end(msgs)
    engine.on_session_end("hooks", msgs)
    rows = engine.store.conn.execute("SELECT * FROM messages WHERE conversation_id=?", (engine.conversation_id,)).fetchall()
    assert len(rows) == 2


def test_status_includes_maintenance_and_provenance_metadata(tmp_path):
    store = LosslessStore(tmp_path / "lcm.db")
    cid = store.get_or_create_conversation("meta")
    store.ingest_messages(cid, [{"role": "tool", "content": "result", "source_type": "tool", "source_identifier": "abc"}])
    row = store.conn.execute("SELECT source_type, source_identifier FROM messages WHERE conversation_id=?", (cid,)).fetchone()
    assert row["source_type"] == "tool"
    assert row["source_identifier"] == "abc"
    report = store.integrity_report()
    assert report["maintenance"]["schema_version"] == 3
    assert "wal_bytes" in report["maintenance"]


def test_optional_llm_summarizer_uses_client_when_supplied():
    from lossless_context.summarizers import OptionalLLMSummarizer
    def client(messages, focus_topic=None):
        return "client summary " + str(focus_topic)
    summarizer = OptionalLLMSummarizer(client=client, model_name="client-model")
    assert summarizer.model_name == "client-model"
    assert summarizer.summarize([], focus_topic="focus") == "client summary focus"


def test_lcm_expand_returns_exact_content_json_for_truncated_text(tmp_path):
    engine = LosslessContextEngine(db_path=tmp_path / "lcm.db")
    engine.on_session_start("exact")
    long_text = "x" * 210_000
    cid = engine.conversation_id or 0
    engine.store.ingest_messages(cid, [{"role": "user", "content": long_text}])
    chunk = engine.store.unsummarized_messages(cid, protect_last_n=0, limit_tokens=1_000_000)
    sid = engine.store.create_leaf_summary(cid, chunk, "summary")
    payload = json.loads(engine.handle_tool_call("lcm_expand", {"id": sid}, messages=[]))
    assert payload["messages"][0]["content_json"] == json.dumps(long_text, ensure_ascii=False, sort_keys=True, default=str)
    assert payload["messages"][0]["content_text"].endswith("…[truncated]")


def test_partial_transcript_replay_does_not_append_duplicates(tmp_path):
    store = LosslessStore(tmp_path / "lcm.db")
    cid = store.get_or_create_conversation("partial")
    full = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    assert store.ingest_messages(cid, full) == 2
    assert store.ingest_messages(cid, [{"role": "assistant", "content": "b"}]) == 0
    assert store.ingest_messages(cid, [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]) == 0
    rows = store.conn.execute("SELECT * FROM messages WHERE conversation_id=?", (cid,)).fetchall()
    assert len(rows) == 2


def test_protected_head_tail_do_not_overlap_for_short_transcripts(tmp_path):
    engine = LosslessContextEngine(db_path=tmp_path / "lcm.db")
    original = [{"role": "user", "content": str(i)} for i in range(4)]
    out = engine._assemble_messages(0, original)
    assert [m["content"] for m in out] == ["0", "1", "2", "3"]


def test_parent_summary_creation_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_LCM_PARENT_SUMMARY_FANOUT", "2")
    engine = LosslessContextEngine(db_path=tmp_path / "lcm.db", context_length=200)
    engine.on_session_start("parent-idempotent")
    cid = engine.conversation_id or 0
    for batch in range(2):
        engine.store.ingest_messages(cid, [{"role": "user", "content": f"batch {batch} msg {i}"} for i in range(4)])
        chunk = engine.store.unsummarized_messages(cid, protect_last_n=0, limit_tokens=1000)
        engine.store.create_leaf_summary(cid, chunk, f"leaf {batch}")
    engine._maybe_create_parent_summary(cid)
    engine._maybe_create_parent_summary(cid)
    assert engine.store.conn.execute("SELECT COUNT(*) FROM summaries WHERE kind='parent'").fetchone()[0] == 1


def test_source_messages_for_summary_handles_deep_dag(tmp_path):
    store = LosslessStore(tmp_path / "lcm.db")
    cid = store.get_or_create_conversation("deep")
    store.ingest_messages(cid, [{"role": "user", "content": f"m{i}"} for i in range(8)])
    leaves = []
    for msg in store.unsummarized_messages(cid, protect_last_n=0, limit_tokens=1000):
        leaves.append(store.get_summary(store.create_leaf_summary(cid, [msg], f"leaf {msg.seq}")))
    level = [x for x in leaves if x is not None]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            group = level[i:i+2]
            if len(group) == 1:
                nxt.append(group[0])
            else:
                nxt.append(store.get_summary(store.create_parent_summary(cid, group, "parent")))
        level = [x for x in nxt if x is not None]
    sources = store.source_messages_for_summary(level[0].summary_id)
    assert [m.seq for m in sources] == list(range(1, 9))


def test_budget_zero_selects_no_summaries_explicitly(tmp_path):
    engine = LosslessContextEngine(db_path=tmp_path / "lcm.db", context_length=0)
    engine.threshold_tokens = 0
    engine.on_session_start("zero-budget")
    cid = engine.conversation_id or 0
    engine.store.ingest_messages(cid, [{"role": "user", "content": "hello"}])
    msg = engine.store.unsummarized_messages(cid, protect_last_n=0, limit_tokens=1000)
    engine.store.create_leaf_summary(cid, msg, "summary")
    out = engine._assemble_messages(cid, [{"role": "user", "content": "tail"}])
    assert not any("lossless_context_summary" in m.get("content", "") for m in out)
    assert engine.last_assembly["budget_tokens"] == 0


def test_lcm_describe_returns_summary_metadata(tmp_path):
    engine = LosslessContextEngine(db_path=tmp_path / "lcm.db")
    engine.on_session_start("describe")
    cid = engine.conversation_id or 0
    engine.store.ingest_messages(cid, [{"role": "user", "content": "describe me"}])
    msg = engine.store.unsummarized_messages(cid, protect_last_n=0, limit_tokens=1000)
    sid = engine.store.create_leaf_summary(cid, msg, "summary body")
    payload = json.loads(engine.handle_tool_call("lcm_describe", {"id": sid}, messages=[]))
    assert payload["summary"]["summary_id"] == sid
    assert payload["sourceMessageCount"] == 1
    assert payload["sourceSeqs"] == [1]


def test_public_release_metadata_versions_are_consistent():
    from pathlib import Path
    import tomllib
    import yaml
    import lossless_context

    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    package_version = pyproject["project"]["version"]
    plugin_manifest = yaml.safe_load((root / "plugin_entry" / "plugin.yaml").read_text(encoding="utf-8"))
    skill_text = (root / "skills" / "lossless-context" / "SKILL.md").read_text(encoding="utf-8")

    assert lossless_context.__version__ == package_version
    assert plugin_manifest["version"] == package_version
    assert f"version: {package_version}" in skill_text
