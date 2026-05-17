from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import content_to_text, escape_like_pattern, estimate_tokens, safe_snippet, sanitize_fts_query, stable_hash

SCHEMA_VERSION = 1


@dataclass
class MessageRecord:
    id: int
    conversation_id: int
    seq: int
    role: str
    content_text: str
    token_count: int
    created_at: float


@dataclass
class SummaryRecord:
    summary_id: str
    conversation_id: int
    kind: str
    depth: int
    content: str
    token_count: int
    earliest_seq: int | None
    latest_seq: int | None
    created_at: float


class LosslessStore:
    """SQLite store for source-preserving context summaries.

    The store is append-friendly and never deletes source messages during automatic
    operation. It can be used by an active ContextEngine or by passive tools.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS conversations(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL UNIQUE,
            session_key TEXT NOT NULL,
            title TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            seq INTEGER NOT NULL,
            role TEXT NOT NULL,
            content_json TEXT,
            content_text TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            identity_hash TEXT NOT NULL,
            created_at REAL NOT NULL,
            UNIQUE(conversation_id, seq),
            UNIQUE(conversation_id, identity_hash)
        );
        CREATE TABLE IF NOT EXISTS summaries(
            summary_id TEXT PRIMARY KEY,
            conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            kind TEXT NOT NULL,
            depth INTEGER NOT NULL,
            content TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            earliest_seq INTEGER,
            latest_seq INTEGER,
            descendant_count INTEGER NOT NULL DEFAULT 0,
            source_message_token_count INTEGER NOT NULL DEFAULT 0,
            model TEXT,
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS message_summaries(
            message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            summary_id TEXT NOT NULL REFERENCES summaries(summary_id) ON DELETE CASCADE,
            PRIMARY KEY(message_id, summary_id)
        );
        CREATE TABLE IF NOT EXISTS summary_edges(
            parent_summary_id TEXT NOT NULL REFERENCES summaries(summary_id) ON DELETE CASCADE,
            child_summary_id TEXT NOT NULL REFERENCES summaries(summary_id) ON DELETE CASCADE,
            PRIMARY KEY(parent_summary_id, child_summary_id)
        );
        CREATE TABLE IF NOT EXISTS context_items(
            conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL,
            item_type TEXT NOT NULL CHECK(item_type IN ('message','summary')),
            message_id INTEGER REFERENCES messages(id) ON DELETE CASCADE,
            summary_id TEXT REFERENCES summaries(summary_id) ON DELETE CASCADE,
            created_at REAL NOT NULL,
            PRIMARY KEY(conversation_id, ordinal)
        );
        CREATE INDEX IF NOT EXISTS idx_messages_conv_seq ON messages(conversation_id, seq);
        CREATE INDEX IF NOT EXISTS idx_summaries_conv_depth ON summaries(conversation_id, depth, created_at);
        """
        )
        try:
            cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(content_text, content='messages', content_rowid='id')")
            cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS summaries_fts USING fts5(content, content='summaries', content_rowid='rowid')")
            self.conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('fts5','1')")
        except sqlite3.DatabaseError:
            self.conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('fts5','0')")
        self.conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)", (str(SCHEMA_VERSION),))
        self.conn.commit()

    def fts_enabled(self) -> bool:
        row = self.conn.execute("SELECT value FROM meta WHERE key='fts5'").fetchone()
        return bool(row and row[0] == "1")

    def get_or_create_conversation(self, session_id: str, session_key: str | None = None, title: str | None = None) -> int:
        now = time.time()
        session_key = session_key or session_id or "default"
        row = self.conn.execute("SELECT id FROM conversations WHERE session_id=?", (session_id,)).fetchone()
        if row:
            self.conn.execute("UPDATE conversations SET updated_at=?, title=COALESCE(?, title) WHERE id=?", (now, title, row[0]))
            self.conn.commit()
            return int(row[0])
        cur = self.conn.execute(
            "INSERT INTO conversations(session_id, session_key, title, created_at, updated_at) VALUES(?,?,?,?,?)",
            (session_id, session_key, title, now, now),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def ingest_messages(self, conversation_id: int, messages: list[dict[str, Any]]) -> int:
        """Idempotently ingest OpenAI-format messages. Returns inserted count."""
        next_seq = self._next_seq(conversation_id)
        inserted = 0
        for idx, msg in enumerate(messages):
            role = str(msg.get("role") or "unknown")[:40]
            content_json = json.dumps(msg.get("content"), ensure_ascii=False, sort_keys=True, default=str)
            text = content_to_text(msg.get("content"), max_chars=200_000)
            identity = stable_hash(role, content_json)
            tokens = estimate_tokens(msg.get("content"))
            try:
                cur = self.conn.execute(
                    "INSERT INTO messages(conversation_id, seq, role, content_json, content_text, token_count, identity_hash, created_at) VALUES(?,?,?,?,?,?,?,?)",
                    (conversation_id, next_seq + idx, role, content_json, text, tokens, identity, time.time()),
                )
                mid = int(cur.lastrowid)
                self._insert_message_fts(mid, text)
                self._append_context_item(conversation_id, "message", mid, None)
                inserted += 1
            except sqlite3.IntegrityError:
                continue
        self.conn.commit()
        return inserted

    def _insert_message_fts(self, rowid: int, text: str) -> None:
        if self.fts_enabled():
            try:
                self.conn.execute("INSERT OR REPLACE INTO messages_fts(rowid, content_text) VALUES(?,?)", (rowid, text))
            except sqlite3.DatabaseError:
                pass

    def _insert_summary_fts(self, rowid: int, text: str) -> None:
        if self.fts_enabled():
            try:
                self.conn.execute("INSERT OR REPLACE INTO summaries_fts(rowid, content) VALUES(?,?)", (rowid, text))
            except sqlite3.DatabaseError:
                pass

    def _next_seq(self, conversation_id: int) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(seq),0)+1 FROM messages WHERE conversation_id=?", (conversation_id,)).fetchone()
        return int(row[0])

    def _next_ordinal(self, conversation_id: int) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(ordinal),0)+1 FROM context_items WHERE conversation_id=?", (conversation_id,)).fetchone()
        return int(row[0])

    def _append_context_item(self, conversation_id: int, item_type: str, message_id: int | None, summary_id: str | None) -> None:
        self.conn.execute(
            "INSERT INTO context_items(conversation_id, ordinal, item_type, message_id, summary_id, created_at) VALUES(?,?,?,?,?,?)",
            (conversation_id, self._next_ordinal(conversation_id), item_type, message_id, summary_id, time.time()),
        )

    def unsummarized_messages(self, conversation_id: int, *, protect_last_n: int, limit_tokens: int) -> list[MessageRecord]:
        rows = self.conn.execute(
            """
            SELECT * FROM messages m
            WHERE conversation_id=?
              AND seq <= (SELECT COALESCE(MAX(seq),0) FROM messages WHERE conversation_id=?) - ?
              AND NOT EXISTS(SELECT 1 FROM message_summaries ms WHERE ms.message_id=m.id)
            ORDER BY seq ASC
        """,
            (conversation_id, conversation_id, protect_last_n),
        ).fetchall()
        out: list[MessageRecord] = []
        total = 0
        for r in rows:
            if out and total + int(r["token_count"]) > limit_tokens:
                break
            out.append(MessageRecord(int(r["id"]), int(r["conversation_id"]), int(r["seq"]), r["role"], r["content_text"], int(r["token_count"]), float(r["created_at"])))
            total += int(r["token_count"])
        return out

    def create_leaf_summary(self, conversation_id: int, messages: list[MessageRecord], content: str, model: str | None = None) -> str:
        if not messages:
            raise ValueError("cannot summarize empty message list")
        sid = "sum_" + stable_hash(conversation_id, messages[0].seq, messages[-1].seq, content)[:16]
        token_count = estimate_tokens(content)
        source_tokens = sum(m.token_count for m in messages)
        self.conn.execute(
            """INSERT OR IGNORE INTO summaries(summary_id, conversation_id, kind, depth, content, token_count, earliest_seq, latest_seq, descendant_count, source_message_token_count, model, created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (sid, conversation_id, "leaf", 0, content, token_count, messages[0].seq, messages[-1].seq, len(messages), source_tokens, model, time.time()),
        )
        rowid = self.conn.execute("SELECT rowid FROM summaries WHERE summary_id=?", (sid,)).fetchone()[0]
        self._insert_summary_fts(int(rowid), content)
        for m in messages:
            self.conn.execute("INSERT OR IGNORE INTO message_summaries(message_id, summary_id) VALUES(?,?)", (m.id, sid))
        self._append_context_item(conversation_id, "summary", None, sid)
        self.conn.commit()
        return sid

    def get_summary(self, summary_id: str) -> SummaryRecord | None:
        r = self.conn.execute("SELECT * FROM summaries WHERE summary_id=?", (summary_id,)).fetchone()
        if not r:
            return None
        return SummaryRecord(r["summary_id"], int(r["conversation_id"]), r["kind"], int(r["depth"]), r["content"], int(r["token_count"]), r["earliest_seq"], r["latest_seq"], float(r["created_at"]))

    def source_messages_for_summary(self, summary_id: str) -> list[MessageRecord]:
        rows = self.conn.execute(
            """SELECT m.* FROM messages m JOIN message_summaries ms ON ms.message_id=m.id WHERE ms.summary_id=? ORDER BY m.seq""",
            (summary_id,),
        ).fetchall()
        return [MessageRecord(int(r["id"]), int(r["conversation_id"]), int(r["seq"]), r["role"], r["content_text"], int(r["token_count"]), float(r["created_at"])) for r in rows]

    def grep(self, pattern: str, *, mode: str = "full_text", scope: str = "both", conversation_id: int | None = None, limit: int = 20) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 20), 100))
        results: list[dict[str, Any]] = []
        params_conv = " AND conversation_id=?" if conversation_id else ""
        conv_args: tuple[Any, ...] = (conversation_id,) if conversation_id else ()
        if scope in {"messages", "both"}:
            if mode == "full_text" and self.fts_enabled():
                q = sanitize_fts_query(pattern)
                sql = f"SELECT m.*, bm25(messages_fts) rank FROM messages_fts JOIN messages m ON messages_fts.rowid=m.id WHERE messages_fts MATCH ?{params_conv} ORDER BY rank LIMIT ?"
                rows = self.conn.execute(sql, (q, *conv_args, limit)).fetchall()
            else:
                like = f"%{escape_like_pattern(pattern)}%"
                rows = self.conn.execute(f"SELECT * FROM messages WHERE content_text LIKE ? ESCAPE '\\'{params_conv} ORDER BY created_at DESC LIMIT ?", (like, *conv_args, limit)).fetchall()
            for r in rows:
                results.append({"type": "message", "id": int(r["id"]), "conversationId": int(r["conversation_id"]), "seq": int(r["seq"]), "role": r["role"], "snippet": safe_snippet(r["content_text"], pattern), "tokenCount": int(r["token_count"])})
        if scope in {"summaries", "both"}:
            if mode == "full_text" and self.fts_enabled():
                q = sanitize_fts_query(pattern)
                sql = f"SELECT s.*, bm25(summaries_fts) rank FROM summaries_fts JOIN summaries s ON summaries_fts.rowid=s.rowid WHERE summaries_fts MATCH ?{params_conv} ORDER BY rank LIMIT ?"
                rows = self.conn.execute(sql, (q, *conv_args, limit)).fetchall()
            else:
                like = f"%{escape_like_pattern(pattern)}%"
                rows = self.conn.execute(f"SELECT * FROM summaries WHERE content LIKE ? ESCAPE '\\'{params_conv} ORDER BY created_at DESC LIMIT ?", (like, *conv_args, limit)).fetchall()
            for r in rows:
                results.append({"type": "summary", "id": r["summary_id"], "summaryId": r["summary_id"], "conversationId": int(r["conversation_id"]), "kind": r["kind"], "depth": int(r["depth"]), "snippet": safe_snippet(r["content"], pattern), "tokenCount": int(r["token_count"])})
        return results

    def integrity_report(self) -> dict[str, Any]:
        issues = []
        orphan_edges = self.conn.execute("""SELECT e.* FROM summary_edges e LEFT JOIN summaries p ON p.summary_id=e.parent_summary_id LEFT JOIN summaries c ON c.summary_id=e.child_summary_id WHERE p.summary_id IS NULL OR c.summary_id IS NULL""").fetchall()
        if orphan_edges:
            issues.append({"kind": "orphan_summary_edges", "count": len(orphan_edges)})
        orphan_links = self.conn.execute("""SELECT ms.* FROM message_summaries ms LEFT JOIN messages m ON m.id=ms.message_id LEFT JOIN summaries s ON s.summary_id=ms.summary_id WHERE m.id IS NULL OR s.summary_id IS NULL""").fetchall()
        if orphan_links:
            issues.append({"kind": "orphan_message_summary_links", "count": len(orphan_links)})
        bad_ranges = self.conn.execute("SELECT summary_id FROM summaries WHERE earliest_seq IS NOT NULL AND latest_seq IS NOT NULL AND earliest_seq > latest_seq").fetchall()
        if bad_ranges:
            issues.append({"kind": "invalid_summary_ranges", "count": len(bad_ranges)})
        return {"ok": not issues, "issues": issues}
