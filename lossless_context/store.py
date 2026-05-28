from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import content_to_text, escape_like_pattern, estimate_tokens, safe_snippet, sanitize_fts_query, stable_hash

SCHEMA_VERSION = 3


@dataclass
class MessageRecord:
    id: int
    conversation_id: int
    seq: int
    role: str
    content_json: str | None
    content_text: str
    token_count: int
    created_at: float
    source_type: str | None = None
    source_identifier: str | None = None


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
    descendant_count: int = 0
    source_message_token_count: int = 0


class LosslessStore:
    """SQLite store for source-preserving context summaries.

    The store is append-friendly and never deletes source messages during automatic
    operation. It can be used by an active ContextEngine or by passive tools.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.RLock()
        self._init_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        """Return a SQLite connection owned by the current thread.

        Hermes gateway runs blocking agent work in a thread pool. A context
        engine instance can therefore be created in one worker and used in
        another. Python's sqlite3 connections are thread-affine by default, so
        the store must not keep a single process-wide connection object.
        """
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path), timeout=10.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def _init_schema(self) -> None:
        with self._write_lock:
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
            source_type TEXT,
            source_identifier TEXT,
            created_at REAL NOT NULL,
            UNIQUE(conversation_id, seq)
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
            self._migrate_schema(cur)
            try:
                cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(content_text, content='messages', content_rowid='id')")
                cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS summaries_fts USING fts5(content, content='summaries', content_rowid='rowid')")
                self.conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('fts5','1')")
                try:
                    cur.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
                    cur.execute("INSERT INTO summaries_fts(summaries_fts) VALUES('rebuild')")
                except sqlite3.DatabaseError:
                    pass
            except sqlite3.DatabaseError:
                self.conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('fts5','0')")
            self.conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)", (str(SCHEMA_VERSION),))
            self.conn.commit()

    def _migrate_schema(self, cur: sqlite3.Cursor) -> None:
        row = cur.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        try:
            version = int(row[0]) if row else 0
        except Exception:
            version = 0
        columns = {r[1] for r in cur.execute("PRAGMA table_info(messages)").fetchall()}
        if "source_type" not in columns:
            cur.execute("ALTER TABLE messages ADD COLUMN source_type TEXT")
        if "source_identifier" not in columns:
            cur.execute("ALTER TABLE messages ADD COLUMN source_identifier TEXT")
        # v3 removes the old content-only uniqueness by rebuilding messages when needed.
        indexes = cur.execute("PRAGMA index_list(messages)").fetchall()
        for idx in indexes:
            name = idx[1]
            if not idx[2]:
                continue
            cols = [r[2] for r in cur.execute(f"PRAGMA index_info({name})").fetchall()]
            if cols == ["conversation_id", "identity_hash"]:
                before_count = int(cur.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
                cur.execute("PRAGMA foreign_keys=OFF")
                cur.executescript("""
                CREATE TABLE IF NOT EXISTS messages_new(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    seq INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content_json TEXT,
                    content_text TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    identity_hash TEXT NOT NULL,
                    source_type TEXT,
                    source_identifier TEXT,
                    created_at REAL NOT NULL,
                    UNIQUE(conversation_id, seq)
                );
                INSERT OR IGNORE INTO messages_new(id, conversation_id, seq, role, content_json, content_text, token_count, identity_hash, source_type, source_identifier, created_at)
                SELECT id, conversation_id, seq, role, content_json, content_text, token_count, identity_hash, source_type, source_identifier, created_at FROM messages ORDER BY conversation_id, seq, id;
                DROP TABLE messages;
                ALTER TABLE messages_new RENAME TO messages;
                CREATE INDEX IF NOT EXISTS idx_messages_conv_seq ON messages(conversation_id, seq);
                """)
                after_count = int(cur.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
                if after_count != before_count:
                    cur.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('migration_warning',?)", (f"messages_row_count_changed:{before_count}->{after_count}",))
                cur.execute("PRAGMA foreign_keys=ON")
                break
        if version < SCHEMA_VERSION:
            cur.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('last_migration',?)", (f"v{version}_to_v{SCHEMA_VERSION}",))

    def fts_enabled(self) -> bool:
        row = self.conn.execute("SELECT value FROM meta WHERE key='fts5'").fetchone()
        return bool(row and row[0] == "1")

    def get_or_create_conversation(
        self,
        session_id: str,
        session_key: str | None = None,
        title: str | None = None,
        conversation_id: int | None = None,
    ) -> int:
        with self._write_lock:
            now = time.time()
            session_key = session_key or session_id or "default"
            if conversation_id is not None:
                row = self.conn.execute("SELECT id FROM conversations WHERE id=?", (conversation_id,)).fetchone()
                if row:
                    self.conn.execute("UPDATE conversations SET updated_at=?, title=COALESCE(?, title) WHERE id=?", (now, title, row[0]))
                    self.conn.commit()
                    return int(row[0])
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

    def get_or_create_conversation_for_session_key(
        self,
        session_id: str,
        session_key: str,
        title: str | None = None,
    ) -> int:
        """Bind a host session to the stable logical conversation key."""
        with self._write_lock:
            now = time.time()
            row = self.conn.execute(
                "SELECT id FROM conversations WHERE session_key=? ORDER BY created_at ASC LIMIT 1",
                (session_key,),
            ).fetchone()
            if row:
                self.conn.execute("UPDATE conversations SET updated_at=?, title=COALESCE(?, title) WHERE id=?", (now, title, row[0]))
                self.conn.commit()
                return int(row[0])
            return self.get_or_create_conversation(session_id, session_key=session_key, title=title)

    def conversation_id_for_session(self, session_id: str) -> int | None:
        row = self.conn.execute("SELECT id FROM conversations WHERE session_id=?", (session_id,)).fetchone()
        return int(row[0]) if row else None

    def messages_for_conversation(self, conversation_id: int, *, limit: int | None = None) -> list[dict[str, Any]]:
        sql = "SELECT role, content_json, source_type, source_identifier FROM messages WHERE conversation_id=? ORDER BY seq ASC"
        params: tuple[Any, ...] = (conversation_id,)
        if limit is not None and limit > 0:
            sql += " LIMIT ?"
            params = (conversation_id, int(limit))
        rows = self.conn.execute(sql, params).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            try:
                content = json.loads(row["content_json"]) if row["content_json"] is not None else None
            except Exception:
                content = None
            msg: dict[str, Any] = {"role": row["role"], "content": content}
            if row["source_type"]:
                msg["source_type"] = row["source_type"]
            if row["source_identifier"]:
                msg["source_identifier"] = row["source_identifier"]
            out.append(msg)
        return out

    def ingest_messages(self, conversation_id: int, messages: list[dict[str, Any]]) -> int:
        """Idempotently ingest OpenAI-format messages. Returns inserted count.

        Hermes usually passes full transcripts repeatedly. We preserve repeated
        identical messages by counting occurrences of (role, content hash), while
        avoiding duplicate replay of a transcript or partial transcript already
        present in the store.
        """
        with self._write_lock:
            inserted = 0
            seen_in_batch: dict[tuple[str, str], int] = {}
            existing_counts: dict[tuple[str, str], int] = {}
            for r in self.conn.execute("SELECT role, identity_hash, COUNT(*) c FROM messages WHERE conversation_id=? GROUP BY role, identity_hash", (conversation_id,)).fetchall():
                existing_counts[(r["role"], r["identity_hash"])] = int(r["c"])
            for idx, msg in enumerate(messages):
                role = str(msg.get("role") or "unknown")[:40]
                content_json = json.dumps(msg.get("content"), ensure_ascii=False, sort_keys=True, default=str)
                text = content_to_text(msg.get("content"), max_chars=200_000)
                identity = stable_hash(role, content_json)
                key = (role, identity)
                seen_in_batch[key] = seen_in_batch.get(key, 0) + 1
                if existing_counts.get(key, 0) >= seen_in_batch[key]:
                    continue
                tokens = estimate_tokens(msg.get("content"))
                seq = idx + 1
                existing_at_seq = self.conn.execute(
                    "SELECT role, identity_hash FROM messages WHERE conversation_id=? AND seq=?",
                    (conversation_id, seq),
                ).fetchone()
                if existing_at_seq is not None:
                    seq = self._next_seq(conversation_id)
                source_type = msg.get("source_type") or msg.get("sourceType")
                source_identifier = msg.get("source_identifier") or msg.get("sourceIdentifier") or msg.get("tool_call_id")
                cur = self.conn.execute(
                    "INSERT INTO messages(conversation_id, seq, role, content_json, content_text, token_count, identity_hash, source_type, source_identifier, created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (conversation_id, seq, role, content_json, text, tokens, identity, source_type, source_identifier, time.time()),
                )
                mid = cur.lastrowid
                if mid is not None:
                    self._insert_message_fts(int(mid), text)
                self._append_context_item(conversation_id, "message", int(mid) if mid is not None else None, None)
                existing_counts[key] = existing_counts.get(key, 0) + 1
                inserted += 1
            if inserted:
                self._maybe_checkpoint(inserted)
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
            out.append(MessageRecord(int(r["id"]), int(r["conversation_id"]), int(r["seq"]), r["role"], r["content_json"], r["content_text"], int(r["token_count"]), float(r["created_at"]), r["source_type"], r["source_identifier"]))
            total += int(r["token_count"])
        return out

    def create_leaf_summary(self, conversation_id: int, messages: list[MessageRecord], content: str, model: str | None = None) -> str:
        if not messages:
            raise ValueError("cannot summarize empty message list")
        with self._write_lock:
            sid = "sum_" + stable_hash(conversation_id, messages[0].seq, messages[-1].seq, content)[:32]
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

    def create_parent_summary(self, conversation_id: int, children: list[SummaryRecord], content: str, model: str | None = None) -> str:
        if not children:
            raise ValueError("cannot summarize empty child summary list")
        with self._write_lock:
            depth = max(c.depth for c in children) + 1
            earliest = min((c.earliest_seq for c in children if c.earliest_seq is not None), default=None)
            latest = max((c.latest_seq for c in children if c.latest_seq is not None), default=None)
            descendant_count = sum(max(1, c.descendant_count) for c in children)
            source_tokens = sum(c.source_message_token_count for c in children)
            sid = "sum_" + stable_hash(conversation_id, depth, earliest, latest, content, [c.summary_id for c in children])[:32]
            token_count = estimate_tokens(content)
            self.conn.execute(
                """INSERT OR IGNORE INTO summaries(summary_id, conversation_id, kind, depth, content, token_count, earliest_seq, latest_seq, descendant_count, source_message_token_count, model, created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (sid, conversation_id, "parent", depth, content, token_count, earliest, latest, descendant_count, source_tokens, model, time.time()),
            )
            rowid = self.conn.execute("SELECT rowid FROM summaries WHERE summary_id=?", (sid,)).fetchone()[0]
            self._insert_summary_fts(int(rowid), content)
            for child in children:
                self.conn.execute("INSERT OR IGNORE INTO summary_edges(parent_summary_id, child_summary_id) VALUES(?,?)", (sid, child.summary_id))
            self._append_context_item(conversation_id, "summary", None, sid)
            self.conn.commit()
            return sid

    def child_summaries_for_summary(self, summary_id: str) -> list[SummaryRecord]:
        rows = self.conn.execute("""SELECT s.* FROM summaries s JOIN summary_edges e ON e.child_summary_id=s.summary_id WHERE e.parent_summary_id=? ORDER BY s.earliest_seq, s.created_at""", (summary_id,)).fetchall()
        summaries = []
        for r in rows:
            summary = self.get_summary(r["summary_id"])
            if summary is not None:
                summaries.append(summary)
        return summaries

    def get_summary(self, summary_id: str) -> SummaryRecord | None:
        r = self.conn.execute("SELECT * FROM summaries WHERE summary_id=?", (summary_id,)).fetchone()
        if not r:
            return None
        return SummaryRecord(
            summary_id=r["summary_id"],
            conversation_id=int(r["conversation_id"]),
            kind=r["kind"],
            depth=int(r["depth"]),
            content=r["content"],
            token_count=int(r["token_count"]),
            earliest_seq=r["earliest_seq"],
            latest_seq=r["latest_seq"],
            created_at=float(r["created_at"]),
            descendant_count=int(r["descendant_count"]),
            source_message_token_count=int(r["source_message_token_count"]),
        )

    def source_messages_for_summary(self, summary_id: str) -> list[MessageRecord]:
        seen: set[str] = set()
        message_ids: set[int] = set()

        def collect(sid: str) -> None:
            if sid in seen:
                return
            seen.add(sid)
            for row in self.conn.execute("SELECT message_id FROM message_summaries WHERE summary_id=?", (sid,)).fetchall():
                message_ids.add(int(row["message_id"]))
            for row in self.conn.execute("SELECT child_summary_id FROM summary_edges WHERE parent_summary_id=?", (sid,)).fetchall():
                collect(str(row["child_summary_id"]))

        collect(summary_id)
        if not message_ids:
            return []
        placeholders = ",".join("?" for _ in message_ids)
        rows = self.conn.execute(f"SELECT * FROM messages WHERE id IN ({placeholders}) ORDER BY seq", tuple(sorted(message_ids))).fetchall()
        return [MessageRecord(int(r["id"]), int(r["conversation_id"]), int(r["seq"]), r["role"], r["content_json"], r["content_text"], int(r["token_count"]), float(r["created_at"]), r["source_type"], r["source_identifier"]) for r in rows]

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

    def _maybe_checkpoint(self, writes: int = 1) -> None:
        count = getattr(self._local, "write_count", 0) + writes
        self._local.write_count = count
        if count >= 50:
            try:
                self.conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except sqlite3.DatabaseError:
                pass
            self._local.write_count = 0

    def maintenance_stats(self) -> dict[str, Any]:
        wal = self.db_path.with_name(self.db_path.name + "-wal")
        return {
            "db_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
            "wal_bytes": wal.stat().st_size if wal.exists() else 0,
            "schema_version": int((self.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone() or [0])[0]),
            "last_migration": (self.conn.execute("SELECT value FROM meta WHERE key='last_migration'").fetchone() or [None])[0],
            "migration_warning": (self.conn.execute("SELECT value FROM meta WHERE key='migration_warning'").fetchone() or [None])[0],
        }

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
        return {"ok": not issues, "issues": issues, "maintenance": self.maintenance_stats()}
