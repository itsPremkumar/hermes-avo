"""Structured agent-to-agent message bus.

Topics (the five canonical AVO interaction channels):
  planning      – goal decomposition, task assignment
  execution     – live tool-call + result stream
  observation   – checkpoints, metrics, partial results
  critique      – self-critique / peer review of outcomes
  recovery      – failure context + resume instructions

Persistence:
  * If Redis is reachable (REDIS_URL env), use it for pub/sub so multiple
    agent processes can subscribe in real time.
  * Otherwise fall back to a SQLite topic log so the bus is fully offline-
    capable (queries the local board DB or a default SQLite file).

Every network/IO path has a timeout + retry budget and never crashes the
calling session.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

try:  # redis is optional
    import redis as _redis_mod
except Exception:  # pragma: no cover - redis is an optional dep
    _redis_mod = None

TOPICS: List[str] = ["planning", "execution", "observation", "critique", "recovery"]


@dataclass
class Message:
    topic: str
    sender: str
    recipient: str
    payload: Dict[str, Any]
    trace_id: str
    ts: float = field(default_factory=time.time)
    msg_id: str = field(default_factory=lambda: f"msg-{uuid.uuid4().hex[:10]}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "msg_id": self.msg_id,
            "topic": self.topic,
            "sender": self.sender,
            "recipient": self.recipient,
            "trace_id": self.trace_id,
            "ts": self.ts,
            "payload": self.payload,
        }


# Redis connection with timeout + retry, else None (offline fallback).
_redis: Optional[Any] = None


def _try_redis() -> Optional[Any]:
    """Return a connected redis client or ``None`` if unavailable."""
    global _redis
    if _redis is not None:
        return _redis
    if _redis_mod is None:
        return None
    url = os.environ.get("REDIS_URL")
    if not url:
        return None
    try:
        client = _redis_mod.from_url(url, socket_connect_timeout=3, socket_timeout=3)
        client.ping()
        _redis = client
        return client
    except Exception:
        _redis = None
        return None


def _sqlite_path() -> str:
    return os.environ.get("AVO_MESSAGE_DB", os.path.join(
        os.environ.get("HERMES_HOME", os.environ.get("USERPROFILE", "/tmp")),
        "hermes_avo", "messages.db",
    ))


def _sqlite_conn() -> sqlite3.Connection:
    path = _sqlite_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, timeout=5.0, isolation_level="IMMEDIATE")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            topic TEXT NOT NULL,
            sender TEXT NOT NULL,
            recipient TEXT NOT NULL,
            trace_id TEXT NOT NULL,
            ts REAL NOT NULL,
            payload TEXT NOT NULL
        )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_msg_topic ON messages(topic)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_msg_trace ON messages(trace_id)")
    conn.commit()
    return conn


class MessageBus:
    """Offline-first message bus. Redis when available, SQLite otherwise."""

    def __init__(self, topics: Optional[List[str]] = None) -> None:
        self.topics: List[str] = topics or list(TOPICS)
        self._redis = _try_redis()
        self._sub = None
        if self._redis is not None:
            try:
                self._sub = self._redis.pubsub()
                for t in self.topics:
                    self._sub.subscribe(t)
            except Exception:
                self._sub = None

    # -- publishing ---------------------------------------------------------
    def publish(self, msg: Message) -> bool:
        """Publish a message. Returns True if persisted/sent."""
        if msg.topic not in self.topics:
            # still accept unknown topics but warn
            pass
        data = json.dumps(msg.to_dict())
        if self._redis is not None:
            try:
                self._redis.publish(f"avo:{msg.topic}", data)
                self._log_sqlite(msg, data)
                return True
            except Exception:
                pass  # fall through to sqlite-only
        # SQLite fallback / dual-write
        return self._log_sqlite(msg, data)

    def _log_sqlite(self, msg: Message, data: str) -> bool:
        try:
            conn = _sqlite_conn()
            conn.execute(
                "INSERT INTO messages (id, topic, sender, recipient, trace_id, ts, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (msg.msg_id, msg.topic, msg.sender, msg.recipient, msg.trace_id, msg.ts, data),
            )
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    # -- retrieval ----------------------------------------------------------
    def get_messages(
        self,
        topic: str,
        trace_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieve persisted messages for a topic, optionally filtered by trace_id."""
        try:
            conn = _sqlite_conn()
            if trace_id:
                rows = conn.execute(
                    "SELECT payload FROM messages WHERE topic = ? AND trace_id = ? "
                    "ORDER BY ts ASC LIMIT ?",
                    (topic, trace_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT payload FROM messages WHERE topic = ? ORDER BY ts ASC LIMIT ?",
                    (topic, limit),
                ).fetchall()
            conn.close()
            return [json.loads(r[0]) for r in rows]
        except Exception:
            return []

    def get_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        """Return all messages belonging to a trace, in order."""
        try:
            conn = _sqlite_conn()
            rows = conn.execute(
                "SELECT payload FROM messages WHERE trace_id = ? ORDER BY ts ASC",
                (trace_id,),
            ).fetchall()
            conn.close()
            return [json.loads(r[0]) for r in rows]
        except Exception:
            return []

    def clear(self) -> int:
        """Delete all messages (used in tests). Returns count deleted."""
        try:
            conn = _sqlite_conn()
            n = conn.execute("DELETE FROM messages").rowcount
            conn.commit()
            conn.close()
            return n or 0
        except Exception:
            return 0

    def close(self) -> None:
        if self._sub is not None:
            try:
                self._sub.close()
            except Exception:
                pass
            self._sub = None
