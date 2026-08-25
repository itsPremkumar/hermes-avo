"""MCP trace package: emits AVO execution traces to the AgentWatch trace engine.

Mirrors ``packages/sdk/mcp-trace-server`` architecture: a SQLite ``agent_log``
table (run_id, event_type, timestamp, data) is the durable event store. The
:class:`TraceHandler` writes AVO planning/execution/observation/critique/recovery
events into that table so the trace server's ``agentwatch.query_traces`` tool
can reconstruct the causality graph.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


DEFAULT_TRACE_DB = os.path.join(
    os.environ.get("HERMES_HOME", os.environ.get("USERPROFILE", "/tmp")),
    "hermes_avo", "traces.db",
)

_EVENT_TYPES = {
    "agent.planning",
    "agent.decision",
    "agent.observation",
    "agent.critique",
    "tool.call",
    "tool.result",
    "checkpoint.created",
    "circuitbreaker.tripped",
    "cost.update",
}


@dataclass
class TraceEvent:
    event_id: str
    run_id: str
    type: str
    timestamp: str
    data: Dict[str, Any]

    def to_row(self) -> tuple:
        return (self.event_id, self.run_id, self.type, self.timestamp, json.dumps(self.data))


class TraceHandler:
    """Writes structured execution traces to a SQLite ``agent_log`` table.

    The schema is intentionally identical to the reference trace server so the
    existing ``agentwatch.query_traces`` / ``recover_point`` / ``costs`` /
    ``alerts`` tools can read AVO traces without modification.

    All writes are best-effort: a DB failure is logged (not raised) so the
    calling agent session never crashes.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path: str = db_path or os.environ.get("AGENTWATCH_TRACE_DB") or DEFAULT_TRACE_DB
        self._lock = threading.Lock()
        self._init_db()

    # -- schema -------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=5.0, isolation_level="IMMEDIATE")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS agent_log (
                        id        INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id    TEXT    NOT NULL,
                        event_type TEXT   NOT NULL,
                        timestamp TEXT    NOT NULL,
                        data      TEXT    NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_log_run     ON agent_log(run_id);
                    CREATE INDEX IF NOT EXISTS idx_log_ts      ON agent_log(timestamp);
                    CREATE INDEX IF NOT EXISTS idx_log_type    ON agent_log(event_type);
                    """
                )
                conn.commit()
            finally:
                conn.close()

    # -- writing ------------------------------------------------------------
    def record(
        self,
        run_id: str,
        event_type: str,
        data: Dict[str, Any],
        timestamp: Optional[str] = None,
        event_id: Optional[str] = None,
    ) -> str:
        """Persist one trace event. Returns the event_id."""
        eid = event_id or f"evt-{uuid.uuid4().hex[:10]}"
        eid = eid if eid else f"evt-{uuid.uuid4().hex[:10]}"
        ev = TraceEvent(
            event_id=eid,
            run_id=run_id,
            type=event_type,
            timestamp=timestamp or _iso_now(),
            data=data,
        )
        with self._lock:
            try:
                conn = self._connect()
                try:
                    conn.execute(
                        "INSERT INTO agent_log (run_id, event_type, timestamp, data) "
                        "VALUES (?, ?, ?, ?)",
                        ev.to_row()[1:],  # row tuple is (event_id, run_id, type, ts, data)
                    )
                    conn.commit()
                finally:
                    conn.close()
            except Exception as e:  # never crash the session
                import logging
                logging.getLogger("hermes_avo.trace").warning("trace write failed: %s", e)
        return eid

    # -- domain-level convenience ------------------------------------------
    def planning(self, run_id: str, avatar: str, goal: str, plan: Optional[Any] = None) -> str:
        return self.record(run_id, "agent.planning", {
            "run_id": run_id, "avatar": avatar, "goal": goal, "plan": plan or []
        })

    def decision(self, run_id: str, avatar: str, decision: str, rationale: str = "",
                 cause_event_ids: Optional[List[str]] = None) -> str:
        return self.record(run_id, "agent.decision", {
            "run_id": run_id, "avatar": avatar, "decision": decision,
            "rationale": rationale, "cause_event_ids": cause_event_ids or [],
        })

    def tool_call(self, run_id: str, avatar: str, tool_name: str, arguments: Dict[str, Any]) -> str:
        eid = f"call-{uuid.uuid4().hex[:8]}"
        self.record(run_id, "tool.call", {
            "event_id": eid, "run_id": run_id, "avatar": avatar,
            "tool_name": tool_name, "arguments": arguments,
        }, event_id=eid)
        return eid

    def tool_result(self, run_id: str, avatar: str, tool_name: str,
                    cause_event_id: str, result: Any, error: Optional[str] = None,
                    latency_ms: int = 0) -> str:
        eid = f"result-{uuid.uuid4().hex[:8]}"
        return self.record(run_id, "tool.result", {
            "event_id": eid, "run_id": run_id, "avatar": avatar,
            "tool_name": tool_name, "cause_event_id": cause_event_id,
            "result": result, "error": error, "latency_ms": latency_ms,
        }, event_id=eid)

    def observation(self, run_id: str, avatar: str, data: Dict[str, Any]) -> str:
        return self.record(run_id, "agent.observation", {
            "run_id": run_id, "avatar": avatar, "data": data,
        })

    def critique(self, run_id: str, avatar: str, target: str, score: float,
                 critique: str) -> str:
        return self.record(run_id, "agent.critique", {
            "run_id": run_id, "avatar": avatar, "target": target,
            "score": score, "critique": critique,
        })

    def checkpoint(self, run_id: str, avatar: str, step: int, state: Dict[str, Any]) -> str:
        return self.record(run_id, "checkpoint.created", {
            "checkpoint_id": f"ckpt-{uuid.uuid4().hex[:8]}", "run_id": run_id, "avatar": avatar,
            "step": step, "state": state,
        })

    def cost_update(self, run_id: str, avatar: str, model: str,
                    tokens: int, spend: float) -> str:
        return self.record(run_id, "cost.update", {
            "run_id": run_id, "avatar": avatar, "model": model,
            "tokens": tokens, "spend": spend,
        })

    def circuit_breaker(self, run_id: str, avatar: str, cb_type: str,
                        threshold: float, actual: float, message: str) -> str:
        return self.record(run_id, "circuitbreaker.tripped", {
            "run_id": run_id, "avatar": avatar, "circuit_breaker_type": cb_type,
            "threshold": threshold, "actual": actual, "message": message,
        })

    def recovery(self, run_id: str, avatar: str, reason: str,
                  resume_state: Optional[Dict[str, Any]] = None) -> str:
        return self.record(run_id, "agent.recovery", {
            "run_id": run_id, "avatar": avatar, "reason": reason,
            "resume_state": resume_state or {},
        })

    def _read_rows(self, query: str, params: tuple) -> List[Dict[str, Any]]:
        """Helper: run a read query under the lock, never raising."""
        with self._lock:
            try:
                conn = self._connect()
                try:
                    rows = conn.execute(query, params).fetchall()
                finally:
                    conn.close()
                return rows
            except Exception:
                return []

    # -- reading ------------------------------------------------------------
    def query(self, run_id: str, event_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if event_types:
            placeholders = ",".join("?" for _ in event_types)
            rows = self._read_rows(
                f"SELECT event_type, timestamp, data FROM agent_log "
                f"WHERE run_id = ? AND event_type IN ({placeholders}) ORDER BY id ASC",
                (run_id, *event_types),
            )
        else:
            rows = self._read_rows(
                "SELECT event_type, timestamp, data FROM agent_log "
                "WHERE run_id = ? ORDER BY id ASC",
                (run_id,),
            )
        return [{"type": r[0], "timestamp": r[1], "data": json.loads(r[2])} for r in rows]

    def latest_checkpoint(self, run_id: str) -> Optional[Dict[str, Any]]:
        rows = self._read_rows(
            "SELECT data FROM agent_log WHERE run_id = ? AND event_type = 'checkpoint.created' "
            "ORDER BY id DESC LIMIT 1",
            (run_id,),
        )
        return json.loads(rows[0][0]) if rows else None

    def close(self) -> None:
        pass


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def make_trace_handler(db_path: Optional[str] = None) -> TraceHandler:
    return TraceHandler(db_path)
