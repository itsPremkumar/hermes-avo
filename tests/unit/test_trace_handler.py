"""Unit tests for hermes_avo.mcp_bridge.trace.trace_handler (5 tests)."""
from __future__ import annotations

import json
import os
import sqlite3

import pytest

from hermes_avo.mcp_bridge.trace.trace_handler import (
    TraceHandler,
    make_trace_handler,
)


class TestTraceHandler:
    """Tests for the trace event store."""

    def test_record_returns_event_id(self, tmp_path):
        """TraceHandler.record persists an event and returns its event_id."""
        db = str(tmp_path / "trace.db")
        handler = TraceHandler(db_path=db)
        eid = handler.record("run-1", "agent.planning", {"goal": "test"})
        assert eid.startswith("evt-")
        handler.close()

    def test_planning_event_stored(self, tmp_path):
        """The .planning() convenience method writes an agent.planning event."""
        db = str(tmp_path / "trace2.db")
        handler = TraceHandler(db_path=db)
        handler.planning("run-2", "@ceo", "decompose the monolith", ["step1", "step2"])

        rows = handler.query("run-2")
        assert len(rows) == 1
        assert rows[0]["type"] == "agent.planning"
        data = rows[0]["data"]
        assert data["avatar"] == "@ceo"
        assert data["goal"] == "decompose the monolith"
        assert data["plan"] == ["step1", "step2"]
        handler.close()

    def test_tool_call_and_result_linked(self, tmp_path):
        """tool_call writes a tool.call event; tool_result writes a tool.result event."""
        db = str(tmp_path / "trace3.db")
        handler = TraceHandler(db_path=db)
        call_eid = handler.tool_call("run-3", "@agent-builder", "avio_build_code", {"goal": "build"})
        assert call_eid.startswith("call-")

        result_eid = handler.tool_result(
            "run-3", "@agent-builder", "avio_build_code", call_eid,
            {"status": "ok"}, error=None, latency_ms=42,
        )
        assert result_eid.startswith("result-")

        rows = handler.query("run-3")
        assert len(rows) == 2
        assert rows[0]["type"] == "tool.call"
        assert rows[1]["type"] == "tool.result"
        # The result event should reference the call event
        assert rows[1]["data"]["cause_event_id"] == call_eid
        handler.close()

    def test_query_filtered_by_event_type(self, tmp_path):
        """query accepts event_types filter to return only matching events."""
        db = str(tmp_path / "trace4.db")
        handler = TraceHandler(db_path=db)
        handler.planning("run-4", "@ceo", "plan")
        handler.observation("run-4", "@ceo", {"metric": 42})
        handler.critique("run-4", "@qa-lead", "@ceo", 0.9, "good")

        rows_all = handler.query("run-4")
        assert len(rows_all) == 3

        rows_filtered = handler.query("run-4", event_types=["agent.planning", "agent.critique"])
        assert len(rows_filtered) == 2
        types_returned = {r["type"] for r in rows_filtered}
        assert types_returned == {"agent.planning", "agent.critique"}
        handler.close()

    def test_latest_checkpoint_returns_most_recent(self, tmp_path):
        """latest_checkpoint returns the most recently recorded checkpoint."""
        db = str(tmp_path / "trace5.db")
        handler = TraceHandler(db_path=db)
        handler.checkpoint("run-5", "@ceo", 1, {"step": 1})
        handler.checkpoint("run-5", "@ceo", 2, {"step": 2})
        handler.checkpoint("run-5", "@ceo", 3, {"step": 3})

        latest = handler.latest_checkpoint("run-5")
        assert latest is not None
        assert latest["checkpoint_id"].startswith("ckpt-")
        assert "step: 3" in json.dumps(latest) or latest.get("step") == 3
        handler.close()

