"""Unit tests for hermes_avo.communication.message_bus (6 tests)."""
from __future__ import annotations

import json
import os
import sqlite3

import pytest

from hermes_avo.communication.message_bus import (
    Message,
    MessageBus,
    TOPICS,
    _sqlite_path,
    _sqlite_conn,
)


class TestMessageBus:
    """Tests for the offline-first message bus."""

    def test_publish_and_retrieve_single_message(self):
        """A published message is retrievable via get_messages by topic."""
        bus = MessageBus()
        bus.clear()
        msg = Message(
            topic="execution",
            sender="@ceo",
            recipient="@agent-builder",
            payload={"goal": "build a server"},
            trace_id="trace-001",
        )
        assert bus.publish(msg) is True
        msgs = bus.get_messages("execution")
        assert len(msgs) == 1
        assert msgs[0]["payload"]["goal"] == "build a server"
        assert msgs[0]["trace_id"] == "trace-001"
        assert msgs[0]["sender"] == "@ceo"
        bus.clear()

    def test_get_messages_by_topic_filtered_by_trace(self):
        """get_messages filters by trace_id when provided."""
        bus = MessageBus()
        bus.clear()
        for i in range(3):
            msg = Message(
                topic="planning",
                sender="@ceo",
                recipient="@agent-builder",
                payload={"goal": f"goal-{i}"},
                trace_id="trace-A",
            )
            bus.publish(msg)
        msg2 = Message(
            topic="planning",
            sender="@ceo",
            recipient="@writer",
            payload={"goal": "goal-X"},
            trace_id="trace-B",
        )
        bus.publish(msg2)

        trace_a = bus.get_messages("planning", trace_id="trace-A")
        assert len(trace_a) == 3
        assert all(m["trace_id"] == "trace-A" for m in trace_a)

        trace_b = bus.get_messages("planning", trace_id="trace-B")
        assert len(trace_b) == 1
        bus.clear()

    def test_get_trace_returns_all_messages_for_trace(self):
        """get_trace returns all messages for a trace_id, ordered by ts."""
        bus = MessageBus()
        bus.clear()
        for topic in TOPICS:
            msg = Message(
                topic=topic,
                sender="@ceo",
                recipient="@agent",
                payload={"step": topic},
                trace_id="trace-T",
            )
            bus.publish(msg)

        trace_msgs = bus.get_trace("trace-T")
        assert len(trace_msgs) == len(TOPICS)
        # Messages should be ordered by timestamp ascending
        timestamps = [m["ts"] for m in trace_msgs]
        assert timestamps == sorted(timestamps)
        bus.clear()

    def test_clear_deletes_all_messages(self):
        """clear() removes all rows and returns the count."""
        bus = MessageBus()
        bus.clear()
        msg1 = Message(
            topic="execution",
            sender="a",
            recipient="b",
            payload={"n": 1},
            trace_id="t1",
        )
        msg2 = Message(
            topic="execution",
            sender="c",
            recipient="d",
            payload={"n": 2},
            trace_id="t1",
        )
        assert bus.publish(msg1) is True
        assert bus.publish(msg2) is True

        msgs_before = bus.get_messages("execution")
        assert len(msgs_before) == 2
        count = bus.clear()
        assert count == 2
        assert bus.get_messages("execution") == []

    def test_unknown_topic_accepted(self):
        """Messages on topics not in TOPICS are still accepted (with a warning)."""
        bus = MessageBus()
        bus.clear()
        msg = Message(
            topic="custom-topic",
            sender="x",
            recipient="y",
            payload={"data": "test"},
            trace_id="trace-C",
        )
        assert bus.publish(msg) is True
        msgs = bus.get_messages("custom-topic")
        assert len(msgs) == 1
        assert msgs[0]["payload"]["data"] == "test"
        bus.clear()

    def test_sqlite_persistence_survives_reopen(self):
        """Messages written by one MessageBus survive reopening the DB."""
        bus1 = MessageBus()
        bus1.clear()
        msg = Message(
            topic="observation",
            sender="@agent-builder",
            recipient="@ceo",
            payload={"result": "done"},
            trace_id="trace-PERSIST",
        )
        bus1.publish(msg)

        # Create a new bus pointing at the same DB file
        bus2 = MessageBus()
        msgs = bus2.get_messages("observation", trace_id="trace-PERSIST")
        assert len(msgs) == 1
        assert msgs[0]["payload"]["result"] == "done"
        bus2.clear()
