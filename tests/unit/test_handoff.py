"""Unit tests for hermes_avo.communication.handoff (6 tests)."""
from __future__ import annotations

import pytest

from hermes_avo.communication.handoff import (
    HandoffContext,
    HandoffManager,
    ResourceBudget,
    TaskState,
)
from hermes_avo.communication.message_bus import MessageBus


class TestHandoff:
    """Tests for the formal handoff protocol."""

    def test_create_handoff_publishes_on_planning(self):
        """create_handoff publishes a planning-topic message with context."""
        bus = MessageBus()
        bus.clear()
        mgr = HandoffManager(bus=bus)
        budget = ResourceBudget(budget=0.5, used=0.0, timeout=300)

        ctx = mgr.create_handoff(
            trace_id="trace-H1",
            from_avatar="@ceo",
            to_avatar="@agent-builder",
            goal="Build an MCP server",
            budget=budget,
        )
        assert isinstance(ctx, HandoffContext)
        assert ctx.trace_id == "trace-H1"
        assert ctx.from_avatar == "@ceo"
        assert ctx.to_avatar == "@agent-builder"
        assert ctx.goal == "Build an MCP server"
        assert ctx.budget.budget == 0.5
        assert ctx.handoff_id.startswith("h-")

        msgs = bus.get_messages("planning", trace_id="trace-H1")
        assert len(msgs) == 1
        assert msgs[0]["sender"] == "@ceo"
        assert msgs[0]["recipient"] == "@agent-builder"
        bus.clear()

    def test_accept_handoff_publishes_on_execution(self):
        """accept_handoff publishes an execution-topic ack message."""
        bus = MessageBus()
        bus.clear()
        mgr = HandoffManager(bus=bus)
        budget = ResourceBudget(budget=0.3, used=0.0, timeout=180)

        ctx = mgr.create_handoff(
            trace_id="trace-H2",
            from_avatar="@ceo",
            to_avatar="@mcp-specialist",
            goal="Create MCP server tools",
            budget=budget,
        )
        ack = mgr.accept_handoff(ctx)
        assert ack is not None
        assert ack.topic == "execution"
        assert ack.sender == "@mcp-specialist"
        assert ack.recipient == "@ceo"
        assert ack.payload["ack"] is True
        assert ack.payload["goal"] == "Create MCP server tools"
        bus.clear()

    def test_get_handoffs_for_trace(self):
        """get_handoffs_for_trace retrieves all planning messages for a trace."""
        bus = MessageBus()
        bus.clear()
        mgr = HandoffManager(bus=bus)
        budget = ResourceBudget(budget=0.5, timeout=300)

        mgr.create_handoff("trace-H3", "@ceo", "@agent-builder", "goal-1", budget)
        mgr.create_handoff("trace-H3", "@agent-builder", "@qa-lead", "goal-2", budget)

        handoffs = mgr.get_handoffs_for_trace("trace-H3")
        assert len(handoffs) == 2
        assert handoffs[0]["payload"]["to_avatar"] == "@agent-builder"
        assert handoffs[1]["payload"]["to_avatar"] == "@qa-lead"
        bus.clear()

    def test_get_handoffs_for_avatar(self):
        """get_handoffs_for_avatar filters by recipient avatar."""
        bus = MessageBus()
        bus.clear()
        mgr = HandoffManager(bus=bus)
        budget = ResourceBudget(budget=0.5, timeout=300)

        mgr.create_handoff("trace-H4", "@ceo", "@agent-builder", "g1", budget)
        mgr.create_handoff("trace-H4", "@ceo", "@writer", "g2", budget)
        mgr.create_handoff("trace-H4", "@ceo", "@agent-builder", "g3", budget)

        builder_handoffs = mgr.get_handoffs_for_avatar("@agent-builder")
        assert len(builder_handoffs) == 2
        assert all(h["payload"]["to_avatar"] == "@agent-builder" for h in builder_handoffs)
        bus.clear()

    def test_resource_budget_default_values(self):
        """ResourceBudget defaults: budget=0, used=0, timeout=300."""
        rb = ResourceBudget()
        assert rb.budget == 0.0
        assert rb.used == 0.0
        assert rb.timeout == 300
        d = rb.to_dict()
        assert d == {"budget": 0.0, "used": 0.0, "timeout": 300}

    def test_task_state_to_dict(self):
        """TaskState.to_dict serializes all fields including notes."""
        ts = TaskState(
            task_id="task-abc",
            status="pending",
            step=2,
            progress=0.5,
            notes=["note1", "note2"],
        )
        d = ts.to_dict()
        assert d["task_id"] == "task-abc"
        assert d["status"] == "pending"
        assert d["step"] == 2
        assert d["progress"] == 0.5
        assert d["notes"] == ["note1", "note2"]
