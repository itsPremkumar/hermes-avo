"""Unit tests for hermes_avo.mcp_bridge.server (4 tests)."""
from __future__ import annotations

import json

import pytest

from hermes_avo.mcp_bridge.server import AvoBridge
from hermes_avo.mcp_bridge.bot_adapter import AVATAR_SPEC


class TestServer:
    """Tests for the MCP bridge server / AvoBridge core."""

    def test_list_tools_returns_14_tools(self):
        """AvoBridge.list_tools returns one MCP Tool per avatar (14 total)."""
        bridge = AvoBridge()
        tools = bridge.list_tools()
        assert len(tools) == 14
        # Verify all tool names from AVATAR_SPEC are present
        tool_names = {t.name for t in tools}
        spec_names = {at.tool_name for at in AVATAR_SPEC}
        assert tool_names == spec_names
        # Verify MCP Tool object shape
        t = tools[0]
        assert t.name == "avio_execute_goal"
        assert t.description
        assert t.input_schema["type"] == "object"

    def test_call_tool_valid_goal(self):
        """call_tool with a valid goal returns a completed result."""
        bridge = AvoBridge()
        result = bridge.call_tool("avio_build_code", {"goal": "Generate a Python CLI"})

        assert result.is_error is False
        assert result.content is not None
        assert len(result.content) == 1
        text = result.content[0].text
        data = json.loads(text)
        assert data["status"] == "completed"
        assert data["avatar"] == "@agent-builder"
        assert data["tool"] == "avio_build_code"
        assert "task_id" in data
        assert "trace_id" in data
        assert data["budget_used"] == 0.5

    def test_call_tool_unknown_tool_error(self):
        """call_tool with an unknown tool name returns an error result."""
        bridge = AvoBridge()
        result = bridge.call_tool("nonexistent_tool", {"goal": "test"})
        assert result.is_error is True
        assert "Unknown tool" in result.content[0].text

    def test_call_tool_missing_goal_error(self):
        """call_tool with an empty goal returns a validation error."""
        bridge = AvoBridge()
        result = bridge.call_tool("avio_execute_goal", {"goal": ""})
        assert result.is_error is True
        assert "non-empty" in result.content[0].text.lower() or "goal" in result.content[0].text.lower()
