"""Unit tests for hermes_avo.mcp_bridge.bot_adapter (5 tests)."""
from __future__ import annotations

import pytest

from hermes_avo.mcp_bridge.bot_adapter import (
    AVATAR_SPEC,
    AVATAR_TOOLS,
    TOOL_TO_AVATAR,
    ALL_AVATARS,
    AvatarTool,
    avatar_names,
    tool_for_avatar,
    tool_schema,
)

EXPECTED_AVATARS = [
    "@ceo",
    "@agent-builder",
    "@mcp-specialist",
    "@qa-lead",
    "@agent-architect",
    "@fullstack-dev",
    "@research-analyst",
    "@devops-engineer",
    "@writer",
    "@reviewer",
    "@security-engineer",
    "@backend",
    "@product-manager",
    "@cto",
]

EXPECTED_TOOLS = [
    "avio_execute_goal",
    "avio_build_code",
    "avio_create_server",
    "avio_test_suite",
    "avio_design_system",
    "avio_frontend",
    "avio_research",
    "avio_run_pipeline",
    "avio_write_content",
    "avio_code_review",
    "avio_security_audit",
    "avio_backend_service",
    "avio_product_plan",
    "avio_infra_strategy",
]


class TestBotAdapter:
    """Tests for the bot-to-AVO adapter."""

    def test_all_avatars_present(self):
        """Every canonical Hermes avatar has a tool definition."""
        assert len(AVATAR_SPEC) == 14
        assert ALL_AVATARS == EXPECTED_AVATARS
        assert avatar_names() == EXPECTED_AVATARS

    def test_avatar_to_tool_mapping(self):
        """Each avatar maps to its expected tool name."""
        for avatar, tool_name in zip(EXPECTED_AVATARS, EXPECTED_TOOLS):
            assert TOOL_TO_AVATAR[tool_name] == avatar
            assert AVATAR_TOOLS[avatar].tool_name == tool_name

    def test_tool_for_avatar_returns_correct_tool(self):
        """tool_for_avatar returns the AvatarTool for a known avatar."""
        tool = tool_for_avatar("@agent-builder")
        assert isinstance(tool, AvatarTool)
        assert tool.avatar == "@agent-builder"
        assert tool.tool_name == "avio_build_code"
        assert tool.default_budget == 0.5
        assert tool.default_timeout == 300

    def test_tool_for_avatar_unknown_returns_none(self):
        """tool_for_avatar returns None for an unregistered avatar."""
        assert tool_for_avatar("@nonexistent") is None
        assert tool_for_avatar("") is None

    def test_tool_schema_raises_key_error_for_unknown(self):
        """tool_schema raises KeyError for an unknown tool name."""
        schema = tool_schema("avio_execute_goal")
        assert schema["type"] == "object"
        assert "goal" in schema["required"]
        with pytest.raises(KeyError):
            tool_schema("nonexistent_tool")
