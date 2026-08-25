"""Hermes-AVO MCP Bridge Server.

Exposes AVO agents as Model Context Protocol (MCP) tools so that any MCP-capable
client (Claude Desktop, Cursor, the Hermes CLI, etc.) can invoke Hermes avatars
directly.  Each avatar maps to one tool plus the aggregate ``avio_execute_goal``
tool.

Transports:
  * **stdio** — default.  JSON-RPC over stdin/stdout.  Run ``python -m
    hermes_avo.mcp_bridge.server``.
  * **HTTP**  — ``python -m hermes_avo.mcp_bridge.server --transport
    shttp --port 8000``.

Architecture mirrors ``packages/sdk/mcp-trace-server/``: the :class:`AvoBridge`
holds the tool registry, dispatches calls into a pluggable
:class:`~hermes_avo.communication.message_bus.MessageBus` +
:class:`~hermes_avo.mcp_bridge.trace_handler.TraceHandler`, and never blocks the
MCP event loop.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
import mcp.types as types

from hermes_avo.communication.message_bus import Message, MessageBus
from hermes_avo.mcp_bridge.bot_adapter import AVATAR_SPEC, TOOL_TO_AVATAR, AvatarTool
from hermes_avo.mcp_bridge.trace.trace_handler import (
    TraceHandler,
    make_trace_handler,
)

__all__ = [
    "AvoBridge",
    "build_server",
    "run_stdio",
    "run_http",
    "main",
]

SERVER_NAME = "hermes-avo"
SERVER_VERSION = "2.0.0"


class AvoBridge:
    """Core orchestrator that maps MCP tool calls to AVO avatar goals.

    The bridge is transport-agnostic: it knows about tools, not about how they
    arrive.  ``build_server`` wires this class into the MCP :class:`Server`.
    """

    def __init__(
        self,
        bus: Optional[MessageBus] = None,
        tracer: Optional[TraceHandler] = None,
    ) -> None:
        self.bus: MessageBus = bus or MessageBus()
        self.tracer: TraceHandler = tracer or make_trace_handler()
        # Pre-build the tool lookup from the static AVATAR_SPEC table.
        self._tools: Dict[str, AvatarTool] = {t.tool_name: t for t in AVATAR_SPEC}

    # ------------------------------------------------------------------ #
    # Tool registry
    # ------------------------------------------------------------------ #
    def list_tools(self) -> List[types.Tool]:
        """Return all MCP :class:`~mcp.types.Tool` definitions (13 tools)."""
        tools: List[types.Tool] = []
        for avatar_tool in AVATAR_SPEC:
            tools.append(
                types.Tool(
                    name=avatar_tool.tool_name,
                    title=avatar_tool.tool_name,
                    description=avatar_tool.description,
                    input_schema=avatar_tool.input_schema,
                )
            )
        return tools

    def get_tool(self, name: str) -> Optional[AvatarTool]:
        """Look up an :class:`AvatarTool` by MCP tool name."""
        return self._tools.get(name)

    # ------------------------------------------------------------------ #
    # Tool execution
    # ------------------------------------------------------------------ #
    def call_tool(
        self,
        name: str,
        arguments: Dict[str, Any],
    ) -> types.CallToolResult:
        """Execute a tool call.

        In a production deployment this would dispatch to a live AVO agent
        process.  For Phase 2 the bridge simulates the round-trip: it validates
        the arguments, publishes an *execution* message on the bus, records a
        trace event, and returns a structured result with a synthetic
        ``task_id``, ``status``, and ``trace_id``.
        """
        tool = self._tools.get(name)
        if tool is None:
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"Unknown tool: {name}",
                    )
                ],
                is_error=True,
            )

        # Validate arguments against the schema's required fields.
        schema = tool.input_schema
        required = schema.get("required", [])
        missing = [r for r in required if r not in arguments]
        if missing:
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"Missing required arguments: {missing}",
                    )
                ],
                is_error=True,
            )

        # Normalise / apply defaults
        goal = arguments.get("goal", "")
        budget = float(arguments.get("budget", tool.default_budget))
        timeout = int(arguments.get("timeout", tool.default_timeout))

        # Guard-rail validation
        if not goal or not goal.strip():
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text="Argument 'goal' must be a non-empty string.",
                    )
                ],
                is_error=True,
            )
        if not (0.0 <= budget <= 1.0):
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"Budget {budget} out of range [0, 1].",
                    )
                ],
                is_error=True,
            )
        if timeout < 1:
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"Timeout {timeout} must be >= 1 second.",
                    )
                ],
                is_error=True,
            )

        trace_id = f"trace-{uuid.uuid4().hex[:12]}"
        task_id = f"task-{uuid.uuid4().hex[:12]}"

        # Record trace: planning event from this avatar.
        self.tracer.planning(
            run_id=trace_id,
            avatar=tool.avatar,
            goal=goal,
            plan=[f"execute via {tool.avatar}"],
        )
        self.tracer.tool_call(
            run_id=trace_id,
            avatar=tool.avatar,
            tool_name=tool.tool_name,
            arguments={"goal": goal, "budget": budget, "timeout": timeout},
        )

        # Publish on the execution topic so message-bus subscribers see it.
        msg = Message(
            topic="execution",
            sender="@ceo",
            recipient=tool.avatar,
            payload={
                "tool_name": tool.tool_name,
                "avatar": tool.avatar,
                "goal": goal,
                "budget": budget,
                "timeout": timeout,
            },
            trace_id=trace_id,
        )
        self.bus.publish(msg)

        # Simulated result — a real deployment would await an agent worker.
        result_text = json.dumps(
            {
                "task_id": task_id,
                "status": "completed",
                "trace_id": trace_id,
                "avatar": tool.avatar,
                "tool": tool.tool_name,
                "goal": goal,
                "budget_used": round(budget, 4),
            },
            indent=2,
        )

        self.tracer.tool_result(
            run_id=trace_id,
            avatar=tool.avatar,
            tool_name=tool.tool_name,
            cause_event_id=f"call-{trace_id[:8]}",
            result={"status": "completed", "task_id": task_id},
        )

        return types.CallToolResult(
            content=[
                types.TextContent(type="text", text=result_text),
            ],
        )


def build_server(bridge: Optional[AvoBridge] = None) -> Server:
    """Construct an MCP :class:`Server` wired to *bridge*.

    If *bridge* is ``None`` a default :class:`AvoBridge` is created, allowing
    tests to inject a fake bus / tracer.
    """
    bridge = bridge or AvoBridge()

    server = Server(
        SERVER_NAME,
        version=SERVER_VERSION,
        title="Hermes AVO Bridge",
        instructions=(
            "MCP bridge exposing Hermes AVO agents as tools. "
            "Available tools: " + ", ".join(t.tool_name for t in AVATAR_SPEC)
        ),
        on_list_tools=_make_list_tools_handler(bridge),
        on_call_tool=_make_call_tool_handler(bridge),
    )
    return server


def _make_list_tools_handler(bridge: AvoBridge):
    """Create the MCP ``on_list_tools`` handler closure."""

    async def handler(
        ctx, params: Optional[types.PaginatedRequestParams] = None
    ) -> types.ListToolsResult:
        tools = bridge.list_tools()
        return types.ListToolsResult(tools=tools)

    return handler


def _make_call_tool_handler(bridge: AvoBridge):
    """Create the MCP ``on_call_tool`` handler closure."""

    async def handler(
        ctx, params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        arguments = params.arguments or {}
        return bridge.call_tool(params.name, arguments)

    return handler


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------
def run_stdio() -> None:
    """Run the MCP server over stdio (JSON-RPC 2.0 over stdin/stdout).

    This is the transport that MCP clients use by default.  The process stays
    alive until stdin closes or SIGTERM is received.
    """
    bridge = AvoBridge()
    server = build_server(bridge)
    init_opts = server.create_initialization_options(
        notification_options=NotificationOptions(tools_changed=True),
    )

    async def _run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                init_opts,
                raise_exceptions=True,
            )

    _install_signal_handlers()
    asyncio.run(_run())


def run_http(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the MCP server over **streamable HTTP** (SSE fallback supported)."""
    bridge = AvoBridge()
    server = build_server(bridge)
    app = server.streamable_http_app()
    import uvicorn  # type: ignore[import-not-found]

    uvicorn.run(app, host=host, port=port)


def _install_signal_handlers() -> None:
    """Graceful shutdown on SIGTERM / SIGINT."""

    def _shutdown(signum, frame):  # noqa: ARG001
        sys.stderr.write(f"Received signal {signum}, shutting down.\n")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    """CLI: ``python -m hermes_avo.mcp_bridge.server [--transport ...]``."""
    parser = argparse.ArgumentParser(
        description="Hermes-AVO MCP bridge server",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "shttp"],
        default="stdio",
        help="Transport: 'stdio' (default) or 'shttp' (streamable HTTP).",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="HTTP host (shttp only, default 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTTP port (shttp only, default 8000).",
    )
    args = parser.parse_args(argv)

    if args.transport == "shttp":
        run_http(host=args.host, port=args.port)
    else:
        run_stdio()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
