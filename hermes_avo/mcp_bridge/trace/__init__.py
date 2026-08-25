"""MCP Trace Integration.

Emits AVO execution traces to the AgentWatch trace engine, mirroring the
``packages/sdk/mcp-trace-server`` architecture.
"""
from hermes_avo.mcp_bridge.trace.trace_handler import (
    DEFAULT_TRACE_DB,
    TraceEvent,
    TraceHandler,
    make_trace_handler,
)

__all__ = [
    "DEFAULT_TRACE_DB",
    "TraceEvent",
    "TraceHandler",
    "make_trace_handler",
]
