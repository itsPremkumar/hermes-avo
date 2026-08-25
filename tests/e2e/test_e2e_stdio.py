"""End-to-end stdio protocol tests for the AVO-MCP bridge server (7 tests).

These tests spawn the real MCP server process over a stdio transport and drive
it with raw JSON-RPC 2.0 messages — exactly what a real MCP client would do.

Cross-platform note: ``select.select`` on Windows only works on sockets, not
on file handles.  We therefore read stdout in a background thread and collect
complete newline-delimited JSON messages, which works identically on Linux,
macOS, and Windows.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from typing import Any

import pytest

# The server module path for `python -m hermes_avo.mcp_bridge.server`
SERVER_MODULE = "hermes_avo.mcp_bridge.server"

_JSON_RPC_CONTENT_LENGTH_LIMIT = 64 * 1024  # safety guard


def _run_server(env: dict) -> subprocess.Popen:
    """Spawn the MCP server in stdio mode."""
    proc = subprocess.Popen(
        [sys.executable, "-m", SERVER_MODULE],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=False,  # binary mode for precise control
    )
    return proc


def _send_request(
    proc: subprocess.Popen, method: str, params: dict | None = None, request_id: int = 1
) -> None:
    """Send a JSON-RPC request to the server's stdin."""
    msg: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
    }
    if params is not None:
        msg["params"] = params
    data = json.dumps(msg).encode() + b"\n"
    proc.stdin.write(data)
    proc.stdin.flush()


class _StdoutReader:
    """Reads newline-delimited JSON from a subprocess stdout in a thread.

    This avoids ``select.select`` which is socket-only on Windows.
    """

    def __init__(self, proc: subprocess.Popen) -> None:
        self._proc = proc
        self._queue: list[dict] = []
        self._errors: list[str] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        def _reader() -> None:
            assert self._proc.stdout is not None
            buf = b""
            while not self._stop.is_set():
                try:
                    chunk = self._proc.stdout.read(1)
                except (ValueError, OSError):
                    break
                if not chunk:
                    if buf.strip():
                        self._errors.append(
                            f"Truncated message without newline: {buf!r}"
                        )
                    break
                buf += chunk
                if buf.endswith(b"\n"):
                    line = buf.strip()
                    buf = b""
                    if line:
                        try:
                            self._queue.append(json.loads(line))
                        except json.JSONDecodeError as exc:
                            self._errors.append(
                                f"JSON parse error: {exc} — line: {line!r}"
                            )

        self._thread = threading.Thread(target=_reader, daemon=True)
        self._thread.start()

    def read_message(self, timeout: float = 10.0) -> dict:
        """Wait up to *timeout* seconds for the next complete message."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._queue:
                return self._queue.pop(0)
            if self._errors:
                raise AssertionError(
                    f"stdout reader error: {self._errors[0]}"
                )
            if self._proc.poll() is not None:
                raise TimeoutError(
                    "Server process exited unexpectedly (read_message)"
                )
            time.sleep(0.02)
        raise TimeoutError(
            f"No response from server within {timeout}s "
            f"(queue={self._queue!r}, errors={self._errors!r})"
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def stderr_text(self) -> str:
        """Return any stderr output (non-blocking best-effort)."""
        if self._proc.stderr is None:
            return ""
        # Set non-blocking so we don't block on read
        import fcntl  # Unix only

        try:
            fd = self._proc.stderr.fileno()
            fl = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
            data = self._proc.stderr.read()
            return data.decode("utf-8", errors="replace") if data else ""
        except (OSError, ImportError):
            return ""


def _send_notification(
    proc: subprocess.Popen, method: str, params: dict | None = None
) -> None:
    """Send a JSON-RPC notification (no id)."""
    msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    data = json.dumps(msg).encode() + b"\n"
    proc.stdin.write(data)
    proc.stdin.flush()


def _wait_for_init(
    proc: subprocess.Popen, reader: _StdoutReader, request_id: int = 1
) -> dict:
    """Send initialize, read the response, send initialized notification."""
    _send_request(
        proc,
        "initialize",
        {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "prompts": {},
                "sampling": {},
                "tools": {},
            },
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0",
            },
        },
        request_id=request_id,
    )
    init_response = reader.read_message()
    assert init_response["jsonrpc"] == "2.0", f"Bad jsonrpc: {init_response!r}"
    assert "result" in init_response or "error" not in init_response
    _send_notification(proc, "notifications/initialized")
    return init_response


@pytest.fixture
def server_env(tmp_path):
    """Environment for the server subprocess with isolated paths."""
    env = dict(os.environ)
    env["HERMES_HOME"] = str(tmp_path)
    env["AVO_MESSAGE_DB"] = str(tmp_path / "messages.db")
    env["AGENTWATCH_TRACE_DB"] = str(tmp_path / "traces.db")
    env["PYTHONPATH"] = os.environ.get("PYTHONPATH", "")
    return env


@pytest.fixture
def initialized_server(server_env):
    """Spawn the server, complete MCP initialization, yield (proc, reader)."""
    proc = _run_server(server_env)
    reader = _StdoutReader(proc)
    reader.start()
    try:
        _wait_for_init(proc, reader)
        yield proc, reader
    finally:
        reader.stop()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


class TestE2EStdio:
    """End-to-end stdio protocol tests (7 tests)."""

    def test_initialize_handshake(self, server_env):
        """The server responds to initialize with its capabilities."""
        proc = _run_server(server_env)
        reader = _StdoutReader(proc)
        reader.start()
        try:
            _send_request(
                proc,
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "prompts": {},
                        "sampling": {},
                        "tools": {},
                    },
                    "clientInfo": {"name": "test-client", "version": "1.0.0"},
                },
                request_id=1,
            )
            resp = reader.read_message()
            assert resp["jsonrpc"] == "2.0"
            assert resp["id"] == 1
            result = resp["result"]
            assert result["protocolVersion"] == "2024-11-05"
            assert "capabilities" in result
            assert result["serverInfo"]["name"] == "hermes-avo"
        finally:
            reader.stop()
            proc.terminate()
            proc.wait(timeout=5)

    def test_tools_list_request(self, initialized_server):
        """tools/list returns all 14 AVO tools."""
        proc, reader = initialized_server
        _send_request(proc, "tools/list", {}, request_id=2)
        resp = reader.read_message()
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 2
        tools = resp["result"]["tools"]
        assert len(tools) == 14
        tool_names = {t["name"] for t in tools}
        assert "avio_execute_goal" in tool_names
        assert "avio_build_code" in tool_names
        assert "avio_create_server" in tool_names

    def test_tools_call_execute_goal(self, initialized_server):
        """tools/call on avio_execute_goal returns a completed task."""
        proc, reader = initialized_server
        _send_request(
            proc,
            "tools/call",
            {
                "name": "avio_execute_goal",
                "arguments": {"goal": "Decompose and build MCP bridge"},
            },
            request_id=3,
        )
        resp = reader.read_message()
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 3
        content = resp["result"]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "text"
        data = json.loads(content[0]["text"])
        assert data["status"] == "completed"
        assert data["avatar"] == "@ceo"
        assert data["tool"] == "avio_execute_goal"

    def test_tools_call_unknown_tool(self, initialized_server):
        """tools/call with an unknown tool name returns an error."""
        proc, reader = initialized_server
        _send_request(
            proc,
            "tools/call",
            {"name": "nonexistent_tool", "arguments": {"goal": "test"}},
            request_id=4,
        )
        resp = reader.read_message()
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 4
        assert resp["result"]["isError"] is True

    def test_tools_call_missing_goal(self, initialized_server):
        """tools/call with empty goal returns a validation error."""
        proc, reader = initialized_server
        _send_request(
            proc,
            "tools/call",
            {"name": "avio_build_code", "arguments": {"goal": ""}},
            request_id=5,
        )
        resp = reader.read_message()
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 5
        assert resp["result"]["isError"] is True

    def test_tools_call_budget_out_of_range(self, initialized_server):
        """tools/call with budget > 1.0 returns a validation error."""
        proc, reader = initialized_server
        _send_request(
            proc,
            "tools/call",
            {
                "name": "avio_build_code",
                "arguments": {"goal": "test", "budget": 2.0},
            },
            request_id=6,
        )
        resp = reader.read_message()
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 6
        assert resp["result"]["isError"] is True

    def test_tools_call_timeout_out_of_range(self, initialized_server):
        """tools/call with timeout=0 returns a validation error."""
        proc, reader = initialized_server
        _send_request(
            proc,
            "tools/call",
            {
                "name": "avio_build_code",
                "arguments": {"goal": "test", "timeout": 0},
            },
            request_id=7,
        )
        resp = reader.read_message()
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 7
        assert resp["result"]["isError"] is True
