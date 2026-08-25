"""Shared pytest configuration for the hermes_avo test suite."""
import sys
import os
import tempfile

# Insert the repo root so `hermes_avo` resolves regardless of cwd.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Use a temp SQLite DB for message bus + trace handler so tests are hermetic
# and never touch the real user HOME directory.
_TMP_HOME = tempfile.mkdtemp(prefix="hermes_avo_test_")
os.environ.setdefault("HERMES_HOME", _TMP_HOME)
os.environ.setdefault("AVO_MESSAGE_DB", os.path.join(_TMP_HOME, "messages.db"))
os.environ.setdefault("AGENTWATCH_TRACE_DB", os.path.join(_TMP_HOME, "traces.db"))

import pytest


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Ensure each test gets a fresh SQLite DB for message bus and trace handler."""
    msg_db = str(tmp_path / "messages.db")
    trace_db = str(tmp_path / "traces.db")
    monkeypatch.setenv("AVO_MESSAGE_DB", msg_db)
    monkeypatch.setenv("AGENTWATCH_TRACE_DB", trace_db)
    yield
    # Reset any cached module-level redis connection state
    import hermes_avo.communication.message_bus as mb_mod
    mb_mod._redis = None
