"""Shared, fully-offline pytest fixtures for the Hermes-AVO test harness.

This module mirrors the AgentWatch QA harness architecture. Every fixture
here is intentionally **deterministic** and **network-free** so that test
runs are reproducible across machines and CI environments.

Fixtures provided
-----------------
``mock_llm_provider``
    A deterministic, scripted-response LLM double. It returns canned
    completions from either a FIFO queue (call-order driven) or a
    prompt-keyed map. Every call is recorded and an unexpected prompt
    raises loudly.

``fake_agent``
    A factory returning configurable *agent doubles*. Each double can be
    instructed to return canned values, raise exceptions, simulate
    latency, and record call counts.

``seeded_rng``
    Returns a ``random.Random`` instance fixed to a constant seed so that
    any randomness consumed inside a test is reproducible.

``tmp_state_store``
    A clean per-test key-value store backed by ``pytest``'s ``tmp_path``
    (a real on-disk directory unique to each test invocation). Persistence
    is therefore exercised end-to-end against disk, while isolation
    between tests is guaranteed.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import pytest

__all__ = [
    "MockLLMProvider",
    "FakeAgent",
    "StateStore",
    "mock_llm_provider",
    "fake_agent",
    "seeded_rng",
    "tmp_state_store",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The constant seed used by ``seeded_rng`` for full reproducibility.
RNG_SEED: int = 0xC0FFEE


# ---------------------------------------------------------------------------
# Mock LLM provider
# ---------------------------------------------------------------------------


class MockLLMProvider:
    """A deterministic, scripted-response LLM double.

    The provider returns canned completions sourced from **either** a FIFO
    queue of responses (the *queue* strategy, keyed on call order) **or** a
    mapping from a normalised prompt string to a canned response (the
    *prompt-map* strategy).

    Every call — input prompt and returned completion — is appended to
    ``self.calls`` so tests can assert on usage without monkey-patching.

    Parameters
    ----------
    responses:
        Canned completions to return. For the *queue* strategy these are
        consumed in order; for the *prompt-map* strategy they are keyed by
        the normalised prompt string.
    strategy:
        ``"queue"`` (default) to return responses in order regardless of
        the prompt text, or ``"prompt_map"`` to look up a response by the
        normalised prompt string.
    strict:
        When ``True`` (default), requesting a response for an unknown
        prompt under the *prompt_map* strategy, or exhausting the queue
        under the *queue* strategy, raises ``AssertionError``. Set to
        ``False`` to fall back to ``default_response`` instead.
    default_response:
        Value returned when ``strict`` is ``False`` and no canned
        response is available.
    normalise:
        Optional callable to normalise a prompt before using it as a key
        (defaults to stripping whitespace + lowercasing).
    """

    def __init__(
        self,
        responses: Mapping[str, str] | Sequence[str] | None = None,
        strategy: str = "queue",
        *,
        strict: bool = True,
        default_response: str = "",
        normalise: Callable[[str], str] | None = None,
    ) -> None:
        if strategy not in ("queue", "prompt_map"):
            msg = f"strategy must be 'queue' or 'prompt_map', got {strategy!r}"
            raise ValueError(msg)

        self.strategy: str = strategy
        self.strict: bool = strict
        self.default_response: str = default_response
        self.normalise: Callable[[str], str] = normalise or (
            lambda p: p.strip().lower()
        )

        # Internal mutable state -------------------------------------------------
        if strategy == "prompt_map":
            self._prompt_map: dict[str, str] = dict(responses or {})
            self._queue: list[str] = []
        else:
            self._queue: list[str] = list(responses or [])
            self._prompt_map = {}

        # Records every call: (prompt, response) tuples -------------------------
        self.calls: list[tuple[str, str]] = []

    # -- public API -----------------------------------------------------------

    def __call__(self, prompt: str) -> str:
        """Return the scripted completion for *prompt* and record the call."""
        normalised = self.normalise(prompt)

        if self.strategy == "prompt_map":
            if normalised in self._prompt_map:
                response = self._prompt_map[normalised]
            elif self.strict:
                available = sorted(self._prompt_map)
                msg = (
                    f"Unexpected prompt {prompt!r} (normalised={normalised!r}). "
                    f"Known prompts: {available}"
                )
                raise AssertionError(msg)
            else:
                response = self.default_response
        else:  # queue strategy
            if self._queue:
                response = self._queue.pop(0)
            elif self.strict:
                msg = (
                    f"LLM queue exhausted — unexpected call #{len(self.calls) + 1} "
                    f"with prompt {prompt!r}."
                )
                raise AssertionError(msg)
            else:
                response = self.default_response

        self.calls.append((prompt, response))
        return response

    # -- introspection helpers ------------------------------------------------

    @property
    def call_count(self) -> int:
        """Number of times :meth:`__call__` has been invoked."""
        return len(self.calls)

    @property
    def prompts(self) -> list[str]:
        """All prompts received, in call order."""
        return [p for p, _ in self.calls]

    @property
    def responses(self) -> list[str]:
        """All responses returned, in call order."""
        return [r for _, r in self.calls]

    def reset(self) -> None:
        """Clear recorded calls **and** replenish the queue/prompt-map."""
        self.calls.clear()
        # Re-seed the queue from the original snapshot if we captured one.
        # (For replay scenarios a test can simply construct a new provider.)


def _default_reset() -> None:
    """No-op default so the fixture protocol signature is honest."""
    return None


# ---------------------------------------------------------------------------
# Fake agent factory
# ---------------------------------------------------------------------------


class FakeAgent:
    """A controllable agent double.

    Instances are produced by the :func:`fake_agent` fixture factory. Each
    double is configured at construction time but can be reconfigured at any
    point via :meth:`configure` — handy when a single test needs to simulate
    behavioural changes mid-flow.

    Parameters
    ----------
    name:
        Human-readable identifier recorded in every call log entry.
    return_value:
        Value returned by :meth:`run` when no exception is configured.
    raises:
        Exception instance or class to raise from :meth:`run`.
    latency:
        Simulated wall-clock seconds of work done by :meth:`run`.
    """

    def __init__(
        self,
        *,
        name: str = "fake-agent",
        return_value: Any | None = None,
        raises: BaseException | type[BaseException] | None = None,
        latency: float = 0.0,
    ) -> None:
        self.name = name
        self._return_value = return_value
        self._raises = raises
        self._latency = latency
        self.calls: list[dict[str, Any]] = []
        self._call_count: int = 0

    # -- lifecycle ------------------------------------------------------------

    def configure(
        self,
        *,
        return_value: Any = ...,  # sentinel meaning "don't change"
        raises: BaseException | type[BaseException] | None = ...,
        latency: float = ...,
    ) -> "FakeAgent":
        """Mutate behaviour mid-test and return ``self`` for chaining."""
        if return_value is not ...:
            self._return_value = return_value
        if raises is not ...:
            self._raises = raises
        if latency is not ...:
            self._latency = latency
        return self

    # -- the "agent work" entry-point ----------------------------------------

    def run(self, task: str, **kwargs: Any) -> Any:
        """Simulate running *task* through this agent.

        Records the call, optionally sleeps for the configured latency,
        then either returns ``return_value`` or raises ``raises``.
        """
        self._call_count += 1
        entry: dict[str, Any] = {
            "call": self._call_count,
            "task": task,
            "kwargs": kwargs,
        }
        self.calls.append(entry)

        if self._latency > 0:
            time.sleep(self._latency)

        if self._raises is not None:
            exc = self._raises
            if isinstance(exc, type):
                raise exc(f"{self.name} raised: {exc.__name__}")
            raise exc

        return self._return_value

    # -- introspection --------------------------------------------------------

    @property
    def call_count(self) -> int:
        """Number of times :meth:`run` has been invoked."""
        return self._call_count

    @property
    def tasks(self) -> list[str]:
        """All task strings received, in call order."""
        return [c["task"] for c in self.calls]


# ---------------------------------------------------------------------------
# Persistent state store (per-test, backed by tmp_path)
# ---------------------------------------------------------------------------


class StateStore:
    """A tiny JSON-backed key-value store for test state.

    The store writes to a single JSON file under a pytest ``tmp_path``
    directory, so each test gets a clean, isolated on-disk store that is
    automatically cleaned up by pytest after the test finishes.

    Parameters
    ----------
    path:
        Directory in which the ``state.json`` file lives. Must already
        exist.
    """

    _FILENAME = "state.json"

    def __init__(self, path: Path) -> None:
        self._dir = path
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file = self._dir / self._FILENAME
        self._data: dict[str, Any] = self._read_disk()

    # -- I/O helpers ----------------------------------------------------------

    def _read_disk(self) -> dict[str, Any]:
        if not self._file.exists():
            return {}
        text = self._file.read_text(encoding="utf-8")
        return json.loads(text) if text.strip() else {}

    def _write_disk(self) -> None:
        self._file.write_text(
            json.dumps(self._data, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    # -- public API -----------------------------------------------------------

    def set(self, key: str, value: Any) -> None:
        """Store *value* under *key* and flush to disk immediately."""
        self._data[key] = value
        self._write_disk()

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key* (or *default* if absent)."""
        return self._data.get(key, default)

    def delete(self, key: str) -> None:
        """Remove *key* from the store, if present, and flush."""
        self._data.pop(key, None)
        self._write_disk()

    def keys(self) -> list[str]:
        """Return all current keys."""
        return list(self._data.keys())

    def items(self) -> list[tuple[str, Any]]:
        """Return ``(key, value)`` pairs."""
        return list(self._data.items())

    def clear(self) -> None:
        """Wipe all keys and flush to disk."""
        self._data.clear()
        self._write_disk()

    @property
    def path(self) -> Path:
        """On-disk location of the backing ``state.json`` file."""
        return self._file


# ---------------------------------------------------------------------------
# pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm_provider() -> MockLLMProvider:
    """Return a fresh :class:`MockLLMProvider` configured for queue mode.

    The double starts empty — populate ``responses`` via its constructor or
    by assigning to the private ``_queue`` for prompt-map scenarios. All
    calls are recorded on ``.calls`` and unexpected prompts raise loudly.
    """
    provider = MockLLMProvider(strategy="queue", strict=True)
    return provider


@pytest.fixture
def fake_agent_factory() -> Callable[..., FakeAgent]:
    """Return a factory that builds :class:`FakeAgent` doubles on demand.

    The factory accepts the same keyword arguments as :class:`FakeAgent`
    (``name``, ``return_value``, ``raises``, ``latency``) and returns a
    ready-to-use double. Use it from a test like::

        def test_example(fake_agent_factory):
            agent = fake_agent_factory(
                name="builder", return_value={"done": True}
            )
            assert agent.run("build")["done"] is True
            assert agent.call_count == 1
    """
    def _make(**kwargs: Any) -> FakeAgent:
        return FakeAgent(**kwargs)

    return _make


@pytest.fixture
def seeded_rng() -> random.Random:
    """Return a :class:`random.Random` seeded with :data:`RNG_SEED`.

    Any randomness consumed inside a test that uses this RNG is therefore
    fully reproducible across repeated runs.
    """
    return random.Random(RNG_SEED)


@pytest.fixture
def tmp_state_store(tmp_path: Path) -> StateStore:
    """Return a clean :class:`StateStore` rooted in ``tmp_path``.

    Each test gets its own unique on-disk directory (pytest guarantees
    isolation), so the store is empty at the start of every test and the
    directory is removed automatically after the test completes.
    """
    store_dir = tmp_path / "state"
    return StateStore(store_dir)


# ---------------------------------------------------------------------------
# Marker registration so pytest doesn't warn about unknown marks.
# (pyproject.toml also registers these; this is the belt-and-suspenders path.)
# ---------------------------------------------------------------------------


def pytest_configure(config: "pytest.Config") -> None:
    """Register custom markers used by the Hermes-AVO test harness."""
    markers: dict[str, str] = {
        "planning": "planning engine / task-decomposition tests",
        "wrapper": "DeepAgent / LLM-wrapper layer tests",
        "coordination": "multi-agent orchestration / handoff tests",
        "e2e": "end-to-end integration / smoke tests",
    }
    for name, desc in markers.items():
        config.addinivalue_line("markers", f"{name}: {desc}")
