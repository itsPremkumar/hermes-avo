"""Example tests that exercise every fixture defined in ``tests/conftest.py``.

These double as scratch tests: they verify that the shared fixtures are
collectable, behave deterministically, and never touch the network. They also
serve as usage templates that real feature tests can copy/adapt.
"""

from __future__ import annotations

import random

import pytest

from tests.conftest import (
    MockLLMProvider,
    FakeAgent,
    StateStore,
)


# ------------------------------------------------------------------
# Queue-strategy mock_llm_provider
# ------------------------------------------------------------------


def test_mock_llm_provider_queue_returns_in_order(
    mock_llm_provider: MockLLMProvider,
) -> None:
    """The queue strategy must drain canned responses in FIFO order."""
    mock_llm_provider._queue = ["first", "second"]

    assert mock_llm_provider("prompt A") == "first"
    assert mock_llm_provider("prompt B") == "second"
    # The original prompt text is recorded verbatim.
    assert mock_llm_provider.prompts == ["prompt A", "prompt B"]
    assert mock_llm_provider.call_count == 2


def test_mock_llm_provider_fails_on_unexpected_exhaustion(
    mock_llm_provider: MockLLMProvider,
) -> None:
    """An unexpected call when the queue is empty must raise loudly."""
    with pytest.raises(AssertionError, match="queue exhausted"):
        mock_llm_provider("anything")


def test_mock_llm_provider_prompt_map_returns_canned(
    mock_llm_provider: MockLLMProvider,
) -> None:
    """The prompt-map strategy must key on the normalised prompt string."""
    provider = MockLLMProvider(
        responses={"hello world": "greeting-back"},
        strategy="prompt_map",
    )
    # Whitespace/case should not affect the lookup.
    assert provider("  Hello World  ") == "greeting-back"
    assert provider.call_count == 1


def test_mock_llm_provider_prompt_map_strict_unknown(
    mock_llm_provider: MockLLMProvider,
) -> None:
    """In strict mode an unknown prompt under prompt-map must raise."""
    provider = MockLLMProvider(responses={"a": "x"}, strategy="prompt_map")
    with pytest.raises(AssertionError, match="Unexpected prompt"):
        provider("b")


# ------------------------------------------------------------------
# fake_agent factory
# ------------------------------------------------------------------


def test_fake_agent_returns_value(fake_agent_factory: object) -> None:
    """A fake agent must return its configured ``return_value``."""
    agent = fake_agent_factory(
        name="builder", return_value={"ok": True}
    )
    result = agent.run("build")
    assert result == {"ok": True}
    assert agent.call_count == 1
    assert agent.tasks == ["build"]


def test_fake_agent_records_call_args(fake_agent_factory: object) -> None:
    """Keyword arguments passed to ``run`` must be recorded verbatim."""
    agent = fake_agent_factory(name="planner")
    agent.run("plan", priority="high", notes=["a", "b"])
    recorded = agent.calls[0]
    assert recorded["kwargs"]["priority"] == "high"
    assert recorded["kwargs"]["notes"] == ["a", "b"]


def test_fake_agent_raises_configured_exception(
    fake_agent_factory: object,
) -> None:
    """The ``raises`` field must propagate an instance or a class."""
    agent = fake_agent_factory(name="failer", raises=ValueError("boom"))
    with pytest.raises(ValueError, match="boom"):
        agent.run("task")
    # The call is still recorded even when it raises.
    assert agent.call_count == 1


def test_fake_agent_reconfigure_mid_test(fake_agent_factory: object) -> None:
    """``configure`` must allow behaviour changes while a test runs."""
    agent = fake_agent_factory(name="chameleon", return_value="v1")
    assert agent.run("a") == "v1"
    agent.configure(return_value="v2", raises=KeyError("nope"))
    with pytest.raises(KeyError, match="nope"):
        agent.run("b")
    assert agent.call_count == 2


# ------------------------------------------------------------------
# seeded_rng
# ------------------------------------------------------------------


def test_seeded_rng_is_reproducible(
    seeded_rng: random.Random,
) -> None:
    """Two runs seeded identically must produce identical sequences."""
    a = [seeded_rng.randint(0, 1_000_000) for _ in range(10)]
    # Build a second, independent RNG from the same constant seed.
    replay = random.Random(0xC0FFEE)
    b = [replay.randint(0, 1_000_000) for _ in range(10)]
    assert a == b


def test_seeded_rng_shuffle_deterministic(seeded_rng: random.Random) -> None:
    """``shuffle`` must be reproducible across identical seeds."""
    base = ["x", "y", "z", "1", "2", "3"]
    order_a = base[:]
    seeded_rng.shuffle(order_a)
    order_b = base[:]
    random.Random(0xC0FFEE).shuffle(order_b)
    assert order_a == order_b


# ------------------------------------------------------------------
# tmp_state_store
# ------------------------------------------------------------------


def test_tmp_state_store_persists_to_disk(tmp_state_store: StateStore) -> None:
    """A value written then re-read from disk must round-trip exactly."""
    tmp_state_store.set("answer", 42)
    # The file must really exist on disk.
    assert tmp_state_store.path.exists()

    # Re-open the store from the same file and confirm persistence.
    reopened = StateStore(tmp_state_store.path.parent)
    assert reopened.get("answer") == 42


def test_tmp_state_store_delete_and_clear(tmp_state_store: StateStore) -> None:
    """``delete`` and ``clear`` must mutate the backing store."""
    tmp_state_store.set("a", 1)
    tmp_state_store.set("b", 2)
    tmp_state_store.delete("a")
    assert tmp_state_store.get("a") is None
    assert tmp_state_store.get("b") == 2

    tmp_state_store.clear()
    assert tmp_state_store.keys() == []
    assert tmp_state_store.path.exists()  # cleared file still exists
