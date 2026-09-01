"""Pytest unit tests for the hermes_avo scaffold contract layer.

These are pure-data / pure-logic tests against `hermes_avo.core.types`.  No
network access is required — the LLM seam is exercised via Protocol stubs.
"""
from __future__ import annotations

import pytest

from hermes_avo.core.types import (
    AgentState,
    CriticProtocol,
    Goal,
    ModelConfig,
    Observation,
    Plan,
    PlannerProtocol,
    SubTask,
    ToolSpec,
    TraceSink,
)


# ---------------------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------------------
def test_agent_state_has_all_lifecycle_states():
    expected = {
        "PENDING", "PLANNING", "EXECUTING", "OBSERVING",
        "RECOVERING", "COMPLETE", "FAILED",
    }
    assert {s.value for s in AgentState} == expected


def test_agent_state_is_str_enum():
    assert AgentState.PENDING.value == "PENDING"
    assert isinstance(AgentState.COMPLETE, AgentState)


# ---------------------------------------------------------------------------
# ToolSpec
# ---------------------------------------------------------------------------
def test_tool_spec_defaults():
    ts = ToolSpec(name="grep", description="search text")
    assert ts.name == "grep"
    assert ts.mcp_server is None
    assert ts.parameters == {}


def test_tool_spec_with_mcp_server():
    ts = ToolSpec(name="read", mcp_server="fs-mcp", parameters={"path": "str"})
    assert ts.mcp_server == "fs-mcp"
    assert ts.parameters == {"path": "str"}


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------
def test_model_config_defaults():
    mc = ModelConfig()
    assert mc.model == "gpt-4o-mini"
    assert mc.api_key is None
    assert mc.temperature == 0.7
    assert mc.max_tokens == 4000


def test_model_config_overrides():
    mc = ModelConfig(model="gpt-4o", temperature=0.1, max_tokens=8192)
    assert mc.model == "gpt-4o"
    assert mc.temperature == 0.1
    assert mc.max_tokens == 8192


# ---------------------------------------------------------------------------
# Goal
# ---------------------------------------------------------------------------
def test_goal_round_trip():
    g = Goal(
        goal_id="g1",
        description="Write a blog post",
        constraints=["length<=1000"],
        required_tools=["write", "search"],
        priority=9,
    )
    assert g.goal_id == "g1"
    assert g.constraints == ["length<=1000"]
    assert g.required_tools == ["write", "search"]
    assert g.priority == 9
    assert g.metadata == {}


# ---------------------------------------------------------------------------
# SubTask
# ---------------------------------------------------------------------------
def test_subtask_dependency_edges():
    a = SubTask(task_id="a", goal_id="g1", description="step a")
    b = SubTask(task_id="b", goal_id="g1", description="step b", depends_on=["a"])
    c = SubTask(task_id="c", goal_id="g1", description="step c",
                depends_on=["a", "b"])
    assert a.depends_on == []
    assert b.depends_on == ["a"]
    assert c.depends_on == ["a", "b"]


def test_subtask_default_fields():
    s = SubTask(task_id="x", goal_id="g1", description="do thing")
    assert s.depends_on == []
    assert s.tool_calls == []
    assert s.estimated_tokens == 0


# ---------------------------------------------------------------------------
# Plan topological ordering
# ---------------------------------------------------------------------------
def _chain_plan() -> Plan:
    a = SubTask(task_id="a", goal_id="g1", description="alpha")
    b = SubTask(task_id="b", goal_id="g1", description="beta", depends_on=["a"])
    c = SubTask(task_id="c", goal_id="g1", description="gamma", depends_on=["a", "b"])
    return Plan(plan_id="p1", goal_id="g1", subtasks=[c, a, b])


def test_plan_ordered_subtasks_topological():
    plan = _chain_plan()
    ids = [t.task_id for t in plan.ordered_subtasks()]
    assert ids == ["a", "b", "c"]


def test_plan_ordered_subtasks_idempotent():
    plan = _chain_plan()
    first = [t.task_id for t in plan.ordered_subtasks()]
    second = [t.task_id for t in plan.ordered_subtasks()]
    assert first == second


def test_plan_cycle_detection_raises():
    a = SubTask(task_id="a", goal_id="g1", description="a", depends_on=["c"])
    b = SubTask(task_id="b", goal_id="g1", description="b", depends_on=["a"])
    c = SubTask(task_id="c", goal_id="g1", description="c", depends_on=["b"])
    plan = Plan(plan_id="cyc", goal_id="g1", subtasks=[a, b, c])
    with pytest.raises(ValueError, match="Cycle detected"):
        plan.ordered_subtasks()


def test_plan_dangling_dependency_raises():
    s = SubTask(task_id="s", goal_id="g1", description="s", depends_on=["ghost"])
    plan = Plan(plan_id="p2", goal_id="g1", subtasks=[s])
    with pytest.raises(ValueError, match="unknown task"):
        plan.ordered_subtasks()


def test_plan_empty_subtasks():
    plan = Plan(plan_id="p3", goal_id="g1")
    assert plan.ordered_subtasks() == []


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------
def test_observation_success():
    obs = Observation(task_id="a", plan_id="p1", success=True,
                      output="done", token_cost=10)
    assert obs.success is True
    assert obs.output == "done"
    assert obs.token_cost == 10
    assert obs.error is None


def test_observation_failure():
    obs = Observation(task_id="a", plan_id="p1", success=False, error="boom")
    assert obs.success is False
    assert obs.error == "boom"


# ---------------------------------------------------------------------------
# Protocol conformance (seams for parallel Phase 1 builds)
# ---------------------------------------------------------------------------
def test_planner_protocol_runtime_checkable():
    class StubPlanner:
        def decompose(self, query: str | Goal) -> Plan:  # type: ignore[override]
            return Plan(plan_id="stub", goal_id="g1",
                        subtasks=[SubTask(task_id="s1", goal_id="g1",
                                          description="stub")])

    assert isinstance(StubPlanner(), PlannerProtocol)


def test_critic_protocol_runtime_checkable():
    class StubCritic:
        def evaluate(self, result: Observation) -> bool:
            return result.success

    critic = StubCritic()
    assert isinstance(critic, CriticProtocol)
    obs = Observation(task_id="t", plan_id="p", success=True)
    assert critic.evaluate(obs) is True


def test_trace_sink_protocol_runtime_checkable():
    class StubSink:
        def emit(self, step: SubTask, result: Observation) -> None:
            self.last = (step.task_id, result.success)

    sink = StubSink()
    assert isinstance(sink, TraceSink)
    sink.emit(SubTask(task_id="t", goal_id="g1", description="d"),
              Observation(task_id="t", plan_id="p", success=True))
    assert sink.last == ("t", True)


def test_planner_protocol_accepts_goal_or_str():
    """The seam signature must accept both str and Goal (per child task spec)."""
    class StubPlanner:
        def decompose(self, query: str | Goal) -> Plan:  # type: ignore[override]
            assert isinstance(query, (str, Goal))
            return Plan(plan_id="stub", goal_id="g1",
                        subtasks=[])

    p = StubPlanner()
    assert isinstance(p.decompose("hello"), Plan)
    assert isinstance(p.decompose(Goal(goal_id="g1", description="hi")), Plan)
