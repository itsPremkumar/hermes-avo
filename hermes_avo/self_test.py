#!/usr/bin/env python3
"""Self-test for the `hermes_avo` scaffold package.

Run:  python hermes_avo/self_test.py self-test

This exercises the *shared contracts* in `hermes_avo/core/types.py` with real
assertions and no network access.  It is the verification gate for the scaffold
task (t_d9ed86aa) and is auto-discovered by the QA harness because the
literal ``'self-test'`` appears in this file.
"""
from __future__ import annotations

import sys
from typing import runtime_checkable

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


def _fail(msg: str) -> None:
    print(f"  FAIL: {msg}", file=sys.stderr)
    raise AssertionError(msg)


def _ok(msg: str) -> None:
    print(f"  ok:   {msg}")


def run_self_test() -> int:
    print("hermes-avo self-test")
    print("=" * 60)

    # --- AgentState enum -----------------------------------------------
    print("\n[1] AgentState values")
    expected = {
        "PENDING", "PLANNING", "EXECUTING", "OBSERVING",
        "RECOVERING", "COMPLETE", "FAILED",
    }
    actual = {s.value for s in AgentState}
    assert actual == expected, f"got {actual}"
    _ok(f"7 lifecycle states present ({len(actual)})")

    # --- ToolSpec -------------------------------------------------------
    print("\n[2] ToolSpec model")
    ts = ToolSpec(name="read", description="read a file",
                  parameters={"path": "str"}, mcp_server="fs-mcp")
    assert ts.name == "read"
    assert ts.mcp_server == "fs-mcp"
    assert ts.parameters == {"path": "str"}
    _ok("ToolSpec round-trips name/desc/params/server")

    # --- ModelConfig ----------------------------------------------------
    print("\n[3] ModelConfig model")
    mc = ModelConfig(model="gpt-4o-mini", temperature=0.3, max_tokens=2048)
    assert mc.model == "gpt-4o-mini"
    assert mc.temperature == 0.3
    assert mc.max_tokens == 2048
    assert mc.api_key is None  # default
    _ok("ModelConfig defaults + overrides")

    # --- Goal -----------------------------------------------------------
    print("\n[4] Goal model")
    g = Goal(goal_id="g1", description="Build a treehouse",
             constraints=["budget<=5000"], required_tools=["saw", "drill"],
             priority=8)
    assert g.goal_id == "g1"
    assert len(g.constraints) == 1
    assert "saw" in g.required_tools
    assert g.priority == 8
    _ok("Goal with constraints/tools/priority")

    # --- SubTask with dependency edges ---------------------------------
    print("\n[5] SubTask + dependency edges")
    t1 = SubTask(task_id="t1", goal_id="g1", description="buy lumber")
    t2 = SubTask(task_id="t2", goal_id="g1", description="cut lumber",
                 depends_on=["t1"], tool_calls=["saw"])
    t3 = SubTask(task_id="t3", goal_id="g1", description="assemble",
                 depends_on=["t1", "t2"])
    assert t1.depends_on == []
    assert t2.depends_on == ["t1"]
    assert t3.depends_on == ["t1", "t2"]
    assert t2.tool_calls == ["saw"]
    _ok("SubTask edges: t1->[], t2->['t1'], t3->['t1','t2']")

    # --- Plan.ordered_subtasks() topological sort ----------------------
    print("\n[6] Plan.ordered_subtasks() topo sort")
    plan = Plan(plan_id="p1", goal_id="g1",
                subtasks=[t3, t1, t2])  # deliberately out of order
    ordered = plan.ordered_subtasks()
    ids = [t.task_id for t in ordered]
    assert ids == ["t1", "t2", "t3"], f"order was {ids}"
    _ok(f"topo order correct: {ids}")

    # --- Plan cycle detection ------------------------------------------
    print("\n[7] Plan cycle detection")
    cyc1 = SubTask(task_id="c1", goal_id="g1", description="a", depends_on=["c3"])
    cyc2 = SubTask(task_id="c2", goal_id="g1", description="b", depends_on=["c1"])
    cyc3 = SubTask(task_id="c3", goal_id="g1", description="c", depends_on=["c2"])
    cyc_plan = Plan(plan_id="p2", goal_id="g1", subtasks=[cyc1, cyc2, cyc3])
    try:
        cyc_plan.ordered_subtasks()
        _fail("expected ValueError for cyclic plan")
    except ValueError as exc:
        assert "Cycle detected" in str(exc)
        _ok(f"ValueError raised on cycle: {exc}")

    # --- Plan dangling dependency ---------------------------------------
    print("\n[8] Plan dangling dependency")
    dangle = SubTask(task_id="d1", goal_id="g1", description="x",
                     depends_on=["nonexistent"])
    dplan = Plan(plan_id="p3", goal_id="g1", subtasks=[dangle])
    try:
        dplan.ordered_subtasks()
        _fail("expected ValueError for dangling dep")
    except ValueError as exc:
        assert "unknown task" in str(exc).lower()
        _ok(f"ValueError raised on dangling dep: {exc}")

    # --- Observation ----------------------------------------------------
    print("\n[9] Observation model")
    obs = Observation(task_id="t1", plan_id="p1", success=True,
                      output="lumber acquired", token_cost=42, timestamp=1.5)
    assert obs.success is True
    assert obs.token_cost == 42
    assert obs.error is None
    _ok("Observation records success/output/tokens")

    obs_fail = Observation(task_id="t2", plan_id="p1", success=False,
                           error="saw missing")
    assert obs_fail.success is False
    assert obs_fail.error == "saw missing"
    _ok("Observation failure path")

    # --- Protocol runtime checkability ----------------------------------
    print("\n[10] Seam protocols are runtime-checkable")
    # A concrete planner stub satisfying PlannerProtocol.
    class StubPlanner:
        def decompose(self, query):  # noqa: D401
            return Plan(plan_id="stub", goal_id="g1",
                        subtasks=[SubTask(task_id="s1", goal_id="g1",
                                          description="stub step")])

    sp = StubPlanner()
    assert isinstance(sp, PlannerProtocol), "StubPlanner must satisfy PlannerProtocol"
    plan = sp.decompose("anything")
    assert isinstance(plan, Plan)
    _ok("PlannerProtocol.isinstance(StubPlanner) == True")

    class StubCritic:
        def evaluate(self, result: Observation) -> bool:
            return result.success

    sc = StubCritic()
    assert isinstance(sc, CriticProtocol)
    assert sc.evaluate(obs) is True
    assert sc.evaluate(obs_fail) is False
    _ok("CriticProtocol evaluates Observation correctly")

    class StubSink:
        def __init__(self):
            self.emitted = []

        def emit(self, step: SubTask, result: Observation) -> None:
            self.emitted.append((step.task_id, result.success))

    ss = StubSink()
    assert isinstance(ss, TraceSink)
    ss.emit(t1, obs)
    assert ss.emitted == [("t1", True)]
    _ok("TraceSink.emit() records step+result")

    # --- import surface -------------------------------------------------
    print("\n[11] import surface")
    import hermes_avo
    assert hermes_avo.__version__ == "0.1.0"
    assert hasattr(hermes_avo, "AgentState")
    assert hasattr(hermes_avo, "Plan")
    _ok("hermes_avo.__version__ = '0.1.0' and re-exports present")

    print("\n" + "=" * 60)
    print("ALL SELF-TEST ASSERTIONS PASSED (11 groups)")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] != "self-test":
        print("usage: python hermes_avo/self_test.py self-test")
        return 2
    return run_self_test()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
