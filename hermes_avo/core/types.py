"""Hermes AVO — shared core types, state enum, and seam Protocols.

This module is the *contract layer* for the AVO planning engine.  The three
domain subpackages (``planning/``, ``agents/``, ``orchestrator/``) and their
tests all import from here, so that the next three build tasks can proceed in
parallel without circular dependencies:

    * planning  -> implements ``PlannerProtocol``
    * agents     -> implements ``CriticProtocol`` + ``TraceSink``
    * orchestrator -> consumes all three via duck-typed protocols

The models below are intentionally minimal: they are the data shapes that flow
between planner, agent, and orchestrator.  Behaviour lives in the subpackages.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Agent lifecycle
# ---------------------------------------------------------------------------
class AgentState(str, Enum):
    """Finite states for an AVO agent run.

    Lifecycle: PENDING -> PLANNING -> EXECUTING -> OBSERVING ->
               RECOVERING -> COMPLETE | FAILED
    """

    PENDING = "PENDING"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    OBSERVING = "OBSERVING"
    RECOVERING = "RECOVERING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


# ---------------------------------------------------------------------------
# Domain models (pydantic v2)
# ---------------------------------------------------------------------------
try:
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - pydantic is a hard dependency
    BaseModel = object  # type: ignore[misc,assignment]
    def Field(**_: Any) -> Any:  # type: ignore[misc]
        return None


class ToolSpec(BaseModel):
    """Minimal descriptor of a tool available to an agent."""

    name: str
    description: str = ""
    parameters: Dict[str, Any] = Field(default_factory=dict)
    mcp_server: Optional[str] = None


class ModelConfig(BaseModel):
    """OpenAI / LLM configuration consumed by agents."""

    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = 4000
    top_p: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Goal(BaseModel):
    """A high-level objective handed to the planner."""

    goal_id: str
    description: str
    constraints: List[str] = Field(default_factory=list)
    required_tools: List[str] = Field(default_factory=list)
    priority: int = 5  # 1..10
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SubTask(BaseModel):
    """A decomposed sub-task inside a plan.

    ``depends_on`` holds *other SubTask ids* that must complete before this one
    (explicit dependency edges), enabling topological ordering by the planner
    and the orchestrator's conflict resolver.
    """

    task_id: str
    goal_id: str
    description: str
    depends_on: List[str] = Field(default_factory=list)
    tool_calls: List[str] = Field(default_factory=list)  # tool names required
    estimated_tokens: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    """A full plan: a list of sub-tasks with explicit dependency edges."""

    plan_id: str
    goal_id: str
    subtasks: List[SubTask] = Field(default_factory=list)
    token_budget: int = 8000
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def ordered_subtasks(self) -> List[SubTask]:
        """Return subtasks in dependency order (topological sort).

        Raises ``ValueError`` if a cycle is detected so callers can surface a
        clear error instead of silently looping.
        """
        by_id: Dict[str, SubTask] = {t.task_id: t for t in self.subtasks}
        ordered: List[SubTask] = []
        seen: set[str] = set()
        visiting: set[str] = set()

        def visit(task: SubTask) -> None:
            if task.task_id in seen:
                return
            if task.task_id in visiting:
                raise ValueError(
                    f"Cycle detected in plan {self.plan_id} at task "
                    f"{task.task_id}"
                )
            visiting.add(task.task_id)
            for dep in task.depends_on:
                if dep in by_id:
                    visit(by_id[dep])
                else:
                    # Unknown dependency — surface as a dangling edge error
                    # rather than silently dropping it.
                    raise ValueError(
                        f"Task {task.task_id} depends on unknown task "
                        f"{dep!r} in plan {self.plan_id}"
                    )
            visiting.discard(task.task_id)
            seen.add(task.task_id)
            ordered.append(task)

        for t in self.subtasks:
            visit(t)
        return ordered


class Observation(BaseModel):
    """An observation emitted after a sub-task executes."""

    task_id: str
    plan_id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    token_cost: int = 0
    timestamp: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Seam protocols — the interfaces the subpackages implement / consume
# ---------------------------------------------------------------------------
@runtime_checkable
class PlannerProtocol(Protocol):
    """Planner seam: decompose a query into a ``Plan``.

    Implemented by ``hermes_avo.planning.planner.HTNPlanner``.  Injecting this
    protocol keeps the agent testable without live LLM calls.

    Accepts either a raw query string or a fully-formed ``Goal``.
    """

    def decompose(self, query: "str | Goal") -> Plan:
        ...


@runtime_checkable
class CriticProtocol(Protocol):
    """Critic seam: judge whether an observation/result is acceptable."""

    def evaluate(self, result: Observation) -> bool:
        ...


@runtime_checkable
class TraceSink(Protocol):
    """Trace sink seam: persist/emit a trace step (e.g. to an MCP server)."""

    def emit(self, step: SubTask, result: Observation) -> None:
        ...


# Re-export so child tasks can `from hermes_avo.core.types import *` cleanly.
__all__ = [
    "AgentState",
    "Goal",
    "SubTask",
    "Plan",
    "Observation",
    "ModelConfig",
    "ToolSpec",
    "PlannerProtocol",
    "CriticProtocol",
    "TraceSink",
]
