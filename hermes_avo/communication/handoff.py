"""Formal handoff protocol between AVO agents.

A handoff is a structured envelope that preserves context (task state, resource
budgets, prior observations) when control transfers from one avatar to the
next — e.g. @ceo decomposing a strategic goal and handing a sub-task to
@agent-builder. Handoffs are published on the ``planning`` topic and are
queryable via :class:`HandoffManager`.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from hermes_avo.communication.message_bus import Message, MessageBus, TOPICS


@dataclass
class ResourceBudget:
    """Remaining resources for a handed-off task."""
    budget: float = 0.0
    used: float = 0.0
    timeout: int = 300

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskState:
    """Snapshot of task progress carried across a handoff."""
    task_id: str
    status: str
    step: int
    progress: float  # 0.0–1.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HandoffContext:
    """Full context envelope for a single agent -> agent handoff."""

    trace_id: str
    from_avatar: str
    to_avatar: str
    task_id: str
    goal: str
    task_state: TaskState
    budget: ResourceBudget
    prior_observations: List[Dict[str, Any]] = field(default_factory=list)
    instructions: str = ""
    ts: float = field(default_factory=time.time)
    handoff_id: str = field(default_factory=lambda: f"h-{uuid.uuid4().hex[:12]}")

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "handoff_id": self.handoff_id,
            "trace_id": self.trace_id,
            "from_avatar": self.from_avatar,
            "to_avatar": self.to_avatar,
            "task_id": self.task_id,
            "goal": self.goal,
            "task_state": self.task_state.to_dict(),
            "budget": self.budget.to_dict(),
            "prior_observations": self.prior_observations,
            "instructions": self.instructions,
            "ts": self.ts,
        }
        return d


class HandoffManager:
    """Create, persist, and retrieve handoffs via the message bus."""

    def __init__(self, bus: Optional[MessageBus] = None) -> None:
        self.bus = bus or MessageBus()

    def create_handoff(
        self,
        trace_id: str,
        from_avatar: str,
        to_avatar: str,
        goal: str,
        budget: ResourceBudget,
        task_state: Optional[TaskState] = None,
        prior_observations: Optional[List[Dict[str, Any]]] = None,
        instructions: str = "",
    ) -> HandoffContext:
        """Create a HandoffContext and publish it on the planning topic."""
        ctx = HandoffContext(
            trace_id=trace_id,
            from_avatar=from_avatar,
            to_avatar=to_avatar,
            task_id=f"task-{uuid.uuid4().hex[:8]}",
            goal=goal,
            task_state=task_state or TaskState(task_id=goal[:40], status="handoff", step=0, progress=0.0),
            budget=budget,
            prior_observations=prior_observations or [],
            instructions=instructions,
        )
        msg = Message(
            topic="planning",
            sender=from_avatar,
            recipient=to_avatar,
            payload=ctx.to_dict(),
            trace_id=trace_id,
        )
        self.bus.publish(msg)
        return ctx

    def accept_handoff(self, ctx: HandoffContext) -> Message:
        """Acknowledge receipt of a handoff on the execution topic.

        Returns the published ``Message`` for inspection by callers/tests.
        """
        msg = Message(
            topic="execution",
            sender=ctx.to_avatar,
            recipient=ctx.from_avatar,
            payload={"handoff_id": ctx.handoff_id, "ack": True, "goal": ctx.goal},
            trace_id=ctx.trace_id,
        )
        self.bus.publish(msg)
        return msg

    def get_handoffs_for_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        """Return all planning-topic messages for a trace (handoffs)."""
        return self.bus.get_messages("planning", trace_id=trace_id)

    def get_handoffs_for_avatar(self, avatar: str, trace_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return handoffs addressed to ``avatar``."""
        msgs = self.bus.get_messages("planning", trace_id=trace_id) if trace_id else self.bus.get_messages("planning")
        return [m for m in msgs if m.get("payload", {}).get("to_avatar") == avatar]
