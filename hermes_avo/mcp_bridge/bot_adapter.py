"""Bot-to-AVO adapter — maps each Hermes avatar to an MCP tool definition.

The bridge server exposes one MCP tool per avatar plus the aggregate
``avio_execute_goal`` tool. Mapping is fully data-driven so adding a new avatar
is a single-table edit (no code changes).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


def _goal_schema(default_budget: float = 0.5, default_timeout: int = 300) -> Dict[str, Any]:
    """Common {goal, budget, timeout} JSON schema for avatar tools."""
    return {
        "type": "object",
        "properties": {
            "goal": {"type": "string", "description": "Natural-language objective to execute."},
            "budget": {"type": "number", "description": "Resource budget (normalized 0.0–1.0).",
                       "default": default_budget},
            "timeout": {"type": "integer", "description": "Per-step timeout in seconds.",
                       "default": default_timeout},
        },
        "required": ["goal"],
    }


@dataclass(frozen=True)
class AvatarTool:
    """Static metadata for an AVO MCP tool derived from a Hermes avatar."""

    tool_name: str
    avatar: str
    description: str
    input_schema: Dict[str, Any]
    default_budget: float
    default_timeout: int


# ---------------------------------------------------------------------------
# The complete, canonical set of Hermes avatars bridged to AVO as MCP tools.
# 13 avatars total — @ceo plus 12 specialists (one tool per avatar).
# ---------------------------------------------------------------------------
AVATAR_SPEC: List[AvatarTool] = [
    AvatarTool(
        tool_name="avio_execute_goal",
        avatar="@ceo",
        description=(
            "Execute a high-level strategic goal through the Hermes-AVO autonomous "
            "loop. Orchestrates the @ceo perspective: decompose, delegate to "
            "specialist AVO agents, and report."
        ),
        input_schema=_goal_schema(0.5, 300),
        default_budget=0.5,
        default_timeout=300,
    ),
    AvatarTool(
        tool_name="avio_build_code",
        avatar="@agent-builder",
        description="Build or refactor code: scaffold repos, generate modules, create files.",
        input_schema=_goal_schema(0.5, 300),
        default_budget=0.5,
        default_timeout=300,
    ),
    AvatarTool(
        tool_name="avio_create_server",
        avatar="@mcp-specialist",
        description="Create, patch, or harden an MCP server (schema-validated tools, README, tests).",
        input_schema=_goal_schema(0.5, 300),
        default_budget=0.5,
        default_timeout=300,
    ),
    AvatarTool(
        tool_name="avio_test_suite",
        avatar="@qa-lead",
        description="Run/expand test suites, validate schema conformance, report coverage.",
        input_schema=_goal_schema(0.5, 300),
        default_budget=0.5,
        default_timeout=300,
    ),
    AvatarTool(
        tool_name="avio_design_system",
        avatar="@agent-architect",
        description="Design system architecture, module boundaries, data flow.",
        input_schema=_goal_schema(0.5, 300),
        default_budget=0.5,
        default_timeout=300,
    ),
    AvatarTool(
        tool_name="avio_frontend",
        avatar="@fullstack-dev",
        description="Implement frontend/UI or fullstack features end-to-end.",
        input_schema=_goal_schema(0.5, 300),
        default_budget=0.5,
        default_timeout=300,
    ),
    AvatarTool(
        tool_name="avio_research",
        avatar="@research-analyst",
        description="Research a topic, gather sources, summarize findings.",
        input_schema=_goal_schema(0.5, 300),
        default_budget=0.5,
        default_timeout=300,
    ),
    AvatarTool(
        tool_name="avio_run_pipeline",
        avatar="@devops-engineer",
        description="CI/CD pipeline, deployment, infra-as-code, environment provisioning.",
        input_schema=_goal_schema(0.5, 300),
        default_budget=0.5,
        default_timeout=300,
    ),
    AvatarTool(
        tool_name="avio_write_content",
        avatar="@writer",
        description="Generate documentation, copy, or creative written content.",
        input_schema=_goal_schema(0.3, 180),
        default_budget=0.3,
        default_timeout=180,
    ),
    AvatarTool(
        tool_name="avio_code_review",
        avatar="@reviewer",
        description="Review code for correctness, style, security, and maintainability.",
        input_schema=_goal_schema(0.4, 240),
        default_budget=0.4,
        default_timeout=240,
    ),
    AvatarTool(
        tool_name="avio_security_audit",
        avatar="@security-engineer",
        description="Security audit, threat modelling, secret scanning, compliance check.",
        input_schema=_goal_schema(0.6, 480),
        default_budget=0.6,
        default_timeout=480,
    ),
    AvatarTool(
        tool_name="avio_backend_service",
        avatar="@backend",
        description="Design and implement backend services, APIs, databases.",
        input_schema=_goal_schema(0.5, 300),
        default_budget=0.5,
        default_timeout=300,
    ),
    AvatarTool(
        tool_name="avio_product_plan",
        avatar="@product-manager",
        description="Product planning, roadmap, feature scoping, prioritisation.",
        input_schema=_goal_schema(0.4, 240),
        default_budget=0.4,
        default_timeout=240,
    ),
    AvatarTool(
        tool_name="avio_infra_strategy",
        avatar="@cto",
        description="Technology strategy, infra decisions, stack selection.",
        input_schema=_goal_schema(0.5, 300),
        default_budget=0.5,
        default_timeout=300,
    ),
]

# Lookup tables
_AVATAR_TOOLS: Dict[str, AvatarTool] = {t.avatar: t for t in AVATAR_SPEC}
TOOL_TO_AVATAR: Dict[str, str] = {t.tool_name: t.avatar for t in AVATAR_SPEC}

# Canonical ordered list + lookup tables consumed by the bridge server.
ALL_AVATARS: List[str] = [t.avatar for t in AVATAR_SPEC]
AVATAR_TOOLS: Dict[str, AvatarTool] = dict(_AVATAR_TOOLS)


def tool_for_avatar(avatar: str) -> Optional[AvatarTool]:
    """Return the AvatarTool mapping for a Hermes avatar handle."""
    return _AVATAR_TOOLS.get(avatar)


def avatar_names() -> List[str]:
    """All 13 canonical Hermes avatar handles (@ceo + 12 specialists)."""
    return list(ALL_AVATARS)


def tool_schema(tool_name: str) -> Dict[str, Any]:
    """Extract the JSON schema for a single tool by name."""
    avatar = TOOL_TO_AVATAR.get(tool_name)
    if avatar is None:
        raise KeyError(tool_name)
    return _AVATAR_TOOLS[avatar].input_schema
