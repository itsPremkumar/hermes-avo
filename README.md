# Hermes AVO

NVIDIA AVO-style planning engine with a DeepAgent wrapper for autonomous agent
orchestration.  Built on LangGraph + LangChain + OpenAI, offline-first.

## Project layout

```
hermes-avo/
└── hermes_avo/
    ├── __init__.py            # package root, re-exports core types
    ├── core/
    │   ├── __init__.py        # core public API
    │   └── types.py           # shared contracts (models + Protocol seams)
    ├── planning/              # HTN planner, goal decomposer, plan validator
    ├── agents/                # AVODeepAgent + lifecycle state machine
    └── orchestrator/          # multi-agent orchestrator + task router
```

The `hermes_avo/core/types.py` module is the **contract layer** — the three
domain subpackages (`planning/`, `agents/`, `orchestrator/`) and their tests
all import from here, so each phase builds against a stable seam without
circular dependencies.

## Shared contracts

Defined in `hermes_avo/core/types.py`:

| Symbol              | Kind        | Purpose                                         |
|---------------------|-------------|-------------------------------------------------|
| `AgentState`        | Enum        | PENDING → PLANNING → EXECUTING → OBSERVING → RECOVERING → COMPLETE/FAILED |
| `Goal`              | pydantic    | A high-level objective to decompose             |
| `SubTask`           | pydantic    | A decomposed step with explicit `depends_on` edges |
| `Plan`              | pydantic    | ordered sub-task list; `ordered_subtasks()` topological sort |
| `Observation`       | pydantic    | outcome emitted after a sub-task executes       |
| `ModelConfig`       | pydantic    | OpenAI / LLM configuration                      |
| `ToolSpec`          | pydantic    | minimal tool descriptor (name, params, MCP srv) |
| `PlannerProtocol`   | Protocol    | `decompose(query) -> Plan`                      |
| `CriticProtocol`    | Protocol    | `evaluate(result) -> bool`                      |
| `TraceSink`         | Protocol    | `emit(step, result)`                            |

## Quick start

```bash
# editable install (pulls langgraph, langchain-core, openai, pydantic)
pip install -e ".[dev]"

# import the contract layer
python -c "from hermes_avo.core.types import Plan, SubTask, PlannerProtocol; print('ok')"

# run the built-in self-test (real asserts, no network)
python hermes_avo/self_test.py self-test

# run the test suite
pytest -q
```

## Configuration

Copy `.env.example` to `.env` and fill in your provider keys.  Tests never
make live network calls — inject mocks against `PlannerProtocol` / `CriticProtocol`.

## License

MIT — see [LICENSE](LICENSE).
