# Hermes-AVO Test Harness

This directory contains the pytest-based QA harness for the Hermes-AVO
project, mirroring the AgentWatch QA harness architecture.

## Layout

The test suite is organised into category subpackages. Each subpackage has
an `__init__.py` (documenting its scope) and will hold test modules relevant
to that category. Every test module is decorated with a category marker so
tests can be selected by focus area via `pytest -m <marker>`.

```
tests/
├── __init__.py
├── README.md                 <-- this file
├── planning/
│   ├── __init__.py
│   └── test_placeholder.py   # placeholder, @pytest.mark.planning
├── wrapper/
│   ├── __init__.py
│   └── test_placeholder.py   # placeholder, @pytest.mark.wrapper
├── coordination/
│   ├── __init__.py
│   └── test_placeholder.py   # placeholder, @pytest.mark.coordination
└── e2e/
    ├── __init__.py
    └── test_placeholder.py   # placeholder, @pytest.mark.e2e
```

## Category subpackages

- **planning/** — Tests that exercise the planning / task-decomposition
  subsystem (step breakdown, goal parsing, deterministic planning with mock
  LLM responses and fake agent doubles).

- **wrapper/** — Tests for the LLM-wrapper layer: prompt templating,
  tool-call formatting, tool-result parsing, and response handling against
  scripted mock providers.

- **coordination/** — Tests for orchestration flows, agent-to-agent handoffs,
  and consensus logic using controllable fake agent doubles.

- **e2e/** — Light-weight integration ("smoke") tests that stitch together
  multiple layers of the harness and assert on overall behaviour.

## Category markers

The following custom markers are used (registered in `pyproject.toml` by the
config task):

| Marker          | Subpackage    |
|-----------------|---------------|
| `planning`      | planning/     |
| `wrapper`       | wrapper/      |
| `coordination`  | coordination/ |
| `e2e`           | e2e/          |

## Running tests

From the repository root:

```bash
# Collect only (no execution)
pytest --collect-only

# Run all tests
pytest

# Run a single category
pytest -m planning
pytest -m wrapper
pytest -m coordination
pytest -m e2e
```

## Notes

- The placeholder tests are trivially passing stubs that ensure every
  category subpackage is collectable. Replace them with real tests as
  features land.
- Fixtures and shared helpers will be added under `tests/conftest.py` and
  `tests/_fixtures/` in a follow-up task (not this one).
