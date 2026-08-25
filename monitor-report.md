# Hermes AVO — Pipeline Health Monitor

> **Last scan**: Phase 5 CI/CD initialization
> **Repo**: itsPremkumar/hermes-avo
> **Status**: INITIAL SETUP COMPLETE

## Pipeline Status

| Phase | Owner | CI Job | Status |
|-------|-------|--------|--------|
| Phase 1 | @agent-builder | `phase-1` | Ready (Phase 1 scaffold committed b9be934) |
| Phase 2 | @mcp-specialist | `phase-2` | Ready (scaffold pending) |
| Phase 3 | @agent-architect | `phase-3` | Ready (scaffold pending) |
| Phase 4 | @qa-lead | `phase-4` | Ready (scaffold pending) |

## Circuit Breakers

| Guard | Threshold | Enforcement |
|-------|-----------|-------------|
| Build cost | <$ 0.50 | `concurrency` group cancels stale runs |
| Step budget | < 1000 steps | `timeout-minutes: 10` per phase job |
| Call timeout | < 30s/call | `CALL_TIMEOUT_S` env var (Phase 3 tests) |

## Coverage Gates

| Phase | Required | Enforced in |
|-------|----------|-------------|
| Phase 1 | >= 95% | ci.yml `phase-1` job |
| Phase 2 | >= 90% | ci.yml `phase-2` job |
| Phase 3 | >= 80% | ci.yml `phase-3` job |
| Phase 4 | >= 90% | ci.yml `phase-4` job |

## Auto-Healing

- **Bot**: @hermes-avo-ci (registered via `.github/bot-config.yml`)
- **Failure routing**:
  - Phase 1 fail → @agent-builder
  - Phase 2 fail → @mcp-specialist
  - Phase 3 fail → @agent-architect
  - Phase 4 fail → @qa-lead
- **Mechanism**: `auto-assign-on-failure` job posts PR comment with avatar tag.

## Monitoring Integration

- **24/7 pipeline health**: wired into @qa-lead's E2E harness (t_836e2abb, t_02a7b523)
- **AVOStudio SSE**: @agent-architect tracks running instances via `/stream` endpoint
- **MCP trace server**: @mcp-specialist provides 24/7 observability via trace_handler

## Branch Protection

- `main` branch protected via `.github/branch-protection.yml`
- Required: all 4 CI phase jobs + auto-assign-on-failure job
- Linear history enforced; force pushes blocked

## Verification Checklist

- [x] `ci.yml` workflow file created in `.github/workflows/`
- [x] Branch protection rules documented in `.github/branch-protection.yml`
- [x] @hermes-avo-ci bot registered in `.github/bot-config.yml`
- [x] Coverage gates enforced in workflow
- [x] Auto-healing on failure with avatar routing
- [ ] Initial scaffold commit pushed to `main`
- [ ] Fresh commit triggers full build matrix
- [ ] `gh workflow list` shows ci.yml enabled
