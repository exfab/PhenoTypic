# Execution Entry Point — `--process-only` CLI mode + `.phenotypic` cache

This directory holds the implementation plans for the `cli-processing-mode`
feature and the orchestration for executing them. **Start here.**

## What we're building

1. **`.phenotypic` machine-state migration** — move `progress/`,
   `processing_state.json`, `processing_events.log` into a hidden
   `<output>/.phenotypic/` cache (both forward CLI and process-only), with
   backward compatibility for pre-migration runs.
2. **`--process-only {rgb|gray|detect_mat|objmap}`** — an apply-only CLI mode
   that runs `pipeline.apply()` and exports a single image layer mirroring the
   input tree (TIFF at the image's bit depth; objmap as 16-bit raw-label PNG),
   skipping the measurement/analysis suite.

## Source documents

| Doc | Purpose |
|-----|---------|
| [`../../specs/2026-06-03-cli-process-only-and-phenotypic-cache-design.md`](../../specs/2026-06-03-cli-process-only-and-phenotypic-cache-design.md) | Approved design spec (decisions D1–D14, §1–§11) |
| [`phase1-phenotypic-cache-migration.md`](phase1-phenotypic-cache-migration.md) | Phase 1 — 7 TDD tasks |
| [`phase2-process-only-mode.md`](phase2-process-only-mode.md) | Phase 2 — 9 TDD tasks (depends on Phase 1) |

## Environment

- **Worktree:** `/Users/alex/Projects/PhenoTypic/.claude/worktrees/cli-processing-mode`
- **Branch:** `worktree-cli-processing-mode` (base: `origin/main` @ PR #114)
- **Runner:** `uv` only — `uv run pytest`, `uv run mypy src/phenotypic`,
  `uv run ruff check --fix`. Never bare `python`/`pip`.

## Orchestration

Two implementation phases, each run by a **single Opus subagent**, with a
**code-review agent after each phase**, and a **single simplify pass after both
phases**. The orchestrator (main session) reviews at every boundary — reads the
diff, runs the full gate, applies/dispatches fixes — before advancing.

```
Phase 1 (Opus subagent)
   └─ orchestrator review: diff + gate
   └─ code-review agent (Phase 1 diff)
   └─ orchestrator applies fixes  ──► commit, Phase 1 DONE
Phase 2 (Opus subagent)            (depends on Phase 1)
   └─ orchestrator review: diff + gate
   └─ code-review agent (Phase 2 diff)
   └─ orchestrator applies fixes  ──► commit, Phase 2 DONE
Simplify agent (whole feature diff)
   └─ orchestrator applies fixes + regression  ──► FEATURE DONE
```

Rationale for one-agent-per-phase (vs per-task): Phase 1 is **wide but
shallow** — 39 one-line path swaps across 17 files — gated by a deterministic
"no hand-joined state paths" grep test, so a single agent gets objective
completeness feedback. Phase 2 is **narrow but novel** — one ~120-line module +
small additive edits, each with its own unit test.

### Agent roster

| # | Agent | Model | Scope | Done when |
|---|-------|-------|-------|-----------|
| 1 | Phase 1 impl | Opus | Execute `phase1-…md` Tasks 1–7 | grep-gate green + suites green |
| 2 | Phase 1 review | (review) | Review Phase 1 diff for correctness/regressions | findings triaged |
| 3 | Phase 2 impl | Opus | Execute `phase2-…md` Tasks 1–9 | unit + e2e green |
| 4 | Phase 2 review | (review) | Review Phase 2 diff | findings triaged |
| 5 | Simplify | (simplify) | Whole-feature quality pass (reuse/clarity, no behavior change) | fixes applied + regression green |

### Subagent briefing contract

Each implementation subagent is told:
- Work **only** in this worktree on branch `worktree-cli-processing-mode`.
- Be the **sole committer**; commit per task with the message in the plan;
  scope commits with `git commit -- <paths>` (shared-worktree git index).
- Use `uv run` for everything.
- Follow the plan's TDD steps in order; do not skip the "run test to verify it
  fails" steps.
- Do **not** start the other phase.
- Use the **code-review-graph MCP tools first** for any exploration beyond the
  plan (project convention).
- Finish by running the phase's full gate and reporting: commits made, gate
  output, and any deviation from the plan.

## Per-phase verification gate

**Phase 1:**
```
uv run pytest tests/unit/tools_/test_io_constants.py -k "PhenotypicCache or BackCompat or MigrateLegacy or NoHandJoined or ReporterReads" -v
uv run pytest tests/unit/tools_ tests/unit/cli tests/unit/gui tests/integration/cli -q
uv run mypy src/phenotypic
uv run ruff check --fix
```
The grep-gate (`TestNoHandJoinedStatePaths`) must return `offenders == []`
(allowlisting `_io_constants.py`, `/sweep/`, and `checkpoint_handler` L200).

**Phase 2:**
```
uv run pytest tests/unit/cli tests/unit/gui tests/integration/cli -q
uv run mypy src/phenotypic
uv run ruff check --fix
```
Plus the process-only e2e + classifier discovery tests.

## CI gates to satisfy before the PR

- `gui-checks` **features-md-gate**: both phases touch `src/phenotypic/gui/`, so
  each must edit `gui/FEATURES.md` (Phase 1: status-read row; Phase 2:
  process-only discoverability row) with a real `Test ref`.
- `workflows-md-gate`: **not** triggered (no new end-to-end GUI flow).

## Refinements baked into the plans (beyond the spec text)

1. **Migrate-on-resume** (Phase 1 Task 1/2): the CLI moves a pre-migration
   run's state into `.phenotypic/` once at resume rather than writing new state
   beside legacy state (avoids event-log split-brain). GUI stays read-only via
   `resolve_*`.
2. **Classifier signal for process-only** (Phase 2 Task 8): `is_cli_output`
   keys on `results/` + `deliverables/`, which a process-only run lacks — so
   D13 ("visible in run console") needs a new `is_process_only_output`
   capability signalled by `.phenotypic/progress/manifest.json`.

(If we want the spec text reconciled to these, do it as a follow-up doc edit.)

## Status

- [ ] Phase 1 — implementation (Agent 1)
- [ ] Phase 1 — code review (Agent 2) + fixes
- [ ] Phase 2 — implementation (Agent 3)
- [ ] Phase 2 — code review (Agent 4) + fixes
- [ ] Simplify pass (Agent 5) + regression
- [ ] Final: full suite + `mypy` + `ruff` green; ready for PR
