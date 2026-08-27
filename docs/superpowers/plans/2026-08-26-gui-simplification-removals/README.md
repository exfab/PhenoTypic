# GUI simplification — removals: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a GUI whose results viewer has exactly two tabs (Plate, Colony), whose
Browse tab has no view-mode toggle, and whose Tune sub-app is unreachable — deleting the
Timeline engine outright and unmounting Tune / Heatmap / Error / QC without deleting them.

**Architecture:** Two mechanisms, kept strictly distinct. **Delete** removes modules,
tests, assets, ledger rows, tutorial pages and capture functions from the tree. **Unmount**
removes the mount, the nav leaf, the `dbc.Tab`, *and the callback registration*, while the
package stays on disk with its unit tests passing and its e2e tests skip-marked. Work is
sequenced consumer-first: both Timeline consumers die before the shared engine they share,
so no phase ever leaves an import dangling.

**Tech Stack:** Python 3.11+, Dash / dash-bootstrap-components, Flask blueprints, pytest,
Playwright (e2e), `uv` as the sole runner.

**Spec:** [`docs/superpowers/specs/2026-08-26-gui-simplification-removals/design.md`](../../specs/2026-08-26-gui-simplification-removals/design.md)

**Baseline:** branch `feat/gui-ome-zarr-sync`, restacked onto
`worktree-ome-zarr-image-store` head `bf0d01a1`. Every `file:line` in the spec was
re-verified against this tree on 2026-08-26 and holds (see §Verified baseline).

---

## Global Constraints

- **`uv` is the sole runner.** Never bare `python`/`pip`. `uv run <cmd>`.
- **`QT_QPA_PLATFORM=offscreen` is mandatory** for any pytest invocation. Without it the
  interpreter aborts at ~79% with no summary.
- **Never `pytest -n auto`.** `nproc` reports the node's cores, not the allocation's.
  Pass an explicit `-n 4` or omit `-n`. Use the **`run-phenotypic-test`** skill for any
  non-trivial run; the full `tests/unit` suite is ~65 minutes and is a Slurm job
  (`docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch`).
- **`uv run ruff check --fix <explicit paths you changed>`.** Never bare `ruff check --fix`.
- **Known-failing baseline test, not caused by this work:**
  `tests/unit/cli/test_cli_terminal_failures.py::test_concurrent_process_appends_do_not_lose_records`
  fails on a 4-core allocation because it spawns 8 processes with a 20 s join timeout.
  Report the suite as "green except this one" and re-check it is still *this* test failing
  for *this* reason.
- **Three CI gates in `.github/workflows/gui-checks.yml` bind every phase that touches
  `src/phenotypic/gui/`:**
  - `features-md-gate` — a PR touching `gui/` **must** modify
    `src/phenotypic/gui/FEATURES.md`; then `scripts/check_features_md.py` and
    `--strict` must pass.
  - `workflows-md-gate` — `scripts/check_workflows_md.py -v` enforces the
    WORKFLOWS.md ↔ capture-function ↔ tutorial-page round trip.
  - `smoke-capture` — runs `scripts/capture_gui_tutorial_screenshots.py`.
  Ledger, capture-script and tutorial edits therefore live **inside** the phase that
  removes the surface, never in a follow-up phase.
- **The ledgers are at `src/phenotypic/gui/FEATURES.md` and
  `src/phenotypic/gui/WORKFLOWS.md`**, not the repo root. The spec cites them by bare
  filename; the paths above are the real ones.
- **Unmounted ≠ deleted in the ledger.** An unmounted surface's FEATURES.md row is
  **edited to say unmounted, with a pointer to this spec** — not removed. A deleted
  surface's row is removed. `check_features_md.py` only resolves refs for `✅ shipping`
  rows, so an unmounted row must not carry that status.
- **`colony_view/` is not touched by this plan.** See spec §5. If a test under
  `tests/unit/gui/results_viewer/test_colony_callbacks_helpers.py` needs editing, the
  plan has been violated — stop and escalate.
- **Browse Single mode behaviour is unchanged.** Every J/K nav, filmstrip, keep-position
  and preparation affordance stays. Single stops being *one of two modes* and becomes the
  whole tab; that is a layout change, not a behaviour change.

---

## Verified baseline

Re-verified in this worktree on 2026-08-26. These are facts the phases depend on:

| Claim | Verified |
|---|---|
| All 10 delete-target paths exist | yes |
| All 6 delete-target test paths exist | yes |
| `results_viewer/_layout.py` imports at `:65` `_error_tab`, `:66` `_heatmap_tab`, `:72` `_qc_tab`, `:74` `timeline_view` | yes |
| `results_viewer/_layout.py` bodies at `:609` heatmap, `:610` error, `:615` qc; `dbc.Tabs` holds **6** tabs at `:622-655`; `active_tab=ids.TAB_PLATE_ID` at `:656` | yes |
| `results_viewer/_callbacks.py` register calls at `:113` heatmap, `:114` qc, `:115` error, `:116` timeline | yes |
| `browse/_ids.py` carries **64** `BROWSE_TL_*` names plus `BROWSE_VIEW_MODE_TOGGLE` (`:47`), `BROWSE_SINGLE_BODY` (`:48`), `BROWSE_TIMELINE_BODY` (`:49`) | yes |
| `browse/_callbacks.py` imports the four doomed modules at `:39, :44, :46, :50` | yes |
| `browse/_app.py` registers `_thumb_routes` at `:33` (import) and `:84` (call) | yes |
| `browse/_layout.py:320` `build_timeline_body`, toggle at `:274`, single body at `:296` | yes |
| Capture fns at `:1156, :1246, :1750, :1810, :1900, :1947, :2813` | yes |
| WORKFLOWS.md rows at `:46, :47, :51, :52, :54, :55, :56` | yes |
| Tutorial pages `10, 11, 15, 16, 17, 19, 20` exist; highest is `20_results_timeline.md` | yes |

---

## Phases

Strict order. Phase 3 **must** follow 1 and 2 — the shared engine cannot be deleted while
either consumer still imports it.

| # | Phase | Deliverable | Doc |
|---|---|---|---|
| 1 | Results Timeline tab — delete | Results viewer has 5 tabs; `timeline_view/` gone | [phase-1](phase-1-results-timeline.md) |
| 2 | Browse Timeline mode — delete | Browse has no view-mode toggle; Single is the tab | [phase-2](phase-2-browse-timeline.md) |
| 3 | Shared timeline engine — delete | `_shared/timeline/` gone; no dangling imports | [phase-3](phase-3-shared-timeline-engine.md) |
| 4 | Tune — unmount | `/tune/` 404s; `gui/tune/` still imports and unit-tests | [phase-4](phase-4-tune-unmount.md) |
| 5 | Heatmap / Error / QC — unmount | Results viewer has 2 tabs; 3 packages retained | [phase-5](phase-5-analysis-tabs-unmount.md) |
| 6 | Verification & docs | Layout-shape tests, dangling-ref test, `gui/CLAUDE.md` | [phase-6](phase-6-verification.md) |

## Definition of done

1. `uv run pytest tests/unit/gui -n 4` green (minus the known baseline failure).
2. `uv run python scripts/check_features_md.py --strict` exits 0.
3. `uv run python scripts/check_workflows_md.py -v` exits 0.
4. `uv run python scripts/capture_gui_tutorial_screenshots.py --smoke` exits 0.
5. The three new tests from phase 6 pass.
6. `tests/unit/gui/results_viewer/test_colony_callbacks_helpers.py` passes **unmodified** —
   `git diff --stat` shows zero lines changed in that file across the whole plan.
