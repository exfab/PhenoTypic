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
- **NEVER edit FEATURES.md by line number. Anchor on its `##` / `###` headings.**
  **Every line range the spec §6 gives for FEATURES.md is wrong**, by a consistent ≈ +7
  offset past line ~400 — verified 2026-08-26. The worst case is not cosmetic: spec §6 and
  an earlier draft of phase 1 said "delete the Results Timeline rows at `FEATURES.md:372-394`",
  but `:372-377` are **Colony curation rows** (`Colony radial lazy-populate`,
  `Custom folder + ＋ Add custom`, `Bulk "Mark N as ▾" (colony)`, `Pixel layer toggle`) and
  the `### Results Timeline tab` heading is at `:379`. Following the spec literally
  **deletes four curation rows** — a direct spec §5 violation, and phase 5's `git diff`
  guard would not catch it because that guard watches `colony_view/` and the radial modules,
  not the ledger.

  | Section | Heading to anchor on | Spec said | Actually |
  |---|---|---|---|
  | Browse Timeline | `## Browse tab (source image viewer)` block | 104-126 | 102-126 |
  | Results Timeline | `### Results Timeline tab` | 372-394 | **379**-400 |
  | Tune co-pilot | `` ## Tune co-pilot (`/tune/`) `` | 419-486 | **426**-493 |
  | Timeline shared engine | rows naming `gui/_shared/timeline/` | 536-537 | **543**-544 |
  | QC tab | `## QC tab` | 587 | **594** |
  | QC Review | `## QC Review sub-view` | 617 | **624** |
  | Heatmap tab | `## Heatmap tab` | 658 | **665** |
  | Error analysis | `## Error analysis tab` | 677 | **684** |

  **WORKFLOWS.md's row numbers `:46/47/51/52/54/55/56` ARE correct** — verified individually.
  The defect is specific to FEATURES.md.
- **Add FEATURES.md to phase 5's spec-§5 diff guard.** The curation rows live in the ledger
  as well as in code, and only the code half was being watched.
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

**Phase 1 before phases 2 and 5. Phase 4 is independent of all of them.**
Phase 5 shares `_layout.py` and `_callbacks.py` with phase 1 and *edits*
`test_layout_tab_shape.py`, which phase 1 task 1.1 creates — so it cannot run first.

| # | Phase | Deliverable | Doc |
|---|---|---|---|
| 1 | Results Timeline tab — delete | Results viewer has 5 tabs; `timeline_view/` gone | [phase-1](phase-1-results-timeline.md) |
| 2 | Browse Timeline mode **+ the shared engine** — delete | Browse has no view-mode toggle; `_shared/timeline/` gone | [phase-2](phase-2-browse-timeline.md) |
| 4 | Tune — unmount | `/tune/` 404s; `gui/tune/` still imports and unit-tests | [phase-4](phase-4-tune-unmount.md) |
| 5 | Heatmap / Error / QC — unmount | Results viewer has 2 tabs; 3 packages retained | [phase-5](phase-5-analysis-tabs-unmount.md) |
| 6 | Verification & docs | Shell-build test, curation gate, `gui/CLAUDE.md` | [phase-6](phase-6-verification.md) |

**Phase 3 was folded into phase 2** (tasks 2.5-2.6). The shared engine's consumer count
reaches zero at the end of phase 2's first four tasks, so a separate phase bought nothing
and cost a same-PR conditional in both directions. Numbering is left with a gap rather than
renumbered, so the phase names in the ledger and commits keep resolving.

## Definition of done

1. `uv run pytest tests/unit/gui tests/gui -n 4` green (minus the known baseline failure).
   **`tests/gui` is not optional** — browse GUI tests and the colony-view package live
   there, so a `tests/unit/gui`-only gate never reaches what phases 2 and 6 touch.
2. `uv run python scripts/check_features_md.py --strict` exits 0.
3. `uv run python scripts/check_workflows_md.py -v` exits 0.
4. `uv run python scripts/capture_gui_tutorial_screenshots.py --skip-cli` exits 0.
5. The two new checks from phase 6 pass, and the three curation-chain tests it names are
   green (one of them lives under `tests/gui/`, which a `tests/unit/gui` run never reaches).
6. `tests/unit/gui/results_viewer/test_colony_callbacks_helpers.py` passes **unmodified** —
   `git diff --stat` shows zero lines changed in that file across the whole plan.
