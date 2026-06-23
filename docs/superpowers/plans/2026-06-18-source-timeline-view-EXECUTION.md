# Source Timeline View — Execution Plan (All Phases 1–6)

**Date:** 2026-06-18
**Worktree:** `/bigdata/exfab/anguy344/PhenoTypic/.worktrees/worktree-source-timeline-view`
**Branch:** `worktree-source-timeline-view` (isolated — **all** work happens here)
**Method:** subagent-driven-development — a fresh **Opus** implementation agent per task-group, an orchestrator two-gate review between groups, a dedicated `implementation-test-reviewer` after each phase, an **end-only** `code-simplifier` pass, a full regression run, and a **live Playwright-MCP verification** gate against the real UCR_029 reference data.

**Interaction model:** **focus-and-navigate** (spec §16) — supersedes the original scrollable-matrix model (D4 / §4.4). All six plans are authored/revised to it and **each has passed a `plan-reviewer` pass** (fixes applied).

**Governs these plans:**
- Phase 1 — shared engine: `docs/superpowers/plans/2026-06-18-source-timeline-view-phase1-shared-engine.md`
- Phase 2 — Browse surface: `docs/superpowers/plans/2026-06-18-source-timeline-view-phase2-browse.md`
- Phase 3 — Results surface: `docs/superpowers/plans/2026-06-18-source-timeline-view-phase3-results.md`
- Phase 4 — synced Compare strip: `docs/superpowers/plans/2026-06-18-source-timeline-view-phase4-compare-strip.md`
- Phase 5 — CLI `deliverables/metadata.csv` copy: `docs/superpowers/plans/2026-06-18-source-timeline-view-phase5-cli-metadata-copy.md`
- Phase 6 — integration, docs & verification: `docs/superpowers/plans/2026-06-18-source-timeline-view-phase6-integration-docs-verification.md`

---

## Execution model

- **Implementation agents run on Opus** (`model: opus`). Task-groups are sized so one Opus agent gets a coherent, self-contained chunk bounded at a clean dependency/review boundary. Each agent owns one group end-to-end (all its tasks' TDD cycles + per-task commits).
- **Strictly sequential.** Implementation agents run **one at a time** — never in parallel. They share this worktree's working tree and git index; concurrent committers corrupt each other (project memory: *background_job_shared_worktree*, *feedback_agent_team_decision_oscillation*). Each group fully completes (commits landed) before the next is dispatched.
- **Per-task commits stay inside the agent.** The phase plans embed `git commit` steps; the implementation agent performs them. Because agents are sequential, there is exactly one committer at any moment.
- **Two-gate review between groups (orchestrator):** after a group returns, the orchestrator (1) reads every changed file, and (2) runs the group's tests + lint/type checks. A group is "done" only when both pass; otherwise dispatch a fix to a fresh agent with the specific failures.
- **Phase code review:** after all groups in a phase are green, dispatch an `implementation-test-reviewer` (Opus) scoped to the phase's diff. Triage; fix material/correctness findings via a follow-up agent before starting the next phase.
- **End-only simplify (user-decided):** the `code-simplifier` runs **once**, over the whole-feature diff, in **Phase 6 Task 3** — *not* per phase. Then the full regression run (Phase 6 Task 4).

---

## Standing constraints (every agent)

- **`uv` only.** Tests `uv run pytest …`; e2e `PLAYWRIGHT=1 uv run pytest tests/e2e/gui/…`; lint `uv run ruff check`; types `uv run mypy …`.
- **Follow the phase plan task exactly** — TDD order (write failing test → confirm fail → implement → confirm pass → commit). No skipping the "confirm fail" step.
- **Worktree-absolute paths only.** The worktree is `/bigdata/exfab/anguy344/PhenoTypic/.worktrees/worktree-source-timeline-view`. **Never** `cd` to the main repo (memory: *worktree_cd_redirects_edits_to_main*); verify `phenotypic.__file__` resolves under this worktree if unsure. (The branch is `worktree-source-timeline-view`.)
- **Stay in lane.** Touch only the files the assigned tasks name. Do not refactor neighbours.
- **Report back** the exact commands run + their pass/fail output (not a summary) so the orchestrator gate is evidence-based (skill: *verification-before-completion*).
- **`tests/gui` is collected** (Phase 1 Task 0 adds it to `pyproject.toml` `testpaths`) — Dash Flask-route tests must set `app.layout` or they 500 (memory: *gui-test-collection-and-route-fixtures*).
- **Byte-identical `timeline.js` (Phases 2 & 3):** the focus-navigate controller is vendored into BOTH `src/phenotypic/gui/browse/_assets/timeline.js` and `src/phenotypic/gui/results_viewer/_assets/timeline.js`; a CI byte-equality guard enforces they match. **Any edit to the controller must be applied to both copies identically** (it is surface-agnostic by design — finds its sibling controls by class scoped to `.timeline-body`, never by hardcoded ids).

---

## Dependency order (the spine)

```
Phase 1 ─▶ Phase 2 ─▶ Phase 3 ─▶ Phase 4 ─▶ Phase 6
                                   ▲
                       Phase 5 ────┘  (independent; slot before Phase 6)
```

- **Phase 2** consumes Phase 1's engine (`build_matrix`, `build_timeline_grid`, `register_thumbnail_route`, constants).
- **Phase 3** vendors Phase 2's `timeline.js` **byte-for-byte** (byte-equality guard) and reuses the engine; it only adds the surface-agnostic classes + a Results clientside `attach`. It does **not** edit the controller.
- **Phase 4** extends the controller with selection + the Compare strip across both surfaces; its **Results** wiring (Task 8) is a Phase-3-dependent follow-up and may ship `🔭 planned`.
- **Phase 5** (CLI copy) is independent of the GUI; run it any time before Phase 6.
- **Phase 6** verifies the whole lattice and must not be dispatched until 1–5 are landed and green (its Task 0 is an explicit existence gate).

---

## Task groups

> Task ranges reference each phase plan's final task list — confirm against the plan at dispatch (the plans were revised post-review). Gate commands are the group's stated `uv run …` suites.

### Phase 1 — shared engine (2 Opus groups)
| Group | Tasks | Scope | Gate |
|-------|-------|-------|------|
| **P1-A** | 0–3 | `testpaths` pre-flight (Task 0); `_config` constants + `snap_thumb_bucket`; `_matrix` (`_natural_sort_key` w/ NaN/inf guard, `build_matrix`, dataclasses) | `uv run pytest tests/gui/_shared/timeline/test_constants.py tests/gui/_shared/timeline/test_matrix.py -v` + (Task 0) `uv run pytest tests/gui -q` green baseline |
| **P1-B** | 4–8 | `_thumbnail` (naming, downscale, route factory + **per-source lock**); `_grid` (`build_timeline_grid` — `ref_builder`, `data-row/col-index` on cells AND axis labels, hover-⤢); package `__init__` + `🧪 internal` FEATURES row | `uv run pytest tests/gui/_shared/timeline/ -v` + `uv run ruff check src/phenotypic/gui/_shared/timeline src/phenotypic/gui/_config.py` + `uv run mypy src/phenotypic/gui/_shared/timeline` |

### Phase 2 — Browse surface (4 Opus groups)
| Group | Tasks | Scope | Gate |
|-------|-------|-------|------|
| **P2-A** | 1–5 | `_ids` (+ nav/position ids); `_capture_time` (exifread, static JPEG fixture); `_plate_pattern`; `_timeline_records` (per-axis builder + folder-scoped CSV); `_thumb_routes` | `uv run pytest tests/gui/browse/test_ids.py tests/gui/browse/test_capture_time.py tests/gui/browse/test_plate_pattern.py tests/gui/browse/test_timeline_records.py tests/gui/browse/test_thumb_routes.py -v` |
| **P2-B** | 6, 8 | `_layout` (view-mode toggle, no-scroll viewport, edge buttons, position readout, `.timeline-*` classes, static `data-focus-margin`/`-mount-cap`/`-warm-concurrency`); `_callbacks` (record-build → render, pattern preview, tile-size stepper, CSV-column population, nudge) + `read_metadata_csv_table` (utf-8-sig) | `uv run pytest tests/gui/browse/test_layout.py tests/gui/browse/test_timeline_callbacks_helpers.py -v` |
| **P2-C** | 7, 9 | `timeline.js` **focus-navigate controller** (class-based lookup, clamp-translate, margin-ring mount/offload, rAF attach-resilience, neighborhood-first warm); `browse.js` `_mountOSD`/`applyPopoutImage` refactor; Enter + hover-⤢ pop-out bridge; `live_browse_timeline` e2e (sidebar-tree seeding) | `PLAYWRIGHT=1 uv run pytest tests/e2e/gui/test_browse_timeline.py -v` |
| **P2-D** | 10 | `create_app` thumb-route registration; FEATURES/WORKFLOWS rows; `_capture_browse_timeline` + tutorial `19_browse_timeline.md`; screenshot regen | `uv run pytest tests/gui/browse -v` + `uv run python scripts/check_features_md.py` + `uv run python scripts/check_workflows_md.py` + ruff/mypy |

### Phase 3 — Results surface (3 Opus groups)
| Group | Tasks | Scope | Gate |
|-------|-------|-------|------|
| **P3-A** | 1–5 | `TAB_TIMELINE_ID` + **vendor `timeline.js` byte-for-byte** (+ byte-equality guard test) + package skeleton; uncapped `selectable_axis_columns(max_cardinality=None)` (shared-file change; colony caller stays 50); `timeline_view/_grid` (`selectable_time_columns`, `is_large_time_axis`, `has_eligible_time_axis`, `build_timeline_records`); `_ids`; `_thumb_routes` `(dataset,stem)` over overlays | `uv run pytest tests/gui/results_viewer/timeline_view/ -v` |
| **P3-B** | 6–7 | `_layout` (tab body + `.timeline-*` classes) + `_callbacks` (filter-aware dropdowns, render, empty state, tile stepper, clientside `attach("timeline-grid")`, pop-out via overlay DZI); wire 6th tab + thumb route into `_layout`/`_callbacks`/`_app` + e2e | `PLAYWRIGHT=1 uv run pytest tests/e2e/gui/<results timeline e2e>.py -v` + `uv run pytest tests/integration/gui -k timeline -v` |
| **P3-C** | 8 | package exports + asset-sync byte guard + FEATURES/WORKFLOWS rows + `_capture_results_timeline` + tutorial `20_results_timeline.md` | `uv run pytest tests/gui/results_viewer/timeline_view -v` + `uv run python scripts/check_features_md.py` + `uv run python scripts/check_workflows_md.py` + ruff/mypy |

### Phase 4 — synced Compare strip (3 Opus groups)
| Group | Tasks | Scope | Gate |
|-------|-------|-------|------|
| **P4-A** | 1–3 | `TIMELINE_COMPARE_CAP` const; pure `compare_selection_plan(refs, cap) → ComparePlan` (cap/over-cap + exact notice string); engine `__init__` export | `uv run pytest tests/gui/_shared/timeline/test_compare_plan.py -v` |
| **P4-B** | 4–6 | shared `openCompareStrip` (≤cap OSD viewers, feedback-guarded shared viewport, `ns.__compareViewers` seam, pinned modal DOM, teardown); Browse compare ids + layout; selection + triggers in `timeline.js` (DOM-class single source of truth; shift/ctrl toggle; row-header opens) | `uv run pytest tests/gui/browse/test_compare_layout.py -v` (+ vendored-copy byte guard re-checked) |
| **P4-C** | 7, 9 | Browse Compare e2e (multi-select→N viewers, viewport sync, row-header, over-cap exact notice); FEATURES rows. **Task 8 (Results compare wiring) = deferred follow-up** (ship `🔭 planned` if not done) | `PLAYWRIGHT=1 uv run pytest tests/e2e/gui/test_browse_compare_strip.py -v` + `uv run python scripts/check_features_md.py` |

### Phase 5 — CLI metadata copy (1 Opus group)
| Group | Tasks | Scope | Gate |
|-------|-------|-------|------|
| **P5** | 1–3 | `DELIVERABLES_METADATA_CSV` + `metadata_csv_deliverable_path` (sdk_); guarded best-effort copy in `finalize_post_master_outputs`; chunk-writer left untouched (`git diff --stat` guard) | `uv run pytest tests/unit/sdk_/test_io_constants.py tests/unit/cli/test_cli_output_manager.py -v` + ruff/mypy |

### Phase 6 — integration, docs & verification (orchestrator-run + 1 small Opus task)
Run by the orchestrator (mostly gates), with one dispatched Opus task:
- **Task 0** — preconditions existence gate (STOP unless 1–5 landed: `metadata_csv_deliverable_path` importable, finalize copy block present, byte-equality test exists, both `timeline.js` copies exist, `_capture_*_timeline` defined; `tests/gui ∈ testpaths`).
- **Task 1** *(Opus)* — CLI end-to-end integration test (`tests/integration/cli/`, `CliRunner` + `--metadata`, byte-match `deliverables/metadata.csv`). Gate: `uv run pytest tests/integration/cli/test_cli_metadata_deliverable.py -v`.
- **Task 2** — docs/CI reconciliation: `check_features_md.py` (+`--strict`) + `check_workflows_md.py` green across all timeline rows; full screenshot regen committed wholesale.
- **Task 3** — **end-only** `code-simplifier` over the whole-feature diff (`_shared/timeline`, `browse/` timeline + `browse/_assets/timeline.js`, `results_viewer/timeline_view` + its vendored `_assets/timeline.js`, `_config.py`, CLI copy). Simplify both `timeline.js` copies identically or neither; never let it break the §8.3 never-raise on the metadata copy.
- **Task 4** — full regression (all phases' unit + e2e + gates + ruff + mypy).
- **Task 5** — **live Playwright-MCP verification** (orchestrator, live HPCC GUI server, SSH-tunnel): Browse over `~/bigdata_exfab/ucr_029_e_d_Maresca/data/processed/` (flat folder → **filename-pattern row axis**) and Results over `~/bigdata_exfab/ucr_029_e_d_Maresca/data/results/2026-06-16/` (X=`Metadata_ImageNumber`, Y=`Metadata_PlateNum`, one overlay/cell). Drive arrows/edge-buttons, confirm clamp-translate + bounded mount + margin-ring pre-mount, open a pop-out, open Compare on a row with synced pan/zoom; screenshot to a dated scratch dir.
- **Task 6** — finish the branch (`superpowers:finishing-a-development-branch`) → PR/hand-off.

---

## Dispatch protocol (per implementation group)

Brief each Opus implementation agent with this shape:

```
You are implementing <GROUP ID> of an approved, committed plan. Work ONLY in the worktree at
/bigdata/exfab/anguy344/PhenoTypic/.worktrees/worktree-source-timeline-view
(worktree-absolute paths; never cd to the main repo; branch worktree-source-timeline-view).

Read the plan: <phase plan path>. Implement exactly Tasks <N..M> — every step, in order, TDD
discipline (write the failing test, RUN it and confirm it fails, implement, RUN and confirm pass,
commit per the task's git step). Use `uv run` for everything.

Touch only the files those tasks name. Do not refactor neighbours. If a task's stated
command/signature does not match the real code, STOP and report the mismatch rather than guessing.
For any controller (timeline.js) edit, apply it identically to BOTH vendored copies.

When done, report: the exact pytest/ruff/mypy commands you ran and their full pass/fail output,
the list of files changed, and the commit hashes. Do not claim success without the passing output.
```

After the agent returns, the orchestrator: (1) `git log --oneline` + `git diff` the group's range, read every changed file; (2) re-run the group's gate commands independently; (3) green → next group, else dispatch a fresh fix agent with the precise failure.

---

## Sequence (one line per step)

1. **P1-A** (Opus) → gate. **P1-B** → gate. **Phase 1 review** (`implementation-test-reviewer`) → fix.
2. **P2-A** → gate. **P2-B** → gate. **P2-C** → gate (e2e). **P2-D** → gate (FEATURES/WORKFLOWS + screenshots). **Phase 2 review** → fix.
3. **P3-A** → gate. **P3-B** → gate (e2e + integration). **P3-C** → gate (docs). **Phase 3 review** → fix.
4. **P4-A** → gate. **P4-B** → gate (+ byte guard). **P4-C** → gate (e2e + FEATURES). **Phase 4 review** → fix.
5. **P5** → gate. **Phase 5 review** (light) → fix.
6. **Phase 6:** Task 0 preconditions gate → **Task 1** (Opus CLI integration test) → gate → Task 2 docs reconciliation → **Task 3 end-only simplify** → Task 4 full regression → Task 5 live-MCP verification → Task 6 finish branch.

---

## Risk notes carried from the plan reviews

- **P2-C is the highest-risk group** — the `timeline.js` focus-navigate controller + the `browse.js` `_mountOSD` refactor (keep `applyImage` byte-identical; existing browse tests are the guard) + the `live_browse_timeline` fixture's **sidebar-tree source seeding** (NOT localStorage injection — that was the corrected blocker; copy `tests/e2e/gui/test_shared_source_root.py`'s idiom). Budget extra orchestrator review.
- **`timeline.js` byte-equality guard (Phases 2/3, re-checked in P4-B and Phase 6 Task 3):** the two vendored copies must stay identical — edit both or neither.
- **`selectable_axis_columns(max_cardinality=None)` is a shared-file change** (P3-A) touching `colony_view/_grid.py`; the colony caller must stay at the default 50 — re-run colony tests in the P3-A gate.
- **Screenshot churn (P2-D, P3-C, Phase 6 Task 2):** `capture_gui_tutorial_screenshots.py` regenerates the full set; commit them all, do not cherry-pick (CLAUDE.md).
- **Phase 5 best-effort/never-raise:** the metadata copy must never raise out of `finalize_post_master_outputs` (spec §8.3); the simplifier (Phase 6 Task 3) must not collapse its `try/except`.
- **Live-MCP gate (Phase 6 Task 5)** needs a live HPCC GUI server (`uv run phenotypic-gui --root <sandbox-parent> --port <p>`, SSH-tunnel; `--root` is the sandbox parent, source/output selected in-app) + the real data. Orchestrator-run at the very end; expect slow Browse first-open over 3663 TIFFs (§15.8 warm) — confirm the focus window mounts a **bounded** set.
- **Shared-worktree hazard:** if the user runs git operations in this worktree mid-execution, re-check `git status` before trusting an agent's result (memory: *background_job_shared_worktree*).
