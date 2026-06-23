# Source Timeline View — Phase 6: Integration Test, Docs/CI Reconciliation & Verification

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **One** task here is a real TDD test (Task 1 — the CLI end-to-end integration test); the rest are **process / verification gates** — clear checklist steps the orchestrator runs after all surface phases (1–5) have landed.

**Goal:** The cross-cutting **finalization** phase for the Source Timeline View feature. It (1) adds the single end-to-end CLI integration test that Phase 5 deliberately deferred — proving the real wiring `CLI → aggregate_measurements → finalize` produces `deliverables/metadata.csv` (spec §8/§10); (2) reconciles the docs/CI gates holistically across **all** the timeline surfaces that already shipped their own FEATURES/WORKFLOWS/screenshots in Phases 2–4; (3) runs **one** end-of-feature `code-simplifier` pass over the whole-feature diff; (4) runs the consolidated regression suite spanning every phase; (5) drives a documented **live Playwright-MCP verification** against the real reference data (spec §16.9); and (6) finishes the branch.

**Scope boundary — do NOT re-do per-surface docs work.** Each surface phase already added its own FEATURES.md rows, WORKFLOWS.md row(s), `_capture_<id>`, tutorial page, and e2e:

| Surface phase | FEATURES rows | WORKFLOWS row | Capture fn | Tutorial | e2e |
|---|---|---|---|---|---|
| Phase 1 (engine) | `🧪 internal` infra row | — | — | — | `tests/gui/_shared/timeline/` unit |
| Phase 2 (Browse) | view-mode/axes/pattern/stepper/nudge/thumb/focus-navigate/edge-buttons/keyboard/readout/pop-out | `browse_timeline` — *find ideal starting time* | `_capture_browse_timeline` | `docs/source/tutorials/gui/<NN>_browse_timeline.md` (tutorial **19**) | `tests/e2e/gui/test_browse_timeline.py` |
| Phase 3 (Results) | Y dropdown / time-col selector / empty state / focus-navigate / edge-buttons / tab re-attach / pop-out + a `timeline.js` byte-equality guard | `results_timeline` — *trait emergence over time* | `_capture_results_timeline` | `docs/source/tutorials/gui/<NN>_results_timeline.md` (tutorial **20**) | `tests/e2e/gui/test_results_timeline.py` |
| Phase 4 (Compare strip) | Browse compare-strip / viewport-sync / multi-select / row-header / over-cap rows (Results compare rows are a Task-8 follow-up) | **no new row** (folded into the two existing tutorials — Phase 4 Task 9 decision) | — (a Compare-strip screenshot is captured *inside* the existing `_capture_*_timeline` fns) | — | `tests/e2e/gui/test_browse_compare_strip.py` (+ deferred `test_results_compare_strip.py`) |
| Phase 5 (CLI copy) | none (no `gui/` touch) | — | — | — | `tests/unit/cli/test_cli_output_manager.py` (unit) |

Phase 6 **verifies that lattice as a whole**; it does not add or re-enumerate per-surface rows.

**Tech Stack:** Python 3, `click.testing.CliRunner` + `phenotypic.phenotypicCLI.phenotypic_cli` (the real CLI entry — verified harness, see Task 1), polars (reading the mirror in the test assertion), pytest; the docs/CI gate scripts `scripts/check_features_md.py` / `scripts/check_workflows_md.py` / `scripts/capture_gui_tutorial_screenshots.py`; the Playwright MCP (`mcp__plugin_playwright_playwright__*`) for the live verification gate.

## Global Constraints

- **`uv` is the sole runner.** Every command is `uv run …`; never bare `python`/`pip`.
- **Phases 1–5 must already be landed and green — and Phase 6 is NOT dispatchable until the Task 0 existence gate below passes.** Phase 6 is the *last* phase; it assumes the engine (Phase 1), Browse (Phase 2), Results (Phase 3), Compare strip (Phase 4), and the CLI metadata copy (Phase 5) are all committed on the branch. **Do not trust phase labels** — Task 0 concretely asserts each predecessor's artifacts exist and STOPs if any is absent. The reconciliation/regression/simplify gates here are meaningless against a half-built feature.
- **Test collection.** `tests/integration/cli/` is in `pyproject.toml` `testpaths` (so the new Task 1 integration test auto-collects). `tests/gui` is collected **only if Phase 1 Task 0 already added it** to `testpaths` — Task 0 below verifies this with a `grep` precondition rather than asserting it as done. Do not modify `testpaths` in this phase (adding `tests/gui` is Phase 1 Task 0's job, not Phase 6's).
- **The CLI copy itself is Phase 5's code — do NOT re-implement it here.** Task 1 is a *test only*; it exercises the already-shipped `finalize_post_master_outputs` copy through the real CLI. If the copy behavior is missing/broken, that is a Phase 5 regression — STOP and report; do not patch production code in this phase.
- **Simplify is quality-only.** The Task 3 `code-simplifier` pass applies **non-behavioral** reuse/clarity fixes only; reject any edit that changes behavior. The two vendored `timeline.js` copies (`browse/_assets/timeline.js` and `results_viewer/_assets/timeline.js`) are kept byte-identical by a CI guard (Phase 3 `test_viewer_timeline_js_is_byte_identical_to_browse`) — **simplify BOTH identically or NEITHER**; never touch one alone.
- **Screenshots are regenerated wholesale (CLAUDE.md).** `scripts/capture_gui_tutorial_screenshots.py` regenerates the **full** PNG set; cross-surface font-rendering noise shifts unrelated tutorials' PNGs by a few bytes. **Commit them all — do not cherry-pick or `git checkout --` the collateral.**
- **Worktree-absolute paths only.** Work in `/bigdata/exfab/anguy344/PhenoTypic/.worktrees/worktree-source-timeline-view`; never `cd` to the main repo (memory: *worktree_cd_redirects_edits_to_main*).
- **The live-MCP gate (Task 5) and finish-branch (Task 6) are orchestrator/human gates** — not CI, not pytest. They run last, by the orchestrator, on a live HPCC session.

---

### Task 0: Predecessor-landed existence gate (process — Phase 6 is not dispatchable until green)

**Files:** none (verification only — STOP-on-absence preflight).

**Why:** Phase 6 finalizes a feature built across Phases 1–5. The earlier draft of this plan asserted post-execution tree state (e.g. "`tests/gui` is in `testpaths`", "the metadata-copy block exists", "`metadata_csv_deliverable_path` is exported") as already-true. None of that is guaranteed when Phase 6 starts — those are *outputs* of Phases 1–5. This gate **concretely confirms** each predecessor's artifacts exist and **STOPs** if any is absent, so a downstream task never trusts a "verified" label that has not actually executed yet.

- [ ] **Step 1: Phase 5 — the metadata-copy helper is exported and the copy block exists**

```bash
uv run python -c "from phenotypic.sdk_ import metadata_csv_deliverable_path"   # must succeed
grep -n "metadata_csv_deliverable_path" src/phenotypic/_cli/_cli_output_manager.py   # the copy block must reference it
```

Expected: the import succeeds (the `phenotypic.sdk_.metadata_csv_deliverable_path` helper is exported), **and** `_cli_output_manager.py` contains a reference to `metadata_csv_deliverable_path` inside `finalize_post_master_outputs` (the best-effort copy block Phase 5 added). **If the import fails or the grep is empty → STOP and report:** Phase 5 has not landed; Phase 6's Task 1 integration test cannot pass and the reconciliation gates are premature.

> Note (verified at authoring time): the **forward CLI already threads** `metadata_csv` into `finalize_post_master_outputs` (`_cli_output_manager.py:906`), so once Phase 5's copy block lands, the wiring is complete. This grep confirms the *copy block itself* is present, not just the threading.

- [ ] **Step 2: Phase 1 Task 0 — `tests/gui` is in the default `testpaths`**

```bash
grep -q 'tests/gui' pyproject.toml && echo "tests/gui in testpaths: OK" || echo "MISSING — STOP"
```

Expected: `OK`. **If MISSING → STOP and report:** Phase 1 Task 0 (which widens `testpaths` to include `tests/gui`) has not executed, so the timeline unit suites are not in the default CI lane and the Task 4 regression command will under-collect. Do **not** add it here — re-open Phase 1 Task 0.

- [ ] **Step 3: Phases 2 + 3 — both vendored `timeline.js` copies + the byte-equality guard exist**

```bash
ls src/phenotypic/gui/browse/_assets/timeline.js
ls src/phenotypic/gui/results_viewer/_assets/timeline.js
grep -n "test_viewer_timeline_js_is_byte_identical_to_browse" \
  tests/gui/results_viewer/timeline_view/test_assets.py
```

Expected: both files exist **and** the byte-equality test is defined. **If any is absent → STOP and report:** Phase 2 (Browse `timeline.js`) or Phase 3 (Results vendored copy + guard) has not landed; the Task 3 simplify-safety constraint and the Task 4 e2e have no foundation.

- [ ] **Step 4: Phases 2 + 3 — both capture functions are defined in the screenshot script**

```bash
grep -n "def _capture_browse_timeline\|def _capture_results_timeline" \
  scripts/capture_gui_tutorial_screenshots.py
```

Expected: **both** `_capture_browse_timeline` and `_capture_results_timeline` are defined. **If either is missing → STOP and report:** the WORKFLOWS round-trip (Task 2) cannot pass — the capture fn must be defined *and* dispatched, and Task 2 verifies dispatch, but a missing definition is a predecessor gap, not a Phase 6 authoring task.

- [ ] **Step 5: Record the gate**

Record each command's actual output (not a summary). Phase 6 proceeds **only** when all of Steps 1–4 are green. Any STOP here is a predecessor regression to be fixed in the owning phase before Phase 6 is dispatched.

---

### Task 1: CLI end-to-end integration test for `deliverables/metadata.csv` (TDD)

**Files:**
- Create: `tests/integration/cli/test_cli_metadata_deliverable.py`

**Why this exists (Phase 5 deferred it):** Phase 5 unit-tested the copy by calling `finalize_post_master_outputs(..., metadata_csv=source, ...)` **directly** (`tests/unit/cli/test_cli_output_manager.py::TestFinalizeCopiesMetadataCsv`). That proves the function copies, but **not** that the real CLI threads the `--metadata` path all the way through `aggregate_measurements` into finalize. Phase 6 adds the missing integration-level assertion: a real `phenotypic_cli` run with `--metadata <csv>` produces `<output>/deliverables/metadata.csv` byte-matching the source.

**Verified harness (mirror this exactly):** `tests/integration/cli/test_phenotypic_cache_layout.py` is the canonical smallest real CLI run that produces `deliverables/`:
- It invokes the **real** entry point: `from phenotypic.phenotypicCLI import phenotypic_cli` + `CliRunner().invoke(phenotypic_cli, [...])`.
- Fixtures (`tests/integration/cli/conftest.py`, already on the path for this dir): `synth_plate_dir` (a `plates/` dir with one synth plate `plate_001.png`) and `simple_pipeline_json` (a serialized `RoundPeaksPipeline`).
- Flags for a fast, SLURM-free, local run: `--force-local --skip-validation --njobs 1`.
- It asserts `(out / "deliverables").exists()` after a 0-exit run.

**Verified `--metadata` contract** (`phenotypicCLI.py:770-776`): `--metadata` is a `click.Path(exists=True, dir_okay=False)` whose CSV is **inner-joined onto the post-applied mirror** (`deliverables/measurements.parquet`) *on shared columns* — **not** onto `master_measurements.*`, which stays metadata-free per spec §8.2 + CLAUDE.md ("Master vs. mirror outputs"). So the test's CSV should carry a column name that exists in the run's measurement frame (e.g. `Metadata_ImageFile`) plus the per-plate value the synth run produces — **but the copy is independent of join success** (spec §8.2: the copy preserves the full original mapping precisely *because* the inner join can drop unmatched rows from the mirror). The test asserts the **copy**, so it does not need a perfect join; it only needs `--metadata` to point at a real, readable CSV (the `exists=True` Click guard requires the file to exist on disk).

**Interfaces:**
- Consumes: `phenotypic_cli`, `synth_plate_dir`, `simple_pipeline_json`, and the `phenotypic.sdk_.metadata_csv_deliverable_path` helper Phase 5 added.
- Produces: a single integration test proving end-to-end CLI → finalize copy.

- [ ] **Step 1: Write the failing-first test**

Create `tests/integration/cli/test_cli_metadata_deliverable.py`:

```python
"""End-to-end: a ``--metadata`` CLI run copies the source CSV to
``deliverables/metadata.csv`` (spec §8 / D6).

Phase 5 unit-tested the copy by calling ``finalize_post_master_outputs``
directly; this asserts the REAL wiring CLI → aggregate_measurements →
finalize threads ``--metadata`` through and produces the co-located copy.
Mirrors the smallest real run in ``test_phenotypic_cache_layout.py``
(``--force-local --skip-validation --njobs 1`` over the ``synth_plate_dir``
fixture), then asserts the byte-for-byte deliverable.
"""
from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from phenotypic.phenotypicCLI import phenotypic_cli
from phenotypic.sdk_ import metadata_csv_deliverable_path


def test_metadata_run_copies_source_csv_to_deliverables(
    tmp_path: Path, synth_plate_dir: Path, simple_pipeline_json: Path
) -> None:
    out = tmp_path / "out"
    # A real, readable metadata CSV. ``Metadata_ImageFile`` is a column the
    # measurement frame carries, so this also exercises a normal inner-join
    # onto the post-applied MIRROR (deliverables/measurements.parquet) — the
    # master_measurements.* archive stays metadata-free (spec §8.2). A non-ASCII
    # cell guards against an accidental text-mode re-encode in a future
    # refactor (a byte-for-byte copy preserves the UTF-8 bytes).
    source = tmp_path / "meta.csv"
    source.write_text(
        "Metadata_ImageFile,Metadata_Strain\nplate_001,Säccharomyces\n",
        encoding="utf-8",
    )

    res = CliRunner().invoke(
        phenotypic_cli,
        [
            "--pipeline",
            str(simple_pipeline_json),
            "--input",
            str(synth_plate_dir),
            "--output",
            str(out),
            "--metadata",
            str(source),
            "--force-local",
            "--skip-validation",
            "--njobs",
            "1",
        ],
    )

    assert res.exit_code == 0, res.output
    copied = metadata_csv_deliverable_path(out)
    assert copied.exists(), f"expected {copied} to exist after a --metadata run"
    assert copied.read_bytes() == source.read_bytes()
```

> **If the join column differs:** the assertion that matters is the **copy** (`copied.read_bytes() == source.read_bytes()`), which is join-independent. If a future schema change renames the join column, the run still exits 0 and the copy still lands. Should the run unexpectedly exit non-zero, read `res.output` — a `--metadata` inner-join (onto the mirror) that produces an empty frame is the only realistic failure mode, and the chosen `Metadata_ImageFile,plate_001` pair is taken from the synth fixture's image basename (`synth_plate_dir` writes `plate_001.png`) to avoid it. Adjust the CSV's key column/value to a column the run's measurement frame actually carries if needed; **do not** weaken the byte-equality assertion.

- [ ] **Step 2: Run the test to verify it passes (the copy is Phase 5 production code, already landed)**

Run: `uv run pytest tests/integration/cli/test_cli_metadata_deliverable.py -v`

Expected: **PASS.** Unlike a normal TDD task, the production behavior already exists (Phase 5). This test is the deferred *integration-level* assertion, so it is green on first write **iff** Phase 5 wired the copy correctly through the real CLI. **If it FAILS** on `assert copied.exists()`, the `--metadata` path is not reaching `finalize_post_master_outputs` in the forward CLI — that is a **Phase 5 wiring regression**, not something to patch here: STOP and report (re-open Phase 5 Task 2 / its call-site assumption that `aggregate_measurements` already threads `metadata_csv`).

- [ ] **Step 3: Confirm it is collected by the default lane**

Run: `uv run pytest --collect-only -q tests/integration/cli/test_cli_metadata_deliverable.py | tail -3`
Expected: the test item appears (it lives under `tests/integration`, already in `testpaths`).

- [ ] **Step 4: Commit**

```bash
git add tests/integration/cli/test_cli_metadata_deliverable.py
git commit -m "test(cli): e2e assert --metadata run copies source CSV to deliverables/metadata.csv"
```

---

### Task 2: Docs/CI reconciliation gate (process — verify, do not author rows)

**Files:** none expected to change for the gates themselves; **only** the regenerated screenshot PNGs (Step 4) and possibly a `_static/gui_images/<id>/` directory commit.

**Why:** Each surface phase added its own FEATURES/WORKFLOWS rows + capture fn + tutorial. Phase 6 confirms the **whole lattice** reconciles after everything has landed — a row could pass its own phase's local gate yet collide or orphan once a *later* phase's rows land (e.g. a duplicate `Test ref`, a capture fn defined but never dispatched, an empty screenshot dir).

> **Verified gate behavior (read before running):**
> - `scripts/check_features_md.py` parses `src/phenotypic/gui/FEATURES.md`; `✅ shipping` rows must resolve `Test ref` to a real `path::test`; `🧪 internal` rows are skipped; `🔭 planned` rows allow not-yet-existing files; `--strict` (the merge gate) rejects `🚧 in progress`. Run **both** the plain and `--strict` forms.
> - `scripts/check_workflows_md.py` AST-walks `scripts/capture_gui_tutorial_screenshots.py`, requiring each WORKFLOWS row's `_capture_<id>` to be **defined AND dispatched** (called inside `capture_workflow_screenshots` / `capture_standalone_viewer_screenshots`), and for `✅ shipping` rows: a non-empty `docs/source/_static/gui_images/<ID>/*.png` **and** the referenced tutorial page under `docs/source/tutorials/`. It also flags **orphan** capture fns no row references.

- [ ] **Step 1: FEATURES.md gate — plain and strict**

```bash
uv run python scripts/check_features_md.py
uv run python scripts/check_features_md.py --strict
```

Expected: both pass. Confirm by eye that the timeline rows from Phases 2–4 are present and their `✅ shipping` rows resolve: Browse (view-mode, row/time source selectors, CSV dropdowns, pattern input, advanced-regex toggle, pattern preview, tile-size stepper, CSV nudge, thumbnail route, focus-navigate matrix, four edge buttons, keyboard nav, position readout, pop-out), Results (Y dropdown, time-column selector, empty state, focus-navigate, edge buttons, tab re-attach, pop-out), Compare strip (Browse compare-strip, viewport-sync, multi-select, row-header, over-cap notice). **If a Results compare-strip row is `🔭 planned`** (Phase 4 deferred the Results compare e2e to its Task 8 follow-up), that is expected — `--strict` does not reject `🔭 planned`; do **not** flip it to `✅ shipping` unless `test_results_compare_strip.py` exists.

- [ ] **Step 2: WORKFLOWS.md round-trip gate**

```bash
uv run python scripts/check_workflows_md.py
```

Expected: pass. Verify holistically:
- The two timeline workflow rows exist — `browse_timeline` (*find ideal starting time*) and `results_timeline` (*trait emergence over time*).
- `_capture_browse_timeline` **and** `_capture_results_timeline` are both **defined** in `scripts/capture_gui_tutorial_screenshots.py` **and dispatched** from `capture_workflow_screenshots` (the AST walk requires the call, not just the def — verified gate behavior).
- `docs/source/_static/gui_images/browse_timeline/` and `docs/source/_static/gui_images/results_timeline/` each contain ≥1 `.png`.
- Tutorial pages `docs/source/tutorials/gui/<NN>_browse_timeline.md` (19) and `<NN>_results_timeline.md` (20) exist.
- No **orphan** `_capture_*` (e.g. a Compare-strip capture fn must NOT exist as its own row — Phase 4 folded the Compare screenshot **into** the two existing `_capture_*_timeline` fns, so there is no `_capture_compare_strip` row).

> **OQ-A (tutorial filename consistency — proceeding on the recommended default):** Phase 2's plan names the tutorial `docs/source/tutorials/gui/browse_timeline.md` (no `NN_` prefix) while every existing sibling uses `NN_*.md` (the highest today is `18_browse.md`) and Phase 3 uses `<NN>_results_timeline.md`. **Default (endorsed):** rename Browse's to `19_browse_timeline.md` and Results' to `20_results_timeline.md` so both follow the sibling convention. If a phase shipped the un-prefixed name, a docs-only rename here must update **three** things together — the gate (`check_workflows_md.py`) validates the page path *as written*, and a `_capture_*` that hardcodes its output filename/path would otherwise write a stale name:
> 1. the tutorial file itself (`git mv` to the `NN_` name);
> 2. the WORKFLOWS.md `Tutorial page` cell;
> 3. **the `_capture_browse_timeline` / `_capture_results_timeline` function** in `scripts/capture_gui_tutorial_screenshots.py` **if it hardcodes the page path/filename or the `_static/gui_images/<ID>/` output dir** — grep the fn body for the old name and patch it so the regenerated screenshot lands under the ID the WORKFLOWS row references. (If the capture fn derives its paths purely from the row `ID`, only 1+2 change — verify by reading the fn.)
>
> Confirm with the orchestrator before renaming, then re-run `check_workflows_md.py` + regenerate screenshots (Step 4).

- [ ] **Step 3: Confirm the two vendored `timeline.js` copies are byte-identical (pre-simplify baseline)**

Run: `uv run pytest tests/gui/results_viewer/timeline_view/test_assets.py::test_viewer_timeline_js_is_byte_identical_to_browse -v`
Expected: PASS. This is the guard the Task 3 simplify pass must not break.

- [ ] **Step 4: Regenerate the FULL screenshot set and commit wholesale**

Run: `uv run python scripts/capture_gui_tutorial_screenshots.py`

This regenerates **every** tutorial PNG (not just the timeline ones). Per CLAUDE.md, cross-surface font-rendering noise shifts unrelated PNGs by a few bytes — **commit them all**; do **not** cherry-pick or `git checkout --` the collateral. Then re-run the workflows gate to confirm the freshly-captured `browse_timeline/` and `results_timeline/` dirs are non-empty:

Run: `uv run python scripts/check_workflows_md.py`
Expected: pass.

> **Capture environment note (CLAUDE.md):** CI's `smoke-capture` regenerates on Ubuntu and uploads as an artifact for spot-checking, but committed PNGs should come from a developer/HPCC workstation, not CI, because of cross-platform font rendering. Regenerate here on the live HPCC session.

- [ ] **Step 5: Commit the regenerated screenshots (+ any docs-rename from OQ-A)**

```bash
git add docs/source/_static/gui_images src/phenotypic/gui/WORKFLOWS.md docs/source/tutorials/gui
git commit -m "docs(gui-timeline): reconcile FEATURES/WORKFLOWS gates + regenerate full screenshot set"
```

> If Step 1/2 found **no** drift and Step 4 produced only PNG churn, the only staged change is `docs/source/_static/gui_images/**`. If OQ-A required a rename, the WORKFLOWS.md + tutorial path also stage. If the gates reported a **real** orphan/duplicate/missing-row, STOP and report which surface phase owns it — do not silently author a new row in Phase 6.

---

### Task 3: End-of-feature `code-simplifier` pass over the whole-feature diff (process)

**Files (the simplify surface — the whole-feature diff):**
- `src/phenotypic/gui/_shared/timeline/` (engine: `_matrix.py`, `_thumbnail.py`, `_grid.py`, `__init__.py`)
- `src/phenotypic/gui/browse/` timeline files (`_timeline_records.py`, `_plate_pattern.py`, `_capture_time.py`, `_thumb_routes.py`, timeline `_layout`/`_callbacks` additions, `_ids.py` timeline ids) **+ `src/phenotypic/gui/browse/_assets/timeline.js`**
- `src/phenotypic/gui/results_viewer/timeline_view/` (`_ids.py`, `_layout.py`, `_grid.py`, `_thumb_routes.py`, `_callbacks.py`) **+ its vendored `src/phenotypic/gui/results_viewer/_assets/timeline.js`**
- `src/phenotypic/gui/_config.py` (timeline constants block)
- The CLI metadata-copy files: `src/phenotypic/sdk_/_io_constants.py`, `src/phenotypic/sdk_/__init__.py`, `src/phenotypic/_cli/_cli_output_manager.py`

**Why end-only:** This matches the user's decision (one simplify at the end, not per-phase — see the EXECUTION plan's "Simplify cadence" note). A single pass over the consolidated diff catches cross-phase reuse opportunities (e.g. a helper duplicated between Browse and Results) that a per-phase pass would miss.

- [ ] **Step 1: Capture the whole-feature diff range**

Run: `git log --oneline` and identify the first feature commit (the Phase 1 Task 0 `testpaths` commit or the first `feat(gui-timeline)` commit). Record `<feature-base>` (the commit *before* the feature started). The simplify target is `git diff <feature-base>..HEAD -- <the files above>`.

- [ ] **Step 2: Dispatch a `code-simplifier` (Opus) scoped to that diff**

Brief it explicitly:

> Simplify ONLY the Source Timeline View feature diff (range `<feature-base>..HEAD`), files: the `gui/_shared/timeline/` engine, the Browse timeline files **incl. `browse/_assets/timeline.js`**, the `results_viewer/timeline_view/` package **incl. `results_viewer/_assets/timeline.js`**, `gui/_config.py`'s timeline block, and the CLI metadata-copy files (`sdk_/_io_constants.py`, `sdk_/__init__.py`, `_cli/_cli_output_manager.py`). Apply **non-behavioral** reuse/clarity/dedup fixes only. **Reject any behavior change.**
>
> HARD CONSTRAINT: `browse/_assets/timeline.js` and `results_viewer/_assets/timeline.js` are kept **byte-identical** by a CI guard (`tests/gui/results_viewer/timeline_view/test_assets.py::test_viewer_timeline_js_is_byte_identical_to_browse`). If you touch one `timeline.js`, apply the **identical** edit to the other so the two stay byte-for-byte equal — or touch **neither**. Never edit one alone.
>
> Do NOT touch `gui/_config.py`'s non-timeline constants, the colony/QC code, or anything outside the named files.

- [ ] **Step 3: Review every proposed edit; apply only non-behavioral ones**

Read each diff hunk. Apply reuse/clarity/dedup fixes; **reject** anything that alters behavior (different sort, changed cache key, different error code, altered focus math, etc.). For any `timeline.js` change, verify the same bytes landed in both copies.

**Explicit reject criterion — the metadata-copy `try/except` is load-bearing (spec §8.3):** the simplifier must **NOT** collapse, remove, or narrow the `try/except Exception:` around the `shutil.copy` in `finalize_post_master_outputs` into anything that could let an exception propagate out of `finalize_post_master_outputs`. Spec §8.3 + Phase 5's Global Constraints require the copy to be **best-effort, failure-logged, never-raised** — a missing/unreadable source CSV (or any copy failure) must not abort finalize or lose the master/mirror outputs. Reject any "simplification" that (a) drops the broad `except Exception` for a narrower one, (b) hoists the copy out of the guard, or (c) removes the WARNING log. Clarity-only edits inside the guarded block (renaming a local, deduping a path-join) are fine **iff** the guard's catch-all-and-log shape is preserved verbatim.

- [ ] **Step 4: Re-prove the byte-equality guard after simplification**

Run: `uv run pytest tests/gui/results_viewer/timeline_view/test_assets.py::test_viewer_timeline_js_is_byte_identical_to_browse -v`
Expected: PASS. **If it fails, the two `timeline.js` copies drifted** — revert the asymmetric edit (apply to both or neither) and re-run.

- [ ] **Step 5: Commit the simplify pass (only if edits were applied)**

```bash
git add -A
git commit -m "refactor(gui-timeline): end-of-feature code-simplifier pass (non-behavioral)"
```

> If the simplifier proposed nothing worth applying, record "no simplify edits applied" and skip the commit — that is a valid outcome.

---

### Task 4: Full consolidated regression run (process)

**Files:** none (verification only).

The consolidated command list spans every phase. Run them in order; **all must be green** before Tasks 5–6.

- [ ] **Step 1: Timeline unit + CLI-copy unit tests (default lane, no browser)**

```bash
uv run pytest tests/gui/_shared/timeline tests/gui/browse tests/gui/results_viewer/timeline_view \
  tests/unit/sdk_/test_io_constants.py tests/unit/cli/test_cli_output_manager.py -q
```

Expected: PASS (engine, Browse, Results timeline unit suites + the Phase 5 unit tests). These also run in the default CI lane **provided** Phase 1 Task 0 added `tests/gui` to `testpaths` — Task 0 Step 2 confirmed that precondition; if it had been missing, this command's explicit paths still collect, but CI's default lane would silently skip the timeline unit tests.

- [ ] **Step 2: The CLI integration test (Task 1) + the broader CLI integration lane it lives in**

```bash
uv run pytest tests/integration/cli/test_cli_metadata_deliverable.py \
  tests/integration/cli/test_phenotypic_cache_layout.py -q
```

Expected: PASS (the new e2e metadata-copy test + the harness it mirrors, confirming no regression in the shared CLI run path).

- [ ] **Step 3: Playwright e2e — every interactive timeline surface (spec §16.9)**

```bash
PLAYWRIGHT=1 uv run pytest \
  tests/e2e/gui/test_browse_timeline.py \
  tests/e2e/gui/test_browse_compare_strip.py \
  tests/e2e/gui/test_results_timeline.py -v
```

Expected: PASS. Notes:
- `PLAYWRIGHT=1` is mandatory (the conftest module-skips the e2e lane without it — `tests/CLAUDE.md`).
- Phase 4's `test_browse_compare_strip.py` is marked `ci_flaky` (OSD-mount + tile-fetch budget on shared runners); it runs locally by default. CI skips it via `-m "not ci_flaky"`.
- `tests/e2e/gui/test_results_compare_strip.py` is a **deferred Phase-4 Task-8 follow-up** — include it here **only if it exists** on the branch; if absent, note it as deferred (do not author it in Phase 6).

- [ ] **Step 4: Lint the touched trees**

```bash
uv run ruff check src/phenotypic/gui src/phenotypic/_cli
```

Expected: clean.

- [ ] **Step 5: Type-check the timeline packages**

```bash
uv run mypy src/phenotypic/gui/_shared/timeline src/phenotypic/gui/browse src/phenotypic/gui/results_viewer/timeline_view
```

Expected: clean.

- [ ] **Step 6: The two docs/CI gates one more time (post-simplify, post-screenshot)**

```bash
uv run python scripts/check_features_md.py --strict
uv run python scripts/check_workflows_md.py
```

Expected: both pass.

- [ ] **Step 7: State the all-green bar**

Record the pass/fail of every command above (the actual output, not a summary — skill *verification-before-completion*). The feature is **regression-green** only when **all** of: timeline unit suites, the Phase 5 unit suite, the new CLI integration test, all timeline e2e (Browse + Browse-compare + Results), ruff, mypy, and **both** docs/CI gates (`--strict` FEATURES + WORKFLOWS) pass. Any red → STOP, triage, and fix before Task 5.

---

### Task 5: Live Playwright-MCP verification against the real reference data (process; spec §16.9)

**Files:** none (manual orchestrator gate; not CI, not pytest).

**Why this is a checklist, not a test:** It drives the **running** `phenotypic-gui` against **real** reference data — it needs a live server + on-disk reference images, neither available in CI. The committed pytest e2e (Task 4 Step 3) is the CI-enforced guard; this is the human/agent confirmation layer above it, confirming **real-image rendering + correct chronological ordering** that a fixture-only e2e cannot (spec §16.9). It runs at the **END** of execution, by the orchestrator, on a live HPCC session.

**Verified launch facts:**
- Entry point: `phenotypic-gui` → `phenotypic.gui.shell._launcher:main` (`pyproject.toml:131`).
- Flags: `--root <sandbox>` (required; the file browser is sandboxed to it), `--port <int>` (default 8050), `--host` (default loopback `127.0.0.1`), `--url-prefix`, `--debug` (`_launcher._build_parser` + `_config.add_launcher_args`).
- SSH-tunnel the port from a workstation: `ssh -L <port>:localhost:<port> user@cluster`.
- Reference data (verified to exist on this HPCC account):
  - Browse: `~/bigdata_exfab/ucr_029_e_d_Maresca/data/processed/` — a **flat** folder of **3663** TIFFs (e.g. `d000367_280_001_2026-04-06_12-06-57.tif`). A flat folder = a **single** dataset-folder row, so the **row axis needs a FILENAME PATTERN** (`{plate}` source) to split plates; `{time}` (or EXIF) supplies the time axis.
  - Results: `~/bigdata_exfab/ucr_029_e_d_Maresca/data/results/2026-06-16/` — the run whose mirror is `deliverables/measurements.parquet` (verified present). Set **X = `Metadata_ImageNumber`**, **Y = `Metadata_PlateNum`** — one overlay per `(PlateNum, ImageNumber)` cell, ~826 cells.

> **Sandbox-root note:** `--root` is the GUI's sandbox; point it at a directory that **contains** the reference tree so the in-app file browser / source-root picker can reach it. For Browse, `--root ~/bigdata_exfab/ucr_029_e_d_Maresca/data` then select `processed/` as the source image root in the sidebar; for Results, the viewer reads the output root selected in-app, so `--root ~/bigdata_exfab/ucr_029_e_d_Maresca/data` and open `results/2026-06-16/`. Confirm the exact in-app selection idiom against the running UI.

**Playwright MCP tools (verified available, namespace `mcp__plugin_playwright_playwright__`):** `browser_navigate`, `browser_snapshot`, `browser_click`, `browser_hover`, `browser_press_key`, `browser_take_screenshot`, `browser_wait_for`, `browser_evaluate`, `browser_console_messages`. Fetch their schemas via `ToolSearch("select:mcp__plugin_playwright_playwright__browser_navigate,...")` before first use (they are deferred tools).

> **Warm caveat (spec §15.8 — read before judging first-open latency):** the Browse run is over **~3663** TIFFs. On entering Timeline mode the background-warm sweep fires at **concurrency 2** over unique source images, each via the heavy RAW-aware `normalize_to_png` — so the *full* warm takes a long time, and first-open latency scales with the unique-source count (documented, not hidden). **Do NOT wait for full warm to call the surface working.** The §16.3 contract is that the **focus window mounts a BOUNDED set** (the focused neighborhood + the `TIMELINE_FOCUS_MARGIN` ring) regardless of total matrix size; confirm *that* — the centered window paints its bounded set of thumbnails promptly and arrow-navigation stays responsive — rather than blocking on the warm queue draining. Use `browser_evaluate` to confirm the mounted-`<img>` count stays bounded (≈ window + 2·margin) even while warm is still running.

- [ ] **Step 1: Browse surface — launch + drive**
  - Launch: `uv run phenotypic-gui --root ~/bigdata_exfab/ucr_029_e_d_Maresca/data --port <port>` (background it); SSH-tunnel `<port>`.
  - `browser_navigate` to the Browse tab; switch the `Single | Timeline` toggle to **Timeline**.
  - Set the **row source = Filename pattern** with a `{plate}` (+ `{time}`) pattern that splits the `d000367_280_001_2026-04-06_12-06-57.tif`-style names into plate vs. timestamp (use the live **pattern preview** to confirm matched/unmatched counts before trusting the grid). Set the **time source** to the `{time}` capture or EXIF.
  - Navigate focus with the **arrow keys** (←/→ along time, ↑/↓ along plates) **and** the on-edge **◀ ▶ ▲ ▼** buttons. Confirm: focus highlight moves; the centered window stays **full at the corner** (clamp-translate — the window does not shrink at matrix bounds); far cells are **offloaded** (`<img>` removed) while the **margin ring is pre-mounted** (cells just outside the viewport already carry an `<img>`). Use `browser_evaluate` to count mounted `<img>` vs. placeholder `<div data-src>` if a visual check is ambiguous.
  - Open a **deep-zoom pop-out**: press **Enter** on the focused cell, and separately **hover** a visible tile and click the revealed **⤢** button. Confirm the OSD modal opens on the real image.
  - Open the **Compare strip** on a **row** (click an axis row-header) and confirm **synced pan/zoom** (pan/zoom one viewer → peers follow).
  - `browser_take_screenshot` of: the focus-navigated grid (real TIFF thumbnails visible), a corner-clamped window, the pop-out, and the synced Compare strip.

- [ ] **Step 2: Results surface — launch + drive**
  - Point the viewer at `results/2026-06-16/` (the mirror at `deliverables/measurements.parquet`).
  - Open the **Timeline** tab (6th tab). Set **Y = `Metadata_PlateNum`** (74 plates — selectable because the Y axis is uncapped, spec §16.5) and **X = `Metadata_ImageNumber`** (Int64 1..24, strictly monotonic — spec §16.6).
  - Navigate focus with arrows + edge buttons; confirm the same clamp-translate / margin-ring / offload behavior over **overlay** images.
  - Open a pop-out (Enter + hover-⤢) on a real overlay.
  - `browser_take_screenshot` confirming **REAL overlay rendering + correct chronological ordering** — scan one plate's row and confirm **ImageNumber 1→24** reads left-to-right (the assertion a fixture-only e2e cannot make).

- [ ] **Step 3: Record the verification**
  - Archive the screenshots to a **dated scratch dir kept OUT of `docs/` and out of git** — e.g. `~/timeline_mcp_verification/2026-06-18/` (or `/tmp/...`). These are verification *evidence*, NOT committed tutorial assets (the committed tutorial PNGs come from Task 2 Step 4's capture script). Do not stage them; do not place them under `docs/source/_static/`.
  - Note any console errors (`browser_console_messages`).
  - Write a short orchestrator note that **references the dated scratch-dir path** and lists: surfaces driven, behaviors confirmed (focus-navigate, clamp-translate corner, margin-ring pre-mount + bounded offload **while warm still runs** per the §15.8 caveat, pop-out, Compare sync, chronological ordering 1→24), and any discrepancy vs. the e2e expectations. **This note (with its scratch-dir reference) is the gate's deliverable** — it confirms the live, real-data behavior above the CI-enforced pytest e2e.

> This gate cannot run in CI (live server + real data). It is the final orchestrator/human confirmation layer; it does not commit anything.

---

### Task 6: Finish the branch (process)

**Files:** none (the finishing-a-development-branch workflow handles the PR / hand-off).

- [ ] **Step 1: Confirm every gate is green**
  Confirm: Task 1 integration test passes; Task 2 docs/CI gates pass (`--strict` + WORKFLOWS); Task 3 simplify applied (or recorded as no-op) with the byte-equality guard still green; Task 4 full regression all-green bar recorded; Task 5 live-MCP note written.

- [ ] **Step 2: Hand off via the finishing workflow**
  Use the `superpowers:finishing-a-development-branch` skill to present the integration options (merge / PR / cleanup). Per the standing rules, this phase does **not** itself run `git push` / open the PR — the finishing workflow guides that with the user. The branch is **`worktree-source-timeline-view`** (the worktree at `/bigdata/exfab/anguy344/PhenoTypic/.worktrees/worktree-source-timeline-view`); the PR body should summarize the full feature (shared engine + Browse + Results + Compare strip + CLI metadata copy + docs/CI) and link the spec `docs/superpowers/specs/2026-06-18-source-timeline-view-design.md`.

---

## Phase 6 deliverable

- A single **CLI end-to-end integration test** (`tests/integration/cli/test_cli_metadata_deliverable.py`) proving the real `phenotypic_cli --metadata` wiring produces `deliverables/metadata.csv` byte-matching the source — the assertion Phase 5 deferred.
- A **holistic docs/CI reconciliation**: `check_features_md.py` (plain + `--strict`) and `check_workflows_md.py` pass across all the timeline rows (Browse, Results, Compare); the WORKFLOWS round-trip (`_capture_browse_timeline` + `_capture_results_timeline` defined + dispatched, `_static/gui_images/{browse,results}_timeline/` populated, tutorial pages 19 + 20 present); and the **full** screenshot set regenerated + committed wholesale.
- A single **end-of-feature `code-simplifier` pass** over the whole-feature diff (engine + Browse + Results + both vendored `timeline.js` + `_config.py` timeline block + CLI metadata-copy files), non-behavioral only, with the two `timeline.js` copies kept byte-identical.
- A **full consolidated regression** all-green bar (timeline unit + Phase 5 unit + the new CLI integration test + Browse/Compare/Results e2e + ruff + mypy + both docs/CI gates).
- A **live Playwright-MCP verification** against the real UCR_029 reference data (Browse over `data/processed/`, Results over `data/results/2026-06-16/` with X=`Metadata_ImageNumber`, Y=`Metadata_PlateNum`) confirming real-image rendering + chronological ordering — the manual orchestrator gate above CI.
- The branch handed off for PR via the finishing-a-development-branch workflow.

## Open Questions (need a human decision)

- **OQ-A — tutorial filename convention (PROCEEDING on the default).** Phase 2's plan names its tutorial `docs/source/tutorials/gui/browse_timeline.md` (no `NN_` prefix), but every sibling uses `NN_*.md` (highest today `18_browse.md`) and Phase 3 uses `<NN>_results_timeline.md`. **Default (endorsed):** `19_browse_timeline.md` + `20_results_timeline.md` (next two integers after 18). The rename is a docs-only change done in Task 2 Step 2's OQ-A note **if** a phase shipped the un-prefixed name, and it must update **three** things together (the file, the WORKFLOWS `Tutorial page` cell, **and** any hardcoded path/filename in the `_capture_*` fn) — `check_workflows_md.py` validates the path as written.
- **OQ-B — Results Compare-strip e2e.** Phase 4 deferred `tests/e2e/gui/test_results_compare_strip.py` (+ the Results compare FEATURES rows flipping `🔭 planned → ✅ shipping`) to its Task-8 follow-up. **Recommended default:** if that follow-up has NOT landed by Phase 6, leave the Results compare rows `🔭 planned` (the `--strict` gate accepts it) and note the deferral; do **not** author the Results compare e2e inside Phase 6 (out of this phase's scope). Decide whether the feature ships with Browse-only Compare e2e coverage or blocks on the Results follow-up.
- **OQ-C — live-MCP gate operator (PROCEEDING on the default).** Task 5 needs a live HPCC session with the reference data and an SSH-tunnelled port. **Default (endorsed):** the orchestrator runs it interactively at the very end (it is the explicit user-requested MCP confirmation, spec §16.9). Two specifics now baked into Task 5: (1) the Browse run over ~3663 TIFFs triggers the §15.8 background-warm at concurrency 2 over unique sources → slow first-open; the operator confirms the **bounded focus-window mount** (§16.3) rather than waiting for full warm; (2) the verification screenshots are archived to a **dated scratch dir kept out of `docs/` and out of git** (referenced in the orchestrator note), distinct from the committed tutorial PNGs.
- **OQ-D — simplify scope on the CLI files (PROCEEDING on the default).** Task 3 includes the Phase 5 CLI metadata-copy files in the simplify surface. **Default (endorsed):** include them (they are part of the feature diff), but the bar is high — the copy block is already minimal/guarded, so expect near-zero edits there. Task 3 Step 3 now carries an **explicit reject criterion**: the simplifier must not collapse/narrow/hoist the `try/except Exception` guard around `shutil.copy` in any way that could let an exception escape `finalize_post_master_outputs` (spec §8.3 best-effort/failure-logged/never-raised).
