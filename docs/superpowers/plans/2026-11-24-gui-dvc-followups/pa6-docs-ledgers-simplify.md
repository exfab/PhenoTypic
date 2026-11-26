# PA6 · Docs, ledgers, tutorial text, screenshots, simplify

**Spec:** PA7 (D4, D11, D17, N7), PA8 "Docs that name the file", N3's exit-5 surfaces; D15, D18

---

**Goal:** Every doc, ledger and tutorial page describes the shipped behaviour; screenshots are
regenerated; one simplify pass.

**Depends on:** PA1–PA5 merged on the main line.

### Task 1: docs and the exit-5 surfaces (C12)

**Files:** exactly spec PA8's list, plus N3's:
- `CLAUDE.md` ("Output layout": `measurements.{tsv,parquet}`);
- `src/phenotypic/_cli/CLAUDE.md:663, 697-745, 777, 1549, 1659` and its output inventory
  (`:1659`'s "Two generated CSVs and one input snapshot stay CSV" becomes "one generated CSV,
  `error_analysis.csv` (D18), and one input snapshot stay CSV"); the `phenotypic-wait` bullets
  say `--accept-failures` maps only `3` (D15), and that after cancelling a SLURM run and
  continuing it locally, the CLI's own exit code is the one to use, not `phenotypic-wait`'s
  (spec R15; the same line goes in `dvc.md`);
- `src/phenotypic/_gui/CLAUDE.md:257, 324, 427`, and the readers rule gains
  `run_finished_with_failures`;
- `docs/source/contrib_guide/tracked_state.md` (check every row PA1/PA5 added is present);
- `docs/source/how_to/pages/dvc.md:76-94, 363-392` (the "until #283" caveat removed; "curate
  after the last `dvc repro`", D11; the exit-code table's `5` row and the text about `3`, N3);
- `docs/source/api_reference/cli_reference.rst`, `how_to/pages/{reference_metadata,migrate_ome_zarr,rembi_metadata,generate_reports}.md`,
  `tutorials/notebooks/07_measuring_and_exporting.ipynb` (edited as JSON text, never executed).

**Interfaces:** none.

- [ ] **Step 1: Make the edits.** Wording for exit `5` everywhere: "no job is alive, and the
  run ended without finishing: nothing was published, accepted images were never attempted,
  or the published outputs no longer cover the run (for example, a store the finalizer could
  not project was left out)".
- [ ] **Step 2: Check no stale mention survives:**
  `grep -rn 'measurements\.csv\|until #283\|json\.pht-\|reads `4` as a failed' CLAUDE.md src/phenotypic/*/CLAUDE.md docs/source`
  returns only lines that describe older trees or legacy reads.
- [ ] **Step 3: Docs build on Slurm** (`batch`, 16 CPU, 32 GB, 1 h, detached worktree):
  `uv run sphinx-build -j "$SLURM_CPUS_PER_TASK" -D nbsphinx_execute=never -b html docs/source <out>`.
  Then read the rendered HTML of `dvc`, `generate_reports` and the notebook page.
- [ ] **Step 4: Commit.**

### Task 2: ledgers, tutorial text, screenshots (C13)

**Files:**
- `src/phenotypic/_gui/FEATURES.md`: spec PA7's new rows and row updates (`:53, 54, 92, 114,
  123, 356, 371, 433, 451, 456, 486, 496, 512, 522, 527, 537, 585, 586, 604-622, 626`). Each
  `✅ shipping` row's `Test ref` names a test PA1–PA5 added.
- `src/phenotypic/_gui/WORKFLOWS.md`: row descriptions only where they name a changed state
  or file (no new row).
- `docs/source/tutorials/gui/{01_setup,04_run_local,05_run_slurm,06_view_results}.md`: spec
  PA7's paragraphs (status, `--retry-failures` relaunch and N7's "resume from the CLI", badge,
  curation, "curate after the last re-run", and D17's "never run the same output from the CLI
  and the GUI at once"). The same D17 sentence goes in `docs/source/how_to/pages/dvc.md` and
  `src/phenotypic/_gui/CLAUDE.md`'s run-console section.
- `docs/source/_static/gui_images/**`: regenerated.

- [ ] **Step 1: Edit the ledgers and the four tutorial pages.**
- [ ] **Step 2: Run the ledger gates:** `uv run python scripts/check_features_md.py --strict`
  and `uv run python scripts/check_workflows_md.py -v`; both exit 0.
- [ ] **Step 3: Regenerate the full screenshot set** (a Slurm job on `batch` with a headless
  Chromium; `gui-tutorial-capture` skill): `uv run python scripts/capture_gui_tutorial_screenshots.py`.
  Commit every regenerated PNG, collateral included. No frame is added for
  `complete_with_failures` (D4).
- [ ] **Step 4: Playwright:** `PLAYWRIGHT=1 $PT tests/e2e/gui -m "not slow and not ci_flaky"`
  (Slurm). Tune specs do not count toward this gate (the app is unmounted).
- [ ] **Step 5: Commit.**

### Task 3: simplify (C14)

- [ ] **Step 1:** dispatch `code-simplifier` (Opus) over `git diff <PR A base>..HEAD -- src/`:
  quality only, no behaviour change.
- [ ] **Step 2:** apply its fixes; run the affected surface (importers of every module it
  touched) on Slurm; mypy shows no new error.
- [ ] **Step 3:** commit.
