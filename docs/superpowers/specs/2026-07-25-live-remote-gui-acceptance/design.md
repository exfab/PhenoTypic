# Live Remote GUI Acceptance Specification

**Date:** 2026-07-25  
**Target:** `https://4qbp9pqt-8050.usw3.devtunnels.ms/`  
**Cluster:** `cluster.hpcc.ucr.edu`  
**Project root:** `/rhome/anguy344/bigdata_exfab/ucr_029_e_d_Maresca`  
**Results mutation fixture:** `data/results/2026-07-16-test-gui`

## 1. Purpose

Re-test the deployed GUI after updating the remote `debug-gui` checkout. This
acceptance run covers every shipped GUI workflow in `WORKFLOWS.md`, with
additional end-to-end checks for the reliability contracts introduced by the
2026-07-23 GUI reliability specification.

This is a live-system acceptance test, not a substitute for the committed
automated suite. Browser behavior, remote filesystem evidence, scheduler
evidence, and user-visible command text must agree.

## 2. Status and evidence rules

- `[ ]` not started
- `[~]` in progress
- `[x]` passed with evidence
- `[!]` failed; issue recorded
- `[-]` blocked or intentionally skipped, with reason

For every state-changing test, record:

1. the exact selected input, pipeline/spec, and output;
2. the GUI-visible generation or status transition;
3. the resulting remote artifacts and scheduler state;
4. cleanup state, without deleting or modifying unrelated data.

Major issues are P0/P1 failures, destructive or source-mutating behavior,
incorrect scheduler state, stale path authority, silent fallback, lost
configuration, partial publication, or an advertised workflow that cannot
complete.

## 3. Safety boundaries

### 3.1 Protected state

- Do not select, cancel, overwrite, migrate, curate, or publish into any active
  processing output.
- Do not use a `latest`, production, or date-named result directory as a Run or
  Tune output.
- Do not alter the source completed run from which
  `data/results/2026-07-16-test-gui` was copied.
- Do not cancel scheduler jobs unless their output belongs to this acceptance
  run and their generation was created by this test.
- Do not recursively remove any project, data, results, or user directory.

### 3.2 Allowed writes

- Results/QC/Error/Analysis mutations are allowed only inside
  `data/results/2026-07-16-test-gui`.
- Run and Tune jobs use a unique acceptance namespace confirmed during the
  preflight, with one-image or otherwise minimal input.
- GUI-owned presets, external viewer caches, logs, and generation records may
  be written under their documented `.phenotypic-gui`, `.phenotypic`, or Tune
  cache locations.

### 3.3 Stop conditions

Stop state-changing work if:

- the active production output cannot be distinguished from the test output;
- the scheduler job identity cannot be tied to the GUI test generation;
- the GUI resolves a typed or picked path outside the sandbox;
- the results copy is not independently writable;
- a callback targets the source run instead of the test copy;
- scheduler cancellation cannot be fenced to test-owned job IDs.

## 4. Test fixtures

The preflight must fill in this table before browser mutations begin.

| Purpose | Path or identity | Required property | Verified |
| --- | --- | --- | --- |
| GUI sandbox | `/rhome/anguy344/bigdata_exfab/ucr_029_e_d_Maresca` | Matches shell settings | `[x]` |
| Results copy | `data/results/2026-07-16-test-gui` | Writable and not active; completion consistency checked separately | `[x]` |
| Minimal images | `gui_e2e_acceptance/b7879e1e2deeff681cd085445a9391aa510ce657/cases/gui-v1-live-44208909b63f4442b36ff5d261f09546/input/single-small-colony.tiff` | Existing 96×96, 28,020-byte one-image fixture | `[x]` |
| Secondary image | `data/gui_e2e_test_inputs/slurm_smoke.png` | Existing 581,853-byte smoke fixture | `[x]` |
| CPU pipeline | `gui_e2e_acceptance/b7879e1e2deeff681cd085445a9391aa510ce657/cases/gui-v1-live-44208909b63f4442b36ff5d261f09546/pipeline.json.pht-pipe` | 419-byte `OtsuDetector` + `MeasureSize` pipeline | `[x]` |
| Minimal staged pipeline | `gui_e2e_acceptance/a51f9620feae229b6c78a7cd697cb41833b4a809/cases/gui-v1-live-b40dfad3ce15407a96ca1d1a071e67ba/pipeline.json.pht-pipe` | 667-byte test-owned `FakeGpuDetector` + `MeasureSize` pipeline; requires preload below | `[x]` |
| Minimal staged image | `gui_e2e_acceptance/a51f9620feae229b6c78a7cd697cb41833b4a809/cases/gui-v1-live-b40dfad3ce15407a96ca1d1a071e67ba/input/single-small-colony.tiff` | One-image staged fixture | `[x]` |
| Staged preload | `PHENOTYPIC_PRELOAD_MODULES=tests._fakes.register_fake_gpu` | Test module importable by GUI and workers | `[ ]` |
| Production GPU pipeline | `config/UCR_029_E_D_Maresca_v12.json.pht-pipe` | SAM2/staged; do not use except for read-only inspection | `[x]` |
| Tune spec/layout | To be chosen after revision gate | Canonical typed spec and metadata | `[ ]` |
| Run output namespace | `gui_e2e_acceptance/9e646c16078bf969/live-20260725-<uuid>/` | New, unique, outside production results | `[x]` |
| Active protected job | SLURM `26725257`, interactive allocation on `gpu12` | Never cancel or replace | `[x]` |
| Protected code workdir | `/bigdata/exfab/anguy344/PhenoTypic` | GUI checkout and active allocation workdir | `[x]` |
| Protected production run | `data/results/2026-07-16` and every other production result | Never mutate or resume | `[x]` |

## 5. Acceptance checklist

### A. Preflight and deployment identity

- [!] **A01** Confirm the remote checkout contains final commit
  `9e646c16078bf969d80227a8bb624381ee437e3b`.
- [-] **A02** Confirm the GUI process serves the updated Run controls and
  canonical typed-file extensions.
- [x] **A03** Record active SLURM jobs, their output roots, and protected
  generations before testing. `squeue` showed exactly one job: interactive GUI
  allocation `26725257` on `gpu12`; no batch or array processing job was
  active.
- [!] **A04** Validate the completed-run copy layout, permissions, manifest,
  measurements mirror, pipeline recipe, overlays, QC state, and completion
  evidence.
- [x] **A05** Confirm the dedicated Run/Tune output namespace is absent or
  empty and cannot resolve to an active output.
- [-] **A06** Restart the controlled GUI with
  `PHENOTYPIC_PRELOAD_MODULES=tests._fakes.register_fake_gpu`, confirm the
  module imports in the GUI/worker environment, and prove the minimal staged
  pipeline deserializes before G08.

### B. Shell, Home, and file explorer

- [ ] **B01** Home loads with shell chrome, grouped Pipeline/Results
  navigation, RSS readout, sandbox label, and capability counts.
- [ ] **B02** Help and Settings open and close without layout or callback
  errors.
- [ ] **B03** Input-folder picker writes a valid shared V2 source payload;
  Clear removes browser authority without deleting files.
- [ ] **B04** Metadata picker accepts the intended CSV and rejects
  non-CSV/out-of-sandbox paths.
- [ ] **B05** Sidebar expands/collapses lazily and shows accurate
  `img`/`cfg`/`out`/bundle badges.
- [ ] **B06** Hidden and external-symlink toggles change visibility without
  escaping the sandbox.
- [ ] **B07** Refresh updates badges, open pickers, source labels, and page
  inputs through one shared revision.
- [ ] **B08** Sidebar handoff offers only context-valid actions and does not
  treat stale labels as selected paths.
- [ ] **B09** Recent Runs excludes private legacy backups and reports
  incomplete generation-less historical outputs as `unknown`, not `running`.
- [ ] **B10** Navigation among all mounted apps preserves active-group styling
  and does not produce duplicate IDs or blank mounts.

### C. Browse workflows

- [ ] **C01** Selecting the minimal image directory populates dataset and image
  controls and loads the first image.
- [ ] **C02** Previous/next controls clamp correctly; image dimensions, size,
  and available metadata render.
- [ ] **C03** Deep zoom, pan, home, and full-page controls work without a CDN.
- [ ] **C04** Compare mode mounts the selected images, enforces its cap, and
  propagates linked pan/zoom.
- [ ] **C05** Timeline mode exercises at least one folder/pattern row source
  and one CSV-backed row source when metadata exists, including placeholder or
  advanced-regex preview and join/warning feedback.
- [ ] **C06** Timeline time-source selection exercises EXIF/folder/pattern/CSV
  options that are available, then builds a matrix, focuses the first
  populated cell, navigates by arrows/buttons, and keeps a bounded mounted
  window.
- [ ] **C07** Timeline tile-size controls and row-header comparison work.
- [ ] **C08** Timeline hover/Enter opens and reopens the deep-zoom popout.
- [ ] **C09** Returning to Browse after shared Refresh retains a valid source
  or clearly reports it unavailable.

### D. Builder workflows

- [ ] **D01** Builder loads the canonical pipeline through the picker and
  preserves the typed extension.
- [ ] **D02** Synthetic and real-image source paths both render a usable input
  node; point-picker selection round-trips when offered.
- [ ] **D03** Palette insertion builds a linear chain; zoom, fit, selection,
  inspector, and documentation controls work.
- [ ] **D04** Scalar operation-valued aux targets accept, replace, clear, and
  drill into compatible operations.
- [ ] **D05** Embedded pipeline aux creation, breadcrumb drill-in, nested
  editing, and drill-out work.
- [ ] **D06** Required-side-value and whole-pipeline validation states identify
  the correct operation without destructive repair.
- [ ] **D07** Run Preview publishes a complete generation and keeps the preview
  DOM mounted across reselection.
- [ ] **D08** Editing a parameter marks the old preview stale; rerun replaces
  it atomically with the new revision.
- [ ] **D09** Save writes a canonical `.json.pht-pipe`; Load round-trips the
  saved pipeline without shared-instance aliasing.
- [ ] **D10** Unsupported nonlinear/development DAG input fails closed with a
  recovery explanation instead of silent data loss.

### E. Tune co-pilot

- [ ] **E01** Setup loads a pipeline or existing `.json.pht-tune` spec and
  preserves existing strategy, budget, storage, scorer, and extensions.
- [ ] **E02** Search-space editors expose supported domains, preserve typed
  values, and block invalid/no-knob configurations.
- [ ] **E03** Metadata-backed scorer replacement is explicit; credentials are
  not rendered into browser-visible state or commands.
- [ ] **E04** Continue writes an atomic canonical spec and switches to Run.
- [ ] **E05** Run source/output/strategy/budget/storage/compute/evaluation
  controls produce one valid launch command.
- [ ] **E06** Copied command text and Deploy argv are identical after shell
  parsing/redaction rules.
- [ ] **E07** A minimal Local Tune deployment reaches terminal state and
  produces a bindable Tune output.
- [ ] **E08** Monitor polls progress, binds through the read-only run picker,
  and exercises Local-only cancel behavior or its precise non-local fallback.
- [ ] **E08a** Curate shortlists and pins A/B trials, renders linked overlays
  and difference view, propagates pan/zoom, and selects a winner.
- [ ] **E08b** Space toggles tunable knobs, edits supported domains, exports a
  canonical next spec, and Launch refreshes to a valid command using it.
- [ ] **E09** Best-pipeline export writes a canonical pipeline without
  modifying the source spec.
- [ ] **E10** SLURM Tune mode either completes a minimal test-owned deployment
  or is marked blocked with precise scheduler/UI evidence.

### F. Run Console, Local, and Validate

- [ ] **F01** Pipeline/input/output pickers accept only sandbox-valid paths and
  show canonical typed files.
- [ ] **F02** Rapid Local/SLURM mode changes use the final visible controls, not
  stale derived state.
- [ ] **F03** Dry-run, Resume, metadata, canonical image extensions, and
  advanced fields appear in the generated request exactly once.
- [ ] **F04** Save/Load preset round-trips all CPU, GPU, staged, and SLURM
  controls.
- [ ] **F05** Validate records before launch, streams logs, reaches a terminal
  registry state, and publishes no run output.
- [ ] **F06** Minimal Local run records a unique generation before spawn,
  streams incremental logs, and reaches completion only with matching
  publication evidence.
- [ ] **F07** Local cancellation affects only the test generation and reaches a
  terminal state after the process is inactive.
- [ ] **F08** Recent Runs refreshes by registry revision; row selection points
  the dashboard iframe at the correct output.
- [ ] **F09** Fresh-output ownership rejects accidental reuse; Resume is
  explicit and generation-checked.
- [ ] **F10** A pre-seeded process-mode output is classified and shown in
  Recent Runs without inventing a full-run dashboard.

### G. Run Console and live SLURM

- [ ] **G01** SLURM form accepts minutes, `HH:MM:SS`, and `D-HH:MM:SS`, while
  rejecting malformed durations and an empty profile.
- [ ] **G02** Ordinary one-image submission persists intent before `sbatch`,
  records array/finalizer roles, and shows queued/running/reconciling states.
- [ ] **G03** Scheduler logs stream incrementally and polling remains bounded.
- [ ] **G04** Ordinary completion requires every ledgered job, finalizer
  success, generation marker, and complete manifest.
- [ ] **G05** Test-owned ordinary cancellation fences submission, cancels every
  recovered ID, and remains `cancelling` until quiescent.
- [ ] **G06** GUI restart/Refresh rehydrates a nonterminal test generation from
  durable owner, intent, role ledger, and scheduler state.
- [ ] **G07** Scheduler-unavailable behavior becomes `unknown`, never a false
  terminal or running state.
- [ ] **G08** Staged GPU one-image submission records controller, recovery,
  CPU-stage, GPU-stage, continuation, and finalizer roles as applicable.
- [ ] **G09** Staged completion requires orchestration completion, per-image
  Stage-3 evidence, matching epoch, and cleanup of transient sidecars.
- [ ] **G10** Staged cancellation deactivates the epoch before `scancel` and
  leaves no test-owned continuation active.
- [ ] **G11** No job or output belonging to the protected active run changes
  throughout G01-G10.

### H. Results binding and read-only views

All H items are currently blocked by A01. H01-H10 read-only behavior may be
tested after the revision gate; mutation-dependent interpretations remain
blocked by A04 until the fixture is authoritative.

- [-] **H01** Binding `data/results/2026-07-16-test-gui` validates the layout
  and atomically updates Results and Analysis to one snapshot.
- [-] **H02** Binding, tab activation, first tile, Refresh, and compatibility
  preflight do not write under the bound output.
- [-] **H03** Snapshot header shows path, timestamp, fingerprint, and
  Current/Stale/Active status accurately.
- [-] **H04** Refresh atomically swaps Results and Analysis; a failed or
  superseded Refresh preserves the prior coherent sessions.
- [-] **H05** Plate view image picker, stepper, deep zoom, details, filters,
  cards, and measurement table work.
- [-] **H06** Colony view grid, axis/group controls, tile-size controls,
  selection, curation labels, lock views, and crop routes work.
- [-] **H07** Standalone/full-run layer affordances match available artifacts;
  missing layers fail clearly.
- [-] **H08** Heatmap measurement/QC color choices, aggregation, image
  previous/next controls, time slider, and removed-cell styling work.
- [-] **H09** Results Timeline axis pickers, matrix navigation, bounded mount,
  tab re-entry, and deep-zoom popout work.
- [-] **H10** An active/nonterminal owner makes every Results and Analysis
  mutation fail closed.

### I. QC configuration, migration, rebuild, and review

All I items are blocked by A01 and A04. Do not mutate the inconsistent copy.

- [-] **I01** Compatibility preflight classifies the copied recipe as
  compatible, migratable, or blocked with exact reasons and fingerprints.
- [-] **I02** If migratable, explicit migration rechecks the source, creates an
  exact backup/receipt, writes atomically, and is idempotent.
- [-] **I03** Add/edit/duplicate/toggle/delete Count and SE checks update only
  the copied pipeline recipe.
- [-] **I04** QC changes wait for explicit `Rebuild QC database`; rebuild uses
  the current measurements mirror and validates `qc.duckdb`.
- [-] **I05** Configure summary counts and status badges refresh after rebuild.
- [-] **I06** Review selects a module, shows worst-first members, navigates
  groups, and preserves splitter state.
- [-] **I07** Curation on the copy updates shared removed/category state and
  recomputes before/after metrics without touching the source run.
- [-] **I08** Mark reviewed persists in `deliverables/qc/review_state.json`;
  vanished keys reconcile safely.

### J. Error analysis and transactional publication

All J items are blocked by A01 and A04. Do not publish into the inconsistent
copy.

- [-] **J01** Opening Error and switching categories is compute-only and leaves
  canonical artifacts byte-identical.
- [-] **J02** Baseline, measurement, direction, cutoff, score table, pagination,
  and plot controls work for every category.
- [-] **J03** Publish remains disabled or blocked when source fingerprints,
  compatibility, ownership, or required category state is invalid.
- [-] **J04** `Publish all categories` writes one complete transactional
  generation with every category and a receipt.
- [-] **J05** Refresh/reopen reads the published generation without partial or
  stale category state.

### K. Analysis authoring and publication

All K items are blocked by A01 and A04. Do not save or publish into the
inconsistent copy.

- [-] **K01** Analysis opens on the same Results snapshot and displays pipeline
  summary, compatibility warnings, and missing-column diagnostics.
- [-] **K02** Post, filter, edge, model, and plot sections render and preview
  without writing on activation.
- [-] **K03** Known edits preserve opaque/unknown sibling nodes exactly;
  explicit replacement drops only the replaced node's extensions.
- [-] **K04** Save rechecks the source fingerprint and writes the copied recipe
  atomically; stale concurrent edits fail closed.
- [-] **K05** Recompile guidance uses current output paths and command forms.
- [-] **K06** Analysis publication produces all configured class-named
  artifacts and plot outputs as one guarded generation.
- [-] **K07** Results Refresh exposes the new coherent Analysis generation.

### L. Final cross-app and filesystem verification

- [ ] **L01** V1 payloads remain readable, but a sandbox relocation or
  fingerprint mismatch makes both stored V1/V2 source and metadata descriptors
  unavailable and non-authoritative until explicit reselection; reselection
  writes V2 path/fingerprint payloads.
- [ ] **L02** Shared source, metadata, output binding, and Refresh propagate
  consistently across Browse, Builder, Tune, Run, Results, and Analysis.
- [ ] **L03** Pipeline, Tune, and image-extension displays use canonical
  extensions everywhere.
- [ ] **L04** Copy/deploy command parity holds for Run and Tune.
- [ ] **L05** Bound source trees remain unchanged by read-only interactions;
  only approved test outputs and explicit copied-results mutations change.
- [ ] **L06** Every submitted test job is terminal and no test-owned scheduler
  continuation remains.
- [ ] **L07** Active production jobs and outputs match their preflight state
  except for changes produced by their own pre-existing processing.
- [ ] **L08** Record every major issue below with reproduction, affected path,
  evidence, severity, and recommended fix.

## 6. Findings

### F-001: Remote GUI is not running the requested final revision

**Severity:** P0 acceptance blocker  
**Checklist:** A01, A02, and every browser test after preflight

Read-only SSH inspection found:

- remote checkout: `/bigdata/exfab/anguy344/PhenoTypic`;
- remote branch: `debug-gui`, clean and tracking `origin/debug-gui`;
- remote HEAD: `4c2b792fdce0bb90e7427c8998a84f34e13c6380`;
- required local HEAD:
  `9e646c16078bf969d80227a8bb624381ee437e3b`;
- local branch state before this specification: 12 commits ahead of
  `origin/debug-gui`.

The remote `git pull` succeeded only against the older published branch. It
could not retrieve the unpushed implementation commits. Browser acceptance
must not proceed because failures would test the wrong program.

**Required resolution:** publish the intended branch revision, fast-forward the
remote checkout, and restart the GUI process in a controlled way without
cancelling allocation `26725257`. The controlled restart must also satisfy A06
for staged acceptance. Then rerun A01, A02, and A06 before any other item.

### F-002: The copied Results fixture has contradictory completion evidence

**Severity:** P1 fixture/data consistency issue  
**Checklist:** A04, H01-H10, I01-I08, J01-J05, K01-K07

The copy exists, is owned by `anguy344:exfab`, has directory mode `2750`, no
symlinks, approximately 44,857 files, and disk usage near 475 GB. The current measurements,
master measurements, pipeline, manifest, and finalization marker are
byte-identical to `data/results/2026-07-16`.

However:

- the manifest reports `total=2970`, `completed=5940`, `failed=496`,
  `success_rate=0.922933`, and `is_complete=false`;
- a staged finalization-complete marker exists;
- `slurm_info` still embeds the original
  `data/results/2026-07-16` output path;
- current artifacts include 2,970 measurements/overlays/Stage-3 markers,
  3,218 HDF files, and 248 objmaps;
- copied backup trees from prior compatibility work are also present.

This is useful for stale-path and compatibility testing, but it does not satisfy
the clean-completed-output precondition for mutation acceptance. The viewer
should expose the conflict and fail closed. QC, Error, and Analysis mutation
tests need either an authoritative repair through supported GUI actions or a
separate coherent completed-run copy. The copied run must never be resumed.

## 7. Completion summary

Preflight stopped after A05 because A01 is a P0 version-identity failure. No
browser mutation, Results write, job submission, cancellation, or cleanup
action was performed. Active allocation `26725257` and all production outputs
were left untouched.
