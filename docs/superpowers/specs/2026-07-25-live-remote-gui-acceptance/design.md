# Live Remote GUI Acceptance Specification

**Date:** 2026-07-25  
**Target:** `https://4qbp9pqt-8050.usw3.devtunnels.ms/`  
**Cluster:** `cluster.hpcc.ucr.edu`  
**Project root:** `/rhome/anguy344/bigdata_exfab/ucr_029_e_d_Maresca`  
**Results read-only acceptance fixture:** `data/results/2026-07-16-test-gui`

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

- Results/QC/Error/Analysis mutations are conditionally allowed only inside
  `data/results/2026-07-16-test-gui`, after A04 establishes coherent
  completion evidence. A04 failed in this run, so the copy remains read-only.
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
| Staged preload | `PHENOTYPIC_PRELOAD_MODULES=tests._fakes.register_fake_gpu` | Test module importable by GUI and workers | `[-]` unverified in the live allocation |
| Production GPU pipeline | `config/UCR_029_E_D_Maresca_v12.json.pht-pipe` | SAM2/staged; do not use except for read-only inspection | `[x]` |
| Browse multi-image source | `data/subset/subset_only/outlier` | Eight distinct non-production TIFF images | `[x]` |
| Browse metadata | `metadata/UCR_029_E_D-Metadata_subset.csv` | Contains `Metadata_ImageFileName` and legacy `Metadata_ImageName`, but neither literal `ImageName` nor current `MetadataImage_ImageName`; Timeline-valid only | `[x]` |
| Tune spec/layout | Verified minimal CPU pipeline above; Setup never produced a spec | Canonical typed spec and metadata | `[-]` blocked by F-010 |
| Run output namespace | `gui_e2e_acceptance/ab547d29f/live-20260726-codex/` | New, unique, outside production results | `[x]` |
| Active protected job | SLURM `26749027`, interactive allocation on `gpu12` | Never cancel or replace | `[x]` |
| Protected code workdir | `/bigdata/exfab/anguy344/PhenoTypic` | GUI checkout and active allocation workdir | `[x]` |
| Protected production run | `data/results/2026-07-16` and every other production result | Never mutate or resume | `[x]` |

## 5. Acceptance checklist

### A. Preflight and deployment identity

- [x] **A01** Confirm the remote checkout contains final commit
  `ab547d29fbca082f3d801196cf952eaf562535f9`.
- [x] **A02** Confirm the GUI process serves the updated Run controls and
  canonical typed-file extensions.
- [x] **A03** Record active SLURM jobs, their output roots, and protected
  generations before testing. Fresh `squeue` showed exactly one job:
  interactive GUI allocation `26749027` on `gpu12`; no batch or array
  processing job was active.
- [!] **A04** Validate the completed-run copy layout, permissions, manifest,
  measurements mirror, pipeline recipe, overlays, QC state, and completion
  evidence.
- [x] **A05** Confirm the dedicated Run/Tune output namespace is absent or
  empty and cannot resolve to an active output.
- [-] **A06** Restart the controlled GUI with
  `PHENOTYPIC_PRELOAD_MODULES=tests._fakes.register_fake_gpu`, confirm the
  module imports in the GUI/worker environment, and prove the minimal staged
  pipeline deserializes before G08. The checkout contains the module, but the
  live allocation environment cannot be inspected externally and no startup
  file establishes the variable.

### B. Shell, Home, and file explorer

- [x] **B01** Home loads with shell chrome, grouped Pipeline/Results
  navigation, RSS readout, sandbox label, and capability counts.
- [x] **B02** Help and Settings open and close without layout or callback
  errors.
- [x] **B03** Input-folder picker writes a valid shared V2 source payload;
  Clear removes browser authority without deleting files.
- [x] **B04** Metadata picker accepts the intended CSV and rejects
  non-CSV/out-of-sandbox paths.
- [x] **B05** Sidebar expands/collapses lazily and shows accurate
  `img`/`cfg`/`out`/bundle badges.
- [x] **B06** Hidden and external-symlink toggles change visibility without
  escaping the sandbox.
- [!] **B07** Refresh updates badges, open pickers, source labels, and page
  inputs through one shared revision.
- [x] **B08** Sidebar handoff offers only context-valid actions and does not
  treat stale labels as selected paths.
- [!] **B09** Recent Runs excludes private legacy backups and reports
  incomplete generation-less historical outputs as `unknown`, not `running`.
- [x] **B10** Navigation among all mounted apps preserves active-group styling
  and does not produce duplicate IDs or blank mounts.

### C. Browse workflows

- [x] **C01** Selecting the minimal image directory populates dataset and image
  controls and loads the first image.
- [x] **C02** Previous/next controls clamp correctly; image dimensions, size,
  and available metadata render.
- [x] **C03** Deep zoom, pan, home, and full-page controls work without a CDN.
- [x] **C04** Compare mode mounts the selected images, enforces its cap, and
  propagates linked pan/zoom.
- [!] **C05** Timeline mode exercises at least one folder/pattern row source
  and one CSV-backed row source when metadata exists, including placeholder or
  advanced-regex preview and join/warning feedback.
- [x] **C06** Timeline time-source selection exercises EXIF/folder/pattern/CSV
  options that are available, then builds a matrix, focuses the first
  populated cell, navigates by arrows/buttons, and keeps a bounded mounted
  window.
- [x] **C07** Timeline tile-size controls and row-header comparison work.
- [!] **C08** Timeline hover/Enter opens and reopens the deep-zoom popout.
- [!] **C09** Returning to Browse after shared Refresh retains a valid source
  or clearly reports it unavailable.

### D. Builder workflows

- [x] **D01** Builder loads the canonical pipeline through the picker and
  preserves the typed extension.
- [x] **D02** Synthetic and real-image source paths both render a usable input
  node; point-picker selection round-trips when offered.
- [!] **D03** Palette insertion builds a linear chain; zoom, fit, selection,
  inspector, and documentation controls work.
- [!] **D04** Scalar operation-valued aux targets accept, replace, clear, and
  drill into compatible operations.
- [x] **D05** Embedded pipeline aux creation, breadcrumb drill-in, nested
  editing, and drill-out work.
- [!] **D06** Required-side-value and whole-pipeline validation states identify
  the correct operation without destructive repair.
- [!] **D07** Run Preview publishes a complete generation and keeps the preview
  DOM mounted across reselection.
- [-] **D08** Editing a parameter marks the old preview stale; rerun replaces
  it atomically with the new revision.
- [-] **D09** Save writes a canonical `.json.pht-pipe`; Load round-trips the
  saved pipeline without shared-instance aliasing.
- [ ] **D10** Unsupported nonlinear/development DAG input fails closed with a
  recovery explanation instead of silent data loss.

The canonical D09 save passed, but the saved file was not reloaded; the
round-trip and aliasing requirement remains unverified.

### E. Tune co-pilot

- [!] **E01** Setup loads a pipeline or existing `.json.pht-tune` spec and
  preserves existing strategy, budget, storage, scorer, and extensions.
- [-] **E02** Search-space editors expose supported domains, preserve typed
  values, and block invalid/no-knob configurations.
- [-] **E03** Metadata-backed scorer replacement is explicit; credentials are
  not rendered into browser-visible state or commands.
- [-] **E04** Continue writes an atomic canonical spec and switches to Run.
- [-] **E05** Run source/output/strategy/budget/storage/compute/evaluation
  controls produce one valid launch command.
- [-] **E06** Copied command text and Deploy argv are identical after shell
  parsing/redaction rules.
- [-] **E07** A minimal Local Tune deployment reaches terminal state and
  produces a bindable Tune output.
- [-] **E08** Monitor polls progress, binds through the read-only run picker,
  and exercises Local-only cancel behavior or its precise non-local fallback.
- [-] **E08a** Curate shortlists and pins A/B trials, renders linked overlays
  and difference view, propagates pan/zoom, and selects a winner.
- [-] **E08b** Space toggles tunable knobs, edits supported domains, exports a
  canonical next spec, and Launch refreshes to a valid command using it.
- [-] **E09** Best-pipeline export writes a canonical pipeline without
  modifying the source spec.
- [-] **E10** SLURM Tune mode either completes a minimal test-owned deployment
  or is marked blocked with precise scheduler/UI evidence.

E02 editors and the E03 scorer panel rendered, but invalid-domain behavior,
credential handling, replacement, and persistence could not be exercised
because F-010 kept Continue disabled.

### F. Run Console, Local, and Validate

- [!] **F01** Pipeline/input/output pickers accept only sandbox-valid paths and
  show canonical typed files.
- [!] **F02** Rapid Local/SLURM mode changes use the final visible controls, not
  stale derived state.
- [!] **F03** Dry-run, Resume, metadata, canonical image extensions, and
  advanced fields appear in the generated request exactly once.
- [ ] **F04** Save/Load preset round-trips all CPU, GPU, staged, and SLURM
  controls.
- [x] **F05** Validate records before launch, streams logs, reaches a terminal
  registry state, and publishes no run output.
- [x] **F06** Minimal Local run records a unique generation before spawn,
  streams incremental logs, and reaches completion only with matching
  publication evidence.
- [-] **F07** Local cancellation affects only the test generation and reaches a
  terminal state after the process is inactive.
- [-] **F08** Recent Runs refreshes by registry revision; row selection points
  the dashboard iframe at the correct output.
- [!] **F09** Fresh-output ownership rejects accidental reuse; Resume is
  explicit and generation-checked.
- [ ] **F10** A pre-seeded process-mode output is classified and shown in
  Recent Runs without inventing a full-run dashboard.

### G. Run Console and live SLURM

- [-] **G01** SLURM form accepts minutes, `HH:MM:SS`, and `D-HH:MM:SS`, while
  rejecting malformed durations and an empty profile.
- [-] **G02** Ordinary one-image submission persists intent before `sbatch`,
  records array/finalizer roles, and shows queued/running/reconciling states.
- [-] **G03** Scheduler logs stream incrementally and polling remains bounded.
- [-] **G04** Ordinary completion requires every ledgered job, finalizer
  success, generation marker, and complete manifest.
- [-] **G05** Test-owned ordinary cancellation fences submission, cancels every
  recovered ID, and remains `cancelling` until quiescent.
- [-] **G06** GUI restart/Refresh rehydrates a nonterminal test generation from
  durable owner, intent, role ledger, and scheduler state.
- [-] **G07** Scheduler-unavailable behavior becomes `unknown`, never a false
  terminal or running state.
- [-] **G08** Staged GPU one-image submission records controller, recovery,
  CPU-stage, GPU-stage, continuation, and finalizer roles as applicable.
- [-] **G09** Staged completion requires orchestration completion, per-image
  Stage-3 evidence, matching epoch, and cleanup of transient sidecars.
- [-] **G10** Staged cancellation deactivates the epoch before `scancel` and
  leaves no test-owned continuation active.
- [x] **G11** No job or output belonging to the protected active run changes
  throughout G01-G10.

Only the `HH:MM:SS` value `00:05:00` was entered for G01. The other accepted
forms and rejection cases remain blocked because F-014 prevents validation.

### H. Results binding and read-only views

Read-only binding was attempted after A01 passed. Mutation-dependent
interpretations remain blocked by A04 until the fixture is authoritative;
all post-bind read-only workflows are blocked by F-015.

- [!] **H01** Binding `data/results/2026-07-16-test-gui` validates the layout
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

The H02 immutability portion passed: key pre/post hashes and modification times
were unchanged. Tab, tile, Refresh, and compatibility-preflight portions never
became reachable.

### I. QC configuration, migration, rebuild, and review

All I items are blocked by A04 and F-015. Do not mutate the inconsistent copy.

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

All J items are blocked by A04 and F-015. Do not publish into the inconsistent
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

All K items are blocked by A04 and F-015. Do not save or publish into the
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

- [!] **L01** V1 payloads remain readable, but a sandbox relocation or
  fingerprint mismatch makes both stored V1/V2 source and metadata descriptors
  unavailable and non-authoritative until explicit reselection; reselection
  writes V2 path/fingerprint payloads.
- [!] **L02** Shared source, metadata, output binding, and Refresh propagate
  consistently across Browse, Builder, Tune, Run, Results, and Analysis.
- [x] **L03** Pipeline, Tune, and image-extension displays use canonical
  extensions everywhere.
- [-] **L04** Copy/deploy command parity holds for Run and Tune.
- [x] **L05** Bound source trees remain unchanged by read-only interactions;
  only approved test outputs and explicit copied-results mutations change.
- [x] **L06** Every submitted test job is terminal and no test-owned scheduler
  continuation remains.
- [x] **L07** Active production jobs and outputs match their preflight state
  except for changes produced by their own pre-existing processing.
- [!] **L08** Record every major issue below with reproduction, affected path,
  evidence, severity, and recommended fix.

L06 passes only as an isolation invariant: no test scheduler job was submitted,
so there is no test-owned continuation. Scheduler lifecycle behavior remains
blocked by F-014.

L08 remains partial because the live browser evidence did not expose the
registry generation UUID for F-012 through F-014. The affected output paths
and all other available evidence are recorded below.

## 6. Findings

### F-001: Resolved initial remote revision mismatch

**Severity:** Resolved P0 acceptance blocker
**Checklist:** A01, A02, and every browser test after preflight

Pre-restart read-only SSH inspection found:

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

**Resolution:** `debug-gui` was published and the user restarted the GUI.
Fresh read-only SSH inspection on 2026-07-26 found both local and remote clean
at `ab547d29fbca082f3d801196cf952eaf562535f9`, which contains the required
implementation. The sole active job is now interactive GUI allocation
`26749027`. A01 and A02 pass. A06 remains independently blocked because the
live process environment does not expose whether the fake-GPU preload was set.

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

### F-003: Shared Refresh changes an explicitly selected source to its parent

**Severity:** P1 stale-path authority failure
**Checklist:** B07, C09, L01, L02

The Shell input picker explicitly selected
`data/gui_e2e_test_inputs` and showed `source: gui_e2e_test_inputs`. Clicking
the one shared Refresh button changed the settings label to `source: data`
without a user selection. Browse then had no valid dataset option and did not
show the previously selected image.

The source was manually reselected afterward. Refresh must preserve the exact
V2 path/fingerprint descriptor or mark it unavailable; silently authorizing the
parent directory is incorrect.

### F-004: Browse Timeline retains controls and preview data from the old source

**Severity:** P1 cross-source state-coherence failure
**Checklist:** C05, C09, L02

After changing the source from the one-image `gui_e2e_test_inputs` folder to
the eight-image `data/subset/subset_only/outlier` folder:

- the single-image picker updated to the first outlier TIFF;
- Timeline initially continued to render the old `slurm_smoke.png` matrix;
- toggling Single then Timeline rebuilt the matrix with the eight TIFFs;
- the filename-pattern input and preview table still displayed
  `slurm_{plate}.png` and `slurm_smoke`, even after switching both axes and the
  join key to CSV columns from the new metadata.

The active matrix eventually used the new source, but stale authoring/preview
state remained visible and authoritative-looking. Source revision changes must
invalidate all dependent Timeline state atomically.

### F-005: Compare overlay exposes encoded internal path tokens as titles

**Severity:** P2 presentation and path-identity issue
**Checklist:** C04

Selecting two populated Timeline cells and opening Compare produced two linked
deep-zoom viewers whose zoom state propagated correctly. Their visible titles,
however, were long base64-encoded path tokens such as an encoding of
`data/subset/subset_only/outlier/<image>.tif`, rather than image filenames or
human-readable relative paths.

The encoded transport token should remain internal. The overlay should display
the resolved filename while preserving the token only in callback state.

### F-006: Timeline deep-zoom popout cannot be opened

**Severity:** P1 advertised workflow failure
**Checklist:** C08

With a populated CSV-backed 6×6 Timeline matrix:

- the grid contained exactly one focused cell and six mounted popout controls;
- clicking a visible/populated cell, clicking its `.timeline-cell-popout`
  target, and pressing Enter after focus did not open a modal;
- neither the expected popout modal nor a deep-zoom canvas appeared.

Compare mode can open deep zoom, so image serving is functional. The per-cell
popout event path itself appears disconnected or inaccessible.

### F-007: Browse has two incompatible image-name metadata contracts

**Severity:** P1 stale-schema compatibility failure
**Checklist:** C02, C05, L01, L02

Direct SSH inspection confirmed that
`metadata/UCR_029_E_D-Metadata_subset.csv` contains
`Metadata_ImageFileName` and `Metadata_ImageName`, but not the current
`MetadataImage_ImageName` header. The CSV-backed Timeline is valid when its
image join is explicitly set to `Metadata_ImageFileName`: all eight selected
image stems are represented, and Timeline strips the extension before joining.

The ordinary single-image metadata panel uses a different strict resolver and
rejects `Metadata_ImageFileName`, legacy `Metadata_ImageName`, and literal
`ImageName`; it requires exactly `MetadataImage_ImageName`. The same selected
CSV therefore works in Timeline while the ordinary panel reports that it has
no image-name column. One compatibility resolver and migration warning should
serve both paths.

### F-008: Builder preview, selection, and zoom controls are disconnected

**Severity:** P1 core-authoring workflow failure
**Checklist:** D03, D06-D08

The canonical nine-operation pipeline and both synthetic and real-image
sources loaded. Palette insertion and nested-pipeline drill-in/drill-out also
worked. However:

- selecting existing nodes did not populate the inspector;
- zoom in, zoom out, and fit left the canvas transform unchanged;
- repeated Run Preview actions on the small real image produced neither a
  preview generation nor validation/error feedback after more than ten
  seconds.

Because no preview generation can be published, stale-preview invalidation
cannot be acceptance-tested.

### F-009: Builder scalar aux replacement appends a top-level operation

**Severity:** P1 destructive authoring misroute
**Checklist:** D04

Filling `FilamentousFungiDetector.inoculum_detector` with `OtsuDetector`
worked. Choosing Replace and then `MeanDetector` removed the visible aux value
but appended `MeanDetector` as a new top-level pipeline node instead of
replacing the scalar side value. The target identity is lost across the
replacement callback.

### F-010: Tune Setup populates editors but keeps Continue permanently gated

**Severity:** P1 end-to-end Tune blocker
**Checklist:** E01, E04-E10

Selecting either the Builder-saved canonical pipeline or the verified minimal
CPU fixture updated the source label and populated operation parameter editors.
The scorer panel also rendered. At the same time the Setup gate continued to
say `Choose a pipeline or existing tuning spec`, and Continue remained
disabled. Typed absolute paths also did not load on Enter/blur. All downstream
Tune tabs and deployment tests are blocked by this contradictory Setup state.

### F-011: Run output picker ignores a typed new path and selects project root

**Severity:** P0 unsafe-output selection failure
**Checklist:** F01, F09, L02, L05

Entering the new sandbox-relative acceptance path
`gui_e2e_acceptance/ab547d29f/live-20260726-codex/local-01` and confirming
changed the visible Run output to `.`. Retrying with the full absolute path,
explicitly blurring the field, waiting for state settlement, and confirming
again also selected `.`.

No run was launched while `.` was selected. The exact isolated directory was
created out-of-band under the acceptance namespace and then selected by
directory navigation. Typed non-existent output paths must resolve to the
entered canonical target, or fail closed; silently substituting the project
root is unsafe.

### F-012: A completed Run leaves Cancel enabled

**Severity:** P1 lifecycle-control failure
**Checklist:** F06, F07

The one-image local run at
`gui_e2e_acceptance/ab547d29f/live-20260726-codex/local-01` reached
`status=complete`, wrote matching processing state, manifest, deliverables,
and a 100% completion summary. Run re-enabled, but Cancel remained enabled
after terminal publication and a further callback settlement interval. A
terminal generation must not expose an actionable cancellation control.

### F-013: Shared metadata silently corrupts an unrelated Run aggregation

**Severity:** P1 cross-app state/visibility failure
**Checklist:** F03, L02, L05

The one-image output
`gui_e2e_acceptance/ab547d29f/live-20260726-codex/local-01` showed pipeline,
input, output, mode, Dry-run, Resume, and collapsed advanced controls, but no
visible metadata selection. The previously selected 669-row
`metadata/UCR_029_E_D-Metadata_subset.csv` was nevertheless injected into the
run. Its grid keys did not describe the one-image fixture:

- the one measured row was dropped from the metadata inner join;
- 669 unmatched metadata rows were retained in the editable measurements
  mirror with `QC_MetadataOnly=true`;
- the log reported duplicate metadata join keys.

Shared metadata may be intentional, but Run must display and confirm the exact
metadata descriptor, its compatibility, and the resulting request. It must not
silently apply stale cross-app metadata to a scientifically unrelated source.

### F-014: Fresh live SLURM-mode Validate and Run actions are silent no-ops

**Severity:** P0 live-execution blocker
**Checklist:** F02, G02-G10

After a full page reload cleared the stale Local generation identity described
in F-012, the test explicitly reselected:

- the verified minimal CPU pipeline;
- the one-image input directory;
- the pre-created isolated
  `/rhome/anguy344/bigdata_exfab/ucr_029_e_d_Maresca/gui_e2e_acceptance/ab547d29f/live-20260726-codex/slurm-01`
  output;
- SLURM mode with `partition=short`, `time=00:05:00`, `memory=2G`,
  one CPU, and zero CPU-stage GPUs.

Both buttons remained enabled, focused when activated, and were the topmost
hit-test targets. Neither action produced a status, log, error, registry row,
submission intent, or file. The isolated output stayed empty. No scheduler job
was submitted, so ordinary and staged lifecycle acceptance is blocked at the
Run callback seam.

The exact trigger remains unresolved. Focused local callback tests passed, so
the evidence establishes a live browser/Dash action-seam failure rather than a
specific scheduler-submitter defect. Implementation must first reproduce the
same-page action sequence with callback-network and page-error capture.

### F-015: Large Results binding exceeds the proxy timeout with no progress

**Severity:** P1 production-scale viewer blocker
**Checklist:** H01-H10, K01-K07

On a fresh `/results/` page, the copied run was selected from the sidebar and
classified as a CLI output. The enabled `↩ Open in viewer` action was invoked
exactly once. The empty page showed no pending state while the server
synchronously discovered and fingerprinted the approximately 475 GB output
and built both Results and Analysis candidates. RSS rose from roughly 648 MB
to 686 MB.

After about 95 seconds the inline handoff error became `HTTP 504`. A 30-second
grace period and reload still showed `No output selected`; no atomic bind had
published. Pre/post SHA-256 hashes for the manifest, pipeline, measurements
mirror, and master measurements were identical, and no file under the copy had
a new modification time.

The completed one-image Local acceptance output was then selected to exercise
the viewer on a tiny fixture. That bind also returned `HTTP 504` after the same
timeout. Because binding is serialized, this is consistent with the earlier
large discovery continuing to hold or queue behind the publish lock even after
its client request timed out. The evidence suggests a timed-out bind may deny
service to subsequent otherwise-small viewer binds.

The bind needs an asynchronous job/progress contract, bounded or cached
discovery, idempotent ticket reuse, and a request path that cannot be killed by
the web proxy timeout. Until then the requested production-scale Results,
Analysis, QC, Error, and mutation-blocking UI states cannot be exercised.

### F-016: Recent Runs exposes a root-level legacy backup as a run

**Severity:** P2 stale-artifact discovery issue
**Checklist:** B09

Recent Runs includes
`data/results/2026-05-13_objectlabel-backup` with mode and status both
`unknown`. Read-only SSH inspection shows it is a June 2026 backup tree with
duplicated root-level and `deliverables/` measurement artifacts and no current
generation identity. Reporting its state as `unknown` is correct, but surfacing
an explicitly backup-named tree as a runnable recent output conflicts with the
private-backup exclusion requirement and adds a misleading target to both Run
and Results discovery.

## 7. Completion summary

The checklist records 43 passes, 20 observed failures, 60 blocked or partial
items, and 5 not-started items.

Testing concluded with evidence for deployment identity, Shell/Browse,
partial Builder and Tune Setup behavior, Local Validate/lifecycle behavior,
the pre-submission SLURM form, and Results binding. Unexecuted or unreachable
criteria remain explicitly `[-]` or `[ ]` in the checklist.

One isolated acceptance output was written under
`gui_e2e_acceptance/ab547d29f/live-20260726-codex/local-01`. Its process
lifecycle completed successfully for F06, but F-013 made its scientific
aggregation invalid by silently applying unrelated shared metadata. The separate
`slurm-01` directory remained empty because SLURM Validate and Run were silent
no-ops; no scheduler job was submitted or cancelled.

No production output was modified. Interactive allocation `26749027` remains
the sole active job. The copied Results bind failed with `HTTP 504` while
leaving its key artifacts byte-identical; Analysis and all bound-view
workflows remain blocked behind that bind.

## 8. Re-acceptance attempt on 2026-07-27

This section records the acceptance attempt against the production-file content
published in `770ff7bbaa35437fd3c0b3c11afabb146587ea43`. The checklist state in
Sections 4 through 7 remains the historical 2026-07-26 record and must not be
interpreted as current evidence.

### 8.1 Deployment and safety preflight

- [!] **A01** The checkout is not clean or checked out at the required commit.
  `/bigdata/exfab/anguy344/PhenoTypic` reports `HEAD=ab547d29f` with the
  implementation present as modified and untracked files. A byte-for-byte
  comparison of every changed production file under `src/phenotypic` found
  zero differences from `770ff7bba`, but the live-test exact-source gate
  correctly rejects this checkout identity. One tracked test file,
  `tests/unit/tune/test_atomic_io.py`, is absent from the working tree.
- [-] **A02** The updated Run controls and canonical typed extensions were not
  revisited before the browser session wedged. Updated shared-path, Timeline,
  and Results surfaces were observed separately.
- [x] **A03** The scheduler preflight found one job:
  interactive allocation `26751178` on `gpu12`. No acceptance batch job was
  submitted or cancelled.
- [!] **A04** The copied Results fixture remains contradictory and read-only,
  so it does not satisfy the coherence precondition for mutation testing.
  Before and after the attempted bind, sizes and mtimes were unchanged for
  processing state, manifest, both pipeline copies, and both measurements
  mirrors. SHA-256 values were also unchanged for processing state, manifest,
  and both small pipeline copies.
- [-] **A05** New live output namespaces were not created because A01 failed.
- [-] **A06** The running compute-node process environment could not be read
  from the login node, so the fake-GPU preload remains unverified.

The canonical sandbox actually served by this deployment is
`/bigdata/exfab/anguy344/projects/ucr_029_e_d_Maresca`. The originally recorded
`/rhome/anguy344/bigdata_exfab/ucr_029_e_d_Maresca` path does not exist in this
login environment and must not be used by the live harness.

### 8.2 Locally safe scheduler gates

- [x] 259 scheduler contract tests passed.
- [x] All five fake-scheduler browser scenarios passed: generation-fenced
  submit/cancel, ordinary array plus finalizer publication, process-mode
  success and failure boundaries, and staged dependency retargeting.
- [-] Ordinary live SLURM, cancellation, restart reconciliation, and Tune
  SLURM remain blocked by A01. Staged SLURM is additionally blocked by A06.

### 8.3 Live browser results

- [x] **B01-B02** Home chrome, Help, and Settings worked.
- [~] **B03** Explicit source selection and Clear worked. The V2 payload was
  not inspected directly in this attempt.
- [~] **B07 / [x] F-003 core** Shared Refresh retained the exact explicitly
  selected one-image directory instead of authorizing its parent. Badge, open
  picker, source-label, and page-input propagation were not all rechecked.
- [x] **C01-C02** The one-image and eight-image sources loaded with correct
  image names, dimensions, file sizes, EXIF values, and navigation bounds.
- [~] **C03** Deep-zoom controls mounted, but pan, home, and full-page behavior
  were not exercised in this attempt.
- [~] **C04 / F-005** Compare rendered readable sandbox-relative image paths,
  not encoded transport tokens. Linked pan/zoom was not rechecked.
- [~] **C08 / [x] F-006 keyboard path** Enter on the focusable Timeline
  viewport opened the deep-zoom modal; close and reopen both worked. The hover
  trigger was not rechecked.
- [~] **F-004 core** Clearing the source atomically retired the grid,
  selection, Compare overlay, filename pattern, and axis state. A
  source-to-source switch with authored state and C09 return-after-Refresh
  were not rechecked.
- [x] **F-007** The ordinary Browse metadata panel matched the legacy
  `Metadata_ImageFileName` identity and rendered five matching metadata rows
  for the first outlier image.
- [!] **C05** Selecting `Filename pattern`, typing a pattern, and confirming it
  did not update the preview or rebuild the grid. The DOM value changed, but
  the preview remained `Enter a pattern to preview matches.`. Current browser
  coverage injects this state with `dash_clientside.set_props` and therefore
  does not cover the real typing path.
- [!] **H01 / F-015** Clicking `Open in viewer` for
  `data/results/2026-07-16-test-gui` did not return an immediate asynchronous
  ticket or progress state. The browser interaction remained blocked for more
  than five minutes, and subsequent control requests to that in-app browser
  session also timed out. A separate HTTP health request to the GUI root still
  returned `302` in 0.2 seconds, so the failure is specific to the bind/browser
  request path rather than a total server outage.
- [-] **H02-H10, I-K** The Results bind never published, so downstream Results,
  QC, Error, and Analysis acceptance remains blocked. The fixture was not
  mutated.
- [-] **D, E, F, G, L** Remaining live Builder, Tune, Run, scheduler, and final
  cross-app checks were stopped after the bind wedged the browser session.

### 8.4 Major issues from this attempt

1. **P0 acceptance blocker: non-reproducible deployed checkout.** Production
   files match the target content, but Git identity is still `ab547d29f` with
   a large dirty overlay. Live safety harnesses cannot prove the code they are
   executing and correctly fail closed.
2. **P1 Results blocker: the asynchronous large-bind contract is not active in
   the deployed behavior.** The bind request still blocks for minutes with no
   visible ticket, phase, progress, cancellation, or supersession control.
3. **P1 Browse workflow failure: typed Timeline patterns do not propagate.**
   The current automated browser test bypasses user input by writing component
   state directly and missed the live interaction failure.

No production output or scheduler job was changed during this attempt.

### 8.5 F-015 remediation prepared locally

The follow-up implementation preserves exhaustive processing inventories for
coherent, mutation-capable outputs. Active, incomplete, and contradictory
outputs now bind with a bounded structural inventory, remain permanently
read-only for that binding generation, and validate only the HDF or overlay
used by each pixel request. The bind UI also publishes a non-authoritative
Submitting state synchronously, before awaiting the POST acknowledgement, so
a slow proxy or server acknowledgement cannot leave the page apparently idle.

Local unit, integration, targeted pixel-route, and real-click browser
regressions cover the new split. Remote H01-H10 acceptance remains pending a
clean deployment of the follow-up commit; this section does not change the
failed/blocked states recorded above.
