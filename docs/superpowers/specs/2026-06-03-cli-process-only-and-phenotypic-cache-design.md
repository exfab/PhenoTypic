# Design: `--process-only` CLI mode + `.phenotypic` machine-state cache

- **Date:** 2026-06-03
- **Branch:** `worktree-cli-processing-mode`
- **Status:** Approved (brainstorm) — pending implementation plan
- **Author:** Alexander Nguyen (with Claude)

## 1. Summary

Two related changes to the PhenoTypic CLI:

1. **`--process-only {rgb|gray|detect_mat|objmap}`** — an alternate run mode
   that executes `pipeline.apply()` only (preprocessing/detection, **no
   measurement, no analysis output suite**) and writes a **single chosen image
   layer** per input image into the output folder, **mirroring the input
   directory tree**. `rgb`/`gray`/`detect_mat` are saved as 16-bit TIFFs;
   `objmap` is saved as a raw-label PNG (matching prior CLI iterations). This
   lets users leverage PhenoTypic's preprocessing pipeline as an image
   transformer without paying for the full measurement/dashboard/deliverables
   machinery.

2. **`.phenotypic` hidden machine-state cache** — relocate the run's
   machine-state sidecars (`progress/` and `processing_state.json`) into a
   single hidden `<output>/.phenotypic/` directory, for **both** the existing
   forward CLI and the new process-only mode. This gives the CLI and GUI one
   canonical, de-duplicated reference point for run state, and keeps the
   user-facing output folder clean.

Both build on the existing single-source-of-truth layout module
`phenotypic.tools_._io_constants`.

## 2. Goals / Non-goals

### Goals
- Run the pipeline's `apply()` stage over a directory of images in parallel
  (local **and** SLURM), with resume support, and emit only the requested
  image layer.
- Output mirrors the input tree (collision-free), no `deliverables/`,
  `results/`, dashboards, QC, overlays, or aggregation.
- Consolidate machine-state under `<output>/.phenotypic/` for both run modes.
- Preserve backward compatibility: existing on-disk runs (state at the legacy
  root) still resume and still appear in the GUI run list.

### Non-goals
- No new measurement, post, analysis, or QC behavior.
- No change to the `deliverables/`, `results/`, `qc/`, `logs/`, or
  `slurm_scripts/` locations or contents.
- No multi-layer export in a single run (one `--process-only` value per run).
- No new GUI *control* for process-only (CLI-only feature for v1); the GUI only
  needs to keep discovering/reading runs through the new state location.

## 3. Decisions (locked during brainstorming)

| # | Decision | Choice |
|---|----------|--------|
| D1 | `.phenotypic` scope | **Migrate all machine-state, both modes** |
| D2 | Flat output collision strategy | **Mirror input tree** |
| D3 | Layers per run | **Single layer** |
| D4 | Execution machinery | **Full reuse: local parallel + SLURM + resume** |
| D5 | Worker integration (Fork A) | **Sibling worker** `process_single_apply_only_core` |
| D6 | Migration mechanism (Fork B) | **Re-root path helpers + read-fallback for legacy runs** |
| D7 | Migration set | machine-state sidecars only: `progress/`, `processing_state.json`, `processing_events.log` (**not** `logs/`, `slurm_scripts/`, `results/`, `qc/`, `deliverables/`) |
| D8 | Output filename | `<stem>_<layer>.<ext>` (self-documenting) |
| D9 | objmap without a detector | Warn per image + write the (empty) map; do not fail |

## 4. Background (current state)

- **`ImagePipeline.apply(image, inplace=False, reset=None)`**
  (`_core/_pipeline_parts/_image_pipeline_core.py:858`) runs the operations
  queue (enhancers → detectors → refiners) and returns the processed image. No
  measurement. This is the exact core needed.
- **Layer accessors** all expose `imsave(filepath, bit_depth=None)`:
  - Color/gray/detect_mat go through
    `_accessor_io_handler.imsave` → `_save_image`: for a `.tiff` path it writes
    true 16-bit via `tifffile` when the array is `uint16` (or `bit_depth=16`
    forces conversion); otherwise PIL.
  - `objmap.imsave(path, use_label2rgb=False)`
    (`accessors/_objmap_accessor.py:531`) writes the **raw labeled map**
    (integer labels). For `.png` this is the prior CLI behavior.
- **Per-image worker** `process_single_image_core`
  (`_cli/_cli_process_single.py:34`) currently does
  `apply_and_measure` + save HDF + overlay + measurements parquet.
- **Execution strategies** (`_cli/_cli_execution_strategies.py`):
  `LocalParallelStrategy` (joblib) and `AutonomousSLURMStrategy` (array job
  scripts). Both call `process_single_image_core`. SLURM array scripts invoke
  `python -m phenotypic._cli._cli_process_single` (`main()`), so that worker
  CLI must learn the same flag.
- **Layout source of truth** `tools_/_io_constants.py`:
  - Machine-state helpers currently root at the **output root**:
    - `processing_state_path(output)` → `<output>/processing_state.json`
    - `event_log_path(output)` → `<output>/processing_events.log`
    - `progress_dir(output)` → `<output>/progress/`, and everything composed
      from it: `job_metadata_path`, `failures_jsonl_path`, `manifest_json_path`,
      `chunk_manifest_path`, `chunk_state_path`, `overlay_manifest_path`, plus
      the `progress_dir_`-taking helpers (`chunks_dir`, `recompile_dir`,
      `recompile_status_dir`, `analysis_full_parquet_path`,
      `analysis_scatter_json_path`, `sentinel_resubmitted_path`,
      `chunk_lock_path`, `checkpoint_lock_path`).
  - User-facing helpers root under `deliverables_dir(output)` —
    **unchanged** by this work.
- **GUI coupling**: `gui/_config.py` re-exports `DIR_PROGRESS` etc.; readers
  `shell/_runs_registry.py`, `shell/_classifier.py`,
  `run_console/_recent_runs.py` use the path helpers and `manifest_json_path`.
  Run discovery **already skips hidden directories** (`.phenotypic-gui`,
  `.gui_log`). The GUI already uses a hidden `.phenotypic-gui` dir for
  presets/state — precedent for hidden run-local dirs.

## 5. Detailed design

### 5.1 `.phenotypic` machine-state migration

Add to `_io_constants.py`:

```python
DIR_PHENOTYPIC: Final[str] = ".phenotypic"

def phenotypic_cache_dir(output_dir: Path) -> Path:
    """Return <output>/.phenotypic/ — hidden machine-state root."""
    return output_dir / DIR_PHENOTYPIC
```

Re-root the **machine-state** helpers (and only these) through it:

- `progress_dir(output)` → `phenotypic_cache_dir(output) / DIR_PROGRESS`
  → `<output>/.phenotypic/progress/`
- `processing_state_path(output)` → `phenotypic_cache_dir(output) / PROCESSING_STATE_JSON`
  → `<output>/.phenotypic/processing_state.json`
- `event_log_path(output)` → `phenotypic_cache_dir(output) / PROCESSING_EVENTS_LOG`
  → `<output>/.phenotypic/processing_events.log`

Everything composed from `progress_dir`/`progress_dir_` follows automatically
(no per-file edits to the ~15 dependent helpers — that is the payoff of the
existing single-source-of-truth design and the literal meaning of "dedup
references").

**Unchanged:** `deliverables_dir` and all helpers under it, `results_dir`,
`dataset_*`, `qc_dir`, `logs_dir`, `slurm_scripts_dir`.

**Internal structure preserved**: `progress/` keeps its sub-layout, just nested
one level deeper. This minimizes churn in every writer/reader that already
operates relative to `progress_dir(output)`.

#### Backward compatibility (D6)

A run created before this change has `progress/` + `processing_state.json` at
the output root. We **always write** to `.phenotypic/`, but **reads tolerate
the legacy location**:

- New helper `resolve_processing_state_path(output)`: returns
  `.phenotypic/processing_state.json` if it exists, else the legacy
  `<output>/processing_state.json` if *that* exists, else the new path
  (default for fresh writes). Used by `--resume`/`--restart` state loading.
- New helper `resolve_progress_dir(output)` with the same first-existing
  semantics, used by GUI run discovery and any read-only consumer that must
  see legacy runs (`manifest.json`, failures, event log).
- Writers (`ProcessingState.save`, `append_event`, manifest/failure writers)
  use the canonical (new) helpers — they always create `.phenotypic/`.

This is a soft cutover: no migration of old files, no symlinks. Old runs remain
readable/resumable; new runs use the hidden dir.

> Implementation note: confirm whether `ProcessingState.save`/`load` and the
> dashboard/manifest writers call the helpers (good) or join paths by hand
> (must be updated). Audit `_cli_state_management.py`, `_cli_update_state.py`,
> `_cli_failure_tracker.py`, `_cli_checkpoint_handler.py`, `_cli_sentinel.py`,
> `_cli_chunk_writer.py`, `_cli_recompile_worker.py`.

### 5.2 GUI integration

- `gui/_config.py`: keep re-exporting the directory-name constants; add
  `PHENOTYPIC_CACHE_DIRNAME = DIR_PHENOTYPIC`. Update the doc comments noting
  machine-state now lives under `.phenotypic/`.
- Run discovery (`shell/_runs_registry.py`, `shell/_classifier.py`,
  `run_console/_recent_runs.py`): resolve state via `resolve_progress_dir` /
  `manifest` through the new helper so both new and legacy runs are found.
- Keep the existing "skip hidden dirs" behavior for *dataset/run candidate*
  scanning, but ensure `.phenotypic/` is still **read** for state. (It is never
  itself a run root, so skipping it as a candidate is correct; the fix is only
  that state lookups for a real run root descend into its `.phenotypic/`.)
- `gui/FEATURES.md` (CI-gated): add/adjust the row describing where run state
  is read from, with a test ref.

### 5.3 `--process-only` CLI surface

In `phenotypicCLI.py`:

```
--process-only [rgb|gray|detect_mat|objmap]   (default: None)
```

- **Activates** process-only mode when set.
- **Requires** `--pipeline` and `--input`.
- **Honors**: `--output-dir`, `--image-type`, `--nrows/--ncols`,
  `--bit-depth`, `--detect-mode`, `--n-jobs`, `--slurm`/`--force-local`/`--wait`,
  `--resume`/`--restart`/`--overwrite`/`--retry-failures`,
  `--sample`/`--random-seed`, `--dry-run`, `--skip-validation`.
- **Hard-rejected** (raise `click.UsageError` — these are conflicting run
  modes): `--measure`, `--recompile`.
- **Warn-and-ignore** (valid top-level options that have no effect in
  process-only mode, since there is no measurement/aggregation/QC output):
  `--metadata`, `--no-qc`, `--no-dataset-column`. Emit a single stderr warning
  per such flag that it is ignored; do not fail.

> Note: `--save-inspect` / `--save-overlays` are options on the **per-image
> worker CLI** (`_cli_process_single.main`), not on the top-level
> `phenotypic_cli`, so there is nothing to reject for them at the top level.
> `--ext` (deprecated, overlay-PNG only) and `--overlay-alpha` are simply
> unused by process-only.

Validation lives alongside the existing `--measure`/`--recompile` guard blocks.

### 5.4 Process-only execution flow

- **`ExecutionConfig`** gains an optional `process_only_layer: ProcessOnlyLayer
  | None` field (a `Literal["rgb","gray","detect_mat","objmap"]` alias in
  `tools_/typing_.py` per the CLAUDE.md closed-set rule — type-only, no Enum).
- **Strategy dispatch**: `LocalParallelStrategy` and `AutonomousSLURMStrategy`
  select the per-image callable based on `config.process_only_layer`:
  - unset → existing `process_single_image_core` (or measure path).
  - set → new `process_single_apply_only_core`.
- **New worker** `process_single_apply_only_core(pipeline_path, image_path,
  input_root, output_dir, image_type, layer, read_kwargs, ...)`:
  1. Load pipeline + image (same read-kwargs / grid-shape resolution as the
     forward worker).
  2. `image = pipeline.apply(image, inplace=True)`.
  3. Compute the **mirrored output path** (see 5.5) and write the layer (5.6).
  4. Append the same `started`/`completed`/`failed` events to the event log so
     resume + progress tracking work unchanged.
- **SLURM**: the `_cli_process_single.main` worker CLI gains `--process-only
  LAYER`; the array-script generator (`_cli_slurm_array_scripts.py`) threads it
  through. When set, the script also needs the input root to compute relative
  paths (pass `--input-root`, or derive from the existing per-image args —
  decided in the plan).
- **No finalize**: process-only skips `aggregate_measurements`,
  dashboard/manifest analysis regeneration, QC, overlays, and `deliverables/`.
  It still writes the `.phenotypic/` tracking (state, events, failures) and a
  `.phenotypic/pipeline.json` copy for reproducibility.

### 5.5 Output path (mirror input tree, D2 + D8)

For each input image, the output path preserves the image's path **relative to
the input root**, with a layer-tagged name:

```
rel   = image_path.relative_to(input_root)          # e.g. day1/plateA.tif
ext   = ".png" if layer == "objmap" else ".tiff"
out   = output_dir / rel.parent / f"{rel.stem}_{layer}{ext}"
# day1/plateA.tif , layer=detect_mat  ->  <output>/day1/plateA_detect_mat.tiff
# single file foo.tif , layer=objmap  ->  <output>/foo_objmap.png
```

- Parent directories are created as needed.
- `_layer` suffix prevents overwrite if the same output dir is reused for a
  different layer.

### 5.6 Layer write semantics (D3, D9)

| Layer | Format | Call | Notes |
|-------|--------|------|-------|
| `rgb` | 16-bit TIFF | `image.rgb.imsave(out, bit_depth=16)` | uint8 upcast to 16-bit |
| `gray` | 16-bit TIFF | `image.gray.imsave(out, bit_depth=16)` | |
| `detect_mat` | 16-bit TIFF | `image.detect_mat.imsave(out, bit_depth=16)` | float [0,1] scaled to 16-bit by the accessor's bit-depth handling |
| `objmap` | PNG (raw labels) | `image.objmap.imsave(out)` | `use_label2rgb=False`; 16-bit-capable integer labels |

- **objmap with no detector in the pipeline** → `image.objmap.isempty()` is
  true: log a per-image warning ("pipeline produced no objects; writing empty
  object map for <image>") and write the empty map. The run does not fail.

### 5.7 `--dry-run`

Print the resolved plan: mode = process-only, layer, per-dataset image counts,
sample of mirrored output paths, execution mode (local/SLURM), and the
`.phenotypic/` location — without processing.

## 6. Error handling & edge cases

- Forcing 16-bit goes through the accessor's existing `_check_bit_depth` /
  `_save_image`; no new conversion math here.
- Per-image failures use the **same** failure-tracking path as the forward run
  (`append_failure` into `.phenotypic/progress/failures.jsonl`), so one bad
  image doesn't abort the batch.
- `--process-only` + `--resume`: resume uses the shared event log / processing
  state, so already-written images are skipped. (Confirm the completion check
  keys on image identity, not on measurement-parquet presence — adjust the
  resume predicate for process-only if it currently checks for an HDF/parquet.)
- Input is a single file vs flat dir vs nested dirs: all handled by
  `relative_to(input_root)`; for a single-file input, `input_root` is the
  file's parent.

## 7. Testing strategy

**Unit**
- `_io_constants`: new layout test — machine-state helpers root under
  `.phenotypic/`; `deliverables_dir`/`results_dir`/`qc_dir`/`logs_dir`/
  `slurm_scripts_dir` unchanged. (Extends `tests/unit/tools_/test_io_constants.py`.)
- Back-compat resolvers: legacy-root state file is found when `.phenotypic/`
  is absent; new path wins when both exist; fresh write targets `.phenotypic/`.
- `ProcessOnlyLayer` Literal alias presence (type-only set).
- Output-path mapping: nested / flat / single-file inputs → expected mirrored
  paths and `_layer` suffix; objmap → `.png`, others → `.tiff`.
- Worker: each layer writes a file of the right format/bit-depth (small synth
  image); objmap-without-detector warns and writes an empty map.
- CLI validation matrix: every rejected-flag combination raises `UsageError`;
  required-flag enforcement; happy-path parsing populates
  `ExecutionConfig.process_only_layer`.

**Integration**
- `--process-only detect_mat` over a nested-input fixture, local executor,
  asserts mirrored 16-bit TIFFs exist, no `deliverables/`/`results/`, and
  `.phenotypic/` tracking is present.
- `--dry-run` output contains the plan and `.phenotypic/` path.
- SLURM script generation asserts `--process-only` (+ input root) is threaded
  into the array script (no cluster needed).
- A regression test confirming the existing forward run still works with state
  under `.phenotypic/` and that a legacy-layout run still resumes / is GUI-discovered.

## 8. Implementation phasing

1. **Phase 1 — `.phenotypic` migration**: `_io_constants` re-root + cache-dir
   helper + back-compat resolvers; audit/update all machine-state writers to go
   through helpers; update GUI readers + `FEATURES.md`; tests. Forward CLI
   behavior identical except state location.
2. **Phase 2 — process-only mode**: `ProcessOnlyLayer` alias; `ExecutionConfig`
   field; `process_single_apply_only_core`; strategy dispatch; worker-CLI flag
   + SLURM threading; CLI option + validation; mirrored output writer;
   `--dry-run`; tests + docs (CLI docstring, `CLAUDE.md` CLI section).

Phase 1 lands and is verifiable on its own; Phase 2 depends on it.

## 9. Risks

- **Migration blast radius**: many writers/readers touch `progress/`. Mitigated
  by routing everything through `_io_constants` helpers and a focused audit
  list (§5.1 note). Any hand-joined path is the main failure mode.
- **GUI run discovery regressions**: covered by back-compat resolvers + a
  legacy-layout discovery test + `FEATURES.md` gate.
- **Resume predicate**: if completion is inferred from measurement artifacts,
  process-only needs an artifact-agnostic predicate (event-log based). Flagged
  in §6 for the plan to resolve.

## 10. Open questions for implementation (non-blocking)

- Exact mechanism to pass `input_root` to the SLURM worker (new `--input-root`
  flag vs deriving from existing args).
- Whether `event_log_path` should remain a sibling of `progress/` inside
  `.phenotypic/` or move *into* `.phenotypic/progress/` (cosmetic; default:
  sibling, preserving current relative layout).
