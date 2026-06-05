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
   directory tree** (bounded by the 1-level dataset scanner — D12).
   `rgb`/`gray`/`detect_mat` are saved as TIFFs **at the
   image's bit depth** (8-bit source → 8-bit TIFF, 16-bit source → 16-bit
   TIFF); `objmap` is saved as a 16-bit raw-label PNG (matching prior CLI
   iterations). This
   lets users leverage PhenoTypic's preprocessing pipeline as an image
   transformer without paying for the full measurement/dashboard/deliverables
   machinery.

2. **`.phenotypic` hidden machine-state cache** — relocate the run's
   machine-state sidecars (`progress/`, `processing_state.json`, and
   `processing_events.log`) into a single hidden `<output>/.phenotypic/`
   directory, for **both** the existing
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
| D10 | Layer write mechanism (revised per user) | **Delegate to the layer accessor's `imsave`** — the single, golden-tested writer. Each layer is written at its **native dtype**: `rgb` integer TIFF at the source bit depth, `gray`/`detect_mat` as **float TIFF (full precision preserved)**, `objmap` 16-bit raw-label PNG. PhenoTypic metadata JSON is embedded (`imsave`'s default). **No worker-side quantization**; the writer does not reimplement IO. (See §5.6.) |
| D11 | `phenotypic.sweep` in the migration | **Out of scope for v1.** Migrate the forward CLI + process-only only; sweep's root-level `processing_events.log` stays put (follow-up). |
| D12 | Process-only input discovery / mirror depth | **Reuse the existing 1-level `scan_directory_structure`** (flat dir or one level of dataset subdirs; mixed root+subdir inputs rejected). "Mirror input tree" = mirror that ≤1-level structure; deeper nesting is not discovered. |
| D13 | Process-only GUI visibility | **Progress only, no deliverables.** Write the progress manifest + state under `.phenotypic/` (run console shows progress; `--resume` works) but emit no `deliverables/` / dashboard / results. |
| D14 | Event-log placement (resolved) | **Sibling of `progress/` inside `.phenotypic/`** (`<output>/.phenotypic/processing_events.log`). Forced by `_cli_checkpoint_handler.py:200`, which derives the log as `progress_dir.parent / PROCESSING_EVENTS_LOG`. |

## 4. Background (current state)

- **`ImagePipeline.apply(image, inplace=False, reset=None)`**
  (`_core/_pipeline_parts/_image_pipeline_core.py:858`) runs the operations
  queue (enhancers → detectors → refiners) and returns the processed image. No
  measurement. This is the exact core needed.
- **Layer accessors** all expose `imsave(filepath, bit_depth=None)` and embed the
  PhenoTypic metadata JSON. This is the writer process-only **reuses** (D10):
  - Color/gray/detect_mat go through `_accessor_io_handler.imsave` →
    `_save_image`. For a `.tiff` path it writes true 16-bit via `tifffile`
    when the array is `uint16`, an 8-bit TIFF for `uint8`, and a **float TIFF**
    (PIL mode `F`) for float arrays. For process-only this is exactly the wanted
    behavior: `rgb` (uint8/uint16) → integer TIFF at the source depth;
    `gray`/`detect_mat` (float) → **float TIFF that preserves the enhanced
    matrix's full precision** (no lossy quantization). The `bit_depth` argument
    is only consulted by the PNG/JPEG branches — irrelevant here. See §5.6.
  - `objmap.imsave(path, use_label2rgb=False)`
    (`accessors/_objmap_accessor.py:531`) writes the **raw labeled map**
    (integer labels). `objmap` is `uint16`, and `_save_image`'s PNG branch
    routes `uint16` to `_write_png_cv2`, so `.png` yields a 16-bit raw-label
    image — the prior CLI behavior, and correct as-is.
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

> **Implementation note (self-review — blast radius is large).** Re-rooting the
> helpers is **not** sufficient on its own: ~40 call sites **hand-join** the
> machine-state path instead of calling a helper, so each must be converted to
> the helper (write path) or its `resolve_*` variant (read path). Audit grep
> (run it first in the implementing session):
> ```
> grep -rn --include='*.py' -E '/ ?"progress"|/ ?PROCESSING_STATE_JSON|/ ?"processing_state\.json"|/ ?PROCESSING_EVENTS_LOG|/ ?"processing_events\.log"|/ ?DIR_PROGRESS' src/phenotypic | grep -v _io_constants.py
> ```
> Inventory captured 2026-06-03 (line numbers will drift — re-grep):
> - **`phenotypicCLI.py`** — L1013, L1038 (state), L1514, L1581, L1708, L1776 (progress/recompile)
> - **`_cli_state_management.py`** — L37, L79 (state), L92, L190 (event log)
> - **`_cli_execution_strategies.py`** — L121, L735 (event log), L290, L402, L633, L781 (progress)
> - **`_cli_recompile_worker.py`** — L122, L174, L263, L329
> - **`_cli_process_single.py`** — L465; **`_cli_output_manager.py`** — L303;
>   **`_cli_chunk_writer.py`** — L67; **`_cli_checkpoint_handler.py`** — L48
>   (hand-join `output_dir / DIR_PROGRESS` → convert). *Note:* L200
>   (`progress_dir.parent / PROCESSING_EVENTS_LOG`) is **already correct** once
>   `progress_dir` is the re-rooted value — leave it; it's precisely why D14
>   places the event log as a sibling of `progress/`.
> - **SLURM script gens** — `_cli_slurm_scripts.py` L187,
>   `_cli_slurm_array_scripts.py` L213, `_cli_recompile_slurm_scripts.py` L117
> - **`_dashboard/`** — `_manifest_builder.py` L311, L469; `_analysis_data.py`
>   L52; `_generator.py` L121, L145
> - **`tools_/generate_report.py`** L50, L118 and **`tools_/monitor_slurm_jobs.py`**
>   L55, L56 — read-only consumers; route through the `resolve_*` helpers so they
>   see both new and legacy runs. *(Both were missing from the earlier draft list.)*
> - **`phenotypic.sweep`** (`_sweep_slurm_scripts.py` L54, `_sweep_cli.py` L432,
>   `_sweep_execution.py` L338) — **intentionally NOT migrated** in v1 (D11).
>
> The migration is "done" when the audit grep returns only `_io_constants.py`
> and the sweep sites; enforce with a test/grep gate. Some readers also need new
> `resolve_*` variants beyond `resolve_processing_state_path` /
> `resolve_progress_dir` — see §5.2.

### 5.2 GUI integration

- `gui/_config.py`: keep re-exporting the directory-name constants; add
  `PHENOTYPIC_CACHE_DIRNAME = DIR_PHENOTYPIC`. Update the doc comments noting
  machine-state now lives under `.phenotypic/`.
- Run discovery (`shell/_runs_registry.py`, `shell/_classifier.py`,
  `run_console/_recent_runs.py`): resolve state through `resolve_*` helpers so
  both new and legacy runs are found. **Note:** the registry reads status via
  `manifest_json_path(output_dir)` (`_runs_registry.py:297`), which composes
  from `progress_dir` and therefore re-roots to `.phenotypic/` automatically —
  so for a **legacy** run (manifest at `<output>/progress/manifest.json`) the
  helper would miss it. Add a `resolve_manifest_json_path()` (and, as needed,
  `resolve_*` for `failures`/`job_metadata`) with the same first-existing
  semantics as `resolve_progress_dir`, and have the GUI readers call those.
- Keep the existing "skip hidden dirs" behavior for *dataset/run candidate*
  scanning (add `.phenotypic` to the skip set alongside `.phenotypic-gui`,
  `.gui_log`), but ensure `.phenotypic/` of a real run root is still **read**
  for state.
- **Process-only runs (D13)** write a progress manifest + state but **no
  `deliverables/`**. The classifier's `out`/`has_dashboard` badges already key
  on `deliverables/`, so a process-only run correctly shows progress without a
  dashboard/results affordance — no extra classifier work expected; add a test
  asserting it.
- `gui/FEATURES.md` (CI-gated): the `features-md-gate` job rejects any PR
  touching `src/phenotypic/gui/` without editing `FEATURES.md`, and pre-commit
  validates the `Test ref` on `✅ shipping` rows — so Phase 1 **must** add/adjust
  a row (where run state is read from) with a real test ref.

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

- **`ExecutionConfig`** (a dataclass in `_cli_types.py:82`) gains an optional
  `process_only_layer: ProcessOnlyLayer | None` field (a
  `Literal["rgb","gray","detect_mat","objmap"]` alias in `tools_/typing_.py`
  per the CLAUDE.md closed-set rule — type-only, no Enum).
- **Input discovery (D12)**: reuse `scan_directory_structure` (1-level; mixed
  root+subdir inputs rejected — process-only inherits that rejection). The
  `input_root` for mirroring is the CLI `--input` path (its parent for a single
  file).
- **Strategy dispatch — worker *and* finalize**: `LocalParallelStrategy` and
  `AutonomousSLURMStrategy` branch on `config.process_only_layer`. This is **not
  just a worker swap** (self-review finding):
  - **Per-image callable**: unset → `process_single_image_core` (or measure
    path); set → new `process_single_apply_only_core`.
  - **Finalize**: process-only writes a **progress manifest but no dashboard
    HTML**. ⚠️ `regenerate_dashboard_artifacts()`
    (`_cli_execution_strategies.py:211`, `_generator.py:107`) does **both**
    `build_manifest` *and* `generate_dashboard` (HTML) — so do **not** call it
    wholesale (no manifest → the run console can't show the run, breaking D13)
    **and** do not call it as-is (we don't want the dashboard HTML). Instead
    call **`build_manifest` only** (the manifest half) locally; on SLURM,
    submit the image array **without** the aggregation/sentinel/checkpoint
    chain and add a one-shot finalize task that calls `build_manifest`. The
    manifest's `is_complete` is the completion signal the run console reads.
- **New worker** `process_single_apply_only_core(pipeline_path, image_path,
  input_root, output_dir, image_type, layer, read_kwargs, ...)`:
  1. Load pipeline + image (same read-kwargs / grid-shape resolution as the
     forward worker).
  2. `image = pipeline.apply(image, inplace=True)`.
  3. Compute the **mirrored output path** (see 5.5) and write the layer (5.6).
  4. Append the same `started`/`completed`/`failed` events to the event log so
     resume + progress tracking work unchanged (event-log keying; see §6).
- **`input_root` plumbing**: local worker closure captures it from
  `self.config.input_path`; the SLURM worker CLI (`_cli_process_single.main`)
  gains `--process-only LAYER` **and** `--input-root PATH`, threaded by
  `_cli_slurm_array_scripts.py`. (Default chosen over deriving from per-image
  args — explicit and unambiguous.)
- **Outputs**: process-only writes the mirrored layer files + a progress
  manifest/state/event log + a `.phenotypic/pipeline.json` copy
  (reproducibility). It writes **no** `deliverables/`, `results/`, QC, overlays,
  aggregation, or analysis HTML. Per D13 the progress manifest makes the run
  visible in the run console; the absence of `deliverables/` keeps it out of the
  results viewer's dashboard affordance.

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

### 5.6 Layer write semantics (D3, D9, D10)

**Rule (revised per user): delegate to the layer accessor's `imsave`.** The
process-only writer does **not** reimplement IO or quantize pixels. It calls
`getattr(image, layer).imsave(out_path)` — the single, golden-tested writer
(`_accessor_io_handler.imsave` for `gray`/`detect_mat`, the multichannel
`imsave` for `rgb`, `accessors/_objmap_accessor.imsave` for `objmap`). That
method embeds the PhenoTypic metadata JSON and picks the format from the path
extension. The writer's only added responsibilities are creating the parent
directory and emitting the D9 empty-objmap warning.

```python
def write_process_only_layer(image, layer, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    accessor = getattr(image, layer)
    if layer == "objmap" and accessor.isempty():
        warnings.warn(f"pipeline produced no objects; writing empty object map to {out_path}")
    accessor.imsave(filepath=out_path)   # native dtype + embedded metadata
```

**Resulting per-layer output** (extension chosen in §5.5: `.png` for `objmap`,
`.tiff` otherwise), verified against the runtime dtypes of
`load_synth_yeast_plate()` (an 8-bit image):

| Layer | dtype | `imsave` output | Note |
|-------|-------|-----------------|------|
| `rgb` | `uint8`/`uint16` | integer TIFF at the **source bit depth** | follows `image.bit_depth` for free |
| `gray` | `float` | **float TIFF** (PIL mode `F`) | full precision preserved |
| `detect_mat` | `float` | **float TIFF** | full precision of the enhanced matrix preserved |
| `objmap` | `uint16` | **16-bit** raw-label PNG (`_write_png_cv2`) | `use_label2rgb=False`; label maps stay 16-bit regardless of source depth |

This is a deliberate change from the earlier draft's worker-side
`img_as_ubyte`/`img_as_uint` quantization: writing `gray`/`detect_mat` as
**float TIFFs** *preserves* the enhanced matrix's full precision (the prior
"precision note" concern disappears), and reusing the shared `imsave` keeps a
single, golden-tested writer instead of a parallel one. There is **no** lossy
8-bit collapse of `detect_mat`, and **no** modification of the shared accessor
IO. Output compression follows whatever `imsave`'s writers
(`_write_tiff_tifffile` / PIL / `cv2`) already do — the process-only path adds
none of its own.

> **Implementation check (TDD, Phase 2 Task 3).** The unit test calls
> `write_process_only_layer` for each layer and asserts the produced file is
> readable at the expected dtype (`rgb` → integer, `gray`/`detect_mat` →
> floating, `objmap` → `uint16`). If `imsave`'s float-TIFF path needs a
> `float64 → float32` view for PIL/`tifffile` compatibility, make that the
> *only* adjustment inside the accessor IO (additive, default-preserving) — do
> not reintroduce intensity quantization.

- **objmap with no detector in the pipeline** → `image.objmap.isempty()` is
  true: log a per-image warning ("pipeline produced no objects; writing empty
  object map for <image>") and write the empty (all-zero) map. The run does not
  fail.

### 5.7 `--dry-run`

Print the resolved plan: mode = process-only, layer, per-dataset image counts,
sample of mirrored output paths, execution mode (local/SLURM), and the
`.phenotypic/` location — without processing.

## 6. Error handling & edge cases

- Layer writing delegates to `accessor.imsave` (§5.6): `rgb` → integer TIFF at
  the source bit depth, `gray`/`detect_mat` → float TIFF (full precision),
  `objmap` → 16-bit raw-label PNG; metadata embedded. No worker-side
  quantization.
- Per-image failures use the **same** failure-tracking path as the forward run
  (`append_failure` into `.phenotypic/progress/failures.jsonl`), so one bad
  image doesn't abort the batch.
- `--process-only` + `--resume`: **resolved during self-review** — completion is
  tracked by the append-only event log / `ProcessingState.completed` set
  (`_cli_state_management.get_datasets_with_remaining_images` keys on
  `ds_state.completed`, *not* on any measurement artifact). So as long as the
  process-only worker appends the same `started`/`completed`/`failed` events,
  resume skips already-written images with **no change to the resume
  predicate**.
- Input shapes (bounded by the 1-level scanner, D12): single file → `<output>/
  <stem>_<layer>.<ext>`; flat dir → files mirror at the output root; one level
  of subdirs → `<output>/<subdir>/<stem>_<layer>.<ext>`. Mixed root+subdir
  inputs are rejected by the scanner (process-only inherits this). Deeper
  nesting is **not discovered** — documented limitation, not a bug.

## 7. Testing strategy

**Unit**
- `_io_constants`: new layout test — machine-state helpers root under
  `.phenotypic/`; `deliverables_dir`/`results_dir`/`qc_dir`/`logs_dir`/
  `slurm_scripts_dir` unchanged. (Extends `tests/unit/tools_/test_io_constants.py`.)
- Back-compat resolvers: legacy-root state file is found when `.phenotypic/`
  is absent; new path wins when both exist; fresh write targets `.phenotypic/`.
- `ProcessOnlyLayer` Literal alias presence (type-only set).
- Output-path mapping: one-level-subdir / flat-dir / single-file inputs →
  expected mirrored paths and `_layer` suffix; objmap → `.png`, others →
  `.tiff`. (Deeper nesting is out of scope per D12 — assert the scanner's
  mixed-input rejection too.)
- Worker: each layer writes a file in the right format (small synth image).
  **Explicit dtype assertions** (load the written file via `tifffile`/`cv2`):
  - `rgb` → integer TIFF at the source bit depth (`uint8` for an 8-bit source,
    `uint16` for a 16-bit source).
  - `gray`/`detect_mat` → **float TIFF** (`np.issubdtype(arr.dtype, np.floating)`),
    full precision preserved by `imsave`; **not** quantized to `uint8`.
  - `objmap` PNG is always `uint16` raw labels, **regardless** of
    `image.bit_depth` (assert with an 8-bit source).
  - Metadata round-trips: the PhenoTypic JSON `imsave` embeds is present in the
    written file's tags/chunk.
- objmap-without-detector warns and writes an all-zero map.
- CLI validation matrix: every rejected-flag combination raises `UsageError`;
  required-flag enforcement; happy-path parsing populates
  `ExecutionConfig.process_only_layer`.

**Integration**
- `--process-only detect_mat` over a one-level-subdir fixture, local executor:
  asserts mirrored TIFFs at the fixture's bit depth (uint8 for an 8-bit
  fixture) land at `<output>/<subdir>/<stem>_detect_mat.tiff`, **no**
  `deliverables/`/`results/`, and `.phenotypic/` tracking (state + progress
  manifest) is present.
- `--dry-run` output contains the plan and the `.phenotypic/` path.
- SLURM script generation asserts `--process-only` **and** `--input-root` are
  threaded into the array script, and that the **finalize/aggregation chain is
  omitted** (only the image array + completion marker) (no cluster needed).
- Regression: the existing forward run still works with state under
  `.phenotypic/`; a **legacy-layout** run (state at the old root) still
  resumes and is GUI-discovered (exercises the `resolve_*` fallbacks).
- A "no hand-joined state paths" grep gate test (audit grep returns only
  `_io_constants.py` + the intentionally-excluded sweep sites).

## 8. Implementation phasing

Two phases, each its own plan + PR + green CI. Phase 1 stands alone; Phase 2
depends on it. **Recommended as two separate fresh sessions** given Phase 1's
~40-site blast radius.

1. **Phase 1 — `.phenotypic` migration** (no behavior change except state
   location): add `DIR_PHENOTYPIC` + `phenotypic_cache_dir` + re-root
   `progress_dir`/`processing_state_path`/`event_log_path`; add `resolve_*`
   back-compat readers (incl. `resolve_manifest_json_path`); **convert all ~40
   hand-joined sites** (§5.1 inventory) to helpers; update GUI readers + add
   `.phenotypic` to the hidden-skip set + `FEATURES.md` row; tests incl. the
   grep gate + legacy-resume + legacy-discovery. Sweep (D11) intentionally
   excluded.
2. **Phase 2 — process-only mode**: `ProcessOnlyLayer` alias; `ExecutionConfig`
   field; `process_single_apply_only_core` + `imsave`-delegating layer writer;
   **strategy dispatch for worker *and* finalize** (skip dashboard locally; skip
   aggregation chain on SLURM, emit completion marker); worker-CLI
   `--process-only`/`--input-root` + array-script threading; top-level CLI option
   + validation; mirrored output writer; `.phenotypic/pipeline.json` copy;
   `--dry-run`; tests + docs (CLI docstring, `CLAUDE.md` CLI section, README).

## 9. Risks

- **Migration blast radius (large)**: **~40 hand-joined sites** across the CLI,
  dashboard builders, SLURM script generators, and two `tools_/` reporters
  (full inventory in §5.1) — not the handful first assumed. A missed site is the
  main failure mode → mitigated by converting every site to the `_io_constants`
  helpers/`resolve_*` readers + a "no hand-joined state paths outside
  `_io_constants` (sweep excepted)" grep-gate test + the `_io_constants` layout
  test. `phenotypic.sweep` is deliberately left on the legacy layout (D11), so
  the gate must allowlist its sites.
- **GUI run discovery regressions**: covered by back-compat resolvers + a
  legacy-layout discovery test + `FEATURES.md` gate. Note status reads already
  go through `manifest_json_path()` (`_runs_registry.py:297`), so they follow
  the re-root automatically; the resolver fallback is what keeps **legacy**
  runs visible.
- **Resume predicate**: ✅ resolved (event-log based — see §6). No predicate
  change needed.
- **Layer writing reuses shared IO**: the writer delegates to `accessor.imsave`
  (§5.6), so `rgb` follows `image.bit_depth` for free and `gray`/`detect_mat` are
  written as float TIFFs (full precision). No parallel writer, no quantization;
  the only residual risk is `imsave`'s float-TIFF path needing a `float64→float32`
  view for PIL — caught by the §7 dtype round-trip assertions, fixed additively in
  the accessor IO if needed.

## 10. Open questions for implementation (non-blocking)

Resolved since the first draft: event-log placement → **sibling inside
`.phenotypic/`** (D14, forced by `_cli_checkpoint_handler.py:200`); SLURM
`input_root` → **new `--input-root` flag** (§5.4); sweep scope (D11), mirror
depth (D12), GUI visibility (D13). Remaining:

- **SLURM completion-marker shape for process-only.** Resolved in direction
  (§5.4): a one-shot finalize task calls `build_manifest` (no aggregation, no
  dashboard HTML); `manifest.json::is_complete` is the signal. Remaining detail
  for the plan: whether that finalize task is a trivial sbatch dependent on the
  array, or the last array task self-elects — pick in Phase 2.
- **Process-only dataset/state keying under the 1-level scanner.** Confirm the
  `(dataset, image)` event keys the worker emits are unique across the mirrored
  layout and that resume's `ds_state.completed` lookup matches them (it should,
  since keying is unchanged from the forward path — add an explicit resume test).
- **`.phenotypic/pipeline.json` helper.** Needs a new path helper distinct from
  `pipeline_json_path` (which roots under `deliverables/`). Trivial; name it in
  the plan (e.g. `phenotypic_cache_pipeline_json_path`).

## 11. Fresh-session implementation handoff

A cold-start session should, in order:

1. **Read this spec** end-to-end, then **re-run the §5.1 audit grep** (line
   numbers will have drifted) to regenerate the live hand-joined-path list.
2. **Use the code-review-graph MCP tools first** (per project CLAUDE.md) —
   `query_graph`/`get_impact_radius` on `progress_dir`, `processing_state_path`,
   `manifest_json_path` — before Grep/Read, to catch any site the grep misses
   (e.g. f-strings or aliased imports).
3. **Do Phase 1 and Phase 2 as separate plans/PRs** (§8); land Phase 1 green
   before starting Phase 2.
4. **Watch the CI gates**: the `gui-checks` `features-md-gate` rejects any
   `src/phenotypic/gui/` change without a `FEATURES.md` edit (+ pre-commit
   `Test ref` validation on `✅ shipping` rows); `WORKFLOWS.md` is **not**
   triggered (no new end-to-end GUI flow). Run `uv run mypy src/phenotypic` and
   `uv run ruff check --fix` at completion boundaries; tests via `uv run pytest`.
5. **Don't confuse the two hidden dirs**: `.phenotypic/` is **run-root
   machine-state** (this work); `.phenotypic-gui/` is the **GUI sandbox** dir
   for presets/state (`gui/_config.py:SANDBOX_GUI_DIRNAME`) — unrelated, do not
   merge them.
6. **Tests that will need updating** (expect, don't be surprised): anything
   asserting `<output>/processing_state.json` or `<output>/progress/...` at the
   root — search `tests/` for those literals and the `TestDeliverablesLayout`
   neighbors in `tests/unit/tools_/test_io_constants.py`.
7. **Reproducibility while iterating**: this is a worktree on branch
   `worktree-cli-processing-mode`; commit each completed phase to lock progress
   (shared-worktree git index — be the sole committer, scope commits with
   `git commit -- <paths>`).
