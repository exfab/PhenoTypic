# Reference-metadata operations (`ReferenceContext`, `RefMetadata`, `SubtractBlank`)

**Status:** draft for review · **Date:** 2026-10-05 · **Branch:** `worktree-ref-metadata-ops`

## 1. Objective

Let an `ImageOperation` read **per-image values from an experiment metadata
table** while it runs, and use those values — including to load *another*
image. The motivating case is time-series background removal for ucr_033
(Linzer *Ganoderma*): frame 0 of each plate is the media blank, and every later
frame of that plate should have the blank's signal subtracted from its
`detect_mat` before detection.

Today no metadata reaches `apply()`. CLI `--metadata` is joined onto
measurement tables only at finalization (`_cli_output_manager.py`,
`_cli/_metadata_join.py`), and `JoinMetadata` is a `PostMeasurement` acting on
DataFrames. This spec adds the missing channel and one operation that uses it.

The feature must work identically through the three surfaces: the Python API,
the CLI (local, SLURM, staged GPU), and the GUI.

### Success criteria

1. In Python, `with ReferenceContext(table, image_root=...): pipe.apply(img)`
   runs a pipeline containing `SubtractBlank`; without a context it raises an
   error that names the fix.
2. `python -m phenotypic --metadata blank_map.csv ...` runs the same pipeline
   on ucr_033 frames (full and process modes, local and SLURM); the run
   preflight refuses before submission when the table cannot serve the
   pipeline.
3. The GUI builder previews `SubtractBlank` against a picked table, and the run
   console refuses Run when a needed table is missing.
4. Every image's provenance journal records which table and which blank it
   used.
5. No wrong-metadata case fails silently (§8 catalogue).

## 2. Non-goals

- **Processing the blank through earlier ops** (option B of the design
  discussion). It needs a tree-prefix extractor, breaks per-image independence
  in the CLI, and is silently wrong for per-image-fitted ops
  (`ContrastStretching`, `CalibrateColorRpcc`). If needed later, the extension
  is an explicit optional `blank_ops` sub-pipeline on `SubtractBlank` (§11).
- **Registration/alignment** of the blank to the target. Frames are assumed
  pixel-aligned (fixed imager position); a shape mismatch is refused.
- **`RefMetadata` on measurers, post-measurement, filters, or models.** v1 is
  `ImageOperation`-only (§4.2).
- **Tuning** a pipeline that contains a `RefMetadata` op (`phenotypic-tune`
  refuses it, §5.4).
- **A metadata path stored on the operation.** Deliberately removed (D1).

## 3. Decisions

| # | Decision | Why |
|---|---|---|
| D1 | `RefMetadata` ops carry **no table path**. Reference data comes **only** from an active `ReferenceContext`. | Including metadata is a deliberate act at run time; a pipeline can never silently run against a stale table it embedded. Removes the precedence rule, effective-pipeline rewriting, and per-op table snapshots. |
| D2 | `ReferenceContext` is a **public core class**, `phenotypic.ReferenceContext`, usable as a context manager. | Users can prototype — inspect exactly what an op will see (`lookup`, `load_image`) without running it. |
| D3 | Activation uses a `contextvars.ContextVar`, the same mechanism as `_core/_provenance.py`. | Reaches ops nested at any depth (`CompositeEnhance`, `CompositeDetector`, branch pipelines) without threading an argument through every `_operate`; exception-safe restore. |
| D4 | Subclasses declare their columns as **ordinary typed fields** typed `RefColumn` / `RefImageColumn`. | Each op targets its own columns; the choice serializes with the pipeline; tooling (preflight, GUI dropdowns, `reference_columns()`) discovers them without per-op code. |
| D5 | The CLI's table is the existing `--metadata`, snapshotted to `deliverables/metadata.csv` (full) or `.phenotypic/reference_metadata.csv` (process). | One table serves both the reference lookup and the measurement join, so they cannot disagree; the snapshot is byte-stable provenance. |
| D6 | `SubtractBlank` subtracts the blank's **fresh channel in the target's detect mode**, and requires the target's `detect_mat` to be **fresh** (unmodified since its last reset) with **no `ImageCorrector` earlier** in the application. | Position-independent correctness: valid at root, after `SetDetectMode`, or inside a composite branch; refuses rather than mis-subtracts. `SetDetectMode` discards prior enhancement (`_set_detect_mode.py`; `_image_handler.py:403`), so "must be first" was the wrong rule. |
| D7 | Lookup is **strict**: missing row, missing column, null, or disagreeing values all raise. Self-reference raises. | The ucr_033 table is per-colony (~96 rows per image); strictness turns every malformed-table case into a loud failure. |

## 4. Core components

### 4.1 `ReferenceContext` — `src/phenotypic/_core/_reference_context.py`

Exported lazily from `phenotypic/__init__.py` via `_LAZY_CLASSES`
(`"ReferenceContext": "._core._reference_context"`) and listed in `__all__`.
The module imports only the standard library at module level; polars is
imported inside methods (polars/pandas are in `HEAVY_STARTUP_MODULES`).

```python
class ReferenceContext:
    def __init__(
        self,
        metadata: str | Path | "pd.DataFrame" | "pl.DataFrame",
        *,
        image_root: str | Path | None = None,
        images: Mapping[str, "Image | str | Path"] | None = None,
        dataset: str | None = None,
        read_kwargs: Mapping[str, Any] | None = None,
    ) -> None: ...

    # activation
    def __enter__(self) -> "ReferenceContext": ...
    def __exit__(self, *exc) -> None: ...
    @classmethod
    def current(cls) -> "ReferenceContext | None": ...

    # the surface ops call — public so users can prototype with it
    def lookup(self, image: "Image | str", columns: Sequence[str]) -> dict[str, Any]: ...
    def resolve_image(self, name: str) -> "Path | Image": ...
    def load_image(self, name: str) -> "Image": ...
    def narrow(self, *, dataset=None, image_root=None, images=None) -> "ReferenceContext": ...
    def has_column(self, name: str) -> bool: ...
    def reference_image_digest(self, name: str) -> str | None: ...

    # provenance
    @property
    def table_sha256(self) -> str | None: ...
```

**Construction** reads and validates the table immediately, so a bad table
fails before the first image:

- A `.csv` path is read with `infer_schema=False` and every column is cast to
  string (parquet via polars, then cast): every value is a name, and type
  inference would turn the stem `000123` into the integer `123`.
- Headers are normalized with the same in-memory canonicalization the CLI join
  uses (`normalize_metadata_columns` and friends), so `ImageName` and
  `Metadata_ImageName` are equivalent. The source file is never rewritten.
- The table must contain `Metadata_ImageName`; otherwise `ReferenceTableError`.
- `table_sha256` is the SHA-256 of the file bytes for a path, `None` for an
  in-memory frame.
- `images` takes precedence over `image_root` when resolving a name; at least
  one is required only when an op calls `resolve_image`/`load_image`.

**Activation.** `__enter__` sets the module `ContextVar` and pushes the token
on the instance; `__exit__` resets it in all cases. An inner context
**replaces** the outer one entirely (no field merging);
`narrow(*, dataset=None, image_root=None, images=None)` is the explicit way to
derive a context that shares the parsed table and lookup indexes (each `None`
keeps the parent's value). A single instance must not be entered concurrently from two threads
(documented).

**`lookup(image, columns)`** — the semantics every op inherits:

1. Key: `Metadata_ImageName == image.name` (or the given string), plus
   `Metadata_Dataset == self.dataset` when both the context has a dataset and
   the table has that column.
2. Each requested column resolves against the table like `JoinMetadata.on`:
   the literal first, then the schema-prefixed spelling of a bare label, using
   the schema helpers (never a string-prefix check).
3. Zero matching rows → `ReferenceLookupError` naming the image and key.
4. Per column, the non-null values across the matching rows must collapse to
   **exactly one** distinct value; none → `ReferenceLookupError` (null);
   several → `ReferenceLookupError` listing them (the per-colony disagreement
   case).
5. Returns `{requested_column_name: value}`.

**`resolve_image(name)`**: `images[name]` if present; else the unique file in
`image_root` whose stem is `name` (any suffix), or the file named exactly
`name` when it includes its extension. Zero or several matches →
`ReferenceImageError` listing candidates.

**`load_image(name)`**: resolves, then reads with `Image.imread` using the
**reader settings the context was given** (`read_kwargs`, set by the CLI to
the same kwargs it uses for inputs; empty in Python). Loaded images are held in
a process-level LRU keyed by `(resolved_path, st_mtime_ns, st_size,
read_kwargs)`, so a worker reads each blank once across all its images. An
`images=` entry that is already an `Image` is returned as-is (shared, not
copied — callers must not modify it). A store directory (`x.ome.zarr`) resolves
by its source stem (`x`), and its identity digest is its root `zarr.json`.

**Errors** (all in the same module, all `ValueError` subclasses rooted at
`ReferenceContextError`, re-exported from `phenotypic.sdk_`, §8):
`RefMetadataUnavailableError`, `ReferenceTableError`, `ReferenceLookupError`,
`ReferenceImageError`.

**Boundaries.** A context is per-process and per-thread. Workers (loky, SLURM)
build their own (§5.2). No pipeline or composite code starts threads today; an
op that ever ran children on a thread must use `contextvars.copy_context()`.
Ops must read the context inside `_operate`, never defer it.

### 4.2 `RefMetadata` mixin and `RefColumn` / `RefImageColumn` — `src/phenotypic/abc_/_ref_metadata.py`

Exported from `phenotypic.abc_`. Fieldless, like `PlotImage`.

The column markers reuse the existing `_ColumnRefMarker` (`sdk_/_column_ref.py`),
which the GUI registry already renders as a dropdown, with a new source
`"reference_metadata"`:

```python
# sdk_/_column_ref.py  (ColumnSource gains "reference_metadata")
RefColumn = Annotated[str, _ColumnRefMarker("reference_metadata")]
RefImageColumn = Annotated[str, _ColumnRefMarker("reference_metadata"), _ReferenceImageMarker()]
#   RefColumn: a value.  RefImageColumn: a value naming another image -- the CLI
#   resolves these to files at startup and in its preflight.

# abc_/_ref_metadata.py
class RefMetadata:
    def __init_subclass__(cls, **kw):
        # v1: only ImageOperation subclasses may mix this in (TypeError otherwise)

    def _ref_columns(self) -> tuple[str, ...]:
        # default: values of every model field carrying _ColumnRefMarker("reference_metadata"),
        # in declaration order; override only for computed column names

    def _ref_image_columns(self) -> tuple[str, ...]:
        # the subset whose fields also carry _ReferenceImageMarker

    def _ref_values(self, image) -> dict[str, Any]:
        # ReferenceContext.current() or RefMetadataUnavailableError; then ctx.lookup
        # records the resolved values for provenance (§6)

    def _ref_image(self, name: str) -> "Image":
        # ReferenceContext.current().load_image(name); records name + digest (§6)
```

`RefMetadataUnavailableError`'s message names the op, its columns, and both
fixes:

```
SubtractBlank reads ('Metadata_BlankImage',) from a ReferenceContext, but none is active.
  Python: with phenotypic.ReferenceContext('blank_map.csv', image_root='images/'): pipe.apply(img)
  CLI:    pass --metadata blank_map.csv
```

`RefColumn`/`RefImageColumn` values are plain strings in `pipeline.json`. The tune
annotation-coverage gate applies to numeric fields only, so no `TuneSpec` is
needed; tune auto-search treats them as non-numeric.

### 4.3 `ImagePipeline.reference_columns()`

`reference_columns(*, images_only=False)` returns `{tree_path: columns}` for
every `RefMetadata` op anywhere in the operation tree, using the same tree-path
spelling as the staged-GPU walker (`sdk_._operation_tree.find_operations`,
joined with `/`); `images_only=True` returns only `RefImageColumn` columns. Empty dict when the pipeline
needs no table. Used by the CLI preflight, the GUI, the tune refusal, and
users.

### 4.4 `SubtractBlank` — `src/phenotypic/enhance/_subtract_blank.py`

```python
class SubtractBlank(BackgroundSubtraction, RefMetadata):   # BackgroundSubtraction is an ImageEnhancer
    blank_column: RefImageColumn = "Metadata_BlankImage"
    polarity: Literal["brighter", "darker", "both"] = "brighter"
```

`_operate(image)`:

1. `name = self._ref_values(image)[self.blank_column]`.
2. `name == image.name` → `ReferenceLookupError` (self-reference; a blank
   frame is not its own background — exclude blanks from the input, §5.2).
3. **Freshness guard (D6):**
   - `np.array_equal(image.detect_mat[:], get_detection_mode(image.detect_mode).compute(image))`
     must hold, else `StaleDetectMatError` ("place SubtractBlank before any
     enhancer, or directly after SetDetectMode").
   - No operation record in **any** of the image's provenance applications
     (a staged Stage-2 probe copy opens a fresh one) has an `operation_class`
     naming a loaded `ImageCorrector` subclass, else
     `StaleDetectMatError` (a corrector changed the target's pixels; the raw
     blank does not share that change). Limitation: a corrector inside the
     same composite branch is recorded only when the composite finishes, so it
     is not seen; correctors do not live in composites today.
4. `blank = self._ref_image(name)`; refuse with `ReferenceImageError` if
   `blank.gray.shape != image.gray.shape` or bit depths differ.
5. `b = get_detection_mode(image.detect_mode).compute(blank)` — the blank's
   fresh channel in the **target's** mode (grey in grey mode, `LabL` in `LabL`
   mode, …). Raises the mode's own error if the blank lacks RGB for an RGB mode.
6. With `t = image.detect_mat[:]` (float32, `[0, 1]`):
   - `brighter`: `clip(t − b, 0, 1)` — colonies brighter than media.
   - `darker`: `clip(b − t, 0, 1)` — colonies darker than media; result is
     object-bright.
   - `both`: `|t − b|`.
7. Write back into `image.detect_mat`; `rgb`/`gray` untouched (enhancer
   integrity validation already enforces this).

Docstring follows the `abc_/CLAUDE.md` order with a runnable doctest built on
`load_synth_yeast_plate()` and `ReferenceContext(..., images={...})`, plus a
`Best For` naming time-series plates with a media-only frame and `Consider
Also` pointing at `SubtractGaussian`/`SubtractRollingBall` for single-image
background estimation.

## 5. Surface integration

### 5.1 Python

```python
from phenotypic import Image, ImagePipeline, ReferenceContext
from phenotypic.enhance import SubtractBlank

pipe = ImagePipeline(ops={"sb": SubtractBlank(), "det": OtsuDetector()})
pipe.reference_columns()        # {'sb': ('Metadata_BlankImage',)}

ctx = ReferenceContext("blank_map.csv", image_root="images/")
ctx.lookup("d000426_300_123_2026-05-27_02-41-37", ["Metadata_BlankImage"])
with ctx:
    out = pipe.apply(Image.imread("images/d000426_300_123_2026-05-27_02-41-37.tif"))
```

No `pipeline.apply(..., metadata=)` keyword: the context is the one way.

### 5.2 CLI

**Which table.** The existing `--metadata`. When
`pipeline.reference_columns()` is non-empty:

| Mode | Snapshot | Continuation without re-passing `--metadata` |
|---|---|---|
| `full` | `deliverables/metadata.csv` (existing, unchanged) | existing fallback to the snapshot |
| `process` | **new:** `.phenotypic/reference_metadata.csv`, byte-for-byte, atomic; preserved across `--restart` as `deliverables/metadata.csv` is | **new:** same fallback rule |
| `measure`, `recompile`, `migrate` | — | ops are not applied; no context |

Today `process` mode ignores `--metadata` (`phenotypicCLI.py:2335`); it now
uses it **only** when the pipeline needs reference columns; otherwise its
behaviour is unchanged (ignored), so existing process invocations — including
the run console's, which may pass `--metadata` regardless of mode — keep
working.

**Reference manifest.** Stage-3 and SLURM workers know only the run root and
the dataset name, so the CLI plans references once. At startup, before any
work-id is computed (`_prepare_incremental_startup`), it resolves every input
image's reference values and reference-image files with the same planner the
preflight uses (`_cli/_cli_reference.py`, `plan_references`) and writes
`.phenotypic/reference_manifest.json`: the table path and SHA-256, the input
reader kwargs, each dataset's `{name: absolute path}` map, and each image's
reference digest. A run whose pipeline needs no references deletes any stale
manifest.

**Where the context is entered.** Each worker core wraps its existing apply
call in `worker_reference_context(output_dir, dataset_name)`, which builds one
base context per process from the manifest (refusing if the table's bytes no
longer match the manifest's digest) and narrows it to the dataset and its
resolved image map:

- `_cli_process_single.py:334` (`apply_and_measure`)
- `_cli_process_only.py:347` (`apply`)
- `_cli_staged_workers.py:369` (Stage 1 pre-GPU `apply`)
- `_cli_staged_workers.py:592` (Stage 3 replay `apply`)

The resolver therefore matches a blank stem within the **same dataset's input
directory**, consistent with `Metadata_ImageName` being the input stem
(`Image.imread` sets `name = filepath.stem`, `_image_io_handler.py:796`). A
blank that is not itself an input can still live there.

**Identity.** The `--metadata` digest today enters only the *finalization*
digest (`_cli_identity.py:333`), not the per-image work-id — metadata never
changed per-image science before. Now it can, so `compute_work_id` gains an
optional **per-image reference digest** (canonical digest of the image's
resolved reference values plus each reference image file's SHA-256), read from
the manifest by both the main process and SLURM workers. It is present only
when the pipeline needs references, so every existing work-id is unchanged; and
it is per image, so editing one plate's blank re-runs that plate's frames and
nothing else, in every mode.

**Run preflight** (`_cli_preflight.py`; new codes added to the closed set and
`HINTS`). Checks read only the pipeline, the table, and directory listings —
no image is opened, per the module contract. Severity follows the module rule:
a finding that fails every image is an `error`, one that fails some is a
`warning` (escalated to `error` when it covers every input).

| Code | Severity | Condition |
|---|---|---|
| `PF-REF-NO-TABLE` | error | pipeline needs reference columns; no `--metadata` and no snapshot to fall back to |
| `PF-REF-TABLE` | error | the table cannot be read, or lacks `Metadata_ImageName` |
| `PF-REF-COLUMN` | error | table lacks a column some op names (message lists op path → column) |
| `PF-REF-UNMATCHED` | warning | input images with no row |
| `PF-REF-AMBIGUOUS` | warning | input images whose rows are null or disagree for a needed column |
| `PF-REF-SELF` | warning | input images that name themselves as their blank (hint: exclude them, e.g. `--image-manifest`) |
| `PF-REF-UNRESOLVED` | warning | a named reference image matches 0 or >1 files in its dataset directory |

Preflight runs before `--overwrite` clears anything and before `--dry-run`
exits, like the staged-GPU refusal. It resolves reference-image *names* to
files but never opens or hashes them; startup does the hashing.

**SLURM / staged GPU.** No new jobs. Every stage that applies operations
enters the context: Stage 1 (pre-detector ops); **Stage 2**, because a
`SubtractBlank` inside the GPU detector's own sequence branch is moved into the
`stage2_prefix` (`_cli_pipeline_split.py:84-97`) and applied before inference;
Stage 3, because it re-runs the detector's whole top-level ancestor; and the
`--layer objmap` export, which re-applies the post-detector chain. (An earlier
draft said SubtractBlank "lands in Stage 1 by construction" — false for the
in-branch placement; corrected after plan review B1.)

**Image identity.** The planner, work-ids and resolution all key an image by
`source_image_stem` — what `Image.imread` names it — so `x.ome.zarr` is `x`.

**Guards around the snapshot.** Measure mode never touches the manifest (it
may run beside live forward workers). A run that needs references and passes
`--overwrite` must pass `--metadata` (the snapshot it would fall back to is
deleted). The reference table is parsed early, before any output change and
regardless of `--skip-validation`. The snapshot is on shared
storage under `--output`; workers read it, never the user's original.

### 5.3 GUI

- **Builder parameter form**: `RefColumn` fields are free text in v1; the
  picker's status line lists the picked table's columns. A live dropdown needs
  the picked path carried in builder state (the inspector is rendered by
  `_render_views` from 12 callbacks) and is a follow-up. The GUI registry
  already reports these fields as `column_ref.source == "reference_metadata"`.
- **Builder preview** (`_preview_cache.py:386`, `_callbacks.py:6090`): a
  session-level **Reference metadata** file picker in the preview panel. When
  set, `apply_with_intermediates` runs inside
  `ReferenceContext(table, image_root=<preview image's directory>)`. It is not
  saved into the pipeline. With no table, a `RefMetadata` node shows the
  `RefMetadataUnavailableError` text instead of a preview.
- **Run console**: when the loaded pipeline's `reference_columns()` is
  non-empty, the metadata field becomes required and Run is disabled until it
  is set; the CLI's `PF-REF-*` findings surface through the console's existing
  preflight display.
- `FEATURES.md`, `WORKFLOWS.md`, and tutorial screenshots updated per the
  `gui-tutorial-capture` skill.

### 5.4 Tune

`phenotypic-tune` refuses, at spec load, any pipeline whose
`reference_columns()` is non-empty, with a message naming the op paths.
Entering a context in the tune evaluator (`tune/_evaluation/_evaluator.py:398`)
is a follow-up.

## 6. Provenance

`RefMetadata` overrides the existing `provenance_parameters()` hook
(`_provenance.py:943`) to append a `_references` entry to the op record's
`parameters` (a leading underscore cannot collide with a pydantic field):

```json
"_references": {
  "table_sha256": "…",
  "values": {"Metadata_BlankImage": "d000426_300_123_2026-05-24_11-40-37"},
  "images": {"d000426_300_123_2026-05-24_11-40-37": {"sha256": "…"}}
}
```

The values are resolved within the same apply frame that records them. Image
digests are of the file bytes (`null` for an in-memory `Image`). Everything
recorded is a function of the inputs, so process stores stay bit-reproducible.
On a full run, `Metadata_BlankImage` is also joined onto `measurements.csv`, so
each colony row names its plate's blank.

## 7. Example: ucr_033

The table `UCR-033-E-D_ImageMetadata.csv` (one row per colony) gains a
`Metadata_BlankImage` column holding each plate's frame-0 stem. In the `F1x3`
pipeline, `SubtractBlank` goes directly after `SetDetectMode` inside the
`CompositeDetector` branch, before `SubtractGaussian`. Frame-0 images are left
out of `--input` via `--image-manifest` (otherwise `PF-REF-SELF`).

## 8. Failure catalogue

| Mistake | Where it fails |
|---|---|
| No table supplied | Python: `RefMetadataUnavailableError`; CLI: `PF-REF-NO-TABLE`; GUI: Run disabled |
| Table lacks a needed column | `ReferenceContext(...)`/`lookup`: `ReferenceTableError`/`ReferenceLookupError`; CLI: `PF-REF-COLUMN` |
| Image has no row | `ReferenceLookupError`; CLI `PF-REF-UNMATCHED` |
| Rows null or disagree | `ReferenceLookupError`; CLI `PF-REF-AMBIGUOUS` |
| Image names itself | `ReferenceLookupError`; CLI `PF-REF-SELF` |
| Blank name resolves to 0 or >1 files | `ReferenceImageError`; CLI `PF-REF-UNRESOLVED` |
| Shape or bit-depth mismatch | `ReferenceImageError` at run (per image) |
| `SubtractBlank` after an enhancer or a corrector | `StaleDetectMatError` at run (inside an `ImagePipeline`, wrapped in the pipeline's `RuntimeError`, original as `__cause__`) |
| A later `SetDetectMode` would discard the subtraction | not caught at run; documented in the docstring and the how-to (§11 R2) |
| Well-formed table from the wrong experiment | not detectable; traceable via `table_sha256`, `deliverables/metadata.csv`, and the joined `Metadata_BlankImage` |

All error classes are defined in `_core/_reference_context.py` (plus
`StaleDetectMatError` in `enhance/_subtract_blank.py`) and re-exported from
`phenotypic.sdk_`.

## 9. Testing

Unit (`tests/unit/`):

- `ReferenceContext`: path/pandas/polars inputs; missing `Metadata_ImageName`;
  header canonicalization; lookup grain over a per-colony table (collapse,
  null, disagreement, missing row); dataset narrowing; `images=` precedence;
  `resolve_image` 0/1/many; LRU reuse (one read for N lookups) and
  invalidation on mtime change; `current()` is `None` outside; nesting
  replaces then restores; restore after an exception; `table_sha256`.
- `RefMetadata`: column discovery from `RefColumn`/`RefImageColumn` fields; override;
  `TypeError` on a non-`ImageOperation` subclass; unavailable-error text names
  both fixes; `_references` provenance entry.
- `SubtractBlank`: each polarity on a synthetic blank/target pair with known
  answer; detect-mode matching (`gray`, `LabL`); guard passes at root, after
  `SetDetectMode`, and inside `CompositeEnhance`/`CompositeDetector`
  branches; guard refuses after an enhancer and after an `ImageCorrector`;
  shape and bit-depth refusal; self-reference refusal; `rgb`/`gray`
  unchanged; JSON round-trip.
- `ImagePipeline.reference_columns()` on nested trees.
- CLI preflight: one test per `PF-REF-*` code and severity escalation
  (including escalation on the union of failure reasons).
- CLI run: full and process mode, local, on a tiny synthetic time series;
  two datasets with same-named blanks (proves per-dataset resolution);
  process-mode snapshot written and reused on continuation; a blank edit
  re-runs only the affected image; a staged GPU run with `SubtractBlank`
  inside the detector's branch (Stage 2), in full mode and `--layer objmap`.
- Existing gates stay honest: the zero-arg enhancer invariant gate applies
  reference ops inside a context; the enhancer taxonomy lists `SubtractBlank`.
- Tune refusal.
- GUI: registry reports the reference column source; preview with and
  without a picked table (and fingerprint change); run console Run gating.
- Startup guard: `import phenotypic` still loads nothing in
  `HEAVY_STARTUP_MODULES` (existing `tests/unit/ci/test_startup_imports.py`).
  Accessing `phenotypic.ReferenceContext` loads `phenotypic._core`, which is
  already heavy, exactly as `phenotypic.Image` does — no guard is possible
  there (plan review M5).
- Doctests for `ReferenceContext` and `SubtractBlank` run with
  `load_synth_yeast_plate()`.

No logic-validation script: the design rests on no numeric invariant beyond a
clipped difference, which the unit tests pin directly.

## 10. Documentation

- How-to: `docs/source/how_to/pages/reference_metadata.md` — the context,
  the CLI flag, the ucr_033 placement, the `SetDetectMode` caveat.
- `abc_/CLAUDE.md`: `RefMetadata` beside the `PlotImage` capability entry.
- Root `CLAUDE.md` Gotchas: one bullet — reference ops need a context; CLI
  `--metadata` is that context; process mode now snapshots it.
- `enhance/__init__.py` export and API reference entry for `SubtractBlank`.

## 11. Risks and follow-ups

- **R1 — processed blanks.** If a blank must share earlier enhancement, add an
  optional `blank_ops: ImagePipeline | None = None` to `SubtractBlank`
  (explicit, image-independent ops only). Backward-compatible; not in v1.
- **R2 — a later `SetDetectMode`.** Discards the subtraction like any
  enhancement. A future preflight warning could flag a `SetDetectMode` after a
  `RefMetadata` op in the same sequence.
- **R3 — tune support.** Enter a context in the tune evaluator.
- **R4 — RefMetadata measurers.** Relax the `__init_subclass__` restriction
  once a measurer needs it; `measure` mode would then need a context too.
