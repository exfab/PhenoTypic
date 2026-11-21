# Time-series image operations and measurements: feasibility analysis

**Status:** analysis only, no source code changed · **Date:** 2026-11-21 ·
**Branch:** `claude/focused-curie-mv5pwl` · **Companion check:**
`docs/superpowers/logic_validation_scripts/2026-11-21-time-series-operations/series_memory.py`

Paths are relative to `src/phenotypic/` unless they begin with `docs/`. A claim
followed by a `file:line` citation was read in the code on this branch. A claim
marked *(inferred)* is a conclusion drawn from code that was read, not something
the code states. Statements about effort are judgments anchored to the size of
two earlier changes in this repository (§8), not measurements of work that has
not been done.

## Summary

Letting operations and measurers work across a time series is feasible, and it
does not require rewriting the existing operations (about 135 classes across
`enhance/`, `detect/`, `refine/`, `correction/`, `grid/` and `measure/`, by a
static count). The single-image contract, `_operate(image) -> image`, can stay
as it is for every operation that only ever needs one frame. The cost of the
change sits in the CLI. It treats one image as the unit of work in its work-id,
its completion records, its failure tracker, its store paths and its SLURM
arrays, and its own module guide calls that property the reason its
verification cache is correct (`_cli/CLAUDE.md:799-806`). The design question is
therefore less "how does an operation see several images" than "how does the
CLI order, identify and invalidate work once images depend on each other".

The answer depends on what an operation needs from other frames, and §3
separates four kinds of dependency that differ sharply in cost. An operation
that needs only the raw pixels of other frames keeps per-image independence and
can be built on the existing reference channel. An operation that needs another
frame's processed result can still keep that independence if each frame
recomputes the result from the raw reference through an explicit sub-pipeline,
an extension the reference-metadata design record already anticipated as its
follow-up R1. Only quantities reduced from many frames at once, or processed
references too expensive to recompute per frame, need a per-series barrier, for
which the staged GPU engine is a working precedent. Holding a whole series in
memory is not viable at plate resolution without spatial tiling (§6.4).

Published colony time-lapse tools point the same way. Three of the four
examined establish colony positions on a late frame and carry them back to
earlier frames, and most compute time-derived quantities from per-frame
measurement curves, which PhenoTypic's analysis layer already does (§7).

The recommended path is incremental (§8). The first step makes the reference
channel series-aware, lets measurers use it, and adds derived references (§4,
Design A). The second step, only if needed, adds series artifacts computed
behind a barrier (§4, Design B). A first-class `ImageSeries` type flowing
through every operation (Design C) is not recommended.

## 1. The starting premise needs one correction

`SubtractBlank` does not remove the earliest time point, and it has no notion
of time order. Each image names its own blank in a metadata column
(`blank_column`, default `Metadata_BlankImage`), and the operation subtracts that
one named frame from the target's `detect_mat` (`enhance/_subtract_blank.py:80-90`,
`:176`). Frame 0 is the usual blank by convention of the table, not of the
code, and the table may name any file, including one outside the input
(`docs/source/how_to/pages/reference_metadata.md:29-35`).

Two properties of that mechanism matter for everything that follows. The blank
is read raw from disk and never passes through the pipeline's earlier
operations, so the target and its blank are processed asymmetrically. The
operation therefore refuses to run when the target's `detect_mat` has already
been enhanced or an `ImageCorrector` appears in its history, because the raw
blank could not match it (`enhance/_subtract_blank.py:236-270`). The design
record lists processing the blank through earlier operations as a non-goal,
because it "breaks per-image independence in the CLI"
(`docs/superpowers/specs/2026-10-05-ref-metadata-ops/design.md:39-43`). That
sentence names the central constraint of this analysis.

## 2. Where the one-image assumption lives

### 2.1 Operation and measurer contracts

`ImageOperation.apply(image, inplace=False)` hands one image, copied by default,
to `_operate`, and the provenance wrapper is installed around that same boundary
(`abc_/_image_operation.py:407-427`, `:436-510`). Integrity validation hashes the
protected layers of that one image before and after the call (`abc_/CLAUDE.md`,
"Integrity Validation"). `MeasureFeatures.measure(image)` returns one DataFrame
keyed by `Object_Label`, the labels of that image's `objmap`, and merges grid or
object metadata from the same image (`abc_/_measure_features.py:443-539`).

Changing either signature would touch every subclass, and nothing requires it.
The contracts already tolerate a side channel: `RefMetadata` reads a context
variable instead of adding an argument to `_operate` (`abc_/_ref_metadata.py:61-156`),
so cross-frame inputs can arrive without a signature change. Designs A and B
both rely on that property.

### 2.2 The pipeline

`ImagePipeline.apply` copies one image, opens one provenance application for
it, and runs `ops` in order (`_core/_pipeline_parts/_image_pipeline_core.py:976-1010`,
`:835-953`). `measure` runs every measurer on the same image and merges by
object label (`:1139`). Only the analysis stage sees more than one image. In the
CLI, `post`, `filters` and `model` run once, at finalization, over the
aggregated table (`_cli/_cli_output_manager.py:870-901`, `:1038-1237`), and that
step is the only one in a run that sees every image. It sees tables, not pixels.

### 2.3 The reference channel

`ReferenceContext` holds one metadata table and resolves names in it to image
files (`_core/_reference_context.py:280-669`). Its `lookup` returns exactly one
value per requested column for an image and raises on a missing, empty or
disagreeing value (`:413-477`). Three limits follow from the code as written.

- A reference is one value in one column, so an operation that wanted ten
  neighboring frames would need ten columns *(inferred)*.
- A reference is always an input file, read with the input reader's settings
  and hashed by its bytes, never another image's output store
  (`_cli/_cli_reference.py:362-364`).
- Loaded references pass through a process-wide cache of two images
  (`_core/_reference_context.py:107-112`), sized for the case it was built for:
  many frames of one plate against one blank.

The companion script quantifies the third limit. One worker visiting a series
in frame order reuses the cache for a window of up to three preceding frames,
and the cache thrashes from a window of four. The CLI spreads images across
independent workers, however, so the realistic cost is closer to the no-reuse
count: one read per referenced frame per target, which is quadratic in series
length for a whole-series window (1,128 reads for 48 frames).

`RefMetadata` may only be mixed into an `ImageOperation`; any other base raises
at class definition (`abc_/_ref_metadata.py:80-92`). The design record names
relaxing this for measurers as follow-up R4
(`docs/superpowers/specs/2026-10-05-ref-metadata-ops/design.md:504-516`). A
separate CLI decision refuses `--mode measure` over a pipeline whose measurers
read reference metadata, because measure mode plans no references
(`_cli/CLAUDE.md`, "Reference metadata").

### 2.4 The CLI

The CLI's unit of work is one image from discovery to publication. Discovery
produces `{dataset: [paths]}`, where a dataset is an input directory or a
first-level subdirectory, the only grouping the CLI knows
(`_cli/_cli_directory_scanner.py:83-187`). Local execution flattens that to
`(dataset, image_path)` tasks for joblib (`_cli/_cli_execution_strategies.py:382-448`),
and SLURM arrays carry one image path per array entry
(`_cli/_cli_slurm_array_scripts.py:39-65`, `:215-235`). Every identity and state
record is then keyed by one image:

| Record | One-image key | Evidence |
|---|---|---|
| Work-id | one dataset, one relative path, one input SHA-256, an optional per-image reference digest | `_cli/_cli_failure_tracker.py:309-344` |
| Completion record | `.phenotypic/progress/images/<ds>/<stem>.json` | `_cli/_cli_completion.py:172-188`, `sdk_/_io_constants.py:2402-2403` |
| Terminal failure | one `work_id`, one `relative_image_path` | `_cli/_cli_failure_tracker.py:501-514` |
| Image store | `results/<ds>/zarr/<stem>.ome.zarr` | `sdk_/_io_constants.py:1874-1882` |
| Stage-2 signal | `stage2_raw/<ds>/<slot>/<stem>.npy` and a token | `_cli/_cli_stage2_token.py:162-182` |

The reference channel already runs through this machinery, which is the
strongest argument for building on it. At startup the CLI plans every image's
references from the table and directory listings, hashes each referenced file,
and writes `.phenotypic/reference_manifest.json` (`_cli/_cli_reference.py:272-321`,
`:437-475`). Each image's reference digest covers every image column its
operations declare (`:193-225`) and enters that image's work-id only when the
pipeline reads references (`_cli/_cli_failure_tracker.py:339-343`). A changed
blank changes the digest and hence the work-id, so continuation re-runs exactly
the frames that named it (`_cli/_cli_state_management.py:603-614`).

The staged GPU engine is the one precedent for ordered work. It runs Stage 1 for
every image, then Stage 2, then Stage 3, and hands state between stages through
on-disk artifacts (`_cli/_cli_staged_strategy.py:242-393`). On SLURM a
controller advances through the same phases and reclassifies progress from
those artifacts at each transition (`_cli/_cli_staged_controller.py:59-118`,
`:256-289`). Its barrier is run-wide; nothing in the CLI waits on a per-plate or
per-series condition (`_cli/_cli_staged_slurm.py:352-359`).

Three CLI rules constrain any design that adds cross-frame work. The engine
holds one identity per image for the whole run, not one per stage, because a
later stage certifies an earlier stage's pixels (`_cli/CLAUDE.md`, "Reference
metadata"). Each artifact has one writer per pass, which is what keeps the
verification cache sound (`_cli/CLAUDE.md:799-806`). And the run tracks four
states, every other answer being derived from them and the artifacts on disk,
so "a fifth tracked value appearing is a design regression" (`_cli/CLAUDE.md`,
"Tracked state"; full reference in `docs/source/contrib_guide/tracked_state.md`).

### 2.5 Storage, metadata and provenance

The image model is two-dimensional throughout. An `Image` accepts a 2-D array
or an `(H, W, c)` array with one, three or four channels, and refuses four or
more dimensions as an unknown format (`_core/_image_parts/_image_data_manager.py:599-630`).
`objmap` is a 2-D sparse matrix of uint16 labels (`:58`, `:585-587`). There is no
reader for multi-frame TIFFs; flat files go through scikit-image, which returns
the first TIFF series (`_cli/_cli_input_headers.py:137-143`). On an OME-Zarr
store, `Image.imread` squeezes a time axis of size one and otherwise requires
`t=` to select one frame (`sdk_/ngff_.py:779-792`). The store writer can only
emit `(c, y, x)` or `(y, x)` axes (`sdk_/ngff_.py:127-137`), and `load_zarr` has
no time argument. Design C would need a new store revision for these reasons;
Designs A and B leave the per-image store untouched.

Capture time does not reach the measurement tables. `imread` records EXIF tags
such as `DateTimeOriginal` in the image's `imported` metadata
(`_core/_image_parts/_image_io_handler.py:451-463`), which the `image.metadata`
accessor does not search and which `insert_metadata` does not write into tables
(`_core/_image_parts/accessors/_metadata_accessor.py:69-78`, `:364-375`). Series
order must therefore come from user metadata (`Metadata_FrameIndex`,
`Metadata_Time` or `Metadata_ImageDatetime`), or from a new step that promotes
the EXIF time.

The provenance journal lives on the image, and each application records a
single `input_filename` (`_core/_provenance.py:338-347`). A second image appears
in provenance only as a digest inside one operation's parameters: `RefMetadata`
stores the table digest, the values it resolved and each reference image's
digest under `_references` (`abc_/_ref_metadata.py:123-156`). That record can
grow to a list of frames without a schema change to the journal itself
*(inferred)*.

Two facts about colony identity bear on cross-frame measurement. Grid edges are
not stored: `grid.info()` re-runs the grid finder on every call, and the store
keeps only `nrows`, `ncols` and the finder's parameters
(`_core/_image_parts/accessors/_grid_accessor.py:165`,
`_core/_image_parts/_grid_image_handler.py:481-511`). Grid-cell identity across
frames is therefore only as stable as the finder's result on each frame. Object
labels come from connected-component relabelling of each image's mask
(`_core/_image_parts/accessors/_objmask_accessor.py:203`,
`_core/_image_parts/accessors/_objmap_accessor.py:425-471`), so `Object_Label` is not stable across
frames *(inferred)*. There is no registration or tracking code in `src/`. The
one piece of cross-frame state in the package is `CalibrateColorRpcc`'s
`lattice_prior`, "per-ROI lattices from an earlier frame" supplied as an
operation parameter, and its docstring explains why ECC registration is not
offered: "it needs a reference band shipped with every serialised pipeline"
(`correction/_color_correction/_calibrate_color_rpcc.py:135-139`). That
precedent shows why cross-frame state belongs in a run-time context rather than
in operation parameters.

### 2.6 GUI, tuning and plotting

Every GUI surface that touches pixels takes one image per call. The builder
stores a single image path (`_gui/builder/_ids.py:730`), loads one `Image` or
`GridImage` for preview (`_gui/builder/_callbacks.py:8167-8195`), and keys the
node-preview cache on that image's identity (`_gui/builder/_preview_cache.py:196-200`,
`:386-412`). Its reference-metadata support narrows the table to the preview
image's directory and folds the blanks that one image resolves to into the
preview fingerprint (`_gui/builder/_reference_metadata.py:184-221`, `:289-331`).

`phenotypic-tune` refuses any pipeline that reads reference metadata, because
its evaluator does not enter a `ReferenceContext` (`tune/_spec.py:76-90`), and it
scores images one at a time, sorted by id rather than by time
(`tune/_evaluation/_evaluator.py:306`, `:364-404`). Among plotting lifecycles,
`PlotImage` binds one image weakly and is drawn once per image
(`plotting/_pipeline/_store_figures.py:187-202`), while `PlotMeas` and
`PlotAnalysis` see the whole run's measurement mirror
(`plotting/_pipeline/_coordinator.py:117-140`). No lifecycle receives several
images.

### 2.7 What already works across time: the measurement table

Much of what "time-series analysis" usually means is already supported at the
level of tables. The schema defines `Metadata_Time`, `Metadata_Timepoint`,
`Metadata_FrameIndex` and `Metadata_PlateID`
(`schema/_metadata/_experimental_tags/_culture.py:22-39`,
`schema/_metadata/_experimental_tags/_plate.py:21`). The growth models
(`LogGrowthModel`, `LinearLagModel`, `LinearCapAndLagModel`) aggregate by
`groupby + [time_label]` and fit one curve per group, with `time_label`
defaulting to `Metadata_Time` (`analysis/abc_/_model_fitter.py:24`, `:77`,
`:359-389`). The fitting-window QC check uses the same axis
(`analysis/qc/_fitting_window.py:294-328`). `AddFrameIndex`, a post-measurement
operation, ranks each plate's images over the aggregated table and describes
itself as "a cross-image, whole-table operation" (`post/_add_frame_index.py:19-45`).

In these tools the identity of "the same colony across frames" is whatever
`groupby` columns the user supplies, typically plate plus grid position. A
per-colony quantity that is a function of per-frame scalar measurements, such as
lag time, maximum growth rate or carrying capacity of colony area, needs no
image-level change. The rest of this report concerns quantities that per-frame
scalars cannot supply.

Two implementations of "rank a plate's frames" already exist and disagree on
their base. `AddFrameIndex` is 1-based, matching the schema's documented
`FrameIndex` semantics (`post/_add_frame_index.py:22-27`). The results viewer's
`derive_frame_index` is zero-based and ranks `Metadata_ImageDatetime` within
`Metadata_PlateID` (`_gui/results_viewer/_scatter_tab/_facets.py:104-145`). The
difference may be deliberate, since the viewer's column is a separate
`Computed_FrameIndex`, but a series feature would need one ordering rule shared
by the CLI planner, the post operation and the GUI.

## 3. Four kinds of cross-frame dependency

What an operation needs from other frames determines what it costs. The table
orders the four kinds by how much of the one-image machinery they disturb.

| Kind | What frame *t* needs | Examples |
|---|---|---|
| **1. Raw side inputs** | raw pixels of named or addressed frames | blank subtraction (exists); difference from the previous frame; median of earlier raw frames as a background; change mask since frame 0 |
| **2. Processed side inputs** | another frame's pipeline result | segmentation of the last frame applied to earlier frames; one grid fitted per plate; a blank passed through the same enhancers (design record R1) |
| **3. Series reductions** | one quantity computed from all frames, or a wide window | per-pixel growth-onset map; registration transforms to a common frame; temporal denoising; a per-series background model |
| **4. Cross-frame measurement** | pixel correspondence for one colony across frames | radial expansion between frames; area gained since the previous frame; time of first appearance; per-zone kinetics in filamentous colonies |

The first kind keeps per-image independence outright, because its inputs are
files that the CLI hashes at startup. The second kind breaks it in its natural
form, since frame *t* would wait for frame *u*'s result, but it can be restored
by recomputing that result inside frame *t*'s own task, at the price of repeated
work. The third kind breaks it in a way recomputation cannot repair cheaply,
because every frame depends on all the others and the repeated reads grow
quadratically. The fourth kind inherits the cost of whichever inputs it uses,
and adds a requirement of its own: its rows describe a colony across frames, so
they need a cross-frame key that `Object_Label` cannot supply.

## 4. Three designs

### Design A: series-aware references

Design A extends the reference channel from "one named frame" to "addressed
frames in the same series, raw or derived". It has three parts, and none of
them changes the CLI's unit of work.

**A.1 Frames by address.** `ReferenceContext` would gain a series query: group
the table's rows by series columns (for example `Metadata_Dataset` plus
`Metadata_PlateID`), order each group by one shared rule (for example
`Metadata_FrameIndex`, then `Metadata_Time`), and resolve a window relative to
the target, such as "first", "previous", "last" or "all earlier frames". A
`RefMetadata` operation would declare its window as an ordinary field, as it
declares columns today. The startup planner already reads the table and the
directory listings without opening images, so it can resolve each image's
window to files, hash them once, and fold the set into the existing per-image
reference digest. Work-ids, continuation, preflight and the SLURM manifest then
extend without a new stage.

**A.2 Derived references.** A reference may carry an explicit sub-pipeline that
is applied to the raw reference inside the target's own task. This is the
`blank_ops` extension the design record names as R1
(`docs/superpowers/specs/2026-10-05-ref-metadata-ops/design.md:504-509`),
generalized. The following sketch, which is not tested code, shows the
published late-frame pattern of §7 expressed this way:

```python
class RestrictToReferenceMask(ObjectRefiner, RefMetadata):
    series_by: RefColumnList = ["Metadata_PlateID"]
    order_by: RefColumn = "Metadata_FrameIndex"
    reference: FrameAddress = "last"
    reference_ops: ImagePipeline  # applied to the raw reference frame, not to the target

    def _operate(self, image):
        reference = self._ref_series(image, self.reference, ops=self.reference_ops)
        ...  # keep only target objects inside the reference frame's footprints
```

Per-image independence survives because the target depends only on input files
and on the sub-pipeline's configuration, both of which the work-id can hash. The
price is repeated work: each of a plate's *N* frames recomputes the reference's
result, so the reference is processed *N* times instead of once. For a CPU
detector on one frame that cost is linear and often acceptable; for a GPU
detector it usually is not. The sub-pipeline must also be deterministic and must
not itself read references, or the recomputations could disagree.

**A.3 Measurers.** Relaxing the `ImageOperation`-only restriction (follow-up R4)
lets a measurer read raw or derived references, which covers kind 4 when its
inputs are of kind 1 or 2. Such a pipeline would remain refused by
`--mode measure` until measure mode learns to plan references (§2.3).

Design A has two real costs. Reads grow with window width, and a whole-series
window read per target is quadratic in series length (§2.3). Derived references
repeat computation, as described above. Kind 3 stays effectively out of reach.

### Design B: series artifacts behind a per-series barrier

Design B handles kind 3, and kind 2 when recomputation is too expensive, by
computing each series-level quantity once and letting every frame read the
result. A new operation type, a series reducer, declares where in the pipeline
it reads frames and what it produces. The CLI splits the pipeline at that point,
as it already splits at a GPU detector's top-level ancestor, and runs three
phases. The first phase runs the pipeline's prefix on every frame and writes
each frame's store, exactly as Stage 1 does today. The second runs one task per
series, in which the reducer reads that series' frames from their stores, tile
by tile where it can, and writes one artifact plus a token under
`.phenotypic/progress/`. The third runs the rest of the pipeline per frame; an
operation that consumes the artifact reads it through a context, as
`SubtractBlank` reads its blank, and measurement proceeds per frame.

The CLI's rules shape the details. Continuation should be content-defined, as
it is for Stage-2 signals: a series is complete when its artifact and token are
valid, so no fifth tracked state is needed. Each artifact has exactly one
writer, the reducer task for that series. The identity of each frame should be
computed once at startup, which is possible because the artifact's identity can
be defined from inputs rather than outputs: a digest of the reducer's
fingerprint and the ordered input digests of the series' frames, folded into
every member frame's work-id the way the reference digest is today
*(inferred)*. A changed frame then re-runs every frame of its series and no
other series. That is more than strictly necessary, since the first phase of an
unchanged frame did not depend on its siblings, but the CLI already prefers
invalidating too much to too little (`_cli/CLAUDE.md`, process-layer revision).

In a notebook the same hooks would run eagerly, in the way `GpuDetector` drives
one set of hooks from both its single-image path and the batched CLI engine
(`abc_/_gpu_detector.py:60-71`). The staged engine shows that most of the
machinery exists in some form: an on-disk handoff between phases, a controller
that recovers from artifacts, chunking against `MaxArraySize`, and a replay
operation substituted at a tree path. What is new is a barrier per series rather
than per run, a store location for series artifacts, and a reducer task kind in
both the local strategy and the SLURM controller. A run-wide barrier, as the
staged engine uses, would be an acceptable first version, at the cost of making
early plates wait for the slowest.

### Design C: a first-class `ImageSeries` through the pipeline

Design C makes the series a value: `ImagePipeline.apply(series)`, operations
broadcast over frames by default, and series-aware ABCs override that. Stores
would carry a real time axis. This is the most expressive design and the only
one suited to joint spatiotemporal segmentation, but it is also the most
expensive by a wide margin. It changes what every operation, validator,
provenance record, store reader and GUI surface receives, and the memory
arithmetic in §6.4 rules out its naive form. It is not recommended now.

### Comparison

| | A.1 raw frames | A.2 derived references | B series artifacts | C `ImageSeries` |
|---|---|---|---|---|
| Kinds covered | 1, and 4 with raw inputs | 2, and 4 with processed inputs | 2 and 3 | all, and joint segmentation |
| Existing operations changed | none | none | none | all, at least by broadcasting |
| New CLI unit of work | none | none | a per-series phase | series throughout |
| Redundant work | reads grow with window | reference processed once per frame | none | none |
| Memory per task | the window of raw frames | one extra processed frame | one tile across a series | a whole series unless tiled |
| Invalidation | per frame, by reference digest | per frame, by reference digest | per series, by artifact digest | per series |

## 5. What would change under the recommended path

| Subsystem | Design A | Design B adds |
|---|---|---|
| `Image` / `GridImage` | nothing | nothing |
| ABCs | `RefMetadata` allowed on measurers; series-window and frame-address field types beside `RefColumn`; a sub-pipeline field for derived references | a series-reducer ABC and a consumer mixin, both pydantic and serializable |
| `ImagePipeline` | `reference_columns()` reports windows and sub-pipelines | tree-wide discovery of reducers, as `find_gpu_detectors` does for detectors |
| `ReferenceContext` | series grouping, one shared ordering rule, window resolution, derived-reference application, a cache keyed by series | an artifact context for the third phase |
| Provenance | `_references` records a list of frames, their digests and any sub-pipeline fingerprint | the artifact digest and the frames it was built from |
| Schema | possibly a `SERIES` identity enum | a cross-frame object key; measurement enums for series quantities |
| CLI planner and preflight | windows resolved and hashed at startup; new `PF-REF-*` codes for short, gapped or unordered series | series discovery and a refusal for images that cannot be grouped |
| Work-id and continuation | unchanged in shape | the artifact digest folded into member work-ids; a content-defined series token |
| Local and SLURM execution | unchanged | a reducer phase in the local strategy and the SLURM controller; chunking counts series |
| Finalization | unchanged if measurers emit per-frame rows | unchanged in a first version if series values ride on per-frame rows (§6.1) |
| GUI builder | the new fields render; preview loads the window | a frame or series picker for preview; previewing a reducer means running it |
| Results viewer | nothing | the cross-frame key, to plot colony trajectories |
| Tune | still refused (follow-up R3) | still refused |
| Docs and ledgers | how-to page; `FEATURES.md` and `WORKFLOWS.md` if GUI chrome changes | a contributor guide for reducers; `tracked_state.md` if any record is added |

## 6. Hard problems

### 6.1 Colony identity across frames

Kind 4 measurements need a key that survives across frames, and the right key
depends on the plate layout. On an arrayed plate a grid cell is the natural key,
but only if every frame of a plate receives the same grid. Because each frame
fits its own grid today (§2.5), a per-plate grid is itself a kind 2 dependency.
On a spread or filamentous plate there is no grid, and frame-to-frame tracking
is the general answer. Colonies on agar do not move, however; they grow in
place and sometimes merge. Transferring labels from one reference segmentation,
usually the last frame where colonies are largest, reduces tracking to an
overlap lookup, which is the approach three of the four published tools take
(§7).

Where series-level values live in the output also needs a decision. Writing
them onto every per-frame row keeps finalization unchanged, because rows stay
per image, at the cost of redundancy. A separate per-series table is cleaner,
but it needs a new store location, a new master and a new mirror, and so touches
the finalization path that full, measure, recompile and migrate all share
(`_cli/_cli_output_manager.py:1038-1237`).

### 6.2 Registration

The reference design assumes pixel-aligned frames and refuses a shape mismatch,
listing registration as a non-goal
(`docs/superpowers/specs/2026-10-05-ref-metadata-ops/design.md:44-45`). Every
kind beyond the first is more sensitive to misalignment than blank subtraction,
because it compares colony boundaries rather than background. A plate lifted
from the scanner between frames would need a transform per frame, which is a
kind 3 dependency. Two of the four published tools register their images (§7),
which suggests that a time-series feature without registration will meet this
limit in practice.

### 6.3 Invalidation, and why causality matters

The CLI's continuation procedure is to run the same command again when new
compatible inputs appear (root `CLAUDE.md`, CLI section). A time-lapse that is
still being acquired is exactly that case. A dependency on the whole series,
such as the last frame's segmentation, changes whenever a new frame arrives, so
every frame of that plate re-runs on every invocation. A dependency only on
frames up to *t*, such as frame 0, the previous frame or a running maximum,
leaves earlier frames' identities unchanged. Each window and each reducer should
therefore say whether it is causal, and the documentation should state what an
acausal one costs during live acquisition.

### 6.4 Memory and input/output

One in-memory frame costs 11 bytes per pixel with 8-bit RGB and 14 with 16-bit,
counting only `rgb`, `gray` and `detect_mat`
(`_core/_image_parts/_image_data_manager.py:575-597`; `objmap` is sparse). A CLI
worker also keeps a copy of the decoded RGB unless `--drop-originals` is given
(`_cli/_cli_process_single.py:322-326`), which raises the figures to 14 and 20.
Under illustrative assumptions of 24 megapixels and 48 frames, one 8-bit frame
is about 264 MB and a resident series about 12.7 GB, before the copy that
`apply` makes by default. One 1024 by 1024 float32 tile read from all 48 frames
is about 201 MB. The companion script recomputes these figures from the layer
sizes, and it should be re-run with a real experiment's dimensions. Design B's
reducers should therefore read tiles across time rather than whole frames,
which per-frame OME-Zarr stores already allow, and Design C is not viable in its
naive form.

### 6.5 Failure semantics

Today one bad image fails alone. Under Design B a frame that fails before the
barrier leaves its series incomplete, and the reducer must refuse the series,
proceed with a gap, or treat the frame as missing data. Under Design A a
missing or unreadable reference fails the frames that need it, as a missing
blank does now. Either way the choice must be explicit in the operation's
contract, because a silent gap produces a plausible but wrong result.

## 7. Prior art

Four published tools for colony time-lapse imaging were examined for how they
relate the frames of one plate. Bibliographic records were verified through
PubMed. Method details are quoted from the PMC full texts and, for ScanLag, from
the group's protocol paper. Supplementary materials were not retrievable from
this session, so statements of the form "not stated" mean not stated in the
main text.

- **ColTapp** (Bär et al., 2020) detects colonies once, on a late reference
  frame, in its time-lapse mode: "ColTapp proposes to initially detect all
  colonies on a frame corresponding to a late timepoint, which is set as the
  reference frame." It then tracks each colony's radius on earlier frames from
  sub-images at fixed centers, avoiding per-frame detection because "colony to
  colony center matching between frames needs high accuracy". Radius over time
  comes from a kymograph built across frames, and appearance time and growth
  rate come from a regression on that curve. Drift and center correction are
  optional.
- **Scan-o-matic** (Zackrisson et al., 2016) registers images using orientation
  markers, calibrates intensity on every image, and analyzes them in reverse
  chronological order "to utilize information about colony position from easier
  to analyze later images". Colony identity is a fixed grid. Growth phenotypes
  are extracted from per-frame population-size curves.
- **Pyphe** (Kamrad et al., 2020) states that "in timecourse mode, colony
  positions are determined in the last image and the mask is applied to all
  previous images" (Results, analysis pipelines section). Growth parameters come
  from the resulting curves. Two web searches did not independently corroborate
  this sentence; it rests on the PMC full text.
- **ScanLag** (Levin-Reisman et al., 2010; protocol, Levin-Reisman et al., 2014)
  aligns scanner images in preprocessing, detects and tracks colonies on each
  plate, and defines a colony's appearance time as the time it is first
  detected. The protocol does not describe its tracking or background method.

The tools agree on three points that bear on this design. The dominant way to
establish colony identity is a late-frame segmentation carried back to earlier
frames, which is a kind 2 dependency and the reason Design A includes derived
references. Time-derived quantities are mostly computed after the fact from
per-frame measurement curves, as PhenoTypic's growth models already do. And
registration is part of two of the four pipelines. None of the four main texts
describes subtracting a first frame, which may mean `SubtractBlank` addresses a
less common need than mask propagation; absence from the main text is not
evidence that the tools do not do it.

## 8. Recommendation, sequencing and effort

Build Design A first, beginning with A.1 and A.3 and then A.2. Treat Design B
as a later step whose need is demonstrated by a concrete kind 3 analysis or by
a kind 2 reference too expensive to recompute, such as a GPU segmentation of the
last frame. Decide on Design C only if joint spatiotemporal segmentation becomes
a goal. Design A is useful on its own, covers the late-frame pattern of §7, and
builds what Design B needs anyway: one ordering rule, series grouping in the
planner, and measurers that can read references.

Two earlier changes give a sense of scale. The reference-metadata change, which
added `ReferenceContext`, `RefMetadata` and `SubtractBlank` across the Python
API, the CLI and the GUI, changed about 4,200 lines in 63 files under `src/` and
added about 7,200 lines of tests (merge `2969a392`). The nested GPU staging
change, which generalized the staged engine to detectors anywhere in the
operation tree, changed about 2,100 lines under `src/` and added about 5,500
lines of tests (merge `7973d960`). In the author's judgment Design A is smaller
than the reference-metadata change, because it extends that infrastructure
rather than creating it. Design B is comparable to the two changes combined,
because it adds a new task kind, a new artifact store and a new identity
composition. Design C would be considerably larger than both.

## 9. Open question for the owner

Which analyses are the target decides how far down this path to go. If they
compare raw frames, or need a processed frame that is cheap to recompute (a CPU
segmentation of the last frame, say), Design A is sufficient. If any of them
reduces a whole series to one quantity, or needs a GPU segmentation shared by
every frame, Design B is required.

## References

- Bär, J., Boumasmoud, M., Kouyos, R. D., Zinkernagel, A. S., & Vulin, C. (2020).
  Efficient microbial colony growth dynamics quantification with ColTapp, an
  automated image analysis application. *Scientific Reports*, 10, 16084.
  https://doi.org/10.1038/s41598-020-72979-4
- Kamrad, S., Rodríguez-López, M., Cotobal, C., Correia-Melo, C., Ralser, M., &
  Bähler, J. (2020). Pyphe, a python toolbox for assessing microbial growth and
  cell viability in high-throughput colony screens. *eLife*, 9, e55160.
  https://doi.org/10.7554/eLife.55160
- Levin-Reisman, I., Fridman, O., & Balaban, N. Q. (2014). ScanLag:
  high-throughput quantification of colony growth and lag time. *Journal of
  Visualized Experiments*, (89), e51456. https://doi.org/10.3791/51456
- Levin-Reisman, I., Gefen, O., Fridman, O., Ronin, I., Shwa, D., Sheftel, H., &
  Balaban, N. Q. (2010). Automated imaging with ScanLag reveals previously
  undetectable bacterial growth phenotypes. *Nature Methods*, 7(9), 737-739.
  https://doi.org/10.1038/nmeth.1485
- Zackrisson, M., Hallin, J., Ottosson, L.-G., Dahl, P., Fernandez-Parada, E.,
  Ländström, E., Fernandez-Ricaud, L., Kaferle, P., Skyman, A., Stenberg, S.,
  Omholt, S., Petrovič, U., Warringer, J., & Blomberg, A. (2016). Scan-o-matic:
  high-resolution microbial phenomics at a massive scale. *G3*, 6(9), 3003-3014.
  https://doi.org/10.1534/g3.116.032342
