# Streamlit Run Monitor — Design

**Date:** 2026-08-13
**Status:** Revision 11. Seven rounds of independent design review and data-flow
review, the data-flow rounds validated against real end-to-end CLI runs. Round 7
ran against a frozen, checksummed copy — earlier rounds had reviewed a document
being edited underneath them, which makes a readiness verdict close to
meaningless. It returned **no blocking finding on either track**: the data flow
was judged sound, and the design findings were two textual contradictions, one
overclaim, and three small mechanism clarifications, all corrected here.

**Three inferences, the same mistake three times.** The design spent four
revisions deciding when to read `deliverables/measurements.parquet`, three more
deciding how to merge `_dataset_aggregated.parquet`, and four more deciding which
metadata CSV belongs to a run. Every one ended the same way: stop inferring.
Sections 3.1.1, 3.3 and 3.2 are one lesson written three times — and in each case
it only became visible after several plausible rules had failed *differently*,
which is the actual signal. One failed rule invites a fix; three failed rules
mean the question is wrong.

The third instance is the sharpest, because deleting it also deleted an upstream
CLI change that existed only to feed it. Inference is not merely risky here; it
was pulling scope along behind it.

Two further habits this document had to unlearn, both recorded in place rather
than quietly corrected. Reassurance without verification ("the divergence
direction is safe" — it was backwards, and nobody re-checks a sentence that
sounds settled). And fixes written in the same pass creating each other's edge
cases: the accumulator's removal path exists only because of a quarantine
documented in the same revision, and the two-poll confirmation added to guard it
could never fire, because it was placed inside a cache the same section had just
made immune to re-entry.

**The single most important thing this document learned:** it tried four times
to auto-detect whether a run was still executing — mirror-existence, then a
completion marker, then a composite of both, each fix introducing the mirror
image of the bug it closed. **The first three were reproduced as broken against
real runs; the fourth was rejected on complexity grounds without being tested**
(the distinction matters, and 3.1.1's table keeps it). The design now **does not
detect run state at all**. It reads the
per-image measurements, always, and never reads the finalized mirror. That
deletes the predicate, its four failure modes, an upstream change, and roughly a
section and a half of contingency handling. Where a decision here looks
surprisingly plain, this is usually why.
**Topic:** A standalone, read-only Streamlit dashboard for watching an
in-progress `python -m phenotypic` run: grouped colony time-series plots with
click-through to full-resolution image crops pulled from the per-image HDFs.

---

## 1. Purpose and scope

A collaborator opens a URL, points at a run folder, picks a strain, a grouping,
a time column and a measurement, and watches colony metrics accumulate while the
run is still executing. Clicking a point on any plot shows the actual pixels
behind that measurement, in one of three framings.

The existing Dash results viewer is a curation and QC tool over *finished*
output. This is a different job: read-only observation of *live* output, on a
server, for people who do not have the repo checked out. The Dash viewer is not
modified by this work beyond the extractions in Section 4.

**In scope**

- Run-folder browsing under a containment-checked scope root.
- Line / scatter / box views of one measurement over one time column, faceted by
  strain and grouping metadata, never aggregated across replicates.
- Metadata value filtering.
- Click-to-inspect with three image modes: plate, colony, grid section.
- A Docker image runnable with a single read-only bind mount.

**Out of scope**

- Authentication and per-user permissions. Only the *seam* is built
  (Section 9); no identity, no login, no authorization checks.
- Any write to the run folder: no curation, no QC labels, no error categories.
- Deep-zoom / DZI tiling.
- Comparison across multiple runs in one view.
- Editing or launching runs.
- Fixing the `MetadatasCondition` schema typo (Section 3.2) — tracked
  separately, because it renames a public column in written output.

---

## 2. Decisions

| Decision | Resolution |
|---|---|
| Why a second app | Lightweight shareable read-only monitor, server-deployed. Dash viewer keeps its role. |
| Placement | `apps/monitor/` — separate top-level app dir, its own `pyproject.toml`, a uv workspace member depending on `phenotypic` plus Streamlit. |
| Image weight | Accepted as large. See 5.1 — depending on `phenotypic` at all pulls Dash and the full scientific stack; avoiding `phenotypic.gui` buys architecture, not megabytes. |
| Run handle | `phenotypic.sdk_` path helpers directly. **Not** `OutputRoot` — see 3.5. |
| Data source | Per-image measurements, always. The finalized mirror is never read, and run state is never detected — see 3.1.1. |
| Controls | Strain (page) · groupby columns (subplot columns) · time column from metadata (x) · measurement (y) · metadata value filters. |
| Views | Line, scatter, box. Radio toggle, plus an overlay option composing box behind the line/marker traces, and an off-by-default toggle admitting unmatched colonies as a pinned `unassigned` page (3.2, 6.2). |
| Scatter | Every colony as a point over time, no connecting lines. |
| Box | One box per timepoint, within each facet. |
| Grid view | Crop window = **union of the member bounding boxes** of the colony's grid section. See 7.1 for what this does and does not give. |
| Click identity | `point_id` on `PlotMeasTimeSeries` carrying `[dataset, image name, object label]`. |
| Strain cardinality | All strains by default, rendered in progressive chunks (6.2). |
| Run access | Mounted root plus free path entry, containment-checked against a swappable scope root. |
| Storage | GCS-FUSE bucket mount (Section 11.1). |

---

## 3. Data flow — what exists on disk, and when

This section is the load-bearing one. The app targets **active** runs, and the
artifacts a finished-run tool reaches for are not present during one. Every
claim below was checked against the writers; the timing rows were additionally
confirmed by observing mtimes on a real run.

### 3.1 What the CLI writes, and when

| Artifact | Path | Written |
|---|---|---|
| Per-image measurements | `results/<ds>/measurements/<stem>.parquet` | **During** the run, one per image as it completes, **atomically** |
| Per-image HDF | `results/<ds>/hdf/<stem>.h5` | **During** the run, atomically (`.part` + replace) |
| Overlay PNG | `deliverables/overlays/<ds>/<stem>.png` | **During** the run, per image, when `--save-overlays` (default on). **Not atomic.** |
| SLURM chunk aggregate | `results/<ds>/measurements/_dataset_aggregated.parquet` | **During** SLURM runs, at every checkpoint |
| SLURM chunk files | `.phenotypic/progress/chunks/chunk_*.parquet` | **During** SLURM runs, at every checkpoint |
| Master archive | `deliverables/master_measurements.parquet` | **Local:** at finalize. **SLURM:** also mid-run, at every checkpoint |
| Post-applied mirror | `deliverables/measurements.parquet` | At finalize only, by `_seed_measurements` |
| Metadata CSV copy | `metadata_csv_deliverable_path(output)` | At finalize, and on every recompile. Preserved by `--restart`, so it may belong to a previous run — which is why 3.2 offers it as a suggestion rather than adopting it |

Measurements, HDF and overlay are written together per image at
`_cli_process_single.py:109-112`, three seconds ahead of finalize on an observed
run. Overlays are gated at `_cli_process_single.py:111` on the
`--save-overlays` flag (declared at `:265-267`, default on); they are never
rewritten at finalize, so their absence is a *flag* property, not a timing one.

**On a staged-GPU run, measurements do not appear until Stage 3 — and that is
most of the run.** The table's "during the run, one per image as it completes"
describes the single-pass path. Staged runs write the HDF in Stage 1, an `.npy`
objmap sidecar in Stage 2, and `<stem>.parquet` only in Stage 3
(`_cli_staged_slurm_worker.py:237-239`). The stages are hard barriers: the
controller advances `stage1 → stage2 → stage3 → finalizing` and submits Stage-3
chunks only after Stage 2 completes for the whole run
(`_cli_staged_controller.py:246-274`), and the local strategy is serial in the
same way.

Since 3.1.1 names staged-GPU SLURM as the target deployment, **the expected,
correct behaviour for roughly the first two thirds of a target-deployment run is
an empty page**, then a rapid fill. This is stated here because the first person
to watch one will otherwise conclude the app is broken; §12's "no measurements
yet, keep polling" is the right behaviour but reads as an edge case, and on this
deployment it is the norm. Surfacing stage progress instead of an empty page is
OQ-10 — worth doing, but it means reading orchestration state, which is a
different data source than this design otherwise touches.

Two further consequences that a simpler reading would miss:

- **The master is not a finalization signal.** `_aggregate_chunks_locked`
  (`_cli_chunk_writer.py:153-160`) writes `master_measurements.{csv,parquet}` on
  every SLURM checkpoint. A mid-run SLURM output directory contains a master.
- **The mirror is not a finalization signal either.** `_seed_measurements`
  (`_cli_output_manager.py:744`) does have exactly one call site inside
  `finalize_post_master_outputs` — but that proves only that *a* finalize wrote
  it, not that it belongs to the run now executing. `--restart`, `--resume` and
  `--mode measure` all skip the existing-output guard
  (`phenotypicCLI.py:1392-1394`) and preserve `results/`, `deliverables/` and
  `qc/`; only `--overwrite` removes the directory (`:1406`). Note that of these,
  **only `--restart` clears `.phenotypic/`** — that block is inside `if restart:`
  at `:1367`, a distinction an earlier revision got wrong and paid for.

  This is what 3.1.1 is about, and why the monitor reads neither file.

### 3.1.1 There is no live/finalized discriminator, by decision

The obvious design reads per-image parquets while a run executes and switches to
`deliverables/measurements.parquet` once it finishes. Detecting *once it
finishes* turned out to be the hardest problem in this document, and four
successive predicates were each reproduced as wrong:

| Predicate | How it failed |
|---|---|
| Mirror exists | `--restart` and `--mode measure` preserve `deliverables/` while rewriting per-image parquets, so a mirror from the **previous** run — with its previous *schema* — is served for the entire live run. Reproduced. |
| `run_completion.json` exists | Written only by GUI-launched runs (`_cli_gui_lifecycle.py:89-91`, gated on a GUI-only env var) and non-staged SLURM finalize (`_cli_checkpoint_handler.py:324-325`). A plain `uv run python -m phenotypic` run writes it never; staged-GPU runs write a *different* file (`_cli_staged_orchestration.py:674-699`). The repo's own `test_non_gui_local_cli_does_not_publish_marker` asserts the absence. Every finished local run would read as live forever. |
| Composite of both | `--mode measure` does **not** clear `.phenotypic/` — that block is inside `if restart:` (`phenotypicCLI.py:1367`), while `measure_only` only skips the output guard (`:1393`). So a stale marker survives a live `--mode measure` run and the first failure returns through a different door. Reproduced. |
| Marker generation comparison | Would work, but requires the monitor to know the currently-executing run's generation — reconstructing what `_runs_registry.py:631-660` does, for a predicate already wrong three times. |

**So the monitor does not detect run state.** It always reads the per-image
measurements, and never reads the mirror. Per-image parquets survive finalize on
every ordinary path, so this reads correctly for a finished run as well as a
running one — the same code, the same numbers, before and after.

**One exception, and it is on the target deployment.** Staged-GPU **SLURM**
finalize moves per-image parquets *out* of `results/<ds>/measurements/`, twice:

- `reconcile_stage3_publications` (`_cli_staged_resume.py:275-304`,
  `parquet.replace(destination)` at `:303`) relocates every parquet lacking a
  Stage-3 completion marker into `.phenotypic/progress/unpublished_stage3/`.
  Called from `_cli_checkpoint_handler.py:253-261`, gated on
  `stage3_markers_required`, which derives from `staged_stage3_markers: bool =
  True` (`_cli_types.py:140`) — **on by default**.
- `quarantine_unchanged_restart_parquets` (`_cli_staged_orchestration.py:306-334`)
  relocates parquets unchanged since a `--restart` snapshot.

Both run before aggregation (`_cli_checkpoint_handler.py:248,253`; aggregation at `:267`). Local staged
runs are unaffected — `_cli_staged_strategy.py` passes no epoch.

So on that one path, points a viewer was watching **do** disappear at finalize,
and the "values never change while you watch" property above is qualified rather
than absolute. The banner reports the drop instead of letting points silently
vanish, and 3.3's decision never to read the aggregate prevents the quarantined
rows from being resurrected from it.

**An earlier revision claimed the divergence direction was safe — "quarantined
rows are exactly the ones the deliverable also excludes." That is false**, and
worth recording because it was asserted without checking.
`aggregate_measurements` resolves its sources through
`discover_measurement_sources` (`_cli_output_manager.py:1262-1264` →
`_measurement_sources.py:99-120`), which *prefers the aggregate and skips the
individuals entirely*. So when an aggregate is present the deliverable is built
from it alone and the quarantined rows are **included** in the published result,
not excluded. The monitor, which drops them, diverges from the deliverable in the
opposite direction to the one claimed. The honest statement is simply that the
two disagree on that path, and the banner says the monitor's count changed.

A pathological case is possible and worth naming: a staged run whose images
carry no Stage-3 markers has *every* parquet relocated, leaving the directory
with only `_dataset_aggregated.parquet` or nothing. The CLI itself raises "No
current-epoch measurements were available to aggregate"
(`_cli_checkpoint_handler.py:276-279`) in that situation, so the monitor showing
an empty frame is reporting a real failure, not inventing one. (It cannot show an
aggregate-only frame: `_`-prefixed names are filtered unconditionally, per 3.3.)

What that costs, stated plainly: the monitor shows **pre-post, uncurated**
values. For a pipeline carrying `PostMeasurement` ops these differ from the
published deliverables, and colonies removed during Dash curation still appear.
The banner says so permanently rather than conditionally. Anyone who wants
post-applied, curated numbers uses the Dash results viewer, which exists for
exactly that and does it properly.

What it buys is the whole point: no predicate, no completion markers, no mtime
heuristics, no clock-skew edge case, no upstream change to CLI completion
semantics, and — the reason this matters to a user rather than to an implementer
— **the numbers on the y-axis can never silently change while someone is
watching them.**

### 3.2 The metadata problem

The user's `--metadata` CSV is left-joined onto the master by `join_metadata` at
`_cli_output_manager.py:936` — inside `finalize_post_master_outputs`. Per-image
parquets carry only image-intrinsic metadata (attached by
`image.metadata.insert_metadata`, `_image_pipeline_core.py:1218`) plus the
identity columns. There is no back door: EXIF lands in
`image._metadata.imported`, which `insert_metadata` does not read — it uses
`_public_protected_metadata` only (`_metadata_accessor.py:344-345`).

This matters because every control in this dashboard is a CSV metadata column:

- `MetadataGenetic_Strain` — the page selector
- `MetadatasCondition_Media` — the default grouping
- `MetadataCulture_Time` — the x axis
- `MetadataSample_BioReplicate` — the trace identity

`MetadataCulture_Time` in particular is **only ever consumed** in this codebase;
no pipeline stage produces it. It exists solely because the user's CSV supplied
it.

**Metadata column identity is *discovered from the frame*, never assumed — not
from a literal, and not from a schema enum either.** An earlier revision required
deriving every column name from the schema enums. The validated run refutes that:
**15 of its 20 metadata columns are not members of any `phenotypic.schema` enum**
(3.3.0.1). They are user-CSV columns — `MetadataCulture_AgeHours`,
`MetadataPlate_Well`, `MetadataGenetic_Source`, `MetadataSample_ExperimentalId`
and the rest — and an enum-derived control finds nothing on that run. The five that
are schema members are `MetadataGenetic_Strain`, `MetadataImage_ImageName`,
`MetadataExperiment_Dataset`, `MetadataImage_ImageType` and
`MetadataImage_BitDepth` — and the last three of those are identity or
image-intrinsic, not anything a user would group by.

Where a schema enum *does* name a column the monitor needs structurally (the
`point_id` triple, 4.2), derive it from the enum rather than a literal — that
much of the old rule survives. The media column, for instance, is not
`MetadataCondition_Media`: `CONDITION_METADATA.category()` returns
`"MetadatasCondition"` with a stray `s` (`schema/_experimental_tags/_condition.py:20`),
contradicting its own docstring and every sibling category. It is a pre-existing
bug, tracked separately. A hand-typed `MetadataCondition_Media` matches nothing,
and a user CSV column spelled that way is not recognised by `is_metadata_header`,
so `join_metadata` double-prefixes it to `Metadata_MetadataCondition_Media`.
Deriving from the enum sidesteps the bug and stays correct after it is fixed.

**Resolution: the user chooses the CSV. The monitor does not infer it.**

The sidebar carries a metadata-CSV picker, resolved through the scope root
(Section 9). When `deliverables/metadata.csv` exists it is offered **pre-filled**
as a suggestion the user confirms — visibly chosen, never silently adopted. The
choice is **scoped to the run**: switching runs clears it, so run B can never
inherit run A's labels.

**Why inference was removed, since it looks like an obvious convenience.** Four
successive rules for "which CSV belongs to this run" were each reproduced as
broken, in five distinct ways:

| Rule | Broken by |
|---|---|
| Prefer `deliverables/metadata.csv` | `--restart` preserves `deliverables/`, so it is the *previous* run's copy, served for the whole live run |
| Any `.phenotypic/` record outranks it | `--mode recompile` writes **only** the deliverables copy — it dispatches early and `sys.exit(0)`s at `phenotypicCLI.py:1148`, before every run-start path — so the one correct source ranked last. The validated run is exactly this case, and would have resolved to *no metadata at all* |
| Newest record wins | `--mode recompile --slurm` *does* write `job_metadata.json` with `METADATA_CSV: None` (`phenotypicCLI.py:2252-2262`), so a recompile without `--metadata` posts a newer negative marker than the valid CSV it is recompiling against |
| Newest record wins (timestamps) | `job_metadata.json`'s mtime is not its record's write time — `mirror_job_to_metadata` rewrites the whole blob on every role mirror (`_cli_slurm_lifecycle.py:315`) with `METADATA_CSV` untouched. And on a bucket-backed deployment an uploaded or rsynced tree carries transfer times, not write times, so relative order is destroyed outright |

Each fix was correct for the door it closed and wrong for the next. The pattern is
the same one that ended the run-state predicate (3.1.1) and the aggregate merge
(3.3): several plausible rules failing differently means the question is wrong,
not the answers. A picker cannot be stale, because the user is looking at it.

**What this deletes.** The run-start CSV copy (a CLI change), the negative marker
distinguishing "no CSV" from "unknown", the ordering rule in all four of its
forms, the per-poll source re-resolution, and any dependence on timestamp
reliability across FUSE, rsync or lifecycle rewrites. `job_metadata.json` and the
`.phenotypic/` tree are not read for metadata at all.

**What it costs.** One confirmation per run, on a pre-filled field, in the common
case where the suggestion is right. That is the whole price.

**Three things the picker still owes the user:**

- **A changed file is still noticed.** The chosen CSV's
  `(path, mtime_ns, size)` is part of the freshness token (Section 10), so
  editing it mid-session re-joins rather than serving a cached frame.
- **A vanished file is reported, not silently dropped.** If the chosen path stops
  resolving, the app says so and the controls fall back to frame-intrinsic
  metadata — it never quietly reverts to a different CSV.
- **No choice is a legitimate state.** With nothing selected the app opens with
  grouping restricted to whatever metadata the per-image frames carry, and says
  so plainly. The validated run has four such columns (3.3.0.1), none of them
  groupable, so this state is honest rather than useful — which is the argument
  for the pre-fill, not for inference.
**Join semantics differ mid-run, and this is not cosmetic.** `join_metadata`
puts metadata on the **left** and joins measurements in
(`_cli_output_manager.py:175`), so `how="left"` keeps every metadata row that
matched nothing, flagged `QC_MetadataOnly`. At finalize a phantom means "this
strain was never detected" — real signal. Mid-run it means "this image has not
been processed yet", which early in a run is nearly the entire CSV. The same
function, the same flag, opposite meanings.

**The monitor therefore joins with `how="inner"`** (the default), so a live plot
shows only colonies that were actually measured. Pending work is communicated by
the progress banner (Section 10), not by flooding every facet with null rows.

**But the join drops measured colonies too, under either mode, and that must be
visible.** The docstring is explicit (`:105-107`): a left join "keeps
*metadata*-unmatched rows but still drops *measurement*-unmatched rows, because
measurements are the right frame." So a colony whose image has no CSV row
disappears from the plot regardless of `how`. Demonstrated: 4 measured rows in,
3 out, one image silently absent. The CLI logs a WARNING (`:204-212`); a
dashboard that just renders a thinner facet gives the reader nothing.

**The dominant cause is not what an image-level reading suggests.** On the
validated run the 42 dropped colonies are spread across **29 of the 30 images**,
and *every* image is present in the CSV. The join key is per-colony
`(MetadataImage_ImageName, Grid_ColMajorIdx)` (3.3.0), the CSV holds exactly 16
wells per image, and the detector found 15–21 colonies per image across 32
distinct grid indices. The dropped rows are **over-detections** — objects in grid
cells the plate layout never had.

That matters because over-detection is exactly the kind of thing someone watching
a live run wants to see. The image-level causes an earlier revision listed — a
CSV written for the final plate set, a stray calibration image, a stem typo — are
real but were not what fired here, and a fixture built from them would not
exercise the mode that does.

**The monitor gets the drop count from `prepare_metadata_join_keys`**
(`_cli/_metadata_join.py:32-101`) — the pure, polars-only function
`join_metadata` itself calls at `_cli_output_manager.py:167`. It returns
`analysis.unmatched_measurement_count` (`:95`) computed by anti-join under the
same String-cast normalization the production join uses (`:66-71`).

The count genuinely does not come back from `join_metadata`, which returns a bare
`pl.DataFrame` (`_cli_output_manager.py:83-88`, `return out` at `:237`) and uses
the number only for a `logger.warning` (`:203-212`). But an earlier revision
concluded from that the monitor should hand-roll its own anti-join, which was
wrong twice over: it reimplements a function that already exists, and a
hand-rolled anti-join without the String cast breaks the moment a CSV column and
its parquet counterpart differ in dtype — an inferred `Int64` against a `Utf8`.
That is exactly the drift 4.4 exists to prevent, reintroduced in the section next
door. Importing it is strictly less code and less risk.

**One special case must be handled at the call site.** When the CSV shares *no*
columns with the measurements, `prepare_metadata_join_keys` reports
`unmatched_measurement_count = measurements.height` (`:60`) while `join_metadata`
logs a warning and returns the frame **unjoined and unchanged** (`:133-137`).
Reporting "every colony was dropped" there would be actively misleading — nothing
was dropped; the CSV simply has no join key. This is its own condition with its
own message (Section 12), distinct from both "no CSV supplied" and "some
colonies unmatched".

Otherwise the banner reports the count, worded for the mechanism that actually
fires: *"42 measured colonies across 29 images had no metadata row and are not
shown."* Note the plural framing — the earlier single-image example implied a
broken image, which reads as 29 broken images on this run and is misleading.

**A toggle admits them when wanted, defaulting to off.** Unmatched measured
colonies are dropped by default, so every plotted point carries full metadata and
the facets stay meaningful. A sidebar control includes them as an explicit
**unassigned** group. This exists because over-detection is a detector-quality
signal: the count tells you a problem exists, the toggle tells you where.

The same drop count feeds the anti-join for the toggle — the unmatched set *is*
the group, so nothing is computed twice.

**These rows have no x, and saying "draw them as one series" is not enough.**
An earlier revision said exactly that; executed against the validated run it
produces 42 points with `x = NaN` and intact `y`, which Plotly draws as nothing.
The cause is structural: per-image parquets carry only
`MetadataExperiment_Dataset`, `MetadataImage_ImageName`, `MetadataImage_ImageType`
and `MetadataImage_BitDepth`, and **no time-like column of any kind**. Every
candidate x lives in the CSV these rows by definition failed to match. All three
views take x from `time`, so the toggle would render a blank page in each —
and test 28's predecessor could not catch it, because the trace *does* exist,
carrying 42 invisible points.

**Recovering x, and when to refuse.** In the validated run the time columns are
**image-level**: `MetadataCulture_AgeHours` and `MetadataAcquisition_Datetime`
each have exactly one distinct value per image, while `MetadataGenetic_Strain`
has four. So x is recoverable for an unmatched colony by a second, **image-only**
lookup of the chosen time column on `MetadataImage_ImageName`, leaving strain,
condition and replicate genuinely null.

That is a property of this CSV, not a guarantee. The monitor therefore checks the
precondition — the chosen time column has **exactly one distinct non-null value
per image** — and:

The non-null qualifier is load-bearing. Polars `n_unique` counts null as a
distinct value, so a *mixed* image correctly fails a plain distinct-count check —
but an image whose time cell is blank on every row reports `n_unique == 1`,
passes, contributes a single null-valued lookup row, and lands its unassigned
points back at `x = null`: toggle enabled, trace present, nothing drawn. Blank
cells for a subset of images is an ordinary CSV-authoring gap, not an exotic
input.

- when it holds, joins x image-wise for the unassigned group;
- when it fails, **disables the toggle with an explanation** rather than
  rendering an empty page. A control that silently draws nothing is worse than a
  control that says why it cannot.

**Label the group explicitly — but fill only the columns that are actually
null.** `_display_pairs` renders a null group as `<null>`
(`_plot_meas_time_series.py:350`), so a sentinel `"unassigned"` string is written
into the group columns to produce the intended label and to avoid colliding with
a genuine null strain value.

The fill is scoped to **CSV-derived columns only**. Unmatched rows are not null
everywhere: they retain the four columns per-image parquets always carry —
`MetadataExperiment_Dataset`, `MetadataImage_ImageName`, `MetadataImage_ImageType`,
`MetadataImage_BitDepth` (verified on all 42 rows of the validated run:
dataset `inputs_frame00`, type `GridImage`, depth 16). Filling one of those would
overwrite a real value with a false one, and for `MetadataExperiment_Dataset` it
is worse than cosmetic — that column is the first element of the `point_id`
triple, which Section 7 resolves to `dataset_hdf_dir(root, dataset)`. A sentinel
there sends **every unassigned click to a nonexistent HDF**, so the one group
whose pixels a user most wants to inspect would be the one group that cannot be
inspected. Nothing feeding `point_id` is ever filled.

**Report the third count as well — with the mid-run caveat the other two got.**
`prepare_metadata_join_keys` also returns `unmatched_metadata_count`: CSV rows
that matched no measured colony. On the *finished* validated run that is 10 —
designed wells where nothing was detected, the "this strain never grew" signal
this section calls real above.

Mid-run it means something else entirely, and the divergence is severe. Measured
against the same run truncated to partial states: 464 of 480 after one image, 322
after ten, 10 after all thirty. A banner reading "322 wells never grew" a third
of the way through a run overstates the truth by 32×.

This is the **third** instance of the finalize-versus-mid-run trap in this one
section — after the left-join phantoms and the whole-CSV duplicate count — and it
was very nearly the one that shipped without a caveat, having been added in a
later revision than the pattern it belongs to. The banner reads *"N metadata rows
have no measured colony yet"*, which is true in both regimes.

**Over-matching needs reporting as much as under-matching.** Because metadata is
the left frame, a CSV with more than one row per join key **multiplies** every
measured colony for that key — a common authoring error is one row per well
rather than per image, which silently doubles every point in a facet and inflates
the colony count with no signal. The CLI treats this as warning-worthy in its own
right ("each duplicate fans the join out into extra rows",
`_cli_output_manager.py:193-200`), and `prepare_metadata_join_keys` already
returns `duplicate_metadata_key_count` (`_metadata_join.py:97-99`) — free, since
the monitor imports it for the drop count anyway.

**Word the banner for what that number actually measures.** It counts duplicates
across the *entire* CSV (`normalized_metadata.height - n_unique(subset=common)`,
`:97-99`), not duplicates among keys that matched something. Mid-run — this app's
whole premise — most CSV rows match no measurement yet, so duplicates among
not-yet-processed images inflate the count while multiplying nothing on screen.
Saying "colonies are multiplied" would then be false for most of a run. The
banner says instead: *"the metadata CSV has N duplicate rows for its join key;
matched images are multiplied accordingly."* This is the same finalize-versus-
mid-run semantic trap 3.2 already navigates for the left join, reappearing in the
fix that answered it — worth noticing as a pattern rather than a one-off.

### 3.3 Frame selection

One path, always:

```
per-dataset: read the individual per-image parquets under
             results/<ds>/measurements/, skipping any name starting with "_",
             then join_metadata(how="inner") if a CSV is available
```

**`_dataset_aggregated.parquet` is never read.** This is the same conclusion as
3.1.1, reached for the same reason, and it took three attempts to see.

**Why the aggregate is never read.** It is built *exclusively* from the per-image
parquets in the same directory: `_scan_unchunked_parquets` globs
`results/*/measurements/*.parquet`, skipping `_`-prefixed files
(`_cli_chunk_writer.py:196-208`), and appends those rows to the aggregate (`:163`
→ `:323-349`). Every row it holds came from an individual parquet that was on
disk at some checkpoint. Nothing ever prunes or rebuilds it —
`DATASET_AGGREGATED_PARQUET` is written only at `_cli_chunk_writer.py:338`, and
individual parquets are never deleted, only relocated by the two quarantines of
3.1.1.

Follow that through. For an image whose individual is still on disk, the
aggregate is redundant — the individual is the authoritative and more recent
record. For an image whose individual is *gone*, the aggregate holds either
quarantined rows or rows left stale by a `--restart` that rewrote the image
smaller. **So the aggregate's only possible unique contribution is exactly the
set of rows that must not be shown.** Reading it can never add correct data and
can only add wrong data.

That also disposes of the double-counting problem an earlier revision worried
about: skipping `_`-prefixed names — the same filter the chunk writer itself
applies at `:203-205` — is the whole of the solution.

**Three rules were tried here before this one**, and the progression is worth
recording because each looked reasonable:

| Rule | Why it failed |
|---|---|
| Prefer the aggregate over individuals (the CLI's own rule, `_measurement_sources.py:104-120`) | The aggregate is rewritten only at checkpoints, clamped to every 50–500 images, so plots freeze for up to 500 images at a time. |
| Union both, newest source file wins | The `--restart` collision lives in *one file with one mtime*, so a per-file rule cannot order it; and the aggregate's mtime is newer than every pre-checkpoint individual, so it wins keys it should lose. |
| Union both, image-level authority (individual's existence suppresses the aggregate for that image) | Says nothing about images with *no* individual — precisely the quarantine case — so the aggregate still resurrected exactly the rows the quarantine removed. Half a rule. |

Each fix addressed the previous symptom rather than the shared cause, which was
reading a derived artifact at all when the source it derives from is present and
authoritative.

Four details, each of which is a real bug if omitted:

- **Skip filenames beginning `_` or `.`.** `_update_dataset_parquet`
  (`_cli_chunk_writer.py:323-349`) writes `_dataset_aggregated.parquet` *inside
  the globbed directory*, and `Path.glob("*.parquet")` matches leading
  underscores, so an unfiltered glob double-counts every colony on a SLURM run.
  The chunk writer applies the same filter for the same reason at `:203-205`.

  **The dot is not defensive padding.** On the validated run — an exFAT external
  volume, which is a normal way to hold this data — `glob("*.parquet")` returns
  **60 entries for 30 images**: macOS writes an AppleDouble `._<stem>.parquet`
  sidecar beside every file. None starts with `_`, so an underscore-only filter
  admits all thirty, and each raises `ComputeError: the file must end with PAR1`.
  Under Section 10 they would be reported as permanently unreadable — thirty
  false alarms on a perfectly healthy finished run, plus thirty phantom entries
  in the freshness token. The CLI survives this only because
  `aggregate_parquet_files` catches per file and logs (`_cli_parquet_agg.py:97-105`).

  This was caught by a reviewer re-deriving "all 30 parquets survive finalize"
  *with the rule the spec specifies* rather than by counting the directory, which
  is what a validation section is for.
- **Concat must tolerate schema divergence.** `aggregate_parquet_files`
  (`_cli_parquet_agg.py:83-108`) falls back from a fast multi-path read to
  per-file `diagonal_relaxed` *because* heterogeneous schemas occur. Use
  `diagonal_relaxed`, as `_read_and_concat` does (`_cli_chunk_writer.py:271`).
- **Re-assert the image name from the parquet file stem — and know when that
  cannot work.** `_read_and_concat` overwrites `MetadataImage_ImageName` with
  the file stem (`_cli_chunk_writer.py:235`), which is what guarantees the name
  matches the HDF filename, and repairs the UUID-shaped names staged-HDF reloads
  produce (the reason `_measurement_sources.py:35-59` exists).

  The stem rule works unconditionally here *because* the aggregate is never read.
  It is what `_aggregate_needs_image_name_recovery` exists to guard against —
  the aggregate's own stem is `_dataset_aggregated`, so the rule cannot repair
  it. Not reading the aggregate removes both the failure and the guard: the
  monitor no longer needs that import at all.
- **Back-fill the dataset column.** `MetadataExperiment_Dataset` is present in
  per-image parquets only when `include_dataset_column` is true;
  `--no-dataset-column` removes it. `_read_and_concat:258-265` back-fills it
  from the directory name. Do the same.

### 3.3.0 Validated against a real completed run

Everything above was derived from the CLI source. It has now been checked against
a real, finalized, staged local run (30 images, one dataset, `--metadata`
supplied via recompile). The claims that held:

| Claim | Observed |
|---|---|
| `run_completion.json` absent on a local run (3.1.1) | Absent; only `manifest.json` and `staged_finalization_complete.json` |
| Staged runs write no `_dataset_aggregated.parquet` (3.3) | None present |
| Staged local runs are exempt from the finalize quarantine (3.1.1) | No parquet relocated |
| `Grid_*Interval*` never produced (3.4) | Absent; `Grid_RowNum/ColNum/RowMajorIdx/ColMajorIdx` present |
| `Bbox_Center*` / `Bbox_Min/Max*` always present (3.4) | Present in both per-image parquets and the mirror |
| HDF path composes from `(dataset, stem)` (7) | Resolves; layers `rgb`, `gray`, `detect_mat`, `objmap` |
| `LayerName` omits `gray` though HDFs carry it (7) | `layers/gray` present and not selectable |
| Per-image parquets carry no CSV metadata (3.2) | Confirmed **after a recompile that joined metadata** — the mirror gained 16 metadata columns plus the `QC_MetadataOnly` flag; the per-image parquets gained none |
| Per-image parquets survive finalize (3.1.1) | 30 present — counted *by the `_`/`.` filter rule*, since the directory itself holds 60 entries (3.3) |

**The join is per-colony, not per-image.** The common columns were
`(MetadataImage_ImageName, Grid_ColMajorIdx)` — the CSV carries one row per well,
so `join_metadata`'s key detection produces a composite key. The monitor's own
join inherits this automatically by reusing the same function; it must not assume
the key is the image alone.

**And half that key is a *measured* value, not an identity.** `Grid_ColMajorIdx`
is grid-detector output. So a `--restart`, a re-measure, or any change that
shifts grid geometry can give an image's colonies different indices, which
re-join to **different CSV rows**. A point then keeps its y value while changing
strain, condition, facet and page — its *identity* moves even though its number
does not. Section 3.1.1's guarantee that values never change silently still holds
literally and is, in this one respect, narrower than it sounds: the label can
change under a viewer even when the measurement does not. Worth knowing before
trusting a facet across a restart.

(A corroborating detail that *supports* an existing decision: the parquet's
`Grid_ColMajorIdx` is `UInt16` while the CSV's is `Int64`. Test 19's dtype case
is real in this data, and 3.2's insistence on `prepare_metadata_join_keys`'s
String cast over a hand-rolled anti-join is vindicated rather than theoretical.)

**Real numbers for the join, which 3.2 previously argued in the abstract:**

```
512  measured colonies on disk (30 per-image parquets)
470  matched rows in the mirror        (QC_MetadataOnly = False)
 42  measured colonies silently DROPPED (no matching CSV row)
+10  phantom rows ADDED                 (QC_MetadataOnly = True, null Object_Label/Bbox)
480  mirror rows
```

Both directions fired on a single ordinary run. The 42 dropped colonies are why
3.2 reports a drop count rather than trusting the join, and the 10 phantoms are
why it joins `how="inner"`. Neither was hypothetical.

### 3.3.0.1 The schema defaults do not exist in real data

`PlotColonyMetricOverTime` defaults `time` to `MetadataCulture_Time` and
`groupby` to `[MetadatasCondition_Media]`. **Neither column exists in the
validated run.** Its time axis is `MetadataCulture_AgeHours` (also
`MetadataAcquisition_Datetime`, `MetadataCulture_StackedDatetime`), and its
condition is decomposed across carbon source, nitrogen source and pH rather than
a single "media" field.

A third schema name is also absent: `MetadataSample_BioReplicate`, the default
`replicate_label` and the trace identity, does not exist in this run either — the
nearest real column is `MetadataSample_ExperimentalId`.

So **the monitor never relies on a schema default for any control.** All five —
strain (page), groupby (columns), replicate (trace), time (x), measurement (y) —
are populated from the columns actually present in the loaded frame, and every
one is passed explicitly to
`PlotColonyMetricOverTime` rather than left to default. A default that resolves
to a missing column would fail at `_validate_input_columns`, which is the correct
loud failure, but only after presenting the user a control that could never work.

**The `MetadatasCondition` typo is present in production data, not just in
theory.** The run's CSV uses the documented-correct spelling
(`MetadataCondition_CarbonSource`, …). Because `category()` returns
`"MetadatasCondition"`, `is_metadata_header` does not recognise those columns, so
`ensure_metadata_prefix` double-prefixed every one of them:

```
CSV:    MetadataCondition_CarbonSource
mirror: Metadata_MetadataCondition_CarbonSource
```

A design reviewer predicted this exact failure from the code alone; the run
confirms it reaches user data. It is still out of scope to fix here (renaming a
public column), but the monitor must therefore treat metadata column names as
**discovered, never assumed** — which the rule above already requires.

### 3.3.1 What the monitor shows, and what it therefore is not

Per-image parquets are written with `apply_post=False`
(`_cli_process_single.py:98-101`). Since the monitor reads only those (3.1.1),
its values are **pre-post and uncurated**, permanently and by design:

- For a pipeline carrying `PostMeasurement` ops, the plotted values differ from
  the published deliverables — `EdgeCorrector` rescales, filtering ops remove
  rows. The deliverables are not wrong and neither is the monitor; they are
  different quantities.
- Colonies removed during Dash curation still appear here, because curation is
  written to the mirror (`_curation_labels.py:807-816`), which the monitor does
  not read.

The banner states this permanently. It is not a conditional warning, because the
condition is always true — and a warning that never turns off is better stated as
a label.

**This is a deliberate scope boundary, not a limitation to work around.** A
monitor answers "is this run going the way I expect, right now". A results
viewer answers "what are the final numbers". The Dash results viewer already does
the second job, over exactly the artifacts this app declines to read. Adding
post-application and curation-awareness here would duplicate it, and would
reintroduce the run-state detection that 3.1.1 exists to eliminate.

### 3.4 Column identities, verified

| Meaning | Column | Note |
|---|---|---|
| Dataset | `MetadataExperiment_Dataset` | Conditional — absent under `--no-dataset-column`; back-filled (3.3) |
| Image **stem** | `MetadataImage_ImageName` | `pl.lit(stem)` at `_cli_chunk_writer.py:235`; extension separately in `MetadataImage_FileSuffix` |
| Colony label | `Object_Label` | `_filtered_state.py:56` |
| Crop centre | `Bbox_CenterRR`, `Bbox_CenterCC` | Always present — from the appended `objects.info()` block, not an optional measure op |
| Bbox extent | `Bbox_MinRR`, `Bbox_MaxRR`, `Bbox_MinCC`, `Bbox_MaxCC` | Same; confirmed in both grid and non-grid runs |
| Grid section id | `Grid_RowNum`, `Grid_ColNum`, `Grid_RowMajorIdx` | **GridImage pipelines only** (`_image_pipeline_core.py:1229-1232`) |
| Grid column-major index | `Grid_ColMajorIdx` | Same call path (`grid.info()`, `_grid_finder.py:366,380`). Listed separately because 3.2/3.3.0 make it load-bearing: it is half the metadata join key, and it is `UInt16` in the parquet against `Int64` in the CSV |

**`Grid_RowIntervalStart/End` and `Grid_ColIntervalStart/End` are never produced
by any live pipeline path.** They are declared in `schema/_grid.py` and have no
producer and no consumer in any executing code. (They do appear as literal
headers in two bundled static sample CSVs under `data/meas/`, which is why the
absolute phrasing of an earlier draft was slightly too strong — nothing computes
them, and they will not be in a real run's frame.) An earlier draft cited the
schema file as verification, which confirmed the *name* existed, not the
*column*. Section 7.1 is built on what is actually written.

Grid view is therefore unavailable for non-grid pipelines. The mode is hidden,
not shown-and-broken, when `Grid_RowMajorIdx` is absent from the frame.

### 3.5 Why not `OutputRoot`

`OutputRoot.discover` cannot open a live run. `BundleLayout.detect` raises
`FileNotFoundError` unless `master_measurements.parquet` is a file at the path
or under its `deliverables/` (`sdk_/_io_constants.py:1986-2001`), and
`_output_root.py:320-326` raises again. Reproduced on a real run with the master
removed. Since Section 3.1 establishes the master is absent for the whole of a
local run, `OutputRoot` is structurally unable to serve this app's primary case.

The monitor therefore composes paths from `phenotypic.sdk_` helpers directly —
`results_dir`, `dataset_measurements_dir`, `dataset_hdf_dir`,
`dataset_overlays_dir`, `deliverables_dir`, `measurements_parquet_path`. These
are pure path expressions with no existence requirements, and they are the
canonical source the project rules already require ("always resolve paths via
the `phenotypic.sdk_` helpers, never hand-join names").

Two consequences, both good: the HDF lookup becomes
`dataset_hdf_dir(root, dataset) / f"{stem}.h5"` with an explicit existence check,
rather than `OutputRoot.hdf_path`'s `Path | None` that Section 7 previously
assumed was a `Path`; and **the monitor imports nothing from
`phenotypic.gui`**, which deletes the Dash-free-`OutputRoot` upstream change
from this spec entirely.

---

## 4. Upstream prerequisites

Three additive changes in `phenotypic`, plus a documentation note in 4.5. None
alters existing behaviour, and none is required for the monitor to open a run —
3.2's picker removed the only change that touched CLI write behaviour. Section
4.4 records two private imports and is not a code change.

### 4.1 Extract the crop math to a framework-free module

`crop_hdf_rgb`, `_crop_hdf_layer_window`, `_dim_outside_bbox` and
`_hdf_layer_array_to_rgb` live in `gui/_shared/tiles.py`, which imports `dash`
and `flask` at top level. Move them under `phenotypic.sdk_` (not under `gui/`,
since the monitor must not import `phenotypic.gui`); `tiles.py` re-exports so
every existing Dash import path keeps working — the pattern
`colony_view/_cropper.py` already uses.

The new module lives at `phenotypic/sdk_/_crops.py`, and its symbols
(`crop_hdf_rgb`, the rectangle-window function, the strided reader, `LayerName`)
are **exported publicly from `sdk_/__init__.py`** — unlike the plotting helpers
of 4.3 and the `_cli` internals of 4.4. `sdk_` is already the package's public
utility surface (it publicly exports the path helpers this design relies on), and
these are general-purpose image-window primitives with two in-tree consumers, not
one module's internals.

**These four functions are not self-contained, and moving them alone would
reintroduce the exact dependency the move exists to remove.** Four dangling
references have to travel with them:

- `_hdf_layer_array_to_rgb` lazily imports `_label_map_to_rgb` and
  `_normalize_to_uint8` from `phenotypic.gui.builder._image_renderer`
  (`tiles.py:337-340`) — a module whose top level does `import cv2` and
  `from dash import dash_table` (`_image_renderer.py:35,37`). Every
  `detect_mat`/`objmap` crop goes through this path, so the monitor's layer
  selector would transitively import the Dash builder renderer. Both functions
  are themselves pure (numpy plus optional skimage/matplotlib —
  `_image_renderer.py:125-183`), so they **move into the new module** and
  `_image_renderer.py` re-exports them.
- `_dim_outside_bbox`'s default `bg: tuple[int,int,int] = TILE_DIM_RGB` binds a
  constant imported from `phenotypic.gui._design` (`tiles.py:48`). The new
  module defines its own black default; `gui._design` keeps `TILE_DIM_RGB` for
  Dash-side callers.
- `LayerName` — the `Literal["rgb","detect_mat","objmap"]` in every one of these
  signatures — is declared locally in `tiles.py:276` and moves with them.
- `_clamp` (`tiles.py:82-84`) is called by `_crop_hdf_layer_window` at
  `:454-457` to bound the dim rectangle. It is a three-line pure helper with no
  framework dependency; without it the moved module raises `NameError`.

That last one was missed twice — once in the original extraction list and again
in the revision whose stated lesson was to audit exactly this. Before
implementing the move, resolve every free name in the four function bodies
mechanically rather than by reading, because reading has now failed at it twice.

The general lesson, since it is what a first pass missed: "pure numpy/PIL/h5py"
was asserted from the function bodies' *visible* operations, and the
function-local import three lines in was not counted as a dependency. Audit what
each function actually reaches for, including lazy imports and default-argument
bindings, before treating any of them as movable.

Two additions while moving:

- Generalise the window function to accept an explicit
  `(top, left, bottom, right)` rectangle, with the current centre-plus-size call
  a thin wrapper. Grid and plate modes need the rectangle form.
- Add the strided reader Section 7.3 requires. It belongs here, not in
  `apps/monitor/`, so the same golden fixture covers it and the Dash viewer can
  adopt it for its own full-decode problem.

`_load_hdf_layer_rgb` stays where it is; the monitor never uses the full-decode
path.

### 4.2 Add `point_id` to `PlotMeasTimeSeries`

A new field, forwarded by `PlotColonyMetricOverTime`:

```python
point_id: ColumnRefList = Field(default_factory=list)
```

In `_build_page`, `customdata` is sliced from the same time-sorted frame that
already produces `x` and `y`:

```python
ordered = replicate_frame.sort_values(self.time, kind="mergesort")
go.Scatter(
    x=ordered[self.time].tolist(),
    y=ordered[measurement].tolist(),
    customdata=ordered[self.point_id].to_numpy() if self.point_id else None,
    ...
)
```

Alignment is structural — one row indexer produces all three — rather than
reconstructed by a second, independently-maintained sort.

Two supporting claims, with their evidence stated so a reader can weigh them:
the mixed-dtype `(str, str, int)` round-trip through Plotly's JSON serialization
without coercion loss was **reproduced by execution**; that Streamlit's
`on_select="rerun"` returns a `PlotlyState` whose `selection.points` carry
`customdata` was checked against Streamlit's installed source and documentation
but **not executed against a live session**. Confirm the second during
implementation before building on it.

`PlotColonyMetricOverTime` needs the field **declared on the outer class too**,
not only forwarded — it has no `point_id` surface today, so `configured.point_id`
does not exist. That means a `point_id: ColumnRefList = Field(default_factory=list)`
declaration alongside its existing `connect: bool`, plus the one-line forward in
`inspect`, which builds its delegate field-by-field
(`_plot_colony_metric_over_time.py:73-81`). `point_id` must also join the
`requested` list in `_validate_input_columns`, so a missing column fails loudly.

The monitor passes the colony primary key — three small values per point.
Centroid, bbox and grid membership are *not* shipped through `customdata`; they
are looked up from the in-memory frame by that key on click, as the Dash crop
route does at `tiles.py:731-745`.

**What is deliberately not claimed.** An earlier draft asserted `point_id`
columns must be added to the `excluded` set in `_measurement_columns`, because
`Object_Label` would otherwise be auto-selected as a measurement. That is false.
Line 180 already filters `startswith(("Object_", "Grid_", "Quality", "QC_"))`,
line 179 filters `is_metadata_header`, and `Object_Label` is additionally in
`_nonmeasurement_schema_headers()`. Verified by execution: with `on=[]` and all
three key columns present, `_measurement_columns` returns `['Shape_Area']` alone.
No `_measurement_columns` change is part of this spec, and the test that was
specified to guard it has been removed — it could not fail, which the
repository's test-integrity rule forbids.

### 4.3 Export the grouping helpers

`_group_rows`, `_group_pairs` and `_display_pairs` in `_plot_meas_time_series.py`
exist specifically to avoid pandas' null-category and mixed-type coercion when
grouping. The box builder needs *identical* faceting, so either these are
exported or the monitor reimplements the subtleties and diverges on nulls.

**`_typed_group_key` must be exported too, and `_canonical_group_key` is not
redundant.** A prior revision dropped both on the grounds that
`_canonical_group_key` is "a thin wrapper over the already-public
`canonical_group_key`". That is false, and false precisely on the column most at
risk. `_canonical_group_key` (`:297-307`) encodes `pd.Timedelta` at true
nanosecond precision via `value.value`, and tags `pd.Timestamp` as
`datetime_ns`; the public `canonical_group_key`
(`abc_/plotting/_output.py:69-84`) tags datetimes `"datetime"` and reconstructs
timedeltas from `value.microseconds * 1_000`, truncating at microseconds.

The consequence is concrete: the sort key the line builder actually uses is
`_typed_group_key` (`:293-294`, applied at `:111`, `:193`, `:213`). A box builder
sorting with the public primitive computes different key strings, so facets can
order differently — and in overlay mode the boxes land under the wrong markers.
`MetadataCulture_Time`, the x axis of every plot in this app, is exactly the
column likely to be a Timedelta or Timestamp.

`_group_rows` already calls `_canonical_group_key` internally at `:272`, so it
ships regardless; excluding it from the export list bought nothing and created
the divergence 4.3 exists to prevent. Export `_typed_group_key` and
`_canonical_group_key` alongside the other three.

**"Export" here means a direct import from the private module** —
`from phenotypic.plotting._plot_meas_time_series import _group_rows, ...` — not
adding underscore-prefixed names to `phenotypic/plotting/__init__.py`'s
`__all__`. The project's convention is that `__init__.py` defines the public API
and `_module.py` is private; promoting five private helpers into the public
surface to serve one internal consumer would invert that. This is the same
sanctioned-private-import pattern 3.3 already uses for `_measurement_sources.py`.
The cost is that these five names become de-facto contracts for `apps/monitor`
even though they stay private to the package — worth stating so a future
refactor of `_plot_meas_time_series.py` knows the monitor is a consumer.

### 4.4 Two sanctioned private imports (no code change)

The monitor imports two `_cli` internals rather than reimplementing them:

- `join_metadata` (`_cli_output_manager.py:83-88`) — its column prefixing,
  join-key selection and phantom semantics are subtle enough that a
  reimplementation would drift, which is the failure mode this whole design is
  organised against.
- `prepare_metadata_join_keys` (`_metadata_join.py:32-101`) — the drop count and
  duplicate-key count, with production String-cast semantics (3.2). Pure polars.

A third, `_aggregate_needs_image_name_recovery`, was required by earlier
revisions and is not needed now: it guards a hazard that only exists when the
aggregate is read (3.3).

### 4.5 Record the dashboard as a consumer of the metadata schema

Add to `src/phenotypic/schema/CLAUDE.md`: **any change to a `METADATA`
`MeasurementInfo` class — adding, renaming or re-categorising a member, or
changing a `category()` — must consider this dashboard**, because every one of
its controls is a metadata column.

This is not defensive boilerplate. 3.3.0.1 records live instances: the
`MetadatasCondition` category typo, which silently double-prefixes user CSV
columns and reached production data; and three schema defaults
(`MetadataCulture_Time`, `MetadatasCondition_Media`,
`MetadataSample_BioReplicate`) that do not exist in a real run, so a plot class
defaulting to them presents controls that cannot work. Both originate in the
schema and surface only in a consumer.

**Scope the claim honestly, though.** Only 5 of the validated run's 20 metadata
columns are schema members, and three of those five are identity or
image-intrinsic; the other 15 come from the user's CSV and no
`schema/CLAUDE.md` note governs them. The note is worth adding — it would have
caught the typo and the dead defaults — but it does not protect the dashboard
from the general case, which is why 3.2's discovery rule, not the schema, is the
real defence.

The note belongs in `schema/CLAUDE.md` rather than here because that is where
someone editing a category will actually be reading.

Both are private by the project's convention that only `__init__.py` exports are
public, and neither is re-exported from `sdk_`. This is a deliberate, recorded
coupling rather than an oversight: `apps/monitor` becomes a consumer of two
private `_cli` names, and a refactor of either must account for it.
Promoting them into `sdk_` is the tidier long-term answer and is deliberately
**not** done here — it would widen the upstream surface of a design already
carrying four changes, for no behavioural gain.

**No upstream change publishes a completion marker.** An earlier revision added
one; 3.1.1 removed the need for it along with the entire run-state predicate.

---

## 5. Application architecture

```
apps/monitor/
    pyproject.toml          phenotypic, streamlit, plotly, polars, h5py, pillow, numpy
    Dockerfile
    monitor/
        scope.py            containment-checked path resolution
        runs.py             run discovery, status, path composition
        data.py             frame selection, incremental accumulation, column roles
        filters.py          metadata value filters
        figures.py          line / scatter / box builders
        images.py           three render modes, local HDF cache
        app.py              page assembly, session state, refresh
```

| Unit | Responsibility | Depends on |
|---|---|---|
| `scope.py` | `ScopeRoot(base)`; `resolve`; `list_dirs`. No knowledge of runs or Streamlit. | stdlib |
| `runs.py` | Identify run directories, compose paths via `sdk_` helpers, and suggest a metadata CSV (existence check only — no inference, 3.2). Deliberately does **not** classify run state (3.1.1). | `scope`, `sdk_` |
| `data.py` | Frame selection (3.3), incremental accumulation (11.1), column role classification, **and the polars→pandas conversion**. | `runs`, `sdk_`, polars, pandas |
| `filters.py` | Filter spec → boolean mask. Pure. | polars |
| `figures.py` | The three views and the overlay composition. Frame + selection in, figures out. | `phenotypic.plotting` |
| `images.py` | Key → pixels, three framings, local HDF cache. | crop module, `sdk_` |
| `app.py` | Widgets, session state, refresh policy, click routing. Only Streamlit-aware module. | all |

### 5.1 Packaging, and an honest note on image weight

`uv` is the repository's sole package manager, and no `apps/` directory exists
today. `apps/monitor` is added as a **uv workspace member** with `phenotypic` as
a workspace path dependency, so `uv sync` resolves both from the repo root. The
Docker build installs a pinned `phenotypic` rather than the workspace path, so
the image does not depend on repo layout.

**The image is large, and no arrangement of this design makes it small.**
`pyproject.toml` lists `dash>=4.1.0`, `dash-cytoscape`, `dash-bootstrap-components`,
`opencv-python`, `numba`, `mahotas` and the rest of the scientific stack as
**core** dependencies — the `gui` extra only adds jinja2, jupyter and pyvips.
So depending on `phenotypic` at all pulls the entire Dash stack in, whether or
not the `gui` extra is requested and whether or not any Dash code is ever
executed.

An earlier draft justified this design partly as "no Dash in the image." That
was never achievable this way and has been removed. The real and still-sufficient
benefit of the Section 3.5 and 4.1 decisions is **architectural**: the monitor
does not import `phenotypic.gui`, so it cannot break when viewer internals
change, and it does not duplicate crop geometry that would drift. Image size is
accepted as-is; if it becomes a practical problem the fix is a packaging change
upstream (moving the Dash dependencies into the `gui` extra where they belong),
which is out of scope here.

**Caching and the module boundary.** `data.py` stays framework-free: it exposes
plain functions, and `app.py` wraps them (`cached = st.cache_data(...)(load_frame)`)
rather than `data.py` carrying a `@st.cache_data` decorator. Decorating in
`data.py` would import Streamlit into a module the test suite exercises as a
pure unit, breaking both the dependency table above and Section 13's premise.

---

## 6. Views and figures

### 6.1 Line and scatter come from the same class

`PlotMeasTimeSeries` already has `connect: bool`, and `connect=False` switches
the trace mode to `"markers"` (`_plot_meas_time_series.py:228`). That is exactly
the requested scatter view: every colony as a point over time, no lines, same
faceting, still coloured per replicate. No new builder.

```python
PlotColonyMetricOverTime(
    on=measurement,
    strain_label=strain_column,
    groupby=group_columns,
    time=time_column,
    connect=(view == "line"),
    point_id=KEY_TRIPLE,
).inspect(frame)
```

Box is the only new builder: one `go.Box` per timepoint within each facet,
faceted identically via the helpers exported in 4.3. Box traces carry no
`customdata` — a box is an aggregate with no single colony to point at. Overlay
mode adds box traces beneath the marker/line traces in the same figure; clicks
resolve against marker traces only.

### 6.2 Rendering all strains without rendering all strains

The brief is all strains by default, down a long page. The constraint is that
**Streamlit cannot virtualise a scroll**: it reruns the script top to bottom and
emits every element, with no viewport awareness. There is no supported way to
render only what is on screen, short of a custom JS component.

`st.expander` is not a workaround: its content is rendered server-side and
collapsed client-side with CSS, so a collapsed strain costs a full render.
(OQ-6: confirm against Streamlit 1.61 during implementation — if collapsed
expanders do skip rendering, they become the preferred mechanism.)

The design uses **progressive chunked rendering**: all strains selected by
default, only the first *K* pages (default 3) rendered, with a "Show more"
control appending the next chunk. Each page lives in its own `st.fragment`, so
appending a chunk does not re-render the others. The chunk count is session
state, reset whenever run, strain set or filter spec changes.

This preserves the long-page reading experience and bounds render cost, at the
price of a click rather than a scroll.

**The unassigned page is pinned first and exempt from the chunk budget.** Its
natural sort position is *unpredictable*, which is the actual reason an explicit
pin is required. `_typed_group_key` (`_plot_meas_time_series.py:293-294`) delegates to
`_canonical_group_key` (`:297-307`), which for plain strings and nulls hands off
to the public `canonical_group_key` (`abc_/plotting/_output.py:59-102`). That
encodes each value as a `[column, kind, canonical]` triple, so keys differ on the
**kind** tag rather than on the raw value. (The delegation is only partial — 4.3
records where the two diverge on `pd.Timedelta`/`pd.Timestamp` — but the string
and null cases below go through the public encoder.) Verified by execution:

```
None         -> [["MetadataGenetic_Strain","null",null]]
"unassigned" -> [["MetadataGenetic_Strain","str","unassigned"]]
sorted:  None, 'MUT1', 'WT', 'unassigned', 'zzz_strain'
```

A null group sorts **first** (`"null" < "str"`), and the sentinel string used by
3.2 carries `kind="str"` exactly like a real strain, so it lands wherever
`"unassigned"` falls alphabetically among the run's strain names — before
`zzz_strain`, after `MUT1`, and anywhere at all for a different naming scheme.

Since the page's whole purpose is surfacing over-detection to someone watching a
live run, leaving its position to chance among K rendered pages defeats the
toggle. It renders first, outside the budget, with the budget applying to the
real strain pages after it.

*(An earlier revision claimed a quoted strain name "begins `\"` and sorts below
every letter, putting a null or sentinel group last." That was wrong in both
directions and was written from a reviewer's assertion without executing the
code path — the exact habit the preamble names, occurring inside the fix for the
previous instance of it. The design decision was right for a reason the
justification got backwards.)*

### 6.3 Click resolution

`st.plotly_chart(fig, on_select="rerun", key=...)` returns selected points with
their `customdata`. The handler takes the first, reads the
`(dataset, stem, label)` triple, and stores it in session state. A selection
from a box trace carries no `customdata` and leaves the current selection
unchanged.

---

## 7. Image panel

All three modes resolve the same way: look the key up in the loaded frame, read
`Bbox_*` and `Grid_RowMajorIdx`, and compose the HDF path as
`dataset_hdf_dir(root, dataset) / f"{stem}.h5"`, checking existence explicitly.

The layer selector offers `rgb`, `detect_mat` and `objmap` — the members of
`LayerName`, which is declared inside `phenotypic.gui` (`tiles.py:276`) and
therefore **moves into the new `sdk_` module** with the crop math (4.1); the
monitor may not import it from where it currently lives. Note it omits `gray`
even though every HDF carries a `layers/gray` dataset; widening it is a
follow-up, not part of this design.

| Mode | Window |
|---|---|
| **Colony** | The colony's bbox, with a small margin |
| **Grid** | Union of the `Bbox_Min/Max*` of all rows sharing `(dataset, stem, Grid_RowMajorIdx)` |
| **Plate** | Whole layer, strided, dimmed outside the target colony's bbox |

### 7.1 What grid mode does and does not give

The window is the union of member bounding boxes, mirroring
`_get_section_object_bounds_arrays` (`_grid_accessor.py:311-345`), which
group-aggregates `BBOX.MIN_RR/MAX_RR/MIN_CC/MAX_CC` by `GRID.ROW_MAJOR_IDX`.
Both column families are written to every measurement frame, so this is
computable today with no upstream work.

**The limitation, stated plainly.** The live `GridImage` computes section bounds
as *grid edges ∪ object extents* (`_get_section_bounds_arrays:349-388`) — the
grid-cell rectangle, expanded by any colony spilling past it. The grid edges
come from `get_row_edges()`/`get_col_edges()` on a live image and are not in the
measurement frame, and the interval columns that would carry them are never
written (3.4). So a section containing one small colony crops tight to that
colony rather than showing the whole cell, and an empty section cannot be shown
at all.

For the stated purpose — "is this colony weird relative to its neighbours" —
the member union is the informative part. Populating the declared interval
columns upstream would close the gap and is the natural follow-up if the
tight-crop behaviour proves annoying in use.

### 7.2 Image source tiering

Unchanged from `crop_colony`: full-resolution HDF layer first, overlay PNG as
fallback. Both are present during a live run (3.1). When a run used
`--no-save-overlays` the overlay rung is absent permanently and the tier degrades
to HDF-or-placeholder — acceptable, since in that configuration the HDF is the
only pixel source that ever exists.

Overlays are **not** written atomically (`_cli_output_manager.py:1556-1567`
writes straight to the final path), so a torn overlay read is possible during a
live run. Decode failures on the overlay fall through to the placeholder rather
than propagating.

### 7.3 Plate mode is the performance risk

`_load_hdf_layer_rgb` decodes an entire `/layers/<name>` dataset — hundreds of
megabytes for one plate — and its LRU holds four. On a shared server that is the
failure mode.

Plate mode reads **strided**: `dset[::k, ::k]` with `k` chosen for a target long
edge of roughly 1600 px, and the dim rectangle scaled by `1/k`. Dimming reuses
`_dim_outside_bbox` unchanged.

**This bounds memory, not I/O.** h5py supports multi-axis strided slicing on a
chunked dataset (verified against a real HDF whose `layers/rgb` is chunked
`(75, 200, 1)`), but a full-extent stride still touches every intersecting
chunk — which is all of them. What is avoided is materialising the
full-resolution *array*, not reading the full *dataset*. That is acceptable only
because 11.1's local HDF cache means the expensive read happens once per image
rather than once per view; on a bare FUSE mount plate mode would still be slow.

### 7.4 Degradation

Missing bbox columns → no dimming, crop still served (matching `tiles.py:731`).
Missing layer in the HDF → `KeyError` caught and degraded, as `crop_colony`
already does. Grid mode hidden entirely when `Grid_RowMajorIdx` is absent.

**A measured colony may have no readable HDF, for two distinct reasons.**
Single-pass writes the parquet first and the HDF second
(`_cli_process_single.py:109` then `:110`), and `save_image_hdf` swallows
failures and returns `None` (`_cli_output_manager.py:1604-1648`) — so a
permanently HDF-less measured image is possible. Staged GPU inverts the order:
Stage 1 writes the HDF (`_cli_staged_workers.py:113`) long before Stage 3 writes
the parquet (`:197`), so mid-run HDFs exist with no measurements and **no
`objmap` layer**. The window is wider than "until Stage 2 completes": Stage 3
writes the parquet at `:197` and only re-saves the merged HDF at `:199`, so
there is an interval in which measurements exist and point at a Stage-1 HDF that
still has no `objmap`. The layer selector must expect a `KeyError` there.

---

## 8. Filtering

Per-metadata-column value multi-selects, held as a spec in session state so they
survive reruns and view switches. The mask applies to the plotted frame and to
grid-section membership alike — a user filtered to one replicate sees only that
replicate's colonies in the section, which is the consistent reading.

Filter columns are offered from the metadata present in the *loaded* frame, so
they shrink gracefully when a live run has no CSV metadata yet.

---

## 9. Run browsing and the scope root

```python
@dataclass(frozen=True)
class ScopeRoot:
    base: Path          # fully resolved at construction

    def resolve(self, candidate: str | Path) -> Path: ...
    def list_dirs(self, rel: Path) -> list[Path]: ...
```

`resolve` joins, calls `.resolve()` (collapsing `..` and following symlinks),
then requires `is_relative_to(self.base)`. Resolving *before* the containment
check is what makes a symlink pointing outside the mount fail rather than
succeed. `base` is resolved at construction so a symlinked mount point does not
break every comparison. Violations raise `ScopeViolation`.

The base comes from `PHENOTYPIC_MONITOR_ROOT` (the container sets `/runs`).

**Multi-tenancy seam.** An operator gets a `ScopeRoot` at the mount root and
sees every project; a restricted user gets one at their own subdirectory and
cannot name anything outside it. That difference is one constructor argument —
no other module knows about it. Deciding *who* gets which root is the
authentication work deferred from this spec.

**The metadata-CSV picker lives here too**, since it resolves through the same
scope root and is subject to the same containment check — a CSV outside the root
is as inadmissible as a run outside it. It is **per-run state**: selecting a
different run clears it, which is what stops a session-state selection outliving
the run it was chosen for and labelling the next one with it. When
`deliverables/metadata.csv` exists in the newly-opened run it is pre-filled
again, so the common case is one confirmation rather than a path hunt.

The UI gives a breadcrumb, a directory listing, a free-text path box (resolved
through `ScopeRoot`), and a session-state list of recently opened runs. A
directory that is not a run is listed but not openable.

---

## 10. Refresh and caching

Streamlit reruns the whole script on every interaction, so caching discipline is
what makes this usable.

- **The freshness token is computed outside the cached function, and is a hash
  of the full `(path, mtime_ns, size)` set** — the same triple the accumulator's
  read-set uses (11.1) — plus the **chosen metadata CSV's
  `(path, mtime_ns, size)`**, not merely its mtime.

  Path identity matters because the user can change the selection mid-session. A
  bare mtime cannot express a *switch* between two files whose timestamps
  coincide — plausible under the coarse FUSE granularity this section already
  warns about — which would leave the token unmoved and the cache serving a frame
  built from the previous CSV. Same argument as the paragraph below makes about
  the parquet set, applied to the component that was left as a scalar. Neither the mirror nor
  the aggregate has an entry, because neither is ever read (3.1.1, 3.3), so
  their appearance is not a change in what is shown.

  **The set is the _effective_ one, not the literal directory listing:** a path
  whose absence is not yet confirmed (11.1) stays in it, and leaves only once the
  removal is confirmed. Read literally as the current listing, the token would
  change on the first miss, the loader would run, and rows would be retired
  immediately — defeating the debounce and failing the mutation test 10 requires.

  **The token must not be weaker than the read-set key**, which an earlier
  revision made it: a token built from the path set plus the *maximum* mtime
  misses a rewrite of any non-newest parquet — a targeted re-measure of an early
  image, or any write whose mtime does not exceed the current maximum under
  cross-node clock skew or coarse FUSE mtime granularity. The cached loader is
  then never called, so the read-set comparison 11.1 mandates never runs and its
  protection is unreachable. Hashing the full set costs nothing extra, since
  computing a maximum already requires statting every file.
- **The cache must be bounded.** `st.cache_data` defaults to `ttl=None,
  max_entries=None` — no eviction — and its default `scope="global"` means one
  cache shared by every viewer on the process. Since the token changes on every
  completed image, an unbounded cache against a long run mints a new
  multi-hundred-megabyte entry per poll and never releases one. Both
  `max_entries` (small) and `ttl` are set explicitly.
- **Budget for two copies, not one.** Reading and accumulating happen in polars;
  `phenotypic.plotting` is pandas (`_plot_meas_time_series.py:11`, and 4.2's own
  snippet uses `.sort_values`). `data.py` owns that conversion, and both
  representations of a large frame can be resident at once — so the memory
  budget behind `max_entries` counts two copies per cached entry, not one.
- A manual **Refresh** button, plus an optional auto-poll as
  `st.fragment(run_every=...)` so polling re-renders figures without re-running
  the page.
- A banner states image count and data age — *"47 images · data as of 14:32"* —
  plus the permanent pre-post/uncurated label (3.3.1) and the metadata drop count
  when non-zero (3.2). It deliberately does **not** claim "run in progress" or
  "run complete": that is the predicate 3.1.1 removed, and asserting it in the
  banner would smuggle it back in through the UI. Data age answers the question a
  viewer actually has — is this fresh — without requiring the app to know
  something it cannot reliably determine.

Per-image parquets are written atomically — `atomic_write_with_writer`
(`sdk_/_atomic_io.py:89`) writes to a same-directory `.{name}.{rand}.tmp`
sibling, fsyncs, then `os.replace`, and the temp name does not end in `.parquet`,
so a `*.parquet` glob cannot see it. HDFs use the same pattern with a `.part`
suffix.

**This makes torn reads impossible on a POSIX filesystem, which is not the
deployment target.** Section 11.1 establishes the runs directory is a GCS-FUSE
mount, where `os.replace` is emulated rather than native. An earlier draft
asserted torn parquet reads were impossible full stop while simultaneously
hedging HDF reads against exactly this concern — the two cannot both be right.
Until gcsfuse rename atomicity is confirmed for the actual bucket configuration
(flat namespace versus hierarchical, OQ-9), the conservative position holds for
both: a parquet that fails to decode is treated as in-flight, skipped for that
poll, and retried on the next one rather than propagating.

**"Retried next poll" holds only while the file keeps changing.** The loader runs
when the token changes, and the token is the path/mtime/size set — so a *stably*
undecodable parquet (a partial object left by a job killed mid-rename) freezes
the token and is skipped for the process's lifetime, silently. A torn read
self-heals, because completing the write changes the triple; a truncated leftover
does not. Undecodable paths are therefore tracked in the accumulator and surfaced
in the banner as unreadable, rather than silently omitted. This is the same shape
as the trap in 11.1 — a retry policy placed behind a cache the same section made
immune to re-entry — and it is worth checking for wherever this design says
"retry".

Images that fail write no parquet at all, and a zero-object image raises
`NoObjectsError` inside the measurer rather than producing an empty frame — so
neither failures nor empty detections are a source of schema divergence or
partial rows.

---

## 11. Docker and deployment

- `python:3.12-slim` base, dependencies installed with `uv`.
- Non-root user; the app never writes to the run folder.
- `EXPOSE 8501`; `PHENOTYPIC_MONITOR_ROOT=/runs`.
- `docker run -p 8501:8501 -v /data/runs:/runs:ro phenotypic-monitor`
- The read-only mount is the outer guarantee matching the app's read-only
  design; `ScopeRoot` is the inner one that stops traversal *within* it.

### 11.1 GCS-FUSE read strategy

**The runs directory is a GCS-FUSE bucket mount.** This is the most
consequential deployment fact in the design, because every assumption a POSIX
filesystem allows about cheap `stat` and cheap random reads is false on object
storage. Three distinct problems.

**Directory listing is a bucket operation.** The freshness token requires listing
`results/*/measurements/` on every poll — a paged object listing, billed, and
growing with the run. Mitigations: a floor on the poll interval (30 s, not 2 s),
and gcsfuse stat/type caching with a TTL below the poll interval. Listing cost
scales with images completed, so the default interval should be set by
measurement (OQ-7).

**Re-reading every per-image parquet each poll is untenable.** A naive loader
re-reads all *N* parquets whenever the token changes — *N* object GETs for a
frame that grew by one row group. The loader therefore **accumulates
incrementally**: it keeps the previous concatenated frame plus the set of
parquets already folded into it, reads only what is not in that set, and
concatenates the delta, turning per-poll cost from *O(N)* into *O(new)*.

**Where that state lives, and why not the obvious place.** It cannot live in
`st.cache_data`. That is a lookup keyed on the call's arguments, and the Section
10 freshness token is *designed* to change whenever an image lands — so every
poll is a new key, every poll is a miss, and nothing ever accumulates. (An
earlier draft asserted "the accumulator is the cached object." That is
incompatible with a volatile key and has been removed.)

The accumulator is held in **`st.cache_resource`, keyed on the run path** — one
accumulator per run, shared by every viewer on the process, guarded by an
explicit lock. On this storage backend that is the point: five people watching
one run generate one set of bucket reads, not five. The alternative,
`st.session_state`, is per-session and would multiply both memory and FUSE
traffic by the viewer count.

Five consequences to design for, none optional:

- **It needs its own bound.** `st.cache_resource` has no `ttl`/`max_entries`
  semantics comparable to `st.cache_data`, so the accumulator carries an
  explicit cap on retained runs and evicts least-recently-used itself.
- **It needs real locking.** Two viewers can poll concurrently against the same
  accumulator; the read-set update and the frame append happen under one lock,
  so a concurrent poll either waits or sees a consistent prior state.
- **The read-set is keyed on `(path, mtime_ns, size)`, not on path alone.** A
  resume, re-run or recompile can rewrite a parquet that was already folded in.
  Keyed on path, that file is never re-read and the accumulator serves stale
  rows for the remainder of the process's life — a silent, unbounded
  correctness failure. Keyed on the triple, a rewritten file simply looks new.

  On GCS-FUSE this depends on the stat-cache TTL: a same-size rewrite within the
  TTL window shows an unchanged triple. The TTL floor mandated earlier in this
  section (below the poll interval) is what closes that, and the two
  requirements are the same requirement — stated in both places because either
  alone is insufficient.

- **Re-reading a rewritten file is not enough; its old rows must be retired.**
  The accumulator holds a frame, so re-reading a changed parquet appends a second
  copy of every colony in it unless the previous rows are removed first. The
  frame therefore carries per-row source provenance (the parquet path it came
  from), and folding a path in deletes that path's existing rows before
  appending. Without this, `accumulate` cannot deliver the replacement its test
  asserts — it would only ever grow.

**How the two caching layers interact**, since holding both in mind at once is
the main cognitive cost of this design. `st.cache_resource` holds the *state*
(one accumulator per run, long-lived, shared, mutable under lock).
`st.cache_data` holds the *derived frame* for a given freshness token
(short-lived, keyed, evicted). The accumulator is what makes producing a frame
cheap; the data cache is what stops it being produced repeatedly within one
token. Neither can do the other's job: a keyed cache cannot accumulate across
keys, and a resource cache has no invalidation tied to the data changing.

- **A source that has *disappeared* must retire its rows too.** A removed path is
  not a changed path — it is never read, so a retire-on-read design leaves its
  rows in the frame for the process's lifetime. This is not hypothetical: 3.1.1
  documents staged-SLURM finalize relocating per-image parquets out of the
  directory, which is exactly this case. The accumulator therefore takes the
  **full effective path set** — the same one Section 10 computes the token from,
  with unconfirmed absences still in it — and diffs it internally against the
  read-set, handling additions, rewrites and removals in one place, rather than
  being handed a pre-computed "changed" list that cannot express a deletion.

  **It must be the effective set, not the literal directory listing**, or the
  debounce is bypassed on the ordinary path. Mid-run, new parquets land
  constantly; any one of them changes the token for its own reason and calls the
  loader. If the loader then receives the literal listing, a path that is only
  transiently absent is missing from it and gets retired on its *first* miss —
  the spurious drop the debounce exists to prevent, occurring precisely when the
  run is most active. The token and the loader's input are the same set,
  computed once per poll.

  **Pending state has to be cleared, too.** A path that reappears clears its
  first-absence timestamp, and a path that decodes successfully clears its
  unreadable mark. Without that, a later brief absence of the same path confirms
  instantly against a stale timestamp — a debounce that silently stops
  debouncing after the first flap.

  **A disappearance must persist for a full poll interval before it is retired
  and reported** — and the pending state must live *outside* the token-keyed
  cache, or it can never fire at all. On GCS-FUSE `os.replace` is emulated
  (Section 10), so a parquet being rewritten can plausibly be absent from a
  single listing; retiring on the first miss produces a spurious "points
  disappeared" banner followed by the rows returning, which trains the viewer to
  distrust the one signal that reports real data loss.

  **The trap, which a natural implementation walks straight into.** A removal
  changes the freshness token exactly once. Poll 1: token changes, the loader
  runs, the absence is noted, nothing is retired. Poll 2: the path set is
  identical to poll 1, so the token is unchanged, so `st.cache_data` serves a hit
  and **the loader is never called** — the second observation never happens and
  the rows are never retired. On the motivating path this is permanent: the
  quarantine happens at finalize, after which nothing else in the directory is
  written, so the token freezes forever. A two-poll rule implemented inside the
  cached loader converts a spurious-drop risk into guaranteed silent stale rows —
  strictly worse than retiring on the first miss.

  Absence tracking therefore belongs with the token computation, which already
  runs outside the cache on every poll, and the pending-absence state lives in
  the `st.cache_resource` accumulator, which is not token-keyed. The loader is
  called once the absence is confirmed, by including the confirmed removal in the
  token.

  **Confirm by elapsed time, not by observation count.** The accumulator is
  deliberately shared across viewers (above) while polling is per-session, so a
  count-based rule is not a debounce at all: five viewers at independent phase
  average about six seconds between observations on a thirty-second interval, and
  one user double-clicking Refresh confirms an absence in under a second — well
  inside the emulated-rename window the rule exists to ride out. The accumulator
  therefore records the **timestamp of first absence** and confirms only once a
  full poll interval has elapsed, which holds regardless of how many viewers are
  watching or how eagerly they refresh.

`data.py` stays framework-free through this: it exposes a pure
`accumulate(prior_frame, prior_read_set, effective_paths) -> (frame, read_set)`
which diffs, retires and re-reads internally. The argument is named for what it
must be: the debounce is resolved *before* the call, so `accumulate` itself
carries no timestamps and stays pure. `app.py` owns the `st.cache_resource`
handle, the lock, the eviction, and the pending-absence state.

**Windowed HDF reads become range requests.** h5py issues many small reads per
window at chunk granularity; over FUSE each is a round trip, and a single grid
crop can be tens of them. The design adds a **local HDF cache**: on first access,
copy the whole `.h5` to container-local disk, then serve every subsequent window
from the local copy. Per-image HDFs are read many times — three view modes,
repeated clicks, neighbouring colonies in one section — so the copy amortises.
The cache is an LRU over local disk with an explicit size cap, and it is a cache
in the strict sense: losing it costs latency, never correctness.

Live runs need care here: an HDF may be copied while still being written. The
copy is keyed on `(path, size, mtime_ns)` observed *before* the copy and
revalidated after; a mismatch discards and retries once.

**Required mount flags** (stat, type and file cache TTLs) are part of the
deployment artifact, pinned in documentation alongside the `docker run` line,
since the latency assumptions depend on them.

**Fallback position.** If measured latency is still unacceptable, move the active
run directory to a persistent disk or Filestore and keep GCS for archival. The
read path is isolated behind `images.py` and the frame loader, so the
substitution touches nothing else.

---

## 12. Error handling

| Condition | Behaviour |
|---|---|
| Path outside scope root | `ScopeViolation`; message names the root, not the attempted path |
| Path is not a run folder | Listed as a plain directory, not openable |
| Run has no measurements yet | Open successfully, report "no measurements written yet", keep polling |
| No metadata CSV available | Open; restrict grouping to available columns; explain what is missing |
| Chosen column absent from frame | Fail loudly at figure build via `_validate_input_columns` |
| Non-grid pipeline | Grid mode hidden, not shown-and-broken |
| HDF missing or lacking a layer | Overlay fallback, then placeholder (7.4) |
| `objmap` absent mid-staged-GPU | Layer selector reports the layer as not yet available for that image; other layers still selectable |
| Torn overlay read | Decode failure falls through to placeholder |
| Parquet fails to decode | Treated as in-flight: skipped this poll, retried next, not folded into the accumulator. A file that stays undecodable is reported in the banner rather than silently omitted (10) |
| Measured colonies dropped by the metadata join | Counted by anti-join and reported in the banner, never silent (3.2) |
| Run finalizes while someone is watching | No source switch occurs — the monitor reads per-image measurements in every run state — so no value changes *silently*. On staged-SLURM finalize points may still disappear; see the quarantine row below and 3.1.1 |
| HDF local-cache revalidation fails twice | Serve nothing rather than possibly-torn pixels: placeholder plus a "still being written" note. Never serve the unvalidated copy |
| No metadata CSV chosen | App opens; controls restricted to frame-intrinsic metadata; stated plainly (3.2) |
| Chosen CSV stops resolving | Reported, with controls falling back to frame-intrinsic metadata. Never a silent switch to a different CSV (3.2) |
| Metadata CSV shares no columns with the measurements | Distinct message: the CSV has no join key, so nothing was joined and nothing dropped. Never reported as "every colony dropped" (3.2) |
| Metadata CSV has duplicate join keys | Banner reports `duplicate_metadata_key_count`; colonies are multiplied by the join and the user is told why (3.2) |
| CSV rows matching no measured colony | Banner reports `unmatched_metadata_count`, worded "no measured colony **yet**" so it is true mid-run as well as at finalize (3.2) |
| Unassigned-group toggle enabled | Over-detections render as a pinned first page with x recovered image-wise (3.2, 6.2) |
| Time column not constant within an image | The toggle is disabled with an explanation rather than rendering an empty page (3.2) |
| Per-image parquets quarantined at staged-SLURM finalize | Points disappear once the absence has persisted a full poll interval; the banner reports the count rather than letting them vanish silently (3.1.1, 11.1) |
| Staged run before Stage 3 completes | "No measurements yet, still polling" — the normal state for most of a staged run, not an error (3.1) |

---

## 13. Testing

Framework-free units get ordinary unit tests; the Streamlit layer is thin enough
that almost nothing needs a browser.

1. **`customdata` alignment** — for a frame with known ordering, assert the
   `customdata` at trace position *i* is the key of the row that produced the
   `x`/`y` at position *i*. This is what makes every click trustworthy. Mutation
   to prove it can fail: reverse the sort in one of the two places.
2. **`ScopeRoot` containment** — `..` traversal, absolute paths, a symlink
   pointing outside the base, and a symlinked base.
3. **Grid window math** — golden fixture over synthetic member bboxes, including
   a section whose members are non-contiguous, and assert grid mode is absent
   when `Grid_RowMajorIdx` is missing.
4. **Strided plate dimming** — the dim rectangle at stride `k` covers the same
   image region as the full-resolution rectangle, within one pixel of `k`.
5. **The mirror is never read** — a run directory carrying a
   `deliverables/measurements.parquet` whose contents differ from the per-image
   parquets yields the per-image values, in every run state: mid-run, after
   finalize, and after `--restart` left a previous run's mirror behind. This is
   the test that keeps 3.1.1's decision from silently regressing into
   run-state detection, and it is the single most important one in this list —
   four separate predicates failed here.
6. **The aggregate is never read** — a directory containing
   `_dataset_aggregated.parquet` alongside individuals yields exactly the
   individuals' colonies: no double-count, and **no colony the current
   individuals do not contain**. Three mutations must each make it fail: an
   unfiltered `*.parquet` glob (double-counts), preferring the aggregate (hides
   up to 500 recent images), and unioning it (resurrects quarantined and
   restart-stale rows). This one test replaces four from earlier revisions, each
   of which guarded a rule that turned out to be wrong.
7. **Aggregate-only images stay hidden** — a directory whose aggregate holds an
   image with *no* individual parquet (the quarantine case) does not show that
   image. Mutation: union the aggregate for images lacking an individual, which
   is what image-level authority did.
8. **Incremental accumulation** — adding one parquet reads exactly one new file
    (assert on the read set, not wall-clock), and the frame equals a from-scratch
    concat.
9. **Accumulator invalidation on rewrite** — a parquet already folded into the
    accumulator, then rewritten with different content, is re-read and its old
    rows **retired**, so the colony count does not grow. Two mutations must each
    make it fail: key the read-set on path alone, and skip the retirement step.
    The second is the one an implementation is most likely to omit.
10. **Accumulator invalidation on removal** — a parquet already folded in, then
    *deleted* from the directory (the staged-SLURM quarantine of 3.1.1), has its
    rows retired once the absence has persisted for a **full poll interval**; a
    briefer absence retires nothing and reports nothing. Four mutations must each
    make it fail: retire on the first miss (spurious drops on FUSE); never retire
    at all (the deletion case a changed-paths-only contract cannot express);
    **put the confirmation inside the cached loader**, where the unchanged token
    means it is never called again and the rows are never retired — exercise this
    one with the directory otherwise static, the finalize case where the token
    freezes permanently; and **confirm by observation count rather than elapsed
    time**, exercised with two viewers polling at independent phase, which
    confirms a removal in a fraction of the interval. A fifth case, not a
    mutation: an unrelated parquet landing during an unconfirmed absence must not
    retire the absent path — the loader runs for the newcomer, and if it receives
    the literal listing rather than the effective set, the debounce is bypassed
    exactly when the run is busiest. Assert the absent path's rows survive that
    poll, then retire on the interval as normal.
11. **Undecodable parquet is surfaced, not swallowed** — a stably-undecodable
    parquet is reported in the banner rather than silently omitted for the
    process's lifetime. Mutation: skip-and-retry with no accumulator entry, with
    the directory otherwise static.
12. **Freshness token** — adding a per-image parquet changes it; **rewriting an
    already-folded parquet that is not the newest changes it** (the case a
    max-mtime token misses); a changed metadata CSV mtime changes it; touching
    nothing does not; and the appearance of a mirror does **not**, since the
    mirror is never read.
13. **Box clicks are ignored** — a box-trace selection payload carries no
    `customdata`; the handler leaves the current selection unchanged rather than
    raising or clearing it.
14. **Concat robustness** — parquets with divergent schemas concat via
    `diagonal_relaxed`; `MetadataImage_ImageName` is re-asserted from the file
    stem (including a UUID-named input); a missing dataset column is back-filled
    from the directory name.
15. **HDF local-cache validation** — an HDF whose size changes between the
    pre-copy and post-copy check is discarded and retried, not served; and when
    the retry also fails validation, a placeholder is served rather than the
    unvalidated copy.
16. **`how="inner"` join semantics** — metadata rows with no measured colony do
    not appear (no phantom flooding mid-run), and measured colonies whose image
    is absent from the CSV **are dropped but counted**, with the count reaching
    the banner. The drop is unavoidable under either `how` (3.2); the test
    guards that it is never silent.
17. **Pre-post labelling** — for a pipeline declaring `PostMeasurement` ops, the
    banner states the values are pre-post, and the plotted values match the
    per-image parquets rather than the mirror. Guards 3.3.1's scope boundary.
18. **Facet key precision** — a `MetadataCulture_Time` column of `pd.Timedelta`
    values differing only below microsecond resolution produces distinct facet
    keys via `_typed_group_key`, and the box builder orders facets identically
    to the line builder. Mutation to prove it can fail: substitute the public
    `canonical_group_key`.
19. **Metadata drop count** — the count the banner reports equals the number of
    measured rows the join removes, sourced from `prepare_metadata_join_keys`.
    Includes a case where a CSV column and its parquet counterpart differ in
    dtype, which a hand-rolled anti-join without the String cast would get wrong.
20. **CSV with no common columns** — reports "no join key", not "every colony
    dropped", even though `prepare_metadata_join_keys` returns
    `unmatched_measurement_count == measurements.height` for that input.
21. **Duplicate metadata keys** — a CSV with two rows for one join key fans the
    join out; assert the colony count multiplies **and** that
    `duplicate_metadata_key_count` reaches the banner. Also assert the mid-run
    case: duplicates among CSV rows matching *no* measurement are still counted
    (the number is whole-CSV), so the banner wording must not claim colonies were
    multiplied when none were. Guards the over-matching direction, which no other
    test covers.
22. **Staged run before Stage 3** — a staged output directory with HDFs and
    objmap sidecars but no per-image parquets renders the "no measurements yet"
    state rather than erroring, and keeps polling. This is the normal state for
    most of a target-deployment run (3.1).

23. **No control relies on a schema default** — with a frame whose columns are
    those of the validated run (no `MetadataCulture_Time`, no
    `MetadatasCondition_Media`), every control populates from the frame and the
    figure builds. Mutation: let `PlotColonyMetricOverTime`'s `time` or `groupby`
    default apply, which must fail at `_validate_input_columns`.
24. **Double-prefixed metadata columns are usable** — a frame carrying
    `Metadata_MetadataCondition_CarbonSource` (the real double-prefixed form,
    3.3.0.1) is offered as a grouping column like any other. Guards against the
    monitor filtering metadata by an assumed prefix shape.
25. **Non-schema metadata columns are offered** — `MetadataCulture_AgeHours`,
    `MetadataPlate_Well` and the other 15 columns that belong to no schema enum
    appear as selectable controls. Mutation: filter the offered columns to schema
    members, which leaves this run with five — of which only
    `MetadataGenetic_Strain` is groupable, so every other control empties.
26. **AppleDouble sidecars are excluded** — a measurements directory containing
    `._<stem>.parquet` beside each real file yields only the real files, with no
    unreadable-file report. Mutation: filter on `_` alone, which admits all of
    them and produces one false alarm per image.
27. **Over-detections are dropped by default and admissible by toggle** — a
    colony whose `(image, grid index)` is absent from the CSV does not appear in
    the default view, is included in the reported drop count, and appears as the
    `unassigned` group when the toggle is on. The fixture must use the real
    mechanism — an extra grid index within a matched image — not an unmatched
    image, which is the mode that does *not* dominate (3.2).
28. **The unassigned group is actually visible** — assert on **plotted x
    values**, not on the group's existence. An earlier draft would have passed
    with 42 points at `x = NaN`, drawing nothing, because the trace exists.
    Assert every unassigned point has a finite x recovered image-wise, that the
    page renders first, and that it is exempt from the chunk budget. Mutation:
    take x from the joined frame, which is null for exactly these rows.
29. **The toggle refuses rather than lying** — two fixtures, both of which must
    disable it: a time column that *varies* within an image, and a time column
    that is **blank for every row of one image**. The second passes a naive
    distinct-count check (null counts as a distinct value, so `n_unique == 1`)
    and lands its points back at `x = null`. Mutation for each: enable anyway,
    which yields a page of invisible points.
30. **The sentinel never touches an identity column** — with the toggle on,
    `MetadataExperiment_Dataset` retains its real value for unassigned rows, and
    a click on one resolves to a real HDF path. Mutation: fill every metadata
    column with the sentinel, which sends every unassigned click to a
    nonexistent file.
31. **The metadata CSV is chosen, never inferred** — four assertions, each
    guarding a rule that was tried and failed (3.2). A run whose
    `deliverables/metadata.csv` exists **pre-fills** the picker but is not
    adopted until confirmed. Nothing under `.phenotypic/` — including
    `job_metadata.json` — is read for metadata. Switching runs **clears** the
    selection, so run B is never labelled from run A's CSV. And a chosen CSV
    whose path stops resolving is reported, with controls falling back to
    frame-intrinsic metadata rather than to some other file. Mutations: adopt the
    pre-fill silently; read `job_metadata.json`; hold the selection in
    session state across a run switch.
32. **A changed CSV re-joins** — editing the chosen CSV mid-session changes the
    freshness token and produces a re-joined frame, including when the
    replacement shares the original's mtime. Mutation: put only the mtime in the
    token, not `(path, mtime_ns, size)`.
33. **Mid-run count wording** — with a run truncated to a third of its images,
    `unmatched_metadata_count` is large (322 of 480 on the validated run) and the
    banner does **not** claim those wells never grew. Guards the third instance
    of the finalize-versus-mid-run trap (3.2).

Per the repository's test-integrity rule, each must be shown to fail when the
behaviour it guards is reverted; a skip on a missing fixture is a failure, not a
pass. Tests 5–12, 16–28 and 32 have fixtures derivable from the validated run in
3.3.0 (29–31 need synthetic multi-run or machine-state states a single run
cannot supply) — prefer deriving them from its real column set over inventing a
plausible-looking one, since the invented ones in earlier revisions of this
document were what hid the schema-default problem.

---

## 14. Open questions

### Resolved

- **Metadata CSV availability** — the user chooses it in the sidebar, pre-filled
  from `deliverables/metadata.csv` when present. Auto-discovery was removed after
  four successive rules failed in five distinct ways (3.2), taking an upstream
  CLI change with it.
- **Overlay availability** — overlays exist during the run; `crop_colony`'s
  tiering is unchanged; no objmap-derived rendering (7.2).
- **Storage backend** — GCS-FUSE (11.1).
- **Strain cardinality** — all strains, progressive chunks (6.2).
- **Grid window** — union of member bboxes, with the tight-crop limitation
  documented (7.1).
- **Run handle** — `sdk_` path helpers, not `OutputRoot` (3.5).

### Open

**OQ-2 — Time join on a partial run.** *Largely closed by 3.3.0.* The validated
run is multi-timepoint and its keys line up per-colony; since per-image parquets
are atomic units, a partial run's key set is a strict subset of the finished
one's, so the join degrades by having fewer rows rather than by failing. What
remains open is narrower: whether the *time* column a user picks is dense enough
early in a run to plot usefully, which is a UX question rather than a join
question.

**OQ-6 — Expander render cost.** 6.2 assumes collapsed `st.expander` content is
still rendered server-side. If false on Streamlit 1.61, expanders beat
progressive chunking and 6.2 should be revised. Cheap to settle empirically.

**OQ-7 — Poll interval under FUSE.** 11.1 proposes a 30 s floor by reasoning,
not measurement. Real listing cost on a realistic run should set it.

**OQ-8 — Populating the grid interval columns.** 7.1 accepts the member-union
limitation for now. Whether to make the declared `Grid_*Interval*` columns real
upstream is a follow-up decision, not a blocker.

**OQ-10 — Stage progress instead of an empty page.** On staged-GPU runs the
monitor shows nothing until Stage 3 (3.1), which is most of the run. Surfacing
"Stage 2 of 3, 340/500 images" would be far better, but it means reading
orchestration state — a data source this design otherwise avoids entirely.

**The obvious source is not trustworthy.** On the validated run — finished, 30
images — `.phenotypic/progress/manifest.json` reports `total_images: 60,
completed: 30, pending: 30, is_complete: false`. The 60 is 30 real images plus 30
macOS AppleDouble `._*` sidecars counted as images (3.3), so a progress bar built
naively on it would sit at 50% forever on any exFAT volume. Settle the counting
question before building on the manifest.

**OQ-9 — gcsfuse rename atomicity.** Section 10's read-retry posture exists
because `os.replace` atomicity on the actual bucket configuration (flat versus
hierarchical namespace) is unconfirmed. If rename is atomic there, the parquet
retry path and the HDF revalidation in 11.1 are both unnecessary and can be
simplified away. Settle empirically against the real mount before implementing
either.

---

## 15. Summary of changes

**In `phenotypic`:**

1. Crop math extracted to a framework-free `sdk_` module, with a rectangle-based
   window function and a new strided reader (4.1, 7.3). The move also relocates
   `_label_map_to_rgb`, `_normalize_to_uint8` and `LayerName`, and drops the
   `gui._design` default binding — without which the extraction reintroduces the
   `phenotypic.gui` dependency it exists to remove.
2. `point_id` on `PlotMeasTimeSeries`, forwarded by `PlotColonyMetricOverTime`
   (4.2). No `_measurement_columns` change.
3. Grouping helpers `_group_rows` / `_group_pairs` / `_display_pairs` /
   `_typed_group_key` / `_canonical_group_key` exported (4.3). The last two are
   not redundant with the public `canonical_group_key` — they preserve
   nanosecond time precision that it truncates.
4. `src/phenotypic/schema/CLAUDE.md` records this dashboard as a consumer of the
   `METADATA` `MeasurementInfo` classes (4.5). Documentation only.

An earlier revision carried a fifth: copying the metadata CSV at run start. It
existed solely to feed the auto-discovery 3.2 deleted, and went with it.

Plus two sanctioned private imports that require no code change but are a
recorded coupling: `join_metadata` and `prepare_metadata_join_keys` (4.4).

**New, in `apps/monitor/`:** seven modules, a `pyproject.toml` and a `Dockerfile`
(Section 5).

**Explicitly not changed:** the `MetadatasCondition` schema typo (3.2);
`OutputRoot`, which this design stopped depending on rather than modifying
(3.5); and CLI completion-marker semantics, which an earlier revision proposed
changing before 3.1.1 removed the need to detect completion at all.

**Two out-of-scope defects surfaced during review** and are tracked separately:
the `MetadatasCondition` typo, and a suspected loss of trailing images from
`master_measurements.*` on SLURM runs whose image count is not a multiple of the
checkpoint interval. Neither is caused by this work; both were found by reading
the data flow closely enough to design against it.
