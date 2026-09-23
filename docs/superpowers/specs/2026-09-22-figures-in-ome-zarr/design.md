# Per-image figures stored inside the OME-Zarr store

- **Date:** 2026-09-22
- **Branch:** `claude/figures-ome-zarr-storage-6eba6f`
- **Status:** approved; revised 2026-09-22 after plan review (format set cut to
  `{plotly-json, png}`, no Chrome lane, failure-log fields, clarifications — see
  *Revision 2026-09-22* at the end)
- **Builds on:** `docs/superpowers/specs/2026-09-20-figure-backend-routing/design.md`
  (PR #236 — `@figure(backend=...)`, HTML+PNG publication, `.failures.jsonl`)

## Objective

A downstream dashboard must be able to display an image's figures **from the
`.ome.zarr` store alone** — no `deliverables/` tree, no run directory, no
pipeline file. Every per-image figure a pipeline produces is therefore written
into the image's store, in both `--mode full` and `--mode process`, and the
store becomes the single source of those figures: `deliverables/plots/` is
populated by copying out of the promoted store.

Compatibility is the **figure author's** burden, not the store's. The store is
format-neutral: it holds whatever renderings the author declared, each labelled
with a media type, and a consumer renders what it understands.

## Non-goals

- **Run-level figures.** `PlotMeas`, `PlotAnalysis` and `PlotQc` bindings are
  aggregate by construction and have no per-image store to live in. They keep
  publishing to `deliverables/plots/` exactly as today.
- **A serializer hook.** Authors choose from a closed set of formats (§2). A
  `to_store(fig) -> bytes` escape hatch is deferred until a real figure needs a
  format outside the set; the declaration is shaped so it can be added without
  a break.
- **Embedding the pipeline in the store.** The store records the pipeline by
  basename + SHA-256 only (see Background). Unchanged here.
- **Figures in intermediate or preview stores.** Builder preview stores
  (`save_intermediate_zarr`, `_image_io_handler.py:1428`) get no figures, and
  staged Stage 1 stores get none **except** those of the §3a bindings whose
  producer ran in Stage 1, which Stage 3 keeps in the same run folder (§1a,
  §3a). Stage 2 writes no store.
- **Figures for flat process exports.** `--process-format tiff` has no store,
  so it gets no figures.
- **Backfilling figures during `--mode migrate`.** Migrate never fabricates a
  figure.
- **`html` and `svg` store formats.** Cut at review: no shipped figure produces
  SVG, and a stored HTML page would load Plotly from a CDN, so it is not
  self-contained while `plotly-json` already is. Either can be added later
  without breaking a store (the set is closed per version, the descriptor is
  already media-typed).
- **Exercising Plotly PNG against real Chrome in CI.** Plotly PNG bytes are
  Kaleido's, not ours; our code path is tested with Chrome patched absent and
  present (§5).
- **Re-hashing figures during continuation validation.** See §1, *What the
  hashes bind*.

## Background — what a store holds today

Verified by running both modes on the synthetic 8×12 plate with an
`OtsuDetector` → `MeasureSize` + `MeasureShape` pipeline (2026-09-22).

**Full mode** — `results/<ds>/zarr/<stem>.ome.zarr/`:

```
zarr.json                 root: attributes.ome + attributes.phenotypic (written last)
OME/zarr.json, OME/METADATA.ome.xml
rgb/{0,1}/  rgb/labels/objmap/{0,1}/
gray/{0,1}/  detect_mat/{0,1}/  original/{0,1}/
tables/zarr.json  tables/measurements/{zarr.json,table.parquet}
```

`attributes.phenotypic` keys: `store_schema_version` (3), `phenotypic_version`,
`publication_protocol` (`root-last-immutable-v1`), `series`, `pyramid`,
`detect_mode`, `illuminant`, `gamma`, `metadata`, `image_class`, `labels`,
`work_id`, `grid`, `tables`, `provenance`. Not consolidated.

**Process mode** (`--layer rgb`) — `<out>/<ds>/<stem>.ome.zarr/`: root, `OME/`,
`rgb/{0,1}/` only. Same block minus `image_class`, `labels`, `tables`,
`work_id`. Consolidated.

**Pipeline provenance** lives at `attributes.phenotypic.provenance`, a schema-v2
journal: `applications[]`, each with `kind` (`full`/`process`/…), `pipeline:
{source_path, sha256}`, and `operations[]` (class, parameters,
`pipeline_step_path`). Process mode strips `applied_at_utc` /
`duration_seconds` so the store is byte-reproducible (process-mode spec
§2.3.3). The pipeline JSON itself lives outside the store
(`<out>/pipeline.json`, `deliverables/pipeline.json.pht-pipe`, or
`.phenotypic/pipeline.json.pht-pipe`).

**Precedent for non-OME payloads.** `tables/measurements/table.parquet` is a
Zarr-v3 group holding a non-Zarr file, described by
`attributes.phenotypic.tables`. `figures/` follows the same pattern.

**Where per-image figures come from today.** `PlotCoordinator.emit_image`
(`plotting/_pipeline/_coordinator.py`) calls each `PlotImage` binding's
`inspect(image, for_save=True)` and publishes to
`deliverables/plots/<binding>/<dataset>/<stem>-<hash>.{html,png}` (single
default page) or `.../<stem>-<hash>/` + manifest v2 (multi-page). Call sites:
full mode `_cli_process_single.py:348` (before `save_image_store`), measure
mode `_cli_process_single.py:452` (after the table replace), staged Stage 3
`_cli_staged_workers.py:583` (before `save_image_store`). Process mode emits no
figures.

**What a per-image figure takes as input.** `emit_image` selects bindings by
`isinstance(binding.plot, PlotImage)` over `pipeline.get_plots()`
(`_coordinator.py:349`) and always passes the `Image` explicitly. No binding
receives the measurement DataFrame or any run-level state. Two kinds of
provider exist:

- **Stateless image consumers** — `PlotDiagnostics`, `PlotDetectModes`
  (`plotting/_image_plots.py:35,115`), `_OrientationZonesReport`
  (`_measure_orientation_zones.py:3715`). They compute from the image's layers
  and raise without an image subject.
- **Measurers that are also plots** — `MeasureSymZones`,
  `MeasureOrientationZones`. The binding refers to the same instance as the
  `meas` slot. `_operate` leaves a private cache (weakref to the image,
  per-object intermediates, and a `model_dump_json()` signature); `inspect(image)`
  reuses it only when the image is the same object and the parameters are
  unchanged (`_measure_symzones.py:569`), and otherwise recomputes from
  `image.objmap` / `image.gray` — `MeasureOrientationZones` by calling
  `self.measure(image)` again (`_ensure_diagnostic_cache`,
  `_measure_orientation_zones.py:1970`).

So a stored figure is a function of **(the image's layers and objmap, the
operation's parameters)**; the operation cache is an optimisation, never an
input. This is what makes a figure-carrying store self-contained, and it gives
§4 a second obligation: the cache-hit render (full mode, right after
`measure`) and the recompute render (process mode, measure mode after a
reload) must produce identical bytes.

**Byte-determinism of candidate formats**, measured across two fresh processes
on the same machine (Chrome present):

| Format | Stable? |
|---|---|
| Plotly `fig.to_json()` | yes |
| Plotly PNG (Kaleido) | yes, per environment |
| matplotlib PNG, default `savefig` | yes (`Software` chunk is version-constant) |
| HTML via `to_html(include_plotlyjs="cdn")` | **no** — random div id |
| Plotly SVG | **no** — random clip-path ids |
| matplotlib SVG, even with `Date` dropped | **no** — random ids; `rcParams["svg.hashsalt"]` fixes it |

## §1 — Store layout and descriptor

### Layout

```
<stem>.ome.zarr/
├── zarr.json                        attributes.phenotypic.figures (root still written last)
├── tables/…
└── figures/
    ├── zarr.json                    empty Zarr v3 group document, as tables/zarr.json
    └── <binding_id>/                safe_path_component(binding.id)
        ├── zarr.json                empty Zarr v3 group document
        ├── <page_key>.plotly.json
        └── <page_key>.png
```

`<page_key>` is `safe_path_component(page.key)`; a bare figure is page
`default`. Two keys that sanitize to the same name (case-folded) get a stable
digest suffix, by the same rule the manifest writer already uses. File
extension per format: `plotly-json` → `.plotly.json`, `png` → `.png`. **The folder layout is storage only;
the descriptor is the contract.** A consumer never infers anything from a file
name.

### Descriptor — `attributes.phenotypic.figures`

```json
{
  "schema_version": 1,
  "bindings": {
    "sym": {
      "class": "MeasureSymZones",
      "pages": [
        {
          "key": "default",
          "label": null,
          "backend": "plotly",
          "metadata": {},
          "files": [
            {"format": "plotly-json",
             "media_type": "application/vnd.plotly.v1+json",
             "path": "figures/sym/default.plotly.json",
             "sha256": "<hex>"},
            {"format": "png", "media_type": "image/png",
             "path": "figures/sym/default.png", "sha256": "<hex>"}
          ]
        }
      ]
    }
  },
  "failed": [
    {"binding": "sym", "page": "default", "format": "png",
     "error": "PlotBackendUnavailable: Plotly PNG export needs Chrome (kaleido); install it with plotly_get_chrome"}
  ]
}
```

| Format | `media_type` |
|---|---|
| `plotly-json` | `application/vnd.plotly.v1+json` (the Jupyter mimebundle type) |
| `png` | `image/png` |

**Rules.**

- **Presence.** The `figures` key and the `figures/` group exist **iff** the
  pipeline carries at least one `PlotImage` binding. No binding → neither
  exists (the same convention as `labels`). Bindings present but all failed →
  the key exists with an empty `bindings` map and a populated `failed`.
- **Ordering.** `pages` are in `PlotOutput` order and `files` in the declared
  `store` order; both are lists, so their order is part of the data.
  `bindings` is a map: a consumer must not rely on its key order (a
  measure-mode rewrite serializes the root with sorted keys). Byte order is
  still deterministic for a given writer, which is what §4 needs.
- **`metadata`** is the page's `PlotPage.metadata`, which must be JSON-native.
  A page whose metadata does not serialize is a page failure (`format: null`),
  never a store failure. The copy-out rebuilds manifest v2 from it.
- **Failure granularity — the finest level available.**
  - One format fails (e.g. a declared `png` with no Chrome): one entry with
    that `page` and `format`; the page still lists the files that succeeded.
  - All of a page's formats fail: the page is omitted from `pages`, one entry
    per format in `failed`.
  - `inspect()` itself raises: one entry with `"page": null, "format": null`,
    and the binding is absent from `bindings`.
- **`error` text** is `f"{type(exc).__name__}: {msg}"` with CPython object
  addresses (`0x[0-9a-f]+`) replaced by `0x…`, so a deterministic failure
  produces deterministic bytes. A failure whose message varies for a real
  reason (different machine, different Chrome) is allowed to vary — that is a
  true property of the artifact, not noise.
- **Namespace.** Everything lives under `attributes.phenotypic`. Nothing is
  added to `attributes.ome`, `OME/zarr.json` or `METADATA.ome.xml`; `figures`
  is not registered in `ome.series`.
- **No `store_schema_version` bump.** The key is additive and optional; a
  reader tests for its presence.

### §1a — Run folders: `{date}-{pipeline hash}`, never wiped

*Added 2026-09-22, user decision. **This supersedes the single-generation
layout and descriptor above**, which now describe one run folder. It also
supersedes every "the whole `figures/` group is rebuilt" statement in §3.*

**Layout.** Figures live one level deeper, in a folder per run:

```
figures/
├── zarr.json                               empty group document
├── 2026-09-22-3f9a1c2b7e04/                one run folder
│   ├── zarr.json                           empty group document
│   └── <binding_id>/…                      exactly as §1 described
└── 2026-10-03-a07bc5e91d22/
    └── …
```

**Name.** The folder name is `{date}-{pipeline hash}`:

- `{date}` is the **UTC calendar date on which the run started**, as
  `YYYY-MM-DD`. There is one value for the whole run: every image, every
  stage, every SLURM task. It is recorded once as `figures_run_date` in the
  run's processing state (`state.config`, beside `pipeline_sha256`) when the
  state is created. **A resume on a later day reuses it** (user decision), and
  `--restart` / `--overwrite` start a new one. `--mode measure` has no run
  state of its own, so it uses the UTC date of its own invocation. The date is
  carried to the workers the way `--durable-writes` is, and is **not** part of
  `processing_configuration_digest`: a new day must not invalidate
  continuation. A run that crosses midnight, or staged Stage 1 and Stage 3 on
  different days, therefore stays in one folder. Programmatic callers
  (`save2zarr(figures=...)`) pass the run id explicitly.
- `{pipeline hash}` is the first 12 hex characters of the pipeline's sha256.
  This is the same digest the provenance journal records as `pipeline.sha256`
  for this run's application.

**Never wiped.** No write path deletes or rewrites another run's folder.

- Full mode, measure mode, staged Stage 3 and process mode **carry every other
  run folder across byte-for-byte**, together with its descriptor entry, from
  the store being replaced. This holds even when full or process mode rewrites
  the store from scratch (`--overwrite`, or a re-derived process run). In a
  measure-mode rewrite those folders are hard links. They are never written
  through; only this run's folder is cleared and rewritten, which is the
  existing hard-link rule.
- **The same run id (same day, same pipeline hash) replaces that one folder**,
  subject to §3a's keep rule.
- A pipeline with **no** `PlotImage` binding adds no run folder, and removes
  nothing. A table-only replace likewise changes no figure.

**Descriptor.** It is keyed by run:

```json
"figures": {
  "schema_version": 1,
  "runs": {
    "2026-09-22-3f9a1c2b7e04": {
      "date": "2026-09-22",
      "pipeline_sha256": "<full hex>",
      "bindings": { ... as §1 ... },
      "failed": [ ... as §1 ... ],
      "unavailable": ["cal"]
    }
  }
}
```

- `path` values are store-relative and include the run folder, e.g.
  `figures/2026-09-22-3f9a1c2b7e04/sym/default.plotly.json`.
- `unavailable` lists the §3a bindings that could not be drawn in this run and
  had nothing to keep. It is always present, and may be empty.

**Which run is "current" is left to the consumer** (user decision). There is
no `latest` pointer. A consumer chooses by `date`, by `pipeline_sha256`, or
both. The copy-out (§3) publishes **this run's folder** to
`deliverables/plots/`, because the deliverables tree belongs to the run that
wrote it.

**Process mode determinism** (user decision). Process stores carry run folders
too. §4's byte identity therefore holds for identical runs **on the same UTC
day**. The same image and pipeline run on another day produce a different run
folder name, and so different bytes.

`schema_version` stays `1`: nothing has been published under the flat layout.

### What the hashes bind

The root carries each file's `sha256`, so the image's completion record — which
digests the root — binds those hashes. **It does not re-read figure bytes.**

- The copy-out (§3) verifies every file against its `sha256` before copying.
  A mismatch is a recorded failure and the file is not copied.
- Any consumer may verify the same way.
- Continuation validation does **not** re-hash figures. Figures are
  best-effort, not measurement authority; a corrupted figure must not force an
  image to be re-run.
- `file_sha256` (`_cli_failure_tracker.py:85`), which digests a store *input*'s
  entire tree, covers `figures/` automatically. That is intended: a store whose
  figures differ is a different input.

## §2 — Author declaration on `@figure`

### Signature

```python
StoreFormat = Literal["plotly-json", "png"]

def figure(
    *,
    title: str,
    backend: Literal["plotly", "mpl"],
    store: tuple[StoreFormat, ...] | None = None,   # new
    section: str = "default",
    controls: dict[str, Control] | None = None,
    description: Any = None,
    primary: bool = False,
) -> ...
```

`FigureSpec` gains `store: tuple[StoreFormat, ...]`, resolved at decoration.

### Validation — `TypeError` at class-definition time

- `plotly-json` with `backend="mpl"` (matplotlib has no native form of it).
- An unknown format name, or a duplicate.
- A bare `str` (`store="png"`), which would otherwise iterate as characters.
- **`store=()`**: rejected. An image figure must store at least one format.
  Because `deliverables/plots/` is a copy-out of the store (§3), an opted-out
  figure would appear nowhere; a figure the author does not want stored should
  not be bound.

### Defaults (`store` omitted / `None`)

| Backend | Default | Why |
|---|---|---|
| `plotly` | `("plotly-json",)` | Lossless, interactive via any Plotly consumer, Chrome-free, byte-stable. Size follows the figure: a trace-only figure is KB, an image-backed one (`px.imshow(binary_string=True)`, as the zone measurers use) embeds a full-resolution PNG data URI and is MB |
| `mpl` | `("png",)` | The only lossless-enough native form without extra dependencies; byte-stable |

**Consequence, accepted:** a default Plotly figure no longer yields a PNG in
`deliverables/plots/`, even where Chrome exists (§3 never renders). An author
who wants PNGs declares `"png"`; on a Chrome-less machine that declared PNG is
then a genuine recorded failure, because the author asked for it.

### Which declaration applies

A `PlotImage` binding's store formats are its **primary spec's** `store`
(`PhtPlot.inspect` renders `_primary_spec()`, `_pht_plot.py:419,445`).

A provider that **overrides `inspect()`** has no spec for its pages. Each page
falls back to the backend default, sniffed with `figure_backend_of`
(`abc_/plotting/_output.py:14`). An unrecognised backend is a per-page failure.
This is the documented escape hatch for multi-page and hand-built output; the
deferred strict-backend work (`2026-09-20-figure-backend-routing/DEFERRED.md`)
would later give such pages a declaration channel.

### Serializers

One private module, `plotting/_pipeline/_store_formats.py`, with one function
per format, each `(figure, *, binding_id, page_key) -> bytes`, each
deterministic by construction:

| Format | Serializer |
|---|---|
| `plotly-json` | `fig.to_json()`, UTF-8 (uids removed by default) |
| `png` | Plotly: `plotly.io.to_image(fig, format="png")` (Chrome-gated via `chrome_available()`; absent Chrome raises `PlotBackendUnavailable`); mpl: `fig.savefig(buf, format="png")` |

- All plotting imports stay inside function bodies (lazy-import guards:
  `tests/unit/ci/test_startup_imports.py`, `test_deferred_imports.py`).

## §3 — Write path by mode, and copy-out

`emit_image` is split into two steps.

### Step 1 — build (in memory)

`build_image_figures(pipeline, image) -> StoredFigures | None` (a module
function in `plotting/_pipeline/`; `None` when the pipeline has no `PlotImage`
binding). Process mode has no `plots_base`, so there is no coordinator to hang
it on; the other modes call the same function:

- For each `PlotImage` binding: `inspect(image, for_save=True)` →
  `normalize_plot_output` → for each page, each resolved format → serializer
  bytes. Figures are closed after serialization (`FigureAdapter.close`).
- Failures are captured into `StoredFigures.failed` at the finest level (§1).
  `PlotPublicationBlocked` still propagates, as in every handler today.
  Everything per binding — `inspect`, normalisation, format resolution,
  serialization — is inside that binding's failure boundary, so no figure
  error can fail the image. An `inspect()` that returns `None` is a failure
  (`page: null, format: null`), not an absence: absence means "not
  configured".
- **Writes nothing.** Safe to call anywhere before a store transaction.

`StoredFigures` is an immutable value: an ordered mapping
`binding_id → (class_name, pages → (key, label, backend, ordered (format, bytes)))`
plus `failed`. It is the only thing the store writer and the descriptor
builder receive.

### Step 2 — store write

`write_image_figures(part, stored) -> dict` (new, `sdk_/`, beside
`write_image_tables`) writes `figures/` into a `.part` directory and returns the
descriptor fragment; `apply_image_figures_attributes(phenotypic, fragment)`
sets or removes the `figures` key. Called from `_write_store_part`
(`_image_io_handler.py`) **after the tables block (`:1381`) and before the root
(`:1391`)**, so figures are inside the root-last transaction.

### Step 3 — copy-out (after promotion)

`publish_store_figures(store_path, plots_base, *, dataset, image_stem, …)`:

- Reads the **promoted** store's `figures` descriptor; for each file verifies
  `sha256`, then copies it to today's deliverables path:
  - single `default` page → `<binding>/<dataset>/<stem>-<hash>.<ext>`
    (`_image_output_stem`, `_coordinator.py:530`);
  - multi-page → `<binding>/<dataset>/<stem>-<hash>/<page>.<ext>` + manifest v2.
- For each stored `plotly-json`, additionally writes
  `<…>.html` rendered from it (`plotly.io.from_json` → `write_html` with
  `include_plotlyjs=<relpath to the hoisted plotly.min.js>`), preserving the
  figure-routing rule that a Plotly figure is always browsable as HTML in
  `deliverables/`. PNGs are copied only if stored; copy-out never renders a
  PNG.
- A multi-page directory is published under its `.publication.lock`, like
  `publish_plot_output`, and its manifest is committed by the writer's own
  manifest commit. `renderers` keeps its documented *capability* meaning
  (`html: available` for Plotly pages); outcomes stay in `files`, `partial`
  (formats that failed for a published page) and `failed`.
- Each descriptor `failed` entry becomes one `.failures.jsonl` line through
  `record_plot_failure` (`_failures.py:41`), `lifecycle="image"`. The stored
  `error` is recorded **verbatim**; the page and format go in two new optional
  JSONL fields, `page` and `format`. `plot_class` comes from the pipeline's
  binding of that id (the descriptor records no class for a failed binding).
  The publication guard is checked before the first record is written.
- Stale-sibling removal keeps today's rule: only a page this pass published has
  its leftover renderings removed.
- **Best-effort.** A copy-out error is recorded, never raised, never fails the
  image. Only a refused guard (`PlotPublicationBlocked`) propagates. It runs **before** the image's completion record is published, so a
  crash between promotion and copy-out re-runs the image.

### By mode

| Mode | Build | Store write | Copy-out |
|---|---|---|---|
| Full — `_cli_process_single.py` | after `apply_and_measure`, where `emit_image` runs today | `save_image_store(…, figures=)` → `save2zarr` → `_save_store` → `_write_store_part` | yes |
| Staged Stage 3 — `_cli_staged_workers.py` | same, with `plan.post_pipeline` | same | yes |
| Measure — `_cli_process_single.py` | after `measure()`, **moved before** the table replace | same root-last transaction as the tables: `replace_image_tables` → `_rewrite_store_tables` (`_measurement_tables.py:632`) also clears the part's copied `figures/` (alongside `tables/`) and writes the new one | yes |
| Process, `--process-format zarr` — `_cli_process_only.py` | after `pipeline.apply`, before the provenance status is closed (**new**) | `write_process_only_layer(…, figures=)` → `_save_store`, inside the consolidated part | **no** — no `deliverables/` |
| Process, `--process-format tiff` | not built | — | — |

**Process mode builds every `PlotImage` binding, measurer-backed ones
included.** Process mode runs only `pipeline.apply()`, so no measurer's cache is
ever filled and each measurer-backed figure recomputes inside `inspect()` —
for `MeasureOrientationZones` that is a full `measure(image)` per image, whose
result is used for the figure only and never written as a table. Accepted: a
user who put a measurer's plot in the pipeline asked for that figure, and a
process store should carry the same figures as a full store of the same
pipeline. Without a detector in `apply()`, `objmap` is empty and the figure is
the provider's own "no objects" rendering — a real figure, not a failure.

**Measure mode semantics.** The whole `figures/` group is rebuilt from the
current pipeline's `PlotImage` bindings. A binding removed from the pipeline
disappears from the store; a new one appears. A store's figures and its table
always come from the same pipeline. `replace_embedded_measurement_table` (the
migrate-only path) leaves `figures/` untouched — it hard-links it across like
any other unchanged file. The one exception to "rebuilt" is a binding whose
input no longer exists (§3a): that binding is carried across.

### §3a — Figures whose input exists only where the operation applied

*Added 2026-09-22, user decision, with `CalibrateColorRpcc`'s tile overlay
(PR #238) as the first case.*

The Background rule, that a figure is a function of the image's layers and
the operation's parameters, holds for measurers. It does **not** hold for a
figure drawn from state that only `apply()` can produce. The calibration
overlay draws the **as-shot** checker pixels, and `apply()` overwrites them
with the corrected image. After that, no process can redraw the overlay from
the image. It can only be drawn in the process that ran `apply()`, and only
for the image that `apply()` ran on.

**Signal.** Such a provider raises `FigureInputUnavailable`, a new public
`RuntimeError` subclass in `phenotypic.abc_.plotting`, from `inspect()` when
it cannot draw for the image it was given. That happens when no `apply()` ran
in this process, or when the last `apply()` ran on a different image. It is a
statement about where the figure can be drawn, not a failure of the figure.

**Build.** *(Revised with §1a run folders; user decision: "leave it where it
is".)* The signal never makes a figure disappear, and it never copies one
**between** run folders.

- **The run folder being written already holds this binding.** This happens
  when staged Stage 1 wrote it earlier in the same run, or when a same-day
  rerun of the same pipeline wrote it. The binding's existing entry and files
  in that folder are **kept** unchanged: its descriptor entry (pages, labels,
  backends, metadata, and the page-level `failed` entries naming it) and its
  files. Each file is verified against its recorded `sha256`. A mismatch is a
  binding-level failure, never a silent keep.
- **Otherwise.** The binding is listed in the new run folder's `unavailable`
  list (§1a), not in `failed`: it is a statement about where the figure can
  be drawn. The figure itself stays wherever an earlier run folder already has
  it. A consumer that wants it looks back through the run history.

| Path | Result for a §3a binding |
|---|---|
| Full mode, process mode | `apply()` ran in this process on this image; the provider draws into this run's folder. |
| Measure mode | Usually a new run folder (a different pipeline hash, or another day), where the binding is `unavailable`. The earlier folder holding the overlay is untouched. |
| Staged Stage 1 | Stage 1 runs the pre-detector operations. It builds figures **only for bindings whose producer ran in Stage 1**, into this run's folder. Other bindings (the measurers) are left to Stage 3; building them on a store with no objmap would be wasted work. |
| Staged Stage 3 | The same run, so the same run folder (§1a: one `{date}` per run). Stage 1's overlay is kept; Stage 3 adds the rest. |

A kept figure is byte-identical to the one Stage 1 or the earlier same-day run
wrote. §4's "cache parity" does not apply to these providers, because they
have no recompute path by construction.

**`CalibrateColorRpcc`** becomes a `PlotImage`. Its `inspect(image)` renders
`show_tiles()` (matplotlib, so the default store format is `png`). It raises
`FigureInputUnavailable` when there is no calibration record, or when the
record was built from a different image. It holds a weak reference to the
image `_operate` saw, never the image itself (abc_ rule on image caches).
`inspect` is an undecorated override. The approved overlay look is kept
exactly: no theme context wraps it, and its page backend (`mpl`) selects the
format.

### Continuation

- **Process mode:** bump `PROCESS_LAYER_SEMANTICS_REVISION` 2 → 3
  (`_cli_failure_tracker.py:205`), with a changelog line: *"process-mode stores
  now carry the pipeline's per-image figures."* A process tree resumed across
  the upgrade re-derives instead of mixing stores with and without figures.
  As documented, this also invalidates in-flight `tiff` continuations —
  deliberate; invalidating too much is safe.
- **Full mode:** no revision. A full run resumed across the upgrade with the
  same pipeline reuses figure-less stores. A store with no `figures` key whose
  recorded pipeline has image bindings reads as *"produced before this
  feature"*; `--overwrite` fills it in. Accepted over adding a full-mode
  revision, which would re-process every in-flight run — including GPU
  re-inference on staged runs — for output that is decorative relative to the
  measurements.

## §4 — Determinism contract

Process-mode stores stay **byte-identical across identical runs**, now
including `figures/`. "Identical" means same inputs, pipeline, PhenoTypic
version **and environment**. A difference that comes from the environment — a
different Chrome changing Kaleido PNG bytes, or Chrome being absent so a
declared PNG becomes a `failed` entry — is a true property of the artifact and
is meant to show up in the store's bytes and in `file_sha256`. Only
*stochastic* variation (random ids, timestamps, memory addresses) is removed.

Full-mode stores keep their existing non-reproducible journal timestamps; their
`figures/` bytes are nonetheless produced by the same deterministic serializers.

**Cache parity.** For a given image and pipeline, a measurer-backed figure
rendered from the operation cache (full mode, immediately after `measure`) and
one rendered by recomputation (process mode; measure mode after a reload) must
be byte-identical. Otherwise the same image's figure would depend on which
mode wrote the store, and the background claim that the cache is never an
input would be false. A provider that cannot meet this is a bug in the
provider, not a tolerance in the contract.

## §5 — Testing

Each new test must be shown to fail when the bug it guards is reintroduced
(user rule: a test must be able to fail).

| Test | Guards |
|---|---|
| Full-mode store with a `PlotImage` binding has `figures/` + per-binding groups, the descriptor, and `sha256` equal to each file's bytes | §1 |
| Pipeline with no `PlotImage` binding → no `figures` key, no `figures/` group | §1 presence |
| An independent Zarr v3 reader (zarr-python, not the writer's own reopen) opens `figures/` as groups; `OME/zarr.json` `series` and `METADATA.ome.xml` are unchanged | §1 namespace; store contract step 4 |
| `store=` validation: `plotly-json` with `mpl`, unknown name, duplicate, `()`, a bare `str` → `TypeError` at class definition | §2 |
| `store` omitted → `("plotly-json",)` for plotly, `("png",)` for mpl | §2 defaults |
| A provider overriding `inspect()` stores each page in its backend default | §2 fallback |
| `plotly-json` and mpl `png`, each run in two separate processes, produce identical bytes. Plotly `png` is tested with Chrome patched absent (a recorded `PlotBackendUnavailable`) and present (the Kaleido call is reached); its bytes are Kaleido's and are not pinned by us | §2, §4 |
| A page whose `metadata` is not JSON-native is a recorded page failure; the store still publishes (all modes, including the measure-mode root rewrite) | §1 metadata |
| An `inspect()` returning `None` is recorded as a failure | §3 step 1 |
| The existing process-mode byte-identical-store test, extended to a pipeline with an image binding | §4 |
| A figure whose `inspect()` raises → store still published; `failed` entry with `page: null, format: null`; binding absent | §1 failure |
| A declared `png` with `chrome_available()` false → per-format `failed` entry; the page's other files present | §1 failure |
| `error` text with an object address is normalised | §1 |
| Copy-out lands files at today's `deliverables/plots/` paths (flat and multi-page) | §3 |
| Copy-out writes HTML from stored `plotly-json`, and its `plotly.min.js` `src` resolves from each depth | §3 |
| A tampered stored figure (bytes ≠ `sha256`) is recorded as failed and not copied | §1 hashes, §3 |
| Descriptor `failed` entries become `.failures.jsonl` lines with the stored `error` verbatim, `page`/`format` fields, and the binding's real `plot_class` | §3 |
| Measure mode: removing a binding removes it from the store; adding one adds it; pixel arrays remain hard links (inode check) | §3 measure |
| Measure mode: figures are written before the root (no store write outlives the publication that certifies it) — asserted at `promote_store` time, not from the final tree | §3 ordering |
| Measure mode: a same-name rebuild never writes through a hard link into the live store (a held descriptor on the old file still reads the old bytes) | §3 measure |
| Process mode: `zarr` export carries `figures/`; `tiff` export produces no figures | §3 process |
| Process mode with a measurer-backed binding (`MeasureSymZones`) stores its figure, and writes no measurement table | §3 process |
| Cache parity: for `MeasureSymZones` and `MeasureOrientationZones`, the stored bytes from `inspect()` right after `measure()` equal those from `inspect()` on a freshly loaded copy of the same image | §4 cache parity |
| `PROCESS_LAYER_SEMANTICS_REVISION` bump changes the process work id | §3 continuation |
| A pre-feature store (no `figures` key) remains valid to every reader and to `--mode migrate` | §1 optional |
| Startup/deferred import guards still pass | §2 lazy imports |

Regression surface — derive mechanically from importers, per the project rule:
`tests/unit/plotting/`, `tests/unit/abc_/plotting/`,
`tests/integration/plotting/`, store/table tests under `tests/unit/sdk_/` and
`tests/unit/cli/` touching `_save_store`, `replace_image_tables`,
`write_process_only_layer`, and the three `emit_image` call sites. Use the
**`run-phenotypic-test`** skill; the full suite is a Slurm job, run once at the
end.

No logic-validation script: this design rests on no numeric invariant. The
determinism claims are pinned by the tests above.

## §6 — Documentation

- `.claude/skills/working-with-ome-zarr/SKILL.md` — store-contract table gains a
  *Per-image figures* row; describe `figures/` and its descriptor.
- `src/phenotypic/_cli/CLAUDE.md` — write path per mode; copy-out; continuation.
- Root `CLAUDE.md` — process-mode bullet: stores carry figures; revision 3.
- `src/phenotypic/abc_/CLAUDE.md` — `@figure(store=...)` convention.
- `docs/source/extending/pages/custom_plotter.md` — `store=`, the format set,
  media types, determinism, defaults, that `plotly-json` for an image-backed
  figure is MB-sized, and that `deliverables/plots/` is a copy-out of the
  store.
- `docs/source/how_to/pages/zarr_storage.md` — the `figures/` group and how a
  consumer reads it (descriptor → media type → file).

## Blast radius

- **Additive / not breaking:** `@figure(store=)` is optional; the `figures` key
  and group are optional; no `store_schema_version` bump.
- **Behavioural:**
  - `deliverables/plots/` per-image output is now a copy-out; it gains
    `.plotly.json` files, and default Plotly figures lose their PNG even where
    Chrome exists.
  - Measure mode's per-image plot emission moves before the table replace.
  - Process-mode stores gain `figures/`, and process mode now calls
    `PlotImage.inspect()` (new work per image). A measurer-backed binding
    re-runs its measurement inside `inspect()` there, so a process run with
    `MeasureOrientationZones` bound pays roughly one extra measurement per
    image.
- **Continuation:** in-flight process trees re-derive once (revision 3). Full
  runs resumed across the upgrade keep figure-less stores until `--overwrite`.
- **Unchanged:** aggregate/QC/analysis figures; `attributes.ome` and OME-XML;
  pixel arrays; the completion-record format; intermediate and preview stores.

## Decisions on record

| Decision | Choice | Rationale |
|---|---|---|
| Purpose | Self-contained store (copy travels with the image) | A dashboard must render from the `.ome.zarr` alone |
| Format ownership | The figure author declares; the store is format-neutral | Compatibility burden on the implementor; the store only labels media types. Reverses, for stored figures, the figure-routing spec's "format lives in the publication layer" |
| Declaration shape | `@figure(store=...)` over a closed set, per-backend default | Declarative, validated at class definition, visible in the schema; hook deferred |
| `store=()` | Rejected | Copy-out makes an opted-out figure vanish everywhere |
| Process-mode reproducibility | Figures inside the byte-identical contract; serializers remove stochastic variation | "The store's bytes are its identity" stays unqualified; environment-driven differences are real signal |
| Measure mode | Rebuild `figures/` in the table transaction | A store's figures and table never disagree |
| Process mode, measurer-backed bindings | Built like every other `PlotImage` binding (recompute inside `inspect()`) | Same figures as a full store of the same pipeline; the user bound the plot, so they asked for it |
| Figure inputs | Image layers + objmap + operation parameters; the operation cache is an optimisation only, pinned by a cache-parity test | Keeps the store self-contained and the figure independent of which mode wrote it |
| Figure failure | Publish the store; record in descriptor `failed` | Dashboard can tell "not configured" from "failed"; one bad figure never kills a run |
| Write architecture | Single sink into the store; `deliverables/plots/` copied out after promotion | One source of truth, same pattern as embedded tables → master |
| Copy-out content | Stored files verbatim + HTML generated from stored `plotly-json` | Keeps deliverables browsable without re-rendering from figure objects |
| Default Plotly store | `("plotly-json",)` | Chrome-free and stable; declared PNGs fail honestly |
| Format set | `{plotly-json, png}` (review, 2026-09-22) | `svg` unused by any shipped figure; stored `html` not self-contained and collided with the generated deliverable HTML |
| Chrome in CI | No lane | Plotly PNG bytes are Kaleido's; our path is tested with Chrome patched |
| Failure log | `error` verbatim + optional `page`/`format` fields; class from the pipeline | Keeps `.failures.jsonl`'s "starts with the exception class" contract; no descriptor change |
| Continuation | Process revision → 3; full mode unchanged | Process re-derivation is cheap; full-run re-derivation is not |
| Continuation validation | Does not re-hash figures | Figures are not measurement authority |

## Revision 2026-09-22 (after plan review)

Decided by the user after the plan review and the simplicity review
(`docs/superpowers/reports/2026-09-22-figures-in-ome-zarr/`):

1. Format set cut to `{plotly-json, png}` (Non-goals; §1; §2).
2. No Chrome CI lane; Plotly PNG tested with Chrome patched (§5).
3. `.failures.jsonl`: stored `error` verbatim, new optional `page`/`format`
   fields, `plot_class` from the pipeline (§3).

Clarifications made while planning, not changes of intent:

4. Descriptor pages carry `metadata`; a non-JSON value is a page failure (§1).
5. `build_image_figures(pipeline, image)` is a module function (§3).
6. `bindings` key order is not part of the contract (§1 Ordering).
7. Page filenames that collide after sanitizing get a stable digest suffix (§1).
8. Every per-binding step is inside the failure boundary; `inspect() -> None`
   is a failure (§3).
9. Copy-out holds `.publication.lock` for a manifest directory, reuses the
   writer's manifest commit, keeps `renderers` a capability field (§3).
10. Process mode builds figures before closing provenance status (§3).
11. "KB-sized" corrected: image-backed Plotly JSON is MB-sized (§2).

Added 2026-09-22 after the calibration overlay (PR #238) merged into main:

12. §3a: figures whose input exists only where the operation applied
    (`FigureInputUnavailable`, carried across in measure mode and staged
    Stage 3, built by Stage 1 for its own operations). User decision: keep the
    stored overlay rather than drop it. `CalibrateColorRpcc` is the first
    provider.
13. §1a: run folders `{date}-{pipeline hash}` (UTC start date of the CLI
    invocation; first 12 hex of the pipeline sha256), never wiped by any mode.
    The descriptor is keyed by run, and a consumer chooses the current run.
    Process-mode byte identity holds within a UTC day. User decisions.
    §3a is revised to match: no copy between run folders. A §3a binding is
    kept within its own run folder (staged Stage 1 → Stage 3, or a same-day
    rerun), and is otherwise listed as `unavailable`. The `KEEP_FIGURES`
    sentinel is retired: with nothing ever wiped, `figures=None` means "add
    no run folder" and is safe as the default.
