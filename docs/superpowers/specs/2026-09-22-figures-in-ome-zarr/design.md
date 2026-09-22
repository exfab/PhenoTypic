# Per-image figures stored inside the OME-Zarr store

- **Date:** 2026-09-22
- **Branch:** `claude/figures-ome-zarr-storage-6eba6f`
- **Status:** design approved in conversation, awaiting written-spec review
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
- **Figures in intermediate or preview stores.** Stage 1/2 staged stores and
  builder preview stores (`save_intermediate_zarr`, `_image_io_handler.py:1428`)
  get no figures.
- **Figures for flat process exports.** `--process-format tiff` has no store,
  so it gets no figures.
- **Backfilling figures during `--mode migrate`.** Migrate never fabricates a
  figure.
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
        ├── <page_key>.html
        ├── <page_key>.png
        └── <page_key>.svg
```

`<page_key>` is `safe_path_component(page.key)`; a bare figure is page
`default`. File extension per format: `plotly-json` → `.plotly.json`, `html` →
`.html`, `png` → `.png`, `svg` → `.svg`. **The folder layout is storage only;
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
     "error": "ChromeNotFoundError: <message>"}
  ]
}
```

| Format | `media_type` |
|---|---|
| `plotly-json` | `application/vnd.plotly.v1+json` (the Jupyter mimebundle type) |
| `html` | `text/html` |
| `png` | `image/png` |
| `svg` | `image/svg+xml` |

**Rules.**

- **Presence.** The `figures` key and the `figures/` group exist **iff** the
  pipeline carries at least one `PlotImage` binding. No binding → neither
  exists (the same convention as `labels`). Bindings present but all failed →
  the key exists with an empty `bindings` map and a populated `failed`.
- **Ordering.** `bindings` is keyed by binding id in pipeline order; `pages` in
  `PlotOutput` order; `files` in the declared `store` order. Deterministic
  ordering is part of the reproducibility contract (§4).
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
StoreFormat = Literal["plotly-json", "html", "png", "svg"]

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

- `plotly-json` or `html` with `backend="mpl"` (matplotlib has no native form
  of either; consistent with the figure-routing spec's "no HTML for
  matplotlib").
- An unknown format name, or a duplicate.
- **`store=()`**: rejected. An image figure must store at least one format.
  Because `deliverables/plots/` is a copy-out of the store (§3), an opted-out
  figure would appear nowhere; a figure the author does not want stored should
  not be bound.

### Defaults (`store` omitted / `None`)

| Backend | Default | Why |
|---|---|---|
| `plotly` | `("plotly-json",)` | Lossless, interactive via any Plotly consumer, Chrome-free, byte-stable, KB-sized |
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
| `plotly-json` | `fig.to_json()`, UTF-8 |
| `html` | `plotly.io.to_html(fig, include_plotlyjs="cdn", full_html=True, div_id=<stable id derived from binding_id + page_key>)` |
| `png` | Plotly: `plotly.io.to_image(fig, format="png")` (Chrome-gated via `chrome_available()`); mpl: `fig.savefig(buf, format="png")` |
| `svg` | mpl: `savefig(format="svg", metadata={"Date": None})` under `rc_context({"svg.hashsalt": <stable>})`; Plotly: `to_image(format="svg")` with generated ids rewritten to stable ones |

- Store HTML uses the **CDN** script: a store cannot reference the run's
  hoisted `plotly.min.js`, and embedding 4.8 MB per figure is rejected.
  Declaring `html` means the author accepts that viewing needs network.
- **Plotly SVG is provisional.** If its ids cannot be pinned reliably, `svg`
  becomes mpl-only (a class-definition `TypeError` for Plotly) rather than
  shipping non-deterministic bytes.
- All plotting imports stay inside function bodies (lazy-import guards:
  `tests/unit/ci/test_startup_imports.py`, `test_deferred_imports.py`).

## §3 — Write path by mode, and copy-out

`emit_image` is split into two steps.

### Step 1 — build (in memory)

`PlotCoordinator.build_image_figures(image) -> StoredFigures`:

- For each `PlotImage` binding: `inspect(image, for_save=True)` →
  `normalize_plot_output` → for each page, each resolved format → serializer
  bytes. Figures are closed after serialization (`FigureAdapter.close`).
- Failures are captured into `StoredFigures.failed` at the finest level (§1).
  `PlotPublicationBlocked` still propagates, as in every handler today.
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
  `<…>.html` rendered from it (`plotly.io.from_json` → `to_html` with
  `include_plotlyjs=<relpath to the hoisted plotly.min.js>`), preserving the
  figure-routing rule that a Plotly figure is always browsable as HTML in
  `deliverables/`. PNGs are copied only if stored; copy-out never renders.
- Each descriptor `failed` entry becomes one `.failures.jsonl` line through
  `record_plot_failure` (`_failures.py:41`), `lifecycle="image"`.
- Stale-sibling removal keeps today's rule: only a page this pass published has
  its leftover renderings removed.
- **Best-effort.** A copy-out error is recorded, never raised, never fails the
  image. It runs **before** the image's completion record is published, so a
  crash between promotion and copy-out re-runs the image.

### By mode

| Mode | Build | Store write | Copy-out |
|---|---|---|---|
| Full — `_cli_process_single.py` | after `apply_and_measure`, where `emit_image` runs today | `save_image_store(…, figures=)` → `save2zarr` → `_save_store` → `_write_store_part` | yes |
| Staged Stage 3 — `_cli_staged_workers.py` | same, with `plan.post_pipeline` | same | yes |
| Measure — `_cli_process_single.py` | after `measure()`, **moved before** the table replace | same root-last transaction as the tables: `replace_image_tables` → `_rewrite_store_tables` (`_measurement_tables.py:632`) also clears the part's copied `figures/` (alongside `tables/`) and writes the new one | yes |
| Process, `--process-format zarr` — `_cli_process_only.py` | after `pipeline.apply` (**new**) | `write_process_only_layer(…, figures=)` → `_save_store`, inside the consolidated part | **no** — no `deliverables/` |
| Process, `--process-format tiff` | not built | — | — |

**Measure mode semantics.** The whole `figures/` group is rebuilt from the
current pipeline's `PlotImage` bindings. A binding removed from the pipeline
disappears from the store; a new one appears. A store's figures and its table
always come from the same pipeline. `replace_embedded_measurement_table` (the
migrate-only path) leaves `figures/` untouched — it hard-links it across like
any other unchanged file.

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

## §5 — Testing

Each new test must be shown to fail when the bug it guards is reintroduced
(user rule: a test must be able to fail).

| Test | Guards |
|---|---|
| Full-mode store with a `PlotImage` binding has `figures/` + per-binding groups, the descriptor, and `sha256` equal to each file's bytes | §1 |
| Pipeline with no `PlotImage` binding → no `figures` key, no `figures/` group | §1 presence |
| An independent Zarr v3 reader (zarr-python, not the writer's own reopen) opens `figures/` as groups; `OME/zarr.json` `series` and `METADATA.ome.xml` are unchanged | §1 namespace; store contract step 4 |
| `store=` validation: `plotly-json`/`html` with `mpl`, unknown name, duplicate, `()` → `TypeError` at class definition | §2 |
| `store` omitted → `("plotly-json",)` for plotly, `("png",)` for mpl | §2 defaults |
| A provider overriding `inspect()` stores each page in its backend default | §2 fallback |
| Each serializer, run in two separate processes, produces identical bytes. Chrome-free cases (`plotly-json`, `html`, mpl `png`/`svg`) run everywhere. Chrome-dependent cases (Plotly `png`/`svg`) reuse the existing `requires_kaleido_chrome` marker; the plan must name a lane where they actually run, because a marker that skips on every lane is a silent green (user rule: a check that cannot run must fail) | §2, §4 |
| The existing process-mode byte-identical-store test, extended to a pipeline with an image binding | §4 |
| A figure whose `inspect()` raises → store still published; `failed` entry with `page: null, format: null`; binding absent | §1 failure |
| A declared `png` with `chrome_available()` false → per-format `failed` entry; the page's other files present | §1 failure |
| `error` text with an object address is normalised | §1 |
| Copy-out lands files at today's `deliverables/plots/` paths (flat and multi-page) | §3 |
| Copy-out writes HTML from stored `plotly-json`, and its `plotly.min.js` `src` resolves from each depth | §3 |
| A tampered stored figure (bytes ≠ `sha256`) is recorded as failed and not copied | §1 hashes, §3 |
| Descriptor `failed` entries become `.failures.jsonl` lines | §3 |
| Measure mode: removing a binding removes it from the store; adding one adds it; pixel arrays remain hard links (inode check) | §3 measure |
| Measure mode: figures are written before the root (no store write outlives the publication that certifies it) | §3 ordering |
| Process mode: `zarr` export carries `figures/`; `tiff` export produces no figures | §3 process |
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
  media types, determinism, defaults, and that `deliverables/plots/` is a
  copy-out of the store.
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
    `PlotImage.inspect()` (new work per image).
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
| Figure failure | Publish the store; record in descriptor `failed` | Dashboard can tell "not configured" from "failed"; one bad figure never kills a run |
| Write architecture | Single sink into the store; `deliverables/plots/` copied out after promotion | One source of truth, same pattern as embedded tables → master |
| Copy-out content | Stored files verbatim + HTML generated from stored `plotly-json` | Keeps deliverables browsable without re-rendering from figure objects |
| Default Plotly store | `("plotly-json",)` | Chrome-free and stable; declared PNGs fail honestly |
| Continuation | Process revision → 3; full mode unchanged | Process re-derivation is cheap; full-run re-derivation is not |
| Continuation validation | Does not re-hash figures | Figures are not measurement authority |
