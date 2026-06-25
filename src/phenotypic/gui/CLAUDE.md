# PhenoTypic GUI Module Guide

The GUI is a Dash-based hub composed of four mounted sub-apps:

| Mount        | Module                              | Purpose                                  |
| ------------ | ----------------------------------- | ---------------------------------------- |
| `/`          | `gui/shell/`                        | Top-bar chrome, sidebar, home page       |
| `/builder/`  | `gui/builder/`                      | Pipeline builder (dash-cytoscape graph)  |
| `/results/`  | `gui/results_viewer/`               | Output viewer (OpenSeadragon + tables)   |
| `/run/`      | `gui/run_console/`                  | Run console (form + log tail + recents)  |

Composition lives in [shell/_app.py](shell/_app.py) (`compose_hub`); the
hub's WSGI seam is a `werkzeug.middleware.dispatcher.DispatcherMiddleware`
that strips each mount prefix before forwarding to the sub-app's Flask
server.

---

## Where to put things

| Need                                     | File                                                  |
| ---------------------------------------- | ----------------------------------------------------- |
| New CLI-produced output filename / dirname (consumed by GUI) | [`phenotypic.sdk_._io_constants`](../tools_/_io_constants.py) (`MASTER_MEASUREMENTS_PARQUET`, `DIR_RESULTS`, `JOB_METADATA_JSON`, …) |
| New CLI default / port / log format      | [_config.py](_config.py) (`DEFAULT_*`, `LOG_FORMAT`)  |
| New brand color / type / radius / shadow | [_design.py](_design.py) (`COLOR_*`, `TEXT_*`, …)     |
| New mount prefix                         | [_config.py](_config.py) (`MOUNT_*`) + register in [shell/_app.py](shell/_app.py) |
| New `app.server.config` key              | [_config.py](_config.py) (`CFG_*`; pick `CFG_OPERATION_REGISTRY` for the builder, `CFG_RUN_REGISTRY` for the run console) |
| New sandbox/cache subdirectory name      | [_config.py](_config.py) (`SANDBOX_*`, `*_DIRNAME`)   |
| New tool-internal Dash component ID      | tool's own `_ids.py` (e.g. `builder/_ids.py`)         |
| New cross-tool path store ID             | [shell/_ids.py](shell/_ids.py)                        |

**Default rule:** if a string literal would appear in two or more files,
move it to `_config.py` (Python identifiers) or `_design.py` (CSS-shaped values).
Don't re-spell. **For CLI-produced output artifact filenames (anything the
CLI writes that the GUI reads), the canonical home is
`phenotypic.sdk_._io_constants` instead — `_config.py` re-exports the
shared names so existing imports keep working.**

---

## Shared constants — always import, never re-spell

### Launcher defaults

```python
from phenotypic.gui._config import (
    DEFAULT_HOST,        # "127.0.0.1"
    DEFAULT_PORT,        # 8050
    DEFAULT_URL_PREFIX,  # "/"
    LOG_FORMAT,          # logging.basicConfig format
    SSH_TUNNEL_HINT,     # "ssh -N -L 8050:<server-host>:8050 <cluster>"
    add_launcher_args,   # add --host/--port/--debug to a parser
    configure_launcher_logging,  # logging.basicConfig with LOG_FORMAT
    join_url_prefix,           # prepend optional browser-visible base prefix
    normalize_url_prefix,      # canonicalize path-only --url-prefix values
    print_launcher_banner,       # consistent banner (title + url + hint)
)
```

Every `__main__.py` and `_launcher.py` calls `add_launcher_args(parser)`
and `configure_launcher_logging(debug=args.debug)`. Don't re-implement
either or you re-introduce drift like the `"%(name)s:"` vs
`"%(name)s "` inconsistency we cleaned up.

`--url-prefix` is shared by all GUI launchers and defaults to `/`. It is a
path-only browser prefix for reverse proxies (for example Open OnDemand
`/node/hz01/30099/` or `/rnode/hz01/30099/`), not a full URL. Use
`join_url_prefix` for browser-facing links, API fetches, and iframe URLs that
need to survive a proxy prefix. Open OnDemand `/node` forwards the full prefix
to the backend, so the shared URL-prefix WSGI middleware strips that configured
prefix from incoming `PATH_INFO` before Flask, Dash, or `DispatcherMiddleware`
route the request. Open OnDemand `/rnode` usually strips before the backend, so
the middleware is effectively a no-op while generated browser URLs remain
prefixed.

Every GUI app factory that accepts `url_prefix` must finalize the Dash app with
`configure_url_prefix_routing(app, url_prefix)` from
`phenotypic.gui._url_prefix` immediately before returning it. Do not install
prefix-stripping middleware directly in app factories; keep direct
`install_url_prefix_strip_middleware(...)` calls inside `_url_prefix.py` and
its unit tests so future proxy-routing changes have one implementation point.

### Mount prefixes

```python
from phenotypic.gui._config import (
    MOUNT_HOME,             # "/"
    MOUNT_BUILDER,          # "/builder/"
    MOUNT_VIEWER,           # "/results/"
    MOUNT_RUN,              # "/run/"
    SANDBOX_API_PREFIX,     # "/sandbox/api"
    RUNS_BLUEPRINT_PREFIX,  # "/runs"
)
```

Every dispatcher mount, tab href, redirect, and Flask blueprint
`url_prefix` must spell these via the constant. The shell layout's
`_TAB_HREFS` dict and the composer's dispatcher mounts BOTH read these
so they can never disagree.

### Flask `app.server.config` keys

```python
from phenotypic.gui._config import (
    CFG_URL_PREFIX,            # current Dash app's mount-point prefix
    CFG_OPERATION_REGISTRY,    # builder's OperationRegistry
    CFG_RUN_REGISTRY,          # run console's RunRegistry (also stashed on shell)
    CFG_RUNNER,                # process-wide LocalRunner
    CFG_IMAGE_ROOT,            # builder image root
    CFG_SANDBOX_ROOT,          # frozen sandbox root (string)
    CFG_OUTPUT_ROOT,           # results viewer OutputRoot
    CFG_FILTERED_STATE,        # results viewer CurationLabels (durable, categorized)
)
```

Reads use `app.server.config[CFG_*]` or `current_app.config.get(CFG_*)`;
writes use the same constant. The two `*_REGISTRY` keys are deliberately
distinct — even though the builder and run-console Dash apps live on
separate Flask servers, naming the keys after the registry's TYPE
prevents a future cross-tool callback from grabbing the wrong handle.

### Sandbox / cache paths

```python
from phenotypic.gui._config import (
    SANDBOX_GUI_DIRNAME,           # ".phenotypic-gui"
    SANDBOX_PRESETS_SUBDIR,        # "presets"
    SANDBOX_BUILDER_TILES_SUBDIR,  # "builder_tiles"
    RUN_LOG_DIRNAME,               # ".gui_log" (inside a run's output dir)
    VIEWER_CACHE_DIRNAME,          # ".viewer_cache" (inside an output root)
)

presets = sandbox.root / SANDBOX_GUI_DIRNAME / SANDBOX_PRESETS_SUBDIR
log_dir = output_dir / RUN_LOG_DIRNAME
```

### Branding strings

```python
from phenotypic.gui._config import (
    TITLE_HUB,       # "PhenoTypic GUI"
    TITLE_BUILDER,   # "PhenoTypic Pipeline Builder"
    TITLE_VIEWER,    # "PhenoTypic Results Viewer"
    TITLE_RUN,       # "PhenoTypic Run Console"
)

dash.Dash(..., title=TITLE_BUILDER)
```

### Threading

```python
from phenotypic.gui._config import THREAD_NAME_PREFIX  # "phenotypic-gui"

ThreadPoolExecutor(thread_name_prefix=f"{THREAD_NAME_PREFIX}-slurm")
threading.Thread(name=f"{THREAD_NAME_PREFIX}-idle-release", ...)
```

### CLI-produced output filenames (re-exported from `phenotypic.sdk_`)

```python
from phenotypic.gui._config import (
    MASTER_MEASUREMENTS_PARQUET,  # "master_measurements.parquet"
    MEASUREMENTS_CSV,             # "measurements.csv"
    MEASUREMENTS_PARQUET,         # "measurements.parquet"
    ANALYSIS_CSV,                 # "analysis.csv"
    ANALYSIS_PARQUET,             # "analysis.parquet"
    PIPELINE_JSON,                # "pipeline.json"
    RESULTS_DIRNAME,              # "results"
    PROGRESS_DIRNAME,             # "progress"
    DELIVERABLES_DIRNAME,         # "deliverables"
    DASHBOARD_FILENAME,           # "dashboard.html"
)
```

These are re-exports of the canonical constants in
`phenotypic.sdk_._io_constants`; importing from either location
yields the same string. Use `_config.py` for ergonomic GUI imports; reach
for `phenotypic.sdk_` directly when in non-GUI code that consumes these
filenames (e.g. test fixtures, CLI integration tests).

**Output layout — `deliverables/`.** These are *filenames*, not full
paths. The user-facing run artifacts (`master_measurements.*`,
`measurements.*`, `measurements_by_feature/`, `analysis.*`,
`dashboard.html`, `analysis.html`, `processing_report.html`,
`README.md`, `pipeline.json`) now live under `<output>/deliverables/`
(`DELIVERABLES_DIRNAME` = `"deliverables"`, underlying
`DIR_DELIVERABLES` in `phenotypic.sdk_`). Join them via the
`phenotypic.sdk_` path helpers (`deliverables_dir(output)`,
`master_measurements_parquet_path(output)`, …) so the subfolder stays
single-sourced. Detection overlay PNGs now live under
`deliverables/overlays/<dataset>/` (also accessed via the `phenotypic.sdk_`
path helpers). The durable **QC + curation state** (`qc_summary.parquet`,
`qc_members.parquet`, `review_state.json`, `curation_labels.parquet`,
`custom_categories.json`) now lives under `deliverables/qc/` (`DIR_QC` joined
on `deliverables_dir(output)`; resolve via `qc_dir(output)` /
`qc_summary_parquet_path(output)` / … — never hand-join `qc/`). It moved
*into* `deliverables/` so a deliverables bundle is self-contained and portable;
`resolve_qc_dir(output)` / `BundleLayout.qc_dir` still read the legacy
output-root `qc/` of pre-relocation runs, and `migrate_legacy_qc` MOVES a
legacy `qc/` into `deliverables/qc/` on discovery. The root-level directories
`RESULTS_DIRNAME` (`results/`, per-image hdf/measurements),
`PROGRESS_DIRNAME` (`progress/`), and `processing_state.json` are **not**
deliverables and stay at the output-dir root.

---

## Design tokens — Python and CSS share the same source

[_design.py](_design.py) is the single source of truth for **all** design
tokens. `inject_design_tokens(app)` splices a `<style>` block carrying
`--font-*`, `--font-size-*`, `--color-*`, `--oi-*`, `--text-*`,
`--sp-*`, `--radius-*`, `--shadow-*`, `--ease-*`, and `--transition`
into every Dash app's `index_string`. The hub composer's
`wrap_in_chrome` calls it on every mount, and each sub-app `_app.py`
calls it as well — both paths are idempotent via a marker comment.

### Using tokens in CSS

```css
.my-card {
    background: var(--color-surface);
    color: var(--color-body);
    border: 1px solid var(--color-border);
    padding: var(--sp-3);
    border-radius: var(--radius);
    box-shadow: var(--shadow-sm);
    transition: var(--transition);
}
```

**DO NOT** declare a `:root { --color-* }` block inside a tool's CSS
file — `_design.py`'s injection covers all four mounts. Tool-specific
overrides are fine, but only for values truly unique to that tool. For
example, [builder/assets/builder.css](builder/assets/builder.css)
keeps `--color-interactive: var(--oi-purple)` (point-picker mixin)
which references shared tokens and never redefines a shared value.

### Semantic typography (`--font-size-*` / `FONT_SIZE_*`)

For `font-size` and `font-family`, use the **semantic aliases** —
they read at call sites and survive font-stack swaps without churn.
The raw `--text-*` rem-scale primitives are still injected for
back-compat but new code should reach for the semantic tier:

| Role        | CSS                       | Python                    |
| ----------- | ------------------------- | ------------------------- |
| Display     | `var(--font-size-display)`  | `FONT_SIZE_DISPLAY`     |
| Title       | `var(--font-size-title)`    | `FONT_SIZE_TITLE`       |
| Header 1    | `var(--font-size-header-1)` | `FONT_SIZE_HEADER_1`    |
| Header 2    | `var(--font-size-header-2)` | `FONT_SIZE_HEADER_2`    |
| Body lead   | `var(--font-size-body-lg)`  | `FONT_SIZE_BODY_LG`     |
| Body        | `var(--font-size-body)`     | `FONT_SIZE_BODY`        |
| Label       | `var(--font-size-label)`    | `FONT_SIZE_LABEL`       |
| Caption     | `var(--font-size-caption)`  | `FONT_SIZE_CAPTION`     |

For font families: CSS uses the existing
`var(--font-display | --font-body | --font-mono)`; Python inline
styles and call sites that don't see CSS variables (Cytoscape
stylesheets, Plotly layouts, `dash_table` `style_cell`) import the
matching `FONT_FAMILY_DISPLAY` / `FONT_FAMILY_BODY` /
`FONT_FAMILY_MONO` string constants. Never hardcode a font-family
literal — the active GUI font is owned by `_design.py` and
swapping it should never require touching call sites.

### Using tokens in Python inline styles

```python
from phenotypic.gui._design import (
    COLOR_BG, COLOR_BLUE, COLOR_GOLD, COLOR_MUTED, COLOR_NAVY,
    COLOR_SURFACE, OI_VERMILION,
)

html.Div(style={"color": COLOR_NAVY, "background": COLOR_SURFACE})
```

**Never** write `"color": "#003660"` inline. Pyright will not catch it,
but a future palette change will. The same goes for `f"1px solid #1b75bc"`
— write `f"1px solid {COLOR_BLUE}"`.

### Palette rules (DESIGN.md)

`COLOR_*` (navy/blue/gold/etc.) are UI-only — never use them as data
series colors. `OI_*` (Okabe-Ito) are data-only — never use them for UI
chrome. Series order is fixed: navy, orange, sky, green, blue, purple
(vermilion reserved for error / alert). Yellow may not be used as text
on white backgrounds. See [../../../DESIGN.md](../../../DESIGN.md) for
badge contrast variants and prohibited combinations.

---

## Adding a new GUI feature

1. **Update [FEATURES.md](FEATURES.md)** — every user-visible affordance
   gets a row. The `gui-checks` workflow's `features-md-gate` job
   rejects any PR that touches `gui/` without modifying `FEATURES.md`.
2. **If it's an end-to-end flow worth a tutorial**, also update
   [WORKFLOWS.md](WORKFLOWS.md), add `_capture_<id>` in
   `scripts/capture_gui_tutorial_screenshots.py`, and add a walkthrough
   page under `docs/source/tutorials/gui/`. The `gui-checks` workflow's
   `workflows-md-gate` job enforces the round-trip.
3. **Use [_config.py](_config.py)** for any new shared constant.
4. **Use [_design.py](_design.py)** for any color, type size, radius,
   shadow, or motion value.
5. **Re-run `uv run python scripts/capture_gui_tutorial_screenshots.py`**
   after any visible chrome change and commit refreshed PNGs.

### Column-aware analyzer params

Filter / model params that name a column in
`deliverables/measurements.parquet`
should be annotated with `ColumnRef` / `ColumnRefList` from
`phenotypic.sdk_` instead of bare `str` / `list[str]`. Analyzers are
pydantic models, so these are declared as annotated **class-level
fields** (there is no `__init__`):

```python
from phenotypic.sdk_ import ColumnRef, ColumnRefList

class MyFilter(SetAnalyzer):
    on: ColumnRef
    groupby: ColumnRefList
```

The annotation is purely informational at runtime (`Annotated[str,
...]` is still a `str`), but the GUI's `OperationRegistry` walks
`cls.model_fields` and inspects each `FieldInfo.metadata` for the
marker, rendering matching params as dropdowns populated from
`MeasurementSchema.columns_for(...)`. A `ColumnRef | None` union
renders as a two-button "Column / None" RadioItems toggle so the user
can switch dtype without losing the dropdown affordance. No GUI
registry edit is required when adding a new analyzer — the marker
travels with the field annotation.

---

## Common gotchas

- **`requests_pathname_prefix=url_prefix, routes_pathname_prefix=MOUNT_HOME`** — sub-apps
  mounted under `DispatcherMiddleware` see their mount prefix stripped
  before Dash routes. Standalone launches collapse to identical prefixes
  because `url_prefix` defaults to `MOUNT_HOME` ("/").
- **External proxy prefixes are browser-facing and sometimes backend-facing** —
  keep `DispatcherMiddleware` mount keys at the internal `MOUNT_*` paths, build
  Dash `requests_pathname_prefix`, chrome hrefs, shell API fetches, and
  `/runs/...` iframe URLs through `join_url_prefix(base_prefix, path)`, and let
  the URL-prefix WSGI middleware strip configured `/node/...` prefixes before
  routing when the proxy forwards them to the backend.
- **Finalize URL-prefix routing through the shared helper** — factories should
  return `configure_url_prefix_routing(app, url_prefix)` after registering their
  routes, callbacks, static blueprints, and layouts. For the composed hub, call
  the helper only after `DispatcherMiddleware` has replaced
  `shell_app.server.wsgi_app`; wrapping earlier would leave `/node/...`
  requests visible to the dispatcher.
- **Don't import dash from `_config.py` or `_design.py`** — they stay
  cheap to import everywhere, including from blueprint and test code.
- **Registry keys are split by type** — the builder's
  `OperationRegistry` lives under `CFG_OPERATION_REGISTRY`
  (`"pheno_operation_registry"`) and the run console's `RunRegistry`
  lives under `CFG_RUN_REGISTRY` (`"pheno_run_registry"`). They were
  previously overloaded behind a single `pheno_registry` key; the
  rename makes the type explicit at every call site.
- **`inject_design_tokens` is idempotent** — both the sub-app `_app.py`
  factories AND `wrap_in_chrome` call it; only the first call inserts
  the `<style>` block (marker comment de-dupes).
- **`CFG_FILTERED_STATE` now holds `CurationLabels`, not
  `FilteredMeasurements`** — the live results-viewer curation backend was
  swapped to the durable, categorized store. The config key name is
  unchanged (≈12 duck-typed call sites keep working). `FilteredMeasurements`
  (`results_viewer/_filtered_state.py`) is now a **utility / constants
  module only** (`KEY_DATASET` / `KEY_IMAGE_FILE` / `KEY_OBJECT_LABEL` +
  `decode_removed_keys_payload`) — do not extend it; new curation state lives
  on `CurationLabels` (`results_viewer/_curation_labels.py`).
- **A single-`Output` `allow_duplicate` callback returning a list 500s on
  the empty case** — `STORE_REMOVED_KEYS` payloads are lists, and Dash treats
  an `allow_duplicate` output as multi-mode; a bare `[]` (e.g. restoring the
  last labeled object) makes the multi-return validator see *zero* values and
  raise. Wrap the return in a 1-tuple: `return (payload,)`. Same rule for
  wildcard (`MATCH`) outputs returning a single component (e.g. the radial
  popover body) — wrap it: `return (body,)`.
- **Standalone deliverables bundle** — the results viewer boots from a full
  `python -m phenotypic` run **or** a portable, deliverables-only bundle
  (`deliverables/master_measurements.parquet` + mirror + overlays + `qc/`, no
  per-image `results/`). `BundleLayout.detect(root)` resolves which:
  `layout.output_root is None` (and `OutputRoot.has_results` is `False`) for a
  bundle, where the viewer `root` IS the `deliverables/` folder. Every
  deliverables/qc/error path resolves through `layout` (never re-join `root`,
  or you double-join `deliverables/`). The header **mode badge**
  (`build_mode_badge`, `HEADER_MODE_BADGE_ID`) reads `has_results` to show
  "Full run" vs "Standalone bundle"; the sidebar classifier's
  `Capabilities.is_deliverables_bundle` (deliverables/master, no `results/`)
  drives a distinct `bundle` badge so such a directory is recognizable as
  viewer-openable.
- **Pixel-layer toggle is gated on `results/`** — `build_layer_toggle`
  (colony view) returns `None` for a standalone bundle (`has_results is
  False`) because the RGB/Enhanced/Labels layers source per-image
  `results/<ds>/.../<stem>.h5` HDFs that a bundle does not ship. Curation, QC
  review, overlays, and DZI deep-zoom (overlay-tiled) all still work; only the
  per-image full-res layer switch is absent. The `STORE_ACTIVE_LAYER` store
  stays mounted regardless so the colony render callback's Input is resolvable
  even when the control is hidden.

---

## Error-category triage (curation)

The results viewer's per-colony curation is an **error-category radial
menu**, not a binary remove. The shared component is
`gui/_shared/_radial.py` (one implementation for both tile surfaces):

- **Per-tile trigger** (`build_radial_trigger`) → a `▾` button (or a colored
  category **badge** when labeled) anchoring a **lazily-populated**
  `dbc.Popover`. The body ships empty and a `MATCH` populate-on-click callback
  fills the wedge ring (`build_radial_body`) — mirrors the `_build_stack_popover`
  pattern so a grid of many tiles stays light.
- **Surfaces** are keyed by a `surface` arg (`"colony"` / `"qc"`) baked into
  every id type (`colony-cat-wedge` vs `qc-cat-wedge`, …) so the two tabs'
  pattern-matched callbacks never collide. Colony wiring lives in
  `colony_view/_callbacks.py`; QC in `_qc_tab/review/_callbacks.py`.
- **One wedge click → one `CurationLabels.mark(image_file, label, category)`**;
  the center node carries `RADIAL_RESTORE_SENTINEL` (`"__restore__"`) → `unmark`.
  Category colors come from `_design.category_color` (core = fixed OI slot,
  custom = cycled palette + a `radial-badge--custom` discriminator, decision D).
- **Grid category channel (decision A):** the grid re-render reads
  `filtered_state.labels` under `filtered_state._lock` (a server-side snapshot —
  there is **no** `STORE_LABELS` Dash store) and threads a `category_of` map
  into `build_grid` / `build_tile_grid`.
- **On-disk dual ownership:** the GUI writes `deliverables/errors/<category>.parquet`
  **live** as the user curates (via `CurationLabels._save_locked`), and the CLI
  **re-emits** them on the next finalize/recompile from the durable
  `qc/curation_labels.parquet` (which the CLI never wipes). Resolve these paths
  via `phenotypic.sdk_` helpers (`errors_dir`, `error_category_parquet_path`,
  `curation_labels_parquet_path`), never by hand-joining names.

---

## Error-analysis tab

The results viewer's **Error** tab (`results_viewer/_error_tab/`, the 5th
tab, `TAB_ERROR_ID`) ranks the single measurements that best separate a
chosen error category from a good baseline, via
`phenotypic.analysis.ErrorCutoffFinder`. The package splits the usual way:
`_data.py` (pure good/error frame construction + verified-good resolution),
`_figure.py` (the pure good-vs-error distribution figure), `_layout.py`
(the static layout), `_callbacks.py` (the recompute + cutoff-drag wiring),
and `_ids.py` (all-static component ids, e.g. `error-cutoff-table`,
`error-distribution-figure`, `error-good-mode-toggle`).

- **Good-baseline toggle (`ERROR_GOOD_MODE_TOGGLE_ID`):** `all_unlabeled`
  (default — every unlabeled object is good) vs `verified` (the good class
  is restricted to `verified_good_keys`, the unlabeled members of
  QC-reviewed groups, derived from `qc/qc_summary.parquet` +
  `qc/qc_members.parquet` + `qc/review_state.json`). The verified-good
  count badge (`ERROR_VERIFIED_COUNT_ID`) reports that set's size.
- **Server-side state, recompute on activation:** the recompute callback
  reads `filtered_state.labels` under `filtered_state._lock` (the shared
  `CurationLabels`; there is **no** `STORE_LABELS` Dash store) and builds
  the error frame by looking the labeled keys up in `output_root.master_df`
  (the post-applied mirror). It gates on `active_tab == TAB_ERROR_ID` and
  raises `PreventUpdate` off-tab, so marking a colony on another tab never
  runs the finder — returning to the Error tab recomputes from the current
  durable labels store (R2).
- **Live persistence (focused category):** each recompute writes
  `deliverables/error_analysis.{parquet,csv}` for the *focused* category
  (leading `category` column, transient last-viewed-in-session) via
  `_persist`, and in verified mode writes `deliverables/verified.parquet`
  via `_persist_verified`. The HTML report is written only on explicit
  `Save analysis report`. The per-category `deliverables/errors/<category>.parquet`
  are written by the curation layer (`CurationLabels._save_locked`), not here.
- **CLI finalize re-emits (all categories):** `reemit_error_deliverables`
  (`_cli/_cli_error_outputs.py`, called from `finalize_post_master_outputs`)
  re-runs the finder per labeled category against the clean master and
  authoritatively rewrites `errors/*` + `error_analysis.*` (columns
  `[category, *RESULT_COLUMNS]`) from the durable `qc/curation_labels.parquet`,
  so headless output matches the live GUI. `verified.parquet` is **GUI-only**
  — finalize never writes it (it would be stale: `review_state.json` is reset
  on recompile). Resolve every path via `phenotypic.sdk_` helpers
  (`error_analysis_parquet_path`, `errors_dir`, `verified_parquet_path`, …),
  never by hand-joining names.

## Builder preview cache

The builder's `Run preview` does NOT cache full `Image` instances. Each
intermediate is rendered to a PNG **at preview-run time** by
`render_node_preview` (see `builder/_image_renderer.py`) and the resulting
bytes — together with measurement DataFrames and `PreviewRenderError`
sentinels — are what live in `IntermediatesCache` (`builder/_session.py`).
The inspector base64-wraps the bytes for `<img src=…>` and is otherwise
a no-op on selection.

- **Per-stage rule** (encoded inside `render_node_preview`):
  - Enhancer → PNG of `detect_mat`.
  - Detector / Refiner → overlay PNG (`label2rgb` of `objmap` over
    `detect_mat`, scikit-image-fallback to a tab20 colormap).
  - Corrector / nested `ImagePipeline` / unknown → PNG of `rgb`.
- **Why**: pre-baking collapses worst-case resident memory from ~1–2 GB
  (full Images per node × bounded sessions) to ~10–15 MB of PNG bytes,
  and makes inspector node-switching effectively free.
- **Render failures don't kill the preview** — they're caught per-node
  and stored as `PreviewRenderError(message)` so the inspector can
  surface the error inline while the rest of the run-preview output
  remains valid.
- **Testable seam**: `_callbacks._bake_preview_cache(state, pipeline,
  result, session_id, cache)` is the unit the integration tests target;
  `run_preview` itself is a thin Dash-callback adapter around it.
