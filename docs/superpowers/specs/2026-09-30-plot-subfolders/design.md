# Plot subfolders in stored figures, and per-ROI calibration overlays

**Status:** design approved section by section, 2026-09-30. One PR on
`feat/plot-subfolders`.

## Objective

Each plot that an operation stores gets a folder of its own, and a plot can hold
several figures. The immediate use is the colour calibration.
`CalibrateColorRpcc` stores **one tile overlay per ROI** instead of one figure
with every ROI side by side. With two cards at opposite edges of a 6016 px
frame, the combined figure is a tall, wide sheet whose ROI panels crowd each
other. One file per ROI keeps each one readable.

The layout rule is the same for every plot, in the store and in deliverables:

```text
figures/<run>/<binding>/<plot>/<file>
```

## Non-goals

- **No change to how a plot is drawn**, beyond `CalibrateColorRpcc` splitting
  its overlay by ROI. In particular, `show_tiles()` still returns the combined
  figure for notebook use.
- **No per-ROI ΔE00 chart.** The colour fit is one fit across all ROIs, so
  `delta_e` stays one chart of every patch.
- **No ROI labels in file names.** Files are `roi_<index>` (user decision,
  "keep it simple"). `CheckerRoi.label` is not used for naming.
- **No migration of existing stores.** Run folders are never rewritten (figures
  spec §1a), so old runs keep their flat layout forever. Readers accept both.
- **No GUI work.** Nothing under `_gui/` reads stored figures (checked: the
  only readers are listed under Background).

## Decisions on record (user, 2026-09-30)

| # | Decision |
|---|---|
| D1 | The change lives in PhenoTypic, not in AutoConvertRaw-GC's `process_one`. |
| D2 | **Every plot gets a folder, always**, not only plots with several files (option A). A one-file plot such as `delta_e` becomes `delta_e/delta_e.png`. Chosen because it is one rule, and readers must accept the old flat layout either way. |
| D3 | Per-ROI files are named `roi_<index>`. |
| D4 | `deliverables/plots/` **mirrors the store**. |
| D5 | One PR for the whole change. |

## Background: what exists today

All references are to `origin/main` at `ba725001`.

- **Store layout** (`docs/superpowers/specs/2026-09-22-figures-in-ome-zarr/design.md`
  §1, §1a): `figures/<run>/<binding>/<page>.<ext>`. The binding folder is
  `safe_path_component(binding.id)`, and the page file stem is
  `unique_page_stems` over the page keys
  (`plotting/_pipeline/_store_figures.py:406`).
- **The descriptor is the contract.** `attributes.phenotypic.figures` lists every
  file's store-relative `path`, and the figures spec says "a consumer never
  infers anything from a file name". `FIGURES_SCHEMA_VERSION` is `1`
  (`sdk_/ngff_.py`).
- **One helper does parse paths.** `split_figure_file_path`
  (`sdk_/_image_figures.py:209`) requires exactly
  `figures/<run>/<binding>/<file>`. Its callers:
  - `carry_figure_runs` / `_carry_file` (`_image_figures.py:402`), which carries
    other run folders across a rewrite.
  - `_kept_binding` (`_store_figures.py:371`), which keeps §3a figures such as
    the calibration overlay across a re-measure. It also requires every page of
    a binding to live in **one** directory (`_store_figures.py:386`).
- **Other readers of stored figures**, which need no change:
  - `latest_run_date` (`_cli_process_single.py:482`) reads run names and hashes.
  - The re-measure clear in `_measurement_tables.py:702` clears a whole run
    folder.
  - `known_figures_schema` (`_image_figures.py:506`) gates every write path.
- **Unknown versions are already safe.** A writer whose
  `FIGURES_SCHEMA_VERSION` differs from a store's descriptor leaves that
  descriptor and its files untouched, adds no run, and warns
  (`_image_figures.py:364, 488`; `_measurement_tables.py:786`).
- **Deliverables** (`_store_copyout.py:170-230`, and `publish_plot_output` in
  `_writer.py:309` for publishes without a store):
  - Usually `plots/<binding>/<dataset>/<stem>/<page>.<ext>`, plus `manifest.json`
    (`schema_version` 2).
  - A **flat** special case: a binding whose only page is `default` is published
    as `plots/<binding>/<dataset>/<stem>.<ext>`.
  - Nothing in `src/` reads these manifests.
- **The calibration plot** (`correction/_color_correction/_calibrate_color_rpcc.py:372`)
  returns `PlotOutput(pages=(tiles, delta_e))`. `tiles` is
  `render_calibration_overlay(record)`
  (`_calibration_overlay.py:521`), which lays out one panel per entry of
  `record.rois`.
- **Page keys are not guaranteed slash-free.** Some keys are built from data
  values (`canonical_group_key`, `abc_/plotting/_output.py:86`, used by the
  time-series plot). So structure cannot be encoded inside the key string.
- **Paths are hard-coded in only a few tests:** 3 test files, 6 lines
  (`tests/unit/sdk_/test_image_figures.py` ×4, `test_store_copyout.py` ×1,
  `tests/integration/cli/test_figures_in_store.py` ×1).
- **Baseline:** the 17 non-GUI test files that touch figure storage or plot
  pages give **313 passed, 0 failed** at `ba725001` (Slurm job 29302628,
  11.5 min).

## §1: The `plot` field on `PlotPage`

```python
@dataclass(frozen=True)
class PlotPage:
    key: str
    figure: FigureLike
    label: str | None = None
    metadata: Mapping[str, str | int | float | bool | None] = field(default_factory=dict)
    plot: str | None = None        # new: the folder; None means "same as key"
```

- `plot` names the **folder**, and `key` names the **file** inside it. When
  `plot is None` the page's plot is its `key`, so every existing plot still
  works unchanged and lands at `<key>/<key>.<ext>` (D2).
- **Validation** is in `__post_init__`, beside the key check. `plot`, when
  given, is a non-empty string.
- **Uniqueness:** `PlotOutput` refuses a duplicate `(plot or key, key)` pair.
  It no longer refuses a duplicate bare `key`. `tiles/roi_0` and `masks/roi_0`
  may coexist.
- **Why a field rather than a `"tiles/roi_0"` key convention:** keys from
  `canonical_group_key` can contain `/`. Splitting on it would silently move an
  existing figure.

A small helper used everywhere a page's plot is needed:

```python
def page_plot(page: PlotPage) -> str:
    return page.plot if page.plot is not None else page.key
```

## §2: Store layout and descriptor

### Layout of a run written by this version

```text
figures/
├── zarr.json                         empty group document (unchanged)
└── <run>/
    ├── zarr.json                     empty group document (unchanged)
    └── <binding>/                    safe_path_component(binding.id) (unchanged)
        ├── zarr.json                 empty group document (unchanged)
        └── <plot>/                   new level
            ├── zarr.json             empty group document (new)
            ├── <file>.png
            └── <file>.plotly.json    for a Plotly page, as today
```

**Names.**
- **Plot folders:** `unique_page_stems` over the binding's distinct plots, in
  first-appearance order. So two plots whose names clean to the same name
  (case-folded) get the existing stable digest suffix.
- **Files:** `unique_page_stems` over that plot's pages, in page order.
- Both reuse the existing rule. There is no new naming code.

**Each new folder gets the empty Zarr group document** its parents already
have, through the same `_ensure_group`, so `figures/` still opens as a Zarr
hierarchy.

### Descriptor

`FIGURES_SCHEMA_VERSION` goes from **1 to 2**. Version 2 is version 1 plus:

- **Page entries** gain `"plot": "<logical plot name>"`. That is the unsanitized
  `page_plot(page)`; the folder name is visible in each file's `path`.
- **Failure entries** gain `"plot"`, null where `"page"` is null (an
  `inspect()` failure).
- **File paths** are one level deeper:
  `figures/<run>/<binding>/<plot>/<file>`.

**Reading version 1.** A version 1 page entry has no `"plot"`, and its files are
at `figures/<run>/<binding>/<file>`. This version reads such an entry as a
**flat page**, and never moves or renames it.

**Writing over a version 1 descriptor.**
- `known_figures_schema` accepts **1 and 2**.
- When a run is added, the merged descriptor is written as version 2. Every
  version 1 run is a valid version 2 run with flat pages, so nothing is
  relabelled.
- The old runs' files and entries are carried across byte for byte (§1a,
  unchanged).

**Older PhenoTypic on a version 2 store.** Its `known_figures_schema` sees `2`,
so it adds no run and leaves the figures alone, with its existing warning. It
never reaches its 4-part path parser.

### Path helpers (`sdk_/_image_figures.py`)

```python
def figure_file_path(run_id: str, directory: str, filename: str,
                     plot: str | None = None) -> str: ...
def split_figure_file_path(path: str) -> tuple[str, str, str | None, str]:
    """(run_id, binding_dir, plot_dir or None, filename)"""
```

- `split_figure_file_path` accepts **exactly** 4 parts (version 1, plot `None`)
  or 5 parts (version 2).
- It still refuses `..`, `.`, absolute paths, a first part other than
  `figures`, and any other depth.
- `StoredFigurePage` gains `plot: str | None` (the logical name) and
  `directory: str | None` (the cleaned folder name, `None` for a flat page).
  `StoredFigureFailure` gains `plot`.

### Determinism

The figures spec's §4 contract is unchanged: two identical process runs on the
same UTC day write byte-identical stores. Folder names, ordering and group
documents all derive from page order and names alone.

## §3: Readers and deliverables

### Inside the store

| Reader | Change |
|---|---|
| `_kept_binding` (§3a: keep the calibration overlay across a re-measure) | Rebuild each page with its stored plot folder taken from its own path. Require a single **binding** folder (the existing check, applied to the binding part only); allow any number of plot folders. A binding is kept whole from one stored run, so it is either all version 2 (foldered) or all version 1 (flat, rewritten flat); never mixed. |
| `carry_figure_runs` / `_carry_file` | Accept both path shapes; carry each file byte for byte as today. |
| re-measure run clear (`_measurement_tables.py:702`) | none: it clears the whole run folder |
| `latest_run_date` | none: it reads only run names and hashes |

### Deliverables: mirror the store (D4)

```text
plots/<binding>/<dataset>/<stem>/
├── manifest.json        schema_version 3
├── <plot>/
│   ├── <file>.png
│   ├── <file>.plotly.json
│   └── <file>.html      generated for a Plotly page, as today
└── …
```

- **Manifest version 3** is version 2 with `"plot"` added to each `pages[]` and
  `failed[]` entry, and with file paths relative to the image folder, so they
  include the plot folder.
- **The flat special case is removed.** A binding whose only page is `default`
  is published as `<stem>/default/default.<ext>`, like every other plot. This
  deletes the "a failure must not flip a multi-page binding to the flat
  layout" logic in `_store_copyout.py`.
- **A page kept flat from a version 1 run** (plot `None`) is copied into the
  image folder itself, as today.
- **Both deliverables writers follow the rule:** the store copy-out
  (`_store_copyout.py`) and the direct publish (`publish_plot_output`,
  `_writer.py`). `unique_page_stems` applies per plot folder, as in the store.
- **Stale files: behaviour unchanged, scope narrowed.**
  - `_remove_stale_sibling` (`_writer.py:238`) removes only a page's own
    other-format sibling, for example the old PNG when a rerun writes HTML.
    It now runs inside that page's plot folder.
  - As today, it does **not** sweep a file whose page vanished between runs.
    Its docstring explains that this needs ownership of the whole directory,
    which it does not have. A re-copy that drops a plot therefore leaves that
    plot's old folder in place, exactly as a dropped page's file is left today.
  - Sweeping is out of scope.

## §4: Per-ROI calibration overlays

`CalibrateColorRpcc.plot_pages` (today `_calibrate_color_rpcc.py:372`) returns,
for a record with N ROIs:

```python
PlotOutput(pages=(
    *(PlotPage(key=f"roi_{i}", plot="tiles", figure=..., label=f"Tile overlay, ROI {i}")
      for i in range(N)),
    PlotPage(key="delta_e", figure=self.show_delta_bar_plot(),
             label="Delta E00 before and after"),     # plot omitted → "delta_e"
))
```

- **Each overlay** is `render_calibration_overlay` applied to a copy of the
  record holding only that ROI: `record.model_copy(update={"rois": [roi]})`.
  - The title still names the image, the verdict and the frame-level
    `n_fitted` / `n_expected`, because the fit belongs to the whole frame.
  - The renderer's no-overlap sizing applies per figure.
  - `RoiOverlay.roi_index` keeps the index in the panel title, so a lone
    overlay still says which ROI it is.
- **The index `i`** is the ROI's position in `rois`, which is `roi_index`. It
  is not renumbered after a refusal.
- **"Both pages or neither" still holds as "all pages or none".** A failure
  drawing any page fails the whole binding, which publishes nothing, as today.
- **A refused frame** still draws every per-ROI overlay, each carrying its own
  reasons and no after-values. `plot_pages` keeps raising
  `FigureInputUnavailable` when there is no record, unchanged.
- **`show_tiles()` is unchanged**: one combined figure, for notebooks.
- **Stored result for a 2-ROI rig:** `CalibrateColorRpcc/tiles/roi_0.png`,
  `CalibrateColorRpcc/tiles/roi_1.png`, `CalibrateColorRpcc/delta_e/delta_e.png`.

## §5: Testing

Tests are written first (repo TDD rule). Focused files run per task. The 17
figure test files run once per phase. The full sharded suite runs once at the
end as a Slurm job, using the committed
`docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch`.

| Area | Must prove |
|---|---|
| `PlotPage` / `PlotOutput` | `plot` defaults to `key`; uniqueness is per `(plot, key)`; an empty `plot` is refused |
| store writer | `<binding>/<plot>/<file>`; a group document in every new folder; the folder-name collision digest; descriptor version 2 with `plot` on pages and failures |
| version 1 → 2 | Adding a run to a version 1 store writes version 2 and leaves the version 1 run's files and entries byte-identical. A version 1-only writer, simulated by pinning `FIGURES_SCHEMA_VERSION = 1`, adds no run to a version 2 store and leaves its figures intact. |
| path helpers | 4 and 5 parts accepted; 3 and 6 parts, `..`, `.`, absolute paths and a wrong root refused; `figure_file_path` ↔ `split_figure_file_path` round trip |
| `_kept_binding` | keeps a version 2 binding with plot folders; keeps a version 1 flat binding flat; still refuses a binding spread over two binding folders |
| carry | a version 1 run and a version 2 run both carried byte for byte |
| deliverables | the mirrored layout and manifest version 3, for both writers; `default` no longer flat; a version 1 flat kept page copied flat; a rerun that switches a page's format removes that page's old sibling inside its plot folder |
| `CalibrateColorRpcc` | 2 ROIs give pages `tiles/roi_0`, `tiles/roi_1`, `delta_e/delta_e`; each overlay figure carries one ROI panel; a refused frame draws N overlays; 1 ROI gives `tiles/roi_0` |
| determinism | two same-day `--mode process` runs are byte-identical stores |
| end to end | a `--mode process` store over a 2-ROI pipeline holds the three files, and the descriptor lists them |

The 3 test files that hard-code flat paths are updated, not deleted.

## §6: Documentation

- **`docs/superpowers/specs/2026-09-22-figures-in-ome-zarr/design.md`:** a
  dated note at its §1 Layout and §1a pointing here: "superseded for runs
  written by the release that ships `feat/plot-subfolders`: one more level,
  `<plot>/`".
- **`CLAUDE.md`, CLI section:** "A store also carries the pipeline's per-image
  figures under `figures/<run>/`" gains "`<binding>/<plot>/<file>`".
- **Plotting user docs:** `PlotPage(plot=...)`, with the calibration example.
- **Changelog:**
  - the layout change and descriptor version 2;
  - manifest version 3, and that the flat deliverable form is gone;
  - older PhenoTypic adds no figure run to a version 2 store and leaves it
    intact.

## Blast radius

| Area | Files |
|---|---|
| plotting API | `abc_/plotting/_output.py` |
| store figures | `sdk_/_image_figures.py`, `sdk_/ngff_.py` (version constant), `plotting/_pipeline/_store_figures.py` |
| deliverables | `plotting/_pipeline/_store_copyout.py`, `plotting/_pipeline/_writer.py` |
| calibration | `correction/_color_correction/_calibrate_color_rpcc.py` |
| tests | the 17 figure files above, plus new ones |
| docs | as §6 |

**Out of scope, in AutoConvertRaw-GC** (separate change, after a PhenoTypic
release):
- Re-vendor the wheel. This moves the pipeline from **0.19.0 to 0.20.x or
  later**, which carries more than this feature.
- Rebuild the image.
- Confirm that `verify_store` needs no change: it already takes paths from the
  descriptor.
- Make `process_one`'s own `qc_refused` refusal figure per ROI.
