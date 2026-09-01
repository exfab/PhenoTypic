# Results Viewer — Scatter tab

Status: design approved, not implemented.
Branch: `feat/results-scatter-gui`, stacked on `ome-zarr-merged` @ `73516a23`.
Mockup: `docs/superpowers/artifacts/2026-09-01-results-scatter-tab/scatter-tab-mockup.html`
Logic validation: `docs/superpowers/logic_validation_scripts/2026-09-01-results-scatter-tab/crop_uint16_scaling.py`

## Summary

A third tab in the results viewer that turns
`projects/ucr_029_e_d_Maresca/scripts/plot_strain_growth_scatter.py` into a live,
configurable, clickable surface, and exports the same multi-page PDF at the end.

Every role that script hard-codes becomes a dropdown: pages by strain, a 4x4 grid of
pH x salinity, median radius against a derived frame index, colour by biological
replicate, marker by plate configuration. Clicking a point opens the colony it
represents — crop, plate context, and its measurements.

The tab is named **Scatter** and mounts after Plate and Colony. Heatmap and QC are
deprecated and are not mounted (`results_viewer/_layout.py:563-575` wires two tabs).

## Objective and non-goals

**Objective.** Replace the standalone script with a GUI surface that (a) reproduces its
figure from run data without editing code, (b) exports it as a multi-page PDF, and
(c) resolves a plotted point back to the colony, plate and measurements behind it.

**Non-goals.**
- Joining external condition CSVs (decision Q6). The tab plots the columns the run has.
- Writing curation state (decision Q4). Scatter reads curation; it never mutates it.
- Statistical modelling. Fits and summary bands are v2.
- Replacing the Colony or Plate surfaces. Scatter is a third view over the same frame.

## Decisions (locked 2026-09-01)

| # | Decision | Consequence |
|---|---|---|
| Q1 | One section group on screen at a time, with a pager | The PDF is the all-sections view |
| Q2 | Plotly + kaleido per page, merged with `pypdf` | One renderer; `kaleido>=1.2.0` already a dependency, `pypdf` is new |
| Q3 | Overlay crop by default | Requires server-side compositing — see §3 |
| Q4 | Share filters only | No curation write path; box-select bulk curate dropped |
| Q5 | Design for > 500,000 points | `Scattergl` unconditionally; see §5 |
| Q6 | External metadata join out of scope | Fix upstream with `--metadata` |

## 1. Measured facts

All read from the migrated subset at
`projects/ucr_029_e_d_Maresca/data/results/2026-08-11-migration-test/`
(36 stores) and from `deliverables/measurements.parquet` of the full 2026-08-11 run.
These are measurements, not estimates; anything derived from them is marked.

| Fact | Value |
|---|---|
| Rows in the full run | 231,229 |
| Columns | 149 (148 in a per-store table) |
| Strains (section groups) | 23 |
| Plates | 111 |
| Images | 6,657 |
| Measurers in `pipeline.json["meas"]` | `MeasureNeighborDist`, `MeasureShape`, `MeasureIntensity`, `MeasureColor`, `MeasureTexture` |
| `rgb/0` | `[3, 3132, 5086]` uint16, sharded, inner chunk `[1, 1024, 1024]` |
| `rgb/0` on disk | 79.8 MB over 60 inner chunks (~1.33 MB each) |
| `rgb/labels/objmap/0` on disk | 0.0 MB |
| Pyramid | 5 levels, `stop_px: 512`; `rgb/4` is `[3, 196, 318]`, 374 KB, reads in 39 ms |
| Label path | `attributes.phenotypic.labels` = `{"objmap": "rgb/labels/objmap"}` |

Three consequences the data forces:

- **`Metadata_FrameIndex` is absent** and `Metadata_Timepoint` is 1 for 230,329 rows and
  null for 900 — unusable as an axis. The derived frame index is required, not optional.
- **`Metadata_ImageDatetime` is 1:1 with image name** (6,657 unique). The derived frame
  index ranks on that within `Metadata_PlateID`, which is cleaner than the reference
  script's regex over filenames.
- **`Size_*` does not exist** in this run; the pipeline runs `MeasureShape`, so area is
  `Shape_Area`. Axis defaults must resolve from the run, never from a hard-coded name.

**This run cannot reproduce the reference figure**, and that is correct behaviour under
Q6: it carries no pH, salinity, biological-replicate or configuration column. Those live
only in `UCR_029_E_D-Metadata.csv`. On this data the tab can facet on strain, plate, day
and session. Re-running with those conditions passed through `--metadata` makes the exact
reference figure available with no change to the tab.

## 2. Prerequisite P0 — the crop path is broken on migrated stores

Lands as its own commit **before any Scatter code**. The Colony grid and QC gallery
consume the same route and are equally affected.

### 2.1 Truncation

`_store_layer_array_to_rgb` (`gui/_shared/tiles.py:466`) converts the `rgb` layer with a
bare cast. `np.ndarray.astype` defaults to unsafe casting, so a narrowing integer
conversion keeps the low byte: `value & 0xFF`, not a scale.

Store data is uint16 with real values in 18,315–45,344 — genuine 16-bit acquisition data
(5,799 distinct values in one 256x256x3 window, not multiples of 256 or 257). Measured on
object 24 of `d000466_280_003`, the largest in that image at 9,182 px, centred at
`(1783.158135, 342.748203)`:

```
source window        19061 .. 38171 uint16     (74.6 wraps of the 256 cycle)
rendered PNG         193 KB
mean neighbour delta 85.3                      (smooth imagery reads 0-5)
deltas > 100         36.8% of pixels
```

The mapping is non-monotonic — 18,175 maps to 255 and 18,176 to 0 — so every 256 crossing
snaps full scale. The arithmetic is re-derived in
`logic_validation_scripts/2026-09-01-results-scatter-tab/crop_uint16_scaling.py`.

**The store is not affected.** The loss happens in RAM, after the read. Viv-backed
surfaces read the same chunks and apply contrast client-side, which is why Plate and
Colony look correct and only the PNG crop route does not.

### 2.2 The naive fix introduces a second defect

The same function already imports `_normalize_to_uint8`
(`builder/_image_renderer.py:125`), whose docstring says it exists for this case. But it
min-max stretches *the array it is handed*, which for a crop is that crop's own window:

```
window A: source 18315..31783   a true 24000 renders as 107.6
window B: source 20539..28559   a true 24000 renders as 110.0
window C: source 20445..27877   a true 24000 renders as 122.0
```

Same physical brightness, three renderings. For a gallery whose job is comparison, that
is its own bug.

### 2.3 The fix

Scale, against a range computed **per image** and cached on `(store path, mtime_ns)` —
the key `crop_store_rgb` already accepts.

```python
# _shared/tiles.py — _store_layer_array_to_rgb
- if layer == "rgb":
-     return arr.astype(np.uint8)          # mod-256 truncation
+ if layer == "rgb":
+     lo, hi = image_display_range(store, mtime_ns)
+     return scale_to_uint8(arr, lo, hi)   # scale, then clip
```

`image_display_range` reads the smallest pyramid level whole (374 KB, 39 ms) and returns
its **min/max, not a percentile**: 0.5/99.5 on `rgb/4` gives 21,644–25,993 against a true
range of 17,912–45,344, which clips the colonies — the brightest thing in the frame and
the subject of the picture. Clip on apply instead.

Measured on the same colony crop, scaling against 20,511–44,047:

```
mean neighbour delta   85.3  ->  7.2
deltas > 100          36.8%  ->  0.0%
PNG size              193 KB -> 128 KB
colony vs background    n/a  ->  90 vs 32 (mean, 0-255)
```

**Known limitations, stated not hidden.** (a) 90-against-32 is correct but dim: the top of
the range is set by a specular highlight, so most of the 0–255 budget sits above the
subject. Contrast polish is a follow-on, not part of the correctness fix. (b) A per-image
scale still means two different images map differently, and prev/next steps across images.
A run-level scale fixes that and is v2.

### 2.4 Compositing (Q3)

The store path served **no contours at all**: `layer=rgb` is bare pixels, `layer=objmap`
is a colourised label map with no plate under it, and nothing blends them. Contours
existed only in the overlay-PNG fallback, which is dead whenever a store exists.

Read the objmap window alongside the rgb window, take boundaries with
`skimage.segmentation.find_boundaries` (skimage 0.25.2, already a dependency), tint the
focal `Object_Label` distinctly from its neighbours, encode. Measured on object 24: 330
boundary pixels, 0.50% of the crop; composited PNG 129 KB against 128 KB uncomposited.

**A flag, not a new layer.** `LayerName = Literal["rgb", "detect_mat", "objmap"]`
(`tiles.py:313`) names store *series*; "overlay" is a render *mode*. Adding a fourth
member would conflate the two and break the `get_args(LayerName)` validation at the route
boundary. This becomes a separate `?contours=` parameter, default on.

Resolve the label path from `attributes.phenotypic.labels`, never hard-coded.

### 2.5 P0 tests

- A synthetic uint16 store holding a monotonically increasing ramp renders monotonically
  non-decreasing. This is the invariant truncation violates; the test fails if the bug is
  reintroduced.
- A uint8 store is unchanged by the new path (`_normalize_to_uint8` already short-circuits
  on uint8; the new scale must too).
- Two non-overlapping crops of the same image map an identical source value to an
  identical output value.
- `?contours=1` on a window containing a known label emits boundary pixels; `?contours=0`
  emits none.

## 3. Module layout

```
results_viewer/_scatter_tab/
    __init__.py       public factory + callback registrar
    _ids.py           element ids (TAB_SCATTER_ID added to results_viewer/_ids.py)
    _spec.py          FigureSpec — the pure config object
    _facets.py        facet planning, caps, empty-cell handling
    _figure.py        pure figure builder (side-effect free, unit-testable)
    _callbacks.py     Dash wiring
    _inspector.py     offcanvas layout + click resolution
    _pdf.py           kaleido per page -> pypdf merge
```

`_figure.py` follows `_heatmap_tab/_figure.py`: no Dash imports, no I/O, testable against
synthetic frames without booting a server.

## 4. Data flow

```
OutputRoot.master_df
  -> filter offcanvas (shared, Q4)          -> filtered frame + curation flags
  -> facet plan (section / row / column)    -> FigureSpec
  -> one Plotly figure with Scattergl traces
       -> dcc.Graph            (screen, one section)
       -> kaleido per section  -> pypdf merge -> PDF
```

`FigureSpec` carries roles, sizes, scales and palette. Both destinations consume the same
figure object, so the PDF cannot drift from the screen.

## 5. Rendering at scale (Q5)

`go.Scattergl` unconditionally. SVG `go.Scatter` does not render at this scale, and the
pager only divides the total by the number of section groups. Adaptive threshold switching
is the deferred v2 item; always-on is strictly simpler — one code path, no threshold.

WebGL is established practice in this repo: `go.Scattergl` appears in
`measure/_measure_symzones.py`, `measure/_measure_orientation_zones.py` (traces at `:2528`
inside the `make_subplots(rows=1, cols=3)` figure created at `:2190`) and
`sdk_/branch_pathfinding/_diagnostics.py:21`, which documents the same reason.

**Context cap.** The grid renders as `make_subplots` with gl traces, guarded by
`SCATTER_FACET_CAP` in `gui/_config.py`, mirroring the existing `TIMELINE_COMPARE_CAP = 12`
(`_config.py:757`), whose comment records the reason: "browsers cap live WebGL contexts
(~16 in Chrome)". Over-cap renders the first N facets plus a visible "showing first N of M"
notice — never a silent truncation, following the rule that constant already documents.

**Two caps interact.** Per-axis `max_cardinality` bounds each dropdown's option list;
`SCATTER_FACET_CAP` bounds rows x columns. A 12-value row axis crossed with a 12-value
column axis is 144 panels. The product is checked after both axes are chosen.

**Fallback.** A pH x salinity grid is 16 panels, so a cap below 16 would break the figure
this tab exists to draw. If spike A shows 16 gl subplots do not survive, the grid collapses
to a single gl axes pair with per-facet coordinate offsets and drawn separators — one
context regardless of facet count. Spike A decides which, before layout code is written.

## 6. Click path

Each point's `customdata` is a single `int32` row index into the filtered frame. The click
callback resolves it server-side into `(dataset, stem, Object_Label)` — the key the crop
route, the Viv stage and the curation lookup already take.

This is a design requirement, not an optimisation. Carrying the strings would send a
dataset name, a ~32-character image stem and a label per point; at 231,229 rows that is on
the order of 100 bytes per point (derived, not measured), tens of megabytes per section
render over an SSH tunnel.

## 7. Inspector

Right-docked `dbc.Offcanvas`, the component the filter sidebar already uses.

- **Colony crop** — the `/crops/<ds>/<stem>/<label>.png?size=&contours=` route, server-side
  composite (§2.4). Chosen over two client-side Viv layers on measured grounds: a 256 px
  crop needs at minimum 3 inner chunks (one per channel, inner chunk `[1, 1024, 1024]`),
  ~4 MB compressed, up to ~16 MB straddling a chunk corner, against a ~100 KB PNG. The
  server does the same chunk read on local disk instead of across a tunnel, and spends no
  WebGL context.
- **Plate context** — stays a Viv stage. `rgb/3` is 392x636; the whole level is small and
  Viv gives free pan and zoom. This is the case Viv is good at. Pick the level from the
  `pyramid` block, never a hard-coded index.
- **Contours / Raw** — a segmented control, the Q3 toggle as UI rather than a URL
  parameter. State rides in the same store as the panel width.
- **Collapsing chevron** on the imagery block, reusing `builder-palette-collapse__chevron`
  (`builder/_layout.py:4361`).
- **Draggable width** — reuses the QC worklist splitter (`_assets/results_viewer.js:796`):
  clamp on drag, persist to a Dash store on mouse-up, re-apply across re-renders.
  `clampSidebarWidth` is already exposed for tests.
- **Prev / next** walks the clicked facet's points in x order.
- No **Exclude colony**. Q4 removed the write path.

## 8. Measurement grouping

Section headings are `MeasureFeatures` class names resolved from the run's own
`deliverables/pipeline.json` — the `"meas"` key, which is separate from `pipe_cfgs`.
`OutputRoot` already resolves `layout.pipeline_config_path` and reads that file.

Everything `is_metadata_header()` accepts (`sdk_/_metadata_helpers.py:281`) goes into one
flat **Metadata** group. Columns no measurer claims go to **Unattributed**.

**`get_headers()` is not uniformly zero-argument.** Executed against a real store, the
naive design throws:

```
TEXTURE.get_headers() missing 1 required positional argument: 'scale'
```

`TEXTURE.get_headers(cls, scale, matrix_name=None)` (`schema/_texture.py:160`) is
parameterized because its columns carry the offset (`Texture_Contrast-deg000-scale05`).
Naively that puts 65 of 148 columns into Unattributed.

**Resolution:** try `get_headers()`; on `TypeError`, match the frame's columns against
`info.category()` and record the group as resolved by category rather than exact headers.
Total, no core change, covers any future parameterized schema. The deliverables README
generator does not hit this because it documents members, not headers.

Executed on this run, 148 of 148 columns resolve:

| Group | Columns | Resolved |
|---|---|---|
| `MeasureShape` | 17 | exact |
| `MeasureColor` | 15 | exact — `ColorLab` + `ColorHSV` only, since the run sets `include_XYZ=False, include_xy=False` |
| `MeasureIntensity` | 12 | exact |
| `MeasureNeighborDist` | 8 | exact — emits `GridSpatial_*`, not a `NeighborDist_*` prefix |
| `MeasureTexture` | 65 | by category |
| Metadata | 16 | `is_metadata_header()` |
| Unattributed | 15 | `Object_Label`, `Bbox_*`, `Grid_*` |

`MeasureNeighborDist` emitting `GridSpatial_*` is why the grouping asks the measurer
rather than parsing the column name.

Measurer classes are on `phenotypic.measure`, not the top-level namespace:
`getattr(phenotypic, "MeasureShape")` raises.

A zero-argument `MeasureFeatures.emitted_headers()` on the ABC would make this exact rather
than heuristic. That is a core change and belongs in its own proposal.

## 9. Configuration surface

A popover anchored to the tab-bar actions strip.

| Control | Binds to | Options from | Default |
|---|---|---|---|
| Section group label | one PDF page / one on-screen section | `selectable_axis_columns(max_cardinality=None)` | first metadata column with 2–50 values |
| Figure row label | facet rows | `selectable_axis_columns(max_cardinality=12)` | none -> single row |
| Figure column label | facet columns | `selectable_axis_columns(max_cardinality=12)` | none -> single column |
| Y-axis | point y | numeric columns of the filtered frame | first numeric measurement present |
| X-axis | point x | numeric columns + derived frame index | `Metadata_FrameIndex` if present, else derived |
| Hue | colour + legend | `selectable_axis_columns(max_cardinality=8)` | none -> single series |
| Shape | marker + legend | `selectable_axis_columns(max_cardinality=6)` | none -> circles |
| Sizing | section/facet/axis/tick/legend type sizes, marker size, opacity, facet height | steppers | DESIGN.md §06 |
| Legend | corner, expanded/collapsed, move-to-bottom-on-export | — | bottom-right, expanded, on |
| Curation | show removed colonies as grey x | toggle | on |

`selectable_axis_columns` is `colony_view/_grid.py:201`. Palette is `OKABE_ITO` from
`gui/_design.py`, applied in the DESIGN.md §06 series order.

**Legend.** A floating panel that snaps to whichever of the four corners it is dropped
nearest, collapsible to a pill. On export it leaves the corner and lays out along the
bottom of every page, matching the reference script's `fig.legend(loc="lower center")`.

## 10. Derived frame index

When the X-axis selects "frame index from capture order": rank distinct
`Metadata_ImageDatetime` within `Metadata_PlateID`, ascending, zero-based. Rank on the
parsed datetime, not the image name. Images with a null datetime are excluded from the
ranking and from the plot, with the count surfaced in the pager chip row.

## 11. Export

`kaleido` renders one PDF page per section group; `pypdf` merges them. No cover page
(dropped). Page size is a control, default 16x12 in, matching the reference script's
`figsize`.

`pypdf` is the only new dependency: `uv add pypdf`.

## 12. Spikes — before any layout code

**Spike A — how many gl subplots survive in one figure.** Build a `make_subplots` figure
of N panels, each a `go.Scattergl` trace, and raise N until the browser drops contexts.
3 is known good (`_measure_orientation_zones.py`); 16 is the number in question.
Output: the value of `SCATTER_FACET_CAP`, and whether the single-axes fallback is needed.

**Spike B — does kaleido export a gl layer headless.** Kaleido runs headless Chromium on
the compute node: no display, no GPU, software GL at best. Render one Maresca section to
PDF there. Output: points present or absent, and raster or vector. Absent reopens Q2;
raster is a documentation note unless vector artwork is required.

Spike B runs first — it is the only open item that can reopen a locked decision.

## 13. Testing

- `_figure.py` unit tests against synthetic frames, no Dash (pattern:
  `tests/unit/gui/results_viewer/test_heatmap_figure.py`).
- Facet planning: empty cells, over-cap grids and the "showing first N of M" notice,
  single-row and single-column degenerate cases.
- Click resolution: an index round-trips to the correct `(dataset, stem, Object_Label)`
  under a filtered and re-sorted frame.
- Grouping: the §8 table reproduced against a fixture pipeline, including the
  parameterized-`get_headers` fallback.
- Derived frame index: ranks within plate, ties, nulls excluded.
- P0 crop tests (§2.5).
- `FEATURES.md` / `WORKFLOWS.md` ledgers and a tutorial capture — see the
  `gui-tutorial-capture` skill. Highest tutorial page is currently `18_browse.md`.

## 14. Scope

**v1:** shared axis ranges; curation-aware points (toggle); derived frame index; jitter for
categorical X; floating legend. Plus P0.

**v2:** growth-curve overlay (`LogGrowthModel`); mean +/- band per hue; adaptive WebGL
switching; save/load figure config; copy-as-Python; run-level display scale; crop contrast
polish.

**Dropped:** box-select bulk curate (Q4); provenance cover page; marginal histograms.

## 15. Risks

| Risk | Mitigation |
|---|---|
| 16 gl subplots exceed the browser context cap | Spike A; single-axes fallback specified |
| kaleido cannot render gl headless | Spike B first; reopens Q2 if it fails |
| PDF point layer is raster, not vector | Spike B measures it; documentation note unless vector is required |
| Per-image display scale shifts brightness across prev/next | Stated limitation; run-level scale is v2 |
| The verification run cannot draw the reference figure | Re-run with `--metadata`; no tab change needed |
