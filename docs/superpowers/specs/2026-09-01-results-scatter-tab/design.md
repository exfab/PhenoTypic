# Results Viewer — Scatter tab

Status: revision 2 — independent review incorporated. Not implemented.
Review: `REVIEW.md` beside this file (4 blocking, 11 should-fix). Revision 2
resolves all of them; §16 carries what still needs the user.
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
| Rows in `measurements.parquet` (the mirror) | 231,229 |
| **Plottable colony rows** | **113,814** |
| Metadata-only phantoms (`QC_MetadataOnly`) | 117,415 |
| Rows in `master_measurements.parquet` | 128,598 |
| Master keys absent from the mirror | 14,784 |
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

### 1.1 Half the mirror is not plottable

The headline 231,229 is not a point count. Derived and independently reproduced:

```
  128,598  master_measurements.parquet
 - 14,784  master keys with no mirror row
 =113,814  real colony rows in the mirror
 +117,415  metadata-only phantoms
 =231,229  measurements.parquet
```

Mirror-only keys: **0** — mirror-real rows are a strict subset of master.
`QC_MetadataOnly` partitions the mirror exactly: it is true for precisely the
117,415 rows with a null `Object_Label` (verified as an identical partition).

**The tab plots only rows where `QC_MetadataOnly` is false.** This is a v1
requirement, not an optimisation: a phantom has no colony, no coordinates and no
crop, so 51% of the frame cannot become a point. The filter is applied at plot
time, not at ingest, so the shared filter offcanvas keeps reporting the same
totals as Plate and Colony; the pager chip row reports
"113,814 of 231,229 rows plottable".

Per section group: 113,814 / 23 strains ≈ **4,948 points per page**. This number
governs the export path — see §11.

The 14,784 master rows missing from the mirror are **unexplained by
configuration**: this run's `pipeline.json.pht-pipe` has `"post": {}` and
`"filters": {}`. The likely cause is GUI curation, which rewrites the mirror.
That matters beyond bookkeeping — it means the mirror is session-mutable, which
is why §6 carries a snapshot fingerprint. Open question, §16.

### 1.2 Consequences the data forces

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
(`builder/_image_renderer.py:125`). Its **code** (lines 145–148) min-max stretches
*the array it is handed*, which for a crop is that crop's own window. (Its
docstring claims integers are "rescaled by their global max"; the docstring and
the code disagree — rely on the code.)

```
window A: source 18315..31783   a true 24000 renders as 107.6
window B: source 20539..28559   a true 24000 renders as 110.0
window C: source 20445..27877   a true 24000 renders as 122.0
```

Same physical brightness, three renderings. For a gallery whose job is comparison, that
is its own bug.

### 2.3 The fix

Scale, against a range computed **per image** and cached on
`store_generation_token(store)` (`results_viewer/_zarr_routes.py`, already used by
`_store_source.py:127`).

**Not `(store path, mtime_ns)`.** `crop_colony` stats the store *directory*. A
directory's mtime moves when its dirent set changes, not when a chunk nested under
`rgb/0/c/…` is rewritten in place. It therefore invalidates on a full re-publish
and not on a chunk rewrite — the worst possible shape for a cache key, because it
passes every test one would think to write and fails in production with a silently
wrong brightness. `crop_store_rgb` is uncached today (the only `lru_cache` in
`tiles.py` is on `_load_overlay_rgb`, line 103), so this costs nothing now and
becomes live the moment a display-range cache exists. `image_display_range` needs
its own cache; `(lo, hi)` is 16 bytes, so a generous `maxsize` is free.

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

**Known limitations, stated not hidden.**

(a) 90-against-32 is correct but dim: the top of the range is set by a specular
highlight, so most of the 0–255 budget sits above the subject. Contrast polish is
a follow-on, not part of the correctness fix.

(b) A per-image scale still means two different images map differently, and
prev/next steps across images. §14 accepts this for v1 explicitly rather than
leaving it implied — a comparison surface with known brightness stepping is the
same class of defect §2.2 rejects the naive fix for, one level up. Promoting the
run-level scale is the alternative; it is listed v2.

(c) `rgb/4` is a 16x **mean** downsample (`pyramid.downsample.image == "mean"`),
and averaging contracts the range: 20,511/44,047 against a level-0 truth of
17,912/45,344. So the bottom 2,599 levels — **11% of the true span** — clip to 0
and the top 1,297 saturate. §2.3 rejects percentiles for clipping the colonies and
then adopts a proxy that clips the shadows. That is the better trade, since the
subject is bright, but it is a real third limitation. Mitigate by widening the
derived range by a margin, or by reading level 0 strided (79 ms measured).

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
boundary. This becomes a separate `?contours=` parameter. **Default `1` for the Scatter
segment, `0` for the two existing segments** (`COLONY_CROPS_URL_SEGMENT` at
`colony_view/_crop_routes.py:37` and `QC_CROPS_URL_SEGMENT` at `_app.py:294`).
Defaulting on everywhere would visibly change the Colony grid for existing users —
the composite adds focal-`Object_Label` tinting the baked overlay never had — and
would pull the `FEATURES.md` / `WORKFLOWS.md` / tutorial-capture obligations into
P0. Per-segment defaults keep P0 invisible to existing surfaces.

Resolve the label path from `attributes.phenotypic.labels`, never hard-coded.

### 2.5 P0 tests

- A synthetic uint16 store holding a monotonically increasing ramp renders monotonically
  non-decreasing. This is the invariant truncation violates; the test fails if the bug is
  reintroduced.
- A uint8 store is unchanged by the new path (`_normalize_to_uint8` already short-circuits
  on uint8; the new scale must too).
- Two non-overlapping crops of the same image map an identical source value to an
  identical output value.
- `?contours=1` on a window containing a known label emits boundary pixels;
  `?contours=0` emits none. Use a window with a real colony — object 24 of
  `d000466_280_003` — not bare agar, where objmap is uniformly 0 and the test
  cannot distinguish the two.
- The display-range cache invalidates when a nested chunk is rewritten without
  touching the store's top-level listing. This is the S5 failure mode; the test
  exists because the obvious key does not catch it.

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

`go.Scattergl` for the screen. SVG `go.Scatter` does not render at this scale, and
the pager only divides the total by the number of section groups. Adaptive
threshold switching is the deferred v2 item; always-on is strictly simpler — one
code path, no threshold.

WebGL is established practice here: `go.Scattergl` appears in
`measure/_measure_symzones.py`, `measure/_measure_orientation_zones.py` (traces at
`:2528`, inside the `make_subplots(rows=1, cols=3)` figure created at `:2190`) and
`sdk_/branch_pathfinding/_diagnostics.py:21`.

### 5.1 Facet count does not consume WebGL contexts

Revision 1 capped facets by analogy to `TIMELINE_COMPARE_CAP = 12`
(`gui/_config.py:756`), whose comment cites the ~16-context browser ceiling. **The
analogy is wrong and the cap's rationale is withdrawn.** `TIMELINE_COMPARE_CAP`
bounds 12 *independent OpenSeadragon viewers* — 12 divs, 12 canvases, 12 contexts.
One Plotly figure pools every gl trace into a single shared `gl-container`.

Measured in real chromium, counting canvases inside the graph div:

```
N= 1 subplots -> canvases 3, gl-containers 1
N= 4 subplots -> canvases 3, gl-containers 1
N=16 subplots -> canvases 3, gl-containers 1
N=36 subplots -> canvases 3, gl-containers 1
```

Three canvases, one container, at every N. Consequences:

- **The single-gl-axes fallback is cut.** Revision 1 specified collapsing the grid
  to one axes pair with per-facet coordinate offsets if 16 subplots failed. That
  was the largest piece of speculative complexity in the spec, and the contingency
  does not obtain.
- **Spike A is cut.** "Raise N until the browser drops contexts" cannot terminate,
  because N is not what consumes them.

### 5.2 The cap that is still needed

`SCATTER_FACET_CAP` survives, re-derived from what actually binds:

- **Legibility** — below roughly 200 px per panel a facet stops being readable.
  At a typical 1400 px body that is about 7 columns.
- **Point count per figure**, which is what gl cost scales with.
- **Axis and DOM count** — every subplot is still axes, ticks and labels.

Its comment must state these reasons and must not copy `TIMELINE_COMPARE_CAP`'s,
which is correct for its own case and wrong here. Over-cap renders the first N
facets **in facet-value sort order** (deterministic, and independent of the data's
distribution) plus a visible "showing first N of M" notice — never a silent
truncation.

### 5.3 Three caps, not two

Per-axis `max_cardinality` bounds each dropdown's option list. `SCATTER_FACET_CAP`
bounds rows x columns — a 12-value row axis crossed with a 12-value column axis is
144 panels, so the product is checked after both axes are chosen. And the
**section-group count is a third cap, currently unbounded** — see §9.

## 6. Click path

Each point's `customdata` is a single `int32` row index into the filtered frame. The click
callback resolves it server-side into `(dataset, stem, Object_Label)` — the key the crop
route, the Viv stage and the curation lookup already take.

This is a design requirement, not an optimisation — but the magnitude argument in
revision 1 was wrong and is corrected here. Q1 puts one section on screen, so a
render carries a section, not the run: 113,814 / 23 ≈ 4,948 rows, and at roughly
100 bytes of strings per point that is ~0.5 MB, not tens of megabytes.

The index still wins, for reasons that do not depend on the size: the resolve step
needs an index into a stable frame regardless (§6.1), and an int32 column avoids
browser memory, JSON parse cost and Dash store bloat that scale with every future
run rather than with this one.

### 6.1 The index anchors to `master_df`, not the filtered frame

A positional index into a frame re-derived on every filter or sort change is
invalidated by both. Worse, there is a race with no error path: the user changes a
filter, clicks a point on the still-rendered old figure before the new one lands,
and the callback resolves that index against the new frame. It opens the wrong
colony, silently, and the result looks entirely plausible — a real colony, a real
crop, the wrong one.

`OutputRoot` is `@dataclass(frozen=True)` (`_output_root.py:111`) and `master_df`
is captured once at `discover()`, so a positional index into it is stable for the
whole binding, costs the same four bytes, and is immune to re-filter and re-sort by
construction rather than by discipline.

**Carry the `OutputSnapshotDescriptor` fingerprint (`_output_root.py:75`) beside
the index.** `master_df`'s stability holds within one binding but not across a
curation write followed by a re-discover — and §1.1 shows the mirror *is*
session-mutable. The descriptor's `consumed_state_fingerprint` already covers "the
measurements mirror ... incorporated by an explicit Refresh", which is exactly the
signal needed. A stale index is then refused with a "the run changed, refresh"
message rather than silently mis-resolved.



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
- **Draggable width** — the QC worklist splitter (`_assets/results_viewer.js`
  section F, handler at `:833`, `clampSidebarWidth` at `:818`, clamp `[140, 380]`,
  default 180) is the right *pattern* but **is not reusable as written**: every
  identifier is hard-coded (`#qc-review-splitter`, `#qc-review-worklist`,
  `store-qc-sidebar-width`, the `_qcSplitter` idempotence flag). Reuse means
  generalizing that module the way `timeline.js` was made surface-agnostic — real
  work, scoped into v1, not a free reuse.
  Second-order, inherited: QC is not mounted, so `#qc-review-splitter` never
  enters the DOM, its `setInterval(tryAttach, 100)` never clears and its body-wide
  `MutationObserver` runs for the session's life. Pre-existing; fix while
  generalizing rather than build on it.
- **Prev / next** walks the clicked facet's points in x order.
- No **Exclude colony**. Q4 removed the write path.

## 8. Measurement grouping

Section headings are `MeasureFeatures` class names resolved from the run's own
pipeline config (`deliverables/pipeline.json.pht-pipe`, resolved via
`layout.pipeline_config_path` — never hand-joined) — the `"meas"` key, which is
separate from `pipe_cfgs`.

**Each measurer must be instantiated from its recorded params**, not used as a
class. `get_measurement_infoclasses()` is an instance method and raises on the
class, and its result is parameter-dependent:

```
MeasureColor()                                  -> [ColorLab, ColorHSV]
MeasureColor(include_XYZ=True, include_xy=True) -> [ColorXYZ, Colorxy, ColorLab, ColorHSV]
```

The params round-trip is load-bearing, not incidental. Classes resolve through
`phenotypic.measure`; `getattr(phenotypic, "MeasureShape")` raises.

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

**Resolution:** try `get_headers()`; on `TypeError`, match the frame's columns
against `info.category()` and record the group as resolved by category rather than
exact headers. Total, no core change, covers any future parameterized schema. The
deliverables README generator does not hit this because it documents members, not
headers.

This is a deliberate choice not to special-case per schema, not a necessity —
`pipeline.json["meas"]["MeasureTexture"]["params"]["scale"]` is `[5]`, sitting
right there, and could be passed. The category fallback is preferred because it
generalizes to schemas that do not exist yet.

A zero-argument `MeasureFeatures.emitted_headers()` would **not** fix this
exactly, contrary to revision 1: the emitted header set is a function of instance
params, so any such helper must stay an instance method. Withdrawn as a
recommendation.

Executed on this run, 148 of 148 columns resolve:

| Group | Columns | Resolved |
|---|---|---|
| `MeasureShape` | 17 | exact |
| `MeasureColor` | 15 | exact — `ColorLab` + `ColorHSV` only, since the run sets `include_XYZ=False, include_xy=False` |
| `MeasureIntensity` | 12 | exact |
| `MeasureNeighborDist` | 8 | exact — emits `GridSpatial_*`, not a `NeighborDist_*` prefix |
| `MeasureTexture` | 65 | by category |
| Metadata | 16 | `is_metadata_header()` |
| Curation | 1 | `QC_MetadataOnly` — grouped explicitly, see below |
| Unattributed | 15 | `Object_Label`, `Bbox_*` (10), `Grid_*` (4) |

Totals are 149 against the mirror and 148 against a per-store table, which carries
no `QC_MetadataOnly`.

`QC_MetadataOnly` gets its **own Curation group** rather than falling into
Unattributed. No measurer claims it and `is_metadata_header()` rejects it, so the
naive rule buries the single column §1.1 hinges on inside a bucket named
"Unattributed" — the worst of the available options.

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
| Section group label | one PDF page / one on-screen section | `selectable_axis_columns(max_cardinality=SECTION_GROUP_CAP)` | first metadata column with 2–50 values |
| Figure row label | facet rows | `selectable_axis_columns(max_cardinality=12)` | none -> single row |
| Figure column label | facet columns | `selectable_axis_columns(max_cardinality=12)` | none -> single column |
| Y-axis | point y | numeric columns of the filtered frame | first numeric measurement present |
| X-axis | point x | numeric columns + derived frame index | `Metadata_FrameIndex` if present, else derived |
| Hue | colour + legend | `selectable_axis_columns(max_cardinality=8)` | none -> single series |
| Shape | marker + legend | `selectable_axis_columns(max_cardinality=6)` | none -> circles |
| Sizing | section/facet/axis/tick/legend type sizes, marker size, opacity, facet height | steppers | DESIGN.md §06 |
| Legend | corner, expanded/collapsed, move-to-bottom-on-export | — | bottom-right, expanded, on |
| Curation | show removed colonies as grey x | toggle | on |

`selectable_axis_columns` is `colony_view/_grid.py:201`. Palette is `OKABE_ITO`
from `gui/_design.py:280`, applied in the DESIGN.md §06 series order.

**The section-group control needs a cap** (§5.3's third cap). Revision 1 passed
`max_cardinality=None`, which is unbounded. That matters here because
`_MEASUREMENT_PREFIXES` (`colony_view/_grid.py:93`) lists `TextureGray_` while
`TEXTURE.category()` is **`Texture`**, so this run's 65 continuous `Texture_*`
columns are not excluded by name — only incidentally, by the default
`max_cardinality=50`. With no cap they become selectable section groups, and
choosing one asks for up to 113,814 sections: that many PDF pages and pager steps.

`SECTION_GROUP_CAP` bounds it, with a confirm-before-export threshold above a
smaller number (23 pages is fine; 500 is not). Whether `Texture_` should also join
`_MEASUREMENT_PREFIXES` is §16 — it would fix every consumer but changes the
Colony grid's axis options too.

**Legend.** A floating panel that snaps to whichever of the four corners it is dropped
nearest, collapsible to a pill. On export it leaves the corner and lays out along the
bottom of every page, matching the reference script's `fig.legend(loc="lower center")`.

## 10. Derived frame index

When the X-axis selects "frame index from capture order": rank distinct
`Metadata_ImageDatetime` within `Metadata_PlateID`, ascending, zero-based. Images
with a null datetime are excluded from the ranking and from the plot, with the
count surfaced in the pager chip row. (This run has 0 such nulls — the guard is
defensive and unexercised by the verification data.)

The improvement over the reference script is that `Metadata_ImageDatetime` exists
as a column; the ranking logic is identical. The script parses the datetime out of
the filename and then sorts on the parsed datetime
(`plot_strain_growth_scatter.py:110-115`) — it does not rank on the name, and
revision 1 said otherwise.

**Null grouping values** follow one rule everywhere: a null section-group, facet-row
or facet-column value is **dropped**, with the row count surfaced in the pager chip
row — never a silent omission and never a "(none)" page. On this run the question
is moot: `Metadata_Strain` is null for 900 rows and **all 900 are phantoms**, so
§1.1's predicate removes them first. (The review stated these were real measured
colonies; independently, 0 of the 900 carry an `Object_Label`.)

## 11. Export

`kaleido` renders one PDF page per section group; `pypdf` merges them. No cover
page. Page size is a control, default 16x12 in, matching the reference script.

### 11.1 Export swaps the trace type — kaleido cannot render `Scattergl`

**Measured, on compute node `i38` (`intel`), no display, no GPU; plotly 6.6.0,
kaleido 1.2.0, choreographer 1.2.1.** Identical data through both trace types:

```
gl.png    non-white   624   dark    289
svg.png   non-white 46886   dark  36608
```

624 non-white pixels is the axis frame, ticks and labels — the identical count for
a figure with **zero traces**. The gl marker layer contributes nothing, flat at
~180 ink per panel for every subplot count from 1 to 16. Root cause, confirmed
independently: that headless chromium reports `webglAvailable: false` even with
swiftshader flags, so plotly.js's regl backend has no context and **fails soft**.

**It fails silently.** No warning (`catch_warnings(record=True)` is empty), exit
code 0, a valid well-formed PDF. A green CI job and a clean 23-page PDF of empty
axes. This shapes §13.

Six mitigations were tested and rejected: `fig.write_image` is the same path
(plotly 6 has no engine choice); `enable_gpu=True` leaves ink at 289;
`--use-gl=swiftshader` is unreachable (choreographer accepts only
`enable_gpu`/`headless`/`enable_sandbox`/`tmp_dir`); `--enable-unsafe-swiftshader`
is already default and does not help; `plotly.io.kaleido.scope` was the v0 API and
is gone; rasterize-and-embed is impossible because there are no pixels to
rasterize.

**Resolution: build the figure with `Scattergl`, and substitute `Scatter` for the
export pass only.** One trace-type swap at the PDF boundary; the `FigureSpec` and
the figure construction are unchanged, so the PDF is still the same figure.

This does not reopen Q2. Q5's ">500k points" governs the **screen**, where a
section is drawn from a live frame; the export path draws one section per page,
which for this run is ~4,948 points. Measured with `go.Scatter` in a 4x4
`make_subplots` at 1600x1200: **2.8 s and 0.10 MB per page** at 5,000 points. No
downsampling, so the PDF is not a quiet lie about the figure.

If a future run makes a single section large enough that SVG export stalls, that
is the point to revisit — not now.

### 11.2 Chrome is an undeclared prerequisite

Independent of 11.1, and it blocks export entirely today. On this node
`google-chrome`, `chromium` and `chromium-browser` are absent from `PATH` and
`~/.cache/kaleido` does not exist; plain `write_image` raises
`RuntimeError: Kaleido requires Google Chrome to be installed`. §12's "kaleido
runs headless chromium on the compute node" had nothing to run — every kaleido
measurement above came from pointing `BROWSER_PATH` at the browser Playwright
vendors for the e2e suite
(`~/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome`).

Two options, and this needs a decision (§16): reuse the vendored Playwright
browser via `BROWSER_PATH` (no download, but couples PDF export to the e2e browser
cache and its version pinning), or run `plotly_get_chrome` at environment-build
time (~150 MB from Google, subject to HPCC egress policy, and it must enter the
`uv sync` guidance or every fresh worktree breaks).

Note the two failures do not mask each other: a missing Chrome raises loudly,
while 11.1 fails silently.

`pypdf` remains the only new dependency: `uv add pypdf`.

## 12. Spikes — resolved

Both spikes from revision 1 are closed. Neither survives as work.

**Spike A (gl subplot ceiling) — cut.** It measured nothing: facet count does not
consume WebGL contexts (§5.1). One `gl-container` at N = 1, 4, 16 and 36.

**Spike B (kaleido + gl headless) — answered, and it was the bad answer.** §11.1.
It reshaped the export path rather than reopening Q2.

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
- **Export tests assert on rendered ink, never on file existence.** §11.1's failure
  mode is a valid, well-formed, entirely blank PDF produced with exit code 0 and no
  warning. A test that checks the file exists, or that it has pages, passes against
  a broken export. Rasterize a page and assert a dark-pixel count above a floor;
  the measured separation is 289 against 36,608, so the threshold is not delicate.
- A guard that the export path never emits a `Scattergl` trace — the substitution
  in §11.1 is a correctness requirement, not an optimisation, and nothing else
  would catch its removal.
- Phantom exclusion: a frame with `QC_MetadataOnly` rows plots exactly the
  non-phantom count, and the pager reports both numbers.
- A stale click index — one captured before a curation write and re-discover — is
  refused, not resolved (§6.1).
- `FEATURES.md` / `WORKFLOWS.md` ledgers and a tutorial capture — see the
  `gui-tutorial-capture` skill. Highest tutorial page is currently `18_browse.md`.

## 14. Scope

**v1:** shared axis ranges; curation-aware points (toggle); derived frame index;
jitter for categorical X; floating legend. Plus P0. Added by review: the
phantom-row predicate (§1.1, not optional), generalizing the splitter module
(§7), and the `Scatter`-for-export substitution (§11.1).

**Cut by review:** the single-gl-axes fallback and spike A (§5.1) — the
contingency does not obtain.

**Accepted in v1, explicitly:** prev/next brightness stepping across images
(§2.3 limitation b). The run-level display scale that would fix it stays v2. This
is a conscious trade, recorded so it is not rediscovered as a bug.

**v2:** growth-curve overlay (`LogGrowthModel`); mean +/- band per hue; adaptive WebGL
switching; save/load figure config; copy-as-Python; run-level display scale; crop contrast
polish.

**Dropped:** box-select bulk curate (Q4); provenance cover page; marginal histograms.

## 15. Risks

| Risk | Status |
|---|---|
| kaleido cannot render gl headless | **Confirmed** (§11.1). Resolved by the export-time trace substitution |
| A blank export passes a naive test | **Live.** Mitigated by the ink assertion in §13 — the only defence, since nothing else signals |
| Chrome absent on the compute node | **Confirmed** (§11.2). Needs a decision, §16 |
| 16 gl subplots exceed the context cap | **Withdrawn** (§5.1). Facet count does not consume contexts |
| A stale click index resolves to the wrong colony | Mitigated by the `master_df` anchor plus snapshot fingerprint (§6.1) |
| Display-range cache serves a stale brightness | Mitigated by `store_generation_token` (§2.3) |
| `rgb/4` under-covers the true range | Stated limitation (§2.3c); widen or read level 0 strided |
| Per-image scale shifts brightness across prev/next | Accepted for v1 (§14); run-level scale is v2 |
| The verification run cannot draw the reference figure | Re-run with `--metadata`; no tab change needed |
| 14,784 master rows missing from the mirror, cause unknown | Open (§16). Bounds how far §1.1's arithmetic can be trusted on other runs |

## 16. Open questions for the user

Everything else the review raised is resolved above. These four need a decision or
information I do not have.

1. **Where does Chrome come from on the compute node?** (§11.2) Reuse the
   Playwright-vendored browser via `BROWSER_PATH` — zero download, but PDF export
   inherits the e2e browser cache and its pinning — or `plotly_get_chrome` at
   environment-build time, ~150 MB from Google, subject to HPCC egress policy and
   requiring a `uv sync` guidance change. My lean: `BROWSER_PATH` with an explicit
   error when the cache is absent, since it adds no new download path.

2. **What removed the 14,784 measured rows from the mirror?** (§1.1) This run has
   empty `post` and `filters`, so configuration does not explain it. If it is GUI
   curation, that confirms the mirror is session-mutable and settles §6.1's
   fingerprint requirement. If it is something else, §1.1's arithmetic may not
   generalize to other runs.

3. **Should `Texture_` be added to `_MEASUREMENT_PREFIXES`?** (§9) It is missing —
   the tuple lists `TextureGray_` while `TEXTURE.category()` is `Texture`. Adding
   it fixes every consumer, but also changes the Colony grid's axis options for
   existing users. The alternative is a Scatter-local filter, which leaves the
   inconsistency in place. My lean: fix the shared tuple, since a wrong prefix is
   a bug wherever it is read.

4. **Can curation mutate while Scatter is open?** (§6.1) The Colony tab is
   mounted, so probably yes. If it cannot, the `master_df` anchor suffices alone
   and the snapshot fingerprint is belt-and-braces; if it can, the fingerprint is
   load-bearing. I have specified it as load-bearing.
