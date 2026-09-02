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

## 1. Verification fixture and measured facts

**The fixture is
`projects/ucr_029_e_d_Maresca/data/results/2026-08-11-migration-test/`** — 36
migrated OME-Zarr stores sampled from the 2026-08-11 run, with its own
`deliverables/`. Every number below is read from it. The full run is referenced
only where scale is the point; it is not a test target.

| Fact | Fixture |
|---|---|
| Rows in `measurements.parquet` (the mirror) | 844 |
| Plottable colony rows | 723 |
| Metadata-only phantoms (`QC_MetadataOnly`) | 121 |
| Rows in `master_measurements.parquet` | 723 |
| Master keys absent from the mirror | **0** |
| Columns | 149 (148 in a per-store table, which has no `QC_MetadataOnly`) |
| Stores | 36 |
| Strains | **22** (23 distinct `Metadata_Strain` values counting the 82 nulls as one; a null is dropped rather than becoming a section, so 22 is the number the pager shows) |
| Plates | 28 |
| Images | 36 |
| `Metadata_ImageDatetime` | 32 unique, **81 null** |
| Measurers in `pipeline.json["meas"]` | `MeasureNeighborDist`, `MeasureShape`, `MeasureIntensity`, `MeasureColor`, `MeasureTexture` |
| `rgb/0` | `[3, 3132, 5086]` uint16, sharded, inner chunk `[1, 1024, 1024]` |
| `rgb/0` on disk | 79.8 MB over 60 inner chunks (~1.33 MB each) |
| `rgb/labels/objmap/0` on disk | 0.0 MB |
| Pyramid | 5 levels, `stop_px: 512`; `rgb/4` is `[3, 196, 318]`, `rgb/3` is `[3, 392, 636]` |
| Label path | `attributes.phenotypic.labels` = `{"objmap": "rgb/labels/objmap"}` |

Scale reference, from the full run only: 231,229 mirror rows, 113,814 plottable,
6,657 images, 111 plates, **22 strains** (900 null, 23 counting the null --
measured on the full run, not carried across from the fixture; the two agree). Q5's ">500k points" is a stated design target above both.

### 1.1 Phantoms are the fixture's own behaviour

`QC_MetadataOnly` partitions the mirror exactly — true for precisely the rows with
a null `Object_Label` (verified as an identical partition in the fixture and in
the full run).

**The tab plots only rows where `QC_MetadataOnly` is false.** A phantom has no
colony, no coordinates and no crop, so it cannot become a point. The filter is
applied at plot time, not at ingest, so the shared filter offcanvas keeps
reporting the same totals as Plate and Colony; the pager chip row reports
"723 of 844 rows plottable".

The proportion is what varies, not the rule: 14% of the fixture, 51% of the full
run.

### 1.2 The master/mirror row gap does not occur in the fixture

Master-only keys in the fixture: **0**. Master and mirror agree exactly on 723
plottable rows.

The full run has a 14,784-row gap that is *not* explained by curation
(`deliverables/qc/` is 512 bytes, one lock file), duplicates (0), relabeling
(label sets match where counts match) or `KeepSectionLargest` (14,549 of the
missing rows are the only object in their grid cell). The dropped rows are ~5.5x
smaller by area and no surviving row sits in grid row 0 or 7.

**That is a full-run phenomenon and is out of scope here.** It is recorded because
Scatter plots the mirror, so any run with the gap inherits it — but it is a
question about that run's mirror-write path, not about this tab, and the fixture
does not exhibit it.

### 1.3 What the fixture forces

- **`Metadata_FrameIndex` is absent** and `Metadata_Timepoint` is unusable, so the
  derived frame index is required. `Metadata_ImageDatetime` is 1:1 with image and
  is what the ranking uses.
- **81 of 844 rows have a null `Metadata_ImageDatetime`**, so §10's null guard is
  *exercised* by the fixture rather than merely defensive. `Metadata_Strain` is
  null for 82 rows.
- **`Grid_RowNum` and `Grid_ColNum` are `String`, not integers.** So are most
  metadata columns. Facet and section ordering must sort numeric-looking
  categorical values **numerically when every value parses, lexically otherwise** —
  a plain string sort puts `Grid_ColNum` in the order 0, 1, 10, 11, 2, 3, which is
  wrong and would look like a rendering bug rather than a sort bug.
- **`Size_*` does not exist**; the pipeline runs `MeasureShape`, so area is
  `Shape_Area`. Axis defaults resolve from the run, never from a hard-coded name.
- **The fixture is sparse by construction**: 22 strains across 36 images, median 32
  plottable rows per strain. Good for exercising empty facets, sparse grids and
  the "no data" cell; it will not look like the reference figure, and should not
  be expected to.

**Neither this fixture nor the full run can reproduce the reference figure**, and
that is correct under Q6: there is no pH, salinity, biological-replicate or
configuration column: those live only in `UCR_029_E_D-Metadata.csv`. Faceting here
is on strain, plate, day and session. Re-running with those conditions passed
through `--metadata` makes the reference figure available with no change to the
tab.

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

Scale, against a range computed **per image, per request — with no
cross-request cache**.

```python
# _shared/tiles.py — _store_layer_array_to_rgb
- if layer == "rgb":
-     return arr.astype(np.uint8)          # mod-256 truncation
+ if layer == "rgb":
+     lo, hi = image_display_range(store, layer)
+     return scale_to_uint8(arr, lo, hi)   # scale, then clip
```

**Revision 3 withdraws the cache entirely.** Revisions 1 and 2 specified a
cross-request cache — first on `(store path, mtime_ns)`, then, on review, on
`store_generation_token`. Both are wrong, and the second is wrong for a reason
its own docstring states (`_zarr_routes.py:146-150`):

> The token deliberately keys on the root `zarr.json` **only**. An in-place
> nested-chunk rewrite moves neither the store directory's `st_mtime_ns` nor the
> root, so the token does not move ... which is correct, because the route holds
> no cache.

That last clause is the whole point: the token is sound *because nothing caches
on it*. Introducing a display-range cache keyed on it would break the premise
that makes it correct, and reintroduce exactly the S5 failure — a key that
invalidates on full re-publish but not on chunk rewrite, passing every test one
would write and serving a silently wrong brightness in production.

**Measured, so the trade is not a guess:**

```
read whole rgb/4 (the range computation)   4.4 ms   min of 5
the 256 px level-0 crop the request does  164.6 ms
```

The range is **3% of a request the route already pays**. A cache that saves 3%
is not worth a correctness hazard with no available sound key. Compute it per
request; revisit only if profiling shows it matters.

`image_display_range(store, layer)` reads the smallest pyramid level whole and
returns its **min/max, not a percentile**: 0.5/99.5 on `rgb/4` gives
21,644–25,993 against a true range of 17,912–45,344, which clips the colonies —
the brightest thing in the frame and the subject of the picture. Clip on apply
instead.

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
- Two crops of the same image, taken from different windows, map an identical
  source value to an identical output — the per-image range must not vary with
  the window. (This is what the withdrawn per-crop stretch got wrong.)

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
render carries a section, not the run. Even at full-run scale that is
113,814 / 22 ≈ 5,173 rows, and at roughly 100 bytes of strings per point ~0.5 MB
— not tens of megabytes. In the fixture it is 723 rows total.

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
  generalizing that module the way section (F)'s tile-bridge `BRIDGES` table
  already parameterizes two surfaces — real
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
smaller number (22 pages is fine; 500 is not). Whether `Texture_` should also join
`_MEASUREMENT_PREFIXES` is §16 — it would fix every consumer but changes the
Colony grid's axis options too.

**Legend.** A floating panel that snaps to whichever of the four corners it is dropped
nearest, collapsible to a pill. On export it leaves the corner and lays out along the
bottom of every page, matching the reference script's `fig.legend(loc="lower center")`.

## 10. Derived frame index

When the X-axis selects "frame index from capture order": rank distinct
`Metadata_ImageDatetime` within `Metadata_PlateID`, ascending, zero-based. Images
with a null datetime are excluded from the ranking and from the plot, with the
count surfaced in the pager chip row. (The fixture has 81 such rows, so this
path is exercised rather than merely defensive.)

The improvement over the reference script is that `Metadata_ImageDatetime` exists
as a column; the ranking logic is identical. The script parses the datetime out of
the filename and then sorts on the parsed datetime
(`plot_strain_growth_scatter.py:110-115`) — it does not rank on the name, and
revision 1 said otherwise.

**Null grouping values** follow one rule everywhere: a null section-group, facet-row
or facet-column value is **dropped**, with the row count surfaced in the pager chip
row — never a silent omission and never a "(none)" page. In the fixture `Metadata_Strain` is null for 82
rows, so this path is exercised. (In the full run the equivalent 900 rows are all
phantoms — 0 carry an `Object_Label` — and §1.1's predicate removes them first.)

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
code 0, a valid well-formed PDF. A green CI job and a clean 22-page PDF of empty
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
section is drawn from a live frame; the export path draws one section per page —
723 rows in the fixture, ~5,173 at full-run scale. Measured with `go.Scatter` in a 4x4
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
| Display-range cache serves a stale brightness | **Eliminated** — there is no cache (§2.3). The range costs 4.4 ms against a 164.6 ms crop read |
| `rgb/4` under-covers the true range | Stated limitation (§2.3c); widen or read level 0 strided |
| Per-image scale shifts brightness across prev/next | Accepted for v1 (§14); run-level scale is v2 |
| The verification run cannot draw the reference figure | Re-run with `--metadata`; no tab change needed |
| A run whose mirror drops measured rows | Not present in the fixture (§1.2); a full-run phenomenon, out of scope. Scatter plots whatever the mirror holds |
| `_MEASUREMENT_PREFIXES` is wrong in both directions | `TextureGray` is not a real category; 31 real ones are missing (§16.3). Fix by derivation; changes Colony axis options |

## 16. Answers (2026-09-01) — all four resolved

### 16.1 Chrome: `plotly_get_chrome` at environment-build time

The path is not what differs — both land in a user home cache
(`~/.cache/kaleido` vs `~/.cache/ms-playwright`) and neither is inside the venv.
What differs is **ownership and availability**:

- `plotly_get_chrome` — kaleido owns the lifecycle; stable, independent of other
  tooling. Costs a ~150 MB download from Google, so it must survive HPCC egress
  policy and must enter the `uv sync` guidance or every fresh worktree exports
  blank PDFs.
- `BROWSER_PATH` — no download, but PDF export would silently depend on the
  browser Playwright vendors for the e2e suite: `playwright install` can GC the
  pinned directory, and a machine that never ran the e2e suite has nothing.

**Decided: `plotly_get_chrome`.** Two follow-ons that are now requirements, not
notes: verify the download works from a compute node before relying on it, and
document it in the environment-setup guidance beside `uv sync`.

### 16.2 The 14,784 rows: not curation, cause still unidentified

**Terminology first, since "mirror" was jargon.** The CLI writes two frames into
`deliverables/`:

| File | What it is |
|---|---|
| `master_measurements.parquet` | the **master** — the exact pre-post concatenation of the per-store embedded tables. 128,598 rows here. |
| `measurements.parquet` | the **mirror** — the master with post-operations applied and metadata-only phantoms appended. 231,229 rows here. This is what the GUI reads and curates, and what the Scatter tab plots. |

CLAUDE.md requires analysis and dashboards to read the mirror, not the master, so
Scatter inherits whatever the mirror contains.

**Four hypotheses tested and rejected:**

| Hypothesis | Result |
|---|---|
| GUI curation removed them | **No.** `deliverables/qc/` is 512 bytes — one lock file, no review state |
| Duplicate keys collapsed | **No.** Master has 128,598 rows and 128,598 unique `(image, label)` keys; 0 duplicates |
| `Object_Label` was renumbered | **No.** Where per-image counts match, the label sets are identical |
| `KeepSectionLargest` deduplicated per grid cell | **No.** Only 164 grid cells hold more than one master object, and **14,549** of the dropped rows are the *only* object in their cell |

**What the dropped rows actually look like:**

```
                     dropped        kept
Shape_Area            5,971.8    33,035.1     ~5.5x smaller
Shape_MedianRadius        9.7        23.8
Grid_RowNum          0..7 (all)   1..6 only
```

Two signatures: a strong **size bias**, and **no surviving row in grid row 0 or 7**
— the outer rows of the 8x12 plate — while columns 0 and 11 survive fine.

That is 14,784 measured colonies, **11.5% of the master**, absent from the frame
the GUI reads, on a run whose `pipeline.json.pht-pipe` has `"post": {}` and
`"filters": {}`. Something between master and mirror is applying an edge-row rule
and a size rule that the recorded config does not describe.

**This is not a Scatter question and the spec does not resolve it.** It is a
question about the run, and it wants tracing through the mirror-write path. It is
recorded here because the tab plots the mirror: if the reduction is unintentional,
every Scatter figure inherits it silently. §15 carries it as an open risk.

### 16.3 `_MEASUREMENT_PREFIXES`: derive it, and it is worse than one typo

Derived from `MeasurementInfo` subclasses, the current tuple is wrong in both
directions:

- `TextureGray` **is not a category at all** — no schema declares it. The entry is
  dead, which is why `Texture_*` was never excluded.
- **31 real categories are missing**, including `Size`, `Grid`, `Object`,
  `ColorLab`, `ColorHSV`, `RadialExpansion`, `OrientZones` and the `QC_*` family.

So the fix is to derive, not to add one string. **But a naive derivation is wrong
too:** the discoverable set includes `Metadata`, `Grid`, `Object`, `Curation` and
`Status`, which are exactly the families that *should* stay selectable as axes.
`_MEASUREMENT_PREFIXES` is an exclusion list for continuous per-object
measurements, so it must be "every `MeasurementInfo` category, minus the
metadata/identity/grouping families" — expressed by schema ownership, not by a
hand-maintained tuple and not by string prefixes.

Fix the shared helper, since a wrong prefix list is a bug wherever it is read.
That changes the Colony grid's axis options as a side effect; it needs a
`FEATURES.md` note and a look at the Colony tutorial capture.

### 16.4 Refresh: reuse the existing one, and the fingerprint is load-bearing

The machinery already exists — `BTN_REFRESH_SNAPSHOT` (`_ids.py:107`),
`STORE_PLOT_REFRESH_REVISION` (`:813`), `OutputSnapshotDescriptor.active_run`
(`_output_root.py:106`) and `active_run_is_currently_running()` (`:571`). Scatter
**subscribes to the existing refresh revision store** rather than adding a second
button, so one Refresh re-derives every surface consistently. A Scatter-local
button would let the tab disagree with Plate and Colony about which snapshot it is
showing.

On refresh the tab rebuilds the facet plan and every figure, because new images
can add section groups and facet values, and axis ranges are shared (§9) so they
must be recomputed rather than held.

**Curation can change while Scatter is open, and the run may be live.** That
settles §6.1: the `OutputSnapshotDescriptor` fingerprint beside the click index is
load-bearing, not belt-and-braces. A point clicked on a figure drawn before a
refresh resolves against a frame that no longer matches, and without the
fingerprint it opens the wrong colony silently.

Live-run specifics: a partially written run means facet values appear over time,
so the pager must tolerate its current section disappearing (fall back to the
first available and say so), and the "showing first N of M" notice must recompute
per refresh rather than being cached with the figure.

## 17. Superseded — original open questions

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
