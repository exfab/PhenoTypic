# Independent review — Results Viewer Scatter tab

Reviewer: adversarial spec review, analysis only.
Subject: `docs/superpowers/specs/2026-09-01-results-scatter-tab/design.md`
Branch: `feat/results-scatter-gui` @ `1c6c5e16`, worktree
`/bigdata/exfab/anguy344/PhenoTypic/.worktrees/results-scatter-gui`.

Everything below was re-derived against the real codebase and the real data, not
against the spec's prose. The VERIFIED table is omitted at the requester's
instruction; in summary, **every one of the spec's ~15 `file:line` code citations
resolves to the claimed construct**, off by at most one line in four cases (listed
under NITS), and the store-geometry and column-grouping measurements in §1 and §8
reproduce. The findings below are what did *not* survive.

Environment for every measurement in this document: Slurm job `27994238`,
**compute node `i38`, partition `intel`**, 16 cores, no display, no GPU.
plotly `6.6.0`, kaleido `1.2.0`, choreographer `1.2.1`, zarr/polars from the
worktree `.venv`.

---

## BLOCKING

### B1. Kaleido renders `Scattergl` as blank axes — silently

Spike B's answer, obtained now rather than deferred.

**Exact reproduction:**

```python
import os, numpy as np
os.environ["BROWSER_PATH"] = "/rhome/anguy344/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome"
import plotly.graph_objects as go, kaleido
from PIL import Image
rng = np.random.default_rng(0)
for name, cls in [("gl", go.Scattergl), ("svg", go.Scatter)]:
    fig = go.Figure(cls(x=rng.normal(size=5000), y=rng.normal(size=5000),
                        mode="markers", marker=dict(size=4, color="black")))
    fig.update_layout(width=800, height=600, showlegend=False,
                      paper_bgcolor="white", plot_bgcolor="white")
    kaleido.write_fig_sync(fig, f"{name}.png")
    a = np.asarray(Image.open(f"{name}.png").convert("L"))
    print(name, "non-white", int((a < 250).sum()), "dark", int((a < 128).sum()))
```

```
gl   non-white   624   dark    289
svg  non-white 46886   dark  36608
```

**Literally nothing renders, not a partial render.** 624 non-white pixels is the
axis frame, ticks and tick labels — the identical count for a figure carrying zero
traces. The gl marker layer contributes 0 pixels.

It is not a subplot-count effect. Sweeping gl panel count `N = 1, 2, 3, 4, 6, 8,
12, 16` in a `make_subplots` grid gives a flat **~180 ink px per panel at every
N** — axes only, no points, at any N including 1.

**No warning and no error surfaced.** `warnings.catch_warnings(record=True)`
returns `[]`. Exit code 0. A valid, well-formed PNG/PDF is produced. At
`logging.basicConfig(level=logging.DEBUG)` the only chromium output is
post-shutdown pipe noise (`devtools_pipe_handler.cc:274 Could not write into
pipe`, a teardown `BrokenPipeError`) — nothing about WebGL, nothing during render.

**This silence is the dangerous part: a green CI job and a clean 23-page PDF of
empty axes.** Any regression test here must assert on rendered ink, never on "the
file exists" or "no exception was raised".

**Root cause, confirmed independently:** the headless chromium reports
`webglAvailable: false` from a Playwright page even with swiftshader flags. With
no WebGL context, plotly.js's regl backend has nothing to draw into and fails soft
rather than throwing.

#### Mitigations attempted

| Attempt | Result |
|---|---|
| `fig.write_image(...)` (plotly's own path) | Same code path. plotly 6 has no engine choice left — kaleido is the only engine |
| `kaleido.Kaleido(n=1, enable_gpu=True)` (drops `--disable-gpu`) | ink **289** — unchanged |
| Custom chrome flags `--use-gl=swiftshader` / `--use-angle=swiftshader` | **Not reachable.** `RuntimeError: Chromium.get_cli() received invalid args: dict_keys(['args'])` — choreographer accepts only `enable_gpu`, `headless`, `enable_sandbox`, `tmp_dir` |
| `--enable-unsafe-swiftshader` | Already in choreographer's default CLI (`choreographer/browsers/chromium.py::get_cli`) and does not help |
| `plotly.io.kaleido.scope` settings | **Does not exist in kaleido 1.x.** `scope` was the v0 API and is gone |
| Rasterize the gl layer, embed as an image | **Not possible** — gl produces no pixels headless, so there is nothing to rasterize. Would mean replacing plotly's renderer (datashader-style server-side raster + `go.Image`); large scope |
| **`Scattergl` → `Scatter` for the export pass only** | **Works.** Full fidelity, true vector output, timings below |

#### The Q2 × Q5 conflict is narrower than it looks — recommendation

The two locked decisions only conflict if Q5 is read as governing the *export*
path. It need not be. **Q1 already decided that a section is the unit**: one
section on screen, one section per PDF page. So the per-page point count is a
section's, not the run's.

For this run: 113,814 plottable rows (see B4) / 23 strains ≈ **4,948 points per
page**. Measured export cost with `go.Scatter` in a 4×4 `make_subplots` at
1600×1200:

```
  5,000 pts/page ->  2.8 s   0.10 MB   <- this run's realistic load
 20,000 pts/page ->  4.0 s   0.38 MB
100,000 pts/page -> 10.6 s   1.82 MB
```

A full 23-page export at realistic load is **~64 s**; even at a pessimistic
100k/page it is ~4 min. SVG is not the bottleneck the spec assumed. That
assumption is true for the *interactive* surface — where every pan and zoom
redraws — and false for a one-shot export.

**Recommendation, in priority order:**

1. **Keep Q2 and Q5 both, and scope Q5 to the screen.** `FigureSpec` gains one
   field: the trace constructor. `go.Scattergl` for `dcc.Graph`, `go.Scatter` for
   the kaleido pass. Every other figure input stays identical, so §4's "the PDF
   cannot drift from the screen" survives with one sentence amended — the trace
   *type* is the single permitted difference, and it is the only one, enforced by
   construction. This preserves the substance of both locked decisions, keeps one
   renderer, keeps `pypdf`, and yields true vector output, which §15 listed as an
   open risk and which is now resolved in the good direction for free.
2. **Add an ink guard, because the failure is silent.** A page whose expected
   point count is > 0 must render above an ink threshold, asserted in the export
   test. Without it, a future refactor that puts gl back on the export path ships
   empty PDFs with a green suite.
3. **Only if (1) fails on some figure you need:** matplotlib `PdfPages`. Proven
   for this exact figure — the reference script already produced
   `deliverables/strain_growth_scatter.pdf` with it — and needs no `pypdf`. But
   it means two renderers and real drift risk. Fallback, not plan.

**One verification item for path (1).** I confirmed *Python-side* that `Scatter`
and `Scattergl` accept the same 14 marker symbols (plotly.py shares the schema).
plotly.js's *runtime* symbol support has historically been narrower for gl. Before
wiring §9's Shape control (`max_cardinality=6`), render the chosen 6-symbol set
both ways and diff. A symbol that silently falls back to a circle on screen but
draws correctly in the PDF is exactly the screen/PDF drift §4 promises against.

---

### B2. Chrome is not installed. Kaleido cannot export anything today.

Independent of B1, and the reason B1's reproduction carries a `BROWSER_PATH`
override. Plain `fig.write_image(...)` on this node fails outright:

```
RuntimeError:
Kaleido requires Google Chrome to be installed.
Either download and install Chrome yourself ... or ... $ plotly_get_chrome
```

Searched for a usable binary: `google-chrome`, `chromium`, `chromium-browser` are
all absent from `PATH`; `~/.cache/kaleido` does not exist. The **only** Chrome on
this machine is the one Playwright vendors for the repo's e2e suite:

```
~/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome
~/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome
```

I obtained every kaleido result in this report by setting
`BROWSER_PATH=~/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome`.

§12 states "Kaleido runs headless Chromium on the compute node". There is nothing
for it to run. This is an **undeclared runtime prerequisite** that the spec must
resolve one way or the other:

- reuse the vendored Playwright browser via `BROWSER_PATH` (zero download, but
  couples PDF export to the e2e browser cache and its version pinning); or
- `plotly_get_chrome` at environment-build time (~150 MB pulled from Google,
  subject to HPCC egress policy, and it must be part of `uv sync` guidance or
  every fresh worktree breaks).

Note this also means a naive CI or test that calls `write_image` will fail with a
*different*, loud error than B1's silent one — so the two failures do not mask
each other, but they must both be fixed for export to work at all.

---

### B3. `SCATTER_FACET_CAP`'s WebGL rationale is a category error

§5 caps facets because "browsers cap live WebGL contexts (~16 in Chrome)", by
analogy to `TIMELINE_COMPARE_CAP`. **The analogy does not hold.**
`TIMELINE_COMPARE_CAP` caps 12 *independent OpenSeadragon viewers* — 12 divs, 12
canvases, 12 contexts. One Plotly figure pools every gl trace into a single shared
`gl-container`.

Measured in real chromium via Playwright, counting canvases inside the graph div:

```
N= 1 subplots -> canvases: 3, gl-containers: 1
N= 4 subplots -> canvases: 3, gl-containers: 1
N=36 subplots -> canvases: 3, gl-containers: 1
```

Three canvases, one container, at every N. Consequences:

- **Spike A as written measures nothing.** "Raise N until the browser drops
  contexts" cannot terminate, because N is not what consumes contexts. Kill the
  spike or repoint it at point count.
- **Cut the specified fallback.** "Collapse to a single gl axes pair with
  per-facet coordinate offsets and drawn separators — one context regardless of
  facet count" is the largest piece of contingent complexity in the spec, and the
  contingency does not obtain.
- A facet cap is still worth having — for point count, axis/DOM count, and
  legibility below roughly 200 px per panel. Re-derive the number from those and
  rewrite the constant's comment. Do not copy `TIMELINE_COMPARE_CAP`'s reasoning,
  which is correct for its own case and wrong here.

---

### B4. Half the rows the tab plans to plot carry no measurements

**Requested derivation — where the other half of 231,229 goes.**

`OutputRoot.master_df` is the **mirror**, not the master
(`results_viewer/_output_root.py:348` reads `measurements.parquet`; the master is
kept separately as `clean_master_df`). Measured on the full 2026-08-11 run:

```
deliverables/measurements.parquet         231,229 rows x 149 cols   (the mirror)
deliverables/master_measurements.parquet  128,598 rows x 136 cols   (the master)

mirror QC_MetadataOnly = true   -> 117,415 rows
mirror QC_MetadataOnly = false  -> 113,814 rows
                                   -------
                                   231,229
```

The 117,415 flagged rows have **`Shape_Area` null in all 117,415** and
**`Object_Label` null in all 117,415** — they are the metadata-only phantoms
CLAUDE.md describes ("`measurements.*` appends metadata-only phantoms once").
They are metadata entries with no measured object.

The remaining discrepancy — mirror-real 113,814 vs master 128,598 — resolves
exactly. Keying both on `(Metadata_ImageName, Object_Label)`:

```
unique keys, mirror-real : 113,814
unique keys, master      : 128,598
master-only keys         :  14,784      <- present in master, absent from mirror
mirror-only keys         :       0      <- mirror-real is a strict subset of master
```

So the full accounting is:

```
  128,598   measured objects (master)
 -  14,784   measured objects absent from the mirror
 + 117,415   metadata-only phantoms appended by the mirror
 = 231,229   mirror rows                                    <- exact
```

**Plottable rows are 113,814, not 231,229 — 49.2% of the spec's headline number.**

The 14,784 dropped measured rows are themselves unexplained by the pipeline
config: this run's `pipeline.json.pht-pipe` has `"post": {}` and `"filters": {}`,
both empty, so no post-measurement operation removed them. The most likely
explanation is **GUI curation** — `FilteredMeasurements` rewrites
`measurements.parquet` by removing `(Metadata_ImageName, Object_Label)` keys
(`_filtered_state.py` module docstring), and 14,784 curated removals on a
231k-row run is plausible. That is worth confirming, and it feeds directly into
S8 below: the mirror is *mutable by curation*, so an index into it is unstable
across a curation write.

**What the spec has to absorb:**

- §1's "Rows in the full run | 231,229" is not the point count. Say both numbers.
- §6's `int32` index will address phantom rows. A click resolving to one yields
  `Object_Label = null`, which the crop route rejects with `400 bad request: label
  must be an integer` — a dead-end inspector with a confusing error.
- Section counts, facet-emptiness logic, the "showing first N of M" notice, and
  any "showing N points" chip are all computed on the wrong denominator.
- `column_value_sets` — which drives every §9 dropdown's cardinality against its
  `max_cardinality` cap — is built over the **full** mirror, so phantom-only
  values inflate cardinality and can push a column past a cap it should clear
  (or pull a column under one it should not).

One predicate fixes the plotting, but it must be stated, tested, and every count
in §1 and §5 re-derived beneath it.

---

## SHOULD-FIX

### S1. `Texture_` is not in `_MEASUREMENT_PREFIXES`, and the section control has no cap

`_MEASUREMENT_PREFIXES` (`colony_view/_grid.py:93`) is:

```python
("Bbox_", "Shape_", "Intensity_", "TextureGray_", "SymZones_", "GridSpatial_")
```

`TEXTURE.category()` is **`Texture`**, not `TextureGray` — verified:
`TEXTURE.get_headers(5)[0] == "Texture_AngularSecondMoment-deg000-scale05"`. So
this run's 65 `Texture_*` columns are **not** excluded by
`selectable_axis_columns`. At `max_cardinality=50` they are filtered out
incidentally, by cardinality.

But §9 gives the **Section group label** control `selectable_axis_columns(
max_cardinality=None)`. With no cap, all 65 continuous float columns become
selectable section groups — and selecting one asks for up to 113,814 sections,
i.e. that many PDF pages and that many pager steps.

§5 says "Two caps interact". **There are three**, and the third is unbounded. The
section count needs its own hard cap, or a confirm-before-export threshold —
independently of whether `Texture_` is added to the prefix tuple.

### S2. §8's grouping needs an instance; the spec describes a class

`get_measurement_infoclasses()` is an **instance** method. On the class it raises:

```
TypeError: MeasureFeatures.get_measurement_infoclasses() missing 1 required
positional argument: 'self'
```

And it is parameter-dependent:

```
MeasureColor()                                  -> [ColorLab, ColorHSV]
MeasureColor(include_XYZ=True, include_xy=True) -> [ColorXYZ, Colorxy, ColorLab, ColorHSV]
```

§8's "exact — `ColorLab` + `ColorHSV` only, since the run sets
`include_XYZ=False, include_xy=False`" is therefore only reproducible if each
measurer is constructed **from its recorded `pipeline.json` params**. The spec
never says to instantiate, and the params round-trip is load-bearing.

Corollary: §8's proposed zero-argument `MeasureFeatures.emitted_headers()` on the
ABC would be **wrong for the same reason** — the emitted header set is a function
of instance params, so it must stay an instance method. That is worth saying,
since §8 currently recommends it as the future exact solution.

Also worth stating: `pipeline.json["meas"]["MeasureTexture"]["params"]["scale"]`
is `[5]`, sitting right there. So the `TypeError` → category fallback is a
deliberate choice not to special-case per-schema, not a necessity. Frame it that
way; it is a better justification than the one given.

### S3. The QC splitter is neither reusable nor live

§7: "reuses the QC worklist splitter (`_assets/results_viewer.js:796`)". Reading
the module (section F, lines 794–893), **every identifier is hard-coded**:

```js
document.getElementById("qc-review-splitter")
document.getElementById("qc-review-worklist")
dc.set_props("store-qc-sidebar-width", { data: px })
handle.dataset._qcSplitter        // idempotence flag
```

There is no parameterization and no class/data-attribute contract. "Reuse" means
generalizing the module the way `timeline.js` was made surface-agnostic — real,
unscoped work. `clampSidebarWidth` *is* exposed on the namespace for tests, as the
spec says, and the clamp is `[140, 380]` with a 180 default.

Second-order: QC is **not mounted** (neither `_layout.py` nor `_app.py` references
`_qc_tab`), so `#qc-review-splitter` never appears in the DOM. The module's
`setInterval(tryAttach, 100)` therefore never clears, and its body-wide
`MutationObserver` runs for the life of the session. Pre-existing, not caused by
this spec — but the spec proposes to build on it, so it should be fixed or
explicitly inherited.

### S4. `crop_store_rgb` has no cache today

§2.3: "cached on `(store path, mtime_ns)` — the key `crop_store_rgb` already
accepts." It accepts the *parameter*, but the only `functools.lru_cache` in
`tiles.py` is at line 103, on `_load_overlay_rgb`. `crop_store_rgb` (line 475) is
uncached. `image_display_range` needs its own cache; say so, and size it (per-image
`(lo, hi)` is 16 bytes, so a generous `maxsize` is free).

### S5. `(store path, mtime_ns)` is not a sound cache key — my view: change it

You flagged this as a self-doubt. The doubt is justified.

`crop_colony` computes `os.stat(store).st_mtime_ns` where `store` is a
**directory** — `results/<ds>/zarr/<stem>.ome.zarr/`. A directory's mtime moves
only when its own dirent set changes. It does **not** move when a chunk nested
under `rgb/0/c/…` is rewritten in place.

Today this costs nothing, because nothing is cached (S4). Introducing a
long-lived per-image display-range cache keyed on it makes the weakness live: a
reprocess that overwrites chunk data without disturbing the top-level listing
serves a stale `(lo, hi)` indefinitely, and the symptom is a subtly wrong
brightness with nothing to catch it.

It is *partially* sound — an atomic replace of the store's top-level `zarr.json`
does bump the directory mtime, so a full re-publish is caught. But "invalidates
for one kind of write and not another" is the worst possible shape for a cache
key: it works in every test you would think to write, and fails in production.

**Recommendation:** use `store_generation_token`
(`results_viewer/_zarr_routes.py`), which already exists for exactly this purpose
and is already imported by `colony_view/_grid.py`. Or key on
`rgb/<level>/zarr.json`'s mtime rather than the store directory's. Cost is one
function call; the failure mode avoided is silent.

### S6. The display range read from `rgb/4` systematically under-covers

Measured on the cited store
(`d000466_280_003_2026-07-26_06-34-47.ome.zarr`):

```
rgb/4 min/max = 20,511 / 44,047      <- exactly the IMAGE_LO/IMAGE_HI the
                                        validation script hard-codes: confirmed
§2.3 level-0 true range = 17,912 / 45,344
```

`rgb/4` is a 16× **mean** downsample (`attributes.phenotypic.pyramid.downsample.
image == "mean"`), and averaging contracts the range. So the bottom **2,599
levels — 11% of the true span — clip to 0**, and the top 1,297 saturate.

§2.3 rejects percentiles because they clip the colonies, which is correct, and
then adopts a proxy that clips anyway — in the shadows rather than the highlights.
That is a better trade (the subject is bright), but it is a real third limitation
and the section explicitly promises limitations are "stated not hidden". Add it,
or widen the range by a margin, or read level 0 strided.

### S7. §6's payload argument contradicts Q1

"Carrying the strings would send … on the order of 100 bytes per point; at 231,229
rows that is … tens of megabytes per section render."

Q1 puts **one section group on screen at a time**. 231,229 / 23 strains ≈ 10,053
rows per section → ~1 MB of strings per section render, and after B4, ~0.5 MB. Not
tens of megabytes.

The conclusion — carry an int index — is still right, for other reasons (browser
memory, JSON parse cost, Dash store size, and the fact that the index is needed
for the resolve step regardless). But the stated magnitude is inconsistent with a
locked decision in the same document.

### S8. The int32 click index — my view: anchor to `master_df`, not the filtered frame

Your other self-doubt, and I think the spec has it wrong.

A positional index into a frame re-derived on every filter/sort change is
invalidated by any filter change and any sort change. Worse, there is a live race
with **no error path**: the user changes a filter, clicks a point on the
still-rendered old figure before the new figure lands, and the callback resolves
that index against the *new* frame. It opens the wrong colony, silently, and the
result looks plausible — a real colony, a real crop, the wrong one.

`OutputRoot` is `@dataclass(frozen=True)` (`_output_root.py:111`) and `master_df`
is captured once at `discover()`. **A positional index into `master_df` is stable
for the entire binding**, costs the same 4 bytes, and is immune to re-filter and
re-sort by construction rather than by discipline.

§13's own test — "an index round-trips to the correct `(dataset, stem,
Object_Label)` under a filtered and re-sorted frame" — is precisely the test a
filtered-frame index cannot pass in the general case. The spec's test is evidence
against the spec's choice.

**One caveat that B4 surfaced.** The mirror is not immutable on disk: curation
rewrites `measurements.parquet`, and the 14,784 missing rows are most likely
exactly that. So `master_df`'s stability holds *within one `OutputRoot` binding*
but not across a curation write followed by a re-discover. Carry the
`OutputSnapshotDescriptor` fingerprint (`_output_root.py:74`) alongside the index
so a stale index is **detected and refused** rather than silently mis-resolved.
That is the difference between a correct design and one that merely usually works.

### S9. `?contours=` default-on changes two shipping surfaces

Two consumers of `register_crop_route` exist today — `_app.py:294`
(`QC_CROPS_URL_SEGMENT`) and `colony_view/_crop_routes.py:37`
(`COLONY_CROPS_URL_SEGMENT`); Scatter would be the third.

§2.4's parity argument is sound and I verified its premise: `crop_colony`
(`tiles.py:643-670`) prefers the store and only falls back to the baked overlay
PNG when there is no store or a missing layer, so contours genuinely died on
store-backed surfaces.

But the new render is not the old one. §2.4 adds **focal-`Object_Label` distinct
tinting**, which the baked overlay never had. So the Colony grid's appearance
changes visibly for existing users. That pulls the `FEATURES.md` /
`WORKFLOWS.md` / tutorial-capture obligations into **P0**, which the spec assigns
only to the Scatter commit (§13). Either scope the ledger work into P0, or default
`?contours=0` for the two existing segments and `1` for Scatter.

### S10. §8's table is derived from a different frame than §4 reads

§4 starts the data flow at `OutputRoot.master_df` = the mirror = **149** columns.
§8 groups **148** ("148 in a per-store table"). Re-running the grouping against
the mirror, constructing each measurer from its recorded params:

```
MeasureShape         17   exact
MeasureColor         15   exact
MeasureIntensity     12   exact
MeasureNeighborDist   8   exact   (GridSpatial_*)
MeasureTexture       65   by category
Metadata             16   is_metadata_header()
Unattributed         16   Bbox_*(10), Grid_*(4), Object_Label, QC_MetadataOnly
                    ---
                    149
```

Six of seven groups match the spec exactly. **Unattributed is 16, not 15** — the
extra is `QC_MetadataOnly`, which no measurer claims and `is_metadata_header()`
rejects. It is also the column B4 hinges on. Group it explicitly (a Curation
group) or state that it is deliberately Unattributed — silently landing the
tab's most semantically important column in a bucket named "Unattributed" is the
worst of the three options.

### S11. Null section-group values are unhandled

`Metadata_Strain` is null for **900 rows** (and those 900 are all non-phantom —
they are real measured colonies with no strain assigned). §10 handles nulls for
the derived frame index only.

What happens to a null section-group value — a 24th "(none)" page, or dropped with
the count surfaced in the pager chip row the way §10 does for datetime? Same
question for a null facet row or column value. Pick one and say it.

---

## NITS

**Off-by-one citations** (all resolve to the claimed construct):

- `_config.py:757` → `TIMELINE_COMPARE_CAP` is at **756** (the comment runs
  751–755).
- `schema/_texture.py:160` → `get_headers` is at **159**.
- `results_viewer/_layout.py:563-575` → the `dbc.Tabs` block is **560–577**. The
  claim ("two tabs only") is true.
- `_assets/results_viewer.js:796` → that line is inside section F's comment
  block; the handler is at **833**, `clampSidebarWidth` at **818**.

**Other:**

- §8 and §11 write `deliverables/pipeline.json`; the file on disk is
  **`pipeline.json.pht-pipe`**. Cosmetic — the spec correctly names
  `layout.pipeline_config_path` as the resolver — but it appears twice.
- §1 `rgb/4` "374 KB" → measured **296–317 KiB** across six stores (303,487 B for
  the cited one). ~20% overstated.
- §1 `rgb/4` "reads in 39 ms" → measured **5–7 ms**
  (`zarr.open_array(...)[...]`, warm and cold-ish). Errs safe.
- `crop_uint16_scaling.py` prints `truncated 19.7 vs scaled 0.1` for claim 4,
  while §2.3 quotes `85.3 -> 7.2`. Different sources (a synthetic random walk vs
  the real crop) — add one line to the script saying so, or a reader diffing the
  two will stumble. The script otherwise runs clean, exits 0, and its arithmetic
  is honest.
- §1: "the derived frame index … is cleaner than the reference script's regex over
  filenames." The reference script (`plot_strain_growth_scatter.py:110-115`)
  parses the datetime *out of* the filename and then sorts **on the parsed
  datetime**, then `groupby(PLATE_COL).cumcount()`. It does not rank on the name.
  The genuine improvement is that `Metadata_ImageDatetime` now exists as a column;
  the ranking logic is identical.
- §11: the reference script already emits a multi-page PDF via matplotlib
  `PdfPages`, with no merge step. Not a Q2 relitigation — just noting that
  `pypdf` is a cost a different renderer would not incur, which matters more now
  given B1.
- `_normalize_to_uint8`'s docstring (`builder/_image_renderer.py:126-131`) says
  integers are "rescaled by their global max"; the code (lines 145–148) min-max
  stretches. Pre-existing, but §2.2's argument leans on that docstring, so quote
  the code instead.
- §10 handles a null `Metadata_ImageDatetime`; this run has **0** nulls there
  (6,657 unique, 0 null). Defensive and fine — just not exercised by the
  verification data.

---

## SCOPE AND LAYOUT

- **P0 as its own commit is right**, and the four §2.5 tests are the right four.
  The monotonic-ramp test is a genuine regression pin: the validation script shows
  truncation produces **75 descending steps** across the measured source range
  where scaling produces **0**, so the test fails loudly if the bug returns.
  Caveat: S9 means P0 now also owns a visible Colony-grid change and its ledger
  obligations.
- **Module layout is right.** Eight files is not over-split for this. `_figure.py`
  mirroring `_heatmap_tab/_figure.py` is the correct precedent, and
  `tests/unit/gui/results_viewer/test_heatmap_figure.py` exists as the pattern
  §13 cites.
- **Cut from v1:** the single-gl-axes fallback (B3). The contingency does not
  obtain, and it is the largest piece of speculative complexity in the spec.
- **Add to v1:** the phantom-row predicate (B4). Not optional.
- **Reconsider:** run-level display scale is listed v2, but §2.3's own limitation
  (b) — brightness stepping across images — lands squarely on the inspector's
  prev/next, which **is** v1. A comparison surface shipping with a known
  per-crop brightness inconsistency is the same defect §2.2 rejects the naive fix
  for, one level up. Either promote it or state explicitly that prev/next
  brightness stepping is accepted in v1.

---

## OPEN QUESTIONS FOR THE AUTHOR

1. Do you accept scoping Q5 to the screen and exporting with `go.Scatter` (my
   recommendation, B1), or should the failure be taken back to the user as a Q2
   reopening? The measured export timings say (1) is comfortable.
2. Where does Chrome come from on the compute node (B2) — `BROWSER_PATH` to the
   vendored Playwright browser, or `plotly_get_chrome` at env-build time?
3. Is the phantom filter (B4) applied once at `master_df` ingest for the whole
   tab, or per-plot? The former changes what the shared filter offcanvas reports;
   the latter keeps Scatter consistent with Plate and Colony.
4. What removed the 14,784 measured rows from the mirror on a run with empty
   `post` and `filters` (B4)? If it is GUI curation, that confirms the mirror is
   session-mutable and settles S8's fingerprint requirement.
5. Should `Texture_` be added to `_MEASUREMENT_PREFIXES` — which fixes S1 for
   every consumer but also changes the Colony grid's axis options — or should
   Scatter apply its own additional filter?
6. Can anything still mutate curation while Scatter is open (the Colony tab)?
   That decides whether S8's `master_df` anchor suffices alone or needs the
   snapshot fingerprint beside it.
7. §5's "renders the first N facets" — first by facet-value sort order, or by
   point count? The `TIMELINE_COMPARE_CAP` precedent it cites does not settle it.

---

## ARTIFACTS

Spike outputs, under
`/scratch/anguy344/27994238/claude-5188/-bigdata-exfab-anguy344-PhenoTypic/e6ad7160-0024-44c2-ba1e-552a606cebac/scratchpad/`:

- `gl1.png`, `svg1.png` — the B1 pair (624 vs 46,886 non-white px)
- `glN{1,2,3,4,6,8,12,16}.png` — the gl panel-count sweep, flat ~180 ink/panel
- `glc{1,4,36}.html` — the B3 canvas-count pages
- `svg_{5000,20000,100000}.pdf` — the export timing set
- `gl16.pdf`, `gl_gpu.png`, `gl_dbg.png` — mitigation attempts

These live on node-local scratch and will not survive the job. Re-run the snippets
in B1 and B3 to reproduce.
