# Gate 0 review — Results Viewer Scatter Tab implementation plan

**Reviewed:** `docs/superpowers/plans/2026-09-01-results-scatter-tab.md`
**Against spec:** `docs/superpowers/specs/2026-09-01-results-scatter-tab/design.md` (revision 3)
**Worktree:** `/bigdata/exfab/anguy344/PhenoTypic/.worktrees/results-scatter-gui`, branch `feat/results-scatter-gui`
**Date:** 2026-09-01
**Verdict:** do not start implementation. **9 blocking findings**, 12 should-fix,
11 nits.

## How to read the evidence markers

Every finding is marked with how it was established. The plan contains runnable
code, so "I read the signature" and "I ran it" are different strengths of claim
and are not conflated here.

- **[SRC]** — read directly from the named `file:line`.
- **[MEASURED]** — a spike script was written and executed; the numbers quoted
  are its output, not an estimate.
- **[SPIKE-pending]** — written and queued, output not yet returned.

Seven spike scripts live in
`/scratch/anguy344/27996931/claude-5188/-bigdata-exfab-anguy344-PhenoTypic/e6ad7160-0024-44c2-ba1e-552a606cebac/scratchpad/`.
A–F have been executed (all exit 0); G is queued.

| Script | Settled |
|---|---|
| `spike_a_schema_prefixes.py` | **Clean.** No leaf's `category()` raises (46 leaves); result is NOT import-order dependent; all four Task 4 assertions pass |
| `spike_b_frame_index.py` | **Clean.** All three Task 8 tests pass; left join preserves order at n=2000; `maintain_order='left'` accepted |
| `spike_c_figure_facets.py` | Shared-axes test has **no teeth**; legend bug **confirmed**; `plottable` raises rather than corrupts |
| `spike_d_store_and_grouping.py` | `_read_store_level` misuse **confirmed and worse than named**; Task 7's 149-column claim reproduces exactly |
| `spike_e_dash_and_deps.py` | `callback_map` form, missing symbols, missing asset path, absent Chrome/pypdf/fitz — all confirmed |
| `spike_f_fix_and_vacuity.py` | B1 fix shape; vacuity sweep by mutation |
| `spike_g_rasterizer_and_range.py` | *(queued)* whether pymupdf is needed; simplest correct `image_display_range` |

## Withdrawn after measurement

Two findings from the first pass did not survive their spikes. Recorded so they
are not re-raised:

- **S5(a) and S5(c), import-order dependence and a raising `category()`** —
  disproved. `spike_a` walks all 46 discoverable leaves under both import
  orders: nothing raises, `Status` is present either way, and the derived tuple
  is identical (31 prefixes) with and without `phenotypic.sdk_` loaded. Task 4's
  derivation is sound as written. The docstring-drift point (S5b) and the
  unstated `QC_` behaviour change (S5d) still stand.
- **S7, join-order flakiness** — disproved, and by a wider margin than a
  downgrade. `spike_b` confirms the left join preserves left row order at
  n=2000, the per-key rank values are correct regardless of order, and
  `maintain_order='left'` is accepted by polars 1.41.2 if belt-and-braces is
  wanted. Task 8 needs no change.

---

# BLOCKING

## B1 — The click index is built against the filtered frame and resolved against `master_df` [SRC]

**Accepted by the orchestrator; recorded here for completeness.**

Task 13's figure callback (plan:2026-2027):

```python
frame = plottable(apply_filters(output_root.master_df, filter_state))
frame = frame.with_row_index(CUSTOMDATA_COL)
```

`with_row_index` numbers the rows of the **filtered** frame. Task 11's resolver
(plan:1853-1855):

```python
if index < 0 or index >= master_df.height:
    return None
row = master_df.row(index, named=True)
```

indexes the **unfiltered** frame positionally. With any active filter — or with
no filter at all, because `plottable` drops 121 phantom rows in the verification
fixture — the two frames have different row counts and different orderings, and
every click resolves to a different colony than the one clicked.

Nothing errors. The result is a real colony with a real crop and real
measurements. This is exactly the failure spec §6.1 describes ("It opens the
wrong colony, silently, and the result looks entirely plausible") and exactly
what the plan's own orchestration says cluster E exists to prevent ("a mistake
opens the *wrong* colony plausibly and silently", plan:2253-2255). The plan
ships the bug inside the code written to prevent it.

**Why it survived the plan's own review:** producer and resolver are in
different tasks, different clusters, and different review gates. Task 11's three
tests all pass against the broken wiring because none of them constructs an
index — they hand `resolve_click` a literal `1`.

### The fix, and the API question it raises

The mechanical fix is to index before filtering:

```python
base = output_root.master_df.with_row_index(CUSTOMDATA_COL)
frame = plottable(FilterSpec.from_store(filter_spec).apply_to(base))
```

This depends on `apply_to` preserving a caller-added column. It does: `apply_to`
(`results_viewer/_filter_state.py:255-284`) only ever calls
`normalize_viewer_frame(df)` then `result.filter(expr)` per row.
`normalize_viewer_frame` (`results_viewer/_metadata.py:61-109`) partitions
columns into metadata and non-metadata, renames every non-metadata column to a
shield name, normalizes, and renames back via the `reverse` map — it adds and
removes nothing. `_scatter_row_index` is non-metadata, so it round-trips with
its name and values intact, and `.filter()` preserves both row order and column
set. `spike_f` confirms this end-to-end including the phantom-drop pass. **[SRC,
SPIKE-pending for the round trip]**

**But the mechanical fix leaves the class of bug in place, and I recommend
against it on its own.** The defect is not that someone wrote the wrong line; it
is that the plan's API lets producer and consumer disagree silently across a
cluster boundary. `resolve_click(master_df, index, ...)` accepts a bare `int`
whose meaning is a convention held in two files that no gate compares.

**Recommendation — adopt the `index_frame` helper, owned by Task 11.** Move both
halves behind one function in `_inspector.py`:

```python
def index_frame(master_df: pl.DataFrame) -> pl.DataFrame:
    """Attach the positional index that ``resolve_click`` resolves against.

    The index is positional into ``master_df`` and MUST be attached before any
    filtering: a filtered frame's positions are not master positions, and a
    click resolved against the wrong frame opens the wrong colony silently.
    This function and :func:`resolve_click` are the only two places that know
    the index's meaning.
    """
    return master_df.with_row_index(CUSTOMDATA_COL)
```

Task 13 then calls `index_frame(output_root.master_df)` and can only get it
right; the docstring lives next to the resolver that depends on it; and Task 11
owns the invariant end to end rather than owning half of it. It also gives the
missing test a natural home:

```python
def test_an_index_survives_filtering_and_still_resolves() -> None:
    """The B1 regression pin: filter, then resolve, and land on the SAME row."""
    base = index_frame(master)
    kept = FilterSpec.from_store(
        [{"column": "Metadata_Strain", "values": ["S1"]}]
    ).apply_to(base)
    for row in kept.iter_rows(named=True):
        ref = resolve_click(master, row[CUSTOMDATA_COL], "fp", "fp")
        assert ref == ColonyRef(row[KEY_DATASET], row[KEY_IMAGE_FILE],
                                row[KEY_OBJECT_LABEL])
```

That test fails loudly against the plan as written and passes against either
fix. It is the one test this plan most needs and does not have.

One caveat on the helper: it must be called on `master_df`, never on a frame
that already carries the column, or `with_row_index` raises a duplicate-column
error. Either guard it or document it.

## B2 — `apply_filters` and `STORE_FILTER_STATE` do not exist [SRC]

Task 13 (plan:2022, 2026) calls `apply_filters(...)` and reads
`State(rv_ids.STORE_FILTER_STATE, "data")`. Neither symbol exists anywhere in
the codebase. The real API:

| Plan writes | Reality |
|---|---|
| `rv_ids.STORE_FILTER_STATE` | `STORE_FILTER_SPEC` — `results_viewer/_ids.py:29` |
| `apply_filters(df, state)` | `FilterSpec.from_store(payload).apply_to(df)` — `_filter_state.py:181` (class), `:198` (`from_store`), `:229` (`apply_to`) |

Working reference for the whole pattern: `results_viewer/_viewer_card.py:1111`
(`State(STORE_FILTER_SPEC, "data")`) and `:1128-1130`
(`FilterSpec.from_store(filter_payload)` → `spec.apply_to(slice_df)`).

**Second, functional, defect in the same lines:** the plan makes the filter
store a `State`. None of the callback's ten `Input`s fires when the user edits a
filter, so the Scatter figure silently goes stale — defeating decision Q4
("Share filters only"), which is the tab's entire relationship to the rest of
the viewer. It must be an `Input`.

## B3 — Task 13's registration test cannot pass, before or after implementation [SRC, SPIKE-pending]

plan:1983-1987:

```python
inputs = {dep.component_id for cb in app.callback_map.values() for dep in cb["inputs"]}
```

On Dash 4.1.0, `callback_map[...]["inputs"]` holds plain dicts keyed `"id"` and
`"property"`, not dependency objects. `dep.component_id` raises `AttributeError`
in both the red and the green phase, so the test gates nothing.

**The exact working form**, copied from the repo's own passing example at
`tests/unit/gui/results_viewer/test_filter_panel.py:215-219`:

```python
    registered_input_ids = {
        spec["id"]
        for entry in app.callback_map.values()
        for spec in entry["inputs"]
    }
    assert ids.STORE_PLOT_REFRESH_REVISION in registered_input_ids
```

Note that file also demonstrates the idiom for pattern-matching ids, where
`spec["id"]` is a dict rather than a string — it joins them into a blob and
substring-matches (`:222-226`). Scatter's ids are plain strings, so the set
membership above is correct as written. `spike_e` and `spike_f` both print the
live structure to confirm.

## B4 — Task 9's `test_an_empty_facet_still_occupies_its_cell` is vacuous [SRC, SPIKE-pending]

plan:1342-1351. The frame is:

```python
"r": ["0", "0"], "c": ["0", "0"]
```

so `plan_facets` returns a **1×1** grid. There is no empty facet in the fixture
the test builds. Its only assertion is:

```python
assert isinstance(fig, go.Figure)
```

which cannot fail once `build_scatter_figure` exists and returns anything at
all. The test's name and docstring both describe behaviour it never exercises.
`spike_c` prints the grid dimensions to make this concrete.

**Replacement:** use ≥2 row values × ≥2 column values with one combination
absent from the data, then assert on the grid's geometry rather than its type —
`len(fig.layout.annotations)`, or that the axis domains still tile 2×2, or that
the trace count is 3 while the subplot count is 4.

## B5 — Task 1's `test_the_display_range_does_not_depend_on_the_crop_window` is vacuous [SRC]

plan:113-124:

```python
assert image_display_range(store, "rgb") == image_display_range(store, "rgb")
```

This is `f(x) == f(x)` on a deterministic pure function. It cannot fail.

It is worse than merely weak: `image_display_range(store_path, layer)` **takes
no window argument** (plan:166), so no implementation of that function could
ever fail this test — including the per-crop min-max stretch that spec §2.2
spends three paragraphs rejecting. The test named for the §2.2 defect is
structurally incapable of detecting it.

Spec §2.5 states the real requirement: "Two crops of the same image, taken from
different windows, map an identical source value to an identical output." That
needs two `crop_store_rgb` calls with different centres over an overlapping
region, comparing the overlap pixel for pixel:

```python
def test_two_windows_map_a_shared_pixel_identically(store: Path) -> None:
    a = _decode_rgb(crop_store_rgb(store, "rgb", 100, 100, 64, 0))
    b = _decode_rgb(crop_store_rgb(store, "rgb", 116, 116, 64, 0))
    # The two 64px windows overlap; the pixel at (100,100) in image space is
    # a[32,32] and b[16,16]. A per-crop stretch renders it two different ways.
    np.testing.assert_array_equal(a[32, 32], b[16, 16])
```

## B6 — Task 10's page-count test hard-fails wherever Chrome is absent, which the spec says is here [SRC, SPIKE-pending]

`test_one_page_is_written_per_section` (plan:1588-1595) calls
`export_sections_pdf`, which calls `kaleido.write_fig_sync`, which needs a
Chrome binary. The test has no availability guard — only the separate ink test
guards `pymupdf`.

Spec §11.2 states plainly that on this node `google-chrome`, `chromium` and
`chromium-browser` are all absent from `PATH` and `~/.cache/kaleido` does not
exist, and that plain `write_image` raises `RuntimeError: Kaleido requires
Google Chrome to be installed`. So the plan's "Expected: the page-count test
passes" (plan:1723) is wrong in the environment the plan itself documents, and
the default `uv run pytest` lane goes red.

`_pdf.py` catches that `RuntimeError` and re-raises it with an install hint
(plan:1706-1709) — which converts the failure into a *different* failure, not
into a skip.

**Fix, one of:** add a module-level Chrome guard
(`pytest.mark.skipif(shutil.which("google-chrome") is None and not
Path("~/.cache/kaleido").expanduser().exists(), ...)`), or make §16.1's decided
`plotly_get_chrome` an actual Task-10 step with a verification that it works
from a compute node — §16.1 already lists that as "a requirement, not a note",
and the plan does not carry it into any task.

## B7 — The blank-PDF guard, which §15 calls the only defence, never executes [SRC, SPIKE-pending]

`test_the_exported_page_contains_ink_not_just_axes` (plan:1599) is the sole
protection against §11.1's silent failure — a valid, well-formed, entirely blank
PDF produced with exit code 0 and no warning. It cannot run, for two independent
reasons:

1. It is decorated `@pytest.mark.slow` (plan:1598), and `pyproject.toml:222`
   sets `addopts = "--verbose --capture=no -m 'not slow'"`. Deselected by
   default.
2. It requires `fitz`/`pymupdf`, which is **not a dependency** — absent from
   `uv.lock` and from `.venv/lib/python3.12/site-packages/`. Task 10 Step 1 adds
   only `pypdf`. So even under `-m slow` it skips permanently.

Spec §13 says "Export tests assert on rendered ink, never on file existence" and
§15 lists "A blank export passes a naive test" as **Live**, "Mitigated by the
ink assertion in §13 — the only defence, since nothing else signals". A defence
that is deselected by default and skipped when selected is not mitigation. As
the plan stands, §15's live risk is unmitigated and the risk table is wrong.

**Fix:** add `pymupdf` to the dev group and drop the `slow` marker. The
measured separation is 289 dark pixels against 36,608, so the assertion is fast
and not delicate; the cost is one kaleido render, which the page-count test in
the same file already pays.

---

# SHOULD-FIX

## S1 — `_read_store_level`'s second parameter is `layer`, not `member` [SRC, SPIKE-pending]

Real signature, `gui/_shared/tiles.py:407-412`:

```python
def _read_store_level(
    store_path: Path | str,
    layer: str,
    level: int,
    window: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
```

It resolves the member itself at `:436-437`. Task 1 (plan:198) calls:

```python
smallest = _read_store_level(store_path, member, levels[-1])
```

passing `member` into the `layer` slot and a **`str`** (`levels[-1]`, from
`d.name`) into the `level: int` slot.

This appears to work only by coincidence. I read the fixture store's
`attributes.phenotypic` block: `series` is
`{'detect_mat': 'detect_mat', 'gray': 'gray', 'rgb': 'rgb'}` — an identity map,
so `member == "rgb"` and the mistaken argument happens to be a valid layer name.
It breaks for any non-identity member, and `labels` is already one:
`{'objmap': 'rgb/labels/objmap'}`.

Two further consequences:
- `mypy src/phenotypic/gui/_shared/tiles.py` (Task 1 Step 7) will reject `str`
  for `level: int`.
- The plan's own hedge — "If `image_display_range` raises, check
  `_read_store_level`'s actual signature" (plan:211) — will not fire, because on
  this fixture it does not raise. The safety net is positioned where the failure
  isn't.

**Correct call:** `_read_store_level(store_path, layer, int(levels[-1]))`. The
`_store_member_path` lookup at plan:191 is then needed only to locate the
directory for the level scan.

## S2 — `src/phenotypic/gui/_assets/results_viewer.js` does not exist [SRC]

That directory does not exist at all. The real file is
`src/phenotypic/gui/results_viewer/_assets/results_viewer.js` — splitter block
at `:798`, `clampSidebarWidth` at `:818`, drag handler at `:834`,
`#qc-review-splitter` lookups at `:868` and `:884`.

Spec §7 cites the right lines. The plan's File Structure block (plan:53) and
Task 12 Step 3 (plan:1889, 1923) both give the wrong path. An agent following
the plan literally creates a new file at a path Dash does not serve; the
splitter silently never attaches and nothing fails.

## S3 — Task 12 breaks an existing test and does not say so [SRC]

`tests/unit/gui/results_viewer/test_layout_tab_shape.py:33-37` asserts exact
equality:

```python
def test_results_tabs_expose_exactly_the_mounted_surfaces(built_results_layout):
    assert _tab_ids(built_results_layout) == [
        ids.TAB_PLATE_ID,
        ids.TAB_COLONY_ID,
    ]
```

Mounting a third tab fails it. Task 12 Step 5's "Expected: pass" (plan:1935) is
wrong.

Two further details the plan gets wrong about this file:
- the helper is `_tab_ids(layout)` (`:13`), not `_tab_ids_in_layout()`, and it
  takes an argument;
- it needs the `built_results_layout` fixture (`conftest.py:38`), which the
  plan's snippet never requests.

The file's docstring says the edit is deliberate and expected ("This test is
edited deliberately as surfaces are removed. Each edit is the executable
statement that a tab came off"), so updating it is correct — the plan just has
to say so.

## S4 — Task 3 has no test for the only risky part of Task 3 [SRC]

Steps 1-4 test `composite_contours` against synthetic arrays. Step 5 — resolving
the objmap member from `attributes.phenotypic.labels`, reading the matching
window, threading `contours` through `crop_store_rgb` → `crop_colony` →
`register_crop_route` — is prose with no test and no verification step. Spec
§2.5 explicitly requires "`?contours=1` on a window containing a known label
emits boundary pixels; `?contours=0` emits none", on a window with a real
colony.

Two specific hazards left unguarded:

**(a) The shape guard fails silently.** plan:409-410:

```python
if labels.shape != rgb.shape[:2]:
    return out
```

Returns the crop unchanged with no log and no error. If compositing is applied
after the `size × size` PIL paste rather than to the clamped window array, every
crop silently loses its contours and the feature quietly does nothing. I checked
the shapes on the fixture — `rgb/0` is `[3, 3132, 5086]` uint16 and
`rgb/labels/objmap/0` is `[3132, 5086]` — so the *clamped-window* pairing does
match, but the plan's prose does not specify which pairing it means. Say it
explicitly, and log at `warning` rather than returning silently.

**(b) The overlay fallback is unspecified.** `crop_colony` falls through to
`crop_overlay` when no store exists (`tiles.py:667-671`), and `crop_overlay`
takes no `contours`. The plan does not say whether that is an error, a silent
no-op, or acceptable (the baked overlay already has contours drawn in, so it is
probably acceptable — but it should be stated, since the two paths would then
render different-looking contours).

## S5 — Task 4's derivation is import-order dependent and reintroduces a dead entry [SRC, SPIKE-pending]

Four separate problems in `_derive_measurement_prefixes` (plan:531-555):

**(a) `"Status"` may not be discoverable.** `PIPE_STATUS.category()` returns
`"Status"` and lives at `src/phenotypic/sdk_/constants_.py:141-146` — **outside**
`phenotypic.schema`. `from phenotypic.schema import MeasurementInfo` does not
import it, so `MeasurementInfo.__subclasses__()` reaches it only if something
else already loaded `phenotypic.sdk_`. It happens to work because `_grid.py`
imports `from phenotypic.sdk_ import is_metadata_header` at module scope
(`colony_view/_grid.py:67`), but `_MEASUREMENT_PREFIXES` is computed at **module
import time** and is therefore a function of import order. This is the same
class of latent wrongness as the `TextureGray_` entry the task exists to remove.
Import the owning enums explicitly instead of relying on `__subclasses__()`
reachability. `spike_a` measures the difference with and without `sdk_` loaded.

**(b) The docstring becomes a second stale copy.** `selectable_axis_columns`
spells the tuple's contents out in prose at `colony_view/_grid.py:212-214`
(`Bbox_`, `Shape_`, `Intensity_`, `TextureGray_`, `SymZones_`, `GridSpatial_`).
Task 4 changes the tuple and never touches that docstring, recreating exactly
the drift it is fixing.

**(c) `hasattr(c, "category")` is vacuous.** `MeasurementInfo` declares
`category()` on the base (`schema/_measurement_info.py:342-343`), so the guard is
always true. If the intent was to skip leaves that do not override it, the guard
does not do that — and the base's `Raises:` section means a non-overriding leaf
would raise at module import, taking the Colony tab down with it. `spike_a`
enumerates every leaf and reports any that raise. **If any does, this becomes
BLOCKING.**

**(d) An unstated behaviour change.** `METADATA_MATCH.category()` and
`QUALITY_CHECK.category()` both return `"QC"`
(`schema/_metadata_match.py:19`, `schema/_quality_check.py:33`), so `"QC_"`
enters the exclusion tuple and `QC_MetadataOnly` stops being offered as a
section or facet column. That is probably desirable, but it is a user-visible
change to the Colony grid's axis options that the plan does not mention and
FEATURES.md would need to reflect.

*Not a defect, for the record:* `analysis/_error_cutoffs.py:32-42` keeps its own
`MEASUREMENT_PREFIXES` and its comment says the independence is deliberate ("This
list is defined independently — the colony grid's `_MEASUREMENT_PREFIXES` is a
UI axis-exclusion list, not an authoritative phenotype-measurement set"). Leave
it alone.

## S6 — `"QC_MetadataOnly"` is hard-coded against a firm convention [SRC]

plan:657 (`_spec.py`) and plan:1037 (`_grouping.py`). The column is owned by
`METADATA_MATCH.METADATA_ONLY` (`schema/_metadata_match.py:21`), and every
existing site spells it through the schema: `_cli/_cli_output_manager.py:196`,
`:860`, `:894`; `sdk_/_metadata_helpers.py:760`. CLAUDE.md's rule — "Metadata
queries use schema ownership, never string prefixes" — governs.

```python
from phenotypic.schema import METADATA_MATCH
CURATION_PHANTOM_COL = str(METADATA_MATCH.METADATA_ONLY)
```

**Related, and more than cosmetic:** `plottable` does
`pl.col(CURATION_PHANTOM_COL).cast(pl.Boolean)` (plan:713). The existing pandas
equivalent `metadata_only_mask` (`sdk_/_metadata_helpers.py:712-765`) devotes a
paragraph to why that coercion must be refused: "The dtype check is deliberately
**strict**: only a real boolean column is trusted. An object/string column is
rejected rather than coerced, because `pd.Series(["False", "True"]).astype(bool)`
is `[True, True]` — the string `"False"` is truthy — which would silently mark
every row a phantom." Polars may raise on the cast instead of coercing, which
turns a silent corruption into a 500; `spike_c` reports which. Either way, mirror
the existing helper: test `df.schema[col] == pl.Boolean` and treat anything else
as "no flag present".

## S7 — Task 8's left join has no order guarantee [SPIKE-pending]

plan:1248: `df.join(ranked, on=[plate_col, time_col], how="left")`. Polars 1.41.2
does not guarantee output row order for joins unless `maintain_order` is passed,
and `test_index_ranks_within_each_plate_independently` (plan:1173) asserts an
exact positional list.

The *values* stay correctly keyed — the join carries them by key, so the plotted
data is not corrupted — but the test can flake and any positional consumer
downstream breaks. `spike_b` measures whether the order actually moves at n=2000
and whether `maintain_order="left"` is accepted on this version.

Pass `maintain_order="left"` regardless; it costs nothing and removes the
question.

## S8 — A hue absent from the first facet never appears in the legend [SRC, SPIKE-pending]

plan:1490 sets `showlegend=first_cell`, and plan:1503 clears `first_cell` after
the first `(row, col)` cell. If that cell has no rows for hue *k*, the
`if part.height == 0: continue` at plan:1475-1476 skips it, and hue *k* is then
drawn in every other facet with `showlegend=False` — invisible in the legend
despite being on the plot.

This is not hypothetical for this data. Spec §1.3 says the verification fixture
is "sparse by construction": 23 strains across 36 images, median 32 plottable
rows per strain, explicitly kept for "exercising empty facets, sparse grids and
the 'no data' cell". The first facet having a gap is the normal case, not the
edge case.

**Fix:** track emitted legend groups instead of cell position.

```python
seen_legend: set[str] = set()
...
showlegend = label not in seen_legend
seen_legend.add(label)
```

## S9 — The facet row label is never rendered [SRC]

plan:1438-1441 builds `subplot_titles` as `f"{spec.col_col}={c}"` repeated once
per row. The row facet value appears nowhere in the figure. Spec §9 lists
"Figure row label" as a first-class control with its own dropdown, so a user who
sets it gets a grid that splits by that column but never says which row is
which.

The list length is correct (`rows * cols`), so `make_subplots` will not
complain — this fails as a silent usability gap, not an exception. Row labels
normally go on the left via `fig.update_yaxes(title_text=r_val, row=i, col=1)`
or a left-edge annotation.

## S10 — The shared-axes test passes when it finds nothing [SRC, SPIKE-pending]

plan:1354-1361 ends `assert len(ranges) <= 1`, which is satisfied by
`len(ranges) == 0`.

I traced the layout walk and it does find something: with `make_subplots(2, 2)`
plus `fig.update_yaxes(range=...)`, all four `yaxis*` entries carry the same
range, so the set has exactly one element and a per-facet-range regression would
produce four. The test is therefore not vacuous in the B4/B5 sense — but it
silently degrades to always-true if `_axis_range` starts returning `None`, if the
`share_axes` branch is dropped, or if plotly changes its layout key naming.

Make it `== 1`. `spike_c` re-runs the builder with `share_axes=False` to confirm
the assertion actually flips.

## S11 — The `_config.py` ownership claim is false [SRC]

The Orchestration section (plan:2249-2250) states: "A owns *all three*
`_config.py` constants, including the two Task 6 needs, so C never touches that
file and the seam stays in one cluster."

Task 3 Step 6 adds only `SCATTER_CROPS_URL_SEGMENT` (plan:450-457). Task 6
Step 4 adds `SCATTER_FACET_CAP` and `SECTION_GROUP_CAP` (plan:906-923) — and
Task 6 is in cluster **C**, not A. The plan's own "Shared files" line correctly
contradicts the claim: "`_config.py` (3, 6)" (plan:2230).

No write race results, because C runs after A. But the stated invariant is wrong
and should not be relied on if the clustering is ever re-cut. Either move the
two caps into Task 3, or delete the claim.

**Also within Task 6:** Step 3 writes `from phenotypic.gui._config import
SCATTER_FACET_CAP` (plan:822) before Step 4 creates the constant. Swap the two
steps or Step 3 leaves the tree un-importable.

## S12 — Tasks 2 and 10 put non-hermetic work in `tests/unit/` [SRC]

`tests/CLAUDE.md` scopes the directory: "`unit/` ← deterministic, no I/O beyond
`tmp_path`".

- Task 2 reads an absolute path under
  `/rhome/anguy344/bigdata_exfab/projects/ucr_029_e_d_Maresca/...` — correctly
  `skipif`-guarded, but still a machine-specific dependency inside the unit
  lane.
- Task 10 spawns headless Chrome.

Both belong under `tests/integration/`. `tests/unit` is in `testpaths`
(`pyproject.toml:219`), so this is the suite everyone runs by default.

## S13 — `resolve_click` will raise on a null `Object_Label` [SRC, SPIKE-pending]

plan:1859: `label=int(row[KEY_OBJECT_LABEL])`.

`master_df` **is the mirror** — `_output_root.py:348` reads `mirror_path` into
`master_df` — and the mirror carries phantom rows with a null `Object_Label`
(121 of 844 in the fixture, per spec §1.1). A fingerprint-matching index should
never point at one, but `resolve_click`'s documented contract is that it returns
`None` "when the index is stale or out of range"; a `TypeError` escaping into a
Dash callback is neither, and surfaces as a 500. Guard it and return `None`.

---

# NITS

1. **plan:215** — "change its signature and both call sites".
   `_store_layer_array_to_rgb` has exactly one caller: `tiles.py:571`. **[SRC]**
2. **plan:244** — predicts `test_crop_matches_the_full_resolution_slice` may
   break. It won't: that test exercises the `detect_mat` layer
   (`test_tiles_zarr.py:83`), which the change does not touch. The rgb test is
   `test_rgb_crop_returns_channel_last_pixels` (`:138-142`), and it survives only
   because `load_synth_yeast_plate` is 8-bit and `scale_to_uint8` short-circuits
   on `uint8`. Name the right test. **[SRC]**
3. Consequence of that short-circuit: `image_display_range` performs a whole
   smallest-level read *before* `scale_to_uint8` discards its result, on every
   crop from every uint8 store. Check the dtype first. **[SRC]**
4. **plan:348** — the contours test imports `OI_ORANGE` and never uses it.
   `ruff check --fix` in Step 7 will strip it, or F401 fires. **[SRC]**
5. **plan:1112-1121** — Task 7's fixture check hand-joins
   `D + '/pipeline.json.pht-pipe'`, violating the plan's own Global Constraint
   (plan:22) to resolve through `layout.pipeline_config_path`. The check itself
   is sound — I confirmed the file's `meas` block has exactly the assumed shape,
   with `MeasureColor` carrying `include_XYZ: false, include_xy: false` and
   `MeasureTexture` carrying `scale: [5]` — it is just spelled against the rule
   it sits below. **[SRC]**
6. **plan:674** — `FigureSpec.marker_size` documented as "Marker area in points
   squared". Plotly's `marker.size` is a diameter in pixels. **[SRC]**
7. **spec header** still reads "Status: revision 2" while its §2.3 and the plan
   both say revision 3. **[SRC]**
8. **plan:1575** — module-level `pytest.importorskip("pypdf")` sits *after* the
   `_pdf` import; harmless only because `_pdf` imports pypdf lazily inside the
   function. **[SRC]**
9. **plan:1459, 1462** — `r_val != ""` means a facet value that is genuinely the
   empty string applies no filter, silently drawing the whole frame in that cell.
   Use a sentinel rather than `""`. **[SRC]**
10. **plan:1394** — `_SYMBOLS` includes `"x"`, which spec §9 also reserves for
    curation-removed colonies ("show removed colonies as grey x"). With
    ≥5 shape values the two become indistinguishable. **[SRC]**
11. **Task 10** has no test that `export_sections_pdf` actually passes
    `for_export=True`. Spec §13 asks for "a guard that the export path never
    emits a `Scattergl` trace"; Task 9's trace-type test covers the *builder*,
    not the export function that calls it. One line: assert every trace in the
    figure `_pdf` builds is `"scatter"`. **[SRC]**

---

# VERIFIED CORRECT

Everything below was checked against source and found accurate as the plan
states it.

**Signatures and symbols**
- `_readable_block(store_path)` → `tiles.py:333`; `_store_member_path(block,
  store_path, layer)` → `:353`. Both called correctly by the plan.
- `crop_store_rgb(store_path, layer, center_rr, center_cc, size, mtime_ns, *,
  dim_alpha, bbox)` → `tiles.py:475-485`. Task 2's positional call matches
  exactly.
- `crop_colony(output_root, dataset, stem, layer, center_rr, center_cc, size, *,
  dim_alpha, bbox)` → `tiles.py:595-606`. Task 3's argument order matches.
- `register_crop_route(app, output_root, segment)` → `tiles.py:838-840`; adding
  `default_contours: int = 0` is purely additive and leaves the two existing
  segments unchanged, as the plan intends.
- `KEY_DATASET` / `KEY_IMAGE_FILE` / `KEY_OBJECT_LABEL` →
  `_filtered_state.py:52/55/58`, resolving to `Metadata_Dataset` /
  `Metadata_ImageName` / `Object_Label`. Task 11's `_master()` fixture columns
  match.
- `OutputRoot.snapshot.consumed_state_fingerprint` → `_output_root.py:75`
  (descriptor class), `:104` (field), `:163` (`snapshot` field), with a
  convenience property at `:533`.
- `STORE_PLOT_REFRESH_REVISION` → `_ids.py:813`, exported at `:1087`.
  `TAB_HEATMAP_ID` (`:631`) and `TAB_QC_ID` (`:628`) exist and are unmounted, so
  Task 12's negative assertions are meaningful rather than trivially true.
- `OKABE_ITO` → `_design.py:280`, a 7-tuple in DESIGN.md §06 series order
  (navy, orange, sky, green, blue, purple, vermilion). `OI_ORANGE` `:263`,
  `OI_SKY` `:264`.
- `phenotypic.measure` exports all five measurers Task 7 names
  (`measure/__init__.py:9-19`). `get_measurement_infoclasses()` is an instance
  method returning `tuple[type[MeasurementInfo], ...]`
  (`abc_/_measure_features.py:333-335`), and `MeasureColor` overrides it on
  `include_XYZ` / `include_xy` (`_measure_color.py:73-101`) — so the plan's
  "instantiate from recorded params" rule is necessary, exactly as §8 argues.
- `TEXTURE.get_headers(cls, scale, matrix_name=None)` → `schema/_texture.py:159-160`.
  A bare call raises `TypeError`, so the fallback branch is genuinely reachable,
  and `TEXTURE.category()` is `"Texture"` (`:41`) so the prefix match finds
  `Texture_Contrast-deg000-scale05`.
- `is_metadata_header` is exported from `phenotypic.sdk_`
  (`sdk_/__init__.py:256, 347`) and accepts both `Metadata_Strain` and
  `Metadata_PlateID` (`_metadata_helpers.py:289`).
- `kaleido.write_fig_sync(fig, path)` exists
  (`.venv/.../kaleido/__init__.py:174`), forwarding to `Kaleido.write_fig(fig,
  path, opts=None, ...)` (`kaleido/kaleido.py:421-430`); the `.pdf` extension
  selects the format, as the plan assumes.

**Versions** — polars 1.41.2, plotly 6.6.0, dash 4.1.0, kaleido 1.2.0,
pytest 9.0.1 (so `importorskip(reason=...)` is supported). `with_row_index` and
`cum_count().over()` are current API.

**Logic I traced and found sound**
- Task 8's expression is semantically right for all three of its tests:
  `.unique().drop_nulls().sort()` then `cum_count().over(plate).sub(1)` yields
  `[0,1,2,0,1]`; `.unique()` collapses repeated timestamps so colonies in one
  image share a frame; and a null-keyed left join yields `None` rather than 0,
  because polars does not join on nulls by default. Only the *order* guarantee
  is at risk (S7).
- Task 6's cap loop terminates correctly on the 12×12 case, shrinking
  alternately to 4×4 = 16 ≤ 16 with `truncated=True` and `total=144`.
- `sort_facet_values(["10", "a", "2"])` returns `["10", "2", "a"]`: CPython's
  `sorted` computes all keys up front, so the `ValueError` fires before any
  comparison and the lexical fallback is reached with nothing half-sorted.
- Task 9's `subplot_titles` list length is `rows * cols`, which is what
  `make_subplots` expects.
- Task 1's three arithmetic tests (monotonic ramp, clip, uint8 pass-through) are
  sound and have teeth: they fail on `ImportError` in the red phase, and the ramp
  test genuinely detects the mod-256 regression.
- Task 3's two `composite_contours` tests are correct — `find_boundaries(...,
  mode="outer")` on the synthetic squares produces non-zero pixels, and the
  all-zero label map short-circuits both branches so the no-op test holds.

**Fixture facts I re-derived rather than took from the spec**
- The migration-test store exists at the path Task 2 hard-codes.
- `attributes.phenotypic.series` is `{'detect_mat': 'detect_mat', 'gray':
  'gray', 'rgb': 'rgb'}`; `labels` is `{'objmap': 'rgb/labels/objmap'}`;
  `pyramid` is `{"downsample": {"image": "mean", "label": "nearest"}, "levels":
  5, "stop_px": 512}`.
- `rgb/0` is `[3, 3132, 5086]` uint16; `rgb/labels/objmap/0` is `[3132, 5086]`.
- `deliverables/pipeline.json.pht-pipe` has a top-level `meas` key shaped
  `{name: {"class": str, "params": dict}}`, carrying exactly the five measurers
  the plan assumes.

**Structure**
- `results_viewer/_layout.py:561-577` is the `dbc.Tabs` block with exactly Plate
  and Colony, matching the plan's "currently lines 560-577".
- `FEATURES.md:312` carries `## Results Viewer integration`;
  `scripts/check_features_md.py` and `scripts/check_workflows_md.py` both exist;
  the highest tutorial page is `docs/source/tutorials/gui/18_browse.md`, so
  Task 14's `19_scatter.md` is the right number.
- `docs/superpowers/logic_validation_scripts/2026-09-01-results-scatter-tab/crop_uint16_scaling.py`
  exists, so Task 15 Step 4 is runnable.
- Clusters A (`tiles.py` + `test_tiles_zarr.py`) and B (`colony_view/_grid.py` +
  a new test file) are genuinely disjoint — that half of the parallelism claim
  holds. `_MEASUREMENT_PREFIXES` has exactly one consumer,
  `colony_view/_grid.py:244`.
- No task depends on a symbol an earlier task fails to produce, apart from the
  intra-task Step 3/Step 4 ordering in Task 6 noted under S11.

---

# Recommended order of fixes

1. **B1** with the `index_frame` helper, plus its round-trip test. This is the
   only finding that produces silently wrong output for the user.
2. **B2** — replace `apply_filters`/`STORE_FILTER_STATE` with
   `FilterSpec`/`STORE_FILTER_SPEC`, and promote the store to an `Input`.
3. **B6, B7** together — decide Chrome provisioning, add `pymupdf`, unmark the
   ink test. Until then the export half of the feature is unverifiable.
4. **B3, B4, B5** — the three tests that cannot fail. Cheap, and they are what
   the later gates will lean on.
5. **S1, S2, S3** — three wrong references that each cost an implementing agent
   a debugging cycle.
6. Everything else in cluster order.

**One structural suggestion.** B1 survived because the producer and the resolver
sat in different clusters with a gate between them, and each cluster's tests
passed in isolation. The plan's gate list (plan:2258-2265) runs
`implementation-test-reviewer` after D and after G — B1's two halves are in G and
E, so the first gate that sees both is the last one. Consider adding a
cheap cross-cluster check to the gate after G specifically: for every value that
crosses a cluster boundary, name its producer and its consumer and confirm they
agree. That is the one thing per-cluster review structurally cannot catch.
