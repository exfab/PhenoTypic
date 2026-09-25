# MeasureNeighborDist — nearest-object measurement

**Date:** 2026-09-24
**Status:** Approved in brainstorming; awaiting written-spec review
**Branch:** `worktree-measure-neighbor-nearest`
**Validation script:**
`docs/superpowers/logic_validation_scripts/2026-09-24-measure-neighbor-nearest/nearest_object_bounds.py`

## 1. Objective

`MeasureNeighborDist` reports, for every colony, the edge-to-edge distance to the
nearest colony in each of the four edge-sharing grid cells. It can't see the
object that is actually closest when that object is:

- in the colony's **own** grid cell (a satellite colony, fragment, or debris),
- in a **diagonal** cell,
- **further** than one cell away (the adjacent cell is empty).

Add a plate-wide **nearest object** measurement alongside the directional
columns, so a single table answers both "how crowded is this colony by its
arrayed neighbours" and "what is the closest thing to this colony at all".
Make the measurer usable on a plain `Image`, where only the nearest-object
half is meaningful.

### Non-goals

- Changing the values of the eight existing directional columns on a
  `GridImage`. They keep their Voronoi-restricted semantics, including
  shielding (`test_shielded_target_returns_nan`).
- Renaming the class. `FilamentousFungiPipeline` and serialized pipelines
  (`{"class": "MeasureNeighborDist"}`) key on it. (The schema **category** *is*
  renamed; see §3.9.)
- A general legacy-header alias layer for measurement columns. The existing
  `LEGACY_HEADER_TO_MEMBER` registry covers metadata headers only; §3.9 names
  the one consumer that keeps reading the old prefix.
- Diagonal directional columns (`UpperLeftDistance`, …). The nearest-object
  relation code covers the diagonal case without four more column pairs.
- A distance threshold or search radius parameter. The search is exact and
  cheap enough without one.
- Physical units. Distances stay in pixels, like the existing directional
  columns.

## 2. Current behaviour (as of `81d19ec66`)

`src/phenotypic/measure/_measure_neighbor_dist.py`, a `GridMeasureFeatures`
(rejects a plain `Image` at `abc_/_grid_measure.py:265`), schema
`NEIGHBOR_DIST` (`schema/_neighbor_dist.py`, `QualityInfo`,
`category() == "GridSpatial"`).

- Emits `Label` plus 8 columns: `{Left,Right,Above,Under}NeighborObjLabel` and
  `{Left,Right,Above,Under}Distance`.
- It emits one row per object in `grid.info()`. Objects whose `Grid_RowNum`/`Grid_ColNum` is
  `NaN` are skipped as targets and as neighbours, and keep `NaN` in every column.
- Per target cell, one `ndi.distance_transform_edt(~target_mask,
  return_indices=True)` over a window covering the target and its in-bounds,
  non-empty 4-neighbours. Each neighbour pixel is attributed to its nearest
  target object; a target's directional distance is the minimum EDT value over
  the neighbour pixels attributed to it.
- **Distance convention:** Euclidean distance between pixel centres; two
  4-adjacent pixels are distance 1. Touching colonies therefore report 1, not 0.

Observed while scoping: `ManualGridFinder` clamps an object outside the grid
edges into the nearest edge cell (an object at row 82 of a grid ending at row
50 gets `Grid_RowNum == 0`). Off-grid (`NaN` grid position) objects are not
expected from the shipped grid finders; the existing `pd.isna` guard is
defensive, and this design keeps it.

## 3. Design

### 3.1 Interface

- **Base class:** `MeasureFeatures` (was `GridMeasureFeatures`). The measurer
  branches on `hasattr(image, "grid")`, the duck-typing precedent set by
  `MeasureOrientationZones` (`_measure_orientation_zones.py:1166`) and by
  `MeasureFeatures.measure` itself (`abc_/_measure_features.py:455`).
- **Parameters:** none added. The class stays keyword-only-constructible with
  no fields.
- **Name:** unchanged (`MeasureNeighborDist`).
- **Category:** renamed from `GridSpatial` to `NeighborDist`, so all 11 headers
  become `NeighborDist_*` (§3.9).

### 3.2 Columns

Every call emits the same 11 measurement columns plus `Object_Label`,
whatever the input type, so the schema is stable across `Image` and
`GridImage`:

| Column (`NeighborDist_…`) | `GridImage` | plain `Image` |
|---|---|---|
| 8 directional columns | unchanged | all `NaN` |
| `NearestObjLabel` | label of the closest eligible other object | same, over all objects |
| `NearestDistance` | min pixel-centre Euclidean distance between the two masks | same |
| `NearestRelation` | integer code, §3.4 | all `NaN` |

`NearestObjLabel` and `NearestRelation` are stored as float columns holding
integer values, with `NaN` for missing. That matches how the existing
`*NeighborObjLabel` columns are stored (`np.full(n, np.nan)`).

### 3.3 Eligibility

| Input | Rows | Eligible as target and as candidate |
|---|---|---|
| `GridImage` | every object in `grid.info()` (unchanged) | objects with non-`NaN` `Grid_RowNum` **and** `Grid_ColNum` |
| `Image` | every object in `objects.info()` | every object |

An ineligible object on a `GridImage` keeps a row, per the one-row-per-object
contract of `MeasureFeatures.measure`, with `NaN` in all three new columns, and
is never reported as another object's nearest.

### 3.4 `NearestRelation` code

From the absolute grid offsets `dr = |row_nearest − row_self|`,
`dc = |col_nearest − col_self|`:

| Code | Name | Rule | Typical reading |
|---|---|---|---|
| 0 | same cell | `dr == 0 and dc == 0` | satellite, fragment, or debris in the well |
| 1 | adjacent | `dr + dc == 1` | ordinary crowding by an arrayed neighbour |
| 2 | diagonal | `dr == 1 and dc == 1` | crowding the directional columns can't see |
| 3 | distant | anything else | intervening cell(s) empty |

`NaN` when the object has no nearest (fewer than two eligible objects) or the
input is a plain `Image`. There is no off-grid code: on a `GridImage`
off-grid objects are ineligible, so a candidate always has a grid position.

The code is an integer rather than a string because no measurer in `measure/`
emits a non-numeric column today, and downstream consumers (scatter axes,
`analysis/` tools) assume numeric measurements. The code table lives in the
`NearestRelation` `desc` (§3.7), which the deliverables README publishes.

### 3.5 Algorithm (nearest object)

This lives in a private helper that takes the object map and an eligible-label
array and returns `(nearest_label, nearest_distance)` arrays. It doesn't use the
grid, so it can be tested on its own, and the off-grid exclusion test can drive
it directly (no shipped grid finder produces an off-grid object; §2).

1. **Boundary pixels.** `ndi.find_objects(objmap)` gives each label's tight
   slice. Within it, `mask = crop == label` and
   `boundary = mask & ~ndi.binary_erosion(mask)` (default 4-connected cross).
   Crop edges count as outside, which is correct because the slice is tight.
   Keep boundary coordinates in full-image space.
2. **Exact bounding boxes** are the inclusive min/max of each object's boundary
   coordinates. The `Bbox_Max*` columns are *not* used: they are regionprops'
   exclusive max (`Bbox_MaxRR == 25` for pixels 20..24, observed in scoping).
   That would give only a looser bound, not a wrong one, but deriving the box
   from pixels makes it exact and removes a dependency.
3. **Branch and bound per target `a`:**
   - Compute the vectorized bounding-box lower bound
     `lb(a, o) = hypot(max(0, o.min_r − a.max_r, a.min_r − o.max_r),
     max(0, o.min_c − a.max_c, a.min_c − o.max_c))` against every other
     eligible object `o`.
   - Visit candidates in ascending `(lb, label)` order. Stop at the first
     candidate whose `lb > best` (strict).
   - For each visited candidate, compute the exact distance
     `d = cKDTree(boundary[o]).query(boundary[a]).min()`.
   - Accept it when `d < best`, or when `d == best` and its label is smaller
     (the tie-break).
   - Build each object's `cKDTree` lazily, only when the object is first
     visited as a candidate, and reuse it for later targets.
4. **Relation** (grid inputs only): look up both objects' `Grid_RowNum`/
   `Grid_ColNum` from the `grid.info()` frame already loaded for the
   directional pass, and apply §3.4.

**Change from what was discussed:** the brainstorm described seeding the
upper bound from the ~8 nearest objects by centroid. Ordering candidates by
the bounding-box lower bound seeds the bound just as well, needs no `k`, and
is exact by construction (C3 below), so the centroid seed is dropped.

**Cost.** Memory is O(total boundary pixels) plus the lazily built trees:
about 300k points for a 1536-colony plate. There is no full-plate distance
transform. Lower-bound vectors are O(n) per target, so O(n²) scalar work
overall, which is trivial at plate scales (1536² ≈ 2.4M).

### 3.6 Correctness argument

Each claim is re-derived independently by the validation script (numpy/scipy
only, 300 random trials of non-convex and holed blobs; mutation-tested: an
early-exit mutant gives 149 failures and a dropped tie-break gives 5).

- **C1 Boundary sufficiency.** If the closest pair `(a, b)` had `a` interior,
  then stepping from `a` along the axis of `b − a`'s largest component lands
  on a pixel still in `A` (all 4-neighbours of an interior pixel are in `A`).
  That pixel is strictly closer to `b`, because `|v − e_i|² = |v|² − 2|v_i| + 1
  < |v|²` for `|v_i| ≥ 1`. So the minimum is attained on 4-connected boundary
  pixels. Holes add boundary pixels, which is harmless.
- **C2 Box lower bound.** `|r_a − r_b| ≥ gap_r` and `|c_a − c_b| ≥ gap_c` for
  every pixel pair, so `hypot(gap_r, gap_c)` ≤ the mask distance.
- **C3 Branch-and-bound exactness.** Every unvisited candidate has
  `lb > best`, and therefore distance `> best`. Candidates tied at `best`
  have `lb ≤ best`, so the strict stop visits them and the tie-break sees
  them.
- **C4 Directional dominance.** A directional distance is the minimum over a
  Voronoi-restricted subset of one neighbour's pixels, which can't be smaller
  than the unrestricted pair minimum. So `NearestDistance ≤` every non-`NaN`
  directional distance, **exactly**. Both are `sqrt` of the same integer
  squared distance, so no float tolerance is needed.

### 3.7 Schema (`NEIGHBOR_DIST`)

Three new members. Only `label` and `desc` are authored; `bio_desc` stays
`""` and `image` stays unset, for a human domain author (CLAUDE.md Gotchas):

- `NEAREST_OBJ_LABEL = Entry("NearestObjLabel", …)`: the label of the
  closest other object anywhere on the plate. On a `GridImage` only objects
  with a grid position are considered.
- `NEAREST_DISTANCE = Entry("NearestDistance", …)`: the minimum pixel-centre
  Euclidean distance between the two object masks, in pixels. Touching
  objects report 1.
- `NEAREST_RELATION = Entry("NearestRelation", …)`: the §3.4 code table in
  full, including `NaN` on a plain `Image`.

The enum and class docstrings get updated: the first paragraph no longer says
"adjacent grid cells" only, and the class docstring gets a `Best For` entry for
satellite and contamination screening. Per the `MeasureFeatures` vs
`MeasurementInfo` docstring split, per-column detail lives only in the enum
`desc`.

### 3.8 Structure of `_operate`

```
_operate(image)
  ├─ labels / info frame  ← grid.info() if hasattr(image, "grid") else objects.info()
  ├─ directional = _measure_grid_directions(image, grid_info)   # existing body, moved
  │                 (plain Image → all-NaN frame)
  ├─ eligible     = labels with a grid position (grid) | all labels (plain)
  ├─ nearest      = _nearest_objects(objmap, eligible)          # §3.5
  └─ relation     = _nearest_relation(grid_info, nearest)       # grid only
```

The existing directional body moves unchanged into
`_measure_grid_directions`. Its behaviour is pinned by the existing
`TestEdtDistance` suite, which must stay green without edits.

### 3.9 Category rename: `GridSpatial` → `NeighborDist`

`NEIGHBOR_DIST.category()` returns `"NeighborDist"`, so every header, old and
new, is `NeighborDist_<Label>`. The measurer now also runs on images with no
grid, so a "Grid" prefix would be wrong, and the new name matches the enum and
the class.

What reads the prefix (checked with `grep -rn GridSpatial src tests docs`):

| Consumer | Behaviour after the rename | Change |
|---|---|---|
| `ErrorCutoffFinder.MEASUREMENT_PREFIXES` (`analysis/_error_cutoffs.py:40`) | hard-coded `"GridSpatial_"` | add `"NeighborDist_"`; **keep** `"GridSpatial_"` so tables written before the rename still count as phenotype columns |
| `test_prefix_set_detects_phenotype_headers_and_excludes_position` (`tests/unit/analysis/test_error_cutoffs.py:169`) | lists `"GridSpatial_Foo"` | add `"NeighborDist_Foo"` and keep the old entry |
| Results-viewer scatter grouping | resolves columns by asking the measurer, not by prefix | none; new-run columns group under `MeasureNeighborDist`. `GridSpatial_*` columns in an **older** run's table fall into "Unattributed", like any header the current schema doesn't declare |
| Deliverables `README.md` generator | reads `MeasurementInfo` members | none; documents the new headers automatically |
| User docs (`docs/` outside `superpowers/`) | no mentions | none |

**Mixed-version runs.** Stores measured before and after the rename carry
different headers for the same quantity. Aggregation takes the union of
columns, so a run measured partly on each side of the rename shows both
prefixes, each `NaN` on the other side's rows. Re-measuring the older stores
(`--mode measure`) brings them onto the new prefix. This is the same thing
that happens when any column is added, so there's no special handling.

### 3.10 Public grid API only; one grid fit per measurement

**Found while planning.** One `measure(synth_plate)` (96 colonies, 600×800)
took ~45 s, and 99.6% of that was the existing directional pass.
`_section_bbox` ran 402 times, each through the **private**
`grid._adv_get_grid_section_slices`. That calls `get_row_edges()` and
`get_col_edges()`, and on `CenteredAutoGridFinder` each of those re-fits the
whole grid (`_fit_grid` → `MeasureBounds` over every colony). That's 804
identical full-plate fits, so the cost grows roughly with the square of plate
size.

**Rule.** `MeasureNeighborDist` uses only public `image.grid` members:
`info()`, `nrows`, `ncols`, `get_row_edges()`, `get_col_edges()`. It memoizes
locally rather than reaching into accessor internals. It fetches each edge
array **once** per measurement and builds a `{(grid_row, grid_col): window}`
dict for every occupied cell. Each window is the cell's grid rectangle,
widened to cover its colonies and clipped to the image, exactly what the
private helper returned. Cells are keyed by `(row, col)`, which removes the
only use of `grid._idx_ref_matrix`.

**Verified before planning.** On the synth plate the public-API windows
matched `_adv_get_grid_section_slices` on 88/88 occupied cells, and building
all of them took 0.09 s. Directional values are therefore unchanged. Tests pin
three things: no `grid._` in the module source, window equality against the
private helper (used as a test-side oracle), and exactly one call to each
edge getter per measurement.

## 4. Edge cases

| Case | Result |
|---|---|
| Fewer than two eligible objects | the three new columns are `NaN` for every row |
| Two candidates at exactly equal distance | the smaller label wins |
| Touching objects (4-adjacent pixels) | `NearestDistance == 1` |
| Nearest object in a non-adjacent cell | `NearestRelation == 3` |
| Off-grid object on a `GridImage` | a row with `NaN` in the new columns; never a candidate |
| Empty object map | the existing `measure()` contract applies; no rows |

## 5. Testing

In `tests/unit/measure/test_measure_grid_spatial.py`, reusing
`_circle` / `_make_synthetic_grid_image`, with exact expected values from the
disc geometry (±1 px rasterisation slack, as the existing EDT tests use):

1. Same-cell satellite: `NearestRelation == 0`, correct label, correct distance.
2. Empty adjacent cell with an occupied diagonal cell: `NearestRelation == 2`.
3. A 1×3 grid with the middle cell empty: the ends report `NearestRelation == 3`.
4. An adjacent nearest: `NearestRelation == 1` and `NearestDistance` equals
   that direction's distance.
5. Plain `Image` (same discs, no grid): all 11 columns present; directional
   columns and `NearestRelation` all `NaN`; nearest label and distance correct.
6. Single object: the three new columns are `NaN`.
7. Tie-break: two candidates at an identical integer offset, so the smaller
   label wins.
8. Off-grid exclusion: call `_nearest_objects` with an eligible set that omits
   a label physically closest to the target; that label is never returned.
9. Property test on `load_synth_yeast_plate()`: `NearestDistance ≤` every
   non-`NaN` directional distance (exact, C4), and the (label, distance)
   pairs equal a brute-force all-pairs `cKDTree` over full masks.
10. Schema: `NEIGHBOR_DIST.get_headers()` contains the three new headers, and
    the measurer output columns equal the schema's headers exactly.
11. Category: `NEIGHBOR_DIST.category() == "NeighborDist"` and every header
    starts with `NeighborDist_`. `ErrorCutoffFinder.measurement_columns`
    selects both a `NeighborDist_` and a legacy `GridSpatial_` column.

**Mutation gate:** before trusting (9), reintroduce an early exit in the
branch and bound (stop after the first candidate) and confirm the
brute-force comparison fails. Then restore it.

## 6. Side updates

- `docs/superpowers/specs/2026-09-01-results-scatter-tab/design.md` records
  `MeasureNeighborDist | 8` and "emits `GridSpatial_*`". Add a dated note that
  it now emits 11 `NeighborDist_*` columns, rather than rewriting the
  historical table.
- `analysis/_error_cutoffs.py` prefix list and its drift-guard test (§3.9).
- `FilamentousFungiPipeline`: no change, since the constructor is unchanged.
- `abc_` docs mentioning grid measurers: no change needed. The class moves to
  a less restrictive base, and nothing else subclasses it.

## 7. Risks

- **Header rename breaks external readers.** User scripts, notebooks, or R
  analyses that select `GridSpatial_*` columns by name stop finding them on
  newly measured data. The spec accepts this, deliberately: there is no alias
  layer for measurement headers. The PR description must call out the rename.

- **Base-class change — checked, low risk.** Nothing in `src/` routes on
  `isinstance(…, GridMeasureFeatures)`. Its only other subclasses are
  `GridFinder`, `MeasureGridSpread`, and `MeasureGridLinRegStats`. The GUI's
  grid detection (`_pipeline_uses_grid`, `builder/_callbacks.py:7973`) keys on
  `GridOperation`, which `GridMeasureFeatures` does not inherit from. The GUI
  registry groups measurers under `MeasureFeatures` (`_operation_registry.py:224`),
  which the class still is. What changes: `GridMeasureFeatures.measure` does
  the `GridImage` check and then defers to `MeasureFeatures.measure`, which
  carries the same `@validate_measure_integrity()`. So integrity validation is
  unchanged, and the class gains `MeasureFeatures.measure`'s `include_meta`
  argument, which the Grid override lacked.
- **Row set on a plain `Image`.** The measurer now runs on a plain `Image`,
  where it previously raised. That is an extension, not a change, for
  existing callers.
- **Output width.** The measurement tables gain three columns. The embedded
  `table.parquet` has no fixed schema across measurers, so no migration is
  needed.
