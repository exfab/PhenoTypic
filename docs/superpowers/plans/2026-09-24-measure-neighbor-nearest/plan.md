# MeasureNeighborDist Nearest-Object Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add plate-wide `NearestObjLabel` / `NearestDistance` / `NearestRelation` columns to `MeasureNeighborDist`, and let it accept a plain `Image`.

**Architecture:** A grid-free private helper `_nearest_objects(objmap, eligible)` does an exact branch-and-bound nearest search. It orders candidates by a bounding-box lower bound and gets exact distances from `cKDTree` queries over 4-connected boundary pixels. `MeasureNeighborDist` moves from `GridMeasureFeatures` to `MeasureFeatures` and branches on `hasattr(image, "grid")`. The existing directional body moves unchanged into `_measure_grid_directions`; the nearest columns and a grid-offset relation code are appended.

**Tech Stack:** numpy, scipy (`ndimage.find_objects`, `ndimage.binary_erosion`, `spatial.cKDTree`), pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-09-24-measure-neighbor-nearest/design.md`. Read it before starting.
**Validation script:** `docs/superpowers/logic_validation_scripts/2026-09-24-measure-neighbor-nearest/nearest_object_bounds.py` (claims C1–C4 cited below).

## Global Constraints

- `uv` only. Never bare `python`/`pip`.
- Test command form (the `run-phenotypic-test` skill): `QT_QPA_PLATFORM=offscreen uv run pytest <paths> -q --no-header -p no:randomly -o addopts= -m "not slow"`. Never `-n auto`. Never quote an `-x` run as a result.
- `uv run ruff check --fix <explicit paths you changed>`. Never bare `ruff check --fix`.
- Class name `MeasureNeighborDist` is unchanged. Schema category is **renamed** `GridSpatial` → `NeighborDist` (spec §3.9), so every header is `NeighborDist_<Label>`. `ErrorCutoffFinder.MEASUREMENT_PREFIXES` keeps `"GridSpatial_"` alongside the new `"NeighborDist_"`.
- Distances are pixel-centre Euclidean, in pixels; touching (4-adjacent) objects = `1.0`.
- `NearestRelation` codes: `0` same cell, `1` adjacent (`dr + dc == 1`), `2` diagonal (`dr == dc == 1`), `3` anything else. `NaN` on a plain `Image` or when there is no nearest.
- Ties go to the smaller label.
- On a `GridImage`, objects with `NaN` `Grid_RowNum` or `Grid_ColNum` keep a row but are never targets or candidates for the nearest search.
- `MeasurementInfo` members: author `label` and `desc` only. Leave `bio_desc` as `""` and `image` unset (CLAUDE.md Gotchas).
- **Timing (measured 2026-09-24):** one `MeasureNeighborDist().measure(synth_plate)` takes **~40 s**. 99.6% of that is the pre-existing directional pass, where `_section_bbox` → `_adv_get_grid_section_slices` → `get_row_edges`/`get_col_edges` re-fits the whole grid on each of 402 calls. The baseline test file takes **6 min 12 s** for 16 tests. Set a ≥600 s timeout on file-level runs, and share one measurement across read-only assertions (the `synth_neighbor_df` fixture in Task 3). This plan does **not** fix that hotspot; see the note in the handoff.
- Keep the 8 existing directional columns' values byte-for-byte unchanged on a `GridImage`. The existing `TestEdtDistance` and `TestMeasureGridSpatialIntegration` classes must pass **without edits**.

## Review Focus

1. **Non-contiguous labels** (objmap `{1, 3, 7}` after a refiner dropped objects). `ndi.find_objects` indexes by `label - 1`, so an off-by-one returns the wrong object's pixels. Pinned in Task 2 (`test_non_contiguous_labels`).
2. **Objects touching the image border.** The crop from `find_objects` touches the array edge. Erosion must still mark the edge pixels as boundary, and coordinates must be shifted back to full-image space. Pinned in Task 2 (`test_border_touching_objects`).
3. **A satellite inside a ring colony's hole.** The nearest pair crosses a hole boundary. Boundary pixels must include hole edges, or the distance comes out as the outer rim's (far too large). Pinned in Task 2 (`test_satellite_inside_ring_hole`).
4. **Touching colonies.** Two 4-adjacent masks must report exactly `1.0`, not `0.0`, and match the directional columns' convention. Pinned in Task 2 (`test_touching_objects_distance_one`) and via Task 3's dominance property.
5. **`include_meta=True`.** The base-class change makes `MeasureFeatures.measure(image, include_meta=True)` reachable, which the Grid override lacked. It must merge without duplicating `Object_Label` or dropping rows. Pinned in Task 3 (`test_include_meta_merges_grid_info`).

## File Map

| File | Change | Task |
|---|---|---|
| `src/phenotypic/schema/_neighbor_dist.py` | category → `NeighborDist`, add 3 members, rewrite class docstring | 1 |
| `src/phenotypic/analysis/_error_cutoffs.py` | add `"NeighborDist_"` prefix, keep `"GridSpatial_"` | 1 |
| `tests/unit/analysis/test_error_cutoffs.py` | prefix drift guard covers both | 1 |
| `tests/unit/measure/test_measure_grid_spatial.py` | add `TestNearestSchema`, `TestNearestObjectsHelper`, `TestNearestColumns`, `TestNearestProperty` | 1–3 |
| `src/phenotypic/measure/_measure_neighbor_dist.py` | add `_nearest_objects`, `_nearest_relation`; base class change; `_operate` split; docstring | 2, 3 |
| `docs/superpowers/specs/2026-09-01-results-scatter-tab/design.md` | dated note: column count is now 11 | 4 |

## Task DAG

`Task 1 → Task 3`, `Task 2 → Task 3`, `Task 3 → Task 4`. Tasks 1 and 2 touch different source files but share the test file, so run them **sequentially** (1, then 2) to avoid merge tax.

---

### Task 1: Schema: `NeighborDist` category and nearest-object members

**Files:**
- Modify: `src/phenotypic/schema/_neighbor_dist.py`
- Test: `tests/unit/measure/test_measure_grid_spatial.py` (append a class)

**Interfaces:**
- Produces: `NEIGHBOR_DIST.category() == "NeighborDist"`. `NEIGHBOR_DIST.NEAREST_OBJ_LABEL` (header `NeighborDist_NearestObjLabel`), `NEIGHBOR_DIST.NEAREST_DISTANCE` (`NeighborDist_NearestDistance`), `NEIGHBOR_DIST.NEAREST_RELATION` (`NeighborDist_NearestRelation`), declared **after** `UNDER_DISTANCE`, so `get_headers()` lists the 8 directional headers first and then these 3. `ErrorCutoffFinder.MEASUREMENT_PREFIXES` contains both `"NeighborDist_"` and `"GridSpatial_"`.

**Also modifies:** `src/phenotypic/analysis/_error_cutoffs.py:34-42` and `tests/unit/analysis/test_error_cutoffs.py:162-178`.

- [ ] **Step 1: Write the failing test.** Append to the test file:

```python
class TestNearestSchema:
    """NEIGHBOR_DIST declares the three nearest-object members, in order."""

    def test_headers_append_nearest_members_after_directional(self):
        headers = NEIGHBOR_DIST.get_headers()
        assert headers[-3:] == [
            "NeighborDist_NearestObjLabel",
            "NeighborDist_NearestDistance",
            "NeighborDist_NearestRelation",
        ]
        assert len(headers) == 11

    def test_category_is_neighbor_dist_for_every_header(self):
        assert NEIGHBOR_DIST.category() == "NeighborDist"
        assert all(h.startswith("NeighborDist_")
                   for h in NEIGHBOR_DIST.get_headers())

    def test_nearest_members_have_no_authored_bio_desc(self):
        for member in (NEIGHBOR_DIST.NEAREST_OBJ_LABEL,
                       NEIGHBOR_DIST.NEAREST_DISTANCE,
                       NEIGHBOR_DIST.NEAREST_RELATION):
            assert member.bio_desc == ""

    def test_relation_desc_documents_every_code(self):
        desc = NEIGHBOR_DIST.NEAREST_RELATION.desc
        for code in ("0", "1", "2", "3"):
            assert code in desc
```

Before running, check that `Entry` members expose `.bio_desc` and `.desc` attributes: `grep -n "bio_desc\|def desc" src/phenotypic/schema/_measurement_info.py`. If the accessor names differ, use the real ones in the test. Don't add accessors.

In `tests/unit/analysis/test_error_cutoffs.py`, edit `test_prefix_set_detects_phenotype_headers_and_excludes_position`: change the `pheno` list to include **both** prefixes, keeping the legacy one:

```python
    pheno = [
        "Size_Area", "Shape_Circularity", "Intensity_MeanIntensity",
        "SymZones_Foo", "NeighborDist_Foo", "GridSpatial_Foo",
        "RadialExpansion_Foo", "TextureGray_Contrast",
    ]
```

- [ ] **Step 2: Run them and confirm they fail.**
Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/measure/test_measure_grid_spatial.py::TestNearestSchema tests/unit/analysis/test_error_cutoffs.py::test_prefix_set_detects_phenotype_headers_and_excludes_position -q --no-header -p no:randomly -o addopts= -m "not slow"`
Expected: FAIL. `AttributeError: NEAREST_OBJ_LABEL` / `len(headers) == 8` / `category() == "GridSpatial"`, and the error-cutoff test fails because `NeighborDist_Foo` isn't selected.

- [ ] **Step 3: Implement.**

In `src/phenotypic/analysis/_error_cutoffs.py`, change `MEASUREMENT_PREFIXES` so it contains both prefixes:

```python
    "SymZones_",
    "NeighborDist_",
    "GridSpatial_",  # pre-2026-09-24 name of NeighborDist_; keeps older tables analysable
    "RadialExpansion_",
```

In `src/phenotypic/schema/_neighbor_dist.py`, change `category()` to return `"NeighborDist"`:

```python
    @classmethod
    def category(cls) -> str:
        return "NeighborDist"
```

Then replace the class docstring and append the members after `UNDER_DISTANCE`:

```python
class NEIGHBOR_DIST(QualityInfo):
    """Measure distances from each colony to its grid neighbours and its nearest object.

    Two families of columns. The directional columns report, for each colony,
    the nearest object in the left, right, above, and below grid cells and the
    minimum Euclidean distance between their pixel masks, computed via a
    per-section distance transform so round colonies are not over-estimated by
    their bounding boxes; edge and corner colonies report ``NaN`` beyond the
    plate boundary. The nearest-object columns report the single closest other
    object anywhere on the plate (same cell, diagonal, or further away) and its
    grid relation to the colony. On a plain ``Image`` only the nearest label and
    distance are populated.
    """
```

```python
    NEAREST_OBJ_LABEL = Entry(
            "NearestObjLabel",
            "The object label of the closest other object anywhere on the plate,"
            " by minimum pixel-to-pixel distance between object masks. On a"
            " GridImage only objects assigned to a grid cell are considered;"
            " on a plain Image every object is. NaN when fewer than two"
            " eligible objects exist. Ties resolve to the smaller label."
    )
    NEAREST_DISTANCE = Entry(
            "NearestDistance",
            "The minimum Euclidean distance, in pixels, between the pixel centres"
            " of this object's mask and the nearest object's mask. Two objects"
            " whose masks share an edge report 1. Always less than or equal to"
            " every non-NaN directional distance for the same object."
    )
    NEAREST_RELATION = Entry(
            "NearestRelation",
            "Integer code for where the nearest object sits relative to this"
            " object's grid cell, from the absolute row offset dr and column"
            " offset dc between the two cells: 0 = same cell (dr = dc = 0);"
            " 1 = edge-adjacent cell (dr + dc = 1); 2 = diagonal cell"
            " (dr = dc = 1); 3 = any cell further away. NaN on a plain Image"
            " (no grid) or when there is no nearest object."
    )
```

- [ ] **Step 4: Run it and confirm it passes**, together with the schema classification guard (asserts every `NEIGHBOR_DIST` member resolves as `quality`) and the whole error-cutoff file:
Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/measure/test_measure_grid_spatial.py::TestNearestSchema tests/unit/schema/test_classification.py tests/unit/analysis/test_error_cutoffs.py -q --no-header -p no:randomly -o addopts= -m "not slow"`
Expected: PASS.

- [ ] **Step 5: Confirm nothing else spells the old prefix.** Run `grep -rn "GridSpatial" src tests`. The only hits allowed are the kept legacy entry in `_error_cutoffs.py`, the legacy `"GridSpatial_Foo"` in its test, and the unchanged test class names `TestMeasureGridSpatial` / `TestMeasureGridSpatialIntegration`. Those class names are left alone so the existing tests stay unedited.

- [ ] **Step 6: Commit.**

```bash
uv run ruff check --fix src/phenotypic/schema/_neighbor_dist.py src/phenotypic/analysis/_error_cutoffs.py tests/unit/measure/test_measure_grid_spatial.py tests/unit/analysis/test_error_cutoffs.py
git add src/phenotypic/schema/_neighbor_dist.py src/phenotypic/analysis/_error_cutoffs.py tests/unit/measure/test_measure_grid_spatial.py tests/unit/analysis/test_error_cutoffs.py
git commit -m "feat(schema): rename GridSpatial category to NeighborDist; add nearest-object members"
```

---

### Task 2: `_nearest_objects` helper (grid-free, exact)

**Files:**
- Modify: `src/phenotypic/measure/_measure_neighbor_dist.py` (add module-level functions above the class)
- Test: `tests/unit/measure/test_measure_grid_spatial.py` (append a class)

**Interfaces:**
- Produces:
  - `_nearest_objects(objmap: np.ndarray, eligible: np.ndarray) -> tuple[np.ndarray, np.ndarray]`. `eligible` is a 1-D int array of labels present in `objmap`. Returns `(nearest_label, nearest_distance)`, both `float64` arrays of length `len(eligible)`, aligned to `eligible`'s order. `NaN` everywhere when `len(eligible) < 2`. Only labels in `eligible` are ever returned. Raises `ValueError` if an eligible label has no pixels.
  - `_nearest_relation(self_rc: np.ndarray, nearest_rc: np.ndarray) -> np.ndarray`. Both arrays are `(n, 2)` float grid `(row, col)`. Returns a float array of codes 0–3.

- [ ] **Step 1: Write the failing tests.** Add these imports to the **top-of-file** import block (not mid-file, which would trip E402):

```python
from scipy.spatial import cKDTree

from phenotypic.measure._measure_neighbor_dist import (
    _nearest_objects,
    _nearest_relation,
)
```

Then append:

```python
def _brute_nearest(objmap: np.ndarray, eligible: np.ndarray):
    """All-pairs nearest over FULL masks (no boundary shortcut, no pruning)."""
    coords = {int(lab): np.argwhere(objmap == lab) for lab in eligible}
    out_label, out_dist = [], []
    for a in eligible:
        best, best_lab = np.inf, np.nan
        for b in sorted(int(x) for x in eligible if x != a):
            d = float(cKDTree(coords[b]).query(coords[int(a)], k=1)[0].min())
            if d < best:
                best, best_lab = d, b
        out_label.append(best_lab)
        out_dist.append(best if np.isfinite(best) else np.nan)
    return np.asarray(out_label, float), np.asarray(out_dist, float)


class TestNearestObjectsHelper:
    """Exact nearest-object search on raw label maps (no grid)."""

    def test_two_discs_edge_to_edge(self):
        objmap = np.zeros((60, 120), np.int32)
        _circle(objmap, 1, 30, 30, 5)
        _circle(objmap, 2, 30, 90, 5)
        lab, dist = _nearest_objects(objmap, np.array([1, 2]))
        assert lab.tolist() == [2.0, 1.0]
        # centre gap 60 minus two radii; ±1 px rasterisation slack
        assert np.all(np.abs(dist - 50.0) <= 1.0)

    def test_fewer_than_two_eligible_is_nan(self):
        objmap = np.zeros((20, 20), np.int32)
        _circle(objmap, 4, 10, 10, 3)
        lab, dist = _nearest_objects(objmap, np.array([4]))
        assert np.isnan(lab).all() and np.isnan(dist).all()
        lab0, dist0 = _nearest_objects(objmap, np.array([], dtype=np.int64))
        assert lab0.shape == (0,) and dist0.shape == (0,)

    def test_ineligible_label_is_never_returned(self):
        # Label 2 is physically closest to 1 but not eligible (an off-grid
        # object on a GridImage); 1 must fall through to 3.
        objmap = np.zeros((40, 120), np.int32)
        _circle(objmap, 1, 20, 20, 4)
        _circle(objmap, 2, 20, 40, 4)
        _circle(objmap, 3, 20, 100, 4)
        lab, _ = _nearest_objects(objmap, np.array([1, 3]))
        assert lab.tolist() == [3.0, 1.0]

    def test_tie_resolves_to_smaller_label_even_when_visited_second(self):
        # Target 5 is a single pixel at (20, 20).
        # Label 2: one pixel at (20, 30): distance 10, bbox lower bound 10.
        # Label 9: pixels (26, 28) and (12, 40): distance 10 via (6, 8), but
        #          bbox rows 12..26 x cols 28..40 gives a lower bound of 8, so 9 is
        #          visited FIRST. The strict `lb > best` stop must still visit 2,
        #          and the tie-break must prefer it.
        objmap = np.zeros((50, 50), np.int32)
        objmap[20, 20] = 5
        objmap[20, 30] = 2
        objmap[26, 28] = 9
        objmap[12, 40] = 9
        lab, dist = _nearest_objects(objmap, np.array([5, 2, 9]))
        assert lab[0] == 2.0
        assert dist[0] == 10.0

    def test_non_contiguous_labels(self):
        objmap = np.zeros((40, 160), np.int32)
        _circle(objmap, 1, 20, 20, 4)
        _circle(objmap, 3, 20, 60, 4)
        _circle(objmap, 7, 20, 140, 4)
        lab, _ = _nearest_objects(objmap, np.array([1, 3, 7]))
        assert lab.tolist() == [3.0, 1.0, 3.0]

    def test_border_touching_objects(self):
        objmap = np.zeros((30, 30), np.int32)
        objmap[0:5, 0:5] = 1        # touches top-left corner of the array
        objmap[25:30, 20:30] = 2    # touches bottom-right edges
        lab, dist = _nearest_objects(objmap, np.array([1, 2]))
        # closest pixels (4, 4) and (25, 20): offset (21, 16)
        assert dist[0] == np.sqrt(21 ** 2 + 16 ** 2)
        assert lab.tolist() == [2.0, 1.0]

    def test_satellite_inside_ring_hole(self):
        objmap = np.zeros((80, 80), np.int32)
        rr, cc = np.ogrid[:80, :80]
        r2 = (rr - 40) ** 2 + (cc - 40) ** 2
        objmap[(r2 <= 30 ** 2) & (r2 > 15 ** 2)] = 1   # ring, hole radius 15
        objmap[r2 <= 3 ** 2] = 2                        # satellite in the hole
        lab, dist = _nearest_objects(objmap, np.array([1, 2]))
        # gap across the hole is ~15 - 3; the outer rim would be ~27 away
        assert lab.tolist() == [2.0, 1.0]
        assert abs(dist[0] - 12.0) <= 1.0

    def test_touching_objects_distance_one(self):
        objmap = np.zeros((20, 20), np.int32)
        objmap[5:10, 5:10] = 1
        objmap[5:10, 10:15] = 2   # shares an edge with label 1
        _, dist = _nearest_objects(objmap, np.array([1, 2]))
        assert dist.tolist() == [1.0, 1.0]

    def test_eligible_label_missing_from_objmap_raises(self):
        objmap = np.zeros((10, 10), np.int32)
        objmap[2, 2] = 1
        with pytest.raises(ValueError, match="label 4"):
            _nearest_objects(objmap, np.array([1, 4]))

    @pytest.mark.parametrize("seed", range(20))
    def test_matches_brute_force_on_random_blobs(self, seed):
        rng = np.random.default_rng(seed)
        objmap = np.zeros((64, 64), np.int32)
        label = 1
        for _ in range(40):
            if label > 9:
                break
            r, c, rad = rng.integers(4, 60), rng.integers(4, 60), rng.integers(1, 5)
            rr, cc = np.ogrid[:64, :64]
            disc = (rr - r) ** 2 + (cc - c) ** 2 <= rad ** 2
            if (objmap[disc] != 0).any():
                continue
            objmap[disc] = label
            label += 1
        eligible = np.unique(objmap[objmap > 0]).astype(np.int64)
        got = _nearest_objects(objmap, eligible)
        want = _brute_nearest(objmap, eligible)
        np.testing.assert_array_equal(got[0], want[0])
        np.testing.assert_array_equal(got[1], want[1])


class TestNearestRelation:
    def test_codes(self):
        self_rc = np.array([[2, 2], [2, 2], [2, 2], [2, 2], [2, 2]], float)
        near_rc = np.array([[2, 2], [2, 3], [1, 2], [3, 3], [2, 4]], float)
        assert _nearest_relation(self_rc, near_rc).tolist() == [0, 1, 1, 2, 3]

    def test_empty(self):
        empty = np.empty((0, 2))
        assert _nearest_relation(empty, empty).shape == (0,)
```

Why `assert_array_equal` with no tolerance: both sides are `sqrt` of the same integer squared distance, computed exactly in float64 and then correctly rounded, so they're bit-identical (spec §3.6 C4). Any tolerance would let the anchor drift.

- [ ] **Step 2: Run them and confirm they fail.**
Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/measure/test_measure_grid_spatial.py -k "NearestObjectsHelper or NearestRelation" -q --no-header -p no:randomly -o addopts= -m "not slow"`
Expected: collection ERROR, `ImportError: cannot import name '_nearest_objects'`.

- [ ] **Step 3: Implement.** In `src/phenotypic/measure/_measure_neighbor_dist.py`, add `from scipy.spatial import cKDTree` next to the `ndi` import. Then add these functions between the imports and the class:

```python
def _boundary_coords(
        objmap: np.ndarray, label: int, sl: tuple[slice, slice]
) -> np.ndarray:
    """Full-image (row, col) coordinates of ``label``'s 4-connected boundary.

    A pixel is boundary when at least one 4-neighbour is outside the object;
    hole edges count. The minimum distance between two disjoint masks is always
    attained on these pixels (spec §3.6 C1). Pixels on the crop edge are
    boundary because ``binary_erosion`` treats outside-the-crop as background,
    which is correct for the tight ``find_objects`` slice.
    """
    mask = objmap[sl] == label
    edge = mask & ~ndi.binary_erosion(mask)
    coords = np.argwhere(edge)
    coords[:, 0] += sl[0].start
    coords[:, 1] += sl[1].start
    return coords


def _nearest_objects(
        objmap: np.ndarray, eligible: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Exact nearest other object for every eligible label.

    Branch and bound: candidates are visited in ascending (bounding-box lower
    bound, label) order, and the search stops at the first candidate whose
    lower bound strictly exceeds the best exact distance found. The box gap
    never exceeds the mask distance (C2), so no nearer object is skipped, and
    equal-distance candidates are still visited for the smaller-label
    tie-break (C3).

    Args:
        objmap: 2-D integer label map.
        eligible: 1-D labels to search among; only these are targets or
            candidates.

    Returns:
        ``(nearest_label, nearest_distance)`` float arrays aligned to
        ``eligible``; ``NaN`` where no other eligible object exists.

    Raises:
        ValueError: If an eligible label has no pixels in ``objmap``.
    """
    eligible = np.asarray(eligible, dtype=np.int64)
    n = eligible.size
    nearest_label = np.full(n, np.nan)
    nearest_dist = np.full(n, np.nan)
    if n < 2:
        return nearest_label, nearest_dist

    slices = ndi.find_objects(objmap)
    boundaries: list[np.ndarray] = []
    bboxes = np.empty((n, 4), dtype=np.int64)  # inclusive min_r, max_r, min_c, max_c
    for i, label in enumerate(eligible):
        sl = slices[label - 1] if 0 < label <= len(slices) else None
        if sl is None:
            raise ValueError(f"eligible label {label} has no pixels in objmap")
        coords = _boundary_coords(objmap, int(label), sl)
        boundaries.append(coords)
        bboxes[i] = (coords[:, 0].min(), coords[:, 0].max(),
                     coords[:, 1].min(), coords[:, 1].max())

    trees: dict[int, cKDTree] = {}
    for i in range(n):
        gap_r = np.maximum(0, np.maximum(bboxes[:, 0] - bboxes[i, 1],
                                         bboxes[i, 0] - bboxes[:, 1]))
        gap_c = np.maximum(0, np.maximum(bboxes[:, 2] - bboxes[i, 3],
                                         bboxes[i, 2] - bboxes[:, 3]))
        # sqrt of an exact integer sum, the same rounding path cKDTree uses, so
        # a lower bound equal to a true distance compares equal rather than
        # 1 ulp high (np.hypot is not guaranteed correctly rounded on every libm),
        # and tied candidates are never skipped.
        lower = np.sqrt((gap_r * gap_r + gap_c * gap_c).astype(np.float64))
        lower[i] = np.inf
        best, best_j = np.inf, -1
        for j in np.lexsort((eligible, lower)):
            if j == i or lower[j] > best:
                break
            if j not in trees:
                trees[j] = cKDTree(boundaries[j])
            d = float(trees[j].query(boundaries[i], k=1)[0].min())
            if d < best or (d == best and eligible[j] < eligible[best_j]):
                best, best_j = d, int(j)
        nearest_label[i] = eligible[best_j]
        nearest_dist[i] = best
    return nearest_label, nearest_dist


def _nearest_relation(self_rc: np.ndarray, nearest_rc: np.ndarray) -> np.ndarray:
    """Grid relation code per spec §3.4 from (row, col) cell positions."""
    dr = np.abs(nearest_rc[:, 0] - self_rc[:, 0])
    dc = np.abs(nearest_rc[:, 1] - self_rc[:, 1])
    step = dr + dc
    return np.select(
            [step == 0, step == 1, (dr == 1) & (dc == 1)],
            [0.0, 1.0, 2.0],
            default=3.0,
    )
```

The `j == i` guard is reachable only after every other candidate is exhausted, because `lower[i] = inf` sorts self last. With `n >= 2`, the first candidate is always finite, so `best_j` is set before the guard can fire.

- [ ] **Step 4: Run the tests and confirm they pass.** Same command as Step 2. Expected: PASS (`TestNearestObjectsHelper` has 29 tests with the parametrization, plus 2 in `TestNearestRelation`).

- [ ] **Step 5: Mutation gate (the test must be able to fail).** Apply each mutation, run the Step 2 command, confirm the listed test fails, then revert with `git checkout -- src/phenotypic/measure/_measure_neighbor_dist.py`, re-applying the Step 3 code if that file was uncommitted:
  - `if j == i or lower[j] > best:` → `if j == i or best < np.inf:` (stop after the first candidate). Expect `test_tie_resolves_to_smaller_label_even_when_visited_second` and several `test_matches_brute_force_on_random_blobs[*]` to FAIL.
  - `or (d == best and eligible[j] < eligible[best_j])` → delete it. Expect the tie test to FAIL.
  - `edge = mask & ~ndi.binary_erosion(mask)` → `edge = ndi.binary_erosion(mask)` (interior only). Expect `test_touching_objects_distance_one` and `test_border_touching_objects` to FAIL.

  Record the failing test names in the commit message body.

- [ ] **Step 6: Commit.**

```bash
uv run ruff check --fix src/phenotypic/measure/_measure_neighbor_dist.py tests/unit/measure/test_measure_grid_spatial.py
git add src/phenotypic/measure/_measure_neighbor_dist.py tests/unit/measure/test_measure_grid_spatial.py
git commit -m "feat(measure): exact branch-and-bound nearest-object helper"
```

---

### Task 3: Wire nearest columns into `MeasureNeighborDist`; accept plain `Image`

**Files:**
- Modify: `src/phenotypic/measure/_measure_neighbor_dist.py` (class)
- Test: `tests/unit/measure/test_measure_grid_spatial.py` (append classes)

**Interfaces:**
- Consumes: `_nearest_objects`, `_nearest_relation` (Task 2); `NEIGHBOR_DIST.NEAREST_*` (Task 1).
- Produces: `MeasureNeighborDist(MeasureFeatures)`. `measure(image)` accepts `Image` or `GridImage` and returns `Object_Label` + the 11 `NEIGHBOR_DIST` columns in enum order. New private method `_measure_grid_directions(self, image, grid_info, objmap_full) -> dict`.

- [ ] **Step 1: Write the failing tests.** In the top-of-file imports, change `from phenotypic import GridImage` to `from phenotypic import GridImage, Image`. Then append:

```python
@pytest.fixture(scope="module")
def synth_neighbor_df(synth_plate):
    """One MeasureNeighborDist run on the synth plate, shared by the
    read-only assertions below: a run costs ~40 s (see Timing)."""
    return MeasureNeighborDist().measure(synth_plate)


def _row(df: pd.DataFrame, label: int) -> pd.Series:
    return df[df[OBJECT.LABEL] == label].iloc[0]


class TestNearestColumns:
    """End-to-end nearest-object columns on synthetic plates."""

    def test_same_cell_satellite(self):
        image = _make_synthetic_grid_image(
                height=100, width=100,
                row_edges=np.array([0, 100]), col_edges=np.array([0, 50, 100]),
                circles=[(1, 50, 25, 8), (2, 50, 40, 2), (3, 50, 75, 8)],
        )
        df = MeasureNeighborDist().measure(image)
        colony, satellite, other = _row(df, 1), _row(df, 2), _row(df, 3)
        assert colony[NEIGHBOR_DIST.NEAREST_OBJ_LABEL] == 2
        assert colony[NEIGHBOR_DIST.NEAREST_RELATION] == 0
        assert abs(colony[NEIGHBOR_DIST.NEAREST_DISTANCE] - (15 - 8 - 2)) <= 1.0
        assert satellite[NEIGHBOR_DIST.NEAREST_OBJ_LABEL] == 1
        assert other[NEIGHBOR_DIST.NEAREST_OBJ_LABEL] == 2
        assert other[NEIGHBOR_DIST.NEAREST_RELATION] == 1

    def test_diagonal_when_adjacent_cells_empty(self):
        image = _make_synthetic_grid_image(
                height=100, width=100,
                row_edges=np.array([0, 50, 100]), col_edges=np.array([0, 50, 100]),
                circles=[(1, 25, 25, 5), (2, 75, 75, 5)],
        )
        df = MeasureNeighborDist().measure(image)
        a = _row(df, 1)
        assert a[NEIGHBOR_DIST.NEAREST_OBJ_LABEL] == 2
        assert a[NEIGHBOR_DIST.NEAREST_RELATION] == 2
        assert abs(a[NEIGHBOR_DIST.NEAREST_DISTANCE]
                   - (np.hypot(50, 50) - 10)) <= 1.5
        # the directional columns can't see a diagonal neighbour
        assert pd.isna(a[NEIGHBOR_DIST.RIGHT_DISTANCE])
        assert pd.isna(a[NEIGHBOR_DIST.UNDER_DISTANCE])

    def test_distant_when_middle_cell_empty(self):
        image = _make_synthetic_grid_image(
                height=100, width=180,
                row_edges=np.array([0, 100]),
                col_edges=np.array([0, 60, 120, 180]),
                circles=[(1, 50, 30, 5), (2, 50, 150, 5)],
        )
        df = MeasureNeighborDist().measure(image)
        for label, other in ((1, 2), (2, 1)):
            r = _row(df, label)
            assert r[NEIGHBOR_DIST.NEAREST_OBJ_LABEL] == other
            assert r[NEIGHBOR_DIST.NEAREST_RELATION] == 3
            assert abs(r[NEIGHBOR_DIST.NEAREST_DISTANCE] - 110.0) <= 1.0

    def test_adjacent_nearest_equals_directional_distance(self):
        image = _make_synthetic_grid_image(
                height=100, width=100,
                row_edges=np.array([0, 100]), col_edges=np.array([0, 50, 100]),
                circles=[(1, 50, 25, 5), (2, 50, 75, 5)],
        )
        df = MeasureNeighborDist().measure(image)
        a = _row(df, 1)
        assert a[NEIGHBOR_DIST.NEAREST_RELATION] == 1
        # same pixel-centre convention, same pair → bit-identical (spec C4)
        assert a[NEIGHBOR_DIST.NEAREST_DISTANCE] == a[NEIGHBOR_DIST.RIGHT_DISTANCE]

    def test_single_object_nearest_is_nan(self):
        image = _make_synthetic_grid_image(
                height=100, width=100,
                row_edges=np.array([0, 100]), col_edges=np.array([0, 50, 100]),
                circles=[(1, 50, 25, 5)],
        )
        df = MeasureNeighborDist().measure(image)
        r = _row(df, 1)
        for col in (NEIGHBOR_DIST.NEAREST_OBJ_LABEL,
                    NEIGHBOR_DIST.NEAREST_DISTANCE,
                    NEIGHBOR_DIST.NEAREST_RELATION):
            assert pd.isna(r[col])

    def test_plain_image_emits_all_columns(self):
        image = Image(arr=np.zeros((100, 100, 3), dtype=np.uint8))
        objmap = np.zeros((100, 100), dtype=np.uint16)
        _circle(objmap, 1, 50, 25, 5)
        _circle(objmap, 2, 50, 75, 5)
        image.objmap[:] = objmap
        assert not hasattr(image, "grid")

        df = MeasureNeighborDist().measure(image)

        assert [str(c) for c in df.columns] == (
                [str(OBJECT.LABEL)] + NEIGHBOR_DIST.get_headers()
        )
        a = _row(df, 1)
        assert a[NEIGHBOR_DIST.NEAREST_OBJ_LABEL] == 2
        assert abs(a[NEIGHBOR_DIST.NEAREST_DISTANCE] - 40.0) <= 1.0
        directional = NEIGHBOR_DIST.get_headers()[:8]
        assert df[directional].isna().all().all()
        assert df[str(NEIGHBOR_DIST.NEAREST_RELATION)].isna().all()

    def test_grid_output_columns_match_schema_exactly(
            self, synth_plate, synth_neighbor_df
    ):
        df = synth_neighbor_df
        assert [str(c) for c in df.columns] == (
                [str(OBJECT.LABEL)] + NEIGHBOR_DIST.get_headers()
        )
        assert len(df) == synth_plate.num_objects

    def test_include_meta_merges_grid_info(self, synth_plate):
        df = MeasureNeighborDist().measure(synth_plate, include_meta=True)
        assert len(df) == synth_plate.num_objects
        assert str(GRID.ROW_NUM) in [str(c) for c in df.columns]
        assert [str(c) for c in df.columns].count(str(OBJECT.LABEL)) == 1


class TestNearestProperty:
    """Invariants on the real synthetic yeast plate."""

    def test_nearest_never_exceeds_any_directional_distance(self, synth_neighbor_df):
        df = synth_neighbor_df
        nearest = df[NEIGHBOR_DIST.NEAREST_DISTANCE].to_numpy()
        checked = 0
        for col in (NEIGHBOR_DIST.LEFT_DISTANCE, NEIGHBOR_DIST.RIGHT_DISTANCE,
                    NEIGHBOR_DIST.ABOVE_DISTANCE, NEIGHBOR_DIST.UNDER_DISTANCE):
            d = df[col].to_numpy()
            ok = ~np.isnan(d)
            # exact: both are sqrt of the same integer squared distance (C4)
            assert np.all(nearest[ok] <= d[ok])
            checked += int(ok.sum())
        assert checked > 0

    def test_matches_brute_force_on_synth_plate(self, synth_plate, synth_neighbor_df):
        df = synth_neighbor_df
        objmap = synth_plate.objmap[:]
        labels = df[OBJECT.LABEL].to_numpy().astype(np.int64)
        want_label, want_dist = _brute_nearest(objmap, labels)
        np.testing.assert_array_equal(
                df[NEIGHBOR_DIST.NEAREST_OBJ_LABEL].to_numpy(), want_label)
        np.testing.assert_array_equal(
                df[NEIGHBOR_DIST.NEAREST_DISTANCE].to_numpy(), want_dist)
```

`synth_plate` is the session-scoped fixture in `tests/unit/conftest.py`: a `GridImage` with 96 objects on a 600×800 image, as the existing classes use it. `_brute_nearest` over 96 full masks is about 9k KD queries, which takes a few seconds.

- [ ] **Step 2: Run them and confirm they fail.**
Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/measure/test_measure_grid_spatial.py -k "NearestColumns or NearestProperty" -q --no-header -p no:randomly -o addopts= -m "not slow"`
Expected: FAIL. `KeyError` on the `NeighborDist_Nearest*` columns, and `test_plain_image_emits_all_columns` fails with `OperationFailedError` wrapping `GridImageInputError`.

- [ ] **Step 3: Implement.** In `src/phenotypic/measure/_measure_neighbor_dist.py`:

  1. Change the imports:

```python
if TYPE_CHECKING:
    from phenotypic import Image
    from phenotypic._core._grid_image import GridImage
```
```python
from phenotypic.abc_ import MeasureFeatures
```
  and remove `GridMeasureFeatures` from the imports.

  2. Change the class line to `class MeasureNeighborDist(MeasureFeatures):`.

  3. Rename the existing `_operate` to `_measure_grid_directions(self, image: GridImage, grid_info: pd.DataFrame, objmap_full: np.ndarray) -> dict`. Make these edits and keep everything else unchanged:
     - delete its first three statements (`grid_info = ...`, `objmap_full = ...` and `nrows, ncols = ...`), and replace them with `nrows, ncols = image.grid.nrows, image.grid.ncols`;
     - delete the final three lines (`df = pd.DataFrame(results)`, `df.insert(...)`, `return df`) and replace them with `return results`.

  The body in between (label bookkeeping, `section_groups`, the per-section EDT loop) is untouched.

  4. Add the new `_operate` above it:

```python
    def _operate(self, image: Image) -> pd.DataFrame:
        objmap = image.objmap[:]
        directional = [col for _, _, lab, dist in self._DIRECTIONS
                       for col in (lab, dist)]
        if hasattr(image, "grid"):
            info = image.grid.info(include_metadata=False)
            results = self._measure_grid_directions(image, info, objmap)
            grid_rc = info[[GRID.ROW_NUM, GRID.COL_NUM]].to_numpy(dtype=float)
            eligible_rows = np.flatnonzero(~np.isnan(grid_rc).any(axis=1))
        else:
            info = image.objects.info(include_metadata=False)
            results = {col: np.full(len(info), np.nan) for col in directional}
            grid_rc = None
            eligible_rows = np.arange(len(info))

        labels = info[OBJECT.LABEL].to_numpy().astype(np.int64)
        n_objs = labels.size
        nearest_label = np.full(n_objs, np.nan)
        nearest_dist = np.full(n_objs, np.nan)
        relation = np.full(n_objs, np.nan)

        e_label, e_dist = _nearest_objects(objmap, labels[eligible_rows])
        nearest_label[eligible_rows] = e_label
        nearest_dist[eligible_rows] = e_dist

        if grid_rc is not None:
            label_to_row = {int(lab): i for i, lab in enumerate(labels)}
            found = ~np.isnan(e_label)
            rows_self = eligible_rows[found]
            rows_near = np.array(
                    [label_to_row[int(lab)] for lab in e_label[found]],
                    dtype=np.int64,
            )
            relation[rows_self] = _nearest_relation(
                    grid_rc[rows_self], grid_rc[rows_near]
            )

        results[NEIGHBOR_DIST.NEAREST_OBJ_LABEL] = nearest_label
        results[NEIGHBOR_DIST.NEAREST_DISTANCE] = nearest_dist
        results[NEIGHBOR_DIST.NEAREST_RELATION] = relation

        df = pd.DataFrame(results)
        df.insert(0, OBJECT.LABEL, info[OBJECT.LABEL].to_numpy())
        return df
```

  `results` from `_measure_grid_directions` is a dict literal in enum order (Left, Right, Above, Under × label/dist). Appending the 3 nearest keys afterwards gives the full enum order, which `test_grid_output_columns_match_schema_exactly` pins.

  5. Rewrite the class docstring's first paragraph and add to `Best For` (the `Returns:` block lists the new columns in one line each; per-column detail stays in the enum `desc`):

```python
    """Measure distances to grid neighbours and to the nearest object.

    For each detected colony, report (a) the nearest object in the left,
    right, above, and below grid cells with the minimum Euclidean distance
    between their pixel masks, and (b) the single closest other object
    anywhere on the plate, with its distance and its grid relation (same cell,
    adjacent, diagonal, or further). Directional distances use a per-section
    distance transform over a local window; the nearest-object search is an
    exact branch and bound over mask boundaries. Works on a ``GridImage``
    (objects without a grid cell are excluded from the nearest search) and on a
    plain ``Image``, where only the nearest label and distance are populated.

    Returns:
        pd.DataFrame: Object-level neighbor measurements with columns:

            - Label: unique object identifier.
            - LeftNeighborObjLabel, LeftDistance.
            - RightNeighborObjLabel, RightDistance.
            - AboveNeighborObjLabel, AboveDistance.
            - UnderNeighborObjLabel, UnderDistance.
            - NearestObjLabel, NearestDistance, NearestRelation.
            - ``NaN`` where no neighbor exists (out of plate, empty
              neighbor cell, shielded by a cellmate, no grid, or fewer than
              two eligible objects).

    Best For:
        - Quantifying colony spacing and crowding to assess nutrient
          competition risk on arrayed plates, especially for round or
          irregular colony shapes where bounding-box geometry overstates
          proximity.
        - Screening for satellite colonies, fragments, and contaminants:
          a small ``NearestDistance`` with a same-cell ``NearestRelation``.
        - Flagging closely spaced colonies that may cross-contaminate.
        - Enabling neighbor-aware paired statistical comparisons for
          competition or cooperation studies.
```
  Keep `Consider Also` / `See Also` unchanged.

- [ ] **Step 4: Run the whole test file and confirm it passes.** The existing classes must pass unedited:
Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/measure/test_measure_grid_spatial.py -q --no-header -p no:randomly -o addopts= -m "not slow"`
Expected: PASS, all tests.

- [ ] **Step 5: Mutation gate.** In `_operate`, change `eligible_rows = np.flatnonzero(~np.isnan(grid_rc).any(axis=1))` to `eligible_rows = np.arange(len(info))[:-1]` (drops one object from the search). Run `-k "NearestProperty"` and confirm `test_matches_brute_force_on_synth_plate` FAILS. Then revert.

- [ ] **Step 6: Commit.**

```bash
uv run ruff check --fix src/phenotypic/measure/_measure_neighbor_dist.py tests/unit/measure/test_measure_grid_spatial.py
git add src/phenotypic/measure/_measure_neighbor_dist.py tests/unit/measure/test_measure_grid_spatial.py
git commit -m "feat(measure): MeasureNeighborDist nearest-object columns; accept plain Image"
```

---

### Task 4: Side updates and phase gate

**Files:**
- Modify: `docs/superpowers/specs/2026-09-01-results-scatter-tab/design.md` (after the grouping table ending "Unattributed | 15 | …")

- [ ] **Step 1: Add a dated note** directly under that table:

```markdown
> **2026-09-24:** `MeasureNeighborDist` now emits 11 columns under the
> renamed `NeighborDist_*` prefix (was `GridSpatial_*`). It adds
> `NeighborDist_NearestObjLabel`, `_NearestDistance`, and `_NearestRelation`
> (spec `2026-09-24-measure-neighbor-nearest`). The counts and prefix above
> are the historical record of the run they were taken from.
```

- [ ] **Step 2: Run the affected surface once.** These are the measurer, schema, analysis (it treats `NeighborDist_*` and legacy `GridSpatial_*` as phenotype columns), the GUI scatter grouping (keys on the measurer), and the prefab that constructs it:
Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/measure tests/unit/schema tests/unit/analysis/test_error_cutoffs.py tests/unit/gui/results_viewer/test_scatter_grouping.py -q --no-header -p no:randomly -o addopts= -m "not slow"`
Then find and run any prefab test: `grep -rln "FilamentousFungiPipeline" tests | head`.
Expected: PASS. Run any failure in isolation before attributing it to this change.

- [ ] **Step 3: Static checks against the baseline.** `uv run ruff check src/phenotypic/measure/_measure_neighbor_dist.py src/phenotypic/schema/_neighbor_dist.py tests/unit/measure/test_measure_grid_spatial.py` should be clean. For mypy, run `uv run mypy src/phenotypic/measure/_measure_neighbor_dist.py src/phenotypic/schema/_neighbor_dist.py` on this branch and on `main`'s versions of those files, and confirm the error count did not rise. The repo-wide mypy baseline is already red, so don't report it as "passes".

- [ ] **Step 4: Re-run the independent witness.** `uv run python docs/superpowers/logic_validation_scripts/2026-09-24-measure-neighbor-nearest/nearest_object_bounds.py`. Expected: `OK`, exit 0.

- [ ] **Step 5: Commit.**

```bash
git add docs/superpowers/specs/2026-09-01-results-scatter-tab/design.md
git commit -m "docs: note MeasureNeighborDist column count change in scatter spec"
```

- [ ] **Step 6: End of implementation.** The full sharded regression runs **once**, here, per CLAUDE.md's "Focused between phases" table. Use the committed batch script `docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch` (`slurm-job` skill), or leave it to CI on the PR. Don't run it mid-plan.
