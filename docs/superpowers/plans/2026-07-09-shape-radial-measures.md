# Shape Radial Measures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `MeasureShape`'s misnamed distance-transform "radius" columns with an honest set of radial measurements computed per-object, and fix two correctness bugs found along the way.

**Architecture:** All per-object geometry moves into two private methods on `MeasureShape` that take the object's cropped boolean mask (`regionprops.image`) and return a dict of `SHAPE` headers. Because that crop is already label-isolated, computing the Euclidean distance transform on it fixes the merged-colony bug for free. The new `RobustMeanRadius` comes from a radial signature sampled uniformly **by angle**, not along the boundary, which is what makes it resistant to runners.

**Tech Stack:** Python 3.12, pydantic v2, numpy, scipy (`ndimage`, `spatial`, `stats`), scikit-image (`measure.find_contours`), pandas, pytest, uv.

## Global Constraints

- `uv` is the sole runner. Never invoke bare `python` or `pip`.
- Operations are pydantic v2 models: keyword-only construction, class-level annotated fields, **no `__init__`**. Guards go in `Field(...)` bounds or a `field_validator`.
- Google-style docstrings everywhere. `MeasureFeatures` docstrings explain **parameters** and give a high-level overview; per-column detail lives on the `MeasurementInfo` enum members.
- **Never author or edit `bio_desc`** on any `Entry`. New members get `bio_desc` omitted (defaults to `""`) and `image` unset. Human-authored only.
- Never create example files or notebooks. Examples go in docstrings.
- A check that cannot run must **fail**, not skip. Every numeric tolerance in this plan is derived from a stated mechanism, and every test is paired with a mutation that must make it fail.
- `SHAPE` gains no new `kind`. All new members are ordinary `PrimaryMeasure` entries with `tier=1`. **No QC / `QualityInfo` columns.**
- Do **not** add `TuneSpec` annotations to the new fields. The tune annotation-coverage gate walks `ANNOTATED_MODULES = (detect, enhance, refine, grid, correction)` only (`tests/unit/tune/_annotation_introspect.py:32`). `measure/` is out of scope, so plain `Field(...)` bounds are sufficient and nothing needs adding to `tests/fixtures/tune/annotation_allowlist.json`.
- Run `uv run ruff check --fix` and `uv run mypy src/phenotypic` before each commit.

## Background: what is actually wrong today

Three defects, all in `src/phenotypic/measure/_measure_shape.py`, all verified against the installed scipy 1.16.3 and `load_synth_yeast_plate()`:

1. **`Shape_ConvexArea` is a perimeter.** `scipy.spatial.ConvexHull.area` is the *surface area* of the hull, which in 2D is its **perimeter**; `.volume` is the area. The class docstring says so verbatim. So `Shape_Solidity = area / perimeter` is not even dimensionless. On a real colony it currently reports **9.6197**. Note that swapping in `.volume` is *not* the right fix either: a hull through integer pixel centers undercuts the pixel count, giving Solidity **1.0789** on a filled rectangle. `regionprops.area_convex` counts pixels of the convex image and gives exactly **1.0**.

2. **The EDT merges touching colonies.** `distance_transform_edt` binarizes its input on the first line of its body (`_morphology.py:2569`: `np.where(input, 1, 0)`), so passing the whole `objmap` throws the labels away. Two labels sharing an edge see no background between them. On a 43×43 fixture with labels 1 (41×20) and 2 (41×21) sharing their full internal edge, the current code reports `MaxRadius` of **20.0 / 21.0** where the true inscribed radii are **10.0 / 11.0**.

3. **`MeanRadius` and `MedianRadius` are not radii.** They are the mean and median of the EDT over the object's interior, i.e. mean depth-from-boundary. On an ideal disk of radius R they equal R/3 and R(1−1/√2)≈0.293R. Measured on the 96 real colonies with `Area > 200`: `MeanRadius / MaxRadius` has median **0.3321** (ideal 0.3333) and `MedianRadius / MaxRadius` has median **0.2788** (ideal 0.2929). The names are wrong, not the arithmetic.

## Why the radial signature is sampled by angle

For a colony of radius 40 with a runner of half-width 3 reaching out to 90, the runner's share of the samples depends entirely on how the boundary is sampled:

| sampling scheme | fraction of samples with r > 45 | 20% trimmed mean |
|---|---|---|
| inner boundary pixels | 29.9% | 42.03 |
| contour vertices (≈ arc length) | 23.3% | 40.65 |
| **uniform in angle, K=360** | **2.5%** | **40.03** |

A 20% trimmed mean has a breakdown point of 20%. Boundary-pixel sampling hands the runner 29.9% contamination, which **exceeds the breakdown point**, so the estimator is broken by construction. Sampling by angle gives the runner its true 2.4% angular width, well inside the breakdown point. The sampling scheme does more work here than the choice of robust estimator.

## Naming decision

`Shape_MaxRadius` currently holds the inscribed radius. It is **retired**, not repurposed. Reusing a public column name with new semantics is the exact failure this plan exists to unwind.

| old column | new column | meaning |
|---|---|---|
| `Shape_MeanRadius` | `Shape_MeanBoundaryDist` | mean EDT over the interior (unchanged value, honest name) |
| `Shape_MedianRadius` | `Shape_MedianBoundaryDist` | median EDT over the interior |
| `Shape_MaxRadius` | `Shape_InscribedRadius` | `edt.max()`, the largest inscribed circle |
| — | `Shape_RobustMeanRadius` | 20% trimmed mean of the angular radial signature |
| — | `Shape_ReachRadius` | max of the radial signature, the furthest boundary point |

Two candidate QC columns (`NumRadialPeaks`, `RadialAreaRatio`) were considered and **dropped**. `RadialAreaRatio` does not discriminate once the center is the EDT peak (crescent 1.0436, equal doublet 1.0077, ellipse 1.0058 — no band separates them). `NumRadialPeaks` does discriminate, but detecting merged colonies is `refine.SeparateObjects`'s job, and the column has near-zero variance on correctly segmented data (95 of 96 real colonies report 1).

## File Structure

| File | Responsibility |
|---|---|
| `src/phenotypic/schema/_shape.py` | Modify: rename 3 members, add 2. Pure data. |
| `src/phenotypic/measure/_measure_shape.py` | Modify: 3 bug fixes, 2 new private methods, 3 new pydantic fields. |
| `tests/unit/measure/test_measure_shape.py` | Create: integration tests through `MeasureShape.measure()`. |
| `tests/unit/measure/test_radial_profile.py` | Create: unit tests of the two private methods against analytic shapes. |
| `tests/unit/schema/test_classification.py` | Modify: `tier1` set in `test_shape_straddles_tier1_and_tier2` (line ~124). |
| `src/phenotypic/schema/CLAUDE.md` | Modify: straddler example mentions "radii". |

`tests/unit/analysis/test_edge_correction.py` contains the string `"MeanRadius"` at lines 587-609, but it is a **fabricated column in a synthetic DataFrame**, unrelated to the `SHAPE` enum. Do not touch it. It will show up in greps; ignore it.

There is no existing `tests/unit/measure/test_measure_shape.py` and no golden parquet covering `Shape_*`, so no fixtures need regenerating.

---

## Execution: dependency DAG and clusters

Derived from the per-task `Files` / `Interfaces` blocks via the
`orchestration-clustering` procedure.

```
T1 ──▶ T2 ──▶ T3 ──▶ T4 ──▶ T5
```

Shared files (why the chain is strict):

| file | tasks |
|---|---|
| `src/phenotypic/measure/_measure_shape.py` | T1, T2, T3, T4 |
| `src/phenotypic/schema/_shape.py` | T3, T4 |
| `tests/unit/schema/test_classification.py` | T3, T4 |
| `tests/unit/measure/test_measure_shape.py` | T1, T2, T3 |

**No pair of tasks has zero file overlap, so there are no parallel-worktree
candidates.** Everything runs sequentially.

| task | shape | rationale |
|---|---|---|
| T1 | Leaf | two-line fix, own test, independent of the rest |
| T2 | Keystone | introduces `_measure_radial_profile`, changes column semantics |
| T3 | **Seam** | breaking public rename; one missed site silently breaks downstream |
| T4 | Keystone | the radial-signature algorithm |
| T5 | Leaf | one docs line plus verification |

| cluster | tasks | model / effort | why |
|---|---|---|---|
| **A** | T1 + T2 | Opus, high | both are correctness bugs in one file, one reviewable diff, two commits |
| **B** | T3 | Opus, high | Seam isolated for a focused gate — risk ≠ size |
| **C** | T4 | Opus, high | Keystone: novel geometry, needs judgment |
| **D** | T5 | orchestrator, inline | trivial (one line, one file) |

Gates: light diff review + tests after each cluster. Deep code-review agent over
the combined diff after C. Simplify pass, then regression over
`tests/unit/measure tests/unit/schema tests/unit/util tests/unit/post tests/smoke`.

Never review with a model weaker than the implementer: all gates run on Opus.

---

### Task 1: Fix `ConvexArea` and `Solidity`

**Files:**
- Modify: `src/phenotypic/measure/_measure_shape.py:167-180`
- Test: `tests/unit/measure/test_measure_shape.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: a `_square_objmap_image()` pytest fixture reused by Task 2.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/measure/test_measure_shape.py`:

```python
"""Unit tests for MeasureShape."""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic import Image
from phenotypic.measure import MeasureShape


@pytest.fixture
def split_rectangle_image() -> Image:
    """A 43x43 image whose objmap holds two labels sharing their full internal edge.

    Label 1 is a 41x20 rectangle, label 2 is a 41x21 rectangle. They touch
    along a 41-pixel edge with no background between them, which is what a
    watershed split of a merged colony pair looks like.
    """
    rgb = np.zeros((43, 43, 3), dtype=np.uint8)
    rgb[1:42, 1:42] = 200
    image = Image(rgb)
    objmap = np.zeros((43, 43), dtype=int)
    objmap[1:42, 1:21] = 1
    objmap[1:42, 21:42] = 2
    image.objmap[:] = objmap
    return image


def test_convex_area_is_an_area_not_a_perimeter(split_rectangle_image):
    """A filled rectangle is its own convex hull: ConvexArea == Area, Solidity == 1."""
    measurements = MeasureShape().measure(split_rectangle_image)

    # Label 1 is a 41x20 rectangle => 820 pixels. It is convex, so its convex
    # hull is itself. Exact equality is correct here: both sides are integer
    # pixel counts from regionprops, not floating-point geometry.
    assert measurements["Shape_Area"].iloc[0] == 820.0
    assert measurements["Shape_ConvexArea"].iloc[0] == 820.0
    assert measurements["Shape_Solidity"].iloc[0] == pytest.approx(1.0, abs=1e-12)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/unit/measure/test_measure_shape.py::test_convex_area_is_an_area_not_a_perimeter -v
```

Expected: FAIL. `Shape_ConvexArea` is `118.0` (the hull perimeter) and `Shape_Solidity` is `6.9492`.

- [ ] **Step 3: Fix the implementation**

In `src/phenotypic/measure/_measure_shape.py`, the block currently reading:

```python
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Qhull")
                    convex_hull = ConvexHull(current_props.coords)

            except QhullError:
                convex_hull = None

            measurements[str(SHAPE.CONVEX_AREA)][idx] = (
                convex_hull.area if convex_hull else np.nan
            )
            measurements[str(SHAPE.SOLIDITY)][idx] = (
                (current_props.area / convex_hull.area) if convex_hull else np.nan
            )
```

becomes:

```python
            # ConvexArea and Solidity come from regionprops, which counts the
            # pixels of the convex image. scipy's ConvexHull is retained only
            # for the Feret calipers below: in 2D its `.area` is the hull
            # perimeter and its `.volume` is the hull area, and even `.volume`
            # undercounts because the hull passes through pixel centres.
            measurements[str(SHAPE.CONVEX_AREA)][idx] = current_props.area_convex
            measurements[str(SHAPE.SOLIDITY)][idx] = current_props.solidity

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Qhull")
                    convex_hull = ConvexHull(current_props.coords)

            except QhullError:
                convex_hull = None
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/unit/measure/test_measure_shape.py::test_convex_area_is_an_area_not_a_perimeter -v
```

Expected: PASS.

- [ ] **Step 5: Verify a real colony is now sane**

```bash
uv run python -c "
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureShape
df = MeasureShape().measure(OtsuDetector().apply(load_synth_yeast_plate()))
big = df[df['Shape_Area'] > 200]
print('solidity range:', round(big['Shape_Solidity'].min(),4), '-', round(big['Shape_Solidity'].max(),4))
assert (big['Shape_Solidity'] <= 1.0).all(), 'solidity must not exceed 1'
print('OK')
"
```

Expected: every solidity in `(0, 1]`, ending with `OK`. Before the fix, values were around 9.6.

- [ ] **Step 6: Lint, typecheck, commit**

```bash
uv run ruff check --fix
uv run mypy src/phenotypic
git add src/phenotypic/measure/_measure_shape.py tests/unit/measure/test_measure_shape.py
git commit -m "fix(measure): ConvexArea was a hull perimeter, Solidity was area/perimeter

scipy.spatial.ConvexHull.area is the hull perimeter in 2D, not its area.
Use regionprops.area_convex/.solidity, which count pixels of the convex
image and so stay consistent with Shape_Area."
```

---

### Task 2: Compute the distance transform per object

**Files:**
- Modify: `src/phenotypic/measure/_measure_shape.py:120-140`
- Test: `tests/unit/measure/test_measure_shape.py`

**Interfaces:**
- Consumes: the `split_rectangle_image` fixture from Task 1.
- Produces: `MeasureShape._measure_radial_profile(self, obj_mask: np.ndarray) -> dict[str, float]`, returning keys `Shape_MeanRadius`, `Shape_MedianRadius`, `Shape_MaxRadius`. Task 3 renames those keys; Task 4 adds two more.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/measure/test_measure_shape.py`:

```python
def test_touching_labels_do_not_inflate_each_others_radii(split_rectangle_image):
    """distance_transform_edt binarizes its input, so a whole-objmap EDT merges
    adjacent labels. Each object must be transformed in isolation.

    Label 1 is 41x20, so its largest inscribed circle has radius 20/2 = 10.
    Label 2 is 41x21, so the discrete inscribed radius is 11. Computed over the
    merged 41x41 block instead, the two report 20.0 and 21.0.
    """
    measurements = MeasureShape().measure(split_rectangle_image)

    # Exact: the EDT of an axis-aligned rectangle is exact integer arithmetic
    # (scipy reconstructs distances from an int32 feature transform), so the
    # inscribed radius of a rectangle of even width is exactly half that width.
    assert measurements["Shape_MaxRadius"].iloc[0] == 10.0
    assert measurements["Shape_MaxRadius"].iloc[1] == 11.0

    # Mean depth-from-boundary. Tolerance 1e-3 is far below the 2.56-pixel
    # error the merged EDT produces, so this assertion cannot pass by accident.
    assert measurements["Shape_MeanRadius"].iloc[0] == pytest.approx(4.6951, abs=1e-3)
    assert measurements["Shape_MeanRadius"].iloc[1] == pytest.approx(4.8676, abs=1e-3)
    assert measurements["Shape_MedianRadius"].iloc[0] == pytest.approx(4.0, abs=1e-9)


def test_merged_edt_would_fail_this_test():
    """Mutation control: prove the fixture above can detect the bug it guards.

    Reproduce the old whole-objmap EDT and assert it gives the wrong answer.
    If this test ever starts reporting the correct values, the fixture has
    stopped exercising the merge and the test above is no longer load-bearing.
    """
    from scipy.ndimage import distance_transform_edt

    objmap = np.zeros((43, 43), dtype=int)
    objmap[1:42, 1:21] = 1
    objmap[1:42, 21:42] = 2

    merged = distance_transform_edt(objmap)
    assert merged[objmap == 1].max() == 20.0  # not 10.0
    assert merged[objmap == 2].max() == 21.0  # not 11.0
```

- [ ] **Step 2: Run the tests to verify the first fails**

```bash
uv run pytest tests/unit/measure/test_measure_shape.py -v -k "touching or merged_edt"
```

Expected: `test_touching_labels_do_not_inflate_each_others_radii` FAILS (`Shape_MaxRadius` is 20.0, not 10.0). `test_merged_edt_would_fail_this_test` PASSES.

- [ ] **Step 3: Add the helper and delete the whole-objmap transform**

In `src/phenotypic/measure/_measure_shape.py`, add this method to `MeasureShape` immediately after `_calculate_feret_diameters`:

```python
    def _measure_radial_profile(self, obj_mask: np.ndarray) -> dict[str, float]:
        """Compute distance-transform measures for one cropped object.

        The Euclidean distance transform binarizes its input, so it must be
        given a single object's mask rather than the whole labelled objmap;
        otherwise two touching colonies see no background between them and
        both report inflated distances. The mask is padded by one pixel so
        that the transform sees background on every side of the bounding box.

        Args:
            obj_mask (np.ndarray): Boolean mask of a single object within its
                bounding box, as produced by ``regionprops.image``. Already
                isolated from neighbouring labels.

        Returns:
            dict[str, float]: Mapping of ``SHAPE`` column header to value.
        """
        # asarray narrows the scipy stub's tuple-or-ndarray return union; with
        # return_indices=False the call yields an ndarray and copies nothing.
        edt = np.asarray(distance_transform_edt(np.pad(obj_mask, 1)))[1:-1, 1:-1]
        interior = edt[obj_mask]
        return {
            str(SHAPE.MEAN_RADIUS): float(interior.mean()),
            str(SHAPE.MEDIAN_RADIUS): float(np.median(interior)),
            str(SHAPE.MAX_RADIUS): float(edt.max()),
        }
```

Delete these lines from `_operate` (they sit just above `obj_props = image.objects.props`):

```python
        # Calculate width-based measurements using distance transform
        # Distance transform gives the distance from each object pixel to the nearest background pixel
        dist_matrix = distance_transform_edt(image.objmap[:])
        measurements[str(SHAPE.MEAN_RADIUS)] = self._calculate_mean(
                array=dist_matrix, objmap=image.objmap[:]
        )
        measurements[str(SHAPE.MEDIAN_RADIUS)] = self._calculate_median(
                array=dist_matrix, objmap=image.objmap[:]
        )
        measurements[str(SHAPE.MAX_RADIUS)] = self._calculate_maximum(
                array=dist_matrix, objmap=image.objmap[:]
        )
```

Inside the `for idx, obj_image in enumerate(image.objects):` loop, immediately after `current_props = obj_props[idx]`, add:

```python
            for header, value in self._measure_radial_profile(current_props.image).items():
                measurements[header][idx] = value
```

- [ ] **Step 4: Run the tests to verify both pass**

```bash
uv run pytest tests/unit/measure/test_measure_shape.py -v
```

Expected: all PASS.

- [ ] **Step 5: Confirm the real-colony ratios are unchanged**

Separated colonies must be unaffected by this fix.

```bash
uv run python -c "
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureShape
df = MeasureShape().measure(OtsuDetector().apply(load_synth_yeast_plate()))
big = df[df['Shape_Area'] > 200]
r = (big['Shape_MeanRadius'] / big['Shape_MaxRadius']).median()
print('MeanRadius/MaxRadius median =', round(r, 4), '(ideal disk 0.3333)')
assert 0.32 < r < 0.35
print('OK')
"
```

Expected: `0.3321`-ish, ending with `OK`. The synth plate has no touching colonies, so the ratio is preserved.

- [ ] **Step 6: Lint, typecheck, commit**

```bash
uv run ruff check --fix
uv run mypy src/phenotypic
git add src/phenotypic/measure/_measure_shape.py tests/unit/measure/test_measure_shape.py
git commit -m "fix(measure): compute the EDT per object, not over the whole objmap

distance_transform_edt binarizes its input (_morphology.py:2569), so
passing a labelled objmap discards the labels and touching colonies see
no background between them. Two labels sharing an edge reported inscribed
radii of 20/21 where the true values are 10/11."
```

---

### Task 3: Rename the three misnamed members

**Files:**
- Modify: `src/phenotypic/schema/_shape.py:52-66`
- Modify: `src/phenotypic/measure/_measure_shape.py` (the `_measure_radial_profile` keys and the class docstring)
- Modify: `tests/unit/schema/test_classification.py:124-126`
- Modify: `tests/unit/measure/test_measure_shape.py` (column names)

**Interfaces:**
- Consumes: `_measure_radial_profile` from Task 2.
- Produces: `SHAPE.MEAN_BOUNDARY_DIST`, `SHAPE.MEDIAN_BOUNDARY_DIST`, `SHAPE.INSCRIBED_RADIUS` with values `Shape_MeanBoundaryDist`, `Shape_MedianBoundaryDist`, `Shape_InscribedRadius`. `SHAPE.MEAN_RADIUS`, `SHAPE.MEDIAN_RADIUS`, `SHAPE.MAX_RADIUS` no longer exist.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/measure/test_measure_shape.py`:

```python
def test_radius_columns_are_named_for_what_they_measure():
    """MeanRadius/MedianRadius were means of the distance transform, i.e. depth
    from the boundary, not radii. MaxRadius was the inscribed radius. All three
    are renamed; the old names must be gone.
    """
    from phenotypic.schema import SHAPE

    headers = set(SHAPE.get_headers())
    assert "Shape_MeanBoundaryDist" in headers
    assert "Shape_MedianBoundaryDist" in headers
    assert "Shape_InscribedRadius" in headers
    assert "Shape_MeanRadius" not in headers
    assert "Shape_MedianRadius" not in headers
    assert "Shape_MaxRadius" not in headers
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/unit/measure/test_measure_shape.py::test_radius_columns_are_named_for_what_they_measure -v
```

Expected: FAIL with `assert 'Shape_MeanBoundaryDist' in headers`.

- [ ] **Step 3: Rename the members in the schema**

In `src/phenotypic/schema/_shape.py`, replace the `MEDIAN_RADIUS` / `MEAN_RADIUS` / `MAX_RADIUS` block (lines 52-66) with:

```python
    MEDIAN_BOUNDARY_DIST = Entry(
        "MedianBoundaryDist",
        "Median Euclidean distance from each colony pixel to the nearest background "
        "pixel, computed on the object in isolation. This is a measure of interior "
        "thickness, not a radius: for an ideal disk of radius R it equals "
        r":math:`R(1 - 1/\sqrt{2}) \approx 0.293R`. More robust to boundary raggedness "
        "than MeanBoundaryDist. See InscribedRadius and RobustMeanRadius for the "
        "colony's actual radial extent.",
        tier=1,
    )
    MEAN_BOUNDARY_DIST = Entry(
        "MeanBoundaryDist",
        "Mean Euclidean distance from each colony pixel to the nearest background "
        "pixel, computed on the object in isolation. This is a measure of interior "
        "thickness, not a radius: for an ideal disk of radius R it equals "
        r":math:`R/3`. High values relative to InscribedRadius indicate a compact, "
        "convex colony; low values indicate a thin or filamentous one.",
        tier=1,
    )
    INSCRIBED_RADIUS = Entry(
        "InscribedRadius",
        "Radius of the largest circle that fits entirely inside the colony, equal to "
        "the maximum of the object's Euclidean distance transform. Attained at the "
        "colony's distance-transform peak, which is the center used for "
        "RobustMeanRadius and ReachRadius. For an ideal disk it equals the disk "
        "radius. Formerly reported under the name MaxRadius.",
        tier=1,
    )
```

- [ ] **Step 4: Update the three consumers**

In `src/phenotypic/measure/_measure_shape.py`, the `_measure_radial_profile` return dict becomes:

```python
        return {
            str(SHAPE.MEAN_BOUNDARY_DIST): float(interior.mean()),
            str(SHAPE.MEDIAN_BOUNDARY_DIST): float(np.median(interior)),
            str(SHAPE.INSCRIBED_RADIUS): float(edt.max()),
        }
```

In the `MeasureShape` class docstring, replace the line

```
            - MeanRadius, MedianRadius, MaxRadius (distance-transform
              based).
```

with

```
            - MeanBoundaryDist, MedianBoundaryDist (mean/median depth from
              the boundary; not radii).
            - InscribedRadius (largest inscribed circle).
```

In `tests/unit/schema/test_classification.py`, `test_shape_straddles_tier1_and_tier2` (line ~124), replace the `tier1` set with:

```python
    tier1 = {SHAPE.AREA, SHAPE.CONVEX_AREA, SHAPE.MEDIAN_BOUNDARY_DIST,
             SHAPE.MEAN_BOUNDARY_DIST, SHAPE.INSCRIBED_RADIUS,
             SHAPE.MIN_FERET_DIAMETER, SHAPE.MAX_FERET_DIAMETER,
             SHAPE.MAJOR_AXIS_LENGTH, SHAPE.MINOR_AXIS_LENGTH, SHAPE.BBOX_AREA,
             SHAPE.PERIMETER}
```

In `tests/unit/measure/test_measure_shape.py`, update `test_touching_labels_do_not_inflate_each_others_radii` to read `Shape_InscribedRadius`, `Shape_MeanBoundaryDist`, and `Shape_MedianBoundaryDist` instead of the old names. The expected values are unchanged (10.0 / 11.0, 4.6951 / 4.8676, 4.0).

- [ ] **Step 5: Run the full schema and measure suites**

```bash
uv run pytest tests/unit/schema tests/unit/measure -q
```

Expected: all PASS. `test_shape_straddles_tier1_and_tier2` asserts `tier1 | tier2 == set(SHAPE)`, so a missed rename fails loudly here.

- [ ] **Step 6: Lint, typecheck, commit**

```bash
uv run ruff check --fix
uv run mypy src/phenotypic
git add src/phenotypic/schema/_shape.py src/phenotypic/measure/_measure_shape.py tests/unit/schema/test_classification.py tests/unit/measure/test_measure_shape.py
git commit -m "refactor(schema)!: rename Shape_{Mean,Median}Radius and Shape_MaxRadius

BREAKING CHANGE: Shape_MeanRadius -> Shape_MeanBoundaryDist,
Shape_MedianRadius -> Shape_MedianBoundaryDist, Shape_MaxRadius ->
Shape_InscribedRadius. The first two were means of the distance transform
(depth from the boundary, R/3 and 0.293R on a disk), never radii. No
column name is reused with new semantics."
```

---

### Task 4: Add `RobustMeanRadius` and `ReachRadius`

**Files:**
- Modify: `src/phenotypic/schema/_shape.py` (append 2 members after `INSCRIBED_RADIUS`)
- Modify: `src/phenotypic/measure/_measure_shape.py` (3 pydantic fields, 1 new method, extend `_measure_radial_profile`)
- Modify: `tests/unit/schema/test_classification.py` (`tier1` set)
- Create: `tests/unit/measure/test_radial_profile.py`

**Interfaces:**
- Consumes: `SHAPE.INSCRIBED_RADIUS` and `_measure_radial_profile` from Task 3.
- Produces: `MeasureShape._trace_radial_signature(self, obj_mask: np.ndarray, edt: np.ndarray) -> np.ndarray | None`, returning a length-`self.angular_bins` array of radii, or `None` when no contour exists. `_measure_radial_profile` gains keys `Shape_RobustMeanRadius` and `Shape_ReachRadius`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/measure/test_radial_profile.py`:

```python
"""Unit tests for MeasureShape's radial-signature geometry.

Every expected value here is checked against an analytic shape. Tolerances are
derived from two error sources:

* Angular binning: taking the outermost contour crossing per bin overestimates
  by at most about (1/2) * r * dtheta^2. At K=360, dtheta = 2*pi/360 = 0.01745,
  so for r ~ 40 that is under 0.01 px.
* The marching-squares contour sits on the 0.5 iso-level, which displaces the
  boundary by up to half a pixel.

The half-pixel term dominates, so 0.6 px is the working tolerance: roughly
1.2x the dominant error. It is tight enough to catch the failure this code
exists to prevent -- boundary-pixel sampling of a colony with a runner is off
by 2.03 px -- which `test_boundary_pixel_sampling_would_break_down` proves.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import trim_mean

from phenotypic.measure import MeasureShape

TOL = 0.6  # pixels; see module docstring


def _crop(mask: np.ndarray) -> np.ndarray:
    """Crop a mask to its bounding box, mimicking regionprops.image."""
    return mask[np.ix_(mask.any(axis=1), mask.any(axis=0))]


def _disk(radius: int, half: int = 130) -> np.ndarray:
    y, x = np.mgrid[-half:half, -half:half]
    return x**2 + y**2 <= radius**2


def _disk_with_runner() -> np.ndarray:
    """Radius-40 colony with a half-width-3 runner reaching to r = 90.

    The runner subtends 2*arctan(3/40) = 8.6 degrees, i.e. 2.4% of all
    directions, but it is roughly 2 * 50 = 100 pixels of boundary arc length.
    """
    y, x = np.mgrid[-130:130, -130:130]
    return (x**2 + y**2 <= 40**2) | ((np.abs(y) <= 3) & (x >= 0) & (x <= 90))


def test_signature_of_a_disk_is_its_radius():
    op = MeasureShape()
    mask = _crop(_disk(40))
    profile = op._measure_radial_profile(mask)

    assert profile["Shape_InscribedRadius"] == pytest.approx(40.0, abs=TOL)
    assert profile["Shape_RobustMeanRadius"] == pytest.approx(40.0, abs=TOL)
    # ReachRadius lands on the 0.5 iso-level, half a pixel outside the disk.
    assert profile["Shape_ReachRadius"] == pytest.approx(40.5, abs=TOL)


def test_mean_boundary_dist_of_a_disk_is_one_third_of_its_radius():
    """Pins the interpretation the rename encodes: this column is not a radius."""
    profile = MeasureShape()._measure_radial_profile(_crop(_disk(40)))
    # Analytic: mean EDT over a disk of radius R is R/3. Discretisation of the
    # boundary lifts it slightly, so allow 0.3 px on a predicted 13.33.
    assert profile["Shape_MeanBoundaryDist"] == pytest.approx(40.0 / 3.0, abs=0.3)


def test_angular_sampling_matches_the_analytic_mean_radius_of_an_ellipse():
    """The mean of r(theta) over an ellipse, computed analytically, is 39.2188.

    This is deliberately not the arc-length mean (45.51) nor the
    equivalent-circle radius (41.83). Getting 39.22 proves the signature is
    sampled uniformly in angle.
    """
    op = MeasureShape()
    a, b = 70.0, 25.0
    y, x = np.mgrid[-200:200, -200:200]
    mask = _crop((x / a) ** 2 + (y / b) ** 2 <= 1)

    edt = np.pad(mask, 1)
    from scipy.ndimage import distance_transform_edt

    edt = distance_transform_edt(edt)[1:-1, 1:-1]
    signature = op._trace_radial_signature(mask, edt)

    theta = np.linspace(0, 2 * np.pi, 200_001)[:-1]
    analytic = (a * b / np.hypot(b * np.cos(theta), a * np.sin(theta))).mean()

    assert analytic == pytest.approx(39.2188, abs=1e-3)  # guard the guard
    assert signature.mean() == pytest.approx(analytic, abs=0.25)


def test_runner_does_not_break_down_the_robust_mean():
    """A 20% trimmed mean tolerates up to 20% contamination. Under angular
    sampling the runner occupies its 2.4% angular width, so the estimate holds.
    """
    op = MeasureShape()
    profile = op._measure_radial_profile(_crop(_disk_with_runner()))

    assert profile["Shape_RobustMeanRadius"] == pytest.approx(40.0, abs=TOL)
    assert profile["Shape_ReachRadius"] == pytest.approx(90.5, abs=TOL)
    assert profile["Shape_InscribedRadius"] == pytest.approx(40.0, abs=TOL)


def test_boundary_pixel_sampling_would_break_down():
    """Mutation control. Proves the tolerance in the test above does real work.

    Sample the same colony's boundary by pixel instead of by angle. The runner
    then supplies ~30% of the samples, exceeding the trimmed mean's 20%
    breakdown point, and the estimate lands 2.03 px away from the truth --
    well outside TOL. If this ever falls inside TOL, the test above has
    stopped discriminating.
    """
    from skimage.segmentation import find_boundaries

    mask = _crop(_disk_with_runner())
    op = MeasureShape()
    edt = np.pad(mask, 1)
    from scipy.ndimage import distance_transform_edt

    edt = distance_transform_edt(edt)[1:-1, 1:-1]
    center = np.argwhere(edt >= 0.99 * edt.max()).mean(axis=0)

    pixels = np.argwhere(find_boundaries(mask, mode="inner")).astype(float)
    radii = np.hypot(pixels[:, 0] - center[0], pixels[:, 1] - center[1])

    contamination = float((radii > 45).mean())
    assert contamination > 0.20, "runner must exceed the 20% breakdown point"
    assert abs(trim_mean(radii, 0.2) - 40.0) > TOL

    # And the angular scheme keeps the runner under the breakdown point.
    signature = op._trace_radial_signature(mask, edt)
    assert float((signature > 45).mean()) < 0.05


@pytest.mark.parametrize(
    "mask",
    [
        np.ones((1, 1), dtype=bool),
        np.ones((2, 2), dtype=bool),
        np.ones((1, 5), dtype=bool),
    ],
    ids=["single-pixel", "2x2", "thin-line"],
)
def test_degenerate_objects_do_not_raise(mask):
    """Specks survive detection. They must not crash or warn the measurer."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        profile = MeasureShape()._measure_radial_profile(mask)

    assert np.isfinite(profile["Shape_InscribedRadius"])
    assert np.isfinite(profile["Shape_RobustMeanRadius"])
```

- [ ] **Step 2: Run them to verify they fail**

```bash
uv run pytest tests/unit/measure/test_radial_profile.py -v
```

Expected: FAIL. `KeyError: 'Shape_RobustMeanRadius'` and `AttributeError: 'MeasureShape' object has no attribute '_trace_radial_signature'`.

- [ ] **Step 3: Add the two schema members**

In `src/phenotypic/schema/_shape.py`, immediately after `INSCRIBED_RADIUS`:

```python
    ROBUST_MEAN_RADIUS = Entry(
        "RobustMeanRadius",
        "Symmetrically trimmed mean of the colony's radial signature: the distance "
        "from the colony center to its boundary, resampled uniformly over a fixed "
        "number of directions. The center is the centroid of the distance-transform "
        "peak plateau, which always lies inside the colony. Because the signature is "
        "sampled by angle rather than along the boundary, a narrow protrusion "
        "contributes only its angular width, so a single runner or spur cannot "
        "dominate the estimate. For an ideal disk this equals the disk radius. "
        "Compare InscribedRadius (the smallest radius) and ReachRadius (the largest).",
        tier=1,
    )
    REACH_RADIUS = Entry(
        "ReachRadius",
        "Maximum of the colony's radial signature: the distance from the colony "
        "center to the furthest point on its boundary. A ReachRadius much larger "
        "than RobustMeanRadius indicates a protrusion, spur, or runner extending "
        "from an otherwise compact colony.",
        tier=1,
    )
```

- [ ] **Step 4: Update the classification test**

In `tests/unit/schema/test_classification.py`, add the two new members to `tier1`:

```python
    tier1 = {SHAPE.AREA, SHAPE.CONVEX_AREA, SHAPE.MEDIAN_BOUNDARY_DIST,
             SHAPE.MEAN_BOUNDARY_DIST, SHAPE.INSCRIBED_RADIUS,
             SHAPE.ROBUST_MEAN_RADIUS, SHAPE.REACH_RADIUS,
             SHAPE.MIN_FERET_DIAMETER, SHAPE.MAX_FERET_DIAMETER,
             SHAPE.MAJOR_AXIS_LENGTH, SHAPE.MINOR_AXIS_LENGTH, SHAPE.BBOX_AREA,
             SHAPE.PERIMETER}
```

- [ ] **Step 5: Add the pydantic fields**

In `src/phenotypic/measure/_measure_shape.py`, add these imports at the top:

```python
from pydantic import Field
from scipy.ndimage import label as ndi_label
from scipy.stats import trim_mean
from skimage.measure import find_contours
```

and declare the fields on `MeasureShape`, immediately after `_measurement_infoclass`:

```python
    angular_bins: int = Field(360, ge=8, le=3600)
    trim_proportion: float = Field(0.2, ge=0.0, lt=0.5)
    plateau_tolerance: float = Field(0.01, gt=0.0, lt=1.0)
```

Do **not** pass `description=` to `Field`. `BaseOperation.__init_subclass__` calls
`apply_docstring_descriptions(cls)`, which copies each parameter's description from the
Google-style `Args:` block onto the field. The docstring below is the single source.
This matches `detect/_watershed_detector.py:115-117`.

Add the corresponding `Args:` block to the class docstring, above `Returns:`:

```
    Args:
        angular_bins: Number of equally spaced directions at which the
            radial signature is sampled. Higher values reduce the
            outermost-crossing bias but leave more empty bins to
            interpolate across on small colonies. Typical range: 180--720.
            Default: 360.
        trim_proportion: Fraction of the radial signature trimmed from each
            tail before averaging, which sets the breakdown point of
            RobustMeanRadius. At 0.2 the estimate tolerates up to 20% of
            directions being contaminated by protrusions. Setting it to 0.0
            gives the plain arithmetic mean. Typical range: 0.0--0.3.
            Default: 0.2.
        plateau_tolerance: Relative tolerance defining the near-maximal
            plateau of the distance transform. Its centroid is the colony
            center. Values near 0 make the center an argmax, whose location
            is arbitrary among exact ties; the default averages over the
            plateau instead. Default: 0.01.
```

- [ ] **Step 6: Implement `_trace_radial_signature`**

Add to `MeasureShape`, immediately before `_measure_radial_profile`:

```python
    def _trace_radial_signature(
            self, obj_mask: np.ndarray, edt: np.ndarray
    ) -> np.ndarray | None:
        """Sample the colony boundary's distance from its center, uniformly by angle.

        The center is the centroid of the distance-transform's near-maximal
        plateau, not its argmax: the transform's values are square roots of
        exact integers, so exact ties are common and an argmax would resolve
        them by raster order. The boundary is the subpixel marching-squares
        contour at the 0.5 iso-level. Sampling is by angle rather than along
        the contour so that a narrow protrusion contributes only its angular
        width, which is what keeps the trimmed mean inside its breakdown point.

        Args:
            obj_mask (np.ndarray): Boolean mask of a single object within its
                bounding box (``regionprops.image``).
            edt (np.ndarray): Euclidean distance transform of *obj_mask*,
                computed with one pixel of background padding on every side.

        Returns:
            np.ndarray | None: Radii at ``self.angular_bins`` equally spaced
            angles, or None when the object has no interior or no contour.
        """
        peak = float(edt.max())
        if peak <= 0.0:
            return None

        plateau = edt >= (1.0 - self.plateau_tolerance) * peak
        components, _ = ndi_label(plateau)
        dominant = components[np.unravel_index(np.argmax(edt), edt.shape)]
        center = np.argwhere(components == dominant).mean(axis=0)

        contours = find_contours(np.pad(obj_mask, 1).astype(float), 0.5)
        if not contours:
            return None
        outline = max(contours, key=len) - 1.0

        offsets = outline - center
        radii = np.hypot(offsets[:, 0], offsets[:, 1])
        angles = np.arctan2(offsets[:, 0], offsets[:, 1])

        n_bins = self.angular_bins
        bins = ((angles + np.pi) / (2.0 * np.pi) * n_bins).astype(int) % n_bins
        signature = np.full(n_bins, -np.inf)
        np.maximum.at(signature, bins, radii)

        empty = np.isinf(signature)
        if empty.all():
            return None
        signature[empty] = np.nan
        if empty.any():
            # Circular interpolation: bins with no contour vertex are not
            # missing at random. They cluster where the contour is angularly
            # sparse, so dropping them biases the mean upward.
            index = np.arange(n_bins)
            filled = ~empty
            signature[empty] = np.interp(
                    index[empty],
                    np.concatenate([index[filled] - n_bins, index[filled], index[filled] + n_bins]),
                    np.tile(signature[filled], 3),
            )
        return signature
```

- [ ] **Step 7: Extend `_measure_radial_profile`**

Replace its body with:

```python
        # asarray narrows the scipy stub's tuple-or-ndarray return union; with
        # return_indices=False the call yields an ndarray and copies nothing.
        edt = np.asarray(distance_transform_edt(np.pad(obj_mask, 1)))[1:-1, 1:-1]
        interior = edt[obj_mask]
        values = {
            str(SHAPE.MEAN_BOUNDARY_DIST): float(interior.mean()),
            str(SHAPE.MEDIAN_BOUNDARY_DIST): float(np.median(interior)),
            str(SHAPE.INSCRIBED_RADIUS): float(edt.max()),
            str(SHAPE.ROBUST_MEAN_RADIUS): np.nan,
            str(SHAPE.REACH_RADIUS): np.nan,
        }

        signature = self._trace_radial_signature(obj_mask, edt)
        if signature is not None:
            values[str(SHAPE.ROBUST_MEAN_RADIUS)] = float(
                    trim_mean(signature, self.trim_proportion)
            )
            values[str(SHAPE.REACH_RADIUS)] = float(signature.max())
        return values
```

and extend its `Returns:` docstring to name all five headers.

- [ ] **Step 8: Update the class docstring's `Returns:` block**

Replace the radius lines with:

```
            - MeanBoundaryDist, MedianBoundaryDist (mean/median depth from
              the boundary; not radii).
            - InscribedRadius, RobustMeanRadius, ReachRadius (smallest,
              typical, and largest radius from the colony center).
```

- [ ] **Step 9: Run all the tests**

```bash
uv run pytest tests/unit/measure tests/unit/schema -q
```

Expected: all PASS, including `test_boundary_pixel_sampling_would_break_down`.

- [ ] **Step 10: Eyeball the new columns on real colonies**

```bash
uv run python -c "
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureShape
df = MeasureShape().measure(OtsuDetector().apply(load_synth_yeast_plate()))
big = df[df['Shape_Area'] > 200]
cols = ['Shape_InscribedRadius','Shape_RobustMeanRadius','Shape_ReachRadius','Shape_MeanBoundaryDist']
print(big[cols].describe().loc[['mean','min','max']].round(3).to_string())
ok = (big['Shape_InscribedRadius'] <= big['Shape_RobustMeanRadius'] + 1.0).all()
ok &= (big['Shape_RobustMeanRadius'] <= big['Shape_ReachRadius'] + 1e-9).all()
print('ordering inscribed <= robust <= reach:', ok)
assert ok
"
```

Expected: the ordering invariant holds (the `+1.0` slack absorbs the half-pixel iso-level offset plus the discrete inscribed radius), and `Shape_MeanBoundaryDist` sits near one third of `Shape_InscribedRadius`.

- [ ] **Step 11: Lint, typecheck, commit**

```bash
uv run ruff check --fix
uv run mypy src/phenotypic
git add src/phenotypic/schema/_shape.py src/phenotypic/measure/_measure_shape.py tests/unit/schema/test_classification.py tests/unit/measure/test_radial_profile.py
git commit -m "feat(measure): add Shape_RobustMeanRadius and Shape_ReachRadius

Trimmed mean and maximum of a radial signature sampled uniformly in angle
about the distance-transform peak. Angular sampling gives a runner only its
angular width (2.4% for a typical spur) rather than its arc length (30%),
keeping the trimmed mean inside its 20% breakdown point."
```

---

### Task 5: Documentation and full regression

**Files:**
- Modify: `src/phenotypic/schema/CLAUDE.md`

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces: nothing.

- [ ] **Step 1: Update the straddler example in the schema guide**

In `src/phenotypic/schema/CLAUDE.md`, the "Measurement classification" section reads:

```
its size-magnitude members carry
`Entry(..., tier=1)` (e.g. `AREA`, `PERIMETER`, radii, Feret diameters)
```

Replace `radii` with `boundary distances, radii`:

```
its size-magnitude members carry
`Entry(..., tier=1)` (e.g. `AREA`, `PERIMETER`, boundary distances, radii,
Feret diameters)
```

- [ ] **Step 2: Confirm no stale references survive**

Use `git grep`, not `grep -r`. Much of `docs/` is generated, untracked build output
(`docs/build/`, `docs/source/measurements_ref/`, and the `_static/gui_images/_dataset/`
sample run) which still carries the old column names and is regenerated on the next build.
Only tracked files matter. Also anchor on the old *member* names, since the substring
`MeanRadius` legitimately appears inside the new `RobustMeanRadius`.

```bash
git grep -n "MEAN_RADIUS\|MEDIAN_RADIUS\|MAX_RADIUS" -- src tests docs/source
git grep -n "Shape_MeanRadius\|Shape_MedianRadius\|Shape_MaxRadius" -- src tests docs/source
```

Expected from the first: only `src/phenotypic/schema/_radial_expansion.py` (a different
enum, `RADIAL_EXPANSION`, whose `MEAN_RADIUS` / `MEDIAN_RADIUS` refer to branch path
lengths and are not affected). Expected from the second: only
`tests/unit/measure/test_measure_shape.py`, in the guard test that asserts those columns
are *absent*. No hits under `src/phenotypic/measure/` or `src/phenotypic/schema/_shape.py`.

- [ ] **Step 3: Run the affected test suites**

```bash
uv run pytest tests/unit/measure tests/unit/schema tests/unit/util/test_measurement_outputs.py tests/unit/post tests/smoke/test_serialization.py -q
```

Expected: all PASS. `test_serialization.py` exercises `MeasureShape.to_json()` / `from_json()`; the three new fields all have defaults, so round-tripping is unaffected.

- [ ] **Step 4: Build the docs and check the measurement table renders**

```bash
uv run make -C docs html 2>&1 | tail -20
```

Expected: build succeeds. The Measurements reference picks up the five columns automatically via `SHAPE.rst_table()`; no manual doc edit is needed.

- [ ] **Step 5: Full typecheck and lint**

```bash
uv run ruff check --fix
uv run mypy src/phenotypic
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/schema/CLAUDE.md
git commit -m "docs(schema): note boundary-distance members in the straddler example"
```

---

---

### Task 6: Recapture the `measure.MeasureShape` migration golden

Added during execution. The Task-3 Seam gate found `tests/migration/_goldens/measure.MeasureShape.parquet`
(552x18) carries `Shape_MedianRadius` / `Shape_MeanRadius` / `Shape_MaxRadius` and
`ConvexArea` / `Solidity` values computed with the perimeter bug. `tests/migration/test_equivalence.py`
compares via `assert_frame_equal(check_dtype=True)`, so it will fail once it runs.

It does **not** run on macOS: `_GOLDEN_PLATFORM = "linux"` (`test_equivalence.py:78`) and
`test_operation_matches_golden` calls `pytest.skip()` before any comparison when
`sys.platform != "linux"`. The stale golden is therefore invisible locally and fails only
on Linux CI.

**Do NOT run `scripts/capture_migration_goldens.py`.** It takes no arguments, recaptures
the frozen inputs, and rewrites **all 142** scenario goldens. Its own docstring says it
"must be run before any operation class is migrated" — it is a one-shot bootstrap, not a
refresh tool. Running it here would replace 141 unrelated Linux-captured goldens with
macOS floats against a `_FLOAT_RTOL = 1e-6` comparison.

**Files:**
- Modify: `tests/migration/_goldens/measure.MeasureShape.parquet` (binary, regenerated)

- [ ] **Step 1: Confirm the golden is stale and the suite skips locally**

```bash
uv run pytest tests/migration -k "MeasureShape" -q
```

Expected: `1 skipped` (platform gate), NOT a pass. Then:

```bash
uv run python -c "
import pandas as pd
df = pd.read_parquet('tests/migration/_goldens/measure.MeasureShape.parquet')
print(sorted(c for c in df.columns if 'Radius' in c))
"
```

Expected: `['Shape_MaxRadius', 'Shape_MeanRadius', 'Shape_MedianRadius']`.

- [ ] **Step 2: Recapture that one scenario only**

```bash
uv run python -c "
import sys; sys.path.insert(0, '.')
from tests.migration._scenarios import build_scenarios
from tests.migration._runner import run_scenario, golden_path

target = next(s for s in build_scenarios() if s.scenario_id == 'measure.MeasureShape')
result = run_scenario(target)
result.save(golden_path(target))
print('rewrote', golden_path(target))
"
```

- [ ] **Step 3: Verify the new golden's schema**

```bash
uv run python -c "
import pandas as pd
df = pd.read_parquet('tests/migration/_goldens/measure.MeasureShape.parquet')
cols = list(df.columns)
assert 'Shape_MeanRadius' not in cols and 'Shape_MaxRadius' not in cols
for c in ('Shape_MeanBoundaryDist','Shape_MedianBoundaryDist','Shape_InscribedRadius',
          'Shape_RobustMeanRadius','Shape_ReachRadius'):
    assert c in cols, c
assert (df['Shape_Solidity'] <= 1.0).all()
print('rows', len(df), 'cols', len(cols), 'OK')
"
```

Expected: `rows 552 cols 20 OK`.

- [ ] **Step 4: Commit**

```bash
git add tests/migration/_goldens/measure.MeasureShape.parquet
git commit -m "test(migration): recapture the MeasureShape golden

The golden held Shape_{Mean,Median}Radius/Shape_MaxRadius and ConvexArea/
Solidity values from before the hull-perimeter and per-object-EDT fixes.
Recaptured for measure.MeasureShape only; the capture script rewrites all
142 goldens and must not be run here."
```

**Known limitation, stated plainly.** This golden is captured on macOS, but the suite only
compares it on Linux (`_GOLDEN_PLATFORM`). Its float values are therefore unverified on
the one platform that checks them. If Linux CI reports a mismatch beyond `rtol=1e-6`, the
golden must be recaptured there. The column *names* and the `Solidity <= 1` invariant are
platform-independent and are verified above.

---

## Out-of-plan findings from execution

Recorded so the next reader does not rediscover them.

1. **`scripts/capture_migration_goldens.py` is a bootstrap, not a refresh tool.** See Task 6.
2. **The shipped sample CSVs cannot be regenerated.** `src/phenotypic/data/meas/all_meas.csv`
   and `area_meas.csv` reference 96 source images across 5 timepoints (dataset `S 30C`) that
   do not ship with the package (11 image files total, none matching). Their
   `Shape_MeanRadius` / `Shape_MedianRadius` headers were renamed in place (the values are
   mean/median EDT, exactly what the new names denote, so no value changed). Their
   `Shape_ConvexArea` / `Shape_Solidity` values remain wrong and cannot be fixed without the
   original images — tracked separately.
3. **`np.asarray` is required around `distance_transform_edt`** before subscripting, or mypy
   reports a `call-overload` error from the scipy stub's tuple-or-ndarray return union.
4. **`_measure_shape.py` has 4 pre-existing mypy errors** (`_operate` override, an
   assignment, a dict `.insert`, a return-value). The "25 errors in 6 files" that mypy
   prints for this file mostly come from imported modules. Do not use the larger number to
   justify adding a new one.

## Self-Review

**Spec coverage.** Every item from the discussion maps to a task: ConvexHull bug → Task 1; per-object EDT → Task 2; the two renames plus `MaxRadius` → `InscribedRadius` → Task 3; `RobustMeanRadius` and `ReachRadius` via angular signature → Task 4; docs → Task 5. The two QC columns were dropped by decision and are documented as such under "Naming decision".

**Type consistency.** `_measure_radial_profile(obj_mask) -> dict[str, float]` is introduced in Task 2 with three keys, re-keyed in Task 3, and extended to five keys in Task 4. `_trace_radial_signature(obj_mask, edt) -> np.ndarray | None` appears only in Task 4. Both take `regionprops.image`. No other name is used for either.

**Numbers.** Every expected value in this plan was measured against the working tree before it was written: 118.0 / 6.9492 (buggy `ConvexArea` / `Solidity`), 820.0 / 1.0 (fixed), 20.0 / 21.0 (merged EDT) vs 10.0 / 11.0 (per-object), 4.6951 / 4.8676 / 4.0 (boundary distances), 40.0125 / 40.0069 / 40.5000 (disk inscribed / robust / reach), 40.0292 / 90.5497 (runner robust / reach), 39.2187 vs analytic 39.2188 (ellipse angular mean), 42.0285 (boundary-pixel mutation), 0.3321 (real-colony `MeanBoundaryDist` / `InscribedRadius` ratio).

**Known limitation, not addressed here.** The radial signature assumes the colony is star-shaped about its distance-transform peak. It is not, for crescents, rings, or merged doublets, and in those cases it silently reports the outer envelope rather than raising. `refine.SeparateObjects` is the intended remedy for merged colonies. If a per-object diagnostic is wanted later, the h-maxima peak count (`h_maxima(edt, 0.10 * edt.max())` then count components) separates 1 / 1 / 1 for disk, ellipse, and runner from 2 / 2 / 3 for the two doublets and the crescent, and is stable for prominence between 0.05 and 0.30.
