# Size Measures Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `MeasureSize` the single emitter of colony size magnitudes (area, perimeter, hull and box areas, ellipse axes, and a five-member radius family), leaving `MeasureShape` with form descriptors only. Release this as 0.20.0 with highlighted change notes.

**Architecture:** Two correctness rules move into one private module, `measure/_object_geometry.py`: convex area is `ConvexHull.volume`, and each EDT runs on a padded single-object crop. Every measurer reads the cached `image.objects.props` and no measurer runs another measurer. The work follows decouple-then-flip. First add everything to `SIZE`/`MeasureSize` and cut `MeasureIntensity`/`KeepSectionLargest` loose from other measurers; only then remove the moved members from `SHAPE`. A `MeasurementInfo.change_note()` hook is the single source of the 0.20.0 note, rendered in three doc surfaces.

**Tech Stack:** Python 3.12, pydantic v2, numpy, scipy (`ndimage`, `spatial`, `stats`), scikit-image (`measure.find_contours`, `regionprops`), pandas, pytest, uv, Sphinx (pydata theme).

**Spec:** `docs/superpowers/specs/2026-09-24-size-measures-consolidation/design.md`. Read it before starting any task. The logic-validation script `docs/superpowers/logic_validation_scripts/2026-09-24-size-measures-consolidation/radial_invariants.py` re-derives every geometric number quoted below.

**Port source:** branch `shape-radial-measures` (tip `5cad1dfa5`). Read files from it with `git show shape-radial-measures:<path>`. **Do not merge, rebase or cherry-pick it.**

## Global Constraints

- `uv` is the sole runner. Never invoke bare `python` or `pip`.
- Operations are pydantic v2 models: keyword-only construction, annotated class-level fields, **no `__init__`**. Guards go in `Field(...)` bounds.
- Google-style docstrings. A `MeasureFeatures` docstring gives the **parameters and a high-level overview**; per-column detail lives in the `MeasurementInfo` `Entry` `desc`.
- **Never author or edit `bio_desc`**, and leave `image` unset on new members. The one exception is fixed by the spec: `SHAPE.AREA`'s existing `bio_desc` and `image="shape/area.png"` move **verbatim** onto `SIZE.AREA`.
- Convex area taken from scipy is **`ConvexHull.volume`**, never `.area` (which is a perimeter in 2-D) and never `regionprops.area_convex`.
- **Hard break:** no aliases for retired columns. There is no legacy-header machinery for measurement columns, and none is to be added.
- Version becomes **`0.20.0`** in `src/phenotypic/__init__.py`.
- Change notes use **`.. versionchanged:: 0.20.0`** and never go into member `desc` (descs are published into every run's README).
- No `TuneSpec` annotations on the new `MeasureSize` fields: `measure/` is outside the tune annotation gate.
- `uv run ruff check --fix <explicit paths you changed>`. **Never bare `ruff check --fix`.**
- Vendored `docs/superpowers/**/refs/` sources are read-only.
- A check that cannot run must **fail**, not skip. Derive every numeric tolerance from a stated mechanism. Prove each new guard can fail, by the mutation listed in its step.
- Test runs: per step, the step's own test; per task, the touched test files; per phase gate, the affected surface once; at the end, one full sharded regression via the `run-phenotypic-test` skill. GUI tests need `QT_QPA_PLATFORM=offscreen`.

## Amendments from the pre-dispatch plan review (2026-09-24) — binding

Source: `docs/superpowers/reports/2026-09-24-size-measures-consolidation/plan-review.md`
(0 critical, 5 high, 8 medium, 9 low). Where a task below conflicts with this list, **this
list wins**. Code blocks below have already been corrected inline where marked.

- **A1 Port source by SHA.** Read the branch as `git show 5cad1dfa5:<path>`. Preflight:
  `git cat-file -e 5cad1dfa5^{commit}` must succeed, or stop.
- **A2 Zero objects (user decision).** Keep main's contract: `MeasureSize`/`MeasureShape`
  raise `OperationFailedError` wrapping `NoObjectsError`. The tests pin the raise (corrected
  inline in Tasks 3 and 5). No early return is added.
- **A3 Spec wording (user decision).** InscribedRadius is the minimum only up to half a
  pixel, and the centre can fall in the hole of a ring-shaped colony (spec §4.1). The
  `MEDIAN_RADIUS` desc is corrected inline. Assert radius ordering only on the crescent,
  never on a disk or a degenerate shape.
- **A4 No "Formerly reported as …" in any `desc`** (spec §7). Removed inline from
  `INSCRIBED_RADIUS` and both BoundaryDist Entries.
- **A5 The consumer flip happens before the Shape flip.** Task 7 Steps 3-4 (prefabs gain
  `MeasureSize()`, GUI analysis defaults become `str(SIZE.AREA)`), with their two new tests,
  run as **Task 4b**, right after Task 4 and before Task 5. `Size_Area` exists from Task 3
  on. Do not push between Task 5 and Task 8.
- **A6 Differential proof, not golden proof (user decision).** Task 4b also creates
  `docs/superpowers/plans/2026-09-24-size-measures-consolidation/diff_migration_scenarios.py`,
  per plan-review H4 amendment 1. It drives shipped code, so it lives beside the plan, not
  in `logic_validation_scripts/`. The orchestrator runs it after Task 4b and in Task 10.
  Task 10 then recaptures **all four** goldens (`measure.MeasureShape`, `measure.MeasureSize`,
  `measure.MeasureIntensity`, `refine.KeepSectionLargest`). The commit message names each
  pre-existing drift (below) separately from this change's column moves. Pre-change status
  on unmodified `src/`: Size PASS; Shape FAIL (ConvexArea/Solidity, 99/552 rows, golden
  predates `.volume` fix `d846ca4a`); Intensity FAIL (float32 dtype plus ConvexDensity, same
  cause); KeepSectionLargest FAIL (94 labels vs the golden's 96, golden predates grid fix
  `74401bbd`). "Byte-unchanged" is dropped as a criterion.
- **A7 Rename pipeline hygiene.** The rename script is committed as
  `docs/superpowers/plans/2026-09-24-size-measures-consolidation/size_rename.pl`, never
  `/tmp`. Target lists go to the session scratchpad. Every target grep is
  `grep -rlIE --exclude-dir=__pycache__ …` (`-I` skips binaries). Additional exclusions:
  `src/phenotypic/schema/_measurement_info.py` (its SHAPE is a local toy class in a
  doctest) and `tests/unit/gui/results_viewer/test_measurement_join_migration_run.py` (it
  reads a real run whose stores carry old names). The rule is: **never rename a column name
  that a test reads from historical data on disk.** Fix `src/phenotypic/schema/CLAUDE.md:178`
  by hand to `['Shape_Circularity', 'Shape_Compactness', ...]`.
- **A8 HPCC rules.**
  - No `git stash`. Compare against main in a detached worktree at `81d19ec66` under
    `/bigdata/exfab/anguy344/gate-worktrees/`.
  - The docs build is a Slurm job (Task 6 Step 10, corrected inline).
  - Every multi-file pytest run takes `-o addopts= -m "not slow" -p no:randomly`, and phase
    gates are Slurm jobs submitted by the orchestrator.
- **A9 Wider test surfaces.**
  - Task 3 also runs `tests/gui/builder tests/unit/gui/builder`, since MeasureSize gains its
    first parameters.
  - Task 7 Step 4 also checks the `gui-tutorial-capture` ledgers against the MeasureSize form
    change.
  - Task 10 runs the swept e2e files explicitly (`PLAYWRIGHT=1`): `tests/e2e/gui/test_scatter_tab.py`
    and `tests/e2e/gui/test_analysis_app.py`.
- **A10 Small corrections.**
  - Task 5 Step 3: only the Feret diameters carry `tier=1`, and Circularity does not.
  - Task 4: the `KeepSectionLargest` docstring line "Measures the pixel area … via
    MeasureSize" becomes "counts each object's pixels".
  - Port the branch's `test_merged_edt_would_fail_this_test` (branch
    `tests/unit/measure/test_measure_shape.py:65-81`) into `test_measure_shape.py`, as a
    standing control.
  - Task 7 Step 1: the comment "`meas` returns a dict copy" becomes "`meas` is normalised to
    a dict".
  - Task 7 Step 6: the capture script binds `[str(SHAPE.SOLIDITY), str(SHAPE.CIRCULARITY)]`
    unconditionally, keeping the "stay inside one measurer" invariant. Check `WORKFLOWS.md`.
  - The PR notes that `RemoveByFeature(feature="MeasureSize")` now pays for EDTs and
    signatures.

## Execution clusters (orchestrator, 2026-09-24)

| Cluster | Tasks | Shape | Model | Gate after |
|---|---|---|---|---|
| — | T1 baseline | done `2004cfac` | — | — |
| C1 | T2 + T3 | Keystone | Opus | light |
| C2 | T4 + T4b (A5, A6 script) | Seam | Opus | light + differential run |
| C3 | T5 + T6 | Keystone | Opus | **phase 1 deep: implementation-test-reviewer over T2–T6** |
| C4 | T7 (minus Steps 3-4) | Sweep, consistency-critical | Opus | light |
| C5 ∥ C6 | T8 ∥ T9 (no shared files) | Sweep | Sonnet | **phase 2 deep: code review over T7–T9** |
| — | T10 | orchestrator | — | simplify pass, then full sharded regression |

## Review Focus

These inputs are implied by the spec but no other task exercises them. Each has a test in the owning task.

1. **A plate with zero detected colonies.** `MeasureSize` and `MeasureShape` raise `OperationFailedError` (wrapping `NoObjectsError`), like every other measurer (A2; tests in Tasks 3 and 5).
2. **A degenerate colony: a single pixel or a 1-pixel-wide line.** Qhull fails, so ConvexArea, Solidity and Feret are NaN. The radii stay finite and no warning escapes (Tasks 3 and 5).
3. **A colony cut off by the image border.** The padded crop counts the border as an edge: a 10-row full-width band on the top edge reports InscribedRadius 5, not the old 10 (Task 3).
4. **A concave (crescent) colony.** The five radii keep their ordering, InscribedRadius ≤ MedianRadius ≤ MaxRadius and InscribedRadius ≤ RobustMeanRadius ≤ MaxRadius, and nothing raises (Task 3).
5. **A saved pipeline carrying the new fields.** `MeasureSize(angular_bins=90, trim_proportion=0.1, plateau_tolerance=0.02)` must round-trip through `to_json`/`from_json` (Task 3).

---

## Dependency DAG

```
T1 baseline ──► T3 ──► T4 ──► T5 ──► T7 ──► T8 ──► T9 ──► T10
T2 helpers ──► T3        (T6 needs T5; T7, T8 and T9 need T6 for the note text)
                           T6 ──┘
```

Tasks are sequential in practice: T3–T8 share `schema/` and `measure/` files, so parallel worktrees would only produce merge conflicts. **T1 must run before any `src/` change**, because it captures main's behaviour.

---

### Task 1: Capture main's behaviour as a committed baseline

This golden is the evidence that moved columns kept their values, retained Shape columns did not change, and `MeasureIntensity` is byte-for-byte the same. It is captured from **unmodified** code.

**Files:**
- Create: `tests/unit/measure/_golden/size_consolidation_baseline.parquet`

**Interfaces:**
- Produces: a parquet with `Object_Label` plus every **main** `Shape_*` column and every `Intensity_*` column for the 96 colonies of `load_synth_yeast_plate()`. Tasks 3–5 read it.

- [ ] **Step 1: Confirm the tree is unmodified**

Run: `git status --porcelain src/`
Expected: no output. If `src/` has changes, stop; the baseline must come from main's code.

- [ ] **Step 2: Capture**

```bash
uv run python - <<'EOF'
from phenotypic.data import load_synth_yeast_plate
from phenotypic.measure import MeasureIntensity, MeasureShape
from phenotypic.schema import OBJECT

plate = load_synth_yeast_plate()
shape = MeasureShape().measure(plate)
intensity = MeasureIntensity().measure(plate)
frame = shape.merge(intensity, on=str(OBJECT.LABEL), validate="one_to_one")
assert len(frame) == 96, len(frame)
frame.to_parquet(
    "tests/unit/measure/_golden/size_consolidation_baseline.parquet", index=False
)
print(frame.shape)
print(sorted(frame.columns))
EOF
```

Expected: `(96, N)`, with a column list that contains `Shape_Area`, `Shape_MaxRadius`, `Shape_MeanRadius`, `Shape_MedianRadius`, `Shape_ConvexArea`, `Intensity_Density` and `Intensity_ConvexDensity`.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/measure/_golden/size_consolidation_baseline.parquet
git commit -m "test(measure): capture pre-consolidation Shape/Intensity baseline on the synth plate"
```

---

### Task 2: Shared per-object geometry helpers

**Files:**
- Create: `src/phenotypic/measure/_object_geometry.py`
- Test: `tests/unit/measure/test_object_geometry.py`

**Interfaces:**
- Produces:
  - `convex_hull_area(coords: np.ndarray) -> tuple[ConvexHull | None, float]` returns `(hull, hull.volume)`, or `(None, nan)` on `QhullError`.
  - `object_edt(obj_mask: np.ndarray) -> np.ndarray` returns the EDT of a boolean single-object crop, computed with one pixel of background padding on every side and returned at the crop's shape.

- [ ] **Step 1: Write the failing tests**

```python
"""Unit tests for the shared per-object geometry helpers."""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic.measure._object_geometry import convex_hull_area, object_edt


def _pixel_coords(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(mask).astype(float)


def test_convex_hull_area_is_the_hull_volume_not_its_perimeter():
    """A filled 10x20 pixel block has hull vertices at pixel centres, so the hull is a
    9 x 19 rectangle: area 171, perimeter 56. Qhull on integer coordinates is exact to
    a few ulp, so 1e-9 is ~1e5 ulp of headroom and still 115 below the perimeter.

    Mutation: return `hull.area` instead of `hull.volume` -> 56.0, and this fails.
    """
    hull, area = convex_hull_area(_pixel_coords(np.ones((10, 20), dtype=bool)))
    assert hull is not None
    assert area == pytest.approx(171.0, abs=1e-9)


@pytest.mark.parametrize(
    "mask",
    [np.ones((1, 1), dtype=bool), np.ones((1, 5), dtype=bool)],
    ids=["single-pixel", "collinear-line"],
)
def test_convex_hull_area_is_nan_when_qhull_cannot_build_a_hull(mask):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        hull, area = convex_hull_area(_pixel_coords(mask))
    assert hull is None
    assert np.isnan(area)


def test_object_edt_sees_background_on_every_side_of_the_crop():
    """A 3x3 crop fully set: without padding every pixel would be at infinite/undefined
    distance; with one pixel of padding the centre is 2 px from background.
    Integer geometry, exact.

    Mutation: drop the `np.pad` -> the centre is no longer 2.0, and this fails.
    """
    edt = object_edt(np.ones((3, 3), dtype=bool))
    assert edt.shape == (3, 3)
    assert edt[1, 1] == 2.0
    assert edt[0, 0] == 1.0


def test_object_edt_of_a_41x20_rectangle_has_inscribed_radius_10():
    edt = object_edt(np.ones((41, 20), dtype=bool))
    assert edt.max() == 10.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/measure/test_object_geometry.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'phenotypic.measure._object_geometry'`

- [ ] **Step 3: Implement**

```python
"""Per-object geometry shared by the measurers.

Each helper is the single home of a correctness rule that has been broken
before, so no measurer re-derives it inline:

* :func:`convex_hull_area` -- in 2-D, ``scipy.spatial.ConvexHull.volume`` is
  the hull's area and ``ConvexHull.area`` is its perimeter.
* :func:`object_edt` -- ``distance_transform_edt`` binarizes its input, so it
  must see one object's crop, never the labelled objmap; otherwise touching
  colonies see no background between them.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import ConvexHull, QhullError


def convex_hull_area(coords: np.ndarray) -> tuple[ConvexHull | None, float]:
    """Build an object's convex hull and return it with its area.

    Args:
        coords: ``(N, 2)`` pixel coordinates of one object
            (``regionprops.coords``).

    Returns:
        ``(hull, hull.volume)``, or ``(None, nan)`` when Qhull cannot build a
        hull (a single pixel or collinear pixels). ``volume`` is the area in
        2-D. The hull passes through pixel centres, so it is slightly smaller
        than the pixel count of the same shape.
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Qhull")
            hull = ConvexHull(coords)
    except QhullError:
        return None, float("nan")
    return hull, float(hull.volume)


def object_edt(obj_mask: np.ndarray) -> np.ndarray:
    """Euclidean distance transform of one object's crop, background-padded.

    Args:
        obj_mask: Boolean mask of a single object within its bounding box
            (``regionprops.image``), already isolated from neighbouring labels.

    Returns:
        The distance from each pixel to the nearest background pixel, at the
        shape of ``obj_mask``. The crop is padded by one background pixel on
        every side first, so the bounding box and the image border both
        count as an edge.
    """
    # asarray narrows the scipy stub's tuple-or-ndarray return union; with
    # return_indices=False the call yields an ndarray and copies nothing.
    return np.asarray(distance_transform_edt(np.pad(obj_mask, 1)))[1:-1, 1:-1]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/measure/test_object_geometry.py -q`
Expected: 5 passed.

- [ ] **Step 5: Prove the guards can fail**

Temporarily change `return hull, float(hull.volume)` to `float(hull.area)` and re-run: `test_convex_hull_area_is_the_hull_volume_not_its_perimeter` must FAIL. Temporarily replace `np.pad(obj_mask, 1)` with `obj_mask` and remove the `[1:-1, 1:-1]` slice: `test_object_edt_sees_background_on_every_side_of_the_crop` must FAIL. Revert both changes, then re-run to confirm everything passes.

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff check --fix src/phenotypic/measure/_object_geometry.py tests/unit/measure/test_object_geometry.py
uv run mypy src/phenotypic/measure/_object_geometry.py
git add src/phenotypic/measure/_object_geometry.py tests/unit/measure/test_object_geometry.py
git commit -m "feat(measure): shared convex-hull-area and per-object EDT helpers"
```

---

### Task 3: `SIZE` gains the size magnitudes and the radius family; `MeasureSize` computes them

Purely additive. `SHAPE` is untouched in this task, so both enums temporarily carry Perimeter and similar members, each under its own prefix; this does not collide.

**Files:**
- Modify: `src/phenotypic/schema/_size.py` (whole file)
- Modify: `src/phenotypic/measure/_measure_size.py` (whole file)
- Test: `tests/unit/measure/test_measure_size.py` (create)
- Test: `tests/unit/measure/test_radial_profile.py` (create; ported from the branch)
- Test: `tests/unit/measure/test_size_consolidation_equivalence.py` (create)

**Interfaces:**
- Consumes: `convex_hull_area`, `object_edt` (Task 2).
- Produces:
  - `SIZE` members `AREA`, `INTEGRATED_INTENSITY`, `PERIMETER`, `CONVEX_AREA`, `BBOX_AREA`, `MAJOR_AXIS_LENGTH`, `MINOR_AXIS_LENGTH`, `INSCRIBED_RADIUS`, `MEDIAN_RADIUS`, `MEAN_RADIUS`, `ROBUST_MEAN_RADIUS` and `MAX_RADIUS`, with headers `Size_<Label>`.
  - `MeasureSize` fields `angular_bins: int = 360`, `trim_proportion: float = 0.2` and `plateau_tolerance: float = 0.01`.
  - `MeasureSize._trace_radial_signature(obj_mask, edt) -> np.ndarray | None`.
  - `MeasureSize._measure_radial_profile(obj_mask) -> dict[str, float]`, keyed by the five radius headers.

- [ ] **Step 1: Write the schema.** Replace `src/phenotypic/schema/_size.py` with:

```python
"""The labels and descriptions of the size measurements."""

from ._measurement_info import Entry
from ._tiers import DirectPhenotype


class SIZE(DirectPhenotype):
    """Measure the key size magnitudes of each detected colony.

    Extract colony area, integrated intensity, perimeter, convex-hull and
    bounding-box areas, best-fit-ellipse axis lengths, and a family of radii
    measured from one center inside the colony. These are the starting
    measurements for growth and fitness comparisons; form descriptors
    (circularity, solidity, eccentricity, Feret diameters) live in
    :class:`SHAPE`.
    """

    @classmethod
    def category(cls):
        return "Size"

    AREA = Entry(
        "Area",
        "Total number of pixels occupied by the microbial colony. Represents colony biomass and growth extent on agar plates. Larger areas typically indicate more robust growth or longer incubation times.",
        bio_desc=(
            "Projected 2D footprint of the colony in pixels — a common proxy "
            "for colony size and overall growth in arrayed plate assays. With "
            "matched imaging and incubation, larger area generally reflects "
            "greater proliferation or spreading; it captures only the 2D "
            "footprint, not colony height or cell density."
        ),
        image="shape/area.png",
    )
    INTEGRATED_INTENSITY = Entry(
        "IntegratedIntensity",
        r"The sum of the object's grayscale pixels. Calculated as "
        r":math:`\sum{\text{pixel values}} \times \text{area}`.",
    )
    PERIMETER = Entry(
        "Perimeter",
        "Total length of the colony's outer boundary in pixels. Measures colony edge complexity and surface irregularity. Smooth, circular colonies have shorter perimeters relative to their area compared to irregular or filamentous colonies.",
    )
    CONVEX_AREA = Entry(
        "ConvexArea",
        'Area of the smallest convex polygon that completely contains the colony, computed from the convex hull of its pixel centers. Represents the colony\'s "filled-in" appearance if all indentations and holes were removed. Because the hull passes through pixel centers it is slightly smaller than the pixel count of a convex colony. Useful for detecting colony spreading patterns or invasive growth characteristics.',
    )
    BBOX_AREA = Entry(
        "BboxArea",
        "Area of the smallest rectangle that completely contains the colony. Represents the total spatial shape of the colony including any empty space. In high-throughput assays, this helps assess colony positioning and potential interference with neighboring colonies.",
    )
    MAJOR_AXIS_LENGTH = Entry(
        "MajorAxisLength",
        "Length of the longest axis of the ellipse that best fits the colony shape. Represents the maximum colony dimension. In arrayed microbial growth, this measurement helps identify colonies that have grown beyond their intended grid positions.",
    )
    MINOR_AXIS_LENGTH = Entry(
        "MinorAxisLength",
        "Length of the shortest axis of the ellipse that best fits the colony shape. Represents the minimum colony dimension. Together with major axis length, this helps characterize colony aspect ratio and growth anisotropy.",
    )
    INSCRIBED_RADIUS = Entry(
        "InscribedRadius",
        "Radius of the largest circle that fits entirely inside the colony, equal to "
        "the maximum of the colony's Euclidean distance transform: the distance from "
        "the colony center (see MedianRadius) to its nearest edge. For an ideal disk "
        "it equals the disk radius. It reflects the colony's narrowest dimension, not "
        "its overall extent: an elongated colony reports half its width whatever its "
        "length (a 100 x 20 pixel colony reports 10), and a runner or spur leaves it "
        "unchanged. Use MaxRadius for overall extent. The image border counts as an "
        "edge.",
    )
    MEDIAN_RADIUS = Entry(
        "MedianRadius",
        "Median distance from the colony center to its boundary over the radial "
        "signature: the boundary distance sampled in equal angular directions "
        "(360 by default), keeping the outermost boundary crossing in each. The "
        "center is the centroid of the distance-transform peak plateau; it lies "
        "inside compact colonies, but for a ring-shaped colony it can fall in the "
        "central hole. For an ideal disk it equals the disk "
        "radius; for an ideal 100 x 20 pixel rectangle it is 14.1 pixels. This is "
        "not the value the retired Shape_MedianRadius carried (a median distance to "
        "the nearest edge, now Shape_MedianBoundaryDist).",
    )
    MEAN_RADIUS = Entry(
        "MeanRadius",
        "Mean distance from the colony center to its boundary over the radial "
        "signature (see MedianRadius for the center and sampling). For an ideal "
        "disk it equals the disk radius; for an ideal 100 x 20 pixel rectangle it "
        "is 21.0 pixels. A runner or spur pulls it upward in proportion to its "
        "angular width; use RobustMeanRadius for the compact body. This is not the "
        "value the retired Shape_MeanRadius carried (a mean distance to the nearest "
        "edge, now Shape_MeanBoundaryDist).",
    )
    ROBUST_MEAN_RADIUS = Entry(
        "RobustMeanRadius",
        "Symmetrically trimmed mean (20% from each end by default) of the radial "
        "signature (see MedianRadius). Because the signature is sampled by angle, a "
        "narrow runner contributes only its angular width and is trimmed away, so "
        "this estimates the typical radius of the colony's compact body. The trim "
        "treats genuine elongation the same way: an ideal 100 x 20 pixel rectangle "
        "gives 16.2 pixels against a MeanRadius of 21.0 pixels. For an ideal disk "
        "it equals the disk radius.",
    )
    MAX_RADIUS = Entry(
        "MaxRadius",
        "Largest distance from the colony center to its boundary over the radial "
        "signature (see MedianRadius): the colony's furthest reach. For an ideal "
        "disk it equals the disk radius; for an ideal 100 x 20 pixel rectangle it "
        "is 50.9 pixels. A MaxRadius far above RobustMeanRadius indicates a "
        "protrusion, spur, or runner. This is not the value the retired "
        "Shape_MaxRadius carried (the inscribed radius, now Size_InscribedRadius).",
    )
```

- [ ] **Step 2: Write the failing measurer tests.** Create `tests/unit/measure/test_measure_size.py`:

```python
"""Unit tests for MeasureSize."""

from __future__ import annotations

import warnings

import numpy as np
import pydantic
import pytest

from phenotypic import Image
from phenotypic.measure import MeasureSize
from phenotypic.schema import OBJECT, SIZE


def _image_with_objmap(objmap: np.ndarray) -> Image:
    rgb = np.zeros((*objmap.shape, 3), dtype=np.uint8)
    rgb[objmap > 0] = 200
    image = Image(rgb)
    image.objmap[:] = objmap
    return image


@pytest.fixture
def split_rectangle_image() -> Image:
    """Two labels sharing their full internal edge (a watershed-split pair).

    Label 1 is 41x20 and label 2 is 41x21, touching along 41 pixels with no
    background between them.
    """
    objmap = np.zeros((43, 43), dtype=int)
    objmap[1:42, 1:21] = 1
    objmap[1:42, 21:42] = 2
    return _image_with_objmap(objmap)


def test_emits_exactly_the_size_schema(split_rectangle_image):
    frame = MeasureSize().measure(split_rectangle_image)
    assert list(frame.columns) == [str(OBJECT.LABEL), *SIZE.get_headers()]


def test_area_equals_regionprops_area(split_rectangle_image):
    """Shape's ratios and Intensity's density divide by props.area; the published
    Size_Area must be the same number. Integer pixel counts, compared exactly."""
    frame = MeasureSize().measure(split_rectangle_image)
    props = split_rectangle_image.objects.props
    assert frame[str(SIZE.AREA)].tolist() == [float(p.area) for p in props]
    assert frame[str(SIZE.AREA)].tolist() == [820.0, 861.0]


def test_convex_area_is_the_scipy_hull_volume(split_rectangle_image):
    """Label 1's pixel centres span a 40 x 19 hull: area 760 (the pixel count is 820).

    Mutation: switch the helper to `.area` -> 118.0 (the hull perimeter); switch
    MeasureSize to `props.area_convex` -> 820.0. Both fail.
    """
    frame = MeasureSize().measure(split_rectangle_image)
    assert frame[str(SIZE.CONVEX_AREA)].iloc[0] == pytest.approx(760.0, abs=1e-9)


def test_touching_labels_do_not_inflate_each_others_inscribed_radius(split_rectangle_image):
    """A whole-objmap EDT merges the pair and reports 20/21; per object it is 10/11.
    The EDT of an axis-aligned rectangle is exact integer arithmetic.

    Mutation: compute the EDT over image.objmap[:] -> 20.0/21.0, and this fails.
    """
    frame = MeasureSize().measure(split_rectangle_image)
    assert frame[str(SIZE.INSCRIBED_RADIUS)].tolist() == [10.0, 11.0]


def test_border_touching_colony_counts_the_border_as_an_edge():
    """Review Focus 3. A 10-row band spanning the full width of the top edge: the
    padded crop gives 5; the old whole-image EDT gave 10 (scipy measures only to
    zero pixels inside the array)."""
    objmap = np.zeros((20, 30), dtype=int)
    objmap[0:10, :] = 1
    frame = MeasureSize().measure(_image_with_objmap(objmap))
    assert frame[str(SIZE.INSCRIBED_RADIUS)].iloc[0] == 5.0


def test_no_objects_raises_like_every_other_measurer():
    """Review Focus 1 / amendment A2: main's contract is kept, not changed.
    MeasureFeatures.measure re-raises without chaining, but names the original
    type in the message (abc_/_measure_features.py, the `except Exception` arm)."""
    from phenotypic.sdk_.exceptions_ import OperationFailedError

    with pytest.raises(OperationFailedError, match="NoObjectsError"):
        MeasureSize().measure(_image_with_objmap(np.zeros((20, 20), dtype=int)))


def test_degenerate_objects_measure_without_raising_or_warning():
    """Review Focus 2. A single pixel and a 1-pixel-wide line: Qhull fails, so
    ConvexArea is NaN; the radii come from the EDT and contour and stay finite."""
    objmap = np.zeros((20, 20), dtype=int)
    objmap[2, 2] = 1
    objmap[10, 3:12] = 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        frame = MeasureSize().measure(_image_with_objmap(objmap))
    assert frame[str(SIZE.CONVEX_AREA)].isna().all()
    assert np.isfinite(frame[str(SIZE.INSCRIBED_RADIUS)]).all()
    assert np.isfinite(frame[str(SIZE.MAX_RADIUS)]).all()


def test_trim_proportion_zero_makes_robust_mean_the_plain_mean(split_rectangle_image):
    """With no trim, the trimmed mean is the plain mean. Both average the same
    signature, so they agree to summation-order rounding: 360 terms x 1 ulp
    of ~20 px is ~1.6e-12, so abs=1e-9 has headroom and still catches any real
    wiring slip (a swapped statistic moves the value by more than 1 px here)."""
    frame = MeasureSize(trim_proportion=0.0).measure(split_rectangle_image)
    assert frame[str(SIZE.ROBUST_MEAN_RADIUS)].to_numpy() == pytest.approx(
        frame[str(SIZE.MEAN_RADIUS)].to_numpy(), abs=1e-9
    )


@pytest.mark.parametrize(
    "kwargs",
    [{"angular_bins": 4}, {"trim_proportion": 0.5}, {"plateau_tolerance": 0.0}],
)
def test_field_bounds_are_enforced(kwargs):
    with pytest.raises(pydantic.ValidationError):
        MeasureSize(**kwargs)


def test_non_default_fields_round_trip_through_json():
    """Review Focus 5."""
    op = MeasureSize(angular_bins=90, trim_proportion=0.1, plateau_tolerance=0.02)
    loaded = MeasureSize.from_json(op.to_json())
    assert (loaded.angular_bins, loaded.trim_proportion, loaded.plateau_tolerance) == (
        90, 0.1, 0.02
    )
```

- [ ] **Step 3: Port the radial-profile tests.** Create `tests/unit/measure/test_radial_profile.py` from `git show shape-radial-measures:tests/unit/measure/test_radial_profile.py`, applying exactly these changes:
  - `from phenotypic.measure import MeasureShape` → `from phenotypic.measure import MeasureSize`, and every `MeasureShape(` → `MeasureSize(`.
  - Key renames: `"Shape_InscribedRadius"` → `"Size_InscribedRadius"`, `"Shape_RobustMeanRadius"` → `"Size_RobustMeanRadius"`, and `"Shape_ReachRadius"` → `"Size_MaxRadius"`.
  - **Delete** `test_mean_boundary_dist_of_a_disk_is_one_third_of_its_radius`; it moves to Shape in Task 5.
  - Update the module docstring's first line to `"""Unit tests for MeasureSize's radial-signature geometry.`.

  Then append these tests, which the branch did not have:

```python
def test_disk_median_and_mean_radius_equal_its_radius():
    profile = MeasureSize()._measure_radial_profile(_crop(_disk(40)))
    # The contour sits on the 0.5 iso-level: within TOL of both 40 and 40.5.
    assert profile["Size_MedianRadius"] == pytest.approx(40.0, abs=TOL)
    assert profile["Size_MeanRadius"] == pytest.approx(40.0, abs=TOL)


def test_elongated_colony_matches_the_analytic_rectangle():
    """A 20-row x 100-column pixel block. Its 0.5-iso contour is exactly a 100 x 20
    rectangle about the EDT plateau centroid, so the analytic values from the
    logic-validation script (check 04) apply: 10.0 / 14.1 / 21.0 / 16.2 / 50.9.
    Two rasterisation effects move the measured values, both inside TOL:
    marching squares cuts each corner diagonally, lowering MaxRadius by under
    0.1 px; and max-per-bin takes the extreme vertex in each 1-degree bin, which
    where r(theta) is steep (~4.5 px/degree near the corners) exceeds the
    bin-centre value by up to half a bin x |r'|. Measured on main's code
    (plan-review probe P3): 14.147 / 21.126 / 16.196 / 50.895, largest
    deviation 0.126 px against TOL = 0.6.
    """
    profile = MeasureSize()._measure_radial_profile(np.ones((20, 100), dtype=bool))
    assert profile["Size_InscribedRadius"] == 10.0  # exact: integer EDT
    assert profile["Size_MedianRadius"] == pytest.approx(14.1, abs=TOL)
    assert profile["Size_MeanRadius"] == pytest.approx(21.0, abs=TOL)
    assert profile["Size_RobustMeanRadius"] == pytest.approx(16.2, abs=TOL)
    assert profile["Size_MaxRadius"] == pytest.approx(50.9, abs=TOL)


def _disk_with_wide_runner() -> np.ndarray:
    """Radius-40 colony with a half-width-8 runner reaching to x = 100.

    Analytic (logic-validation script check 06): runner covers 5.6% of directions,
    MeanRadius 42.35, RobustMeanRadius 40.00, MedianRadius 40.00.
    """
    y, x = np.mgrid[-130:130, -130:130]
    return (x**2 + y**2 <= 40**2) | ((np.abs(y) <= 8) & (x >= 0) & (x <= 100))


def test_runner_pulls_the_mean_but_not_the_robust_mean():
    """The analytic gap is 2.35 px, over twice 2 x TOL, so a swap of the two
    estimators cannot pass: each value may drift by at most TOL.

    Mutation: return trim_mean for MEAN_RADIUS and the plain mean for
    ROBUST_MEAN_RADIUS -> the first assertion fails.
    """
    profile = MeasureSize()._measure_radial_profile(_crop(_disk_with_wide_runner()))
    assert profile["Size_RobustMeanRadius"] == pytest.approx(40.0, abs=TOL)
    assert profile["Size_MeanRadius"] - profile["Size_RobustMeanRadius"] > 2 * TOL
    assert profile["Size_MedianRadius"] == pytest.approx(40.0, abs=TOL)


def test_crescent_keeps_the_radius_ordering():
    """Review Focus 4. A concave colony (disk r=40 minus an offset disk r=30) is not
    star-shaped from its center, so several bins cross the boundary more than
    once. The ordering invariants must still hold."""
    y, x = np.mgrid[-60:60, -60:60]
    mask = _crop((x**2 + y**2 <= 40**2) & ((x - 20) ** 2 + y**2 > 30**2))
    p = MeasureSize()._measure_radial_profile(mask)
    assert p["Size_InscribedRadius"] <= p["Size_MedianRadius"] <= p["Size_MaxRadius"]
    assert p["Size_InscribedRadius"] <= p["Size_RobustMeanRadius"] <= p["Size_MaxRadius"]
    assert p["Size_InscribedRadius"] <= p["Size_MeanRadius"] <= p["Size_MaxRadius"]
```

In `test_degenerate_objects_do_not_raise`, extend the `isfinite` assertions with `"Size_MedianRadius"`, `"Size_MeanRadius"` and `"Size_MaxRadius"`.

- [ ] **Step 4: Write the failing baseline-equivalence test.** Create `tests/unit/measure/test_size_consolidation_equivalence.py`:

```python
"""Moved and retained columns keep main's values (baseline: Task 1 of the plan).

Tolerance: every compared value is either a regionprops/Qhull scalar or a
reduction over at most ~1e4 pixels. Cross-platform float differences are
bounded by ~1e4 x 1 ulp ~ 2e-12 relative; rtol=1e-10 leaves 50x headroom
and is still >= 7 orders of magnitude below any real behaviour change (the
convex-area fix alone moved values by > 1e-3).

On the synth plate no colony touches another or the image border (checked
when the plan was written), so the per-object EDT must reproduce main's
whole-image EDT exactly there.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phenotypic.data import load_synth_yeast_plate
from phenotypic.measure import MeasureSize
from phenotypic.schema import OBJECT, SIZE

_BASELINE = Path(__file__).parent / "_golden" / "size_consolidation_baseline.parquet"
RTOL = 1e-10


@pytest.fixture(scope="module")
def baseline() -> pd.DataFrame:
    assert _BASELINE.is_file(), f"missing baseline {_BASELINE}; see plan Task 1"
    return pd.read_parquet(_BASELINE)


@pytest.fixture(scope="module")
def plate():
    return load_synth_yeast_plate()


def _aligned(new: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    merged = new.merge(baseline, on=str(OBJECT.LABEL), validate="one_to_one")
    assert len(merged) == len(baseline) == 96
    return merged


@pytest.mark.parametrize(
    ("size_header", "old_header"),
    [
        (str(SIZE.AREA), "Shape_Area"),
        (str(SIZE.PERIMETER), "Shape_Perimeter"),
        (str(SIZE.CONVEX_AREA), "Shape_ConvexArea"),
        (str(SIZE.BBOX_AREA), "Shape_BboxArea"),
        (str(SIZE.MAJOR_AXIS_LENGTH), "Shape_MajorAxisLength"),
        (str(SIZE.MINOR_AXIS_LENGTH), "Shape_MinorAxisLength"),
        (str(SIZE.INSCRIBED_RADIUS), "Shape_MaxRadius"),
    ],
)
def test_moved_size_columns_keep_mains_values(plate, baseline, size_header, old_header):
    merged = _aligned(MeasureSize().measure(plate), baseline)
    np.testing.assert_allclose(merged[size_header], merged[old_header], rtol=RTOL)
```

The literal `"Shape_*"` strings here are deliberate: they name columns of the baseline file, which is historical data.

- [ ] **Step 5: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/measure/test_measure_size.py tests/unit/measure/test_radial_profile.py tests/unit/measure/test_size_consolidation_equivalence.py -q`
Expected: FAIL. The new SIZE headers are missing from the frame, and `_measure_radial_profile` does not exist.

- [ ] **Step 6: Implement `MeasureSize`.** Replace `src/phenotypic/measure/_measure_size.py` with:

```python
from __future__ import annotations

from typing import ClassVar, TYPE_CHECKING, cast

from phenotypic.schema import OBJECT

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
import pandas as pd
from pydantic import Field
from scipy.ndimage import label as ndi_label
from scipy.stats import trim_mean
from skimage.measure import find_contours

from phenotypic.abc_ import MeasureFeatures
from phenotypic.measure._object_geometry import convex_hull_area, object_edt
from phenotypic.schema import SIZE


class MeasureSize(MeasureFeatures):
    """Measure the key size magnitudes of each detected colony.

    The single source of colony size: area, integrated intensity, perimeter,
    convex-hull and bounding-box areas, best-fit-ellipse axis lengths, and a
    family of five radii that are all measured from one center inside the
    colony (inscribed, median, mean, robust mean and maximum). These are the
    starting measurements for growth and fitness comparisons; see the
    :class:`~phenotypic.schema.SIZE` table below for what each column means.

    The radii come from the colony's *radial signature*: the distance from
    the center to the boundary, sampled in ``angular_bins`` equal directions.
    Sampling by angle rather than along the boundary gives a runner or spur
    only its true angular width, so the trimmed ``RobustMeanRadius`` stays on
    the compact body while ``MeanRadius`` and ``MaxRadius`` show the reach.

    Args:
        angular_bins: Number of equal angular directions in which the radial
            signature samples the boundary. More bins resolve narrower
            protrusions; 360 (one per degree) resolves any runner wider than
            about 1/57 of the colony radius.
        trim_proportion: Fraction trimmed from each end of the radial
            signature for ``RobustMeanRadius``. It tolerates a runner or spur
            covering up to this fraction of all directions; 0 makes
            ``RobustMeanRadius`` equal ``MeanRadius``.
        plateau_tolerance: Relative tolerance defining the distance-transform
            peak plateau whose centroid is the colony center. The default
            (1%) merges the exact ties that integer pixel geometry produces.

    Returns:
        pd.DataFrame: One row per colony with ``Object_Label`` and every
        ``Size_*`` column.

    Best For:
        - Growth and fitness comparisons across strains and conditions
          (area and the radius family).
        - Separating compact growth from runners or spreading
          (``MaxRadius`` against ``RobustMeanRadius``).
        - Filtering debris or aborted growth by minimum size before
          downstream measurement.

    Consider Also:
        - :class:`MeasureShape` for form descriptors (circularity, solidity,
          eccentricity, Feret diameters, interior thickness).
        - :class:`MeasureIntensity` for full intensity statistics.
        - :class:`MeasureGridSpread` for detecting multi-object wells in
          arrayed assays.

    See Also:
        :doc:`/tutorials/notebooks/07_measuring_and_exporting` for a
        walkthrough of measuring and exporting colony data.
        :doc:`/explanation/measurement_metrics_biological_meaning` for
        interpreting size metrics in a biological context.
    """

    _measurement_infoclass: ClassVar[type] = SIZE

    angular_bins: int = Field(360, ge=8, le=3600)
    trim_proportion: float = Field(0.2, ge=0.0, lt=0.5)
    plateau_tolerance: float = Field(0.01, gt=0.0, lt=1.0)

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
        # ndi_label returns int | tuple[ndarray, int]; with no output arg the
        # runtime value is always the tuple, so narrow the stub's union.
        components = cast("tuple[np.ndarray, int]", ndi_label(plateau))[0]
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

    def _measure_radial_profile(self, obj_mask: np.ndarray) -> dict[str, float]:
        """Compute the five radii for one cropped object.

        Args:
            obj_mask (np.ndarray): Boolean mask of a single object within its
                bounding box (``regionprops.image``), already isolated from
                neighbouring labels.

        Returns:
            dict[str, float]: ``SIZE`` header to value for InscribedRadius,
            MedianRadius, MeanRadius, RobustMeanRadius and MaxRadius. The
            four signature radii are NaN when the object has no contour.
        """
        edt = object_edt(obj_mask)
        values = {
            str(SIZE.INSCRIBED_RADIUS): float(edt.max()),
            str(SIZE.MEDIAN_RADIUS): np.nan,
            str(SIZE.MEAN_RADIUS): np.nan,
            str(SIZE.ROBUST_MEAN_RADIUS): np.nan,
            str(SIZE.MAX_RADIUS): np.nan,
        }
        signature = self._trace_radial_signature(obj_mask, edt)
        if signature is not None:
            values[str(SIZE.MEDIAN_RADIUS)] = float(np.median(signature))
            values[str(SIZE.MEAN_RADIUS)] = float(signature.mean())
            values[str(SIZE.ROBUST_MEAN_RADIUS)] = float(
                    trim_mean(signature, self.trim_proportion)
            )
            values[str(SIZE.MAX_RADIUS)] = float(signature.max())
        return values

    def _operate(self, image: Image) -> pd.DataFrame:
        n_objects = image.num_objects
        measurements = {
            str(feature): np.full(shape=n_objects, fill_value=np.nan)
            for feature in SIZE
        }

        objmap = image.objmap[:].copy()
        measurements[str(SIZE.AREA)] = self._calculate_sum(
                array=image.objmask[:], objmap=objmap
        )
        measurements[str(SIZE.INTEGRATED_INTENSITY)] = self._calculate_sum(
                array=image.gray[:].copy(), objmap=objmap
        )

        for idx, props in enumerate(image.objects.props):
            measurements[str(SIZE.PERIMETER)][idx] = props.perimeter
            measurements[str(SIZE.BBOX_AREA)][idx] = props.area_bbox
            measurements[str(SIZE.MAJOR_AXIS_LENGTH)][idx] = props.axis_major_length
            measurements[str(SIZE.MINOR_AXIS_LENGTH)][idx] = props.axis_minor_length
            measurements[str(SIZE.CONVEX_AREA)][idx] = convex_hull_area(props.coords)[1]
            for header, value in self._measure_radial_profile(props.image).items():
                measurements[header][idx] = value

        frame = pd.DataFrame(measurements)
        frame.insert(loc=0, column=OBJECT.LABEL, value=image.objects.labels2series())
        return frame


MeasureSize.__doc__ = SIZE.append_rst_to_doc(MeasureSize)
```

`_trace_radial_signature` is transcribed character-for-character from branch `shape-radial-measures` (`_measure_shape.py` lines 147-210 there), and it is why the imports include `cast`, `ndi_label` and `find_contours`. Do not change its logic; the ported tests pin it, including the max-per-bin mutation.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/measure/test_measure_size.py tests/unit/measure/test_radial_profile.py tests/unit/measure/test_size_consolidation_equivalence.py tests/unit/measure/test_object_geometry.py -q`
Expected: all pass. If `test_elongated_colony_matches_the_analytic_rectangle` misses by more than TOL, **do not widen TOL**. Print the profile and compare it with the validation script's check 04; a miss means the center or the binning differs from the spec.

- [ ] **Step 8: Prove the guards can fail.** Apply each mutation, run the named test, see it FAIL, and revert:
  1. In `_measure_radial_profile`, swap the `MEAN_RADIUS` and `ROBUST_MEAN_RADIUS` expressions → `test_runner_pulls_the_mean_but_not_the_robust_mean`.
  2. In `_trace_radial_signature`, change `np.maximum.at` to accumulate a mean (e.g. `np.add.at` into a sum array, then divide by counts) → `test_reach_uses_the_outermost_crossing_per_bin_not_the_mean`.
  3. Replace `object_edt(obj_mask)` with `np.asarray(distance_transform_edt(obj_mask))` (no padding) → `test_border_touching_colony_counts_the_border_as_an_edge`.
  4. Replace `convex_hull_area(props.coords)[1]` with `props.area_convex` → `test_convex_area_is_the_scipy_hull_volume`.

- [ ] **Step 9: Lint, type-check, commit**

```bash
uv run ruff check --fix src/phenotypic/schema/_size.py src/phenotypic/measure/_measure_size.py tests/unit/measure/test_measure_size.py tests/unit/measure/test_radial_profile.py tests/unit/measure/test_size_consolidation_equivalence.py
uv run mypy src/phenotypic/measure/_measure_size.py src/phenotypic/schema/_size.py
git add src/phenotypic/schema/_size.py src/phenotypic/measure/_measure_size.py tests/unit/measure/test_measure_size.py tests/unit/measure/test_radial_profile.py tests/unit/measure/test_size_consolidation_equivalence.py
git commit -m "feat(measure): MeasureSize emits size magnitudes and a five-member radius family"
```

---

### Task 4: Decouple `MeasureIntensity` and `KeepSectionLargest` from other measurers

**Files:**
- Modify: `src/phenotypic/measure/_measure_intensity.py:62-117` (`_operate`)
- Modify: `src/phenotypic/refine/_keep_section_largest.py` (imports and `_operate`)
- Test: `tests/unit/measure/test_size_consolidation_equivalence.py` (extend)
- Test: `tests/unit/refine/test_keep_section_largest.py` (create)

**Interfaces:**
- Consumes: `convex_hull_area` (Task 2), `MeasureSize` (Task 3, used only as the test oracle).
- Produces: no new names. `MeasureIntensity` and `KeepSectionLargest` stop importing any measurer.

- [ ] **Step 1: Write the failing tests.** Append to `test_size_consolidation_equivalence.py`:

```python
def test_measure_intensity_is_unchanged_from_main(plate, baseline):
    from phenotypic.measure import MeasureIntensity

    new = MeasureIntensity().measure(plate)
    merged = new.merge(baseline, on=str(OBJECT.LABEL), suffixes=("", "_main"),
                       validate="one_to_one")
    for column in new.columns:
        if column == str(OBJECT.LABEL):
            continue
        np.testing.assert_allclose(merged[column], merged[f"{column}_main"], rtol=RTOL)


def test_measure_intensity_does_not_run_another_measurer(plate, monkeypatch):
    """Mutation: restore `MeasureShape().measure(image)` in MeasureIntensity -> fails."""
    from phenotypic.measure import MeasureIntensity, MeasureShape

    def _refuse(self, image):
        raise AssertionError("MeasureIntensity must not run MeasureShape")

    monkeypatch.setattr(MeasureShape, "_operate", _refuse)
    monkeypatch.setattr(MeasureSize, "_operate", _refuse)
    MeasureIntensity().measure(plate)
```

Create `tests/unit/refine/test_keep_section_largest.py`:

```python
"""KeepSectionLargest selects the largest object per grid section."""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureSize
from phenotypic.refine import KeepSectionLargest
from phenotypic.schema import GRID, OBJECT, SIZE


@pytest.fixture(scope="module")
def detected():
    """Otsu on the synth plate over-segments (552 objects for 96 wells), so most
    sections hold several candidates and the selection is non-trivial."""
    image = OtsuDetector().apply(load_synth_yeast_plate())
    assert image.num_objects > 96
    return image


def _labels_by_the_old_algorithm(image) -> np.ndarray:
    """main's implementation, kept as the oracle: MeasureSize area, idxmax per section."""
    table = MeasureSize().measure(image, include_meta=True)
    max_idx = table.groupby(by=GRID.ROW_MAJOR_IDX, observed=True)[SIZE.AREA].idxmax()
    return np.sort(table.loc[max_idx, OBJECT.LABEL].to_numpy())


def test_selects_the_same_labels_as_before(detected):
    expected = _labels_by_the_old_algorithm(detected.copy())
    result = KeepSectionLargest().apply(detected.copy())
    kept = np.unique(result.objmap[:])
    assert np.array_equal(kept[kept > 0], expected)


def test_does_not_run_a_measurer(detected, monkeypatch):
    """Mutation: restore `MeasureSize().measure(...)` in _operate -> fails."""
    def _refuse(self, image):
        raise AssertionError("KeepSectionLargest must not run MeasureSize")

    monkeypatch.setattr(MeasureSize, "_operate", _refuse)
    KeepSectionLargest().apply(detected.copy())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/measure/test_size_consolidation_equivalence.py tests/unit/refine/test_keep_section_largest.py -q`
Expected: both `does_not_run` tests FAIL with the `AssertionError`; the equivalence and selection tests pass (this is behaviour-preserving work).

- [ ] **Step 3: Implement `MeasureIntensity`.** In `_operate`, delete the two local imports (`MeasureShape`, `SHAPE`) and replace the block from `shape_measurements = MeasureShape().measure(image)` down to the final `CONVEX_DENSITY` assignment with:

```python
        props = image.objects.props
        areas = np.array([p.area for p in props], dtype=float)
        convex_areas = np.array(
                [convex_hull_area(p.coords)[1] for p in props], dtype=float
        )
        measurements[INTENSITY.DENSITY] = (
                measurements[INTENSITY.INTEGRATED_INTENSITY] / areas
        )
        measurements[INTENSITY.CONVEX_DENSITY] = (
                measurements[INTENSITY.INTEGRATED_INTENSITY] / convex_areas
        )
```

Add `from phenotypic.measure._object_geometry import convex_hull_area` to the module imports, and `import numpy as np` if it is absent.

- [ ] **Step 4: Implement `KeepSectionLargest`.** Replace the imports and `_operate`:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage

from phenotypic.abc_ import GridObjectRefiner
from phenotypic.schema import GRID, OBJECT
```

```python
    def _operate(self, image: GridImage) -> GridImage:
        # Pixel area per label straight from the objmap: the same numbers as
        # Size_Area without running a measurer (and its radial signatures).
        objmap = image.objmap[:]
        labels = image.objects.labels2series().to_numpy()
        pixel_area = np.bincount(objmap.ravel())[labels]
        # Same merge order as MeasureFeatures.measure(include_meta=True), so
        # idxmax breaks ties between equal areas exactly as before.
        table = image.grid.info(include_metadata=True).merge(
                pd.DataFrame({str(OBJECT.LABEL): labels, "_pixel_area": pixel_area}),
                on=str(OBJECT.LABEL),
        )
        max_idx = table.groupby(by=GRID.ROW_MAJOR_IDX, observed=True)[
            "_pixel_area"
        ].idxmax()
        max_size_labels = table.loc[max_idx, str(OBJECT.LABEL)].to_numpy()

        # Drop objects not the largest
        nonmax_mask = ~np.isin(image.objmap[:], max_size_labels)
        image.objmap[nonmax_mask] = 0
        return image
```

Keep the class docstring unchanged.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/measure/test_size_consolidation_equivalence.py tests/unit/refine/test_keep_section_largest.py -q`
Expected: all pass.

- [ ] **Step 6: Run the directly-touched surface**

Run: `uv run pytest tests/unit/refine tests/unit/measure -o addopts= -q -m "not slow" -p no:randomly`
Expected: no new failures compared with main. If something fails, run that test alone on main — in the orchestrator's detached worktree at `81d19ec66` (A8), **never `git stash`** — before attributing it to this change.

- [ ] **Step 7: Lint, type-check, commit**

```bash
uv run ruff check --fix src/phenotypic/measure/_measure_intensity.py src/phenotypic/refine/_keep_section_largest.py tests/unit/measure/test_size_consolidation_equivalence.py tests/unit/refine/test_keep_section_largest.py
uv run mypy src/phenotypic/measure/_measure_intensity.py src/phenotypic/refine/_keep_section_largest.py
git add src/phenotypic/measure/_measure_intensity.py src/phenotypic/refine/_keep_section_largest.py tests/unit/measure/test_size_consolidation_equivalence.py tests/unit/refine/test_keep_section_largest.py
git commit -m "refactor: MeasureIntensity and KeepSectionLargest read regionprops instead of running measurers"
```

---

### Task 5: Flip `SHAPE` to form descriptors only

**Files:**
- Modify: `src/phenotypic/schema/_shape.py`
- Modify: `src/phenotypic/measure/_measure_shape.py` (whole file)
- Modify: `src/phenotypic/schema/CLAUDE.md` (straddler example)
- Test: `tests/unit/measure/test_measure_shape.py` (create)
- Test: `tests/unit/measure/test_size_consolidation_equivalence.py` (extend)
- Test: `tests/unit/schema/test_classification.py:121-133`, `tests/unit/schema/test_schema_public_api.py:9-14`, `tests/unit/schema/test_dynamic_headers.py:40-43`, `tests/unit/util/test_measurement_outputs.py:25-70`

**Interfaces:**
- Consumes: `convex_hull_area`, `object_edt` (Task 2).
- Produces: the `SHAPE` members `CIRCULARITY`, `COMPACTNESS`, `SOLIDITY`, `EXTENT`, `ECCENTRICITY`, `ORIENTATION`, `MIN_FERET_DIAMETER`, `MAX_FERET_DIAMETER`, `MEAN_BOUNDARY_DIST` and `MEDIAN_BOUNDARY_DIST`. **Removed:** `AREA`, `PERIMETER`, `CONVEX_AREA`, `BBOX_AREA`, `MAJOR_AXIS_LENGTH`, `MINOR_AXIS_LENGTH`, `MEAN_RADIUS`, `MEDIAN_RADIUS` and `MAX_RADIUS`.

- [ ] **Step 1: Write the failing tests.** Create `tests/unit/measure/test_measure_shape.py`:

```python
"""Unit tests for MeasureShape after the size/shape split."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from phenotypic import Image
from phenotypic.measure import MeasureShape
from phenotypic.schema import OBJECT, SHAPE


def _image_with_objmap(objmap: np.ndarray) -> Image:
    rgb = np.zeros((*objmap.shape, 3), dtype=np.uint8)
    rgb[objmap > 0] = 200
    image = Image(rgb)
    image.objmap[:] = objmap
    return image


@pytest.fixture
def split_rectangle_image() -> Image:
    objmap = np.zeros((43, 43), dtype=int)
    objmap[1:42, 1:21] = 1
    objmap[1:42, 21:42] = 2
    return _image_with_objmap(objmap)


def test_emits_exactly_the_shape_schema(split_rectangle_image):
    frame = MeasureShape().measure(split_rectangle_image)
    assert list(frame.columns) == [str(OBJECT.LABEL), *SHAPE.get_headers()]


def test_size_magnitudes_and_misnamed_radii_are_gone():
    headers = set(SHAPE.get_headers())
    for retired in ("Area", "Perimeter", "ConvexArea", "BboxArea", "MajorAxisLength",
                    "MinorAxisLength", "MeanRadius", "MedianRadius", "MaxRadius"):
        assert f"Shape_{retired}" not in headers
    assert {"Shape_MeanBoundaryDist", "Shape_MedianBoundaryDist"} <= headers


def test_touching_labels_do_not_inflate_boundary_distances(split_rectangle_image):
    """Branch-verified values for the 41x20 / 41x21 pair measured per object. The
    merged EDT is off by 2.56 px, so abs=1e-3 cannot pass by accident.

    Mutation: compute the EDT over image.objmap[:] -> fails.
    """
    frame = MeasureShape().measure(split_rectangle_image)
    dist = frame[str(SHAPE.MEAN_BOUNDARY_DIST)]
    assert dist.iloc[0] == pytest.approx(4.6951, abs=1e-3)
    assert dist.iloc[1] == pytest.approx(4.8676, abs=1e-3)
    assert frame[str(SHAPE.MEDIAN_BOUNDARY_DIST)].iloc[0] == pytest.approx(4.0, abs=1e-9)


def test_mean_boundary_dist_of_a_disk_is_one_third_of_its_radius():
    """Pins the rename: this column is interior thickness, not a radius. The analytic
    value is R/3 = 13.33; rasterisation lifts it slightly, so 0.3 px (under 1/R of R
    at R=40, the validation script's check 01 mechanism)."""
    y, x = np.mgrid[-50:50, -50:50]
    objmap = (x**2 + y**2 <= 40**2).astype(int)
    frame = MeasureShape().measure(_image_with_objmap(objmap))
    assert frame[str(SHAPE.MEAN_BOUNDARY_DIST)].iloc[0] == pytest.approx(40 / 3, abs=0.3)


def test_no_objects_raises_like_every_other_measurer():
    """Amendment A2: main's contract is kept."""
    from phenotypic.sdk_.exceptions_ import OperationFailedError

    with pytest.raises(OperationFailedError, match="NoObjectsError"):
        MeasureShape().measure(_image_with_objmap(np.zeros((20, 20), dtype=int)))


def test_degenerate_objects_give_nan_hull_measures_without_warning():
    objmap = np.zeros((20, 20), dtype=int)
    objmap[2, 2] = 1
    objmap[10, 3:12] = 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        frame = MeasureShape().measure(_image_with_objmap(objmap))
    for header in (SHAPE.SOLIDITY, SHAPE.MIN_FERET_DIAMETER, SHAPE.MAX_FERET_DIAMETER):
        assert frame[str(header)].isna().all(), header
```

Append to `test_size_consolidation_equivalence.py`:

```python
@pytest.mark.parametrize(
    ("shape_header", "old_header"),
    [
        ("Shape_Circularity", "Shape_Circularity"),
        ("Shape_Compactness", "Shape_Compactness"),
        ("Shape_Solidity", "Shape_Solidity"),
        ("Shape_Extent", "Shape_Extent"),
        ("Shape_Eccentricity", "Shape_Eccentricity"),
        ("Shape_Orientation", "Shape_Orientation"),
        ("Shape_MinFeretDiameter", "Shape_MinFeretDiameter"),
        ("Shape_MaxFeretDiameter", "Shape_MaxFeretDiameter"),
        ("Shape_MeanBoundaryDist", "Shape_MeanRadius"),
        ("Shape_MedianBoundaryDist", "Shape_MedianRadius"),
    ],
)
def test_retained_shape_columns_keep_mains_values(plate, baseline, shape_header, old_header):
    from phenotypic.measure import MeasureShape

    new = MeasureShape().measure(plate)
    merged = new.merge(baseline, on=str(OBJECT.LABEL), suffixes=("", "_main"),
                       validate="one_to_one")
    right = f"{old_header}_main" if old_header in new.columns else old_header
    np.testing.assert_allclose(merged[shape_header], merged[right], rtol=RTOL)
```

Update the schema tests:
- `tests/unit/schema/test_classification.py`: replace `test_shape_straddles_tier1_and_tier2` with

```python
def test_shape_straddles_tier1_and_tier2():
    from phenotypic.schema import SHAPE

    tier1 = {SHAPE.MIN_FERET_DIAMETER, SHAPE.MAX_FERET_DIAMETER}
    tier2 = {SHAPE.CIRCULARITY, SHAPE.ECCENTRICITY, SHAPE.SOLIDITY, SHAPE.EXTENT,
             SHAPE.COMPACTNESS, SHAPE.ORIENTATION, SHAPE.MEAN_BOUNDARY_DIST,
             SHAPE.MEDIAN_BOUNDARY_DIST}
    for m in tier1:
        assert m.resolved_tier == 1, m
    for m in tier2:
        assert m.resolved_tier == 2, m
    assert tier1 | tier2 == set(SHAPE)  # full coverage, no member missed


def test_size_members_are_all_direct_phenotype():
    from phenotypic.schema import SIZE

    for m in SIZE:
        assert (m.resolved_kind, m.resolved_tier) == ("primary", 1), m
```

- `tests/unit/schema/test_schema_public_api.py:12-14`: change `SHAPE.AREA.value == "Shape_Area"`, `"Shape_Area" in SHAPE.get_headers()` and `"Area" in SHAPE.get_labels()` to `SHAPE.CIRCULARITY.value == "Shape_Circularity"`, `"Shape_Circularity" in SHAPE.get_headers()` and `"Circularity" in SHAPE.get_labels()`.
- `tests/unit/schema/test_dynamic_headers.py:41-43`: `"Shape_Area"` → `"Shape_Circularity"`, `SHAPE.AREA` → `SHAPE.CIRCULARITY`, and `"Shape_Area_extra"` → `"Shape_Circularity_extra"`.
- `tests/unit/util/test_measurement_outputs.py`: in `_base_measurements`, replace `str(SHAPE.AREA): [11.0, 21.0]` and `str(SHAPE.PERIMETER): [12.0, 22.0]` with `str(SHAPE.CIRCULARITY): [0.9, 0.8]` and `str(SHAPE.SOLIDITY): [0.95, 0.85]`. Make the same substitution in the expected `shape_df` column list.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/measure/test_measure_shape.py tests/unit/measure/test_size_consolidation_equivalence.py tests/unit/schema/test_classification.py tests/unit/schema/test_schema_public_api.py tests/unit/schema/test_dynamic_headers.py tests/unit/util/test_measurement_outputs.py -q`
Expected: FAIL with `AttributeError: MEAN_BOUNDARY_DIST`, among others.

- [ ] **Step 3: Rewrite `SHAPE`.** In `src/phenotypic/schema/_shape.py`:
  - **Delete** the Entries `AREA`, `PERIMETER`, `CONVEX_AREA`, `MEDIAN_RADIUS`, `MEAN_RADIUS`, `MAX_RADIUS`, `BBOX_AREA`, `MAJOR_AXIS_LENGTH` and `MINOR_AXIS_LENGTH`.
  - Keep `MIN_FERET_DIAMETER` and `MAX_FERET_DIAMETER` (still `tier=1`), plus `CIRCULARITY`, `ECCENTRICITY`, `SOLIDITY`, `EXTENT`, `COMPACTNESS` and `ORIENTATION` (no tier tag), all unchanged.
  - Change the `tier()` comment to `# default for form descriptors; Feret diameters override via Entry(tier=1)`.
  - Replace the class docstring with:

```python
    """Measure the form of each detected colony.

    Extract dimensionless and angular form descriptors -- circularity,
    compactness, solidity, extent, eccentricity, orientation -- plus the
    Feret caliper diameters and the colony's interior thickness (distance
    from its pixels to the nearest edge). Size magnitudes (area, perimeter,
    radii, axis lengths) live in :class:`SIZE`.
    """
```

  - Add, after `ORIENTATION`:

```python
    MEAN_BOUNDARY_DIST = Entry(
        "MeanBoundaryDist",
        "Mean Euclidean distance from each colony pixel to the nearest background "
        "pixel, computed on the object in isolation. This is a measure of interior "
        "thickness, not a radius: for an ideal disk of radius R it equals "
        r":math:`R/3`. High values relative to Size_InscribedRadius indicate a "
        "compact, convex colony; low values indicate a thin or filamentous one.",
    )
    MEDIAN_BOUNDARY_DIST = Entry(
        "MedianBoundaryDist",
        "Median Euclidean distance from each colony pixel to the nearest background "
        "pixel, computed on the object in isolation. This is a measure of interior "
        "thickness, not a radius: for an ideal disk of radius R it equals "
        r":math:`R(1 - 1/\sqrt{2}) \approx 0.293R`. More robust to boundary "
        "raggedness than MeanBoundaryDist. See Size_InscribedRadius and "
        "Size_RobustMeanRadius for the colony's radial extent.",
    )
```

- [ ] **Step 4: Rewrite `MeasureShape`.** Replace `src/phenotypic/measure/_measure_shape.py` with the following. `_calculate_feret_diameters` is main's method, unchanged.

```python
from __future__ import annotations

from typing import ClassVar, TYPE_CHECKING

from phenotypic.schema import OBJECT

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
import pandas as pd

from phenotypic.abc_ import MeasureFeatures
from phenotypic.measure._object_geometry import convex_hull_area, object_edt
from phenotypic.schema import SHAPE


class MeasureShape(MeasureFeatures):
    r"""Measure the form of each detected colony.

    Extract form descriptors from each colony: circularity and compactness
    (boundary regularity), solidity and extent (how completely the colony
    fills its convex hull and bounding box), eccentricity and orientation
    (elongation and its direction), the Feret caliper diameters, and
    interior thickness (mean and median distance to the nearest edge).
    Colony size -- area, perimeter, radii, axis lengths -- is measured by
    :class:`MeasureSize`; add both to a pipeline for a full profile.

    Returns:
        pd.DataFrame: One row per colony with ``Object_Label`` and every
        ``Shape_*`` column.

    Best For:
        - Distinguishing colony morphotypes (smooth circular wild-type
          vs wrinkled, branching, or invasive mutants).
        - Assessing growth symmetry and directionality via eccentricity
          and orientation.
        - Detecting invasive or spreading growth through low solidity
          values.
        - Morphological clustering for automated strain identification.

    Consider Also:
        - :class:`MeasureSize` for colony area, perimeter, radii and axis
          lengths.
        - :class:`MeasureTexture` for surface roughness and pattern
          features that complement shape metrics.
        - :class:`MeasureBounds` for bounding box and centroid data
          without shape statistics.

    See Also:
        :doc:`/tutorials/notebooks/07_measuring_and_exporting` for a
        walkthrough of measuring and exporting colony data.
        :doc:`/explanation/measurement_metrics_biological_meaning` for
        interpreting shape metrics in a biological context.
    """

    _measurement_infoclass: ClassVar[type] = SHAPE

    @staticmethod
    def _calculate_feret_diameters(hull_points: np.ndarray) -> tuple[float, float]:
        """Calculate minimum and maximum Feret diameters from convex hull points.

        The Feret diameter is the distance between two parallel lines tangent to the object.
        Maximum Feret diameter: longest distance between any two points on the convex hull.
        Minimum Feret diameter: computed using rotating calipers algorithm to find the
        minimum width of the object across all orientations.

        Args:
            hull_points: Nx2 array of coordinates representing convex hull vertices

        Returns:
            tuple: (max_feret, min_feret) diameters
        """
        if len(hull_points) < 2:
            return (np.nan, np.nan)

        # Maximum Feret: compute pairwise distances and find maximum
        # This is the straightforward maximum distance between any two hull vertices
        distances = np.sqrt(
                ((hull_points[:, None, :] - hull_points[None, :, :]) ** 2).sum(axis=2)
        )
        max_feret = np.max(distances)

        # Minimum Feret: use rotating calipers algorithm
        # For each edge of the convex hull, calculate perpendicular distance to all other points
        n = len(hull_points)
        min_feret = np.inf

        for i in range(n):
            # Define edge vector from point i to point i+1
            p1 = hull_points[i]
            p2 = hull_points[(i + 1) % n]
            edge = p2 - p1
            edge_length = np.linalg.norm(edge)

            if edge_length == 0:
                continue

            # Normalized perpendicular direction to the edge
            edge_unit = edge / edge_length
            perpendicular = np.array([-edge_unit[1], edge_unit[0]])

            # Project all hull points onto the perpendicular direction
            projections = np.dot(hull_points - p1, perpendicular)

            # The width in this direction is the range of projections
            width = np.max(projections) - np.min(projections)
            min_feret = min(min_feret, width)

        return (max_feret, min_feret)

    def _operate(self, image: Image) -> pd.DataFrame:
        n_objects = image.num_objects
        measurements = {
            str(feature): np.full(shape=n_objects, fill_value=np.nan)
            for feature in SHAPE
        }

        for idx, props in enumerate(image.objects.props):
            edt = object_edt(props.image)
            interior = edt[props.image]
            measurements[str(SHAPE.MEAN_BOUNDARY_DIST)][idx] = float(interior.mean())
            measurements[str(SHAPE.MEDIAN_BOUNDARY_DIST)][idx] = float(np.median(interior))

            measurements[str(SHAPE.ECCENTRICITY)][idx] = props.eccentricity
            measurements[str(SHAPE.EXTENT)][idx] = props.extent
            measurements[str(SHAPE.ORIENTATION)][idx] = props.orientation

            numer = 4 * np.pi * props.area
            denom = props.perimeter ** 2
            measurements[str(SHAPE.CIRCULARITY)][idx] = (
                numer / denom if denom != 0 else np.nan
            )
            measurements[str(SHAPE.COMPACTNESS)][idx] = (
                denom / numer if numer != 0 else np.nan
            )

            hull, hull_area = convex_hull_area(props.coords)
            if hull is not None:
                measurements[str(SHAPE.SOLIDITY)][idx] = props.area / hull_area
                max_feret, min_feret = self._calculate_feret_diameters(
                        props.coords[hull.vertices]
                )
                measurements[str(SHAPE.MAX_FERET_DIAMETER)][idx] = max_feret
                measurements[str(SHAPE.MIN_FERET_DIAMETER)][idx] = min_feret

        frame = pd.DataFrame(measurements)
        frame.insert(loc=0, column=OBJECT.LABEL, value=image.objects.labels2series())
        return frame


MeasureShape.__doc__ = SHAPE.append_rst_to_doc(MeasureShape)
```

No `scipy.spatial`, `warnings` or `distance_transform_edt` imports remain.

- [ ] **Step 5: Update the straddler example** in `src/phenotypic/schema/CLAUDE.md` (the paragraph starting "Example straddler: `class SHAPE(PrimaryMeasure)`"):

```markdown
Example straddler: `class SHAPE(PrimaryMeasure)` overrides `tier()` to return `2`
(form descriptors default to Descriptive trait); its Feret diameters carry
`Entry(..., tier=1)` so they resolve to Direct phenotype while
`CIRCULARITY`/`ECCENTRICITY`/the boundary distances take the class default of 2.
Size magnitudes (area, perimeter, radii, axis lengths) are not here: they live in
`SIZE(DirectPhenotype)`, which resolves every member to tier 1 without tags.
```

- [ ] **Step 6: Run the tests to verify they pass**

Run the Step 2 command. Expected: all pass.

- [ ] **Step 7: Prove the guards can fail.** Replace `object_edt(props.image)` with the old whole-image EDT, indexed by label (`distance_transform_edt(image.objmap[:])` restricted to `image.objmap[:] == props.label`), and confirm `test_touching_labels_do_not_inflate_boundary_distances` FAILS. Revert.

- [ ] **Step 8: Find every remaining `src/` reference to a removed member**

Run: `grep -rnE "SHAPE\.(AREA|PERIMETER|CONVEX_AREA|BBOX_AREA|MAJOR_AXIS_LENGTH|MINOR_AXIS_LENGTH|MEAN_RADIUS|MEDIAN_RADIUS|MAX_RADIUS)\b" src/ scripts/`
Expected at this point: only docstring or doctest hits (and `scripts/`), all fixed in Task 7. **Any hit in executable code** (not inside a docstring) is a runtime `AttributeError` waiting to happen: fix it now by applying the Task 7 mapping, and list it in the commit message.

- [ ] **Step 9: Phase gate: the affected surface, once.** The surface is every test importing `phenotypic.measure`, `phenotypic.schema`, `phenotypic.refine` or `phenotypic.util`:

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/measure tests/unit/schema tests/unit/refine tests/unit/util tests/unit/docs -q -m "not slow" -p no:randomly
```

Expected: the only failures are tests still spelling a retired `Shape_*` name as data, which Task 8 sweeps. Record their node ids in the commit message body. Any other failure blocks the task.

- [ ] **Step 10: Lint, type-check, commit**

```bash
uv run ruff check --fix src/phenotypic/schema/_shape.py src/phenotypic/measure/_measure_shape.py tests/unit/measure/test_measure_shape.py tests/unit/measure/test_size_consolidation_equivalence.py tests/unit/schema/test_classification.py tests/unit/schema/test_schema_public_api.py tests/unit/schema/test_dynamic_headers.py tests/unit/util/test_measurement_outputs.py
uv run mypy src/phenotypic/measure src/phenotypic/schema
git add -A src/phenotypic/schema src/phenotypic/measure tests/unit/measure tests/unit/schema tests/unit/util
git commit -m "feat(measure)!: MeasureShape emits form descriptors only; size magnitudes live in MeasureSize"
```

---

### Task 6: `change_note()` hook, highlighted 0.20.0 note, version bump

**Files:**
- Create: `src/phenotypic/schema/_change_notes.py`
- Modify: `src/phenotypic/schema/_measurement_info.py` (add `change_note`; update `append_rst_to_doc` at lines ~573-598)
- Modify: `src/phenotypic/schema/_size.py`, `src/phenotypic/schema/_shape.py` (override `change_note`; append to the enum docstring)
- Modify: `docs/source/_extensions/measurements_ref.py:_class_section`
- Modify: `src/phenotypic/__init__.py:17`
- Test: `tests/unit/schema/test_change_note.py` (create), `tests/unit/docs/test_measurements_ref_extension.py` (extend), `tests/unit/sdk_/test_norm_migration.py:89-90`

**Interfaces:**
- Produces: `MeasurementInfo.change_note() -> str` (classmethod, default `""`) and `SIZE_SHAPE_SPLIT_NOTE: str` in `phenotypic.schema._change_notes`.

- [ ] **Step 1: Write the failing tests.** Create `tests/unit/schema/test_change_note.py`:

```python
"""The 0.20.0 size/shape change note renders from one hook in every doc surface."""

from __future__ import annotations

import phenotypic
from phenotypic.measure import MeasureShape, MeasureSize, MeasureTexture
from phenotypic.schema import SHAPE, SIZE, TEXTURE, Entry, MeasurementInfo

MARKER = ".. versionchanged:: 0.20.0"


def test_version_is_0_20_0():
    assert phenotypic.__version__ == "0.20.0"


def test_default_change_note_is_empty_and_leaves_docs_untouched():
    class _Plain(MeasurementInfo):
        @classmethod
        def category(cls):
            return "Plain"

        A = Entry("A", "alpha")

    assert _Plain.change_note() == ""
    assert _Plain.append_rst_to_doc("Doc.") == "Doc.\n\n" + _Plain.rst_table()


def test_size_and_shape_notes_carry_the_rename_table_and_the_trap():
    for info in (SIZE, SHAPE):
        note = info.change_note()
        assert note.startswith(MARKER)
        assert "``Shape_MaxRadius``" in note and "``Size_InscribedRadius``" in note
        assert "``Shape_MeanRadius``" in note and "``Shape_MeanBoundaryDist``" in note
        assert "Same name, different value" in note


def test_note_renders_above_the_table_in_measurer_docs():
    """Anchor on the table directive, not a column name: the note itself spells
    ``Size_Area``, so a header anchor would pass with the note below the table."""
    for measurer in (MeasureSize, MeasureShape):
        doc = measurer.__doc__
        assert MARKER in doc
        assert doc.index(MARKER) < doc.index(".. list-table::"), measurer


def test_note_renders_in_the_enum_docstrings():
    assert MARKER in SIZE.__doc__
    assert MARKER in SHAPE.__doc__


def test_unrelated_classes_carry_no_note():
    assert MARKER not in MeasureTexture.__doc__
    assert TEXTURE.change_note() == ""
```

Append to `tests/unit/docs/test_measurements_ref_extension.py`:

```python
def test_class_section_renders_the_change_note_above_the_table(monkeypatch: MonkeyPatch):
    """Mutation: drop the change_note() line from _class_section -> fails."""
    extension = _load_extension(monkeypatch)
    section = extension._class_section(schema.SIZE)
    marker = ".. versionchanged:: 0.20.0"
    assert marker in section
    assert section.index(marker) < section.index(".. list-table::")
    assert marker not in extension._class_section(schema.TEXTURE)
```

In `tests/unit/sdk_/test_norm_migration.py`, **delete** `test_version_is_0_19_0` (lines 89-90); the version pin now lives in `test_change_note.py`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/schema/test_change_note.py tests/unit/docs/test_measurements_ref_extension.py -q`
Expected: FAIL with `AttributeError: ... change_note` and the version assertion.

- [ ] **Step 3: Create the note text.** Create `src/phenotypic/schema/_change_notes.py`:

```python
"""Release notes rendered above measurement tables by ``MeasurementInfo.change_note``.

One constant per public-column change. Each is RST, rendered verbatim in the
measurer's class docs, the enum's API page, and the Measurements reference
page. Never copy these into ``Entry.desc``: descs are published into every
run's README, so a release note there would ship with every future run.
"""

SIZE_SHAPE_SPLIT_NOTE = """\
.. versionchanged:: 0.20.0
   Colony size magnitudes moved from :class:`~phenotypic.schema.SHAPE` to
   :class:`~phenotypic.schema.SIZE`: :class:`~phenotypic.measure.MeasureSize`
   is now their only source, and :class:`~phenotypic.measure.MeasureShape`
   emits form descriptors only. The radius columns were rebuilt so each name
   matches its value. Retired columns are not aliased, and stores written by
   earlier versions keep their old column names.

   ==============================  ================================
   Retired column                  Successor
   ==============================  ================================
   ``Shape_Area``                  ``Size_Area``
   ``Shape_Perimeter``             ``Size_Perimeter``
   ``Shape_ConvexArea``            ``Size_ConvexArea``
   ``Shape_BboxArea``              ``Size_BboxArea``
   ``Shape_MajorAxisLength``       ``Size_MajorAxisLength``
   ``Shape_MinorAxisLength``       ``Size_MinorAxisLength``
   ``Shape_MaxRadius``             ``Size_InscribedRadius``
   ``Shape_MeanRadius``            ``Shape_MeanBoundaryDist``
   ``Shape_MedianRadius``          ``Shape_MedianBoundaryDist``
   ==============================  ================================

   New columns: ``Size_MedianRadius``, ``Size_MeanRadius``,
   ``Size_RobustMeanRadius`` and ``Size_MaxRadius``, all measured from one
   center inside the colony.

   **Same name, different value:** ``Size_MedianRadius``, ``Size_MeanRadius``
   and ``Size_MaxRadius`` are *not* the retired ``Shape_MedianRadius``,
   ``Shape_MeanRadius`` and ``Shape_MaxRadius``. Compare old data against the
   successor in the table above, never against the same-named ``Size_`` column.
"""
```

- [ ] **Step 4: Add the hook.** In `src/phenotypic/schema/_measurement_info.py`, add after `rembi_module` (near line 370):

```python
    @classmethod
    def change_note(cls) -> str:
        """Return an RST change-note block rendered above this enum's table.

        Rendered by :meth:`append_rst_to_doc` (measurer class docs and the
        enum's own docstring) and by the Measurements reference page. Empty
        by default; an enum overrides it when a release changes its public
        columns. Keep the text in ``phenotypic.schema._change_notes``.

        Returns:
            str: RST, typically a ``.. versionchanged::`` directive, or ``""``.
        """
        return ""
```

Replace the body of `append_rst_to_doc` (from `if isinstance(module, str):` to its final `return`) with:

```python
        doc = module if isinstance(module, str) else (module.__doc__ or "")
        note = cls.change_note()
        parts = [doc, note, cls.rst_table()] if note else [doc, cls.rst_table()]
        return "\n\n".join(parts)
```

In its docstring, add after the first paragraph: `When the enum defines a :meth:`change_note`, it is placed between the docstring and the table.`

- [ ] **Step 5: Override in both enums.** In `_size.py` and `_shape.py`, add `from ._change_notes import SIZE_SHAPE_SPLIT_NOTE` and, inside each class after `category`:

```python
    @classmethod
    def change_note(cls) -> str:
        return SIZE_SHAPE_SPLIT_NOTE
```

At the bottom of `_size.py`: `SIZE.__doc__ = f"{SIZE.__doc__}\n\n{SIZE.change_note()}"`. At the bottom of `_shape.py`: `SHAPE.__doc__ = f"{SHAPE.__doc__}\n\n{SHAPE.change_note()}"`.

- [ ] **Step 6: Render it on the reference page.** In `docs/source/_extensions/measurements_ref.py`, `_class_section`, insert the note between the heading and the table:

```python
    note = info_cls.change_note()
    out = [
        f".. _{_section_label(info_cls)}:",
        "",
        *_heading(linked_heading, "-"),
        *([note, ""] if note else []),
        info_cls.rst_table(
            header=("Column label", "Description"), use_headers=True
        ),
        "",
    ]
```

- [ ] **Step 7: Bump the version.** In `src/phenotypic/__init__.py:17`, set `__version__ = "0.20.0"`.

- [ ] **Step 8: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/schema/test_change_note.py tests/unit/docs/test_measurements_ref_extension.py tests/unit/sdk_/test_norm_migration.py tests/unit/schema -q`
Expected: all pass.

- [ ] **Step 9: Prove the guards can fail.** Remove the `*([note, ""] if note else []),` line and confirm `test_class_section_renders_the_change_note_above_the_table` FAILS. Change `append_rst_to_doc` to ignore `note` and confirm `test_note_renders_above_the_table_in_measurer_docs` FAILS. Revert both.

- [ ] **Step 10: Confirm the note renders as a highlighted block.** The orchestrator submits this as a **Slurm job** (`slurm-job` skill; `short`, default account, `--cpus-per-task=8 --mem=32G --time=01:30:00`, log under the worktree). It is never a local build. Notebooks are not executed:

```bash
PHENOTYPIC_DOCS_BUILD=1 uv run --group docs sphinx-build -b html \
  -j "$SLURM_CPUS_PER_TASK" -D nbsphinx_execute=never \
  docs/source docs/_build/size-note
```

Then read the generated HTML; an exit code of 0 is not evidence. Grep for "Changed in version 0.20.0" in `docs/_build/size-note/measurements_ref/measurements/index.html` **and** in the API pages `api_reference/api/phenotypic.measure.MeasureSize.html`, `…MeasureShape.html`, `…phenotypic.schema.SIZE.html` and `…SHAPE.html`: the three surfaces spec §7 promises. Confirm the rename table rendered as a `<table>`. The implementer does not wait for this job; the orchestrator runs it at the phase-1 gate.

- [ ] **Step 11: Lint, type-check, commit**

```bash
uv run ruff check --fix src/phenotypic/schema/_change_notes.py src/phenotypic/schema/_measurement_info.py src/phenotypic/schema/_size.py src/phenotypic/schema/_shape.py docs/source/_extensions/measurements_ref.py src/phenotypic/__init__.py tests/unit/schema/test_change_note.py tests/unit/docs/test_measurements_ref_extension.py tests/unit/sdk_/test_norm_migration.py
uv run mypy src/phenotypic/schema
git add src/phenotypic/schema docs/source/_extensions/measurements_ref.py src/phenotypic/__init__.py tests/unit/schema/test_change_note.py tests/unit/docs/test_measurements_ref_extension.py tests/unit/sdk_/test_norm_migration.py
git commit -m "feat(schema): highlighted 0.20.0 change note for the size/shape split; bump to 0.20.0"
```

---

### Task 7: Code consumers — prefabs, GUI defaults, bundled data, scripts, docstrings

**Mapping used by this task and Task 8.** Apply it by word boundary. Every other `Shape_*` name is unchanged.

| Old | New |
|---|---|
| `Shape_Area` / `SHAPE.AREA` | `Size_Area` / `SIZE.AREA` |
| `Shape_Perimeter` / `SHAPE.PERIMETER` | `Size_Perimeter` / `SIZE.PERIMETER` |
| `Shape_ConvexArea` / `SHAPE.CONVEX_AREA` | `Size_ConvexArea` / `SIZE.CONVEX_AREA` |
| `Shape_BboxArea` / `SHAPE.BBOX_AREA` | `Size_BboxArea` / `SIZE.BBOX_AREA` |
| `Shape_MajorAxisLength` / `SHAPE.MAJOR_AXIS_LENGTH` | `Size_MajorAxisLength` / `SIZE.MAJOR_AXIS_LENGTH` |
| `Shape_MinorAxisLength` / `SHAPE.MINOR_AXIS_LENGTH` | `Size_MinorAxisLength` / `SIZE.MINOR_AXIS_LENGTH` |
| `Shape_MaxRadius` / `SHAPE.MAX_RADIUS` | `Size_InscribedRadius` / `SIZE.INSCRIBED_RADIUS` |
| `Shape_MeanRadius` / `SHAPE.MEAN_RADIUS` | `Shape_MeanBoundaryDist` / `SHAPE.MEAN_BOUNDARY_DIST` |
| `Shape_MedianRadius` / `SHAPE.MEDIAN_RADIUS` | `Shape_MedianBoundaryDist` / `SHAPE.MEDIAN_BOUNDARY_DIST` |

A mechanical rewrite script, `/tmp/size_rename.pl`:

```perl
#!/usr/bin/perl -pi
s/\bShape_Area\b/Size_Area/g;
s/\bShape_Perimeter\b/Size_Perimeter/g;
s/\bShape_ConvexArea\b/Size_ConvexArea/g;
s/\bShape_BboxArea\b/Size_BboxArea/g;
s/\bShape_MajorAxisLength\b/Size_MajorAxisLength/g;
s/\bShape_MinorAxisLength\b/Size_MinorAxisLength/g;
s/\bShape_MaxRadius\b/Size_InscribedRadius/g;
s/\bShape_MeanRadius\b/Shape_MeanBoundaryDist/g;
s/\bShape_MedianRadius\b/Shape_MedianBoundaryDist/g;
s/\bSHAPE\.AREA\b/SIZE.AREA/g;
s/\bSHAPE\.PERIMETER\b/SIZE.PERIMETER/g;
s/\bSHAPE\.CONVEX_AREA\b/SIZE.CONVEX_AREA/g;
s/\bSHAPE\.BBOX_AREA\b/SIZE.BBOX_AREA/g;
s/\bSHAPE\.MAJOR_AXIS_LENGTH\b/SIZE.MAJOR_AXIS_LENGTH/g;
s/\bSHAPE\.MINOR_AXIS_LENGTH\b/SIZE.MINOR_AXIS_LENGTH/g;
s/\bSHAPE\.MAX_RADIUS\b/SIZE.INSCRIBED_RADIUS/g;
s/\bSHAPE\.MEAN_RADIUS\b/SHAPE.MEAN_BOUNDARY_DIST/g;
s/\bSHAPE\.MEDIAN_RADIUS\b/SHAPE.MEDIAN_BOUNDARY_DIST/g;
```

**Never run it on:**
- `src/phenotypic/schema/_change_notes.py`, `_size.py` or `_shape.py` (they name the retired columns on purpose);
- `tests/unit/measure/test_size_consolidation_equivalence.py`, `test_measure_shape.py` or `test_change_note.py`;
- the Task 1 baseline parquet;
- `docs/superpowers/**`;
- historical fixtures under `tests/**/_golden*/`.

**Files:**
- Modify: the prefabs `src/phenotypic/prefab/{_heavy_watershed_pipeline,_grid_section_pipeline,_heavy_otsu_pipeline,_round_peaks_pipeline,_heavy_round_peaks_pipeline,_filamentous_fungi_pipeline}.py` and the docstring examples in `src/phenotypic/abc_/_prefab_pipeline.py`
- Modify: `src/phenotypic/_gui/analysis/_callbacks.py:97-120`
- Modify: `src/phenotypic/data/meas/area_meas.csv`, `src/phenotypic/data/meas/all_meas.csv`
- Modify: `scripts/capture_gui_tutorial_screenshots.py` (lines ~189, ~195, ~1330-1343), `scripts/make_measurement_example_images.py` (lines ~28 and ~50)
- Modify: `src/` docstrings and doctests found by the grep in Step 5
- Test: `tests/unit/prefab/test_prefab_measures_size.py` (create), `tests/unit/gui/analysis/test_analysis_defaults_use_size_area.py` (create)

**Interfaces:**
- Consumes: the `SIZE` members from Task 3, and the mapping above.

- [ ] **Step 1: Write the failing tests.** Create `tests/unit/prefab/test_prefab_measures_size.py`:

```python
"""Every prefab that measures shape also measures size (area and radii moved there)."""

from __future__ import annotations

import pytest

import phenotypic.prefab as prefab
from phenotypic.measure import MeasureShape, MeasureSize


def _prefab_classes():
    for name in prefab.__all__:
        cls = getattr(prefab, name)
        if isinstance(cls, type):
            yield pytest.param(cls, id=name)


@pytest.mark.parametrize("cls", list(_prefab_classes()))
def test_prefab_with_measure_shape_also_has_measure_size(cls):
    kinds = {type(m) for m in cls().meas.values()}  # `meas` returns a dict copy
    if MeasureShape in kinds:
        assert MeasureSize in kinds, f"{cls.__name__} measures shape but not size"
```

Seven prefab classes are exported. Six carry `MeasureShape`; `SpImagerPipeline` already has only `MeasureSize` and passes.

Create `tests/unit/gui/analysis/test_analysis_defaults_use_size_area.py`:

```python
from phenotypic._gui.analysis import _callbacks
from phenotypic.schema import SIZE


def test_analysis_defaults_target_size_area():
    tables = (_callbacks._FILTER_DEFAULTS, _callbacks._EDGE_DEFAULTS, _callbacks._MODEL_DEFAULTS)
    ons = [params["on"] for table in tables for params in table.values() if "on" in params]
    assert ons and set(ons) == {str(SIZE.AREA)}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/prefab/test_prefab_measures_size.py tests/unit/gui/analysis/test_analysis_defaults_use_size_area.py -q`
Expected: 6 prefab cases FAIL, and the GUI test FAILS (`Shape_Area`).

- [ ] **Step 3: Prefabs.** In each of the six prefab files, add `MeasureSize` to the `from phenotypic.measure import ...` line. In each `meas = [` list, insert `MeasureSize(),` as the first entry; in `_grid_section_pipeline.py`'s dict, insert `"MeasureSize": MeasureSize(),` before `"MeasureShape"`. Update each class docstring's `Measurements:` line to begin with `MeasureSize,`. In `abc_/_prefab_pipeline.py`'s two docstring examples, import `MeasureSize` and add `MeasureSize()` / `custom.add(MeasureSize())` next to the `MeasureShape` lines.

- [ ] **Step 4: GUI defaults.** In `_gui/analysis/_callbacks.py`, replace each of the five `"on": "Shape_Area",` with `"on": str(SIZE.AREA),` and add `SIZE` to the module's `phenotypic.schema` import. Then run the `gui-tutorial-capture` skill's ledger check (`FEATURES.md`/`WORKFLOWS.md`). A changed default is not new chrome, so the expected outcome is no ledger change; confirm it and say so in the commit body.

- [ ] **Step 5: Mechanical rewrite of `src/`, data and scripts.**

```bash
grep -rlE "\bShape_(Area|Perimeter|ConvexArea|BboxArea|MajorAxisLength|MinorAxisLength|MaxRadius|MeanRadius|MedianRadius)\b|\bSHAPE\.(AREA|PERIMETER|CONVEX_AREA|BBOX_AREA|MAJOR_AXIS_LENGTH|MINOR_AXIS_LENGTH|MAX_RADIUS|MEAN_RADIUS|MEDIAN_RADIUS)\b" src scripts \
  | grep -v -E "schema/_(change_notes|size|shape)\.py$" > /tmp/size_rename_targets.txt
cat /tmp/size_rename_targets.txt
xargs perl -pi /tmp/size_rename.pl < /tmp/size_rename_targets.txt
```

Expected targets include the two CSVs, `schema/_measurement_info.py` (doctests), the growth-model and outlier analyzers, `sdk_/_metadata_helpers.py`, `_gui/_shared/_measurement_tint.py`, `_cli/_cli_output_manager.py`, `schema/_{linear_lag,linear_cap_and_lag,log_growth}_model.py`, `schema/CLAUDE.md` and both scripts. Then fix the imports: `uv run ruff check --select F401,F821 $(grep '\.py$' /tmp/size_rename_targets.txt)`. Every F821 `SIZE` needs `SIZE` added to that file's `phenotypic.schema` import, and every F401 `SHAPE` is removed with `--fix`.

- [ ] **Step 6: Hand-fix what the rewrite cannot know.**
  - `scripts/make_measurement_example_images.py`: the rewrite changed the docstring and title text to `Size_Area`. **Keep** `dest = _OUT / "shape" / "area.png"`, because `SIZE.AREA` still points at `image="shape/area.png"`. Rename the function from `_shape_area` to `_size_area` (and its call site).
  - `scripts/capture_gui_tutorial_screenshots.py`: the scatter roles are now `[str(SIZE.PERIMETER), str(SIZE.AREA)]`, which `PIPELINE_DOC` configures (line ~174: `"MeasureSize"`). Rewrite the comment block at ~1336-1340 to say that both come from `MeasureSize`, which `PIPELINE_DOC` configures. Then check whether the "verification run" the old comment mentions (`grep -n "verification" scripts/capture_gui_tutorial_screenshots.py`) configures `MeasureSize`. If it does not, bind `[str(SHAPE.SOLIDITY), str(SHAPE.CIRCULARITY)]` instead and say why in the comment. Also update the `"on": "Shape_Area"` recipe defaults at ~189 and ~195 (the rewrite does this; confirm).
  - `src/phenotypic/analysis/_error_cutoffs.py`: its `MEASUREMENT_PREFIXES` tuple is a prefix list, not a header list. Leave it as it is.
  - Read the diff of every docstring the rewrite touched (`git diff -- src/`) and fix any sentence that no longer reads correctly, e.g. "the Shape category's Size_Area".

- [ ] **Step 7: Run doctests on the rewritten modules**

```bash
uv run pytest --doctest-modules -q -p no:randomly $(grep '^src/.*\.py$' /tmp/size_rename_targets.txt) src/phenotypic/schema/_measurement_info.py
```

Expected: pass. Any doctest output that printed `Shape_Area` must now print `Size_Area`.

- [ ] **Step 8: Run the tests to verify they pass**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/prefab tests/unit/gui/analysis tests/unit/analysis tests/unit/data -q -m "not slow"`
Expected: all pass, except analysis tests that still spell `Shape_Area` as data (Task 8). List them.

- [ ] **Step 9: Lint and commit**

```bash
uv run ruff check --fix $(grep '\.py$' /tmp/size_rename_targets.txt) src/phenotypic/prefab src/phenotypic/abc_/_prefab_pipeline.py src/phenotypic/_gui/analysis/_callbacks.py tests/unit/prefab/test_prefab_measures_size.py tests/unit/gui/analysis/test_analysis_defaults_use_size_area.py
git add -A src scripts tests/unit/prefab/test_prefab_measures_size.py tests/unit/gui/analysis/test_analysis_defaults_use_size_area.py
git commit -m "refactor: point prefabs, GUI defaults, bundled data and docstrings at the Size columns"
```

---

### Task 8: Mechanical sweep of tests that use retired names as data

About 60 test files use `Shape_Area` (and similar) only as an arbitrary column name in synthetic frames.

**Files:**
- Modify: every file listed by Step 1 (expected to include `tests/unit/analysis/test_linear_softplus.py`, `test_log_growth_model.py`, `test_double_softplus.py`, `tests/unit/.../test_metadata_cluster_order.py`, `test_pipeline_analyze.py`, `tests/unit/gui/analysis/test_standalone_bundle.py`, `tests/unit/gui/results_viewer/test_scatter_grouping.py` and `tests/e2e/gui/test_scatter_tab.py`).

**Interfaces:**
- Consumes: the Task 7 mapping and `/tmp/size_rename.pl` (recreate it from Task 7 if `/tmp` was cleared).

- [ ] **Step 1: List the targets**

```bash
grep -rlE "\bShape_(Area|Perimeter|ConvexArea|BboxArea|MajorAxisLength|MinorAxisLength|MaxRadius|MeanRadius|MedianRadius)\b|\bSHAPE\.(AREA|PERIMETER|CONVEX_AREA|BBOX_AREA|MAJOR_AXIS_LENGTH|MINOR_AXIS_LENGTH|MAX_RADIUS|MEAN_RADIUS|MEDIAN_RADIUS)\b" tests \
  | grep -v -E "tests/unit/measure/test_(size_consolidation_equivalence|measure_shape)\.py$|tests/unit/schema/test_change_note\.py$|/_golden" > /tmp/size_rename_tests.txt
wc -l < /tmp/size_rename_tests.txt
```

- [ ] **Step 2: Rewrite and fix imports**

```bash
xargs perl -pi /tmp/size_rename.pl < /tmp/size_rename_tests.txt
uv run ruff check --select F401,F821 --fix $(cat /tmp/size_rename_tests.txt)
uv run ruff check --select F821 $(cat /tmp/size_rename_tests.txt)
```

For each remaining F821 (`SIZE` undefined), add `SIZE` to that file's `from phenotypic.schema import ...` line.

- [ ] **Step 3: Review the semantic cases by hand.** Open the diff for `test_scatter_grouping.py`, `test_scatter_tab.py`, `test_standalone_bundle.py` and `test_measurement_outputs.py`. Where a test's meaning depended on **two columns sharing the Shape category** (grouping by measurer or category), the rewrite has split them across Size and Shape. Restore the intent by choosing two columns that remain in one category (e.g. `SHAPE.CIRCULARITY` and `SHAPE.SOLIDITY`), rather than editing the expected grouping.

- [ ] **Step 4: Run the swept files**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest $(grep -v "tests/e2e" /tmp/size_rename_tests.txt) -q -m "not slow" -p no:randomly
```

Expected: all pass. Run any failure alone, and on main, before attributing it. The e2e file runs in the Task 10 regression.

- [ ] **Step 5: Confirm nothing is left**

Run the Step 1 grep again without `> /tmp/...`. Expected: no output.

- [ ] **Step 6: Commit**

```bash
uv run ruff check --fix $(cat /tmp/size_rename_tests.txt)
git add $(cat /tmp/size_rename_tests.txt)
git commit -m "test: sweep retired Shape_* column names to their Size/Shape successors"
```

---

### Task 9: User-facing documentation

**Files:**
- Modify: `docs/source/explanation/measurement_metrics_biological_meaning.md` (the "Shape Metrics (MeasureShape)" table, lines ~21-30)
- Modify: `docs/source/tutorials/notebooks/07_measuring_and_exporting.ipynb` (cell near source line 123)
- Modify: `docs/source/how_to/notebooks/fit_logistic_growth.ipynb`, `docs/source/how_to/notebooks/correct_edge_effects.ipynb`, `docs/source/explanation/notebooks/linear_softplus_model.ipynb`
- Modify: `docs/source/tutorials/gui/03_build_pipeline.md:40`

**Interfaces:**
- Consumes: the Task 7 mapping and `/tmp/size_rename.pl`; the `SIZE`/`SHAPE` descs from Tasks 3 and 5.

- [ ] **Step 1: Split the explanation table.** In `measurement_metrics_biological_meaning.md`, move the Area, Perimeter, ConvexArea, BboxArea, MajorAxisLength and MinorAxisLength rows, and the radius rows, into a new "Size Metrics (MeasureSize)" table placed **before** the Shape table. Add one row per new radius (InscribedRadius, MedianRadius, MeanRadius, RobustMeanRadius, MaxRadius), each **paraphrasing that member's `desc` in one sentence**. The InscribedRadius row must keep the elongation caveat. Rename the Shape rows MeanRadius/MedianRadius to MeanBoundaryDist/MedianBoundaryDist ("interior thickness, not a radius"). Put this block directly under the Size table heading:

```markdown
```{versionchanged} 0.20.0
Size magnitudes moved from `MeasureShape` to `MeasureSize`, and the radius
columns were rebuilt. `Size_MedianRadius`, `Size_MeanRadius` and
`Size_MaxRadius` are **not** the retired `Shape_*` columns of the same name;
see the rename table on the {ref}`SIZE reference <measurement-info-size>`.
```
```

(MyST directive syntax; check that `myst_parser` is enabled with `grep -n myst docs/source/conf.py`. If it is not, use `.. versionchanged::` in an `{eval-rst}` block.)

- [ ] **Step 2: Notebooks and the GUI tutorial**

```bash
perl -pi /tmp/size_rename.pl docs/source/tutorials/notebooks/07_measuring_and_exporting.ipynb docs/source/how_to/notebooks/fit_logistic_growth.ipynb docs/source/how_to/notebooks/correct_edge_effects.ipynb docs/source/explanation/notebooks/linear_softplus_model.ipynb docs/source/tutorials/gui/03_build_pipeline.md
```

Then read each diff. In `07_measuring_and_exporting`, where a cell lists MeasureShape's columns, move the size columns to a MeasureSize sentence, and make sure the notebook's pipeline adds `MeasureSize()` wherever it reads a `Size_*` column. In `03_build_pipeline.md:40`, rewrite the sentence so that MeasureSize is described as the size half (area, perimeter, radii) and MeasureShape as the form half. Stored cell outputs may still show old column names; they refresh at the next notebook execution, so do not hand-edit outputs.

- [ ] **Step 3: Validate the notebooks are still valid JSON**

```bash
for f in docs/source/tutorials/notebooks/07_measuring_and_exporting.ipynb docs/source/how_to/notebooks/fit_logistic_growth.ipynb docs/source/how_to/notebooks/correct_edge_effects.ipynb docs/source/explanation/notebooks/linear_softplus_model.ipynb; do uv run python -c "import json,sys; json.load(open(sys.argv[1]))" "$f" && echo "ok $f"; done
```

Expected: four `ok` lines.

- [ ] **Step 4: Confirm no retired name survives in `docs/source`**

Run: `grep -rnE "\bShape_(Area|Perimeter|ConvexArea|BboxArea|MajorAxisLength|MinorAxisLength|MaxRadius|MeanRadius|MedianRadius)\b" docs/source --include='*.md' --include='*.rst' --include='*.ipynb' | grep -v "_static/"`
Expected: only the deliberate mentions inside the `versionchanged` block.

- [ ] **Step 5: Commit**

```bash
git add docs/source
git commit -m "docs: document MeasureSize as the source of colony size; highlight the 0.20.0 rename"
```

---

### Task 10: Migration goldens, whole-branch verification, logic-validation re-run

**Files:**
- Modify: `tests/migration/_goldens/measure.MeasureShape.parquet`, `tests/migration/_goldens/measure.MeasureSize.parquet`

- [ ] **Step 0 (A6): Differential proof first.** Run `diff_migration_scenarios.py` (from Task 4b) against a detached main worktree and the branch tip. It must pass before any golden is touched.

- [ ] **Step 1: Recapture exactly FOUR goldens (A6, user decision).** Set `wanted` below to `{"measure.MeasureShape", "measure.MeasureSize", "measure.MeasureIntensity", "refine.KeepSectionLargest"}` and expect exactly those four files modified. The snippet shows the original two-golden form. The commit message must name each pre-existing drift from A6 separately from this change's column moves. Do **not** run `scripts/capture_migration_goldens.py`: it rewrites all 142 goldens and the frozen inputs.

```bash
uv run python - <<'EOF'
import sys
sys.path.insert(0, ".")
from tests.migration._runner import golden_path, run_scenario
from tests.migration._scenarios import build_scenarios

wanted = {"measure.MeasureShape", "measure.MeasureSize"}
done = set()
for scenario in build_scenarios():
    if scenario.scenario_id in wanted:
        run_scenario(scenario).save(golden_path(scenario))
        done.add(scenario.scenario_id)
assert done == wanted, done
print("recaptured", sorted(done))
EOF
git status --porcelain tests/migration/_goldens
```

Expected: exactly the two parquet files are modified. **`measure.MeasureIntensity.parquet` must not appear**; if it does, stop, because the decoupling changed values. Goldens are compared on Linux only (`_GOLDEN_PLATFORM`), and this capture ran on the local platform. The commit message must say which platform it was, as the branch's `ae9c5ddef` did.

- [ ] **Step 2: Check the recaptured goldens' columns**

```bash
uv run python -c "
import pandas as pd
s = pd.read_parquet('tests/migration/_goldens/measure.MeasureShape.parquet')
z = pd.read_parquet('tests/migration/_goldens/measure.MeasureSize.parquet')
assert not any(c in s.columns for c in ['Shape_Area','Shape_MaxRadius','Shape_MeanRadius']), list(s.columns)
assert {'Size_InscribedRadius','Size_MaxRadius','Size_RobustMeanRadius'} <= set(z.columns), list(z.columns)
print('ok', len(s.columns), len(z.columns))"
```

- [ ] **Step 3: Re-run the logic-validation script**

Run: `uv run --no-project --with numpy --with scipy python docs/superpowers/logic_validation_scripts/2026-09-24-size-measures-consolidation/radial_invariants.py`
Expected: `0 failure(s)`.

- [ ] **Step 4: Static checks over everything changed on the branch**

```bash
uv run mypy src/phenotypic
uv run ruff check $(git diff --name-only main...HEAD -- '*.py')
```

Expected: mypy reports no new errors compared with main, and ruff is clean.

- [ ] **Step 5: Full sharded regression, once.** Use the `run-phenotypic-test` skill and the committed batch script (`docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch`) via the `slurm-job` skill. Do not use `-x`, and do not use `-n auto` on a compute node. Record the pass/fail counts. Run each failure alone, and then on `main`, before attributing it to this branch.

- [ ] **Step 6: Commit the goldens**

```bash
git add tests/migration/_goldens/measure.MeasureShape.parquet tests/migration/_goldens/measure.MeasureSize.parquet
git commit -m "test(migration): recapture MeasureShape/MeasureSize goldens for the size/shape split

Recaptured only these two; MeasureIntensity is unchanged (the decoupling
preserves values). Captured on <platform>; goldens compare on Linux only."
```

- [ ] **Step 7: PR description.** It must contain the spec's §6 rename table **and** the same-name warning, so that downstream users see the hard break, and it ends with the attribution lines from the session instructions.

---

## Self-Review (done while writing)

- **Spec coverage:**

  | Spec section | Task |
  |---|---|
  | §3.1, §4 | T3 |
  | §3.2 | T5 |
  | §5.1 | T2 |
  | §5.2 | T3, T4, T5 |
  | §6 rename table | T6 note, T10 PR |
  | §7 | T6 (plus T9 in prose docs) |
  | §8 | T5 Step 5, T7, T9 |
  | §9 goldens, mutation proofs, producer-coupled tests | T1, T3–T6, T8, T10 |
  | §4.1 border edge | T3 Review Focus test |

- **Deliberate literal `Shape_*` strings** appear only in baseline-comparison tests, the change note, the enum descs ("Formerly reported as…") and the retired-name assertions. Each rewrite step's exclusion list names these files.
- **Types:** `convex_hull_area -> tuple[ConvexHull | None, float]` and `object_edt -> np.ndarray` are used identically in T3, T4 and T5. The `_measure_radial_profile` keys are the `SIZE` header strings in both the T3 implementation and its tests.
