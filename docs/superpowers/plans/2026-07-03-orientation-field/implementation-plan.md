# MeasureOrientationZones Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Regime-B `MeasureFeatures` operator `MeasureOrientationZones` that quantifies hyphal **concentration** (`R`), **turning** (`⟨|∇φ|⟩`), and **coherence** from the structure-tensor orientation field, reported **overall** and per **dense**/**sparse** growth zone, reusing the zone segmentation of `MeasureSymmetricZones`.

**Architecture:** First extract the colony-ness → zone-radii pipeline out of `MeasureSymmetricZones` into a shared, side-effect-free `_zone_segmentation.py` module (pure refactor, byte-identical, regression-guarded). Then add a self-contained `orientation_field()` structure-tensor helper. Then build the new operator on top of both: per object it obtains scalar zone radii + inoculum centre from `compute_zone_segmentation`, computes the orientation field over a **mask-free tile** (grid section when available, expanded crop otherwise), and aggregates coherence-weighted metrics over **radially defined** regions bounded by the symmetric radius, in both a `Radial` and a raw-`Mask` variant. Two Plotly figure surfaces follow the repo `FigureProvider` convention: `inspect()` (saveable primary, ships quiver + per-zone glyphs) and `dashboard()` (composed diagnostic adding a coherence heatmap).

**Tech Stack:** Python 3, numpy, `scipy.ndimage` (Gaussian derivatives, EDT), scikit-image (`regionprops`), pandas, pydantic v2 (keyword-only operation fields), Plotly (`graph_objects`, `make_subplots`), pytest. Package manager/runner: `uv` (never bare `python`/`pip`).

## Global Constraints

- **Runner:** all commands via `uv run …`. Never bare `python`/`pip`.
- **Worktree:** all work happens in `/Users/alex/Projects/PhenoTypic/.worktrees/orientation-field` on branch `orientation-field`. All paths below are repo-relative to that worktree root.
- **Operations are pydantic v2 models:** no `__init__`; parameters are annotated class-level fields; **keyword-only** construction; input normalization/guards go in `field_validator`s. Operations subclass the relevant ABC; algorithm body is `_operate(self, image) -> pd.DataFrame`.
- **Call convention:** `MeasureFeatures` subclasses run via `.measure(image)`; detectors via `.apply(image)`. Never `op(image)`.
- **Measurement columns are category-prefixed** and assembled as `f"{ENUM.category()}_{member.label}"`. This op's headers are `OrientZones_<Metric>-<Variant>-<Zone>` — a single underscore after the category, then Metric/Variant/Zone hyphen-joined so `header.split("_", 1)` peels the category and `.split("-")` yields `[Metric, Variant, Zone]`. `Metric ∈ {Concentration, Turning, Coherence}`, `Variant ∈ {Radial, Mask}`, `Zone ∈ {Overall, Dense, Sparse}` (18 columns + `Object_Label`).
- **Schema authoring rule (hard):** author `label` and `desc` only. **Leave `bio_desc=""` and `image=None`** — biological-relevance text is human-authored, never machine-generated. `ORIENTATION_ZONES` is a `DescriptiveTrait` (tier 2).
- **Docstrings:** Google-style everywhere; every doctest must be runnable on `load_synth_yeast_plate()` / `load_synth_filamentous_plate()` (both return a `GridImage` with objmap preloaded).
- **`inspect()` is reserved** across the codebase for the single *saveable* primary figure consumed by the CLI `--save-inspect` flag. The richer composed diagnostic is exposed as `dashboard()` (mirroring `AutoGridFinder.dashboard()`), returning a composed `go.Figure`.
- **Extraction discipline:** the §Task-1 refactor changes **no** observable behaviour of `MeasureSymmetricZones`; land it green (regression test passing) before building anything new.
- **Immutability/memory:** operations return copies; images are large — avoid unnecessary intermediate allocations; read accessors as `image.detect_mat[:]`, `image.gray[:]`, `image.objmap[:]`.

---

## File Structure

**New files**
- `src/phenotypic/measure/_zone_segmentation.py` — `ZoneSegmentation` dataclass, `ZoneSegmentationParams` dataclass, `compute_zone_segmentation(...)`, and the relocated pure segmentation helpers + module constants. Single responsibility: turn one detected object into its concentric zone geometry.
- `src/phenotypic/util/_orientation_field.py` — `orientation_field(intensity, sigma_d, sigma_i) -> (phi, coherence, grad_phi)`. Single responsibility: structure-tensor orientation field on a 2-D intensity tile.
- `src/phenotypic/schema/_orientation_zones.py` — `ORIENTATION_ZONES` header enum (18 members).
- `src/phenotypic/measure/_measure_orientation_zones.py` — `MeasureOrientationZones` operator + pure aggregation helpers + `inspect()` + `dashboard()` + transient `_OrientationZonesReport`.
- `tests/unit/measure/test_zone_segmentation_regression.py` — golden-equality regression guard for the extraction.
- `tests/unit/measure/test_orientation_field.py` — analytic-phantom tests for the `orientation_field` helper and the aggregation helpers.
- `tests/unit/measure/test_measure_orientation_zones.py` — operator behaviour, zone restriction, invariances, Radial-vs-Mask, edge cases, figure smoke tests.
- `tests/unit/measure/_golden/symmetric_zones_yeast.parquet`, `…_filamentous.parquet` — captured baselines (committed).

**Modified files**
- `src/phenotypic/measure/_measure_symmetric_zones.py` — relocate segmentation staticmethods + `_SymmetryIntermediates` into `_zone_segmentation.py`; `_compute_intermediates` becomes a thin delegator to `compute_zone_segmentation`. No column/inspect changes.
- `src/phenotypic/measure/__init__.py` — export `MeasureOrientationZones`.
- `src/phenotypic/schema/__init__.py` — export `ORIENTATION_ZONES`.

**Execution ordering / dependencies**
- **Task 1** (extraction) is a prerequisite for Task 5. It is independent of Tasks 2 and 3.
- **Task 2** (`orientation_field`) and **Task 3** (schema enum) are independent of Task 1 and of each other — can be done in any order / parallel.
- **Task 4** (operator core) depends on Tasks 1, 2, 3.
- **Task 5** (`inspect()`) depends on Task 4. **Task 6** (`dashboard()`) depends on Task 5.

---

### Task 1: Extract shared zone-segmentation helper (pure refactor + regression guard)

**Files:**
- Create: `src/phenotypic/measure/_zone_segmentation.py`
- Modify: `src/phenotypic/measure/_measure_symmetric_zones.py`
- Test: `tests/unit/measure/test_zone_segmentation_regression.py`
- Golden: `tests/unit/measure/_golden/symmetric_zones_yeast.parquet`, `…_filamentous.parquet`

**Interfaces:**
- Produces:
  - `@dataclass ZoneSegmentation` — the full per-object segmentation record: **all** fields currently on `_SymmetryIntermediates` (`label`, `bbox_slice`, `centroid_rc`, `density_profile`, `annulus_radii`, `core_radius`, `sholl_counts`, `angular_R_profile`, `angular_coverage`, `symmetric_radius`, `mean_expansion`, `max_expansion`, `obj_mask`, `dist_map`, `gray_crop`, `core_end_radius`, `dense_end_radius`, `sparse_end_radius`, `r_outer_full_per_angle`, `core_area`, `dense_area`, `sparse_area`, `colony_ness_profile`, `mean_profile`, `variance_profile`, `count_profile`, `I_core`, `I_agar`, `zones_computed`) **plus one new field** `centroid_global: tuple[float, float] = (0.0, 0.0)` (plate-frame inoculum centre; frame origin of `dist_map`/`obj_mask`/`gray_crop` = `(bbox_slice[0].start, bbox_slice[1].start)`).
  - `@dataclass(frozen=True) ZoneSegmentationParams` with fields (name/default identical to the current `MeasureSymmetricZones` pydantic fields): `n_annuli: int = 100`, `pelt_penalty: float = 5.0`, `symmetry_threshold: float = 4/6`, `n_angular_bins: int = 6`, `smoothing_window: int = 3`, `method: str = "distance"`, `extent_margin: float = 0.05`, `min_samples_per_ring: int = 5`, `tau_core: float = 0.9`, `tau_dense: float = 0.5`, `tau_sparse: float = 0.1`, `intensity_source: str = "gray"`.
  - `compute_zone_segmentation(image, prop, *, params: ZoneSegmentationParams) -> ZoneSegmentation`.
  - Relocated module-level pure functions (dropped `@staticmethod`, identical bodies): `distance_from_point`, `expand_slice_around_center`, `compute_radial_density_profile`, `find_core_radius`, `extract_mask_boundary`, `compute_sholl_angular_profile`, `find_symmetric_radius`, `compute_radial_expansion`, `per_angle_mask_envelope`, `build_theta_r_maps`, `accumulate_radial_profile`, `accumulate_mask_per_annulus`, `compute_colony_ness_profile`, `extract_zone_radii`, `compute_zone_areas`, and module constants `_N_ANGULAR_SECTORS = 360`, `_ZONE_RADIAL_SMOOTHING = 3`.
    **Note (15 helpers, not 14):** `_extract_mask_boundary` (currently `_measure_symmetric_zones.py:765`) is a 15th staticmethod that `_compute_sholl_angular_profile` (line ~840) and `_compute_radial_expansion` (line ~997) call via a hardcoded `MeasureSymmetricZones._extract_mask_boundary(...)` reference. It MUST be relocated too, and those two call sites rewritten to the bare module function `extract_mask_boundary(...)` — otherwise the relocated bodies reference a deleted/foreign symbol and create an import cycle or `AttributeError`. `min_boundary_per_annulus` (default 8) stays a hardcoded arg on `compute_sholl_angular_profile`, intentionally **not** promoted to `ZoneSegmentationParams` (matches current behaviour).

- [ ] **Step 1: Capture the golden baseline on the CURRENT (unrefactored) code**

Write `tests/unit/measure/test_zone_segmentation_regression.py` with a one-shot capture guarded by an env flag, then a comparison test:

```python
"""Regression guard: MeasureSymmetricZones output is byte-identical across the
zone-segmentation extraction refactor (Task 1 of the orientation-field plan)."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from phenotypic.data import load_synth_yeast_plate, load_synth_filamentous_plate
from phenotypic.measure import MeasureSymmetricZones

_GOLDEN_DIR = Path(__file__).parent / "_golden"
_CASES = {
    "yeast": load_synth_yeast_plate,
    "filamentous": load_synth_filamentous_plate,
}


def _measure(loader) -> pd.DataFrame:
    return MeasureSymmetricZones().measure(loader())


@pytest.mark.parametrize("name", sorted(_CASES))
def test_symmetric_zones_matches_golden(name):
    golden_path = _GOLDEN_DIR / f"symmetric_zones_{name}.parquet"
    assert golden_path.exists(), (
        f"missing golden {golden_path}; regenerate with "
        f"PHENOTYPIC_CAPTURE_GOLDEN=1 uv run pytest "
        f"tests/unit/measure/test_zone_segmentation_regression.py"
    )
    result = _measure(_CASES[name])
    golden = pd.read_parquet(golden_path)
    pd.testing.assert_frame_equal(result, golden)


@pytest.mark.skipif(
    os.environ.get("PHENOTYPIC_CAPTURE_GOLDEN") != "1",
    reason="golden capture only runs when PHENOTYPIC_CAPTURE_GOLDEN=1",
)
def test_capture_golden():
    _GOLDEN_DIR.mkdir(exist_ok=True)
    for name, loader in _CASES.items():
        _measure(loader).to_parquet(_GOLDEN_DIR / f"symmetric_zones_{name}.parquet")
```

- [ ] **Step 2: Generate the golden files from the current code, then commit them**

Run: `PHENOTYPIC_CAPTURE_GOLDEN=1 uv run pytest tests/unit/measure/test_zone_segmentation_regression.py::test_capture_golden -v`
Expected: PASS; two parquet files created under `tests/unit/measure/_golden/`.

Then verify the comparison test passes against the freshly captured baseline:
Run: `uv run pytest tests/unit/measure/test_zone_segmentation_regression.py -k matches_golden -v`
Expected: 2 PASS.

```bash
git add tests/unit/measure/test_zone_segmentation_regression.py tests/unit/measure/_golden/
git commit -m "test(measure): golden baseline for MeasureSymmetricZones before zone-segmentation extraction"
```

- [ ] **Step 3: Create `_zone_segmentation.py` — relocate constants, dataclass, and the 14 pure helpers**

Read `src/phenotypic/measure/_measure_symmetric_zones.py` and move, **verbatim in body**, into the new module:
1. Constants `_N_ANGULAR_SECTORS = 360` and `_ZONE_RADIAL_SMOOTHING = 3` (currently lines ~30–31).
2. The `_SymmetryIntermediates` dataclass (currently lines ~58–101) → rename to `ZoneSegmentation`; append the new field `centroid_global: tuple[float, float] = (0.0, 0.0)`.
3. The **15** `@staticmethod` helpers (currently at lines ~257, ~273, ~698, ~734, ~765 [`_extract_mask_boundary`], ~784, ~904, ~972, ~1010, ~1047, ~1085, ~1151, ~1171, ~1219, ~1285) → module-level functions with the leading underscore dropped from the public name where convenient, keeping bodies identical. Suggested public names: `distance_from_point`, `expand_slice_around_center`, `compute_radial_density_profile`, `find_core_radius`, `extract_mask_boundary`, `compute_sholl_angular_profile`, `find_symmetric_radius`, `compute_radial_expansion`, `per_angle_mask_envelope`, `build_theta_r_maps`, `accumulate_radial_profile`, `accumulate_mask_per_annulus`, `compute_colony_ness_profile`, `extract_zone_radii`, `compute_zone_areas`. **After relocating, grep the two callers of `_extract_mask_boundary` (inside `compute_sholl_angular_profile` and `compute_radial_expansion`) and rewrite `MeasureSymmetricZones._extract_mask_boundary(...)` → `extract_mask_boundary(...)`.**

Copy the exact imports these helpers need (numpy, `scipy.ndimage` funcs, `skimage.measure.regionprops`, `ruptures`/PELT dependency, `uniform_filter1d`, etc.) from the top of `_measure_symmetric_zones.py`.

Add the params dataclass:

```python
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ZoneSegmentationParams:
    """Parameters controlling the colony-ness → zone-radii pipeline."""
    n_annuli: int = 100
    pelt_penalty: float = 5.0
    symmetry_threshold: float = 4 / 6
    n_angular_bins: int = 6
    smoothing_window: int = 3
    method: str = "distance"
    extent_margin: float = 0.05
    min_samples_per_ring: int = 5
    tau_core: float = 0.9
    tau_dense: float = 0.5
    tau_sparse: float = 0.1
    intensity_source: str = "gray"
```

- [ ] **Step 4: Move the pipeline body into `compute_zone_segmentation`**

Copy the **entire body** of the current `MeasureSymmetricZones._compute_intermediates` (lines ~301–587) into a new module-level function in `_zone_segmentation.py`:

```python
def compute_zone_segmentation(image, prop=None, *, params: ZoneSegmentationParams) -> ZoneSegmentation:
    """Compute concentric zone geometry (core/dense/sparse radii) for one object.

    Pure relocation of MeasureSymmetricZones._compute_intermediates. Reads
    ``params.<name>`` where the method read ``self.<name>``; calls the relocated
    module-level helpers; also records ``centroid_global`` (plate-frame inoculum
    centre) on the returned record.
    """
    ...
```

Mechanical edits while copying:
- Replace every `self.<param>` read with `params.<param>` (the 12 params above).
- Replace every `self._<helper>(...)` call with the relocated module function name.
- Replace `_SymmetryIntermediates(...)` constructions with `ZoneSegmentation(...)`.
- At the point where the expanded crop is built (current Stage 9b, the `center_global` variable), pass `centroid_global=tuple(center_global)` into the final `ZoneSegmentation(...)`. For the early-exit branches (tiny object, collapsed symmetric radius), set `centroid_global` to the object's plate-frame centroid (`(bbox_slice[0].start + centroid_rc[0], bbox_slice[1].start + centroid_rc[1])`) so it is always populated.

- [ ] **Step 5: Reduce `_compute_intermediates` to a thin delegator**

In `_measure_symmetric_zones.py`:
1. Delete the relocated dataclass, constants, and 15 staticmethods (including `_extract_mask_boundary`).
2. Add imports:
   ```python
   from phenotypic.measure._zone_segmentation import (
       ZoneSegmentation,
       ZoneSegmentationParams,
       compute_zone_segmentation,
   )
   _SymmetryIntermediates = ZoneSegmentation  # back-compat alias for the rest of this module
   ```
3. Replace the whole `_compute_intermediates` method body with:
   ```python
   def _compute_intermediates(self, image, object_label=None, prop=None) -> ZoneSegmentation:
       return compute_zone_segmentation(image, prop, params=self._zone_params())

   def _zone_params(self) -> ZoneSegmentationParams:
       return ZoneSegmentationParams(
           n_annuli=self.n_annuli,
           pelt_penalty=self.pelt_penalty,
           symmetry_threshold=self.symmetry_threshold,
           n_angular_bins=self.n_angular_bins,
           smoothing_window=self.smoothing_window,
           method=self.method,
           extent_margin=self.extent_margin,
           min_samples_per_ring=self.min_samples_per_ring,
           tau_core=self.tau_core,
           tau_dense=self.tau_dense,
           tau_sparse=self.tau_sparse,
           intensity_source=self.intensity_source,
       )
   ```
   Note: the current `_compute_intermediates` resolves `prop` from `object_label` when `prop is None` (Stage 1). Preserve that: if the relocated `compute_zone_segmentation` expects a `prop`, keep the object-resolution logic (regionprops lookup by label) as the first lines of `compute_zone_segmentation` so the delegator can pass `prop=None`.

- [ ] **Step 6: Verify byte-identical behaviour**

Run: `uv run pytest tests/unit/measure/test_zone_segmentation_regression.py -k matches_golden -v`
Expected: 2 PASS (output unchanged after extraction).

Run the existing symmetric-zones suite to confirm nothing else moved:
Run: `uv run pytest tests/unit/measure/test_measure_symmetric_zones.py -v`
Expected: all PASS.

- [ ] **Step 7: Type-check and lint the touched files**

Run: `uv run mypy src/phenotypic/measure/_zone_segmentation.py src/phenotypic/measure/_measure_symmetric_zones.py`
Expected: no new errors.
Run: `uv run ruff check --fix src/phenotypic/measure/_zone_segmentation.py src/phenotypic/measure/_measure_symmetric_zones.py`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add src/phenotypic/measure/_zone_segmentation.py src/phenotypic/measure/_measure_symmetric_zones.py
git commit -m "refactor(measure): extract shared zone-segmentation helper from MeasureSymmetricZones"
```

---

### Task 2: `orientation_field()` structure-tensor helper

**Files:**
- Create: `src/phenotypic/util/_orientation_field.py`
- Test: `tests/unit/measure/test_orientation_field.py`

**Interfaces:**
- Produces: `orientation_field(intensity: np.ndarray, sigma_d: float = 1.5, sigma_i: float = 4.0, *, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]` returning `(phi, coherence, grad_phi)`, each shape `intensity.shape`, dtype float64. `phi ∈ (-π/2, π/2]` (radians), `coherence ∈ [0, 1]`, `grad_phi ≥ 0` (rad/px, doubled-angle / π-safe).

- [ ] **Step 1: Write failing tests for the helper (analytic phantoms)**

```python
"""Analytic-phantom tests for the structure-tensor orientation field."""
from __future__ import annotations

import numpy as np
import pytest

from phenotypic.util._orientation_field import orientation_field


def _parallel_stripes(n=64, period=8.0):
    yy, xx = np.mgrid[0:n, 0:n]
    return np.sin(2 * np.pi * yy / period).astype(np.float64)


def test_parallel_bundle_is_coherent_and_non_turning():
    phi, coh, grad = orientation_field(_parallel_stripes(), sigma_d=1.5, sigma_i=4.0)
    interior = (slice(12, 52), slice(12, 52))
    assert coh[interior].mean() > 0.8          # highly coherent
    assert grad[interior].mean() < 1e-2        # orientation ~constant -> no turning


def test_isotropic_noise_is_incoherent():
    rng = np.random.default_rng(0)
    field = rng.standard_normal((64, 64))
    _, coh, _ = orientation_field(field, sigma_d=1.5, sigma_i=6.0)
    assert coh[16:48, 16:48].mean() < 0.35     # no dominant orientation


def test_output_shapes_and_ranges():
    phi, coh, grad = orientation_field(_parallel_stripes())
    for a in (phi, coh, grad):
        assert a.shape == (64, 64)
    assert np.all(coh >= -1e-9) and np.all(coh <= 1 + 1e-9)
    assert np.all(grad >= 0)
    assert np.all(phi <= np.pi / 2 + 1e-9) and np.all(phi > -np.pi / 2 - 1e-9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/measure/test_orientation_field.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'phenotypic.util._orientation_field'`.

- [ ] **Step 3: Implement the helper**

```python
"""Structure-tensor orientation field on a 2-D intensity tile.

Derivation: docs/superpowers/explain/2026-07-03-gradient-to-orientation-field-metrics.md
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def orientation_field(
    intensity: np.ndarray,
    sigma_d: float = 1.5,
    sigma_i: float = 4.0,
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the structure-tensor orientation field of an intensity tile.

    Args:
        intensity: 2-D intensity array (e.g. ``image.detect_mat[:]`` crop).
        sigma_d: Gaussian-derivative (gradient) scale in pixels, ~ hypha width.
        sigma_i: Structure-tensor integration scale in pixels.
        eps: Numerical floor for the coherence denominator.

    Returns:
        ``(phi, coherence, grad_phi)`` — orientation in radians in
        ``(-pi/2, pi/2]``, coherence in ``[0, 1]``, and the doubled-angle
        (pi-safe) orientation-gradient magnitude ``|grad phi|`` in rad/px.
    """
    intensity = np.asarray(intensity, dtype=np.float64)
    # Gaussian-derivative gradients at the derivative scale. scipy `order` is
    # per-axis (axis0=rows=y, axis1=cols=x): (0,1) -> d/dx, (1,0) -> d/dy.
    Ix = gaussian_filter(intensity, sigma_d, order=(0, 1))
    Iy = gaussian_filter(intensity, sigma_d, order=(1, 0))
    # Structure-tensor components smoothed at the integration scale.
    Jxx = gaussian_filter(Ix * Ix, sigma_i)
    Jyy = gaussian_filter(Iy * Iy, sigma_i)
    Jxy = gaussian_filter(Ix * Iy, sigma_i)
    # Dominant orientation via the doubled angle.
    phi = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy)
    # Coherence (anisotropy) in [0, 1].
    coherence = np.sqrt((Jyy - Jxx) ** 2 + 4.0 * Jxy ** 2) / (Jxx + Jyy + eps)
    coherence = np.clip(coherence, 0.0, 1.0)
    # |grad phi| via the doubled-angle representation (pi-safe): the field is
    # 2phi, and |grad phi| = 1/2 * |grad(2phi)| recovered from cos2phi/sin2phi.
    c2, s2 = np.cos(2.0 * phi), np.sin(2.0 * phi)
    gc_y, gc_x = np.gradient(c2)
    gs_y, gs_x = np.gradient(s2)
    grad_phi = 0.5 * np.sqrt(gc_x ** 2 + gc_y ** 2 + gs_x ** 2 + gs_y ** 2)
    return phi, coherence, grad_phi
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/measure/test_orientation_field.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Type-check, lint, commit**

Run: `uv run mypy src/phenotypic/util/_orientation_field.py`
Run: `uv run ruff check --fix src/phenotypic/util/_orientation_field.py tests/unit/measure/test_orientation_field.py`
```bash
git add src/phenotypic/util/_orientation_field.py tests/unit/measure/test_orientation_field.py
git commit -m "feat(util): add structure-tensor orientation_field helper"
```

---

### Task 3: `ORIENTATION_ZONES` schema enum

**Files:**
- Create: `src/phenotypic/schema/_orientation_zones.py`
- Modify: `src/phenotypic/schema/__init__.py`
- Test: fold assertions into `tests/unit/measure/test_measure_orientation_zones.py` (created in Task 4); a standalone import check runs here.

**Interfaces:**
- Produces: `ORIENTATION_ZONES` — a `DescriptiveTrait` enum with `category() == "OrientZones"` and 18 members whose `.value` (header) is `OrientZones_<Metric>-<Variant>-<Zone>`. Inherits `get_labels()` / `get_headers()` from `MeasurementInfo`.

- [ ] **Step 1: Write the enum module**

Model it on `src/phenotypic/schema/_symmetric_zones.py`. Author `label`/`desc` only; leave `bio_desc`/`image` at their defaults.

```python
"""Public header schema for MeasureOrientationZones (category ``OrientZones``).

Header pattern: ``OrientZones_<Metric>-<Variant>-<Zone>`` — single underscore
after the category, then Metric/Variant/Zone hyphen-joined. Metric in
{Concentration, Turning, Coherence}; Variant in {Radial, Mask}; Zone in
{Overall, Dense, Sparse}.
"""
from __future__ import annotations

from ._measurement_info import Entry
from ._tiers import DescriptiveTrait

_METRIC_DESC = {
    "Concentration": (
        "Coherence-weighted resultant length R of the doubled-angle "
        "orientation field over the {variant} selector of the {zone} region. "
        "Dimensionless in [0, 1]; 1 = perfectly aligned hyphae, 0 = isotropic. "
        "NaN when the summed coherence over the selector is ~0 or the zone has "
        "zero width."
    ),
    "Turning": (
        "Coherence-weighted mean orientation-gradient magnitude <|grad phi|> "
        "over the {variant} selector of the {zone} region, in radians per pixel "
        "(radians per micron when a pixel scale is set). Higher values indicate "
        "curving/fanning hyphae; ~0 indicates straight parallel growth."
    ),
    "Coherence": (
        "Mean structure-tensor coherence C over the {variant} selector of the "
        "{zone} region. Dimensionless in [0, 1]; a confidence/QC readout for how "
        "well orientation is defined there (low where texture is isotropic)."
    ),
}
_ZONE_MEANING = {
    "Overall": "the full symmetric disk (0 .. symmetric_radius)",
    "Dense": "the dense ring (core_end .. dense_end radii)",
    "Sparse": "the sparse ring (dense_end .. sparse_end radii)",
}
_VARIANT_MEANING = {
    "Radial": "all tile pixels in the radial region (mask-free)",
    "Mask": "the radial region intersected with the detected object mask",
}


def _desc(metric: str, variant: str, zone: str) -> str:
    return (
        _METRIC_DESC[metric].format(variant=_VARIANT_MEANING[variant], zone=_ZONE_MEANING[zone])
    )


class ORIENTATION_ZONES(DescriptiveTrait):
    """Per-zone hyphal orientation traits (concentration, turning, coherence).

    Computed from the structure-tensor orientation field over a mask-free tile,
    aggregated coherence-weighted over radially-defined zones bounded by the
    symmetric radius, in both a ``Radial`` (all tile pixels) and a raw ``Mask``
    variant. See :class:`MeasureOrientationZones` for parameters and method.
    """

    @classmethod
    def category(cls) -> str:
        return "OrientZones"

    CONCENTRATION_RADIAL_OVERALL = Entry("Concentration-Radial-Overall", _desc("Concentration", "Radial", "Overall"))
    CONCENTRATION_RADIAL_DENSE = Entry("Concentration-Radial-Dense", _desc("Concentration", "Radial", "Dense"))
    CONCENTRATION_RADIAL_SPARSE = Entry("Concentration-Radial-Sparse", _desc("Concentration", "Radial", "Sparse"))
    CONCENTRATION_MASK_OVERALL = Entry("Concentration-Mask-Overall", _desc("Concentration", "Mask", "Overall"))
    CONCENTRATION_MASK_DENSE = Entry("Concentration-Mask-Dense", _desc("Concentration", "Mask", "Dense"))
    CONCENTRATION_MASK_SPARSE = Entry("Concentration-Mask-Sparse", _desc("Concentration", "Mask", "Sparse"))
    TURNING_RADIAL_OVERALL = Entry("Turning-Radial-Overall", _desc("Turning", "Radial", "Overall"))
    TURNING_RADIAL_DENSE = Entry("Turning-Radial-Dense", _desc("Turning", "Radial", "Dense"))
    TURNING_RADIAL_SPARSE = Entry("Turning-Radial-Sparse", _desc("Turning", "Radial", "Sparse"))
    TURNING_MASK_OVERALL = Entry("Turning-Mask-Overall", _desc("Turning", "Mask", "Overall"))
    TURNING_MASK_DENSE = Entry("Turning-Mask-Dense", _desc("Turning", "Mask", "Dense"))
    TURNING_MASK_SPARSE = Entry("Turning-Mask-Sparse", _desc("Turning", "Mask", "Sparse"))
    COHERENCE_RADIAL_OVERALL = Entry("Coherence-Radial-Overall", _desc("Coherence", "Radial", "Overall"))
    COHERENCE_RADIAL_DENSE = Entry("Coherence-Radial-Dense", _desc("Coherence", "Radial", "Dense"))
    COHERENCE_RADIAL_SPARSE = Entry("Coherence-Radial-Sparse", _desc("Coherence", "Radial", "Sparse"))
    COHERENCE_MASK_OVERALL = Entry("Coherence-Mask-Overall", _desc("Coherence", "Mask", "Overall"))
    COHERENCE_MASK_DENSE = Entry("Coherence-Mask-Dense", _desc("Coherence", "Mask", "Dense"))
    COHERENCE_MASK_SPARSE = Entry("Coherence-Mask-Sparse", _desc("Coherence", "Mask", "Sparse"))
```

Import paths (confirmed against `_symmetric_zones.py`): `from ._measurement_info import Entry`, `from ._tiers import DescriptiveTrait`. Calling the module-level `_desc(...)` in the class body is ordinary Python — the calls execute before the enum metaclass runs, so this is legal (no need to inline the strings).

- [ ] **Step 2: Export it**

In `src/phenotypic/schema/__init__.py`, add `from ._orientation_zones import ORIENTATION_ZONES` next to the `SYMMETRIC_ZONES` import and add `"ORIENTATION_ZONES"` to `__all__`.

- [ ] **Step 3: Smoke-verify header assembly and count**

Run:
```bash
uv run python -c "
from phenotypic.schema import ORIENTATION_ZONES as OZ
hs = OZ.get_headers()
assert len(hs) == 18, len(hs)
assert 'OrientZones_Concentration-Radial-Overall' in hs
assert 'OrientZones_Coherence-Mask-Sparse' in hs
for h in hs:
    cat, rest = h.split('_', 1)
    parts = rest.split('-')
    assert cat == 'OrientZones' and len(parts) == 3, h
print('ok', len(hs))
"
```
Expected: `ok 18`.

- [ ] **Step 4: Type-check, lint, commit**

Run: `uv run mypy src/phenotypic/schema/_orientation_zones.py`
Run: `uv run ruff check --fix src/phenotypic/schema/_orientation_zones.py src/phenotypic/schema/__init__.py`
```bash
git add src/phenotypic/schema/_orientation_zones.py src/phenotypic/schema/__init__.py
git commit -m "feat(schema): add ORIENTATION_ZONES header enum (OrientZones)"
```

---

### Task 4: `MeasureOrientationZones` operator core (metrics)

**Files:**
- Create: `src/phenotypic/measure/_measure_orientation_zones.py`
- Modify: `src/phenotypic/measure/__init__.py`
- Test: `tests/unit/measure/test_measure_orientation_zones.py`

**Interfaces:**
- Consumes: `compute_zone_segmentation`, `ZoneSegmentationParams`, `ZoneSegmentation`, `distance_from_point`, `expand_slice_around_center` (Task 1); `orientation_field` (Task 2); `ORIENTATION_ZONES` (Task 3); `MeasureFeatures` ABC; `OBJECT` label enum from `phenotypic.schema`.
- Produces:
  - `class MeasureOrientationZones(MeasureFeatures, FigureProvider)` with keyword-only pydantic fields: `intensity_source: str = "detect_mat"`, `sigma_d: float = 1.5`, `sigma_i: float = 4.0`, `quiver_block: int = 12`, plus the 12 zone passthrough fields (same names/defaults as `ZoneSegmentationParams`, except `intensity_source` which the op reuses for both the field and the tensor input).
  - `_operate(self, image) -> pd.DataFrame` (18 columns + `Object_Label`).
  - Instance helpers: `_prep(image) -> (props, label2section)`; `_iter_object_fields(image, props, label2section)` (the single heavy-compute generator, reused by `_operate` and `_coherence_canvas`); `_resolve_tile(...)`; `_fill_metrics(...)`; `_coherence_canvas(image, downsample=4)` (dashboard-only, recompute-and-discard).
  - Pure module-level helpers (unit-tested): `zone_selector(dist_map, r_lo, r_hi, obj_mask, variant)`, `aggregate_orientation(phi, coherence, grad_phi, selector, eps)`, `_downsample_quiver(phi, coherence, block)`, `_resultant_direction(phi, coherence, selector)`.
  - **Lean compact cache** `self._cache[label]`: scalars (`centroid_global`, `centre`, `radii`, `zones_computed`), the block-resolution `quiver`, and the scalar `per_zone` map — **no** full-res arrays, **no** `seg` dataclass.

- [ ] **Step 1: Write failing tests for the pure aggregation helpers**

```python
"""Behaviour tests for MeasureOrientationZones."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phenotypic.data import load_synth_yeast_plate, load_synth_filamentous_plate
from phenotypic.schema import ORIENTATION_ZONES
from phenotypic.measure import MeasureOrientationZones
from phenotypic.measure._measure_orientation_zones import (
    aggregate_orientation,
    zone_selector,
)


def test_aggregate_parallel_field_high_R_zero_turning():
    n = 40
    phi = np.full((n, n), 0.3)          # constant orientation
    coh = np.ones((n, n))
    grad = np.zeros((n, n))
    sel = np.ones((n, n), dtype=bool)
    R, turning, coh_mean = aggregate_orientation(phi, coh, grad, sel, eps=1e-9)
    assert R == pytest.approx(1.0, abs=1e-6)
    assert turning == pytest.approx(0.0, abs=1e-9)
    assert coh_mean == pytest.approx(1.0, abs=1e-9)


def test_aggregate_zero_coherence_is_nan():
    n = 20
    out = aggregate_orientation(
        np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n)),
        np.ones((n, n), dtype=bool), eps=1e-9,
    )
    assert all(np.isnan(v) for v in out)


def test_aggregate_empty_selector_is_nan():
    n = 20
    out = aggregate_orientation(
        np.zeros((n, n)), np.ones((n, n)), np.zeros((n, n)),
        np.zeros((n, n), dtype=bool), eps=1e-9,
    )
    assert all(np.isnan(v) for v in out)


def test_zone_selector_radial_vs_mask():
    n = 21
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    dist = np.hypot(yy - c, xx - c)
    obj = dist < 6                      # imperfect mask: only inner disk
    radial = zone_selector(dist, 0.0, 8.0, obj, "Radial")
    masked = zone_selector(dist, 0.0, 8.0, obj, "Mask")
    assert radial.sum() > masked.sum()          # mask carves out the ring 6..8
    assert np.array_equal(masked, radial & obj)


def test_zone_restriction_inner_vs_outer_orientation():
    # Inner disk oriented one way, outer ring another -> per-zone R directions differ.
    n = 61
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    dist = np.hypot(yy - c, xx - c)
    phi = np.where(dist < 15, 0.0, np.pi / 2 - 1e-6)
    coh = np.ones((n, n))
    grad = np.zeros((n, n))
    obj = np.ones((n, n), dtype=bool)
    inner = aggregate_orientation(phi, coh, grad, zone_selector(dist, 0.0, 15.0, obj, "Radial"))
    outer = aggregate_orientation(phi, coh, grad, zone_selector(dist, 20.0, 28.0, obj, "Radial"))
    assert inner[0] == pytest.approx(1.0, abs=1e-6)   # each zone internally aligned
    assert outer[0] == pytest.approx(1.0, abs=1e-6)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/measure/test_measure_orientation_zones.py -v`
Expected: FAIL (`ModuleNotFoundError` / `ImportError: cannot import name 'MeasureOrientationZones'`).

- [ ] **Step 3: Implement the pure helpers + operator skeleton**

```python
"""MeasureOrientationZones: per-zone hyphal orientation concentration/turning."""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import PrivateAttr, field_validator

# Control/FigureProvider/figure are re-exported from phenotypic.abc_ (this is
# exactly what _measure_symmetric_zones.py imports — confirm at line ~23).
from phenotypic.abc_ import Control, FigureProvider, MeasureFeatures, figure
from phenotypic.schema import OBJECT, ORIENTATION_ZONES
from phenotypic.util._orientation_field import orientation_field
from phenotypic.measure._zone_segmentation import (
    ZoneSegmentation,
    ZoneSegmentationParams,
    compute_zone_segmentation,
    distance_from_point,
    expand_slice_around_center,
)

_VARIANTS = ("Radial", "Mask")
_ZONES = ("Overall", "Dense", "Sparse")
_METRICS = ("Concentration", "Turning", "Coherence")
_EPS = 1e-9


def zone_selector(dist_map, r_lo, r_hi, obj_mask, variant):
    """Boolean selector for a radial zone on a tile; ``Mask`` also ∩ obj_mask."""
    if not np.isfinite(r_lo) or not np.isfinite(r_hi) or r_hi <= r_lo:
        return np.zeros(dist_map.shape, dtype=bool)
    radial = (dist_map >= r_lo) & (dist_map < r_hi)
    if variant == "Mask":
        return radial & obj_mask
    return radial


def aggregate_orientation(phi, coherence, grad_phi, selector, eps=_EPS):
    """Coherence-weighted (R, turning, mean-coherence) over a selector.

    Returns (nan, nan, nan) when the selector is empty or sum(coherence)~0.
    """
    if not selector.any():
        return (np.nan, np.nan, np.nan)
    C = coherence[selector]
    sumC = float(C.sum())
    if sumC < eps:
        return (np.nan, np.nan, np.nan)
    c2 = np.cos(2.0 * phi[selector])
    s2 = np.sin(2.0 * phi[selector])
    Rx = float((C * c2).sum()) / sumC
    Ry = float((C * s2).sum()) / sumC
    R = float(np.hypot(Rx, Ry))
    turning = float((C * grad_phi[selector]).sum()) / sumC
    return (R, turning, float(C.mean()))
```

- [ ] **Step 4: Implement the operator fields + `_operate`**

Add the class. Mirror `MeasureSymmetricZones` field declarations for the 12 passthrough params (copy their defaults and any `field_validator`s). `intensity_source` is shared (default `"detect_mat"` here) and drives both the tensor input and the zone-segmentation intensity.

```python
class MeasureOrientationZones(MeasureFeatures, FigureProvider):
    """Measure per-zone hyphal orientation concentration, turning, and coherence.

    Computes the structure-tensor orientation field over a mask-free tile (grid
    section when the image is a GridImage, else an expanded crop) and aggregates
    coherence-weighted metrics over radially-defined zones bounded by the
    symmetric radius, in both a ``Radial`` and a raw ``Mask`` variant. Emits the
    :class:`~phenotypic.schema.ORIENTATION_ZONES` columns.

    Args:
        intensity_source: Image array for the structure tensor and zone
            segmentation (``"detect_mat"`` default, ``"gray"`` alternative).
        sigma_d: Gaussian-derivative (gradient) scale in pixels, ~ hypha width.
        sigma_i: Structure-tensor integration scale in pixels.
        quiver_block: inspect() quiver downsample block size in pixels.
        n_annuli, pelt_penalty, symmetry_threshold, n_angular_bins,
        smoothing_window, method, extent_margin, min_samples_per_ring,
        tau_core, tau_dense, tau_sparse: passed through to the shared zone
            segmentation (same meaning/defaults as MeasureSymmetricZones).

    Examples:
        >>> from phenotypic.data import load_synth_filamentous_plate
        >>> from phenotypic.measure import MeasureOrientationZones
        >>> image = load_synth_filamentous_plate()
        >>> df = MeasureOrientationZones().measure(image)
        >>> 'OrientZones_Concentration-Radial-Overall' in df.columns
        True
    """

    intensity_source: Literal["gray", "detect_mat"] = "detect_mat"
    sigma_d: float = 1.5
    sigma_i: float = 4.0
    quiver_block: int = 12
    # --- zone passthrough (defaults identical to MeasureSymmetricZones) ---
    n_annuli: int = 100
    pelt_penalty: float = 5.0
    symmetry_threshold: float = 4 / 6
    n_angular_bins: int = 6
    smoothing_window: int = 3
    method: Literal["distance", "intensity"] = "distance"
    extent_margin: float = 0.05
    min_samples_per_ring: int = 5
    tau_core: float = 0.9
    tau_dense: float = 0.5
    tau_sparse: float = 0.1
    # Per-object figure intermediates, populated by _operate. PrivateAttr keeps
    # it out of model_dump()/JSON (mirrors MeasureSymmetricZones' cache pattern).
    _cache: dict = PrivateAttr(default_factory=dict)
    _cache_image: "object | None" = PrivateAttr(default=None)

    @field_validator("sigma_d", "sigma_i")
    @classmethod
    def _positive_sigma(cls, v):
        if v <= 0:
            raise ValueError("sigma_d and sigma_i must be > 0")
        return v

    def _zone_params(self) -> ZoneSegmentationParams:
        return ZoneSegmentationParams(
            n_annuli=self.n_annuli, pelt_penalty=self.pelt_penalty,
            symmetry_threshold=self.symmetry_threshold, n_angular_bins=self.n_angular_bins,
            smoothing_window=self.smoothing_window, method=self.method,
            extent_margin=self.extent_margin, min_samples_per_ring=self.min_samples_per_ring,
            tau_core=self.tau_core, tau_dense=self.tau_dense, tau_sparse=self.tau_sparse,
            intensity_source=self.intensity_source,
        )

    def _resolve_tile(self, image, seg: ZoneSegmentation, prop, label2section):
        """Return (tile_intensity, obj_mask_tile, centre_rc) for one object.

        Preferred: the object's **grid section** via ``image.grid[idx]`` — an
        object-aware cropped Image (only this object's label survives; the crop
        preserves the complete object, so it is a superset of the object's
        pixels). Verified API: ``image.grid[section_idx]`` returns a cropped
        ``Image``; the crop origin is recovered by the public exact identity
        ``origin = prop.centroid(full) - regionprops(section)[label].centroid``.
        Falls back to the mask-free expanded crop when the image is not a
        GridImage, the section lookup fails, or the section does not cover the
        r_max disk around the centre (crowded/overgrown plate).
        """
        from skimage.measure import regionprops
        r_max = max(seg.sparse_end_radius, seg.symmetric_radius) * (1 + self.extent_margin)
        if hasattr(image, "grid") and seg.label in label2section:
            try:
                section = image.grid[label2section[seg.label]]
                sec_props = {p.label: p for p in regionprops(section.objmap[:])}
                sp = sec_props.get(seg.label)
                if sp is not None:
                    origin = (prop.centroid[0] - sp.centroid[0],
                              prop.centroid[1] - sp.centroid[1])
                    centre = (seg.centroid_global[0] - origin[0],
                              seg.centroid_global[1] - origin[1])
                    H, W = section.objmap[:].shape[:2]
                    if (centre[0] - r_max >= 0 and centre[0] + r_max <= H
                            and centre[1] - r_max >= 0 and centre[1] + r_max <= W):
                        tile = np.asarray(getattr(section, self.intensity_source)[:], dtype=np.float64)
                        return tile, (section.objmap[:] == seg.label), centre
            except (KeyError, IndexError, ValueError, AttributeError):
                pass
        # Fallback: expanded crop on the full plate (non-grid / clipped section).
        hw = image.gray[:].shape[:2]            # 2-tuple; image.shape is (H,W,3) for RGB
        sl = expand_slice_around_center(seg.centroid_global, r_max, hw)
        tile = np.asarray(getattr(image, self.intensity_source)[sl], dtype=np.float64)
        obj_mask = (image.objmap[:][sl] == seg.label)
        centre = (seg.centroid_global[0] - sl[0].start, seg.centroid_global[1] - sl[1].start)
        return tile, obj_mask, centre

    def _zone_bounds(self, seg: ZoneSegmentation):
        return {
            "Overall": (0.0, seg.symmetric_radius),
            "Dense": (seg.core_end_radius, seg.dense_end_radius),
            "Sparse": (seg.dense_end_radius, seg.sparse_end_radius),
        }

    def _prep(self, image):
        """Regionprops + label→grid-section map, computed ONCE per image.

        grid.info() is slow on filamentous plates, so never call it per object.
        intensity_image is required so compute_zone_segmentation can read
        prop.centroid_weighted when method="intensity" (else AttributeError).
        """
        from skimage.measure import regionprops
        from phenotypic.schema import GRID
        props = regionprops(image.objmap[:],
                            intensity_image=image.gray[:].astype(np.float64, copy=False))
        label2section = {}
        if hasattr(image, "grid"):
            info = image.grid.info()
            lab, rmi = str(OBJECT.LABEL), str(GRID.ROW_MAJOR_IDX)
            label2section = dict(zip(info[lab].astype(int), info[rmi].astype(int)))
        return props, label2section

    def _iter_object_fields(self, image, props, label2section):
        """Yield (prop, seg, obj_mask, phi, coh, grad, dist_map, centre) per object.

        SINGLE source of truth for the heavy orientation compute — reused by
        _operate() (which keeps only compact summaries) and by dashboard()'s
        coherence panel (which recomputes on demand). The full-resolution arrays
        yielded here are consumed and discarded by each caller; nothing full-res
        is retained on the instance. Tiny objects (area<10) are skipped.
        """
        for prop in props:
            if prop.area < 10:
                continue
            seg = compute_zone_segmentation(image, prop, params=self._zone_params())
            tile, obj_mask, centre = self._resolve_tile(image, seg, prop, label2section)
            phi, coh, grad = orientation_field(tile, self.sigma_d, self.sigma_i)
            dist_map = distance_from_point(tile.shape, centre)
            yield prop, seg, obj_mask, phi, coh, grad, dist_map, centre

    def _operate(self, image) -> pd.DataFrame:
        props, label2section = self._prep(image)
        headers = ORIENTATION_ZONES.get_headers()
        # pre-seed every object's row with NaN so skipped/failed objects still appear
        base = {}
        for prop in props:
            r = {OBJECT.LABEL: prop.label}
            r.update({h: np.nan for h in headers})
            base[prop.label] = r
        self._cache.clear()          # compact per-object figure records only
        self._cache_image = image    # single reference (not a copy) for no-arg figures
        for prop, seg, obj_mask, phi, coh, grad, dist_map, centre in \
                self._iter_object_fields(image, props, label2section):
            per_zone = self._fill_metrics(base[prop.label], seg, obj_mask, phi, coh, grad, dist_map)
            # LEAN CACHE: store compact summaries only — NO full-res tile/phi/coh/
            # grad/dist_map and NO seg dataclass. Bounds memory to O(objects*blocks).
            self._cache[prop.label] = {
                "centroid_global": tuple(seg.centroid_global),
                "centre": centre,
                "radii": {"core": seg.core_radius, "symmetric": seg.symmetric_radius,
                          "core_end": seg.core_end_radius, "dense_end": seg.dense_end_radius,
                          "sparse_end": seg.sparse_end_radius},
                "zones_computed": seg.zones_computed,
                "quiver": _downsample_quiver(phi, coh, self.quiver_block),  # block-res
                "per_zone": per_zone,
            }
        return pd.DataFrame([base[p.label] for p in props], columns=[OBJECT.LABEL, *headers])

    def _fill_metrics(self, row, seg, obj_mask, phi, coh, grad, dist_map):
        """Write the 18 columns for one object; return the compact per_zone dict."""
        per_zone = {}
        for zone, (r_lo, r_hi) in self._zone_bounds(seg).items():
            zone_ok = seg.zones_computed or zone == "Overall"
            for variant in _VARIANTS:
                if not zone_ok:
                    R = t = cm = direction = np.nan
                else:
                    sel = zone_selector(dist_map, r_lo, r_hi, obj_mask, variant)
                    R, t, cm = aggregate_orientation(phi, coh, grad, sel)
                    direction = _resultant_direction(phi, coh, sel)
                per_zone[(variant, zone)] = (R, t, cm, direction)   # scalars only
                row[f"OrientZones_Concentration-{variant}-{zone}"] = R
                row[f"OrientZones_Turning-{variant}-{zone}"] = t
                row[f"OrientZones_Coherence-{variant}-{zone}"] = cm
        return per_zone

    def _coherence_canvas(self, image, downsample: int = 4):
        """Recompute per-object coherence and composite onto a plate canvas.

        Used only by dashboard()'s heatmap. Full-res fields are recomputed via
        _iter_object_fields and discarded here — the heatmap costs compute, not
        persistent memory. Returned canvas is downsampled for a light figure.
        """
        props, label2section = self._prep(image)
        canvas = np.full(image.gray[:].shape[:2], np.nan)
        for _prop, seg, _mask, _phi, coh, _grad, _dist, centre in \
                self._iter_object_fields(image, props, label2section):
            r0 = int(round(seg.centroid_global[0] - centre[0]))
            c0 = int(round(seg.centroid_global[1] - centre[1]))
            h, w = coh.shape
            r1, c1 = min(r0 + h, canvas.shape[0]), min(c0 + w, canvas.shape[1])
            canvas[max(r0, 0):r1, max(c0, 0):c1] = coh[: r1 - max(r0, 0), : c1 - max(c0, 0)]
        return canvas[::downsample, ::downsample]
```

Add two module-level helpers (compact-cache builders, pure/testable):

```python
def _downsample_quiver(phi, coherence, block):
    """Block-mean the doubled-angle field → (rows, cols, phi_block, coh_block).

    Circular-averages cos2φ/sin2φ (coherence-weighted) and means coherence over
    block×block cells. Returns block-centre coords in the TILE frame plus per-block
    orientation and coherence — a few KB, the only array kept in the lean cache.
    """
    h, w = phi.shape
    nr, nc = max(h // block, 1), max(w // block, 1)
    rows = np.empty((nr, nc)); cols = np.empty((nr, nc))
    pb = np.empty((nr, nc)); cb = np.empty((nr, nc))
    c2, s2 = np.cos(2.0 * phi), np.sin(2.0 * phi)
    for i in range(nr):
        for j in range(nc):
            rsl, csl = slice(i * block, (i + 1) * block), slice(j * block, (j + 1) * block)
            cc = coherence[rsl, csl]
            rows[i, j], cols[i, j] = i * block + block / 2, j * block + block / 2
            cb[i, j] = float(cc.mean())
            wsum = float(cc.sum())
            pb[i, j] = (0.5 * np.arctan2((cc * s2[rsl, csl]).sum(), (cc * c2[rsl, csl]).sum())
                        if wsum > 1e-12 else np.nan)
    return rows, cols, pb, cb


def _resultant_direction(phi, coherence, selector):
    """Coherence-weighted mean orientation over a selector (for the inspect glyph)."""
    if not selector.any():
        return np.nan
    C = coherence[selector]
    if float(C.sum()) < _EPS:
        return np.nan
    return 0.5 * np.arctan2(float((C * np.sin(2.0 * phi[selector])).sum()),
                            float((C * np.cos(2.0 * phi[selector])).sum()))
```

**Design note (lean caching — memory bound):** `measure()` retains **only** compact
per-object records — scalar radii, `centre`/`centroid_global`, `zones_computed`,
the block-downsampled `quiver` (a few KB), and the scalar `per_zone` summaries. It
never keeps the full-resolution `tile`/`phi`/`coherence`/`grad_phi`/`dist_map` or
the heavy `seg` dataclass. This bounds persistent memory to **O(objects × blocks)**
(~1–3 MB/plate) instead of **O(objects × tile-pixels × 8)** (~hundreds of MB on
large plates). `inspect()` renders entirely from this compact cache (no recompute);
`dashboard()`'s coherence heatmap recomputes full-res on demand via
`_coherence_canvas` and discards it. `self._cache_image` is a single reference (not
a copy), matching `MeasureSymmetricZones`.

**Design note (grid-section tile — verified 2026-07-03):** the tile comes from
`image.grid[section_idx]` (in `_resolve_tile`), not hand-rolled edge slicing —
confirmed live on `load_synth_filamentous_plate()`: `grid.info()` exposes
`Object_Label → Grid_RowMajorIdx`; `image.grid[idx]` returns a cropped `Image`
whose `objmap` retains only the section's object(s); the crop origin equals
`prop.centroid(full) − regionprops(section)[label].centroid` exactly. The crop is
object-aware (superset of the object); the disk check falls back to the expanded
crop only when the symmetric radius exceeds the section.

- [ ] **Step 5: Export the operator**

In `src/phenotypic/measure/__init__.py` add `from ._measure_orientation_zones import MeasureOrientationZones` and add `"MeasureOrientationZones"` to `__all__`.

- [ ] **Step 6: Run the Task-4 tests**

Run: `uv run pytest tests/unit/measure/test_measure_orientation_zones.py -v`
Expected: the 5 helper/zone tests PASS.

- [ ] **Step 7: Add operator-level integration tests (real fixtures)**

Append to `tests/unit/measure/test_measure_orientation_zones.py`:

```python
def test_measure_returns_18_columns_one_row_per_object():
    image = load_synth_filamentous_plate()
    df = MeasureOrientationZones().measure(image)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == image.num_objects
    for h in ORIENTATION_ZONES.get_headers():
        assert h in df.columns
    # R and coherence within [0,1] where finite
    for col in df.columns:
        if col.startswith(("OrientZones_Concentration", "OrientZones_Coherence")):
            vals = df[col].to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            assert np.all((finite >= -1e-9) & (finite <= 1 + 1e-9))


def test_rotation_invariance_of_R_magnitude_and_turning():
    # A single synthetic tile rotated 90 deg: R magnitude and turning invariant.
    from phenotypic.measure._measure_orientation_zones import (
        aggregate_orientation, zone_selector,
    )
    n = 61
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    dist = np.hypot(yy - c, xx - c)
    base = np.sin(2 * np.pi * xx / 7.0)
    from phenotypic.util._orientation_field import orientation_field
    obj = np.ones((n, n), dtype=bool)

    def metrics(field):
        phi, coh, grad = orientation_field(field, 1.5, 4.0)
        sel = zone_selector(dist, 0.0, 20.0, obj, "Radial")
        return aggregate_orientation(phi, coh, grad, sel)

    R0, t0, _ = metrics(base)
    R90, t90, _ = metrics(np.rot90(base))
    assert R0 == pytest.approx(R90, abs=0.05)
    assert t0 == pytest.approx(t90, abs=0.05)


def test_tiny_objects_are_all_nan():
    image = load_synth_yeast_plate()
    df = MeasureOrientationZones().measure(image)
    # every row has the full column set; NaN allowed, no exceptions
    assert set(ORIENTATION_ZONES.get_headers()).issubset(df.columns)


def test_measure_cache_is_compact():
    # Guard against memory bloat: after measure(), the per-object cache must hold
    # NO full-res arrays and NO seg dataclass — only scalars + the block quiver.
    image = load_synth_filamentous_plate()
    op = MeasureOrientationZones()
    op.measure(image)
    assert op._cache, "cache should be populated"
    forbidden = {"tile", "phi", "coherence", "grad_phi", "dist_map", "seg"}
    for rec in op._cache.values():
        assert forbidden.isdisjoint(rec), f"full-res leaked: {forbidden & set(rec)}"
        assert "quiver" in rec
        for v in rec.values():
            if isinstance(v, np.ndarray):
                assert v.size <= 4096, "only the block-resolution quiver may be cached"
        # the block quiver must be far smaller than a full tile
        rows, cols, pb, cb = rec["quiver"]
        assert pb.shape == cb.shape and pb.size <= 4096
```

Run: `uv run pytest tests/unit/measure/test_measure_orientation_zones.py -v`
Expected: all PASS (including the compact-cache guard). If the grid-section tile mis-resolves on the fixture, fix `_resolve_tile` before proceeding.

- [ ] **Step 8: Type-check, lint, doctest, commit**

Run: `uv run mypy src/phenotypic/measure/_measure_orientation_zones.py`
Run: `uv run ruff check --fix src/phenotypic/measure/_measure_orientation_zones.py tests/unit/measure/test_measure_orientation_zones.py`
Run: `uv run pytest --doctest-modules src/phenotypic/measure/_measure_orientation_zones.py`
Expected: doctest PASS.
```bash
git add src/phenotypic/measure/_measure_orientation_zones.py src/phenotypic/measure/__init__.py tests/unit/measure/test_measure_orientation_zones.py
git commit -m "feat(measure): add MeasureOrientationZones operator (metrics)"
```

---

### Task 5: `inspect()` — saveable primary figure (quiver + per-zone glyphs + rings)

**Files:**
- Modify: `src/phenotypic/measure/_measure_orientation_zones.py`
- Test: `tests/unit/measure/test_measure_orientation_zones.py`

**Interfaces:**
- Consumes: the per-object `self._cache` populated in `_operate` (Task 4); `@figure`, `BASE_LAYER`-style `Control`, shared viz helpers `plotly_imshow` / `add_plotly_obj_labels` (confirm import paths against `_measure_symmetric_zones.py`).
- Produces: `inspect(self, image=None, base_layer="detect_mat", *, for_save=False) -> go.Figure`, decorated `@figure(title=..., primary=True, controls={"base_layer": BASE_LAYER})`.

- [ ] **Step 1: Write a smoke test for `inspect()`**

```python
def test_inspect_builds_figure():
    import plotly.graph_objects as go
    image = load_synth_filamentous_plate()
    op = MeasureOrientationZones()
    op.measure(image)
    fig = op.inspect(image)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    fig_save = op.inspect(image, for_save=True)
    assert isinstance(fig_save, go.Figure)
```

Run: `uv run pytest tests/unit/measure/test_measure_orientation_zones.py::test_inspect_builds_figure -v`
Expected: FAIL (`AttributeError`/no `inspect`).

- [ ] **Step 2: Define the `BASE_LAYER` control and the `inspect()` figure**

Copy the `BASE_LAYER = Control(...)` definition from `_measure_symmetric_zones.py` (label "Base layer", kind "select", default `"detect_mat"`, options `("rgb","gray","detect_mat")`). Implement:

```python
@figure(title="Orientation-field overlay", primary=True, controls={"base_layer": BASE_LAYER})
def inspect(self, image=None, base_layer="detect_mat", *, for_save=False):
    """Plate overview with the coherence-modulated quiver, zone rings, and
    per-zone resultant glyphs. The single saveable primary figure."""
    import plotly.graph_objects as go
    from phenotypic.sdk_._plotly_helpers import plotly_imshow, add_plotly_obj_labels
    if image is None:
        image = self._require_cache_image()   # guard defined below
    fig = go.Figure()
    base = getattr(image, base_layer)[:]
    plotly_imshow(fig, base)                    # match the helper's call signature
    self._add_quiver_trace(fig)                 # A
    self._add_zone_ring_traces(fig)             # rings
    self._add_resultant_glyph_traces(fig)       # C
    add_plotly_obj_labels(fig, image)
    if for_save:
        for tr in fig.data:
            if tr.visible == "legendonly":
                tr.visible = True
    return fig
```

Implement three private trace builders that read **only the compact `self._cache`** (no recompute):
- `_add_quiver_trace(fig)`: for each cached object, read the pre-downsampled `quiver = (rows, cols, phi_block, coh_block)` (already block-resolution — no tile access). Draw one short segment per block (length + opacity ∝ `coh_block`, skip NaN blocks) as a **single NaN-separated** `Scattergl` trace across all objects (append `None` between segments). Convert block (row,col)+`phi_block` to plate-frame x/y using the tile's plate origin `origin = (centroid_global[0] − centre[0], centroid_global[1] − centre[1])` (both cached per record).
- `_add_zone_ring_traces(fig)`: circle polygons at the cached `radii["symmetric"|"core_end"|"dense_end"|"sparse_end"]` centred at the cached `centroid_global` (reuse a 72-vertex circle helper — replicate the small `_circle_xy` from `_measure_symmetric_zones.py` locally or import it).
- `_add_resultant_glyph_traces(fig)`: per object per zone, read the cached `per_zone[(variant, zone)] = (R, turning, coh, direction)`; draw the resultant arrow from `centroid_global` (angle = cached `direction`, length ∝ `R`), plus a text badge of `R`/turning for the `Radial` variant. No recompute — `direction` was stored in `_fill_metrics`.

Keep NaN zones/objects skipped. `self._cache_image` is set in `_operate` (Task 4); add the guard:

```python
def _require_cache_image(self):
    if self._cache_image is None:
        raise RuntimeError("call measure(image) before inspect()/dashboard()")
    return self._cache_image
```

Also import the 72-vertex circle helper `_circle_xy` from `_measure_symmetric_zones` if it is module-accessible; otherwise replicate the tiny function locally (it is a pure `(cx, cy, r) -> (xs, ys)` polygon).

- [ ] **Step 3: Run the smoke test**

Run: `uv run pytest tests/unit/measure/test_measure_orientation_zones.py::test_inspect_builds_figure -v`
Expected: PASS.

- [ ] **Step 4: Full-file test, lint, commit**

Run: `uv run pytest tests/unit/measure/test_measure_orientation_zones.py -v`
Run: `uv run ruff check --fix src/phenotypic/measure/_measure_orientation_zones.py`
```bash
git add src/phenotypic/measure/_measure_orientation_zones.py tests/unit/measure/test_measure_orientation_zones.py
git commit -m "feat(measure): add MeasureOrientationZones.inspect() overview figure"
```

---

### Task 6: `dashboard()` — composed diagnostic (adds coherence heatmap)

**Files:**
- Modify: `src/phenotypic/measure/_measure_orientation_zones.py`
- Test: `tests/unit/measure/test_measure_orientation_zones.py`

**Interfaces:**
- Consumes: `self._cache`, `inspect()` (Task 5), the `FigureProvider` composition pattern (`iter_figures` / `_render_spec` / `make_subplots`) as used by `GridFitReport.dash()` and `AutoGridFinder.dashboard()`.
- Produces: `dashboard(self, image=None, show=True) -> go.Figure`, plus a transient `_OrientationZonesReport(FigureProvider)` whose control-free `@figure` panels compose into one vertically-stacked figure.

- [ ] **Step 1: Write a smoke test for `dashboard()`**

```python
def test_dashboard_builds_composed_figure():
    import plotly.graph_objects as go
    image = load_synth_filamentous_plate()
    op = MeasureOrientationZones()
    op.measure(image)
    fig = op.dashboard(image, show=False)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    # the go.Table summary panel must survive composition (base composer can't
    # host it — proves the custom dash() override with per-row specs works).
    assert any(getattr(tr, "type", None) == "table" for tr in fig.data)
    # coherence heatmap present too
    assert any(getattr(tr, "type", None) == "heatmap" for tr in fig.data)
```

Run: `uv run pytest tests/unit/measure/test_measure_orientation_zones.py::test_dashboard_builds_composed_figure -v`
Expected: FAIL.

- [ ] **Step 2: Implement the transient report + `dashboard()`**

```python
class _OrientationZonesReport(FigureProvider):
    """Transient control-free FigureProvider composing the orientation diagnostic.

    Overrides dash() (GridFitReport pattern) because the base composer builds a
    uniform xy subplot grid that cannot host the go.Table summary panel.
    """

    def __init__(self, op, image, cache):
        self._op = op
        self._image = image
        self._cache = cache

    @figure(title="Orientation-field overlay")
    def _panel_overview(self):
        return self._op.inspect(self._image, for_save=True)

    @figure(title="Coherence map")
    def _panel_coherence(self):
        # B: coherence heatmap. Recomputed on demand (NOT from cache — the lean
        # cache holds no full-res coherence), then discarded. Downsampled canvas.
        import plotly.graph_objects as go
        canvas = self._op._coherence_canvas(self._image)     # recompute + composite
        fig = go.Figure(go.Heatmap(z=canvas, colorscale="Viridis", zmin=0, zmax=1,
                                   colorbar=dict(title="C")))
        fig.update_yaxes(autorange="reversed")
        return fig

    @figure(title="Per-zone concentration & turning")
    def _panel_summary(self):
        # go.Table: one row per (Variant, Zone), columns Concentration & Turning,
        # aggregated across objects (mean over finite values). Requires the
        # custom dash() override below (base composer cannot host go.Table).
        import plotly.graph_objects as go
        variants, zones = ("Radial", "Mask"), ("Overall", "Dense", "Sparse")
        rows = []
        for v in variants:
            for z in zones:
                conc = np.nanmean([r["per_zone"][(v, z)][0] for r in self._cache.values()]) if self._cache else np.nan
                turn = np.nanmean([r["per_zone"][(v, z)][1] for r in self._cache.values()]) if self._cache else np.nan
                rows.append((f"{v} · {z}", f"{conc:.3f}", f"{turn:.4f}"))
        header = ["Variant · Zone", "Concentration (R)", "Turning (rad/px)"]
        cols = list(zip(*rows)) if rows else [[], [], []]
        return go.Figure(go.Table(header=dict(values=header),
                                  cells=dict(values=[list(c) for c in cols])))

    def dash(self, subject=None):
        """Custom composition (per-row specs) so the go.Table panel renders.

        Mirrors GridFitReport.dash(): render each @figure spec, detect table vs
        xy panels, build make_subplots with matching per-row specs, transfer
        traces, apply the house theme. Annotations/shapes on the overview panel
        (zone-ring labels, R/turning badges) are NOT auto-carried by trace copy —
        transfer them explicitly (see the smoke test's shape/annotation assert).
        """
        from plotly.subplots import make_subplots
        specs = self.iter_figures()
        rendered = [self._render_spec(s) for s in specs]
        is_table = [bool(f.data) and f.data[0].type == "table" for f in rendered]
        row_specs = [[{"type": "table"}] if t else [{"type": "xy"}] for t in is_table]
        composed = make_subplots(rows=len(specs), cols=1,
                                 subplot_titles=[s.title for s in specs],
                                 specs=row_specs, vertical_spacing=0.06)
        for i, fig in enumerate(rendered, start=1):
            for tr in fig.data:
                composed.add_trace(tr, row=i, col=1)
        return composed


def dashboard(self, image=None, show=True):
    """Composed notebook diagnostic: inspect() overview + coherence map +
    per-zone summary, stacked vertically (returns a single go.Figure)."""
    if image is None:
        image = self._require_cache_image()
    if not self._cache or self._cache_image is not image:
        self.measure(image)
    report = _OrientationZonesReport(self, image, self._cache)
    fig = report.dash()
    if show:
        try:
            fig.show()
        except Exception:
            pass
    return fig
```

Confirm `iter_figures()`/`_render_spec()` signatures against `FigureProvider` and `GridFitReport.dash()`; apply the house theme the same way `GridFitReport` does if it wraps the composed figure (e.g. `apply_theme(composed)`).

- [ ] **Step 3: Run smoke test**

Run: `uv run pytest tests/unit/measure/test_measure_orientation_zones.py::test_dashboard_builds_composed_figure -v`
Expected: PASS.

- [ ] **Step 4: Full suite for the module, type-check, lint, commit**

Run: `uv run pytest tests/unit/measure/test_measure_orientation_zones.py tests/unit/measure/test_orientation_field.py tests/unit/measure/test_zone_segmentation_regression.py -v`
Expected: all PASS.
Run: `uv run mypy src/phenotypic/measure/_measure_orientation_zones.py`
Run: `uv run ruff check --fix src/phenotypic/measure/_measure_orientation_zones.py tests/unit/measure/test_measure_orientation_zones.py`
```bash
git add src/phenotypic/measure/_measure_orientation_zones.py tests/unit/measure/test_measure_orientation_zones.py
git commit -m "feat(measure): add MeasureOrientationZones.dashboard() composed diagnostic"
```

- [ ] **Step 5: Final regression sweep across measure + schema**

Run: `uv run pytest tests/unit/measure -v`
Expected: all PASS (symmetric-zones golden regression still green; new tests green).
Run: `uv run python -c "from phenotypic.measure import MeasureOrientationZones; from phenotypic.schema import ORIENTATION_ZONES; print('exports ok')"`
Expected: `exports ok`.

---

## Deferred (out of scope for this plan)

Per spec §11, these are intentionally **not** implemented now: the `Core` zone triple, an orientation-entropy metric, per-object zoom panels in `inspect()`, and pruning the `Mask` variant. They are mechanical to add once the dense/sparse/overall pattern lands.

## Self-Review Notes

- **Spec coverage:** §2 schema → Task 3; §3.1 extraction + regression → Task 1; §3.2 operator → Task 4; §4.1 tile selection → Task 4 `_resolve_tile` (grid section via `image.grid[idx]`, expanded-crop fallback); §4.2 field → Task 2; §4.3 selectors/aggregation → Task 4 helpers; §5 NaN semantics → Task 4 `_fill_metrics` (+ Task 4 tests); §6.1 inspect → Task 5; §6.2 dashboard → Task 6; §7 params → Task 4 fields; §8 testing → Tasks 1–6 tests; §9 file inventory → File Structure above.
- **Memory design (lean caching):** `measure()` retains only compact per-object records (scalars + block-downsampled quiver) — O(objects × blocks), ~1–3 MB/plate — never full-res `tile`/`phi`/`coherence`/`grad_phi`/`dist_map` or the `seg` dataclass. `inspect()` renders from that cache with no recompute; `dashboard()`'s coherence heatmap recomputes full-res via `_coherence_canvas` and discards it. `test_measure_cache_is_compact` (Task 4 Step 7) guards against regression. The heavy compute lives in the single `_iter_object_fields` generator (DRY across `_operate` and `_coherence_canvas`).
- **Type consistency:** `ZoneSegmentationParams` field set is identical in Task 1 (definition) and Task 4 (`_zone_params`); `zone_selector`/`aggregate_orientation` signatures are identical across Task 4 definition and Task 5/6 reuse; header pattern `OrientZones_<Metric>-<Variant>-<Zone>` is identical in Task 3 (enum labels) and Task 4 (`_fill_metrics` f-strings) — the Task-3 smoke test and Task-4 header assertions cross-check they match.
- **Verify-before-code hooks flagged inline:** the `Control`/`FigureProvider`/`figure` re-export from `phenotypic.abc_` and `iter_figures()`/`_render_spec()` signatures must be confirmed against `_measure_symmetric_zones.py:~23` and `GridFitReport.dash()` during implementation (each step says so). Import paths for `Entry`/`DescriptiveTrait` (`schema/_measurement_info` / `_tiers`) and `plotly_imshow`/`add_plotly_obj_labels` (`phenotypic.sdk_._plotly_helpers`) are resolved.
- **Plan-review fixes applied (2026-07-03):** (1) `_extract_mask_boundary` added as a 15th relocated helper with its two hardcoded callers rewritten — otherwise Task 1 breaks with an import cycle; (2) `regionprops(..., intensity_image=image.gray[:])` so `method="intensity"` doesn't crash; (3) the grid-section tile is obtained via the verified public `image.grid[section_idx]` (object-aware cropped Image) with the crop origin recovered exactly from the centroid identity, not hand-rolled edge slicing; the expanded-crop fallback uses `image.gray[:].shape[:2]` (not the RGB 3-tuple `image.shape`) and fires only when the section fails to cover the r_max disk; `grid.info()` is computed once per image; (4) `Literal[...]` typing for `method`/`intensity_source`; (5) `_cache`/`_cache_image` via `PrivateAttr`; (6) Task 6 uses a custom `GridFitReport`-style `dash()` override so the `go.Table` panel renders. Author docstrings fresh — do **not** copy `MeasureSymmetricZones.n_angular_bins`'s stale "Defaults to 36" (the default is 6).
