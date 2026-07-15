# Two-`k` Phase Hysteresis — Enhancer + Reconnecting Detector — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship two new operations packaging this session's two-`k` phase-congruency hysteresis — `FocusEdgeTwoKPhase` (a pure branch-response enhancer) and `TwoKFilamentousDetector` (a grid detector that binarizes it, fills inoculum centers, and reconnects fragments) — by first extracting `FilamentousFungiDetector`'s reconnection/labeling machinery into `sdk_.reconnect` as reusable pure functions that both detectors consume.

**Architecture:** Three layers, bottom-up. (1) Pull FFD's Phase 3–5 orchestration (overlap-filter → grid-Voronoi label → cost surface → tiled Dijkstra reconnection → final Voronoi) out of the `FilamentousFungiDetector` monolith into pure array functions in `sdk_.reconnect`, then refactor FFD to consume them behind a golden regression gate. (2) Add the two-`k` kernel + `FocusEdgeTwoKPhase` enhancer in `enhance/`. (3) Add `TwoKFilamentousDetector` in `detect/` = two-`k` branch mask + inline grid center-fill + the extracted `sdk_.reconnect` functions.

**Tech Stack:** Python ≥3.10,<3.13 · `uv` · pydantic v2 operations · numpy/scipy/skimage · phase congruency (Kovesi phasecong3) · multi-source Dijkstra · `pytest`.

## Global Constraints

- **Package manager / tests:** run everything via `uv run` from the repo root `/Users/alex/Projects/PhenoTypic`. Test command: `uv run pytest <path> -v`. Default `addopts` already includes `-m 'not slow'`; do **not** combine `--testmon` with `-n auto`.
- **`sdk_.reconnect` import purity (load-bearing):** every function added to `sdk_.reconnect` MUST accept **plain numpy arrays and scalars only** — never `Image`, pydantic operations, GUI types, or `_PhaseCong3Result`. Importing the package must not pull in `matplotlib`, `napari`, `plotly`, `dash`, `PyQt*`, `PySide*`, `astropy`, `fil_finder`, `filterpy`, or `gudhi` (enforced by `tests/unit/sdk_/reconnect/test_import_rules.py`). This is why `build_reconnect_cost` takes `pc_sum, M, m, orientation` as **separate arrays**, not the dataclass.
- **Coordinate & angle convention:** `(row, column)` coordinates; axial angles in radians (matches existing `sdk_.reconnect` CLAUDE.md).
- **Behavior preservation (FFD refactor):** the FFD refactor (Phase 3) is a pure move — no algorithm change. FFD's `objmap` output on the golden fixture must be **bit-identical** before and after.
- **No gamma preprocessing:** the two-`k` chain runs on the raw cropped image through `FlattenIllumination(300) → ContrastStretching(70,99)` — **no** `adjust_gamma` step (settled this session: gamma marginally *increased* fragmentation).
- **NormalizedOutputMixin MRO:** any new `NormalizedOutputMixin` enhancer must list the mixin **first**: `class X(NormalizedOutputMixin, FocusEdge)` (matches `FocusEdgeMonogenicPhase`).
- **Locked design decisions** (resolve spec §5 open questions): Q1 reuse = extract to `sdk_.reconnect` as **composable pieces** (separate labeling + reconnection functions), FFD refactored to consume them. Q2 preprocessing = a `branch_base` pipeline field defaulting to `FlattenIllumination(300) + ContrastStretching(70,99)`. Q3 `k_loose` default = `4.5`. Q4 = build the Dijkstra detector (Class 2). Center-fill (spec §2) = **inline detector fields** (`center_detector` + `background_subtractor`), matching FFD's `inoculum_detector` pattern — *not* a standalone `FillInoculumCenters` class. Class 1 stays a pure branch enhancer regardless.

---

## File Structure

**New files:**
- `src/phenotypic/sdk_/reconnect/_colony_labeling.py` — pure Voronoi-labeling functions: `filter_mask_by_overlap`, `markers_from_centroids`, `partition_by_grid_voronoi`.
- `src/phenotypic/sdk_/reconnect/_colony_reconnect.py` — `ReconnectConfig` dataclass + pure reconnection functions: `identify_pseudo_fragments`, `build_reconnect_cost`, `reconnect_fragments_tiled`, `compute_full_image_app2_gi_cost`, plus module-private `_generate_tiles`, `_process_tile`, `_merge_tile_into_output`, `_run_app2_gwdt_dijkstra`, `_apply_penalties_inplace`, `_APP2_NEIGHBORS`.
- `src/phenotypic/enhance/_two_k_phase_kernel.py` — `two_k_phase(...)` shared kernel returning `(gated_response, loose_result)`.
- `src/phenotypic/enhance/_focus_edge_two_k_phase.py` — `FocusEdgeTwoKPhase(NormalizedOutputMixin, FocusEdge)`.
- `src/phenotypic/detect/_two_k_filamentous_detector.py` — `TwoKFilamentousDetector(GridObjectDetector)`.
- Tests: `tests/unit/sdk_/reconnect/test_colony_labeling.py`, `tests/unit/sdk_/reconnect/test_colony_reconnect.py`, `tests/unit/detect/test_filamentous_fungi_regression.py`, `tests/unit/enhance/test_two_k_phase_kernel.py`, `tests/unit/enhance/test_focus_edge_two_k_phase.py`, `tests/unit/detect/test_two_k_filamentous_detector.py`.
- Fixture: `tests/fixtures/filamentous_fungi_regression_objmap.npy` (generated in Phase 0).

**Modified files:**
- `src/phenotypic/sdk_/reconnect/__init__.py` — re-export the new public functions + `ReconnectConfig`.
- `src/phenotypic/detect/_filamentous_fungi_detector.py` — Phase 3 refactor: delete migrated private methods, add `_reconnect_config()`, call the extracted functions.
- `src/phenotypic/enhance/__init__.py` — export `FocusEdgeTwoKPhase`.
- `src/phenotypic/detect/__init__.py` — export `TwoKFilamentousDetector`.
- `tests/unit/abc_/test_enhancer_taxonomy.py` — add `"FocusEdgeTwoKPhase"` to the `FocusEdge` tuple.

**Reference (read-only) — the extraction sources in `_filamentous_fungi_detector.py`:**
- `_operate` Phases 3–5: lines 577–655.
- `_identify_pseudo_fragments` 691–732 · `_apply_penalties_inplace` 734–750 · `_build_cost_surface` 752–805 · `_compute_full_image_app2_gi_cost` 807–835 · `_reconnect_fragments_tiled` 837–923 · `_generate_tiles` 925–959 · `_process_tile` 961–1082 · `_merge_tile_into_output` 1084–1105 · `_filter_mask_by_overlap` 1109–1142 · `_create_markers_from_centroids` 1144–1165 · `_separate_colonies` 1167–1182 · module-level `_run_app2_gwdt_dijkstra` + `_APP2_NEIGHBORS` 63–172.

---

## Phase 0 — FFD regression safety net

Establishes the golden fixture that gates the Phase 3 refactor. Must land **before** any FFD change.

### Task 0.1: Golden regression fixture + test for `FilamentousFungiDetector`

**Files:**
- Create: `tests/unit/detect/test_filamentous_fungi_regression.py`
- Create (generated): `tests/fixtures/filamentous_fungi_regression_objmap.npy`

**Interfaces:**
- Consumes: `phenotypic.data.load_synth_filamentous_plate`, `phenotypic.detect.FilamentousFungiDetector`, `phenotypic.detect.OtsuDetector`.
- Produces: a pinned `objmap` array that Phase 3 must reproduce exactly.

- [ ] **Step 1: Write the regression test (fixture-missing = fail, never skip)**

```python
# tests/unit/detect/test_filamentous_fungi_regression.py
"""Golden characterization test pinning FilamentousFungiDetector's objmap.

Guards the Phase-3 extraction refactor: the detector's output on a fixed
synthetic plate + fixed config must not change when its Phase 3-5 helpers
move into sdk_.reconnect.
"""
from pathlib import Path

import numpy as np

from phenotypic.data import load_synth_filamentous_plate
from phenotypic.detect import FilamentousFungiDetector, OtsuDetector

FIXTURE = Path(__file__).parent.parent.parent / "fixtures" / "filamentous_fungi_regression_objmap.npy"


def _run_detector() -> np.ndarray:
    image = load_synth_filamentous_plate().copy()
    detector = FilamentousFungiDetector(inoculum_detector=OtsuDetector(ignore_zeros=True))
    result = detector.apply(image, inplace=False)
    return np.asarray(result.objmap[:])


def test_filamentous_fungi_objmap_matches_golden():
    # A missing fixture must FAIL loudly (regenerate via the __main__ block), not skip.
    assert FIXTURE.exists(), (
        f"Golden fixture missing: {FIXTURE}. Regenerate with "
        f"`uv run python -m tests.unit.detect.test_filamentous_fungi_regression`."
    )
    expected = np.load(FIXTURE)
    actual = _run_detector()
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert np.array_equal(actual, expected), (
        f"objmap changed: {int((actual != expected).sum())} pixels differ"
    )


if __name__ == "__main__":  # regeneration entrypoint (run intentionally, then commit the .npy)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.save(FIXTURE, _run_detector())
    print(f"wrote {FIXTURE}")
```

- [ ] **Step 2: Run the test to confirm it fails on the missing fixture**

Run: `uv run pytest tests/unit/detect/test_filamentous_fungi_regression.py -v`
Expected: FAIL with the "Golden fixture missing" assertion.

- [ ] **Step 3: Generate and commit the fixture**

Run: `uv run python -m tests.unit.detect.test_filamentous_fungi_regression`
Expected: prints `wrote .../filamentous_fungi_regression_objmap.npy`.

- [ ] **Step 4: Run the test to confirm it now passes**

Run: `uv run pytest tests/unit/detect/test_filamentous_fungi_regression.py -v`
Expected: PASS.

- [ ] **Step 5: Prove the gate can fail (test integrity)**

Temporarily edit `_run_detector` to use `edge_noise_threshold=4.0` (a real behavior change), rerun the test, confirm it FAILS, then revert. This proves the anchor does work.
Run: `uv run pytest tests/unit/detect/test_filamentous_fungi_regression.py -v` → expect FAIL, then revert and expect PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/unit/detect/test_filamentous_fungi_regression.py tests/fixtures/filamentous_fungi_regression_objmap.npy
git commit -m "test: pin FilamentousFungiDetector objmap as Phase-3 refactor gate"
```

---

## Phase 1 — Extract Voronoi labeling into `sdk_.reconnect`

Three pure functions moved verbatim from FFD static methods (they already take arrays and reference no `self`).

### Task 1.1: `_colony_labeling.py` — `filter_mask_by_overlap`, `markers_from_centroids`, `partition_by_grid_voronoi`

**Files:**
- Create: `src/phenotypic/sdk_/reconnect/_colony_labeling.py`
- Test: `tests/unit/sdk_/reconnect/test_colony_labeling.py`
- Modify: `src/phenotypic/sdk_/reconnect/__init__.py`

**Interfaces:**
- Produces:
  - `filter_mask_by_overlap(mask: np.ndarray, reference_mask: np.ndarray) -> np.ndarray` — binary mask keeping only CCs of `mask` that overlap `reference_mask`.
  - `markers_from_centroids(objmap: np.ndarray) -> np.ndarray` — int32 marker array, one seed at each positive label's centroid.
  - `partition_by_grid_voronoi(markers: np.ndarray, mask: np.ndarray) -> np.ndarray` — Euclidean-Voronoi label map (`euclidean_voronoi_assign` + `connectivity_correct_labels`).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/sdk_/reconnect/test_colony_labeling.py
import numpy as np

from phenotypic.sdk_.reconnect import (
    filter_mask_by_overlap,
    markers_from_centroids,
    partition_by_grid_voronoi,
)


def test_filter_mask_by_overlap_drops_non_overlapping_cc():
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:5, 2:5] = True          # CC A — overlaps reference
    mask[12:15, 12:15] = True      # CC B — does not
    ref = np.zeros((20, 20), dtype=bool)
    ref[3, 3] = True
    out = filter_mask_by_overlap(mask, ref)
    assert out[3, 3]                # A kept
    assert not out[13, 13]         # B dropped


def test_markers_from_centroids_one_seed_per_label():
    objmap = np.zeros((20, 20), dtype=np.int32)
    objmap[2:6, 2:6] = 1
    objmap[12:16, 12:16] = 2
    markers = markers_from_centroids(objmap)
    assert markers.dtype == np.int32
    assert set(np.unique(markers)) == {0, 1, 2}
    assert int(markers[np.array([3, 4]).mean().round().astype(int), 3]) or markers[markers > 0].size == 2


def test_partition_by_grid_voronoi_labels_two_blobs():
    mask = np.zeros((20, 40), dtype=bool)
    mask[8:12, 4:8] = True
    mask[8:12, 32:36] = True
    markers = np.zeros((20, 40), dtype=np.int32)
    markers[10, 6] = 1
    markers[10, 34] = 2
    labels = partition_by_grid_voronoi(markers, mask)
    assert set(np.unique(labels[mask])) == {1, 2}
    assert labels[10, 6] == 1 and labels[10, 34] == 2
```

- [ ] **Step 2: Run to verify import failure**

Run: `uv run pytest tests/unit/sdk_/reconnect/test_colony_labeling.py -v`
Expected: FAIL with `ImportError: cannot import name 'filter_mask_by_overlap'`.

- [ ] **Step 3: Create the module (move bodies verbatim)**

Move the three FFD static-method bodies verbatim (they use no `self`), stripping the `@staticmethod` decorator and renaming:
- `_filter_mask_by_overlap` (FFD 1109–1142) → `filter_mask_by_overlap`
- `_create_markers_from_centroids` (FFD 1144–1165) → `markers_from_centroids`
- `_separate_colonies` (FFD 1167–1182) → `partition_by_grid_voronoi`

```python
# src/phenotypic/sdk_/reconnect/_colony_labeling.py
"""Grid-Voronoi labeling helpers for filamentous-colony detection.

Pure array functions extracted from FilamentousFungiDetector. No Image,
operation, or GUI types (see package CLAUDE.md import contract).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import center_of_mass
from skimage.measure import label

from ..branch_pathfinding import connectivity_correct_labels, euclidean_voronoi_assign


def filter_mask_by_overlap(mask: np.ndarray, reference_mask: np.ndarray) -> np.ndarray:
    """Retain only connected components of ``mask`` that overlap ``reference_mask``.

    Args:
        mask: Binary mask to filter (2D boolean or uint8).
        reference_mask: Binary mask defining valid regions.

    Returns:
        Filtered binary mask, same dtype/shape as ``mask``.
    """
    labeled = label(mask)
    min_h = min(mask.shape[0], reference_mask.shape[0])
    min_w = min(mask.shape[1], reference_mask.shape[1])
    intersection = labeled[:min_h, :min_w] * reference_mask[:min_h, :min_w]
    overlapping_labels = np.unique(intersection[intersection > 0])
    max_label = int(labeled.max())
    keep = np.zeros(max_label + 1, dtype=labeled.dtype)
    keep[overlapping_labels] = overlapping_labels
    return keep[labeled].astype(mask.dtype, copy=False)


def markers_from_centroids(objmap: np.ndarray) -> np.ndarray:
    """Create Voronoi seed markers at each positive label's centroid.

    Args:
        objmap: Labeled integer array (each object a unique positive ID).

    Returns:
        2D int32 marker array with one seed per centroid.
    """
    labels = np.unique(objmap)
    labels = labels[labels > 0]
    markers = np.zeros(objmap.shape, dtype=np.int32)
    for marker_id, lbl in enumerate(labels, start=1):
        com = center_of_mass(objmap == lbl)
        r = min(int(round(com[0])), objmap.shape[0] - 1)
        c = min(int(round(com[1])), objmap.shape[1] - 1)
        markers[r, c] = marker_id
    return markers


def partition_by_grid_voronoi(markers: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Voronoi-partition ``mask`` pixels by nearest ``markers`` seed and correct connectivity."""
    voronoi_map = euclidean_voronoi_assign(
        markers=markers, mask=mask, restrict_to_seeded_cc=False,
    )
    return connectivity_correct_labels(
        voronoi_labels=voronoi_map, mask=mask, markers=markers,
    )
```

- [ ] **Step 4: Re-export from the package `__init__`**

In `src/phenotypic/sdk_/reconnect/__init__.py`, add after the existing imports (line 13):
```python
from ._colony_labeling import (
    filter_mask_by_overlap,
    markers_from_centroids,
    partition_by_grid_voronoi,
)
```
and add `"filter_mask_by_overlap"`, `"markers_from_centroids"`, `"partition_by_grid_voronoi"` to `__all__`.

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/unit/sdk_/reconnect/test_colony_labeling.py -v`
Expected: PASS (all 3 tests).

- [ ] **Step 6: Confirm import purity still holds**

Run: `uv run pytest tests/unit/sdk_/reconnect/test_import_rules.py -v`
Expected: PASS (the new module imports only numpy/scipy/skimage/branch_pathfinding).

- [ ] **Step 7: Commit**

```bash
git add src/phenotypic/sdk_/reconnect/_colony_labeling.py src/phenotypic/sdk_/reconnect/__init__.py tests/unit/sdk_/reconnect/test_colony_labeling.py
git commit -m "feat(reconnect): extract grid-Voronoi colony labeling helpers"
```

---

## Phase 2 — Extract cost surface + tiled reconnection into `sdk_.reconnect`

The heavy extraction: the `ReconnectConfig` dataclass plus the cost-surface builder, pseudo-fragment identification, and tiled Dijkstra reconnection — all as pure array functions.

### Task 2.1: `ReconnectConfig` + `identify_pseudo_fragments`

**Files:**
- Create: `src/phenotypic/sdk_/reconnect/_colony_reconnect.py`
- Test: `tests/unit/sdk_/reconnect/test_colony_reconnect.py`
- Modify: `src/phenotypic/sdk_/reconnect/__init__.py`

**Interfaces:**
- Produces:
  - `ReconnectConfig` — frozen dataclass holding the 14 scalar reconnection parameters (see Step 3).
  - `identify_pseudo_fragments(colony_labels: np.ndarray, center_objmask: np.ndarray) -> tuple[np.ndarray, np.ndarray]` — `(central_mask, fragment_labels)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/sdk_/reconnect/test_colony_reconnect.py
import numpy as np

from phenotypic.sdk_.reconnect import ReconnectConfig, identify_pseudo_fragments


def _cfg(**kw) -> ReconnectConfig:
    base = dict(
        beta=2.0, gamma=1.2, delta=1.0, coherence_window_radius=15, mad_window=7,
        gap_crossing_penalty=4.0, border_margin_px=50, frag_reach_px=10,
        tile_size=1200, tile_overlap=500, max_gap_length=30,
        path_dilation_radius=2, snr_margin=5, reconnection_tolerance=2.5,
    )
    base.update(kw)
    return ReconnectConfig(**base)


def test_reconnect_config_is_frozen():
    cfg = _cfg()
    assert cfg.beta == 2.0 and cfg.tile_size == 1200
    import dataclasses, pytest
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.beta = 3.0  # type: ignore[misc]


def test_identify_pseudo_fragments_splits_central_and_fragments():
    labels = np.zeros((30, 30), dtype=np.int32)
    labels[4:8, 4:8] = 1            # CC touching a center
    labels[20:24, 20:24] = 1        # CC NOT touching any center -> fragment
    center = np.zeros((30, 30), dtype=bool)
    center[6, 6] = True
    central_mask, fragment_labels = identify_pseudo_fragments(labels, center)
    assert central_mask[6, 6]
    assert not central_mask[22, 22]
    assert fragment_labels[22, 22] > 0
    assert fragment_labels[6, 6] == 0
```

- [ ] **Step 2: Run to verify import failure**

Run: `uv run pytest tests/unit/sdk_/reconnect/test_colony_reconnect.py -v`
Expected: FAIL with `ImportError: cannot import name 'ReconnectConfig'`.

- [ ] **Step 3: Create the module skeleton with `ReconnectConfig` + `identify_pseudo_fragments`**

`ReconnectConfig` fields are exactly the `self.*`/ClassVar values that FFD's Phase 3–5 helpers read. `identify_pseudo_fragments` is moved verbatim from FFD 691–732 (already a `@staticmethod`).

```python
# src/phenotypic/sdk_/reconnect/_colony_reconnect.py
"""Composite-cost + tiled Dijkstra reconnection for filamentous colonies.

Pure array functions extracted from FilamentousFungiDetector. Callers pass
already-computed phase-congruency arrays (pc_sum, M, m, orientation) and a
ReconnectConfig of scalar parameters — never an Image, operation, or the
_PhaseCong3Result dataclass (see package CLAUDE.md import contract).
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label as ndi_label
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import dilation, disk
from skimage.segmentation import find_boundaries

from ..branch_pathfinding import (
    DijkstraResult,
    _apply_border_penalty_inplace,
    _apply_distance_gap_penalty_inplace,
    _apply_structure_mask_inplace,
    _compute_screening_envelope,
    apply_filter_cascade,
    assemble_composite_cost,
    assign_fragments_to_colonies,
    calibrate_screening_threshold,
    calibrate_thresholds,
    compute_anisotropy,
    compute_local_mad_map,
    compute_orientation_coherence,
    extract_calibration_branches,
    extract_fragment_paths,
    prescreen_fragments,
    run_multisource_dijkstra,
)


@dataclass(frozen=True)
class ReconnectConfig:
    """Scalar parameters for cost-surface assembly + tiled Dijkstra reconnection.

    All values are pre-derived by the caller (e.g. a detector's scene-derivation
    validator). Field meanings mirror the identically named FilamentousFungiDetector
    fields / ClassVars.
    """
    beta: float                     # anisotropy exponent in composite cost
    gamma: float                    # MAD penalty weight in composite cost numerator
    delta: float                    # Dijkstra radial retreat penalty
    coherence_window_radius: int
    mad_window: int
    gap_crossing_penalty: float
    border_margin_px: int
    frag_reach_px: int
    tile_size: int
    tile_overlap: int
    max_gap_length: int
    path_dilation_radius: int
    snr_margin: int
    reconnection_tolerance: float


def identify_pseudo_fragments(
    colony_labels: np.ndarray,
    center_objmask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Identify pseudo-fragments: per-label CCs that don't overlap the inoculum.

    Returns (central_mask, fragment_labels) where central_mask is the main
    colony mass (CCs overlapping ``center_objmask``) and fragment_labels is a
    labeled map of the disconnected blobs.
    """
    foreground = colony_labels > 0
    cc_map, n_cc = ndi_label(foreground)
    if n_cc == 0:
        return (np.zeros_like(foreground),
                np.zeros(foreground.shape, dtype=np.int32))
    seeded_ccs = np.unique(cc_map[center_objmask & foreground])
    is_central = np.zeros(n_cc + 1, dtype=bool)
    is_central[seeded_ccs] = True
    central_mask = is_central[cc_map]
    fragment_mask = foreground & ~central_mask
    if fragment_mask.any():
        fragment_labels = label(fragment_mask).astype(np.int32)
    else:
        fragment_labels = np.zeros(foreground.shape, dtype=np.int32)
    return central_mask, fragment_labels
```

- [ ] **Step 4: Re-export `ReconnectConfig` + `identify_pseudo_fragments`**

In `src/phenotypic/sdk_/reconnect/__init__.py`, add:
```python
from ._colony_reconnect import ReconnectConfig, identify_pseudo_fragments
```
and add both names to `__all__`.

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/unit/sdk_/reconnect/test_colony_reconnect.py -v`
Expected: PASS (both tests).

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/sdk_/reconnect/_colony_reconnect.py src/phenotypic/sdk_/reconnect/__init__.py tests/unit/sdk_/reconnect/test_colony_reconnect.py
git commit -m "feat(reconnect): add ReconnectConfig + identify_pseudo_fragments"
```

### Task 2.2: `build_reconnect_cost` (cost surface from raw PCT arrays)

**Files:**
- Modify: `src/phenotypic/sdk_/reconnect/_colony_reconnect.py`
- Modify: `tests/unit/sdk_/reconnect/test_colony_reconnect.py`
- Modify: `src/phenotypic/sdk_/reconnect/__init__.py`

**Interfaces:**
- Consumes: `ReconnectConfig`.
- Produces:
  - `_apply_penalties_inplace(cost, pct_energy, colony_labels, cfg) -> None` (module-private).
  - `build_reconnect_cost(pc_sum, M, m, orientation, enhanced_arr, colony_labels, central_mask, cfg) -> tuple[np.ndarray, np.ndarray]` — `(unmasked_cost, cost_surface)`. Same math as FFD `_build_cost_surface` but taking the four PCT arrays separately instead of `_PhaseCong3Result`.

- [ ] **Step 1: Write the failing test (append)**

```python
def test_build_reconnect_cost_shapes_and_finiteness():
    from phenotypic.sdk_.reconnect import build_reconnect_cost
    rng = np.random.default_rng(0)
    H, W = 40, 40
    pc_sum = rng.random((H, W)).astype(np.float32) * 0.15
    M = rng.random((H, W)).astype(np.float32)
    m = rng.random((H, W)).astype(np.float32) * 0.5
    orientation = (rng.random((H, W)).astype(np.float32) - 0.5) * np.pi
    enhanced = rng.random((H, W)).astype(np.float32)
    colony_labels = np.zeros((H, W), dtype=np.int32)
    colony_labels[10:14, 10:30] = 1
    central = colony_labels > 0
    unmasked, masked = build_reconnect_cost(
        pc_sum, M, m, orientation, enhanced, colony_labels, central, _cfg(mad_window=7),
    )
    assert unmasked.shape == masked.shape == (H, W)
    assert np.all(np.isfinite(unmasked))
    # inside the colony/central mask the masked surface is driven to near-zero traversal cost
    assert masked[12, 20] < unmasked[12, 20] or masked[12, 20] == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/sdk_/reconnect/test_colony_reconnect.py::test_build_reconnect_cost_shapes_and_finiteness -v`
Expected: FAIL with `ImportError: cannot import name 'build_reconnect_cost'`.

- [ ] **Step 3: Add the functions (move `_build_cost_surface` + `_apply_penalties_inplace`, adapt args)**

Move FFD `_apply_penalties_inplace` (734–750) and `_build_cost_surface` (752–805) into `_colony_reconnect.py`, applying these substitutions: `self.gap_crossing_penalty→cfg.gap_crossing_penalty`, `self.border_margin_px→cfg.border_margin_px`, `self.coherence_window_radius→cfg.coherence_window_radius`, `self.mad_window→cfg.mad_window`, `self.beta→cfg.beta`, `self.gamma→cfg.gamma`, and replace the `pct_result.*` reads with the corresponding array parameters.

```python
def _apply_penalties_inplace(
    cost: np.ndarray,
    pct_energy: np.ndarray,
    colony_labels: np.ndarray,
    cfg: ReconnectConfig,
) -> None:
    """Apply distance-gap and border penalties in place."""
    _apply_distance_gap_penalty_inplace(
        cost, pct_energy, colony_labels, cfg.gap_crossing_penalty,
    )
    _apply_border_penalty_inplace(cost, cfg.border_margin_px)


def build_reconnect_cost(
    pc_sum: np.ndarray,
    M: np.ndarray,
    m: np.ndarray,
    orientation: np.ndarray,
    enhanced_arr: np.ndarray,
    colony_labels: np.ndarray,
    central_mask: np.ndarray,
    cfg: ReconnectConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the composite cost surface from phase-congruency feature arrays.

    Args:
        pc_sum, M, m, orientation: Phase-congruency arrays (the fields of a
            _PhaseCong3Result, passed separately to keep this package pure).
        enhanced_arr: 2D contrast-stretched detection matrix for local MAD.
        colony_labels: Labeled colony assignment.
        central_mask: Boolean mask of branch pixels overlapping colonies.
        cfg: Scalar reconnection parameters.

    Returns:
        (unmasked_cost, cost_surface). ``cost_surface`` has colony/central
        pixels set to near-zero traversal cost.
    """
    anisotropy = compute_anisotropy(M, m)
    coherence = compute_orientation_coherence(orientation, cfg.coherence_window_radius)
    mad = compute_local_mad_map(enhanced_arr, cfg.mad_window)
    base_cost = assemble_composite_cost(pc_sum, anisotropy, coherence, mad, cfg.beta, cfg.gamma)

    unmasked_cost = base_cost.copy()
    _apply_penalties_inplace(unmasked_cost, pc_sum, colony_labels, cfg)

    colony_mask = (colony_labels > 0) | central_mask
    _apply_structure_mask_inplace(base_cost, colony_mask.astype(np.int32))
    _apply_penalties_inplace(base_cost, pc_sum, colony_labels, cfg)
    return unmasked_cost, base_cost
```

- [ ] **Step 4: Re-export `build_reconnect_cost`**

Add `build_reconnect_cost` to the `from ._colony_reconnect import ...` line and `__all__` in `src/phenotypic/sdk_/reconnect/__init__.py`.

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/unit/sdk_/reconnect/test_colony_reconnect.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/sdk_/reconnect/_colony_reconnect.py src/phenotypic/sdk_/reconnect/__init__.py tests/unit/sdk_/reconnect/test_colony_reconnect.py
git commit -m "feat(reconnect): add build_reconnect_cost (pure PCT-array cost surface)"
```

### Task 2.3: `reconnect_fragments_tiled` + `compute_full_image_app2_gi_cost` + tile internals

**Files:**
- Modify: `src/phenotypic/sdk_/reconnect/_colony_reconnect.py`
- Modify: `tests/unit/sdk_/reconnect/test_colony_reconnect.py`
- Modify: `src/phenotypic/sdk_/reconnect/__init__.py`

**Interfaces:**
- Consumes: `ReconnectConfig`, `build_reconnect_cost`.
- Produces:
  - `reconnect_fragments_tiled(colony_labels, fragment_labels, cost_surface, unmasked_cost, pct_energy, grayscale, cfg, app2_gi_cost=None) -> np.ndarray`.
  - `compute_full_image_app2_gi_cost(image, *, background) -> np.ndarray`.
  - Module-private `_generate_tiles`, `_process_tile`, `_merge_tile_into_output`, `_run_app2_gwdt_dijkstra`, `_APP2_NEIGHBORS`.

- [ ] **Step 1: Write the failing test (append) — a broken bridge that reconnects**

```python
def test_reconnect_fragments_tiled_bridges_gap():
    from phenotypic.sdk_.reconnect import build_reconnect_cost, reconnect_fragments_tiled, identify_pseudo_fragments
    H, W = 60, 60
    # a colony bar + a nearby fragment separated by a 3px gap, both on a low-cost ridge
    colony = np.zeros((H, W), dtype=np.int32)
    colony[28:32, 8:28] = 1
    frag_src = np.zeros((H, W), dtype=bool)
    frag_src[28:32, 31:50] = True
    branch = (colony > 0) | frag_src
    # high pc_sum ridge along the bar + fragment + gap so Dijkstra can route cheaply
    pc = np.zeros((H, W), dtype=np.float32)
    pc[28:32, 8:50] = 0.6
    M = pc.copy(); m = pc * 0.2
    orient = np.zeros((H, W), dtype=np.float32)  # all-horizontal
    enhanced = pc.copy()
    central_mask, fragment_labels = identify_pseudo_fragments(
        np.where(branch, 1, 0).astype(np.int32), colony > 0,
    )
    cfg = _cfg(tile_size=60, tile_overlap=20, mad_window=5, frag_reach_px=8, max_gap_length=20, border_margin_px=2)
    unmasked, cost = build_reconnect_cost(pc, M, m, orient, enhanced, colony, central_mask, cfg)
    out = reconnect_fragments_tiled(
        colony, fragment_labels, cost, unmasked,
        pc.astype(np.float32), enhanced.astype(np.float32), cfg,
    )
    # the fragment pixels get painted with the colony id (1) after reconnection
    assert out[29, 40] == 1


def test_reconnect_fragments_tiled_noop_without_fragments():
    from phenotypic.sdk_.reconnect import reconnect_fragments_tiled
    colony = np.zeros((20, 20), dtype=np.int32)
    colony[5:9, 5:15] = 1
    empty_frags = np.zeros((20, 20), dtype=np.int32)
    cost = np.ones((20, 20), dtype=np.float64)
    out = reconnect_fragments_tiled(colony, empty_frags, cost, cost, np.zeros((20, 20), np.float32), np.zeros((20, 20), np.float32), _cfg())
    assert np.array_equal(out, colony)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/sdk_/reconnect/test_colony_reconnect.py -k tiled -v`
Expected: FAIL with `ImportError: cannot import name 'reconnect_fragments_tiled'`.

- [ ] **Step 3: Move the tiling machinery**

Into `_colony_reconnect.py`, move verbatim (adjusting as noted):
- Module-level `_APP2_NEIGHBORS` (FFD 63–72) and `_run_app2_gwdt_dijkstra` (FFD 75–172) — verbatim, no substitutions (already free functions using only arrays + `find_boundaries`/`heapq`/`itertools`/`DijkstraResult`). Add `grey_weighted_distance, app2_gwdt_cost` import: `from ._gwdt import app2_gwdt_cost, grey_weighted_distance`.
- `_generate_tiles` (FFD 925–959) — verbatim `@staticmethod` → module function `_generate_tiles(image_shape, tile_size, overlap)`.
- `_merge_tile_into_output` (FFD 1084–1105) — verbatim `@staticmethod` → module function.
- `_compute_full_image_app2_gi_cost` (FFD 807–835) — verbatim `@staticmethod` → **public** `compute_full_image_app2_gi_cost(image, *, background)`.
- `_process_tile` (FFD 961–1082) → module function `_process_tile(tile_cost, tile_raw, tile_colony, tile_frags, tile_pct, tile_gray, pct_noise_ceil, cfg, tile_app2_gi=None)`. Substitutions: `self.delta→cfg.delta`, `self.max_gap_length→cfg.max_gap_length`, `self.path_dilation_radius→cfg.path_dilation_radius`, `self.snr_margin→cfg.snr_margin`, `self.reconnection_tolerance→cfg.reconnection_tolerance`; `_run_app2_gwdt_dijkstra` now the module function.
- `_reconnect_fragments_tiled` (FFD 837–923) → public `reconnect_fragments_tiled(colony_labels, fragment_labels, cost_surface, unmasked_cost, pct_energy, grayscale, cfg, app2_gi_cost=None)`. Substitutions: `self.frag_reach_px→cfg.frag_reach_px`, `self.tile_size→cfg.tile_size`, `self.tile_overlap→cfg.tile_overlap`, `self._generate_tiles(...)→_generate_tiles(...)`, `self._process_tile(...)→_process_tile(..., cfg, tile_app2_gi)` (thread `cfg` through), `self._merge_tile_into_output(...)→_merge_tile_into_output(...)`.

Show the two public signatures (bodies are the verbatim moves above):
```python
def compute_full_image_app2_gi_cost(image: np.ndarray, *, background: np.ndarray) -> np.ndarray:
    ...  # FFD 807-835 body verbatim

def reconnect_fragments_tiled(
    colony_labels: np.ndarray,
    fragment_labels: np.ndarray,
    cost_surface: np.ndarray,
    unmasked_cost: np.ndarray,
    pct_energy: np.ndarray,
    grayscale: np.ndarray,
    cfg: ReconnectConfig,
    app2_gi_cost: np.ndarray | None = None,
) -> np.ndarray:
    ...  # FFD 837-923 body with the substitutions above
```

- [ ] **Step 4: Re-export the two public functions**

Add `reconnect_fragments_tiled` and `compute_full_image_app2_gi_cost` to the `_colony_reconnect` import + `__all__` in `src/phenotypic/sdk_/reconnect/__init__.py`.

- [ ] **Step 5: Run the reconnect tests**

Run: `uv run pytest tests/unit/sdk_/reconnect/test_colony_reconnect.py -v`
Expected: PASS (all tests, including the bridge + no-op cases).

- [ ] **Step 6: Confirm import purity + no regressions in existing reconnect tests**

Run: `uv run pytest tests/unit/sdk_/reconnect/ -v`
Expected: PASS, including `test_import_rules.py`.

- [ ] **Step 7: Commit**

```bash
git add src/phenotypic/sdk_/reconnect/_colony_reconnect.py src/phenotypic/sdk_/reconnect/__init__.py tests/unit/sdk_/reconnect/test_colony_reconnect.py
git commit -m "feat(reconnect): add tiled Dijkstra reconnection (reconnect_fragments_tiled)"
```

---

## Phase 3 — Refactor `FilamentousFungiDetector` to consume `sdk_.reconnect`

Replace FFD's Phase 3–5 private methods with calls to the extracted functions. Behavior must be bit-identical (Phase 0 gate).

### Task 3.1: Add `_reconnect_config()` and rewire `_operate` Phases 3–5

**Files:**
- Modify: `src/phenotypic/detect/_filamentous_fungi_detector.py`

**Interfaces:**
- Consumes (from Phases 1–2): `filter_mask_by_overlap`, `markers_from_centroids`, `partition_by_grid_voronoi`, `identify_pseudo_fragments`, `build_reconnect_cost`, `reconnect_fragments_tiled`, `compute_full_image_app2_gi_cost`, `ReconnectConfig`.
- Produces: `FilamentousFungiDetector._reconnect_config(self) -> ReconnectConfig`.

- [ ] **Step 1: Confirm the gate is green before touching FFD**

Run: `uv run pytest tests/unit/detect/test_filamentous_fungi_regression.py -v`
Expected: PASS (baseline).

- [ ] **Step 2: Update imports in `_filamentous_fungi_detector.py`**

Replace the `from phenotypic.sdk_.branch_pathfinding import (...)` block (lines 39–59) so it imports only what Phase 2 still leaves in the detector's own Phase-2 branch stage — namely nothing from the reconnection helpers. Concretely: delete the now-migrated names (`_apply_distance_gap_penalty_inplace`, `_apply_border_penalty_inplace`, `_apply_structure_mask_inplace`, `_compute_screening_envelope`, `compute_anisotropy`, `compute_orientation_coherence`, `compute_local_mad_map`, `assemble_composite_cost`, `calibrate_screening_threshold`, `prescreen_fragments`, `run_multisource_dijkstra`, `assign_fragments_to_colonies`, `extract_fragment_paths`, `extract_calibration_branches`, `calibrate_thresholds`, `apply_filter_cascade`, `euclidean_voronoi_assign`, `connectivity_correct_labels`) and the `DijkstraResult` import (line 59) and the `app2_gwdt_cost, grey_weighted_distance` import (line 60). Add:
```python
from phenotypic.sdk_.reconnect import (
    ReconnectConfig,
    build_reconnect_cost,
    compute_full_image_app2_gi_cost,
    filter_mask_by_overlap,
    identify_pseudo_fragments,
    markers_from_centroids,
    partition_by_grid_voronoi,
    reconnect_fragments_tiled,
)
```
Also remove now-unused stdlib/skimage imports that only the migrated code used: `heapq`, `itertools` (lines 2–3), `find_boundaries` (line 20), `center_of_mass` (from line 16 — keep `label as ndi_label`? it becomes unused too), `disk, dilation` (line 19), `threshold_otsu` (line 17 — check no other user). Verify with a grep in Step 6 before deleting; keep any still referenced by the retained Phase-2 code.

- [ ] **Step 3: Delete the migrated methods and module-level app2 helpers**

Delete from `_filamentous_fungi_detector.py`: module-level `_APP2_NEIGHBORS` (63–72) and `_run_app2_gwdt_dijkstra` (75–172); and methods `_identify_pseudo_fragments` (691–732), `_apply_penalties_inplace` (734–750), `_build_cost_surface` (752–805), `_compute_full_image_app2_gi_cost` (807–835), `_reconnect_fragments_tiled` (837–923), `_generate_tiles` (925–959), `_process_tile` (961–1082), `_merge_tile_into_output` (1084–1105), `_filter_mask_by_overlap` (1109–1142), `_create_markers_from_centroids` (1144–1165), `_separate_colonies` (1167–1182). Keep the Phase-2 helpers `_subtract_background` and `_combine_bg_removed_with_pct`.

- [ ] **Step 4: Add the `_reconnect_config` builder**

Add this method to `FilamentousFungiDetector` (place near the other helpers). It reads the already-scene-derived fields + ClassVars:
```python
    def _reconnect_config(self) -> ReconnectConfig:
        """Bundle scene-derived scalars for the sdk_.reconnect functions."""
        return ReconnectConfig(
            beta=self.beta,
            gamma=self.gamma,
            delta=self.delta,
            coherence_window_radius=self.coherence_window_radius,
            mad_window=self.mad_window,
            gap_crossing_penalty=self.gap_crossing_penalty,
            border_margin_px=self.border_margin_px,
            frag_reach_px=self.frag_reach_px,
            tile_size=self.tile_size,
            tile_overlap=self.tile_overlap,
            max_gap_length=self.max_gap_length,
            path_dilation_radius=self.path_dilation_radius,
            snr_margin=self.snr_margin,
            reconnection_tolerance=self.reconnection_tolerance,
        )
```

- [ ] **Step 5: Rewire `_operate` Phases 3–5 (lines 577–655) to call the extracted functions**

Apply these call-site replacements inside `_operate` (leave Phases 1–2 unchanged):
- `self._filter_mask_by_overlap(mask=overall_objmask, reference_mask=inoculum_objmask)` → `filter_mask_by_overlap(overall_objmask, inoculum_objmask)`
- `self._create_markers_from_centroids(inoculum_img.objmap[:])` → `markers_from_centroids(inoculum_img.objmap[:])`
- `self._separate_colonies(centroid_markers, inoculum_structure_mask)` → `partition_by_grid_voronoi(centroid_markers, inoculum_structure_mask)`
- `self._identify_pseudo_fragments(colony_labels=colony_labels, center_objmask=inoculum_objmask)` → `identify_pseudo_fragments(colony_labels, inoculum_objmask)`
- the `app2_gi_cost` branch: `self._compute_full_image_app2_gi_cost(enhanced_arr, background=~overall_objmask.astype(np.bool_, copy=False))` → `compute_full_image_app2_gi_cost(enhanced_arr, background=~overall_objmask.astype(np.bool_, copy=False))`
- `self._build_cost_surface(pct_result=pct_result, enhanced_arr=enhanced_arr, colony_labels=colony_labels, central_mask=central_mask)` → `build_reconnect_cost(pct_result.pc_sum, pct_result.M, pct_result.m, pct_result.orientation, enhanced_arr, colony_labels, central_mask, self._reconnect_config())`
- `self._reconnect_fragments_tiled(colony_labels=..., fragment_labels=..., cost_surface=..., unmasked_cost=..., pct_energy=pct_result.pc_sum.astype(np.float32), grayscale=enhanced_gray, app2_gi_cost=app2_gi_cost)` → `reconnect_fragments_tiled(colony_labels, fragment_labels, cost_surface, unmasked_cost, pct_result.pc_sum.astype(np.float32), enhanced_gray, self._reconnect_config(), app2_gi_cost=app2_gi_cost)`
- the final `self._separate_colonies(centroid_markers, final_mask)` → `partition_by_grid_voronoi(centroid_markers, final_mask)`

- [ ] **Step 6: Grep for stragglers and dead imports**

Run: `cd /Users/alex/Projects/PhenoTypic && grep -n "self\._\(build_cost_surface\|reconnect_fragments_tiled\|separate_colonies\|filter_mask_by_overlap\|create_markers_from_centroids\|identify_pseudo_fragments\|process_tile\|generate_tiles\|merge_tile_into_output\|apply_penalties_inplace\|compute_full_image_app2_gi_cost\)\|_run_app2_gwdt_dijkstra\|_APP2_NEIGHBORS" src/phenotypic/detect/_filamentous_fungi_detector.py`
Expected: no matches. Then confirm no unused-import errors via Step 7.

- [ ] **Step 7: Run the regression gate — must be bit-identical**

Run: `uv run pytest tests/unit/detect/test_filamentous_fungi_regression.py tests/unit/detect/test_filamentous_fungi_detector.py tests/unit/detect/test_filamentous_fungi_gwdt_seam.py tests/unit/detect/test_filamentous_fungi_shim.py -v`
Expected: PASS (objmap identical; existing FFD behavior/seam tests still green).

- [ ] **Step 8: Run the broader detect + sdk suites for collateral**

Run: `uv run pytest tests/unit/detect/ tests/unit/sdk_/ -q`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/phenotypic/detect/_filamentous_fungi_detector.py
git commit -m "refactor(detect): FFD consumes sdk_.reconnect for Phase 3-5 (behavior-identical)"
```

---

## Phase 4 — Class 1: two-`k` kernel + `FocusEdgeTwoKPhase` enhancer

### Task 4.1: `two_k_phase` shared kernel

**Files:**
- Create: `src/phenotypic/enhance/_two_k_phase_kernel.py`
- Test: `tests/unit/enhance/test_two_k_phase_kernel.py`

**Interfaces:**
- Produces: `two_k_phase(detect_mat, *, k_strict, k_loose, seed_thresh, cand_thresh, n_orient, min_wavelength) -> tuple[np.ndarray, _PhaseCong3Result]` — `(gated_response, loose_result)` where `gated_response = loose.pc_sum * hysteresis_mask` (continuous, center hole preserved) and `loose_result` is the loose-`k` `_PhaseCong3Result` (for a detector's cost surface).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/enhance/test_two_k_phase_kernel.py
import numpy as np

from phenotypic.enhance._two_k_phase_kernel import two_k_phase


def _synthetic_branches():
    # a bright ring (branches) around a dark core, on a mid-gray field with a faint outlier speck
    img = np.full((80, 80), 0.4, dtype=np.float32)
    yy, xx = np.ogrid[:80, :80]
    ring = ((yy - 40) ** 2 + (xx - 40) ** 2)
    img[(ring > 18 ** 2) & (ring < 24 ** 2)] = 0.95   # edge ring -> strong PCT
    img[70:72, 6:8] = 0.7                              # isolated faint speck (loose-only, no seed)
    return img


def test_two_k_phase_returns_gated_response_and_loose_result():
    img = _synthetic_branches()
    gated, loose = two_k_phase(
        img, k_strict=6.0, k_loose=4.5, seed_thresh="otsu",
        cand_thresh="triangle", n_orient=8, min_wavelength=5.0,
    )
    assert gated.shape == img.shape
    # loose result exposes the cost-surface arrays, all finite & same-shaped
    for arr in (loose.pc_sum, loose.M, loose.m, loose.orientation):
        assert arr.shape == img.shape
        assert np.all(np.isfinite(arr))
    # gated response is continuous (not just 0/1) where branches are confirmed
    nz = gated[gated > 0]
    assert nz.size > 0
    assert np.unique(nz).size > 2  # magnitudes, not a binary mask


def test_two_k_phase_rejects_isolated_loose_only_agar():
    img = _synthetic_branches()
    gated, _ = two_k_phase(
        img, k_strict=6.0, k_loose=4.5, seed_thresh="otsu",
        cand_thresh="triangle", n_orient=8, min_wavelength=5.0,
    )
    # the isolated faint speck has no strict seed -> rejected (stays 0)
    assert gated[70, 6] == 0.0


def test_two_k_phase_otsu_otsu_admits_no_more_than_hysteresis():
    # mutation guard: using otsu on candidates (instead of triangle) must not
    # recover MORE branch pixels than the loose-triangle hysteresis (it recovers fewer).
    img = _synthetic_branches()
    tri, _ = two_k_phase(img, k_strict=6.0, k_loose=4.5, seed_thresh="otsu",
                         cand_thresh="triangle", n_orient=8, min_wavelength=5.0)
    ott, _ = two_k_phase(img, k_strict=6.0, k_loose=4.5, seed_thresh="otsu",
                         cand_thresh="otsu", n_orient=8, min_wavelength=5.0)
    assert (tri > 0).sum() >= (ott > 0).sum()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/enhance/test_two_k_phase_kernel.py -v`
Expected: FAIL with `ModuleNotFoundError`/`ImportError` for `_two_k_phase_kernel`.

- [ ] **Step 3: Implement the kernel**

```python
# src/phenotypic/enhance/_two_k_phase_kernel.py
"""Two-scale-k phase-congruency hysteresis kernel.

Runs phase congruency at a strict and a loose noise-threshold k, keeps the loose
candidates that touch a strict seed (morphological reconstruction), and returns the
loose-k magnitude gated by that hysteresis mask. Shared by FocusEdgeTwoKPhase (the
enhancer) and TwoKFilamentousDetector (which also reuses the returned loose result
for its Dijkstra cost surface, so phase congruency runs only twice total).
"""
from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
from skimage.filters import threshold_otsu, threshold_triangle
from skimage.morphology import reconstruction

from ._focus_edge_phase import FocusEdgePhase, _PhaseCong3Result

_THRESHOLDS = {"otsu": threshold_otsu, "triangle": threshold_triangle}


def two_k_phase(
    detect_mat: np.ndarray,
    *,
    k_strict: float,
    k_loose: float,
    seed_thresh: Literal["otsu", "triangle"],
    cand_thresh: Literal["otsu", "triangle"],
    n_orient: int,
    min_wavelength: float,
) -> Tuple[np.ndarray, _PhaseCong3Result]:
    """Two-k phase-congruency hysteresis.

    Args:
        detect_mat: Prepared (flattened + contrast-stretched) 2D detection matrix.
        k_strict: Strict noise-threshold k -> clean, fragmented seeds.
        k_loose: Loose noise-threshold k -> full branches + agar candidates.
        seed_thresh: Threshold rule for seeds (strict map). "otsu" verified best.
        cand_thresh: Threshold rule for candidates (loose map). "triangle" verified best.
        n_orient: Phase-congruency angular resolution.
        min_wavelength: Smallest log-Gabor wavelength (skips agar micro-texture).

    Returns:
        (gated_response, loose_result):
          - gated_response: loose.pc_sum where the hysteresis mask confirms a real
            branch, 0 elsewhere. Continuous magnitude; inoculum center hole preserved.
          - loose_result: the loose-k _PhaseCong3Result (M/m/orientation/pc_sum).
    """
    strict = FocusEdgePhase(
        n_orient=n_orient, k=k_strict, min_wavelength=min_wavelength,
    )._phasecong3(detect_mat)
    loose = FocusEdgePhase(
        n_orient=n_orient, k=k_loose, min_wavelength=min_wavelength,
    )._phasecong3(detect_mat)

    seed = strict.pc_sum > _THRESHOLDS[seed_thresh](strict.pc_sum)
    cand = loose.pc_sum > _THRESHOLDS[cand_thresh](loose.pc_sum)
    mask = reconstruction(
        (seed & cand).astype(np.uint8), cand.astype(np.uint8), method="dilation",
    ).astype(bool)

    gated = (loose.pc_sum * mask).astype(np.float32)
    return gated, loose
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/unit/enhance/test_two_k_phase_kernel.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/enhance/_two_k_phase_kernel.py tests/unit/enhance/test_two_k_phase_kernel.py
git commit -m "feat(enhance): add two_k_phase hysteresis kernel"
```

### Task 4.2: `FocusEdgeTwoKPhase` enhancer

**Files:**
- Create: `src/phenotypic/enhance/_focus_edge_two_k_phase.py`
- Test: `tests/unit/enhance/test_focus_edge_two_k_phase.py`
- Modify: `src/phenotypic/enhance/__init__.py`
- Modify: `tests/unit/abc_/test_enhancer_taxonomy.py`

**Interfaces:**
- Consumes: `two_k_phase`, `NormalizedOutputMixin`, `FocusEdge`.
- Produces: `FocusEdgeTwoKPhase` — a `FocusEdge` enhancer writing the gated two-`k` response into `detect_mat`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/enhance/test_focus_edge_two_k_phase.py
import numpy as np
import pytest
from pydantic import ValidationError

from phenotypic import Image
from phenotypic.enhance import FocusEdgeTwoKPhase


def _image_from(arr: np.ndarray) -> Image:
    """Wrap a float [0,1] array in an Image by broadcasting to uint8 RGB (codebase idiom)."""
    rgb = np.repeat((arr[..., None] * 255).astype(np.uint8), 3, axis=2)
    return Image(rgb)


def _plate():
    # a solid bright inoculum disk (its interior yields NO edges -> hole preserved) plus a
    # radiating branch line (an edge -> nonzero gated response)
    img = np.full((80, 80), 0.4, dtype=np.float32)
    yy, xx = np.ogrid[:80, :80]
    core = ((yy - 40) ** 2 + (xx - 40) ** 2) < 12 ** 2
    img[core] = 0.95                                   # solid core
    img[39:41, 40:74] = 0.9                            # a branch tendril
    return _image_from(img)


def test_defaults():
    e = FocusEdgeTwoKPhase()
    assert e.k_strict == 6.0 and e.k_loose == 4.5
    assert e.seed_thresh == "otsu" and e.cand_thresh == "triangle"
    assert e.n_orient == 8 and e.min_wavelength == 5.0


def test_writes_gated_response_into_detect_mat():
    im = _plate()
    out = FocusEdgeTwoKPhase().apply(im, inplace=False)
    dm = np.asarray(out.detect_mat[:])
    assert dm.shape == (80, 80)
    assert dm.max() > 0
    assert dm.min() == 0.0            # agar / background gated to zero


def test_center_hole_preserved():
    im = _plate()
    out = FocusEdgeTwoKPhase().apply(im, inplace=False)
    dm = np.asarray(out.detect_mat[:])
    assert dm[40, 40] == 0.0          # solid inoculum core stays a hole (edge detector)


@pytest.mark.parametrize("bad", [dict(k_strict=-1.0), dict(min_wavelength=1.0), dict(n_orient=0)])
def test_parameter_validation(bad):
    with pytest.raises(ValidationError):
        FocusEdgeTwoKPhase(**bad)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/enhance/test_focus_edge_two_k_phase.py -v`
Expected: FAIL with `ImportError: cannot import name 'FocusEdgeTwoKPhase'`.

- [ ] **Step 3: Implement the enhancer**

```python
# src/phenotypic/enhance/_focus_edge_two_k_phase.py
"""FocusEdgeTwoKPhase — two-scale-k phase-congruency hysteresis enhancer."""
from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field

from phenotypic.abc_._enhance_markers._focus_edge import FocusEdge
from phenotypic.sdk_.mixin import NormalizedOutputMixin
from phenotypic.sdk_.typing_ import TuneSpec

from ._two_k_phase_kernel import two_k_phase

if TYPE_CHECKING:
    from phenotypic._core._image import Image


class FocusEdgeTwoKPhase(NormalizedOutputMixin, FocusEdge):
    """Branch-response enhancer via two-k phase-congruency hysteresis.

    Runs phase congruency at a strict k (clean seeds) and a loose k (full
    candidates), keeps loose candidates that touch a strict seed, and writes the
    loose-k magnitude gated by that mask into ``detect_mat``. Continuous output;
    the inoculum center hole is preserved (this is a pure branch enhancer — center
    filling belongs to the detector, not here).

    Assumes ``detect_mat`` is already illumination-flattened + contrast-stretched
    upstream (same contract as FocusEdgePhase).
    """

    n_orient: Annotated[int, TuneSpec(4, 8)] = Field(8, ge=1)
    min_wavelength: Annotated[float, TuneSpec(2.0, 10.0)] = Field(5.0, ge=2.0)
    k_strict: Annotated[float, TuneSpec(4.0, 8.0)] = Field(6.0, ge=0.0)
    k_loose: Annotated[float, TuneSpec(3.5, 6.0)] = Field(4.5, ge=0.0)
    seed_thresh: Literal["otsu", "triangle"] = "otsu"
    cand_thresh: Literal["otsu", "triangle"] = "triangle"

    def _operate(self, image: "Image") -> "Image":
        gated, _loose = two_k_phase(
            image.detect_mat[:],
            k_strict=self.k_strict,
            k_loose=self.k_loose,
            seed_thresh=self.seed_thresh,
            cand_thresh=self.cand_thresh,
            n_orient=self.n_orient,
            min_wavelength=self.min_wavelength,
        )
        image.detect_mat[:] = self._apply_norm(gated)
        return image
```

- [ ] **Step 4: Re-export from `enhance/__init__.py`**

Add `from ._focus_edge_two_k_phase import FocusEdgeTwoKPhase` alongside the other `FocusEdge*` imports (near line 30–31) and add `"FocusEdgeTwoKPhase"` to `__all__` (the `FocusEdge*` group, ~line 56–65). Do **not** export the `two_k_phase` kernel or `FocusEdge` marker here (markers are deliberately unexported per the taxonomy test).

- [ ] **Step 5: Register in the enhancer taxonomy test**

In `tests/unit/abc_/test_enhancer_taxonomy.py`, add `"FocusEdgeTwoKPhase"` to the `FocusEdge` tuple (lines 37–46).

- [ ] **Step 6: Run the enhancer + taxonomy tests**

Run: `uv run pytest tests/unit/enhance/test_focus_edge_two_k_phase.py tests/unit/abc_/test_enhancer_taxonomy.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/phenotypic/enhance/_focus_edge_two_k_phase.py src/phenotypic/enhance/__init__.py tests/unit/enhance/test_focus_edge_two_k_phase.py tests/unit/abc_/test_enhancer_taxonomy.py
git commit -m "feat(enhance): add FocusEdgeTwoKPhase two-k hysteresis enhancer"
```

---

## Phase 5 — Class 2: `TwoKFilamentousDetector`

A `GridObjectDetector` = two-`k` branch mask + inline grid center-fill + the extracted `sdk_.reconnect` functions. Reuses the loose PCT result from `two_k_phase` for the cost surface (no extra phase-congruency pass).

### Task 5.1: `TwoKFilamentousDetector` — fields + scene derivation + `_operate`

**Files:**
- Create: `src/phenotypic/detect/_two_k_filamentous_detector.py`
- Test: `tests/unit/detect/test_two_k_filamentous_detector.py`
- Modify: `src/phenotypic/detect/__init__.py`

**Interfaces:**
- Consumes: `two_k_phase` (enhance), `FlattenIllumination`, `ContrastStretching`, `SubtractGaussian` (enhance), `InoculumDetector`, `KeepSectionLargest`, `ImagePipeline`; from `sdk_.reconnect`: `ReconnectConfig`, `filter_mask_by_overlap`, `markers_from_centroids`, `partition_by_grid_voronoi`, `identify_pseudo_fragments`, `build_reconnect_cost`, `reconnect_fragments_tiled`.
- Produces: `TwoKFilamentousDetector(GridObjectDetector)` writing `image.objmap`.

- [ ] **Step 1: Write the failing end-to-end test**

```python
# tests/unit/detect/test_two_k_filamentous_detector.py
import numpy as np
import pytest

from phenotypic.data import load_synth_filamentous_plate
from phenotypic.detect import TwoKFilamentousDetector, OtsuDetector


def test_defaults_and_construction():
    d = TwoKFilamentousDetector()
    assert d.k_strict == 6.0 and d.k_loose == 4.5
    # scene-derived scalars populate after validation
    assert d.tile_size is not None and d.mad_window is not None and d.mad_window % 2 == 1


def test_end_to_end_labels_colonies():
    image = load_synth_filamentous_plate().copy()
    d = TwoKFilamentousDetector(center_detector=OtsuDetector(ignore_zeros=True))
    result = d.apply(image, inplace=False)
    assert result.objmap[:].max() > 0
    assert result.objmask[:].sum() > 0


def test_reconnection_reduces_fragments():
    # mutation-style: disabling reconnection (tile smaller than any gap) leaves >= as many
    # connected components as the full run. Full run should not have MORE fragments.
    image = load_synth_filamentous_plate().copy()
    from skimage.measure import label
    full = TwoKFilamentousDetector(center_detector=OtsuDetector(ignore_zeros=True))
    r_full = full.apply(image.copy(), inplace=False)
    n_full = label(r_full.objmap[:] > 0).max()
    no_recon = TwoKFilamentousDetector(
        center_detector=OtsuDetector(ignore_zeros=True), max_gap_length=1, frag_reach_px=1,
    )
    r_no = no_recon.apply(image.copy(), inplace=False)
    n_no = label(r_no.objmap[:] > 0).max()
    assert n_full <= n_no
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/detect/test_two_k_filamentous_detector.py -v`
Expected: FAIL with `ImportError: cannot import name 'TwoKFilamentousDetector'`.

- [ ] **Step 3: Implement the detector**

Fields mirror `FilamentousFungiDetector`'s reconnection knobs (same scene derivation), plus two-`k` branch params, `branch_base`, `center_detector`, `background_subtractor`. The `_operate` flow follows the extraction: two-`k` branches → union with center-fill → overlap-filter → grid-Voronoi → identify fragments → cost surface from the **loose** result → tiled reconnection → final Voronoi.

```python
# src/phenotypic/detect/_two_k_filamentous_detector.py
"""TwoKFilamentousDetector — two-k hysteresis branches + grid center-fill + Dijkstra."""
from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, Optional, Union

import numpy as np
from pydantic import model_validator
from skimage.filters import threshold_otsu
from typing_extensions import Self

from phenotypic.abc_ import GridObjectDetector
from phenotypic import ImagePipeline
from phenotypic.enhance import (
    ContrastStretching,
    FlattenIllumination,
    SubtractGaussian,
)
from phenotypic.enhance._two_k_phase_kernel import two_k_phase
from phenotypic.detect._inoculum_detector import InoculumDetector
from phenotypic.refine import KeepSectionLargest
from phenotypic.sdk_.reconnect import (
    ReconnectConfig,
    build_reconnect_cost,
    filter_mask_by_overlap,
    identify_pseudo_fragments,
    markers_from_centroids,
    partition_by_grid_voronoi,
    reconnect_fragments_tiled,
)
from phenotypic.sdk_.typing_ import OperationField, TuneSpec

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage


class TwoKFilamentousDetector(GridObjectDetector):
    """Detect filamentous fungi via two-k phase hysteresis + grid center-fill + Dijkstra.

    Pipeline: ``branch_base`` (flatten + contrast) prepares detect_mat; ``two_k_phase``
    yields a continuous branch response (binarized by non-zero) and a loose PCT result;
    the grid ``center_detector`` intersected with a background-subtraction body fills the
    inoculum cores; the union is overlap-filtered, grid-Voronoi labeled, and its
    disconnected fragments are reconnected by tiled multi-source Dijkstra over a cost
    surface built from the same loose PCT result (no extra phase-congruency pass).
    """

    # ── scene-derivation multipliers / algorithm constants (mirror FFD) ──
    _GAUSS_SIGMA_PER_R: ClassVar[float] = 1.2
    _TILE_SIZE_PER_R: ClassVar[float] = 4.8
    _TILE_OVERLAP_PER_R: ClassVar[float] = 2.4
    _MAD_WINDOW_PER_W: ClassVar[float] = 2.0
    _PATH_DILATION_PER_W: ClassVar[float] = 0.5
    _SNR_MARGIN_PER_W: ClassVar[float] = 0.5
    _COHERENCE_RADIUS_PER_W: ClassVar[float] = 5.0
    beta: ClassVar[float] = 2.0
    gamma: ClassVar[float] = 1.2
    delta: ClassVar[float] = 1.0
    gauss_n_iter: ClassVar[int] = 2

    # ── branch enhancement ──
    branch_base: Union[OperationField, None] = None                 # -> flatten(300)+stretch(70,99)
    n_orient: Annotated[int, TuneSpec(4, 8)] = 8
    min_wavelength: Annotated[float, TuneSpec(2.0, 10.0)] = 5.0
    k_strict: Annotated[float, TuneSpec(4.0, 8.0)] = 6.0
    k_loose: Annotated[float, TuneSpec(3.5, 6.0)] = 4.5
    seed_thresh: Literal["otsu", "triangle"] = "otsu"
    cand_thresh: Literal["otsu", "triangle"] = "triangle"

    # ── center-fill ──
    center_detector: Union[OperationField, None] = None             # -> InoculumDetector pipeline
    background_subtractor: Union[OperationField, None] = None       # -> SubtractGaussian(gauss_sigma, 2)

    # ── reconnection / scene (mirror FFD) ──
    max_colony_radius_px: Annotated[float, TuneSpec(50.0, 500.0, log=True)] = 250.0
    min_branch_width_px: Annotated[int, TuneSpec(2, 10)] = 3
    reconnection_tolerance: Annotated[float, TuneSpec(1.0, 5.0)] = 2.5
    max_gap_length: Annotated[int, TuneSpec(10, 60)] = 30
    border_margin_px: Annotated[int, TuneSpec(20, 100)] = 50
    frag_reach_px: Annotated[int, TuneSpec(5, 30)] = 10
    gap_crossing_penalty: Annotated[float, TuneSpec(1.0, 10.0)] = 4.0
    gauss_sigma: Annotated[Optional[float], TuneSpec(tunable=False)] = None
    tile_size: Annotated[Optional[int], TuneSpec(tunable=False)] = None
    tile_overlap: Annotated[Optional[int], TuneSpec(tunable=False)] = None
    mad_window: Annotated[Optional[int], TuneSpec(tunable=False)] = None
    path_dilation_radius: Annotated[Optional[int], TuneSpec(tunable=False)] = None
    snr_margin: Annotated[Optional[int], TuneSpec(tunable=False)] = None
    coherence_window_radius: Annotated[Optional[int], TuneSpec(tunable=False)] = None

    @staticmethod
    def __build_center_pipe() -> "ImagePipeline":
        return ImagePipeline(ops=[InoculumDetector(), KeepSectionLargest()])

    @model_validator(mode="after")
    def _derive_scene_params(self) -> Self:
        if self.branch_base is None:
            self.branch_base = ImagePipeline(ops=[
                FlattenIllumination(sigma=300.0),
                ContrastStretching(lower_percentile=70, upper_percentile=99),
            ])
        if self.center_detector is None:
            self.center_detector = self.__build_center_pipe()

        R = self.max_colony_radius_px
        w = self.min_branch_width_px
        if self.gauss_sigma is None:
            self.gauss_sigma = self._GAUSS_SIGMA_PER_R * R
        if self.tile_size is None:
            self.tile_size = int(round(self._TILE_SIZE_PER_R * R))
        if self.tile_overlap is None:
            self.tile_overlap = int(round(self._TILE_OVERLAP_PER_R * R))
        if self.tile_size <= self.tile_overlap:
            raise ValueError("tile_size must exceed tile_overlap")
        if self.mad_window is None:
            _mad = int(round(self._MAD_WINDOW_PER_W * w)) + 1
            self.mad_window = _mad + 1 if _mad % 2 == 0 else _mad
        if self.path_dilation_radius is None:
            self.path_dilation_radius = max(1, int(round(self._PATH_DILATION_PER_W * w)))
        if self.snr_margin is None:
            self.snr_margin = max(2, int(round(self._SNR_MARGIN_PER_W * w)))
        if self.coherence_window_radius is None:
            self.coherence_window_radius = int(round(self._COHERENCE_RADIUS_PER_W * w))
        if self.background_subtractor is None:
            self.background_subtractor = SubtractGaussian(
                sigma=self.gauss_sigma, n_iter=self.gauss_n_iter,
            )
        return self

    def _reconnect_config(self) -> ReconnectConfig:
        return ReconnectConfig(
            beta=self.beta, gamma=self.gamma, delta=self.delta,
            coherence_window_radius=self.coherence_window_radius,
            mad_window=self.mad_window, gap_crossing_penalty=self.gap_crossing_penalty,
            border_margin_px=self.border_margin_px, frag_reach_px=self.frag_reach_px,
            tile_size=self.tile_size, tile_overlap=self.tile_overlap,
            max_gap_length=self.max_gap_length, path_dilation_radius=self.path_dilation_radius,
            snr_margin=self.snr_margin, reconnection_tolerance=self.reconnection_tolerance,
        )

    def _fill_centers(self, image: "GridImage", enhanced: "GridImage"):
        """Grid stamps ∩ background-subtraction body -> (center_mask, center_objmap)."""
        if isinstance(self.center_detector, ImagePipeline):
            center_img = self.center_detector.apply(image, inplace=False, reset=False)
        else:
            center_img = self.center_detector.apply(image, inplace=False)
        grid_mask = center_img.objmask[:] > 0

        body_img = self.background_subtractor.apply(enhanced.copy(), inplace=False)
        body = np.asarray(body_img.detect_mat[:], dtype=float)
        body_mask = body > threshold_otsu(body)

        center_mask = grid_mask & body_mask
        return center_mask, center_img.objmap[:]

    def _operate(self, image: "GridImage") -> "GridImage":
        # ── BRANCH: two-k hysteresis on the enhanced base ──
        enhanced = image.copy()
        self.branch_base.apply(enhanced, inplace=True)
        enhanced_arr = np.asarray(enhanced.detect_mat[:], dtype=np.float32)
        enhanced_gray = np.asarray(enhanced.gray[:], dtype=np.float32)

        gated, loose = two_k_phase(
            enhanced_arr, k_strict=self.k_strict, k_loose=self.k_loose,
            seed_thresh=self.seed_thresh, cand_thresh=self.cand_thresh,
            n_orient=self.n_orient, min_wavelength=self.min_wavelength,
        )
        branch_mask = gated > 0

        # ── CENTERS: grid stamps ∩ body ──
        center_mask, center_objmap = self._fill_centers(image, enhanced)
        if center_objmap.max() == 0:
            raise ValueError("No centers detected by center_detector; cannot label colonies.")

        # ── FILTER + GRID VORONOI ──
        # Union centers FIRST so branch rings (PCT leaves the inoculum core a hole) connect
        # through their cores; THEN keep only components overlapping a center — the analogue of
        # FFD's `inoculum_structure_mask = _filter_mask_by_overlap(overall_objmask, inoculum_objmask)`.
        # This drops stray/agar objects not attached to any well (do NOT keep all objects).
        colony_mask = branch_mask | center_mask
        structure_mask = filter_mask_by_overlap(colony_mask, center_mask)
        markers = markers_from_centroids(center_objmap)
        colony_labels = partition_by_grid_voronoi(markers, structure_mask)
        if colony_labels.max() == 0:
            raise RuntimeError("Voronoi assignment produced empty result.")

        # ── RECONNECT (Dijkstra over the loose PCT cost surface) ──
        central_mask, fragment_labels = identify_pseudo_fragments(colony_labels, center_mask)
        cfg = self._reconnect_config()
        unmasked_cost, cost_surface = build_reconnect_cost(
            loose.pc_sum, loose.M, loose.m, loose.orientation,
            enhanced_arr, colony_labels, central_mask, cfg,
        )
        colony_labels = reconnect_fragments_tiled(
            colony_labels, fragment_labels, cost_surface, unmasked_cost,
            loose.pc_sum.astype(np.float32), enhanced_gray, cfg,
        )

        # ── FINAL VORONOI ──
        # Re-partition the overlap-filtered `structure_mask` (NOT the raw branch_mask) together
        # with the reconnected labels, so the final objmap contains ONLY center-overlapping
        # objects — mirrors FFD's `final_mask = (colony_labels > 0) | inoculum_structure_mask`.
        final_mask = (colony_labels > 0) | structure_mask
        colony_labels = partition_by_grid_voronoi(markers, final_mask)

        if colony_labels.dtype != image._OBJMAP_DTYPE:
            colony_labels = colony_labels.astype(image._OBJMAP_DTYPE)
        image.objmap[:] = colony_labels
        return image
```

- [ ] **Step 4: Re-export from `detect/__init__.py`**

Add `from ._two_k_filamentous_detector import TwoKFilamentousDetector` (near line 30) and `"TwoKFilamentousDetector"` to `__all__` (near line 38).

- [ ] **Step 5: Run the detector test**

Run: `uv run pytest tests/unit/detect/test_two_k_filamentous_detector.py -v`
Expected: PASS (construction, end-to-end labels, reconnection-does-not-increase-fragments).

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/detect/_two_k_filamentous_detector.py src/phenotypic/detect/__init__.py tests/unit/detect/test_two_k_filamentous_detector.py
git commit -m "feat(detect): add TwoKFilamentousDetector (two-k + center-fill + Dijkstra)"
```

### Task 5.2: Serialization round-trip + cost-surface reuse assertion

**Files:**
- Modify: `tests/unit/detect/test_two_k_filamentous_detector.py`

**Interfaces:**
- Consumes: `TwoKFilamentousDetector`.

- [ ] **Step 1: Write the failing tests (append)**

```python
def test_serialization_round_trip():
    d = TwoKFilamentousDetector(k_loose=4.0, max_colony_radius_px=200.0)
    payload = d.model_dump_json()
    restored = TwoKFilamentousDetector.model_validate_json(payload)
    assert restored.k_loose == 4.0
    assert restored.max_colony_radius_px == 200.0
    assert restored.tile_size == d.tile_size          # derived scalars survive


def test_cost_surface_uses_loose_result_no_extra_pct(monkeypatch):
    # two_k_phase must be called exactly once per _operate (the only phase-congruency
    # work); the loose result it returns feeds the cost surface — no third PCT pass.
    import phenotypic.detect._two_k_filamentous_detector as mod
    calls = {"n": 0}
    real = mod.two_k_phase

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(mod, "two_k_phase", counting)
    from phenotypic.data import load_synth_filamentous_plate
    from phenotypic.detect import OtsuDetector
    image = load_synth_filamentous_plate().copy()
    TwoKFilamentousDetector(center_detector=OtsuDetector(ignore_zeros=True)).apply(image, inplace=False)
    assert calls["n"] == 1
```

- [ ] **Step 2: Run to verify (expect PASS if implementation is correct; else fix)**

Run: `uv run pytest tests/unit/detect/test_two_k_filamentous_detector.py -k "serialization or cost_surface" -v`
Expected: PASS. (`two_k_phase` is called exactly once; the detector reuses `loose` rather than re-running phase congruency.)

- [ ] **Step 3: Commit**

```bash
git add tests/unit/detect/test_two_k_filamentous_detector.py
git commit -m "test(detect): TwoKFilamentousDetector serialization + single-PCT-pass guards"
```

### Task 5.3: Overlap-keep pin — final objmap excludes objects not overlapping a center

**Load-bearing requirement (mirrors FFD):** the final `objmap` must contain **only** components overlapping a detected center. A bright structure far from every well must be background (`0`) — the detector must **not** "keep all objects." This is the `structure_mask = filter_mask_by_overlap(colony_mask, center_mask)` filter in `_operate`; this test pins it and must be able to fail if the filter is bypassed.

**Files:**
- Modify: `tests/unit/detect/test_two_k_filamentous_detector.py`

**Interfaces:**
- Consumes: `TwoKFilamentousDetector`, `ManualGridPointDetector`, `GridImage`.

- [ ] **Step 1: Write the failing-capable test (append)**

```python
from phenotypic import GridImage
from phenotypic.detect import ManualGridPointDetector


def _plate_two_colonies_and_a_stray():
    """1x2 grid: two colonies (solid core + branch tendril) on known wells, plus one bright
    blob far from both wells. Returns (GridImage, well0_rc, well1_rc, stray_rc)."""
    H, W = 200, 400
    g = np.full((H, W), 60, dtype=np.uint8)
    yy, xx = np.ogrid[:H, :W]

    def disk(cy, cx, r, val):
        g[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = val

    well0, well1 = (100, 100), (100, 300)
    disk(*well0, 22, 235); g[98:103, 100:150] = 215        # colony 0: core + tendril
    disk(*well1, 22, 235); g[98:103, 250:300] = 215        # colony 1: core + tendril
    stray_rc = (32, 200)
    disk(*stray_rc, 18, 240)                                # stray blob, far from both wells
    rgb = np.repeat(g[..., None], 3, axis=2)
    return GridImage(rgb, nrows=1, ncols=2), well0, well1, stray_rc


def test_final_objmap_excludes_objects_not_overlapping_centers():
    img, well0, well1, stray_rc = _plate_two_colonies_and_a_stray()
    detector = TwoKFilamentousDetector(
        center_detector=ManualGridPointDetector(coord1=well0, coord2=well1,
                                                shape="disk", width=40),
    )
    objmap = np.asarray(detector.apply(img, inplace=False).objmap[:])
    assert objmap[well0] > 0 and objmap[well1] > 0     # both colonies are labeled
    assert objmap[stray_rc] == 0                       # stray object is dropped by the overlap filter
```

- [ ] **Step 2: Run to verify it passes**

Run: `uv run pytest tests/unit/detect/test_two_k_filamentous_detector.py::test_final_objmap_excludes_objects_not_overlapping_centers -v`
Expected: PASS. (If the synthetic doesn't robustly produce two labeled colonies — e.g. PCT thresholds don't fire on this scale — adjust sizes/contrast/`width` while keeping the intent: two center-overlapping colonies labeled, one far-from-center object excluded. Do not weaken the `stray_rc == 0` assertion.)

- [ ] **Step 3: Scope note (verified during code review)**

This integration test guards the end-to-end behavior — "only center-overlapping objects are labeled" — which is enforced **jointly** by `filter_mask_by_overlap` AND the grid-Voronoi marker-drop (`partition_by_grid_voronoi` zeroes marker-less connected components). Bypassing *only* the filter (`structure_mask = colony_mask`) does **not** fail the test — verified empirically that the marker-drop still excludes the stray. So this test guards the gross "keep all objects" regression (skipping the Voronoi entirely would leak the stray); the overlap filter itself is unit-tested in `tests/unit/sdk_/reconnect/test_colony_labeling.py::test_filter_mask_by_overlap_drops_non_overlapping_cc`.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/detect/test_two_k_filamentous_detector.py
git commit -m "test(detect): pin final objmap to center-overlapping objects only"
```

---

## Phase 6 — Integration, packaging, and final verification

### Task 6.1: Public-API / packaging + full regression sweep

**Files:**
- Modify (if needed): `tests/integration/packaging/test_package_contents.py` (only if it enumerates detector/enhancer exports).

- [ ] **Step 1: Confirm the new symbols import from their public homes**

Run: `uv run python -c "from phenotypic.enhance import FocusEdgeTwoKPhase; from phenotypic.detect import TwoKFilamentousDetector; from phenotypic.sdk_.reconnect import ReconnectConfig, build_reconnect_cost, reconnect_fragments_tiled, partition_by_grid_voronoi, markers_from_centroids, filter_mask_by_overlap, identify_pseudo_fragments, compute_full_image_app2_gi_cost; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 2: Check packaging test for an export inventory that needs the new names**

Run: `cd /Users/alex/Projects/PhenoTypic && grep -n "FilamentousFungiDetector\|FocusEdgePhase" tests/integration/packaging/test_package_contents.py`
If the test asserts a fixed inventory, add `TwoKFilamentousDetector`/`FocusEdgeTwoKPhase` there; otherwise no change.

- [ ] **Step 3: Full targeted regression sweep**

Run: `uv run pytest tests/unit/sdk_/reconnect/ tests/unit/enhance/ tests/unit/detect/ tests/unit/abc_/test_enhancer_taxonomy.py -q`
Expected: PASS. Notably the FFD golden regression (`test_filamentous_fungi_regression.py`) still passes.

- [ ] **Step 4: Import-purity final check**

Run: `uv run pytest tests/unit/sdk_/reconnect/test_import_rules.py -v`
Expected: PASS.

- [ ] **Step 5: Commit any packaging-test change**

```bash
git add tests/integration/packaging/test_package_contents.py
git commit -m "test: register two-k detector/enhancer in packaging inventory"
```
(Skip if Step 2 required no change.)

### Task 6.2: Code-review + simplify gates

- [ ] **Step 1:** Dispatch a code-review agent over the full diff (`git diff main...HEAD`) focused on: (a) the FFD extraction preserving behavior, (b) `sdk_.reconnect` import purity, (c) the detector `_operate` flow (center union before overlap-filter; loose-result cost surface). Apply confirmed findings.
- [ ] **Step 2:** Run a simplify pass over the newly added modules; apply reuse/clarity fixes that don't change behavior; re-run the Phase 6 sweep.
- [ ] **Step 3:** Final commit of any review/simplify fixes.

---

## Spec coverage map (self-review)

| Spec element | Task(s) |
|---|---|
| Class 1 `FocusEdgeTwoKPhase` (§1) — continuous gated output, center hole preserved | 4.1, 4.2 |
| Shared kernel `two_k_phase` returning `(gated, loose)` (§1) | 4.1 |
| Opposite-threshold binarization (otsu seed / triangle cand), load-bearing (§1) | 4.1 (mutation test), 4.2 |
| Center-fill = separate concern, Class 1 stays pure (§2) | resolved as inline detector fields (5.1); Class 1 has no center logic (4.2 `test_center_hole_preserved`) |
| Class 2 `TwoKFilamentousDetector` (§3) — branch→center→filter→Voronoi→reconnect→final | 5.1 |
| Cost surface fed by the loose PCT result, no extra pass (§3) | 5.1, 5.2 (`test_cost_surface_uses_loose_result_no_extra_pct`) |
| Reuse mechanism = extract to `sdk_.reconnect`, composable (§5 Q1) | 1.1, 2.1–2.3, 3.1 |
| Flatten(300)+Stretch(70,99), no gamma, as `branch_base` (§5 Q2) | 5.1 |
| `k_loose` default 4.5 (§5 Q3) | 4.2, 5.1 |
| Build the Dijkstra detector (§5 Q4) | 5.1 |
| Registration + taxonomy + tests (§4) | 4.2 (taxonomy), 1.1/2.x (reconnect tests), 6.1 (packaging) |
| FFD unchanged behavior (new constraint) | 0.1 (gate), 3.1 |

**Deviation from spec:** §2 suggested a standalone `FillInoculumCenters` class; this plan implements center-fill as **inline detector fields** (`center_detector` + `background_subtractor` + `_fill_centers`), matching FFD's established `inoculum_detector` pattern and the `LightDetectFungi` spec §3. Class 1's purity (the actual §2 goal) is preserved either way. Additive/reversible if a standalone class is later wanted.

---

## Execution Orchestration (cluster-and-isolate)

### Task shapes

| Task | Shape | Why |
|---|---|---|
| 0.1 Golden fixture | Seam-enabler | tiny, but defines the hard gate the FFD refactor is judged against |
| 1.1 Extract labeling | Keystone (small) | cohesive verbatim move into one new module + exports |
| 2.1–2.3 Extract cost+reconnection | Keystone | interdependent, all mutate the **same** `_colony_reconnect.py` + `__init__` + test |
| 3.1 FFD refactor | **Seam** | one risky wiring point; must stay bit-identical (delete 11 methods, rewire `_operate`) |
| 4.1–4.2 Class 1 | Keystone | novel kernel + enhancer; 4.2 imports 4.1; shares intent |
| 5.1–5.2 Class 2 | Keystone | the integration detector; 5.2 is a Leaf folded in |
| 6.1 Packaging sweep | Leaf | small, folds into the final gate |
| 6.2 Review/simplify | Gate | the mandated deep-review + simplify passes |

### Dependency DAG (→ = must finish before; shared files noted)

```
0.1 ─────────────────────────────┐
                                  ▼
C-A(1.1) ──► C-B(2.1►2.2►2.3) ──► C-Seam(3.1) ──► C-E(5.1►5.2)
  │  shared: reconnect/__init__.py       ▲                ▲
  └──────────── sdk_.reconnect fns ──────┴────────────────┘  (E needs the fns too)
C-D(4.1►4.2) ─────────────────────────────────────────────┘   (E imports two_k_phase)
```
- `2.1→2.2→2.3` are strictly sequential (same file `_colony_reconnect.py`).
- **C-E is sequenced after C-Seam even though it has no import dependency on FFD** — the FFD golden regression is the strongest proof the extracted functions are behavior-correct, so it must be green before we build Class 2 on that foundation.
- **C-D shares zero files** with the extraction chain (`enhance/*` + taxonomy test vs `sdk_.reconnect/*` + FFD + detect regression test) → parallel-worktree candidate.

### Clusters, models, effort

| Cluster | Tasks | Files (no overlap between parallel clusters) | Model / effort | Rationale |
|---|---|---|---|---|
| **C0** Regression fixture | 0.1 | `tests/unit/detect/test_filamentous_fungi_regression.py`, `tests/fixtures/…npy` | **sonnet / medium** | fully-specified, mechanical; runs real FFD to mint the fixture |
| **C-A** Extract labeling | 1.1 | `sdk_/reconnect/_colony_labeling.py`, `…/__init__.py`, test | **sonnet / medium** + Opus gate | verbatim move, fully specified; frontier verify at the gate |
| **C-B** Extract cost+reconnect | 2.1–2.3 | `sdk_/reconnect/_colony_reconnect.py`, `…/__init__.py`, test | **opus / high** | large verbatim moves + `self.*→cfg.*` substitution tables; transcription-error prone, consistency-critical |
| **C-Seam** FFD refactor | 3.1 | `detect/_filamentous_fungi_detector.py` | **opus / high** | highest-stakes wiring; bit-identical gate |
| **C-D** Class 1 | 4.1–4.2 | `enhance/_two_k_phase_kernel.py`, `enhance/_focus_edge_two_k_phase.py`, `enhance/__init__.py`, taxonomy test, enhance tests | **opus / high** | novel kernel + enhancer contract (MRO, mutation test) |
| **C-E** Class 2 | 5.1–5.2 | `detect/_two_k_filamentous_detector.py`, `detect/__init__.py`, test | **opus / high** | integration: center-union-before-filter subtlety, loose-result cost-surface reuse |

Never review/verify with a weaker model than implemented: C0/C-A (sonnet-implemented) are gated by an Opus review; everything else is Opus-implemented and Opus-reviewed.

### Execution waves

- **Wave 1 — two parallel tracks (disjoint files):**
  - *Track EXTRACT (critical path, sequential):* **C0 → C-A → C-B**. (C0 first also confirms the FFD baseline is green before extracting.)
  - *Track CLASS1 (worktree, concurrent):* **C-D**. Independent; finishes early and waits.
- **Wave 2 — hard gate:** **C-Seam**. Runs after C0+C-A+C-B. The golden regression MUST be bit-identical; **STOP and debug on failure — do not proceed to C-E.**
- **Wave 3 — integration:** **C-E** (after C-Seam green + C-D merged).
- **Wave 4 — closing gates:** deep code-review agent (Opus) over the full `main...HEAD` diff (behavior preservation + import purity + detector flow) → triage/fix; then one **simplify** pass over the new modules; then the Phase-6 regression sweep.

### Gates (between clusters)

- **Light (every cluster):** orchestrator (Opus) reads the diff + runs that cluster's tests before the next cluster. Surface any design-conflicting finding to the user before continuing.
- **C-Seam hard gate:** `test_filamentous_fungi_regression.py` bit-identical + the FFD seam/shim suites green. Non-negotiable.
- **Deep review (Wave 4):** fresh Opus code-review agent over the combined diff.
- **Simplify (end):** quality-only pass, then affected-area regression.

### Parallelism summary

Only **C-D ∥ Track EXTRACT** is fanned out (zero file overlap, separate worktree). Everything else is sequential because clusters share files (`reconnect/__init__.py`, FFD, `detect/__init__.py`) or are gated dependencies. No other cluster pair is overlap-free enough to parallelize safely.
