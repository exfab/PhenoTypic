# CalibrateColorRpcc (in-frame checker colour correction) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `CalibrateColorRpcc`, an `ImageCorrector` that fits a root-polynomial colour-correction profile from the colour chart visible in the plate image's own frame and applies it — replacing a notebook route whose reconstructed tiles are 49–91 % mirror-fabricated pixels and which breaks when the plate drifts a few pixels.

**Architecture:** Five new modules under `src/phenotypic/correction/_color_correction/`, plus one promoted method on `ColorCheckerProfile`. The operation owns orchestration only; every colorimetric decision is delegated to code that already exists and is tested. The pipeline is: rectangle ROIs in → lattice detected per ROI (tile count inferred, not declared) → patch identity derived by scoring whole rigid placements → per-tile colour by deterministic ΔE2000 medoid → `ColorCheckerProfile` fitted from that mapping → `ColorCorrector` applies it. Nothing about the chart's contents is a constructor argument, so a mislabelled or mis-mounted card surfaces as a low match margin rather than a silently wrong fit.

**Tech Stack:** Python 3.11+, pydantic v2, numpy, `colour-science`, scikit-image, OpenCV (ECC refinement only), pytest, `uv`.

**Spec:** `docs/superpowers/specs/2026-09-21-in-frame-checker-color-correction/README.md`. Evidence for the degree and patch-count rules: `degree-and-patch-count.md` in the same folder. Independent numeric witness: `docs/superpowers/logic_validation_scripts/2026-09-21-in-frame-checker-color-correction/checker_color_correction.py` (claims C0–C4; exits 0).

**Branch / worktree:** `feat/calibrate-color-rpcc` at `.worktrees/calibrate-color-rpcc`, based on `main`. `main` already carries the `robust_color_center` + `GEOMEDIAN_TOL=1e-6` estimator fix this work depends on.

## Global Constraints

- `uv` is the sole runner **on the host**: `uv run <cmd>`, never bare `python` or `pip`.
  `uv` is **not on PATH inside the agent sandbox** and the repo's own `.venv`
  interpreter is blocked there. From the sandbox, run tests with the `snp-dev`
  venv and put the worktree first on `sys.path` — its `phenotypic` editable
  install is a plain `.pth`, so `sys.path.insert(0, 'src')` shadows the main
  checkout:
  `QT_QPA_PLATFORM=offscreen <snp-dev>/bin/python -c "import sys; sys.path.insert(0,'src');
  sys.path.insert(0,'tests'); import pytest; raise SystemExit(pytest.main([...]))"`
  with `-o addopts= -p no:randomly -p no:napari`. `ruff` runs as a binary from
  `/Users/alex/Projects/PhenoTypic/.venv/bin/ruff`.
- Operations are keyword-only pydantic v2 models with `extra="forbid"` and `validate_assignment=True`. Construction with a positional argument raises.
- **Reuse, never reimplement.** `_load_reference_data` for reference values and Bradford adaptation; `ColorCheckerProfile._fit_from_measured` for the solver, outlier rejection and diagnostics; `robust_color_center` for the geometric median; `ColorCorrector` for applying the matrix; `CaptureMetadata` for EXIF. A second implementation of any of these is a review failure.
- **All Lab comes from `Image.color.Lab`.** The prototype fed linear RawTherapee RGB into `skimage.rgb2lab`, which expects sRGB encoding; its Lab values and every threshold expressed in them are an internal contrast measure, not calibrated CIE. That defect must not be ported.
- **Degree is fixed.** Never adapted per frame, never a tuning target. Rank-insufficiency raises; everything else about a short card warns.
- **Identity is derived, never declared, and decided by whole-placement scoring** — never a free per-tile Hungarian assignment, which lets a label wander onto whichever reference fits it best and silently mislabels an occluded tile.
- Existing repo files use **CRLF** line endings. Preserve them when editing; new files may be LF.
- `uv run ruff check <explicit paths>` — never bare, which rewrites unrelated files.
- Tests: `QT_QPA_PLATFORM=offscreen`, explicit `-n`, `-o addopts=`, never `-x` for a baseline. Use the **`run-phenotypic-test`** skill before any non-trivial invocation; the full `tests/unit` suite is ~65 minutes and belongs in a Slurm job (**`slurm-job`** skill).

## Out of scope, by decision

| Item | Decision | Why |
|---|---|---|
| Pinning white balance in RAW development (`Setting=Custom`) | Deferred | Upstream of this operation. The 5 % WB-divergence warning in Task 5 detects the problem; it does not cure it. |
| A chroma-only / white-normalised accuracy metric | Deferred | Absolute ΔE2000 conflates illumination falloff with colour accuracy for edge-mounted tiles. Real need, separate change. |
| Reconciling the two geometric medians in the repo (`util/_geometric_median.py` in Lab, `_helpers.py` in sRGB) | Deferred | Known duplication; neither is wrong after the estimator fix. |
| `_fit_ridge` / `ridge_lambda` dead path | Document only | `ridge_lambda` is a stored field reachable only from an `except Exception` fallback. Removing it is a separate migration. |
| `M3_blob`, `M23_cascade` refinements | Not ported | 1.419 and 1.825 s/band against the rigid snap's 0.357, with no measured gain. |
| Detecting a free-standing, fully-visible chart | Not in scope | `ColorCheckerProfile.fit()` with border-fill segmentation already does this, and does it better. |

## File Structure

| File | Responsibility in this change |
|---|---|
| `src/phenotypic/correction/_color_correction/_checker_roi.py` | `CheckerRoi` (rectangle + optional assertions), `ColumnLattice`, `CheckerLattice`, `CheckerRoi.from_bbox`. Pure data, no image access. |
| `src/phenotypic/correction/_color_correction/_checker_detect.py` | Plateau counting → tile grid; 1-D periodic fit for phase/pitch/duty; `rigid` / `ecc` / `frozen` refinement of a stored lattice. |
| `src/phenotypic/correction/_color_correction/_checker_identity.py` | Placement enumeration over the chart grid, gain-invariant scoring, winner + margin, free-Hungarian corroboration. |
| `src/phenotypic/correction/_color_correction/_checker_measure.py` | Core box trimming; candidate-restricted ΔE2000 medoid; per-tile spread, impurity, robust shift, clipping. |
| `src/phenotypic/correction/_color_correction/_checker_qc.py` | `QcLimits`, the gate table, `qc_record` assembly, `on_qc_fail` dispatch. |
| `src/phenotypic/correction/_color_correction/_calibrate_color_rpcc.py` | `CalibrateColorRpcc` — orchestration, warnings, diagnostics, `patch_census`. |
| `src/phenotypic/correction/_color_correction/_color_checker_profile.py` | **Modify:** promote `_fit_from_patch_colors` to a public `fit_from_patch_colors`, keeping the private name as a deprecated alias. |
| `src/phenotypic/correction/_color_correction/__init__.py`, `correction/__init__.py` | **Modify:** export the new names. |
| `src/phenotypic/correction/CLAUDE.md` | **New:** module guide; `correction/` currently has none. |
| `tests/unit/correction/test_checker_*.py` | One test module per source module above. |
| `tests/smoke/test_operation.py` | **Modify:** exclude `CalibrateColorRpcc` from the bare-construction contract, as `ColorCorrector` already is. |

---

## Task 0: Worktree and documents

**Files:** the three spec/validation documents, moved onto the feature branch.

- [x] **Step 1: Create the worktree** (blocked in-sandbox; run on the host)

```bash
cd /Users/alex/Projects/PhenoTypic
git worktree add -b feat/calibrate-color-rpcc .worktrees/calibrate-color-rpcc main
```

This session's git protection runs in coarse mode, so `.git/worktrees/` is write-denied and the agent cannot do this step.

- [x] **Step 2: Move the documents onto the branch**

`docs/superpowers/specs/2026-09-21-in-frame-checker-color-correction/{README.md,degree-and-patch-count.md}`, `docs/superpowers/logic_validation_scripts/2026-09-21-in-frame-checker-color-correction/checker_color_correction.py`, and this plan. They are currently untracked in the `fix/geo-median-convergence` checkout.

- [x] **Step 3: Confirm the witness still passes on the branch**

Run: `uv run python docs/superpowers/logic_validation_scripts/2026-09-21-in-frame-checker-color-correction/checker_color_correction.py`
Expected: `All claims hold.`, exit 0, ~35 s. It imports neither `phenotypic` nor `colour` by contract, so it is a genuine independent check of C0–C4.

---

## Task 1: `CheckerRoi` and the bounding-box shorthand

**Files:**
- Create: `src/phenotypic/correction/_color_correction/_checker_roi.py`
- Test: `tests/unit/correction/test_checker_roi.py`

**Interfaces:**
- Produces: `CheckerRoi(row: tuple[int,int], col: tuple[int,int], label: str | None, expect_tiles: int | None, anchor_col: int | None)`; `CheckerRoi.from_bbox(seq) -> CheckerRoi` taking `[row_min, col_min, row_max, col_max]`; `ColumnLattice(x0, x1, start, pitch, duty)`; `CheckerLattice(columns, nrows, dy, dx, rot)`.
- Consumes: nothing. This module must not import numpy-heavy or image code — it is the one piece that stays trivially testable.

**Why first:** everything downstream takes these types, and the coercion has a failure mode worth nailing before anything depends on it. `[row_min, col_min, row_max, col_max]` is `skimage.measure.regionprops`' `bbox` order, chosen so `rois=[r.bbox for r in regionprops(label_img)]` drops in; it is *not* the `(row_slice, col_slice)` pair `ColorCheckerProfile.rois` takes and not `(x, y, w, h)`. Both mistakes produce a plausible-looking rectangle that fails much later as a detection problem.

- [x] **Step 1: Write the failing tests**

```python
def test_bbox_shorthand_matches_the_explicit_form() -> None:
    assert CheckerRoi.from_bbox([1170, 5856, 2840, 6016]) == CheckerRoi(
        row=(1170, 2840), col=(5856, 6016)
    )


@pytest.mark.parametrize(
    "bad, expected_message",
    [
        ([1170, 0, 2840], "four"),
        ([1170, 0, 2840, 340, 7], "four"),
        ([1170.5, 0, 2840, 340], "integer"),
        ([1170, 0, 340, 2840], "row_min, col_min, row_max, col_max"),  # (x,y,w,h) shape
        ([1170, 5856, 1170, 6016], "row_min, col_min, row_max, col_max"),  # zero height
    ],
)
def test_bbox_shorthand_rejects_the_wrong_convention(bad, expected_message) -> None:
    with pytest.raises(ValueError, match=expected_message):
        CheckerRoi.from_bbox(bad)


def test_operation_field_coerces_and_stores_canonical() -> None:
    op = CalibrateColorRpcc(rois=[[1170, 0, 2840, 340],
                                  CheckerRoi(row=(1170, 2840), col=(5856, 6016))])
    assert all(isinstance(r, CheckerRoi) for r in op.rois)
    assert op.model_dump()["rois"][0] == {"row": (1170, 2840), "col": (0, 340),
                                          "label": None, "expect_tiles": None,
                                          "anchor_col": None}
```

The third test imports the operation and so lands red until Task 6; keep it in this file and mark it `xfail(strict=True)` until then, or stage it with Task 6. Do not weaken the assertion to make it pass early.

- [x] **Step 2: Run them and watch them fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/correction/test_checker_roi.py -q --no-header -p no:randomly -o addopts=`

- [x] **Step 3: Implement `CheckerRoi` and `from_bbox`**

The error message is the point of this task — it must name the expected order, because that is the only way a `(x, y, w, h)` caller learns what went wrong:

```python
_BBOX_ORDER = "row_min, col_min, row_max, col_max"

@classmethod
def from_bbox(cls, seq, **kwargs) -> "CheckerRoi":
    values = list(seq)
    if len(values) != 4:
        raise ValueError(
            f"A bounding box needs four coordinates ({_BBOX_ORDER}); got {len(values)}."
        )
    if any(not isinstance(v, (int, np.integer)) for v in values):
        raise ValueError(f"Bounding-box coordinates must be integer pixels ({_BBOX_ORDER}).")
    row_min, col_min, row_max, col_max = (int(v) for v in values)
    if row_min >= row_max or col_min >= col_max:
        raise ValueError(
            f"Bounding box {values} is empty or inverted. Expected {_BBOX_ORDER} "
            "(scikit-image regionprops order) — a (row, col, height, width) or "
            "(x, y, width, height) box will look like this."
        )
    return cls(row=(row_min, row_max), col=(col_min, col_max), **kwargs)
```

- [x] **Step 4: Add `ColumnLattice` / `CheckerLattice`**

Plain serialisable models. `CheckerLattice.boxes(dy, dx, core, rot)` and `.centers(**kw)` port directly from the prototype's `SidePrior`, which is already the right shape — keep the method names and the `(row, col, y0, y1, x0, x1)` tuple order so ported detection code reads the same.

- [x] **Step 5: Green, then lint**

Run: the pytest line from Step 2, then `uv run ruff check src/phenotypic/correction/_color_correction/_checker_roi.py tests/unit/correction/test_checker_roi.py`

---

## Task 2: Tile measurement and the deterministic medoid

**Files:**
- Create: `src/phenotypic/correction/_color_correction/_checker_measure.py`
- Test: `tests/unit/correction/test_checker_measure.py`

**Interfaces:**
- Consumes: `robust_color_center(points, max_iter, tol)`; `colour.difference.delta_E_CIE2000`.
- Produces: `candidate_medoid(lab, k=256) -> MedoidResult(index, lab, rank, total_distance)`; `trim_core(box, core_trim) -> box`; `tile_impurity`, `tile_robust_shift`, `tile_clipped`.

**Why second:** it is the only stage with a contract that can be checked exactly rather than by eye — the answer must equal the exhaustive medoid — and it needs no detection, no image, and no chart. Getting it green early means every later task measures colour with something already proven.

- [x] **Step 1: Write the failing tests**

```python
def _tile_cloud(seed: int, n: int = 4200) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centre = np.array([rng.uniform(20, 85), rng.uniform(-45, 60), rng.uniform(-55, 65)])
    sd = np.array([rng.uniform(1.5, 4), rng.uniform(1, 3), rng.uniform(1, 3)])
    return centre + rng.normal(0, 1, (n, 3)) * sd


@pytest.mark.parametrize("seed", range(8))
def test_candidate_medoid_equals_the_exhaustive_medoid(seed: int) -> None:
    lab = _tile_cloud(seed)
    got = candidate_medoid(lab, k=256)
    assert got.index == _exhaustive_medoid_index(lab)   # local O(n^2) reference
    assert got.rank < 0.8 * 256                          # winner not at the edge of K


def test_candidate_medoid_is_deterministic() -> None:
    lab = _tile_cloud(0)
    first, second = candidate_medoid(lab), candidate_medoid(lab)
    assert first.index == second.index
    assert np.array_equal(first.lab, second.lab)


def test_candidate_medoid_widens_k_when_the_winner_is_at_the_edge() -> None:
    """A bimodal cloud puts the medoid away from the geometric median."""
    ...
    assert result.widened is True
```

`_exhaustive_medoid_index` is a local chunked O(n²) reference in the test file — deliberately not imported from the source, so the test is an independent witness the way the validation script is.

- [x] **Step 2: Run them and watch them fail**

- [x] **Step 3: Implement the candidate-restricted medoid**

Weiszfeld geometric median over all core pixels → rank by Euclidean distance to it → score the nearest `k` candidates by total ΔE2000 against **every** pixel → winner. No RNG anywhere. Return the winning pixel's index so the caller can take that pixel's own sRGB value and leave the existing fit path untouched.

Measured on a 4 500-pixel tile: 0.14 s at k=256 against 2.29 s for the exhaustive form, and identical on 8/8 clouds. Do **not** substitute `medoid_ciede2000`: its seeded 1 000-pixel subsample moves by ~0.12 ΔE00 median (worst 0.81) between seeds, and raising the cap costs O(n²) — 41 s per 24-tile frame at 4 000 — while seed spread only falls to ~0.25.

- [x] **Step 4: Port the tile statistics**

`tile_impurity`, `tile_robust_shift`, `tile_clipped` from the prototype, **with the encoding defect fixed**: Lab must come from `Image.color.Lab`, not `skimage.rgb2lab` on linear input. Re-derive the thresholds against correctly-encoded Lab and record in the docstring whether any gate decision moves; the prototype's note says none did on its test set, and the numbers came out slightly smaller, so the shipped limits are the conservative side.

---

## Task 3: Patch identity by placement scoring

**Files:**
- Create: `src/phenotypic/correction/_color_correction/_checker_identity.py`
- Test: `tests/unit/correction/test_checker_identity.py`

**Interfaces:**
- Consumes: `_load_reference_data(checker_type, target_illuminant)`.
- Produces: `placements(chart_shape, tile_shape) -> list[Placement]`; `assign_placement(obs_linear, placements, ref_linear) -> PlacementResult(assignment, score, runner_up, margin, hungarian_agreement)`.

**Why third:** it is the stage that makes "don't declare the layout" safe, and its threshold is calibrated from data rather than chosen.

- [x] **Step 1: Write the failing tests**

Fixtures come from `patch_measurements.npz` (4 frames × 24 patches) reduced to a small committed `.npz` of 8 cards × 12 tiles of linear RGB — the full substrate is external to the repo and must not become a test dependency.

```python
def test_placement_is_recovered_without_being_told(card):
    result = assign_placement(card.observed, placements(chart=(4, 6), tile=(6, 2)), REF)
    assert result.assignment == card.truth          # 12/12, all 8 cards
    assert result.margin >= 0.29


def test_a_rotated_card_is_assigned_correctly_not_merely_flagged(card):
    result = assign_placement(card.observed[::-1], ..., REF)
    assert result.assignment == card.truth[::-1]
    assert result.margin >= 0.29


def test_three_occluded_tiles_are_refused_on_margin_not_mislabelled(card):
    ...
    assert result.margin < 0.20
```

- [x] **Step 2–3: Enumerate placements, score whole hypotheses**

Score in gain-invariant features — chromaticity plus relative luminance — so an exposure difference cannot choose the answer. Rank whole hypotheses by mean feature distance across all tiles at once. A free Hungarian runs **alongside**, reported as `hungarian_agreement`, never as the decision.

Measured on 8 real cards with the fully general hypothesis set (every ordered pair of chart rows × both column directions, 24 hypotheses, nothing rig-specific): 12/12 correct on all 8, margin 0.294–0.334. The rig-specific 8-hypothesis set gives 0.814–0.873, so the general set is decisive and the operation need not be told the rig layout.

- [x] **Step 4: Wire the calibrated margin gate**

Refuse below 0.20, warn below 0.25. Calibration (2 240 occlusion-injection trials, `identity_margin_threshold.csv`): at 0.10 only 64.8 % of wrong identities are caught and 11.5 % mislabel silently; at 0.20, 92.4 % / 2.5 %; at 0.25, 97.8 % / 0.7 % but only 18 % headroom under the 0.294 clean-card minimum. There is **no fallback** to a declared layout — an undetermined placement is refused.

---

## Task 4: Lattice detection

**Files:**
- Create: `src/phenotypic/correction/_color_correction/_checker_detect.py`
- Test: `tests/unit/correction/test_checker_detect.py`

**Interfaces:**
- Produces: `fit_lattice(lab, roi) -> CheckerLattice` (tile count inferred); `refine(lab, prior, method, reference_band=None) -> RefineResult(lattice, dy, dx, rot, confidence)`.

- [x] **Step 1: Tests on synthetic lattices with known displacement**, then the real bands. Capture range to assert, from the prototype's benchmark: rigid ±45 px vertical / ±30 px horizontal; ECC ±60 / ±40; frozen 0/0.
- [x] **Step 2: Plateau counting** — count qualifying plateaus in the smoothed cross-channel standard-deviation signal along each axis to get the grid, *then* refine phase/pitch/duty by a 1-D periodic-square-wave search at that count. Assert the refracted ghost neutrals produce no qualifying plateau (real neutrals peak at 26.2 and 52.4 in the column signal; the ghost zone yields nothing).
- [x] **Step 3: Port `rigid`** — shift measured on the unclipped inner column only, applied to the whole lattice. This is why it has 3× the horizontal range of per-column snapping: a border-clipped column's *visible extent* changes with the shift, so it tracks at half rate.
- [x] **Step 4: Port `ecc`** behind an OpenCV import guard. Do **not** port phase-whitened cross-correlation: it recovers synthetic shifts exactly but collapses the horizontal estimate to zero on real cross-session pairs, a failure that hides behind a perfect synthetic test.

---

## Task 5: QC gate, warnings, and capture metadata

**Files:**
- Create: `_checker_qc.py`; Test: `tests/unit/correction/test_checker_qc.py`
- Modify: `_capture_metadata.py` (as-shot white-balance multipliers)

- [x] **Step 1: `QcLimits` with the calibrated defaults** — displacement 30 px; method disagreement 8 px (reference) / 12 px (reference-free); ECC correlation 0.90; placement margin 0.20 refuse / 0.25 warn; mean tile impurity 0.05; worst tile impurity 0.05 (warn); robust shift 1.5 ΔE; clipped fraction 0.20.
- [x] **Step 2: Fault injection as the test** — flipped card, blanked card, 60 px and 120 px displacement must each trip ≥ 2 signals; a 2.5× exposure change must **pass**, because the gate is keyed to geometry and card integrity, not photometry.
- [x] **Step 3: The missing-tile warning ladder** from the spec — each row gets a test that fires it and asserts the message names the patch or the count.
- [x] **Step 4: WB divergence warning** — record as-shot multipliers in `CaptureMetadata`; warn above 5 % divergence. Measured cost of the mismatch that motivates it: a profile fitted on an auto-WB render applied to a differently-WB'd render of the same exposure gives 5.69 ΔE2000 against 1.10 in-sample (7.10 uncorrected), concentrated in the neutrals.

---

## Task 6: The operation, exports, and registration

**Files:**
- Create: `_calibrate_color_rpcc.py`; Test: `tests/unit/correction/test_calibrate_color_rpcc.py`
- Modify: `_color_checker_profile.py`, both `__init__.py`s, `tests/smoke/test_operation.py`, new `correction/CLAUDE.md`

- [x] **Step 1: Promote the fit entry point** — `ColorCheckerProfile.fit_from_patch_colors(mapping)` public, `_fit_from_patch_colors` retained as a deprecated alias. One solver, one diagnostics schema, one report.
- [x] **Step 2: Assemble `CalibrateColorRpcc._operate`** — stages A–F in order, delegating to `ColorCorrector` for the apply. Stamp `diagnostics["illuminant"] = {"checker", "target", "bradford_adapted"}` and `diagnostics["degree"]`.
- [x] **Step 3: Exclude from the bare-construction smoke contract.** `tests/smoke/test_operation.py` filters `image_ops` by qualname; `CalibrateColorRpcc` requires `rois` and so joins `ColorCorrector` and `GridApply` there, with a comment saying why.
- [x] **Step 4: `patch_census(images, rois=...)`** — counts only. Per-image detected/accepted, the union of missing patches, whether cyan was found in every image, the worst count in the set. It recommends nothing.
- [x] **Step 5: Write `src/phenotypic/correction/CLAUDE.md`**, matching the pattern in `measure/` and `enhance/`.

---

## Task 7: Acceptance run and the PR

- [x] **Step 1: Run the acceptance criteria** against the real frames.

Measured on the three developed *Rhodotorula* frames (`d000220_300_021/_022/_038`),
`rois=[[950, 0, 3100, 340], [950, W-340, 3100, W]]`, `grid=(6, 2)`, degree 3:

| criterion | target | measured | verdict |
|---|---|---|---|
| tiles detected | 24/24 | 24/24 on all three | met |
| identity derived, not declared | margin >= 0.29 | 0.525-0.613 on all six bands | met |
| no fabricated pixel in any mask | — | none: tiles are read where they are, nothing is padded | met |
| patch colour deterministic | exact | equals exhaustive medoid, no RNG | met |
| degree constant across frames | — | 3 on all three, stamped in diagnostics | met |
| Bradford adaptation recorded | True | True | met |
| serialisation round-trip | — | ROIs round-trip; bbox shorthand normalises | met |
| validation script | exit 0 | exit 0 | met |
| **in-sample mean dE2000** | **1.95 +/- 0.15** | **3.40 / 2.88 / 4.74** | **NOT met** |
| **measurement cost** | **<= 5 s/frame** | **68-71 s/frame** | **NOT met** |

- [ ] **Step 2: Close the in-sample accuracy gap.**

The *median* per-patch error is 2.11, at target; the mean is dragged up by a
handful of tiles. On `_022` the worst are purple 8.84, foliage 6.28, and the
four neutrals 4.07-5.86. Two candidate causes, in order of suspicion:

1. **Box centring.** The worst patches also carry the highest within-tile
   spread (foliage 7.5, neutral 6.5 5.0, purple 4.2 against a median near 2),
   which is the signature of a core box straddling a patch edge rather than a
   genuinely noisy patch. Worst core-box standard deviation across a band is
   11-17 where a well-centred box on a uniform patch should be 2-5. The duty
   estimate is the likeliest culprit -- check `fit_phase`'s duty against the
   real tile height before anything else.
2. **The deferred white-balance confound.** Error concentrated in the neutrals
   is its documented signature, and it is upstream of this operation.

Frame `_038` is corrected *worse* than it started (3.70 -> 4.74), which the
first cause would explain and the second would not.

- [ ] **Step 3: Bring the per-frame cost down.** 70 s is not the medoid
  (~3.4 s for 24 tiles at the measured 0.14 s each). Profile before optimising;
  the `median_filter(size=9)` inside `impurity` and `robust_shift`, run per
  tile, and `_pitch_for_row_count`'s phase search are the two candidates.

- [ ] **Step 4: Full `tests/unit` as a Slurm job** on a frozen checkout, against
  a `main` baseline. Not run here: the suite is ~65 minutes and belongs on a
  compute node (**`slurm-job`** + **`run-phenotypic-test`** skills).

- [ ] **Step 5: `uv run mypy`** on the new modules. **Not runnable from the
  agent sandbox** -- it needs the repo's blocked `.venv` interpreter and is not
  installed in the borrowed venv. Type-checking is unverified, not passing.

- [ ] **Step 6: PR** referencing the spec, the evidence doc and the validation
  script.

## Self-review

Checked against the source, 2026-09-21:

- No new module re-implements reference loading, the solver, the geometric
  median or matrix application. `_load_reference_data`,
  `ColorCheckerProfile._fit_from_measured`, `robust_color_center` and
  `ColorCorrector` are each called, not copied.
- Every Lab value in the new code comes from `Image.color.Lab` (operation) or
  is passed in as Lab by the caller (measure, identity, detect). The
  prototype's linear-RGB-into-`rgb2lab` path is not ported.
- `degree` is a plain `int` with no auto mode, no demotion path, and no
  `TuneSpec`. The only raise is rank-insufficiency.
- Identity is never read before `assign_placement` has returned, and its margin
  is gated before the profile is fitted.
- The free Hungarian assignment is reported as `hungarian_agreement` and
  consumed only as a warning.
- `model_dump()` of an operation built from bbox shorthand round-trips to the
  same rectangles (`test_serialisation_round_trips_the_rectangles`).
- Numbers quoted in docstrings are the measured ones, not remembered: the
  medoid timings, the 0.294-0.334 and 0.476-0.606 margin ranges, the
  255.0-256.0 recovered pitch and the 12 enumerated placements were all
  re-measured during implementation.

Two findings from implementation that changed the design, both now documented
in the code they affect:

- The medoid's edge-widening guard is a safety net **at the default candidate
  count only**. On a bimodal cloud at `k <= 64` the winner sits at rank 3 --
  comfortably inside the set -- while being the wrong pixel, so nothing fires.
  Asserted by `test_candidate_medoid_guard_does_not_rescue_a_tiny_candidate_set`.
- Restricting placements to geometrically admissible ones (12 contiguous
  blocks, not 24 arbitrary row pairs) raises the clean-card margin from
  0.294-0.334 to 0.476-0.606 and eliminated silent mislabelling entirely in
  the occlusion trials. The spec's 0.20/0.25 thresholds were calibrated on the
  wider set and are therefore conservative for what ships.
