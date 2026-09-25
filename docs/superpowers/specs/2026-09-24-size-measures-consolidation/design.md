# Size measures consolidation: `MeasureSize` becomes the single source of colony size

**Date:** 2026-09-24 · **Status:** design approved in brainstorming; awaiting spec review
**Release:** minor bump, `0.19.0` → `0.20.0` (public measurement columns break)
**Logic-validation script:**
[`logic_validation_scripts/2026-09-24-size-measures-consolidation/radial_invariants.py`](../../logic_validation_scripts/2026-09-24-size-measures-consolidation/radial_invariants.py)
(`uv run --no-project --with numpy --with scipy python radial_invariants.py`). Every number
this spec quotes about disks, rectangles, touching colonies and runners is re-derived there.

## 1. Objective

`MeasureShape` currently mixes **size magnitudes** (area, perimeter, radii, axis lengths,
hull and box areas, all tagged `tier=1`) with **form descriptors** (circularity, solidity,
eccentricity, …). `MeasureSize` emits only `Area` and `IntegratedIntensity`, so area is
computed and published twice (`Shape_Area`, `Size_Area`).

After this change, `MeasureSize` is the **only** emitter of the key starting size
measurements, and `MeasureShape` keeps the form descriptors. The radius columns are rebuilt
so their names match their values.

### Non-goals

- The Feret diameters stay in Shape (user decision).
- There is no read-side alias for retired column names; this is a hard break (§6).
- `Size_IntegratedIntensity` duplicates `Intensity_IntegratedIntensity`, and its desc formula
  ("sum × area") is wrong. That is a separate dedupe.
- No `EquivalentRadius = sqrt(Area/π)` column is added.
- The unrelated commits on branch `shape-radial-measures` (contrast enhancers, XYZ
  conversion) are out of scope.

## 2. Background: what is wrong on main today

Branch `shape-radial-measures` (tip `5cad1dfa5`, 2026-07-10, never merged and now 1,321
commits behind main) found three defects. Its plan is
`docs/superpowers/plans/2026-07-09-shape-radial-measures.md` on that branch.

1. **`Shape_Mean/MedianRadius` are not radii.** They are the mean and median of the
   Euclidean distance transform (EDT) over the interior, i.e. depth from the nearest edge.
   On a disk of radius R they equal R/3 and R(1−1/√2) ≈ 0.293R (script check 01).
   `Shape_MaxRadius` is the EDT maximum, which is the inscribed-circle radius, not "the
   furthest extent from the centre" as its desc claims.
2. **Touching colonies merge.** `distance_transform_edt(image.objmap[:])` binarizes its
   input, so two labels sharing an edge see no background between them. A 41×20 and a
   41×21 colony sharing their long edge report 20/21 instead of 10/11 (script check 02).
3. **Convex area.** On main, `d846ca4a1` already corrected `ConvexHull.area` (a perimeter
   in 2-D) to `ConvexHull.volume`. **That fix stands**: any convex area taken from scipy
   uses `ConvexHull.volume`. The branch replaced it with `regionprops.area_convex`; this spec
   does **not** port that. Consequence, stated so nobody "fixes" it later: Solidity =
   area / hull volume can slightly exceed 1, because the hull runs through pixel centres.

The branch is used as a **source to port from, not rebased**. Rebasing a 1,321-commit-stale
branch onto a file that is being split costs more than re-implementing its ~200 lines, and
users should see one column break, not two.

## 3. Resulting column sets

### 3.1 `SIZE` (`DirectPhenotype`, category `Size`)

| Column | Computation | Relation to old columns |
|---|---|---|
| `Size_Area` | objmask pixel sum (existing) | `Shape_Area` dropped (same values) |
| `Size_IntegratedIntensity` | existing | unchanged |
| `Size_Perimeter` | `props.perimeter` | was `Shape_Perimeter` |
| `Size_ConvexArea` | `ConvexHull(props.coords).volume`, NaN on `QhullError` | was `Shape_ConvexArea`, identical values |
| `Size_BboxArea` | `props.area_bbox` | was `Shape_BboxArea` |
| `Size_MajorAxisLength` | `props.axis_major_length` | was `Shape_MajorAxisLength` |
| `Size_MinorAxisLength` | `props.axis_minor_length` | was `Shape_MinorAxisLength` |
| `Size_InscribedRadius` | per-object EDT maximum | value of old `Shape_MaxRadius`, now per object (fix 2) |
| `Size_MedianRadius` | median of the radial signature | **new meaning**, not old `Shape_MedianRadius` |
| `Size_MeanRadius` | plain mean of the radial signature | **new meaning**, not old `Shape_MeanRadius` |
| `Size_RobustMeanRadius` | symmetric trimmed mean of the radial signature | new (branch) |
| `Size_MaxRadius` | maximum of the radial signature | **new meaning** (the branch's `ReachRadius`), not old `Shape_MaxRadius` |

The moved members drop their `tier=1` tags. `DirectPhenotype` already resolves to tier 1,
so resolved tiers are unchanged.

### 3.2 `SHAPE` (`PrimaryMeasure`, category `Shape`) keeps

Circularity, Compactness, Solidity, Extent, Eccentricity, Orientation, MinFeretDiameter,
MaxFeretDiameter, **plus**:

| Column | Computation | Relation to old columns |
|---|---|---|
| `Shape_MeanBoundaryDist` | mean of the per-object EDT over the interior | value of old `Shape_MeanRadius`, now per object |
| `Shape_MedianBoundaryDist` | median of the same | value of old `Shape_MedianRadius`, now per object |

The boundary distances describe interior thickness, which is a form property. They drop
`tier=1` and take SHAPE's default tier 2. The Feret diameters keep `tier=1`, so SHAPE stays
a straddler.

## 4. The radius family

### 4.1 Definition

All five radii are statistics of one set of centre-to-edge distances.

- **Centre.** The centroid of the object's EDT peak plateau (`edt >= (1 −
  plateau_tolerance) · max`, restricted to the 8-connected component containing the argmax,
  matching the 8-connected object labelling: a diagonal run of tied pixels is one plateau).
  It lies inside compact colonies; for a ring-shaped colony (e.g. central lysis) the
  plateau is an arc and its centroid can fall in the central hole. A plateau centroid is used rather than the argmax
  because EDT values are square roots of integers, so exact ties are common and an argmax
  would break them in raster order.
- **Radial signature.** Take **all** subpixel marching-squares contours at the 0.5
  iso-level of the padded crop, traced 8-connected (`fully_connected="high"`, matching the
  8-connected object labelling), and pool their vertices. Once pooled, the connectivity
  setting cannot change the result: it only decides how a saddle cell's four edge
  crossings pair into contours, never which crossings exist. Bin the vertices by angle about
  the centre into `angular_bins` equal bins, and keep the outermost distance per bin (max,
  not mean: the branch's mutation-pinned choice). Empty bins are filled by circular
  interpolation. "Outermost crossing per bin" therefore holds over the whole label.
  **Holes are filled before tracing** (`binary_fill_holes` on the contour input only; the
  centre and InscribedRadius stay on the unfilled mask). Along any ray a hole's crossing
  lies inside the outer one, but an outline of radius R has only about 8R vertices, so
  below R ≈ 45 the outer outline leaves bins empty. A hole's vertices would then fill some
  of them with the hole's smaller radius, where interpolation from the outer neighbours
  belongs. Measured without the fill: 18 bins won on a ring of radius 12, 1 on radius 30, and
  a change above 0.05 px in 87 of 552 Otsu colonies on the synthetic plate. For a **fragmented
  label** (pieces joined only at a corner, or separate fragments a merging refiner put under
  one label), every piece contributes: MaxRadius is the reach of the farthest piece, the
  bins a far piece spans carry its distance, and bins between pieces are interpolated. So
  MeanRadius rises with the far pieces' angular span, MedianRadius stays on the central
  piece while the other pieces span under half of all directions, and RobustMeanRadius stays
  on it while they span under `trim_proportion` of them. The branch traced only the longest
  4-connected contour, which dropped diagonally attached runners and, when a non-central
  fragment had the longest outline, measured the distance to that fragment instead (phase-1
  review HIGH-1).
- **InscribedRadius** is the EDT maximum, which is the exact distance from the centre to
  the nearest edge. It is the family's minimum **up to half a pixel**: the EDT measures to
  background pixel *centres*, the signature to the 0.5 iso-contour, which is half a pixel
  nearer. So on a disk of radius 40 MeanRadius (39.998) sits just below InscribedRadius
  (40.0125), and on a one-pixel speck InscribedRadius is 1.0 against a MaxRadius of 0.5.
  Equivalence with the retired `Shape_MaxRadius` (§3.1) forbids "fixing" this by subtracting
  0.5, so it is documented instead. It is taken from the EDT rather than from
  the signature's minimum, because the nearest bin centre sits up to half a bin off the
  perpendicular and overshoots by h/cos(π/K) − h (script check 04).
- **The image border counts as an edge.** Each EDT runs on a one-pixel-padded crop, so a
  colony cut off by the image border measures its distances to that border. The old
  whole-image EDT did not: scipy measures distance only to zero pixels *inside* the array.
  So a border-touching colony now reports a smaller InscribedRadius and smaller
  BoundaryDist values than before. For example, a 10-row band spanning the full image width
  along the top edge reports 5 now and 10 before. This is deliberate, because the visible
  part of the colony is all that is measured.

### 4.2 Why angle sampling

Sampling by angle gives a runner only its true angular width. In the branch's worked case
(r = 40 colony, runner of half-width 3 reaching x = 90, 20% trim):

| Sampling | Share of samples beyond r = 45 | 20% trimmed mean |
|---|---|---|
| Inner boundary pixels | 29.9% (> the 20% breakdown point) | 42.03 |
| Uniform in angle, K = 360 (analytic) | 2.2% | 40.00 |

The branch plan printed 2.5% / 40.03 for the angle row, measured from its pixel contour;
the analytic value is quoted here. Plain mean on the angle signature: 40.77, pulled up by
the runner (script check 05).

### 4.3 Reference values (script checks 03 and 04)

| | Disk of radius R | Ideal 100×20 rectangle |
|---|---|---|
| InscribedRadius | R | 10.0 |
| MedianRadius | R | 14.1 |
| MeanRadius | R | 21.0 |
| RobustMeanRadius | R | 16.2 |
| MaxRadius | R | 50.9 |

The rectangle values are evaluated analytically at the 360 bin centres; rasterised shapes
differ by sub-pixel amounts. The trim discounts **genuine elongation** as well as runners:
on the rectangle it removes the long ends (16.2 against a mean of 21.0).

### 4.4 Required desc content

Authors write `label` and `desc` only. New members get `bio_desc=""` and `image=None`.

- **InscribedRadius:** the radius of the largest circle that fits entirely inside the
  colony, i.e. the distance from the centre to the nearest edge. **Caveat (required):** it
  reflects the colony's *narrowest* dimension, not its overall extent. An elongated colony
  reports half its width whatever its length (a 100×20 px colony → 10), and a runner or
  spur does not change it. Compare MaxRadius for overall extent.
- **MedianRadius / MeanRadius / RobustMeanRadius / MaxRadius:** state the centre definition
  and the angular sampling, and give the disk and rectangle examples. MeanRadius notes its
  sensitivity to runners and points to RobustMeanRadius. RobustMeanRadius states that it
  estimates the typical radius of the compact body, discounting both runners and
  elongation. Median, Mean and Max each state that they are not the values the retired
  `Shape_*` columns of the same name carried.
- **MeanBoundaryDist / MedianBoundaryDist:** the branch's descs (interior thickness, not a
  radius; R/3 and R(1−1/√2) on a disk).
- **SIZE.AREA:** carries the human-authored `bio_desc` and `image="shape/area.png"` verbatim
  from `SHAPE.AREA`. It is the same quantity, so the text is moved, not authored.

## 5. Architecture

**Principle: measurers read the cached `image.objects.props`, and no measurer runs another
measurer.**

### 5.1 New private module `src/phenotypic/measure/_object_geometry.py`

Two helpers, each the single home of a correctness rule that has been broken before:

- **`convex_hull_area(coords) -> tuple[ConvexHull | None, float]`**: wraps the existing
  Qhull-warning-suppressed `ConvexHull(coords)` and returns the hull (or `None` on
  `QhullError`) plus `hull.volume` (or NaN). It guards the `.volume`-not-`.area` rule.
- **`object_edt(obj_mask) -> np.ndarray`**: `distance_transform_edt` on a one-pixel-padded
  single-object crop (`regionprops.image`), with the padding stripped. It guards the
  per-object rule (fix 2).

### 5.2 Measurers

- **`MeasureSize`** owns every §3.1 column. It gains the branch's pydantic fields
  `angular_bins: int = Field(360, ge=8, le=3600)`,
  `trim_proportion: float = Field(0.2, ge=0.0, lt=0.5)` and
  `plateau_tolerance: float = Field(0.01, gt=0.0, lt=1.0)`, plus the branch's
  `_trace_radial_signature` and a radial-profile helper returning the five radius headers.
  No `TuneSpec` annotations: `measure/` is outside the tune annotation-coverage gate.
- **`MeasureShape`** reads `props.area` and `props.perimeter` internally for Circularity and
  Compactness, and `props.extent`. It uses `convex_hull_area` for Solidity and the hull
  vertices for Feret, and `object_edt` for the boundary distances. It emits none of the §3.1
  columns.
- **`MeasureIntensity`** today runs `MeasureShape().measure(image)` only to divide by
  `SHAPE.AREA` and `SHAPE.CONVEX_AREA`. It switches to `props.area` and `convex_hull_area`.
  `Intensity_*` values are unchanged.
- **`refine.KeepSectionLargest`** today runs `MeasureSize()` only to take the per-section
  argmax of area. It switches to a per-label pixel count, so a refine step does not pay for
  EDTs and radial signatures. Its selections are unchanged.

Rejected alternatives: Shape and Intensity calling `MeasureSize()` internally, which pays
the radial-signature cost several times per image; and a lightweight/full flag on
`MeasureSize` (YAGNI).

**Invariant:** `Size_Area == props.area` for every object. Shape's ratios and Intensity's
density are built on `props.area`, so the published area and the internal denominator must
never diverge.

## 6. Compatibility: hard break

- There is no alias machinery, matching the SymZones rename precedent. The only existing
  legacy-header map (`sdk_/_metadata_compatibility.py`) is metadata-only and stays that way.
- Old OME-Zarr stores keep their `Shape_*` columns. `project_embedded_measurement_table`
  projects each store onto its own recorded columns, so a run mixing old and new stores
  carries both names, each NaN where the other exists. This is documented, not handled.
- A saved recipe or pipeline with `on="Shape_Area"` or
  `RemoveByFeature(feature="MeasureShape", value="Area")` fails at run time.
- **The same-name trap:** old `Shape_MedianRadius` (≈0.29R), `Shape_MeanRadius` (≈R/3) and
  `Shape_MaxRadius` (inscribed radius) are **not** the new same-named `Size_*` columns.

**Rename table.** It goes in the PR description and in the highlighted notes (§7):

| Retired | Successor |
|---|---|
| `Shape_Area` | `Size_Area` |
| `Shape_Perimeter`, `Shape_ConvexArea`, `Shape_BboxArea`, `Shape_MajorAxisLength`, `Shape_MinorAxisLength` | `Size_` + same label |
| `Shape_MaxRadius` | `Size_InscribedRadius` |
| `Shape_MeanRadius` | `Shape_MeanBoundaryDist` |
| `Shape_MedianRadius` | `Shape_MedianBoundaryDist` |
| — (new) | `Size_MedianRadius`, `Size_MeanRadius`, `Size_RobustMeanRadius`, `Size_MaxRadius` |

## 7. Versioning and highlighted change notes

- **Version:** `src/phenotypic/__init__.py` `__version__ = "0.20.0"` (minor bump; pyproject
  reads it dynamically). Update the version pin in
  `tests/unit/sdk_/test_norm_migration.py::test_version_is_0_19_0` to 0.20.0 and rename the
  test. Historical fixtures that record `"version": "0.19.0"` (e.g.
  `tests/unit/measure/_golden/orientation_zones_pre_simplification.json`) are back-compat
  locks and stay unchanged.
- **Highlighted note, one source.** Add a classmethod hook to `MeasurementInfo`,
  `change_note() -> str`, which returns an RST block and defaults to `""`. `SIZE` and
  `SHAPE` override it to return a `.. versionchanged:: 0.20.0` directive: a highlighted
  block in the pydata-sphinx theme, and the repo's existing precedent
  (`sdk_/_io_constants.py:2024`). The text summarises the move, carries the rename table,
  and calls out the same-name trap. The note is rendered in three places, all from that one
  hook:
  1. the **measurements reference page**: `docs/source/_extensions/measurements_ref.py`
     `_class_section` emits `change_note()` between the heading and the table;
  2. the **measurer class docs**: `append_rst_to_doc` emits `change_note()` before the
     table, so `MeasureSize` and `MeasureShape` show it automatically;
  3. the **enum API pages** (`phenotypic.schema.SIZE` / `SHAPE`): the enum docstrings append
     `change_note()` the same way.
- The note is **not** put into member `desc`. The deliverables README generator publishes
  `desc` into every run's `README.md`, so a release note there would ship with every future
  run. The only change-related wording in descs is §4.4's one-line "not the retired
  same-named column" clarification.

## 8. Consumers to update

Blast radius from the consumer map; nothing else reads these columns semantically:

- **Prefabs.** Seven prefabs run `MeasureShape` without `MeasureSize`; add `MeasureSize()`
  to each, or their outputs silently lose area and radii. They are
  `prefab/_heavy_watershed_pipeline.py`, `_grid_section_pipeline.py`,
  `_heavy_otsu_pipeline.py`, `_round_peaks_pipeline.py`, `_heavy_round_peaks_pipeline.py`,
  `_filamentous_fungi_pipeline.py`, and `abc_/_prefab_pipeline.py`.
- **GUI.** `_gui/analysis/_callbacks.py:97-120` has five `on="Shape_Area"` defaults. Change
  them to `str(SIZE.AREA)`.
- **Bundled data.** `src/phenotypic/data/meas/{area_meas,all_meas}.csv`: rename the moved
  headers (branch precedent `387cef181`).
- **Docstrings and doctests** using `SHAPE.AREA` as the example: `schema/_measurement_info.py`,
  the growth-model and outlier analyzers, `sdk_/_metadata_helpers.py:783`, and
  `_gui/_shared/_measurement_tint.py`, among others. Rewrite the `MeasureSize` docstring
  (no longer "lightweight") and the `MeasureShape` docstring. The overview goes on the
  measurer and the per-column detail on the enum, per the root CLAUDE.md split.
- **Docs and scripts.** `explanation/measurement_metrics_biological_meaning.md` (split its
  Shape table), the `07_measuring_and_exporting`, `fit_logistic_growth`,
  `correct_edge_effects` and `linear_softplus_model` notebooks,
  `tutorials/gui/03_build_pipeline.md`, `scripts/capture_gui_tutorial_screenshots.py`, and
  `scripts/make_measurement_example_images.py`. Check the GUI ledgers per the
  `gui-tutorial-capture` skill.
- **Schema docs.** Update the straddler example in `src/phenotypic/schema/CLAUDE.md`.

Self-adapting consumers, needing no change: the results viewer's scatter and grid prefixes,
the README generator, the per-measurer output split, and the tier and kind resolvers. All of
these read the schema through `_measurement_infoclass`.

## 9. Testing

- **Port the branch's tests.** Its `test_measure_shape.py` additions and `test_radial_profile.py`
  move to `tests/unit/measure/test_measure_size.py` and `test_radial_profile.py`, retargeted
  to the SIZE names. The boundary-distance assertions go to `test_measure_shape.py`.
  Coverage:
  - disk identities (all five radii ≈ R; boundary distances R/3 and 0.293R);
  - touching colonies, on **both** measurers;
  - a rasterised 100×20 case;
  - a runner case asserting that MeanRadius moves while RobustMeanRadius stays near the
    body radius (this also catches the two being swapped);
  - the max-per-bin mutation pin.

  The branch's `area_convex` rectangle-solidity test is **not** ported.
- **New guards:**
  - `Size_ConvexArea == ConvexHull(coords).volume` on a known polygon;
  - `Size_Area == props.area`;
  - schema ↔ frame column equality for both measurers;
  - Intensity density == IntegratedIntensity / Size_Area;
  - `KeepSectionLargest` selects the same labels as before on `load_synth_yeast_plate()`;
  - `change_note()` renders in `append_rst_to_doc` output and in the measurements-reference
    section;
  - the version is `0.20.0`.
- **Mutation proofs**, per the test-integrity rule. Each guard must be shown to fail on its
  bug: swap `.volume` → `.area`; reintroduce the whole-objmap EDT; swap mean ↔ trimmed mean;
  change max-per-bin to mean-per-bin; drop `change_note()` from `_class_section`.
- **Goldens.** All four relevant migration goldens were already red on main before this
  change, for pre-existing reasons (measured 2026-09-24, plan-review H4): `MeasureShape`
  predates the `.volume` fix (ConvexArea/Solidity differ on 99/552 rows); `MeasureIntensity`
  has a float32/float64 dtype drift plus the same ConvexDensity cause; `KeepSectionLargest`
  keeps 94 labels against the golden's 96 (the golden predates the grid fixes, `74401bbd`);
  `MeasureSize` passes. So a golden cannot prove "no value changed". The proof is
  **differential**: a committed script runs the four scenarios in a worktree at main and one
  at the branch tip and compares them directly (Intensity dtype-exact at rtol 1e-10;
  KeepSectionLargest array-equal; retained Shape and moved Size columns at rtol 1e-10). Only
  after that passes are **all four** goldens recaptured (user decision, 2026-09-24), with a
  commit message naming each pre-existing drift separately from this change's column moves.
- **Producer-coupled tests to update:** `schema/test_classification.py:121-133`,
  `schema/test_schema_public_api.py`, `schema/test_dynamic_headers.py`,
  `util/test_measurement_outputs.py`, `gui/analysis/test_standalone_bundle.py`,
  `gui/results_viewer/test_scatter_grouping.py`, `e2e/gui/test_scatter_tab.py`, and
  `analysis/test_log_growth_model.py`. About 60 other test files use `Shape_Area` only as an
  arbitrary synthetic column name; sweep them to `Size_Area` mechanically.
- **Verification ladder** (root CLAUDE.md): per task, the touched test files; per phase, the
  importer-derived surface of `measure/`, `schema/`, `prefab/` and `refine/`; at the end,
  one full sharded regression via `run-phenotypic-test`. Also `uv run mypy src/phenotypic`,
  `uv run ruff check --fix <changed paths>`, and a docs build to confirm the
  `versionchanged` blocks render.

## 10. Decisions log (brainstorming, 2026-09-24)

| Decision | Choice |
|---|---|
| What "one source" means | Size columns are removed from Shape |
| Which groups move | Radii; perimeter plus hull and box areas; ellipse axes. Feret stays |
| Unmerged radial branch | Port into SIZE; do not rebase |
| Back-compat | Hard break, no alias |
| Convex area | `ConvexHull.volume` wherever scipy supplies it |
| Radius family | Inscribed / Median / Mean / RobustMean / Max; `InscribedRadius` name kept, with the elongation caveat |
| Old edge-distance stats | Stay in Shape as Mean/MedianBoundaryDist |
| Release | Minor bump to 0.20.0, with highlighted `versionchanged` notes on class and measurement docs |

**Pre-dispatch plan review (2026-09-24)**, report
`reports/2026-09-24-size-measures-consolidation/plan-review.md`:

| Decision | Choice |
|---|---|
| Zero-object plate | Keep main's contract: `MeasureSize`/`MeasureShape` raise `OperationFailedError` (`NoObjectsError`), like every other measurer. Tests pin the raise |
| Migration goldens | Differential main-vs-tip proof first, then recapture all four (Shape, Size, Intensity, KeepSectionLargest) |
| §4.1 wording | InscribedRadius is the minimum only up to half a pixel; the centre can fall in the hole of a ring-shaped colony |
| "Formerly reported as …" in descs | Dropped: §7 allows only the same-name clarification in `desc` |

**Phase-1 review (2026-09-24)**, report
`reports/2026-09-24-size-measures-consolidation/phase1-impl-test-review.md`:

| Decision | Choice |
|---|---|
| Which contours the radial signature samples (HIGH-1) | Option A (user decision): all contours of the label, traced 8-connected (`fully_connected="high"`), pooled with `np.concatenate`. A fragmented label is measured whole: MaxRadius is the farthest piece's reach. Not chosen: B, the longest 8-connected contour only, which still measures the wrong piece when a non-central fragment has the longest outline; C, the contour of the centre's piece, under which MaxRadius stops meaning reach; D, keep the longest 4-connected contour and document it |
| Holes under option A | Filled before tracing, which realises A as it was presented ("a hole's contour can never win a bin; ring and disk unchanged"). Pooling alone let hole vertices fill bins that the outer outline left empty (found while implementing A; 96 of 552 synthetic-plate Otsu colonies have holes) |
| Plateau connectivity | 8-connected (`structure=np.ones((3, 3))`), following the user's "treat diagonal contacts as connected". With 4-connectivity a diagonal line's centre was its end pixel |
