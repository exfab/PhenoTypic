# Phase-1 deep review: implementation and tests (Tasks 2–6)

Reviewer: phase-1 implementation-test reviewer. Diff reviewed: `2004cfac..99adf692`
(`4d51d6b7`, `e3acb345`, `b6f14301`, `1decc582`, `34c87594`, `99adf692`). Spec §10 decisions were
taken as given and are not re-argued here. All probe output quoted below came from two read-only
probe scripts that the orchestrator ran and relayed verbatim (`scratchpad/p1probe_out.txt`,
`scratchpad/p1probe2_out.txt`).

**Counts:** CRITICAL 0 · HIGH 1 · MEDIUM 1 · LOW 8. One finding is [USER]: HIGH-1.

---

## HIGH-1 [USER]: The radial signature traces only the longest 4-connected contour, so a label that is not one 4-connected piece gets wrong or partial radii

**Where:** `src/phenotypic/measure/_measure_size.py:117-120`

```python
contours = find_contours(np.pad(obj_mask, 1).astype(float), 0.5)
...
outline = max(contours, key=len) - 1.0
```

**Mechanism.** `find_contours` defaults to `fully_connected="low"`, which traces the
foreground as 4-connected. PhenoTypic labels objects 8-connected
(`skimage.measure.label(mask > 0)` in `_objmask_accessor.py:203,612,674,735`, default
`connectivity=2`). The merging refiners go further: they relabel separate fragments to a
single label without bridging them. Any label that is not one 4-connected piece therefore
yields several contours. Only the longest is kept, and the centre is found independently,
from the EDT plateau, which can sit in a *different* piece. The four signature radii
(`Median`, `Mean`, `RobustMean`, `Max`) then describe either one piece only or a meaningless
centre-to-other-piece distance. InscribedRadius, Area, the hull columns and every Shape column
use the whole label, so the radius family disagrees with the rest of the row.

**Evidence (probe P3/P3b, synthetic; tip = shipped `MeasureSize`; "all/high" = the same
code with `np.concatenate(contours)` and `fully_connected="high"`):**

| Label | tip Med / Mean / Robust / Max | all/high Med / Mean / Robust / Max |
|---|---|---|
| disk r=15 + a separate 1-px line (same label) | **149.89 / 151.86 / 150.56 / 214.27** | 15.19 / 40.51 / 19.74 / 214.27 |
| two disks, r=15 and r=12, 100 px apart (merged fragments) | 15.03 / 15.02 / 15.04 / **15.5** | 15.07 / 19.90 / 15.08 / 112.5 |
| disk r=20 + 1-px *diagonal* runner out to r≈75 | 20.02 / 20.00 / 20.01 / **20.5** | 20.02 / 20.30 / 20.01 / 75.31 |
| 15-px diagonal line | 0.5 / 0.5 / 0.5 / 0.5 | 0.5 / 0.99 / 0.51 / 20.16 |

In row 1 the centre is in the disk (`center (15.0, 30.0)`), but the longest contour is the
line (contour lengths `[443, 125]`), so a 15-px colony reports MedianRadius 150. Rows 3 and 4
are the case `MaxRadius` exists for: the desc calls it "the colony's furthest reach". A runner
that is only diagonally attached is invisible to it.

**Evidence (probe P4, real pipeline outputs on `load_synth_yeast_plate()`):**

| Objmap | objects | not 4-connected | not 8-connected | MaxRadius, max \|tip − all/high\| (n > 0.5 px) | MeanRadius, max \|d\| (n > 0.5 px) |
|---|---|---|---|---|---|
| `OtsuDetector` | 552 | 111 | 0 | 2.70 (38) | 0.49 (0) |
| + `SmallToLargeMerger()` | 96 | 96 | 96 | 2.71 (51) | 0.54 (1) |
| + `NearestNeighborMerger()` | 452 | 168 | 91 | **35.14 (95)** | **17.07 (62)** |

So 20% of plain Otsu objects are joined only diagonally somewhere, and the public mergers
produce labels made of separate pieces as a matter of course. No test and no differential
check sees this: the baseline, the equivalence test and the differential script all compare
columns that existed on main, and the four signature radii are new.

**Provenance.** This is inherited from the port unchanged. The port source
`5cad1dfa5:src/phenotypic/measure/_measure_shape.py` `_trace_radial_signature` is
byte-identical, and its plan (`5cad1dfa5:docs/superpowers/plans/2026-07-09-shape-radial-measures.md:874`)
uses the same call. Spec §4.1 says "take **the** subpixel marching-squares contour … keep the
outermost distance per bin". The singular assumes there is one contour. The spec never says
"longest", and it does not address labels that yield more than one.

**Options** (a definition choice, so this is the user's call):

- **A. All contours, 8-connected (`fully_connected="high"`, `np.concatenate(contours)`).**
  "Outermost crossing per bin" then holds over the whole label, which matches the spec's
  wording and the whole-label columns. A hole's contour can never win a bin, because the
  outer contour encloses it, so the ring and disk results are unchanged. Effect on plain Otsu
  objects: MeanRadius moves by at most 0.49 px, MaxRadius by at most 2.7 px on 38 of 552
  objects, the diagonal specks it now sees. Fragmented labels get a sensible body radius
  (Median/Robust ≈ the main piece) and a MaxRadius that is the reach of the farthest piece.
  The bins between pieces are empty, so interpolation fills them. That inflates MeanRadius
  for widely separated pieces (40.5 in row 1), and RobustMeanRadius absorbs it. It adds no
  cost. **Recommended.**
- **B. `fully_connected="high"` only, still the longest contour.** This fixes diagonal
  joins, so rows 3 and 4 and the 111 Otsu objects are handled, and it agrees with the
  8-connected labelling. Merged-fragment labels still give row-1 garbage whenever a
  non-central piece has the longest outline.
- **C. Contour of the piece that contains the centre** (the plateau's component).
  Deterministic, and never garbage: it gives the compact body of the main piece. But it
  ignores all other pieces, including runners and absorbed fragments, so MaxRadius stops
  meaning reach for those colonies. It needs `fully_connected="high"` as well, or diagonal
  runners are lost as in rows 3 and 4.
- **D. Keep as is and document it.** Not recommended: row 1 is a silently wrong value in a
  public tier-1 column, not a caveat.

**Fix, once chosen:** change `_measure_size.py:117-120` accordingly. Add fixtures for rows
1, 3 and 4 to `tests/unit/measure/test_radial_profile.py`, and assert the chosen semantics
through `MeasureSize().measure(...)`, not only `_measure_radial_profile`, so that the
`props.image` wiring is exercised. Mutation: revert to `max(contours, key=len)` and the new
tests fail. Update the §4.1 wording and the `MEDIAN_RADIUS` desc ("keeping the outermost
boundary crossing in each") to say which contours are sampled. The logic-validation script
does not model multi-piece labels; add a check if option A or C is chosen.

---

## MEDIUM-1: No test contains an object with a hole, so replacing `props.image` with `props.image_filled` passes every test

**Where:** `src/phenotypic/measure/_measure_shape.py:120` and
`src/phenotypic/measure/_measure_size.py:200`. Tests: `tests/unit/measure/test_measure_shape.py`,
`test_measure_size.py`, `test_size_consolidation_equivalence.py`.

`props.image` is the *unfilled* single-label mask, so a hole counts as background. That is
what main's whole-image EDT on the objmap did, and the tip matches main exactly:

```
== P2 ring (hole)                      meanBD   medBD  inscribed
main   (whole-image EDT on objmap)     4.8374   5.0    9.219544457292887
tip                                    4.8374   5.0    9.219544457292887
filled (props.image_filled mutation)  10.2347   9.0   30.01666203960727
```

Every fixture is hole-free: rectangles, disks, runners, specks, and the synth plate. The
equivalence test cannot catch the mutation either. A plausible refactor to `image_filled` or
`image_convex`, for example to "clean" the contour, would therefore double the BoundaryDist
columns and triple InscribedRadius on lysed or ringed colonies with every test green. The
radius-family ring values (`[9.22, 21.36, 25.926, 23.827, 51.179]`) are pinned by nothing.

**Fix:** add a ring fixture (`12 < r ≤ 30`, clear of the border and of other labels) to
`test_measure_shape.py` and assert `MEAN_BOUNDARY_DIST ≈ 4.8374` (abs 1e-4; the value is an
EDT mean over exact integer geometry) and `MEDIAN_BOUNDARY_DIST == 5.0`. Add a matching
`INSCRIBED_RADIUS == sqrt(85)` (9.2195…, exact) assertion to `test_measure_size.py`. The
values equal main's whole-image EDT because the ring touches nothing, so derive them in the
test from `distance_transform_edt(objmap)` as the oracle rather than hard-coding them.
Mutations to prove: `props.image` → `props.image_filled` in each measurer.

---

## LOW-1: The SIZE and SHAPE enum docstrings lose their dedent, so their API pages render the body as a block quote

**Where:** `src/phenotypic/schema/_size.py:118`, `src/phenotypic/schema/_shape.py:83`.

`f"{SIZE.__doc__}\n\n{SIZE.change_note()}"` appends a column-0 block to a docstring whose
body is indented by 4. Probe P6: "SIZE doc min indent after line 0: 0". Sphinx's
`prepare_docstring` then strips nothing:
`['Measure the key size magnitudes of each detected colony.', '', '    Extract colony area, …']`.
The second paragraph onward renders as an RST block quote on
`api_reference/api/phenotypic.schema.SIZE` and `…SHAPE`. The measurer classes already had
margin 0 before this change, because `rst_table` appends at column 0, so they are not newly
affected.

**Fix:** `SIZE.__doc__ = f"{inspect.cleandoc(SIZE.__doc__)}\n\n{SIZE.change_note()}"`, and the
same for SHAPE. `inspect` is stdlib, so the schema import rule holds. Doing the `cleandoc`
inside `append_rst_to_doc` instead would also repair the pre-existing measurer margin, but it
touches every measurer's docs, so it belongs in a separate change. Confirm in the running docs
build's HTML for the two enum pages.

## LOW-2: The `INSCRIBED_RADIUS` desc ties the value to the colony center, which contradicts amendment A3

**Where:** `src/phenotypic/schema/_size.py:66-73`: "the distance from the colony center (see
MedianRadius) to its nearest edge".

The value is the EDT maximum, the distance from the *deepest interior pixel* to the nearest
background pixel centre. The "colony center" in MedianRadius is the plateau centroid. For a
ring that centroid lies in the hole (A3, and the `MEDIAN_RADIUS` desc says so), where the
distance to the nearest edge is not 9.22. **Fix:** "the distance from the colony's deepest
interior point to its nearest edge". Drop "(see MedianRadius)", or say that it coincides with
the center only for compact colonies.

## LOW-3: The BoundaryDist descs omit that the image border now counts as an edge

**Where:** `src/phenotypic/schema/_shape.py:64-80`. Spec §4.1 says border-touching colonies
now report smaller BoundaryDist values than before, which is as much a behaviour change as the
InscribedRadius one, and that desc does state it. **Fix:** append "The image border counts as
an edge." to both Entries' `desc`.

## LOW-4: Stale names and wrong arithmetic in test docstrings

- `tests/unit/measure/test_radial_profile.py:56` and `:128-148` still say "ReachRadius", the
  branch's name. It is `Size_MaxRadius` now.
- `test_runner_pulls_the_mean_but_not_the_robust_mean` (`:213`) says "the analytic gap is 2.35
  px, over twice 2 x TOL", but twice 2×TOL is 2.4. The assertion itself (`> 2*TOL` = 1.2) is
  sound, so correct the sentence to "about twice 2 x TOL" or "over 2 x TOL".
- `test_mean_boundary_dist_of_a_disk_is_one_third_of_its_radius`
  (`test_measure_shape.py:84-86`): "0.3 px (under 1/R of R at R=40 …)" does not derive 0.3.
  Either state the measured rasterisation lift and the margin, or cite the script check's
  number.

## LOW-5: No test pins the positional alignment between `_calculate_sum` and `props` for non-contiguous labels

**Where:** `_measure_size.py:187-201`, `_measure_intensity.py:106-116`. Both depend on
`np.unique(objmap)` order (inside `_calculate_sum`), `regionprops` order and
`labels2series()` order being the same sorted label order. They are, and probe P1 confirms it
with labels `[2, 9, 300]` placed in *reverse* raster order: every row's area, InscribedRadius,
MeanBoundaryDist and Intensity density matched a per-label oracle exactly. But every test uses
labels 1..N in raster order. **Fix (cheap):** keep P1 as a test in `test_measure_size.py`:
three labels, non-contiguous, reverse spatial order, per-label oracle for Area,
InscribedRadius and `Intensity_Density`.

## LOW-6: The `MeasureIntensity` docstring still calls MeasureSize "lightweight"

**Where:** `src/phenotypic/measure/_measure_intensity.py:48-49`. The Task 7 rename sweep keys on
`Shape_*` and `SHAPE.*` tokens, so it will not catch this line. **Fix:** "`MeasureSize` for colony
area, perimeter and the radius family" in Task 7. `_cli/_cli_output_manager.py:530`
(`"MeasureSize" → ["Size_Area", "Size_IntegratedIntensity"]`) is listed in Task 7's targets but
needs a hand edit: the list is incomplete rather than misnamed.

## LOW-7: A zero-object plate now fails KeepSectionLargest with a bare `NoObjectsError`

**Where:** `src/phenotypic/refine/_keep_section_largest.py:57`. Main raised
`OperationFailedError` (from `MeasureSize().measure`). The tip raises `NoObjectsError` from
`image.objects.labels2series()`. `ImageOperation.apply` wraps both in `RuntimeError`, and I
found no consumer that inspects the cause chain for either type (grep of `NoObjectsError` in
`src/`), so nothing breaks. Recorded so the changed `__cause__` is not a surprise. No fix
needed. If one is wanted, add a test that pins `RuntimeError` on an empty objmap.

## LOW-8: Out-of-phase runtime consumers still broken at the tip (already planned)

These are listed so the phase-1 gate's red results are attributed correctly:
`scripts/capture_gui_tutorial_screenshots.py:1342` (`SHAPE.PERIMETER`, `SHAPE.AREA`, an
AttributeError; Task 7 Step 6), `src/phenotypic/data/meas/{area_meas,all_meas}.csv` (Task 7),
`tests/e2e/gui/test_scatter_tab.py` (the only test using a retired `SHAPE.` attribute; Task 8/10),
and `src/phenotypic/schema/CLAUDE.md:179` (A7). No other executable consumer in `src/` builds a
retired name. I checked `SHAPE.<retired>`, `Shape_<retired>` literals, `getattr(SHAPE`/`SHAPE[`,
`"Shape"` category comparisons, `analysis/_error_cutoffs.MEASUREMENT_PREFIXES` (carries both
`Size_` and `Shape_`), `tune/score/_reference_free_scorer.py` (already reads `SIZE.AREA`, and
from Shape only Solidity, Circularity and Eccentricity), `RemoveByFeature`, and the bundled
JSON/YAML. Every other hit is a docstring in Task 7's list.

---

## Checked and found correct (the focus questions)

1. **MeasureSize ordering and NaN.** `image.objects.props`, `_calculate_sum`
   (`np.unique`, nonzero) and `labels2series()` are all sorted by label, so
   `measurements[header][idx]` is safe with non-contiguous labels (P1). A non-empty padded
   mask always has `edt.max() ≥ 1` and at least one contour, so the NaN branch of
   `_measure_radial_profile` cannot be reached in practice. The degenerate cases stay finite
   (tests). Cost is linear in bbox area: 1.79 s for MeasureSize and 1.60 s for MeasureShape
   on one r=1000 colony (3.1 Mpx), and 0.81 s for 552 Otsu objects (P4, P7). Not a concern.
2. **MeasureShape.** The retained columns are line-for-line main's.
   `_calculate_feret_diameters` is byte-identical. A 3-vertex hull works (rotating calipers
   over CCW Qhull vertices). A 2-vertex hull cannot occur, because Qhull raises on collinear
   points and the helper returns `(None, nan)`. `props.image` is the unfilled mask, so holes
   stay background exactly as in main's objmap EDT (P2: identical to 1e-16). The one
   intended direction change: another label *inside* a hole is now background too.
3. **KeepSectionLargest.** Labels come from the objmap, so `bincount` length = max label + 1
   and no label can be out of range. The objmap is uint16 and a label of 65535 works (P5:
   `equal True`, oracle agrees). The merge is `grid.info(include_metadata=True).merge(…,
   on=Object_Label)`, the same left table and key order as `MeasureFeatures.measure(include_meta=True)`,
   so `idxmax` tie-breaking and the dropping of NaN-section objects (`observed=True`) are
   unchanged.
4. **MeasureIntensity.** The frame has a RangeIndex. Dividing a Series by an ndarray is
   positional, and both are in label order (P1: density matches the oracle to 1e-6 on
   non-contiguous labels).
5. **change_note.** Only `MeasureSize`/`MeasureShape` call `SIZE`/`SHAPE.append_rst_to_doc`.
   `QUALITY_CHECK` overrides `append_rst_to_doc` and inherits the empty default note.
   Pydantic field descriptions are still parsed from `Args:` (P6: `angular_bins` description
   intact), and the GUI registry and modal browser read `Args:` and the first line, which are
   unaffected. The README generator publishes `desc` only, not `__doc__` or the note. The
   Measurements reference page shows the same note under both the SIZE and SHAPE headings, as
   spec §7 intends. The only defect is the indentation in LOW-1.
6. **Tests.** The radial tests call `MeasureSize()` directly, not a stale Shape path. TOL=0.6
   is derived (half-pixel iso-level plus binning) and each use has a discriminating margin.
   The mutation controls (`test_boundary_pixel_sampling_would_break_down`,
   `test_merged_edt_would_fail_this_test`) are honest. rtol=1e-10 on the synth equivalence is
   justified, and it is only meaningful because no synth colony touches another or the
   border, as the module docstring says. The gaps are MEDIUM-1, LOW-5, and the missing
   multi-piece fixtures in HIGH-1.
7. **Consumers.** See LOW-8. Nothing else found.
