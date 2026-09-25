# Final review: MeasureNeighborDist nearest-object columns

- **Branch:** `worktree-measure-neighbor-nearest`, `81d19ec6..ab76b86c` (11 commits)
- **Reviewer:** independent fresh-context review, read-only
- **Authority:** `docs/superpowers/specs/2026-09-24-measure-neighbor-nearest/design.md`
- **Scope read:** `src/phenotypic/measure/_measure_neighbor_dist.py`, `src/phenotypic/schema/_neighbor_dist.py`, `src/phenotypic/analysis/_error_cutoffs.py`, `tests/unit/measure/test_measure_grid_spatial.py`, `tests/unit/analysis/test_error_cutoffs.py`, the witness script, the spec, the executor rulings in `.superpowers/sdd/plan/progress.md`, the commit messages (they record the mutation gates), `abc_/_measure_features.py`, `abc_/_grid_measure.py`, both `info()` accessors, and the consumers of `GridSpatial`/`NEIGHBOR_DIST` across `src/`, `tests/` and `docs/`.

**Verification (addendum; the orchestrator ran these after my first handback).**

- **Focused tests:** `test_measure_grid_spatial.py`, `test_error_cutoffs.py`, `test_classification.py` and `test_scatter_grouping.py` gave **105 passed, 1 skipped**.
- **Witness script:** exit 0. Counts were `c1 3452, c2 3452, c3 1477, c4 299, ties 17`.
- **Probe `/tmp/review_nd_probe.py`:**
  - Directional columns are **identical to the base module (`81d19ec6`)** on two synthetic `ManualGridFinder` plates that have multi-object cells, jitter, spills and satellites: 8×12 with n=113, and 16×24 with n=424. The new code is about 2.8× faster.
  - A non-connected label on a `GridImage` is correct (nearest 27 = directional 27).
  - A plain `Image` with non-contiguous labels {1, 2, 5} and an `int32` objmap holding label 4464 are both correct.
  - A plain `Image` with `include_meta=True` gives 3 rows and one `Object_Label` column.
  - An empty `GridImage` gives `(0, 12)`, against `(0, 9)` at base.
  - The empty plain `Image` result and the timings appear under I1 and I3.

---

## Strengths

- **The algorithm is correct as written.** I checked `_nearest_objects` (`_measure_neighbor_dist.py:40-106`) against C1–C4:
  - `binary_erosion` defaults to a 4-connected cross with `border_value=0`, so pixels on the edge of the tight crop are boundary pixels, which is correct.
  - Coordinates are shifted back by `sl[*].start`.
  - `slices[label - 1]` is guarded by `0 < label <= len(slices)`.
  - The stop is strict (`lower[j] > best`).
  - The self entry gets `lower[i] = inf`, so `i` always sorts last. The `j == i` break therefore fires only after every real candidate.
  - `best_j = -1` is never dereferenced, because the first visited `d` is finite and `< inf`.
- **The lower-bound deviation is an improvement.** The spec says `hypot`. The code uses `sqrt` of an exact integer sum (`:93`), the same rounding path cKDTree uses, and the comment says why. A candidate tied at `best` can no longer be pruned by a lower bound that comes out 1 ulp high.
- **The tests are strong where it matters.** `test_matches_brute_force_on_synth_plate` and `test_matches_brute_force_on_random_blobs` compare labels and distances bit-for-bit against an all-pairs cKDTree over full masks, which shares no code with the implementation. The recorded mutation gates show the tests can fail:
  - early exit: 10 failures
  - dropped tie-break: 2
  - interior-only boundary: 26
  - searching all but the last object: fails the synth-plate brute force and the dominance test

  `test_tie_resolves_to_smaller_label_even_when_visited_second` is built to defeat both a non-strict stop and a missing tie-break.
- **The five review-focus items each have a test that can fail:**

  | Focus item | Test | Mutation it catches |
  |---|---|---|
  | Non-contiguous labels | `test_non_contiguous_labels` | an off-by-one on `find_objects` hits a `None` slice and raises |
  | Border-touching objects | `test_border_touching_objects` | square objects with `border_value=1` give an empty boundary; a missing shift gives the wrong distance |
  | Satellite in a ring hole | `test_satellite_inside_ring_hole` | `objmap[sl] > 0` instead of `== label` gives the ring a distance of 0 |
  | Touching colonies | `test_touching_objects_distance_one` | a touching distance other than 1 |
  | `include_meta=True` | `test_include_meta_merges_grid_info` | a duplicated label column or a changed row count |
- **Task 0 is clean.** Only public `image.grid` members are used. The edges are fetched once, pinned by `test_edges_fetched_once_per_measurement`. The windows match the private helper, which serves as a test-side oracle, and a synthetic row-spill fixture was added after the executor found that the synth plate binds only the column terms (a good ruling).
- **The rename is contained.** `git grep` over `src`, `tests` and `docs/source` finds no other consumer of the `GridSpatial_` prefix. `ErrorCutoffFinder` keeps the legacy prefix, and a drift-guard test covers both prefixes. The new schema members author only `label` and `desc` (`bio_desc == ""` is asserted).
- **Lazy-import rules are respected.** A module-level `from scipy.spatial import cKDTree` matches the existing precedent (`_measure_shape.py`, `_measure_grid_spread.py`, three refiners), and `scipy` is already in `HEAVY_STARTUP_MODULES` behind the lazy measure import.
- **The provenance rule does not apply.** The measurer runs no `ImageOperation` or pipeline. `grid.info()` runs the finder's `measure()`, as the pre-branch code already did.

---

## Issues

### Critical

None found.

### Important

**I1. Large objects can make the nearest-object search slow.** (Confirmed by measurement.)

- **Measured:** On a 32×48 plate with 1551 objects, `_nearest_objects` takes **0.73 s** and the full `measure()` takes **4.80 s**. Adding one plate-rim ring of 71,304 px that encloses the colonies raises `_nearest_objects` to **10.56 s**, 14× slower from a single object. A fragmented 36,555 px contaminant on a 96-colony plate stayed cheap at 0.36 s, because its lower bounds are not all 0. So the slow case is a large object whose bounding box encloses many colonies. That is exactly the rim or agar-edge artifact.

- **Where:** `_measure_neighbor_dist.py:96-101`.
- **What:** The search always queries the target's boundary against the candidate's tree: `trees[j].query(boundaries[i])`. When the target is a large object, many candidates have a lower bound of 0, and each of them is queried with the target's whole boundary. Examples of such objects:
  - a plate-rim or agar-edge artifact
  - a spreading fungal colony
  - a fragmented contaminant that shares one label

  With a rim of ~10–50k boundary pixels around a 1536-colony plate, the lower bound is 0 for every colony. The cost is then n × |boundary_rim| × log|boundary_colony|, on the order of 10⁸ tree operations for that one target.
- **Why it matters:** The spec's cost section (§3.5) assumes plate-scale objects. A user who measures raw detections, before a border-removal refiner, gets a measurement that is much slower than the spec implies. The result is still correct.
- **Fix:** Distance is symmetric, so query the smaller point set against the larger set's tree:

  ```python
  small, big = (i, j) if len(boundaries[i]) <= len(boundaries[j]) else (j, i)
  if big not in trees:
      trees[big] = cKDTree(boundaries[big])
  d = float(np.min(trees[big].query(boundaries[small], k=1)[0]))
  ```

  The brute-force tests pin exact equality, so this change is covered.
- **Re-measure after the fix:** rerun `/tmp/review_nd_probe.py`, section "1536 timing + rim", and expect roughly the 0.73 s baseline.

**I2. The eligibility wiring in `_operate` is never exercised end-to-end.** (High confidence that this is a test gap.)

- **Where:** `_measure_neighbor_dist.py:195-223`.
- **What:** The grid branch computes `eligible_rows` from rows with a NaN grid position, scatters `e_label`/`e_dist` back through `eligible_rows`, and maps nearest labels to info rows through `label_to_row`. No test gives `_operate` a frame with a NaN grid row. The spec (§3.5, test 8) drives the off-grid exclusion only through the helper (`test_ineligible_label_is_never_returned`).
- **Mutation the current tests would miss:** replace line 196 with `eligible_rows = np.arange(len(info))`. Off-grid objects then become candidates, and `_nearest_relation` receives NaN rows. Every test stays green, because no fixture produces a NaN grid position. A second mutation also survives: scattering with `nearest_label[:len(e_label)] = e_label`, which differs only when the ineligible rows are not all trailing.
- **Why it matters:** The spec (§3.3 and §4) promises that an off-grid object keeps a row with NaN in the new columns and is never anyone's nearest. That promise is enforced only at helper level.
- **Fix:** Add one `_operate`-level test that monkeypatches `GridAccessor.info` (the class the existing edge-spy test already patches) to return the real frame with one row's `Grid_RowNum` set to NaN. Place that object physically closest to another object. Assert that:
  - its row has NaN in all three new columns
  - no other row reports it as `NearestObjLabel`
  - the other rows' `NearestRelation` values are finite

**I3. An empty plain `Image` raises, while an empty `GridImage` returns 0 rows. Neither case is tested.** (Confirmed by probe.)

- **Where:** `_measure_neighbor_dist.py:198` (`image.objects.info(...)`); spec §4 ("Empty object map | the existing `measure()` contract applies; no rows").
- **What:**
  - Plain `Image` with no objects: `image.objects` raises `NoObjectsError`, which `MeasureFeatures.measure` wraps as `OperationFailedError: ... No objects currently in image`.
  - `GridImage` with no objects, under `ManualGridFinder`: the measurer returns a `(0, 12)` frame. At base it returned `(0, 9)`.

  So the same empty plate gives two different outcomes depending on the input type. The spec's "no rows" is true only for `GridImage`. Raising `NoObjectsError` is probably what every sibling measurer that reads `image.objects` does on a plain `Image`. If so, this is the "existing contract", and the defect is the spec wording plus the missing test, not the code.
- **Why it matters:** A user running a plain-`Image` pipeline on an empty plate (a failed inoculation, a blank control) gets a hard failure from this measurer. Whether that is acceptable depends on how the pipeline or CLI already handles `NoObjectsError` from other measurers. Nothing pins either behaviour, so a later change could flip it silently.
- **Mutation the current tests would miss:** any change to empty-image handling on either input type, for example guarding with `if image.num_objects == 0: return <empty frame>` on only one branch.
- **Fix:** Pick one of two options.
  - (a) Keep the sibling-consistent behaviour. Then correct spec §4 to say a plain `Image` raises `NoObjectsError`, and add `pytest.raises(OperationFailedError, match="No objects")` for the plain `Image` plus a `(0, 12)` exact-columns assertion for the `GridImage`.
  - (b) If the measurer should never fail on an empty plate, short-circuit `image.num_objects == 0` to an empty frame with the 12 columns on both paths, and test both.

  (a) is the smaller change. Check first what `MeasureSize`/`MeasureShape` do on an empty plain `Image`, and match them.

### Minor

**M1. The column descriptions leave out NaN cases.**

- **Where:** `schema/_neighbor_dist.py:54-68`.
- **What:** The `NearestObjLabel` description says NaN happens only "when fewer than two eligible objects exist". It is also NaN for an ineligible (off-grid) object's own row. The `NearestDistance` description names no NaN condition at all.
- **Why it matters:** These `desc` strings are what the deliverables README publishes to users.
- **Fix:** Add "or when this object has no grid cell (GridImage)" to both descriptions.

**M2. `ErrorCutoffFinder` now reports a cutoff for a label column and a category code.**

- **Where:** `analysis/_error_cutoffs.py:39`.
- **What:** The `NeighborDist_` prefix makes `NeighborDist_NearestObjLabel` (an object ID) and `NeighborDist_NearestRelation` (a 0–3 code) "phenotype" columns. The finder would compute an AUC and cutoff on label numbers and would add them to the BH family. This is pre-existing for the four `*NeighborObjLabel` columns under `GridSpatial_`; the branch extends it by two columns.
- **Fix:** Exclude label and code headers, e.g. `NEIGHBOR_DIST` members whose label ends in `ObjLabel` or equals `NearestRelation`. Alternatively, record this as a known limitation in a follow-up.

**M3. The private-API guard is easy to bypass.**

- **Where:** `test_measure_grid_spatial.py:431-434`.
- **What:** The guard checks `"grid._" not in inspect.getsource(mod)`. Both `g = image.grid; g._adv_get_grid_section_slices(...)` and `getattr(image.grid, "_adv_...")` pass it.
- **Mitigation:** The real protection is `test_edges_fetched_once_per_measurement`, which would catch a regression back to the private helper, because the helper calls the edge getters per section. That makes the weak guard acceptable.
- **Fix, if wanted:** Walk the AST for any `Attribute` whose `attr` starts with `_` on a `grid` receiver.

**M4. The witness script's C4 check cannot fail, and the witness does not cover the crop-local path.**

- **Where:** `nearest_object_bounds.py:147-153`.
- **What:**
  - "Subset min ≥ full min" is true by definition, so the check proves nothing about the real C4 claim (EDT and cKDTree agree bit-for-bit). That claim is pinned only by `test_adjacent_nearest_equals_directional_distance` and `test_nearest_never_exceeds_any_directional_distance`.
  - The witness erodes the full image (`:68-71`), not a tight `find_objects` crop, so it never witnesses focus item 2 (crop-edge erosion).
  - The witness uses `np.hypot` (`:86`), while the code uses integer `sqrt`.
- **Fix:** Either replace the C4 block with an EDT-versus-cKDTree equality check on random pairs (scipy only), or relabel it as a sanity check. Add a crop-local boundary variant to the C1 check.

**M5. The spec text still describes the old lower-bound formula.** Spec §3.5 step 3 still says `hypot`. Add a one-line note that the implementation uses `sqrt` of the integer sum, and why. That keeps the spec, the code and the witness from drifting further apart.

**M6. The test file and class names still use the old category name.** `tests/unit/measure/test_measure_grid_spatial.py`, `TestMeasureGridSpatial` and `TestMeasureGridSpatialIntegration` keep the retired name, and `TestMeasureGridSpatial.test_output_has_required_columns` still lists only the 8 directional columns. The exact-schema test now supersedes it. Rename the file in a follow-up so a future grep for `GridSpatial` returns only the legacy-prefix lines.

**M7. A legacy column can now appear as a Colony-tab axis.** `colony_view/_grid.py:140` derives its excluded prefixes from the current schema categories, so `GridSpatial_` is no longer excluded. In an older run's table, a legacy `GridSpatial_*` column with 2–50 distinct values (a small plate) would now be offered as a Colony-tab grid axis. Low impact, since distance columns usually exceed the cardinality cap. It is a side effect of the rename that the spec's §3.9 consumer table does not list.

**M8. The `include_meta` test does not check the measurement values.** `test_include_meta_merges_grid_info` (`:774-778`) asserts the row count, one label column and that `Grid_RowNum` is present. It does not assert that the three new columns survive the merge with the same values as a plain `measure()` call. A one-line `assert_series_equal` against `synth_neighbor_df`, keyed by label, would close this. The plain-`Image` `include_meta` path, which goes through `objects.info`, works in the probe (3 rows, one label column, correct nearest values for labels {1, 2, 5}) but has no test.

---

## Declined to judge

- **The missing migration golden.** `tests/migration/_goldens/measure.MeasureGridSpatial.parquet` is named for a class that no longer exists, so the scenario `measure.MeasureNeighborDist` has no golden, and that was already true before this branch. `tests/migration` is not in `testpaths`. The base-class move also changes the scenario's category from `DETECTED_GRID` to `DETECTED_PLATE` (`tests/migration/_scenarios.py:522,532`). Outside the plan; flagged for whoever maintains that harness.
- **The directional pass's own semantics** (Voronoi shielding, window construction). The spec lists these as non-goals. I reviewed only the Task 0 refactor's equivalence.
- **The cost of three grid fits per measurement** (`info`, row edges, column edges, plus a fourth under `include_meta`). This is down from 805 and consistent with the public-API rule.
- **Labels above 65535 and objmap storage dtype.** Accessor behaviour, not this branch's. `find_objects` handles any positive integer label.
- **Diagonally touching colonies** (distance √2). The math covers it. Non-connected labels were confirmed correct by the probe.
- **The numeric relation code rather than a string.** A spec decision, justified in §3.4.
- **The header rename breaking external user scripts.** The spec accepts this in §7. The PR description must call it out (RESUME.md step 4 already says so).
- **The O(n² log n) cost of `lexsort` across targets.** About 1 s at 3000 objects. Not worth complicating.
- **Missing doctest examples on the class.** Pre-existing; not added or removed by this branch.

---

## Recommendations (in priority order)

1. Add the I2 test for the NaN-grid-row wiring in `_operate`. It closes the only production mutation found that survives the suite.
2. Settle I3. Choose between raising (like the sibling measurers) and returning no rows on an empty plain `Image`, correct spec §4 to match, and pin both input types with tests.
3. Apply the I1 smaller-side query. It is two lines, and the existing brute-force tests pin exactness. The rim case measured 10.56 s against a 0.73 s baseline.
4. Fix the M1 descriptions, which ship to users through the README.
5. Optionally: the M2 exclusion (or a follow-up issue), the M4 witness correction, and the M5 spec note.

---

## Assessment

**Ready to merge: With fixes.** The nearest-object search is exact, and the tests compare bit-for-bit against a brute force that shares no code with it. The recorded mutation gates show those tests can fail, and 105 focused tests pass. The directional values match the base on two new synthetic plates. Before merging:

- add a test for how `_operate` handles a NaN grid row
- decide how an empty plain `Image` should behave, and pin it with tests
- ideally apply the two-line smaller-side query, which removes a measured 14× slowdown when the plate carries a rim-sized object

Every other finding is documentation or cleanup.
