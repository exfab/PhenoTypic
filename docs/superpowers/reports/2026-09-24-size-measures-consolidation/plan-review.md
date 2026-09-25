# Plan review — size measures consolidation

**Reviewed:** `docs/superpowers/plans/2026-09-24-size-measures-consolidation/plan.md` against
spec `docs/superpowers/specs/2026-09-24-size-measures-consolidation/design.md`, the worktree at
`2004cfac` (`src/` = main `81d19ec6`), and branch `shape-radial-measures` @ `5cad1dfa5`.
**Date:** 2026-09-24. **Reviewer role:** pre-dispatch plan gate (analysis only).

Evidence types used below: `file:line` reads; **probe 1** (read-only script against main's code,
run by the lead, output verbatim in the team thread); **probe 2** (migration-golden status of 4
scenarios on the unmodified tree, run by the lead; full output at
`scratchpad/probe2_full.txt`, summarised in H4); diffs against the branch.

Severity counts: **CRITICAL 0 · HIGH 5 · MEDIUM 8 · LOW 9**.

---

## 0. Port source (`shape-radial-measures`) — verified, blocker resolved

At the start of this review the branch was absent locally and `git cat-file -t 5cad1dfa5`
failed. At 2026-09-24 17:13:11 −0700 `origin/shape-radial-measures` was fetched and the local
branch created (reflog). It resolves to `5cad1dfa58658a1962f63abbbbba41508346e893`, tip message
"test(measure): pin the max-per-bin radial aggregation", 11 commits ahead of `main`, 1,321
behind (matches spec §2). Every plan claim about the branch was then checked:

| # | Plan claim | Result |
|---|---|---|
| 1 | Task 3 Step 6 `_trace_radial_signature` is character-for-character branch `_measure_shape.py` lines 147-210 | **Verified by `diff`**: the `def _trace_radial_signature` … `return signature` block extracted from the branch file and from plan.md is IDENTICAL (64 lines; branch def at line 147). Class (a): self-contained in the plan. |
| 2 | Task 3 Step 3 port of `tests/unit/measure/test_radial_profile.py` | **Applies cleanly.** Branch file defines `TOL = 0.6` (module docstring derives it: half-pixel contour offset ×1.2), `_crop`, `_disk(radius, half=130)`, `_disk_with_runner`, and tests `test_signature_of_a_disk_is_its_radius`, `test_mean_boundary_dist_of_a_disk_is_one_third_of_its_radius` (the one the plan deletes), `test_angular_sampling_matches_the_analytic_mean_radius_of_an_ellipse`, `test_runner_does_not_break_down_the_robust_mean`, `test_boundary_pixel_sampling_would_break_down`, `test_reach_uses_the_outermost_crossing_per_bin_not_the_mean` (uses `MeasureShape(angular_bins=16)`, within the new `ge=8` bound) and `test_degenerate_objects_do_not_raise` (1×1, 2×2, 1×5; asserts Inscribed and RobustMean finite). The only header keys used are `Shape_InscribedRadius`, `Shape_RobustMeanRadius`, `Shape_ReachRadius` and `Shape_MeanBoundaryDist` (only in the deleted test), so the plan's rename list is complete. `MeasureShape(` → `MeasureSize(` covers all constructor sites, including the two `op._trace_radial_signature(mask, edt)` callers. |
| 3 | Task 5 values 4.6951 / 4.8676 / 4.0 are "branch-verified" | **Present** in branch `tests/unit/measure/test_measure_shape.py:60-62`. **Also re-derived analytically** (below), so they no longer depend on the branch. |
| 4 | Boundary-dist descs are "the branch's descs" | Plan text = branch text, with `InscribedRadius`/`RobustMeanRadius` → `Size_InscribedRadius`/`Size_RobustMeanRadius`, "actual radial extent" → "radial extent", **plus an added "Formerly reported as Shape_MeanRadius/MedianRadius." sentence** (see L3). Branch had `tier=1`; the plan drops it per spec §3.2. |
| 5 | Commits `387cef181` (CSV header precedent) and `ae9c5ddef` (golden recapture precedent) | Both exist, and both are on the branch: `387cef18 fix(data): rename stale Shape radius headers in the shipped sample CSVs`, `ae9c5dde test(migration): recapture the MeasureShape golden`. `d846ca4a` (spec §2: `.volume` fix on main) exists: "fix(measure): use ConvexHull.volume for 2D convex area". The branch plan `docs/superpowers/plans/2026-07-09-shape-radial-measures.md` exists on the branch. |

**Analytic re-derivation of the touching-pair values (class (c), now independent of the branch).**
Label 1 is 41×20. Its padded per-object EDT is exact integer arithmetic,
`d(r,c) = min(a_r, b_c)`, where `a = min(r+1, 41−r)` and `b = min(c+1, 20−c)`. The `b`
values are 1..10, each appearing twice. Summed over a row:
`Σ_c min(a,b_c) = 21a − a²` for a < 10, and 110 for a ≥ 10. The rows with a = 1..9 occur
twice each: 2·(21·45 − 285) = 1320. The other 23 rows (a ≥ 10) give 23·110 = 2530. Mean =
3850/820 = **4.695122**.
Label 2 is 41×21 (b has 1..10 twice plus one 11):
- rows with a < 10: 2·Σ(22a − a²) = 1410;
- the two rows with a = 10: 2·120 = 240;
- the 21 rows with a ≥ 11: 21·121 = 2541.

Mean = 4191/861 = **4.867596**. Median of label 1: #{d ≥ k} = (43−2k)(22−2k), so
#{d ≤ 3} = 820 − 490 = 330 < 410 and #{d ≤ 4} = 820 − 396 = 424 ≥ 411. Elements 410 and 411
are both 4, so the median is **4.0 exactly**.

**Amendments (LOW, durability):**
- Reference the port source **by SHA** (`git show 5cad1dfa5:<path>`), not by branch name, and
  add a preflight step `git cat-file -e 5cad1dfa5^{commit}` that fails loudly. The branch was
  missing from this clone until today.
- Add a `check_07_touching_pair_boundary_dist` to `radial_invariants.py`. It uses
  numpy/scipy only: `ndi.distance_transform_edt(np.pad(crop, 1))`. The Task 5 values are then
  re-derived in-repo.

---

## HIGH

### H1 — Review Focus 1 (zero objects) cannot pass with the plan's code (Tasks 3 and 5)
*Plan:* Task 3 Step 2 `test_no_objects_returns_an_empty_frame_with_every_column`; Task 5
Step 1, same test for Shape; Review Focus 1.
*Evidence:* `image.objects` raises `NoObjectsError` when `num_objects == 0`
(`src/phenotypic/_core/_image_parts/_image_objects_handler.py:77-78`). The plan's `_operate`
calls `image.objects.props` and `image.objects.labels2series()` unconditionally, and
`MeasureFeatures.measure` wraps every exception in `OperationFailedError`
(`abc_/_measure_features.py:450-469`). **Probe 1, P1:** on main, both `MeasureShape` and
`MeasureSize` raise `OperationFailedError … NoObjectsError` on an empty objmap. Every other
measurer (e.g. `MeasureIntensity`) has the same behaviour.
*Amendment (pick one before dispatch):*
- **(a, recommended — no new code path):** replace both tests with pins of the existing
  contract, `pytest.raises(OperationFailedError)`, and rewrite Review Focus 1 to match.
  Size/Shape then stay consistent with every other measurer and with the pipeline's existing
  empty-image handling.
- **(b)** add `if image.num_objects == 0: return pd.DataFrame(columns=[OBJECT.LABEL, *SIZE.get_headers()])`
  (same for SHAPE) as the first line of each `_operate`. Say in the spec that this is new
  behaviour that only these two measurers have.

### H2 — Task 8's mechanical sweep breaks a real-data test
*Plan:* Task 8 Steps 1-2 (the exclusion list covers only the equivalence/shape/change-note tests
and `/_golden`).
*Evidence:* `tests/unit/gui/results_viewer/test_measurement_join_migration_run.py:31-33` reads a
**real migrated run**:
`/rhome/anguy344/bigdata_exfab/projects/ucr_029_e_d_Maresca/data/results/2026-08-11-migration-test`.
Its stores were written by old code and carry `Shape_Area`, and spec §6 says old stores keep
their names. The file spells `Shape_Area` 7 times (lines 98-216). The Task 8 grep matches this
file (it is in the 62-file target list, measured). The perl pass turns every mention into
`Size_Area`, and when the run is present (skipif at :67 is false) the test fails on data that
did not change.
*Amendment:* add `tests/unit/gui/results_viewer/test_measurement_join_migration_run.py` to the
Task 8 exclusion list. State the general rule in the plan: **never rename a column name that
a test reads from historical data on disk**. Step 3's hand review should also check any test
that models a *legacy* store or a pre-inversion table (`tests/unit/cli/test_embedded_table_inversion.py`,
`tests/unit/cli/test_metadata_namespace_compat.py`, `tests/unit/gui/results_viewer/test_metadata_namespace_compat.py`).
These are internally consistent synthetic data, so the rename is probably harmless, but they
are the files whose meaning is "old data".

### H3 — Task 6 Step 10 docs build violates the HPCC rules and would execute every notebook
*Plan:* Task 6 Step 10 runs `sphinx-build -b html -j auto docs/source /tmp/pht-docs-size`
locally.
*Evidence:* global CLAUDE.md requires three things: sphinx-build as a Slurm job; `-j
"$SLURM_CPUS_PER_TASK"`, never `-j auto`; and output on shared storage. `docs/source/conf.py:157`
sets `nbsphinx_execute = "auto"`, so a local build runs the notebooks, which is the 626% CPU
incident. Task 9 edits four of those notebooks.
*Amendment:* replace the step with a Slurm job using the `slurm-job` skill. `short` partition,
default account, `--cpus-per-task=8 --mem=32G --time=01:30:00`. Log and output go under the
worktree on `/bigdata`: `docs/_build/size-note/` is already gitignored (`.gitignore:62`); the
generated `measurements_ref/` and `api_reference/api/` are also ignored (`:63`, `:76`).
```bash
PHENOTYPIC_DOCS_BUILD=1 uv run --group docs sphinx-build -b html \
  -j "$SLURM_CPUS_PER_TASK" -D nbsphinx_execute=never \
  docs/source docs/_build/size-note
```
Then read the generated HTML. An exit code of 0 is not evidence. Grep
`docs/_build/size-note/measurements_ref/measurements/index.html` **and** the API pages
`api_reference/api/phenotypic.measure.MeasureSize.html`,
`…MeasureShape.html`, `…phenotypic.schema.SIZE.html` and `…SHAPE.html` for "Changed in version
0.20.0". Those are the three surfaces spec §7 promises. Also confirm the rename table rendered
as a `<table>`.

### H4 — Task 10's golden logic proves nothing about `MeasureIntensity`, and may bless pre-existing drift
*Plan:* Task 10 Step 1: "`measure.MeasureIntensity.parquet` must not appear [in git status]";
spec §9: "must stay byte-unchanged; that proves the decoupling changed no values".
*Evidence:*
- The recapture snippet never writes the Intensity golden, so "byte-unchanged" is true by
  construction and tests nothing.
- The check that compares values, `tests/migration/test_equivalence.py`, is **not in
  `testpaths`** (`pyproject.toml:219`). The Task 10 Step 5 full regression therefore never
  runs it.
- `tests/CLAUDE.md` ("tests/migration runs only when you ask for it — and it is currently
  red") records **57 failing scenarios as of 2026-09-05, including `measure` and `refine`**. It
  says: "Do not regenerate the goldens to get a green run."
- **Probe 2 (measured, unmodified `src/`; `git status --porcelain src/` empty).** The command was
  `pytest tests/migration/test_equivalence.py -k "MeasureShape or MeasureSize or MeasureIntensity or KeepSectionLargest"`.
  Result: 3 failed, 1 passed (exit 1).
  - `measure.MeasureSize`: **PASSED**.
  - `measure.MeasureShape`: **FAILED**. `Shape_ConvexArea` differs on 17.93% of rows; first
    diff 1473.9999999999998 (current) vs 137.9456635036857 (golden). The golden holds the old
    `ConvexHull.area` (a perimeter). It was last written by `36eaff80` (2026-06-01), which
    **predates** the `.volume` fix `d846ca4a` (2026-09-18). The drift is pre-existing and
    explained.
  - `measure.MeasureIntensity`: **FAILED**. `Intensity_MinimumIntensity` dtype is float32
    (current) vs float64 (golden). This is consistent with the float32 intensity-layer change
    noted in `MeasureFeatures._as_accumulation_dtype` (`abc_/_measure_features.py:520-543`).
    The golden was last written by `36eaff80`. The drift is pre-existing.
  - `refine.KeepSectionLargest`: **FAILED**. objmask mismatch 17027/480000 (3.55%), exact
    comparison (tolerance 0.0). **Pre-existing by construction**, because the tree was
    unmodified. The golden was last written by `9941f100` (2026-05-30). Grid code changed
    after it: `74401bbd` "fix centered grid duplicate edges" (2026-07-22) and
    `4b07be7e`/`0c666a6e` (June). The `measure.MeasureSize` golden (the area KeepSectionLargest
    ranks on) still passes. Changed grid-section assignment is therefore the plausible cause,
    not area. Not caused by this change.
- So all three goldens the plan leans on for "no value changed" (Intensity, Shape's retained
  columns, KeepSectionLargest) are red **before** the change. "`MeasureIntensity.parquet`
  byte-unchanged" and "exactly two goldens change" cannot serve as evidence. A Task 10
  recapture of `measure.MeasureShape` would also bless the June→September convex-area drift,
  unrecorded.

*Amendment (replaces spec §9 "Goldens" evidence and plan Task 10 Steps 1-2):*
1. **Differential equivalence, not golden equivalence.** Create a detached worktree at
   `81d19ec66` (M4) and a second one at the branch tip being gated. In each, run the four
   scenarios through `tests.migration._runner.run_scenario(scenario)` on the same frozen
   inputs (`tests/migration/_inputs/`, identical in both trees), and save each result
   (`FrameGolden`/`ImageGolden.save`) to shared storage under
   `/bigdata/exfab/anguy344/gate-worktrees/size-diff/{main,tip}/`, never to `_goldens/`. Then
   compare main against tip directly:
   - `refine.KeepSectionLargest`: `objmask` and `objmap` must be **array-equal**. This is the
     Task 4 selection proof on the migration frozen input.
   - `measure.MeasureIntensity`: every column `assert_frame_equal`, **dtype included**
     (float32 on both sides), with `rtol=1e-10`.
   - `measure.MeasureShape` (tip) vs `measure.MeasureShape` (main): the 8 retained form columns
     equal at `rtol=1e-10`. `Shape_MeanBoundaryDist`/`MedianBoundaryDist` (tip) equal
     `Shape_MeanRadius`/`MedianRadius` (main) **only for objects that neither touch a neighbour
     nor the border**. Report the count and IDs of those that differ (fix 2 and §4.1 change
     them on purpose), and assert those differ in the expected direction (tip ≤ main).
   - `measure.MeasureSize` (tip) vs `MeasureShape` + `MeasureSize` (main): `Size_Area`,
     `Size_IntegratedIntensity`, `Perimeter`, `ConvexArea`, `BboxArea`, `Major/MinorAxisLength`
     equal at `rtol=1e-10`; `Size_InscribedRadius == Shape_MaxRadius` under the same
     no-touch/no-border restriction.

   Write this once as a committed script beside the plan (e.g.
   `docs/superpowers/plans/2026-09-24-size-measures-consolidation/diff_migration_scenarios.py`).
   It drives shipped code, so per root CLAUDE.md it belongs beside the plan and **not** in
   `logic_validation_scripts/`. Run it after Task 4 and again in Task 10.
2. **Goldens.** Do not recapture any golden as part of this change without the user's
   explicit decision, because the Shape and Intensity goldens carry pre-existing unexplained
   (Intensity dtype, KeepSectionLargest) or explained-but-unrecorded (Shape convex area)
   drift. Present the user with two options:
   - **(a)** recapture `measure.MeasureShape` + `measure.MeasureSize` only, with a commit
     message naming both the pre-existing `d846ca4a` convex-area drift and this change's
     column moves, and leave Intensity/KeepSectionLargest red and recorded;
   - **(b)** leave all four goldens untouched and record in the PR that
     `tests/migration` is red for pre-existing reasons, citing the numbers above.

   Either way, drop "must stay byte-unchanged" as a success criterion from spec §9 and plan
   Task 10.
3. Record the pre-change status of these 4 scenarios (above) in RESUME, so a later red
   run is not attributed to this branch.
4. `_GOLDEN_PLATFORM = "linux"` (`test_equivalence.py:78`): HPCC is Linux, so the numeric
   comparisons do run here. The differential script in step 1 is platform-independent anyway.

### H5 — Task 7/8 rewrite pipelines feed binaries and `__pycache__` to `perl -pi`, via `/tmp`
*Plan:* Task 7 Step 5, Task 8 Steps 1-2.
*Evidence:* the Task 7 grep (`grep -rlE … src scripts`, run on the current tree) lists **13
`__pycache__/*.pyc` files** alongside the sources, e.g.
`src/phenotypic/schema/__pycache__/_measurement_info.cpython-312.pyc`. `perl -pi` then
rewrites those binaries, and `Shape_Area` → `Size_Area` changes the string length. Stale
`.pyc` files are usually re-validated against the source, but this is corrupting writes into
the tree, and the same pattern would corrupt any parquet, npz or zarr chunk that happens to
contain the string. Today's tests/ target list is all `.py`, which is luck and not a guard.
The script and target lists also live in `/tmp`: that is node-local, it is not the session
scratchpad the user's instructions require, and Task 8 already expects it to be lost ("recreate
it from Task 7 if `/tmp` was cleared").
*Amendment:*
- Use `grep -rlIE --exclude-dir=__pycache__ --include='*.py' --include='*.csv' --include='*.md' …`.
  `-I` skips binaries.
- Commit the rename script as
  `docs/superpowers/plans/2026-09-24-size-measures-consolidation/size_rename.pl`. It is inside
  the excluded `docs/superpowers/**`, so it never rewrites itself.
- Write target lists to the session scratchpad, not `/tmp`.

---

## MEDIUM

### M1 — Task 7 rewrite breaks a self-contained toy doctest and a CLAUDE.md example
*Plan:* Task 7 Step 5 names `schema/_measurement_info.py (doctests)` as an expected target.
*Evidence:* `src/phenotypic/schema/_measurement_info.py:278-317` defines its **own local**
`class SHAPE(MeasurementInfo)` with `category() == 'Shape'` and `AREA`/`PERIMETER` members.
It is a generic naming-scheme example and not a reference to `phenotypic.schema.SHAPE`. The
perl pass rewrites `SHAPE.AREA` → `SIZE.AREA`, which is undefined in the doctest, and changes
the expected outputs to `'Size_Area'`. The local class still says `Shape`, so the doctest
fails. Step 7 catches it, but only after the implementer has been told it is an expected
target. Similarly, `src/phenotypic/schema/CLAUDE.md:178`
`SHAPE.get_headers()  # ['Shape_Area', 'Shape_Perimeter', ...]` becomes the false
`SHAPE.get_headers()  # ['Size_Area', 'Size_Perimeter', ...]`.
*Amendment:* exclude `schema/_measurement_info.py` from the perl pass. Its mentions at
:289-317, :353, :418, :446, :532 and :553 are generic examples of `{category}_{label}` and
remain correct. In `schema/CLAUDE.md`, change line 178 by hand to a retained member, e.g.
`['Shape_Circularity', 'Shape_Compactness', ...]`.

### M2 — "InscribedRadius is the family's minimum" is false for small objects
*Plan/spec:* spec §4.1 ("It is the family's minimum"); Review Focus 4; the
`SIZE.INSCRIBED_RADIUS` desc ("the distance from the colony center … to its nearest edge").
*Evidence (probe 1, P3, plan's code transcribed):*

| mask | Inscribed | Median | Mean | Robust | Max |
|---|---|---|---|---|---|
| 1×1 | 1.0 | 0.5 | 0.5 | 0.5 | **0.5** |
| 1×9 | 1.0 | **0.938** | 1.3586 | **0.9839** | 4.5 |
| disk r=40 | 40.0125 | 40.0281 | 39.9981 | 40.0069 | 40.5 |

The EDT measures to the nearest **background pixel centre**; the signature measures to the
**0.5 iso-contour**, half a pixel nearer. InscribedRadius therefore sits about +0.5 px above
the contour-based radii. On a speck it exceeds even MaxRadius. On the disk, Mean (39.998) is
already below Inscribed (40.0125).
*Amendment:*
- Spec text (user-gated): "the family's minimum up to half a pixel. The EDT is measured to
  background pixel centres, and the signature to the 0.5 iso-contour."
- Keep the crescent ordering test (probe: 15.0 ≤ 21.32 ≤ 62.31 and 15.0 ≤ 24.44 ≤ 62.31
  hold). Do **not** add an ordering assertion for the disk or degenerate shapes.
- The equivalence with `Shape_MaxRadius` (spec §3.1) forbids "fixing" this by subtracting
  0.5, so document it instead.

### M3 — "The center always lies inside the colony" is false for ring or C-shaped colonies
*Plan/spec:* spec §4.1; `SIZE.MEDIAN_RADIUS` desc ("which always lies inside the colony"),
which is published into every run's README. The branch's RobustMeanRadius desc says the same.
*Evidence:* the centre is the centroid of the plateau component that contains the argmax. For
an annular or strongly curved colony the EDT ridge is an arc: every ridge pixel is within the
1% tolerance, so the whole arc is one plateau component. The centroid of an arc lies in its
concave hole, outside the colony. Ring-shaped colonies (central lysis) are a real morphology.
*Amendment:* reword (user-gated) as "the centroid of the distance-transform peak plateau. It
lies inside compact colonies; for a ring-shaped colony it can fall in the central hole." No
test change is needed.

### M4 — Task 4 Step 6 uses `git stash`
*Evidence:* the environment note says the stash stack is shared with other sessions. Bare
`git stash`/`pop` can pop another session's work.
*Amendment:* compare against main in a detached worktree on shared storage:
`git worktree add --detach /bigdata/exfab/anguy344/gate-worktrees/size-main 81d19ec66`. Then run
`uv run pytest <nodeid> -o addopts= -q -p no:randomly` from that directory, and finish with
`git worktree remove`. The same worktree serves H4 step 1.

### M5 — The e2e files swept by Task 8 are never run
*Plan:* Task 8 Step 4: "The e2e file runs in the Task 10 regression."
*Evidence:* `tests/e2e` is not in `testpaths` (`pyproject.toml:219`) and needs `PLAYWRIGHT=1`
(`tests/CLAUDE.md`). Task 8's target list contains `tests/e2e/gui/test_scatter_tab.py` **and
`tests/e2e/gui/test_analysis_app.py`**. The `run_unit_suite.sbatch` regression runs `tests/unit`.
*Amendment:* add an explicit step to Task 10:
`PLAYWRIGHT=1 QT_QPA_PLATFORM=offscreen uv run pytest tests/e2e/gui/test_scatter_tab.py tests/e2e/gui/test_analysis_app.py -o addopts= -m "not slow" -q -p no:randomly`.
Run it as a Slurm job if it takes more than a couple of minutes. Correct the wording in
Task 8.

### M6 — The phase gates are local, lack `-o addopts=`, and miss the builder surface
*Plan:* Task 4 Step 6, Task 5 Step 9, Task 7 Step 8.
*Evidence:* the `run-phenotypic-test` skill (§3) says to clear `addopts` for any run whose
output goes to a file. The default `--capture=no` dominates the runtime on shared FS. The
Task 5 Step 9 surface (measure+schema+refine+util+docs) takes several minutes, so per the
global rules it is a Slurm job. Separately, `MeasureSize` is the builder tests' canonical
"simple" measurer: `tests/gui/builder/test_state_roundtrip.py:109-254`,
`tests/gui/builder/test_callbacks.py:324`, `tests/unit/gui/builder/test_conversion_dag.py:46-437`.
Task 3 gives it three parameters for the first time, so builder forms, param round-trips and
the tutorial node screenshot are affected. None of these is in the Task 3 surface.
*Amendment:*
- Add `-o addopts= -m "not slow"` to every multi-file run.
- Submit the phase gates through the `slurm-job` skill.
- Add `tests/gui/builder tests/unit/gui/builder` to the Task 3 per-task surface.
- In Task 7 Step 4, check the `gui-tutorial-capture` ledgers against the MeasureSize form
  change as well as the analysis defaults.

### M7 — Decouple-then-flip leaves specific consumers broken between Tasks 5 and 7
*Evidence:* running the Task 5 Step 8 grep today gives these executable and
semantic hits outside the Task 5 files:
- `scripts/capture_gui_tutorial_screenshots.py:1342`, `str(SHAPE.PERIMETER), str(SHAPE.AREA)`:
  an **`AttributeError` at runtime** after Task 5. It is run by the CI `gui-checks`
  smoke-capture (`.github/workflows/gui-checks.yml`). Task 7 fixes it.
- `src/phenotypic/_gui/analysis/_callbacks.py:97,103,109,115,120`, `"on": "Shape_Area"`
  defaults: no import error, but a new run has no such column, so the analysis tab's
  default filter/edge/model fail at run time. Task 7 fixes it.
- The 6 prefabs **silently stop emitting area and size** after Task 5; Task 7 fixes it.
- `src/phenotypic/data/meas/{area_meas,all_meas}.csv` stay self-consistent until Task 7
  renames them. Between Tasks 7 and 8, tests that pair these CSVs with `on="Shape_Area"` are
  red (the plan acknowledges this in Task 7 Step 8).
- `measure/_measure_intensity.py:111,116` is gone by Task 4. The `_measurement_info.py:289-313`
  hits are the toy doctest (M1), not consumers.

*Amendment:* move Task 7 Steps 3-4 (prefabs gain `MeasureSize()`, GUI defaults →
`str(SIZE.AREA)`) **before** Task 5. Both are valid as soon as Task 3 lands, because
`Size_Area` exists from Task 3 on. That leaves only the screenshot script and the CSVs in the
T5→T7 window. State "do not push between T5 and T8".

### M8 — Rectangle tolerance mechanism is misstated (the test still passes)
*Plan:* Task 3 Step 3 `test_elongated_colony_matches_the_analytic_rectangle` docstring:
"Marching squares cuts each corner diagonally, which only lowers MaxRadius."
*Evidence (probe 1, P3):* rasterised 20×100 gives Median 14.1466, Mean 21.1256, Robust 16.1959
and Max 50.8945, against the analytic 14.1, 21.0, 16.2 and 50.9. Mean and Median come out
*above* the analytic values. Max-per-bin takes the extreme vertex in each 1° bin, and where
r(θ) is steep (≈4.5 px/° near the corners) that exceeds the bin-centre value. The bias is up
to half a bin × |r′|, and its average over bins is about ½·(total variation of r)/360 ≈ 0.2 px.
The largest measured deviation is 0.126 px, well inside TOL = 0.6.
*Amendment:* correct the docstring. Name both mechanisms: the corner cut lowers MaxRadius by
< 0.1 px, and the per-bin maximum raises the steep-region bins by ≤ ½ bin × |r′|, a
0.13 px mean shift measured. Keep TOL. The other TOL-based assertions measured within margin:
wide runner Robust 40.04 and gap 2.617 > 1.2; disk Median 40.03 and Mean 40.00; crescent
ordering holds.

---

## LOW

- **L1 — Task 5 Step 3 wording.** "Keep `CIRCULARITY`, `MIN_FERET_DIAMETER` and
  `MAX_FERET_DIAMETER` (still `tier=1`)": `CIRCULARITY` has no `tier=1` on main
  (`schema/_shape.py:43-46`), and the plan's own new classification test puts it in tier 2.
  Reword so that only the Feret diameters are called tier 1, or an implementer may add
  `tier=1` and fail the plan's test.
- **L2 — `KeepSectionLargest` docstring.** Task 4 Step 4 says "Keep the class docstring
  unchanged", but `refine/_keep_section_largest.py:19` says "Measures the pixel area … via
  :class:`MeasureSize`", which becomes false. Change it to "counts each object's pixels".
- **L3 — Change wording in descs beyond spec §7.** The plan adds "Formerly reported as
  Shape_MaxRadius." (INSCRIBED_RADIUS) and "Formerly reported as Shape_Mean/MedianRadius."
  (both BoundaryDist). Spec §7 allows only §4.4's "not the retired same-named column"
  sentence, because descs ship in every run's README. Drop these three sentences, or have the
  user amend §7.
- **L4 — Branch mutation control not ported.** The branch's standing control
  `test_merged_edt_would_fail_this_test` (`test_measure_shape.py:65-81` on the branch) proves
  that the fixture exercises the merge, permanently rather than as a one-off manual mutation.
  Port it into `test_measure_shape.py`; it costs 10 lines.
- **L5 — `RemoveByFeature(feature="MeasureSize")` becomes more expensive.**
  `refine/_remove_by_feature.py:190` runs the named measurer, so an area-filter refine step now
  pays for per-object EDT and contour tracing. That is acceptable, but note it in the PR, since
  spec §5.2 used exactly this argument for KeepSectionLargest.
- **L6 — Label reuse.** `SIZE.ROBUST_MEAN_RADIUS`, `MEAN_RADIUS` and `MEDIAN_RADIUS` share bare
  labels with `RADIAL_EXPANSION` (`schema/_radial_expansion.py:20-36`). The Mean/Median overlap
  already exists on main (SHAPE had them). No gate enforces label uniqueness, but bare-label
  resolution such as `RemoveByFeature(value="MeanRadius")` depends on the `feature` scope.
  Informational.
- **L7 — The `meas` comment is wrong.** Task 7 Step 1 says "`meas` returns a dict copy".
  `meas` is a plain attribute normalised to a dict
  (`_core/_pipeline_parts/_image_pipeline_core.py:389-395, 529`). The test works either way;
  fix the comment.
- **L8 — The capture-script decision is open-ended.** Task 7 Step 6 asks the implementer to
  investigate the "verification run". The simpler rule is to bind two retained Shape columns
  (`SHAPE.SOLIDITY`, `SHAPE.CIRCULARITY`) unconditionally. That keeps the existing "stay
  inside one measurer" invariant (`scripts/capture_gui_tutorial_screenshots.py:1336-1340`),
  but it changes a tutorial screenshot's axes, so check `WORKFLOWS.md`.
- **L9 — `scipy.stats` import cost.** `_measure_size.py` imports `scipy.stats.trim_mean` at
  module level. It is allowed (`scipy` is only in `HEAVY_STARTUP_MODULES`, and `measure/` is
  not imported by `import phenotypic`), but it is the only `scipy.stats` import in
  `measure/`. A function-local import would avoid the ~0.3 s cost for pipelines that never
  measure size.

---

## Verified correct (no change needed)

- **MeasureShape rewrite ≡ main for the retained columns.** Main computes Circularity and
  Compactness from `props.area`/`props.perimeter`, Solidity as
  `props.area / ConvexHull(props.coords).volume` under a Qhull-warning filter, and Feret from
  `props.coords[hull.vertices]` (`measure/_measure_shape.py:157-194`). The plan uses the same
  formulas through `convex_hull_area`, which is identical code. Dropping main's unused
  `enumerate(image.objects)` sub-image construction is a pure speed-up.
- **Per-object EDT ≡ main on the synth plate.** Probe 1 P4, 96 colonies: worst relative
  difference against main's `Shape_MaxRadius`/`MeanRadius`/`MedianRadius` is
  0 / 3.4e-15 / 0. No colony touches the border and no other label enters any padded bbox.
  The rtol=1e-10 equivalence tests in Tasks 3 and 5 hold.
- **MeasureIntensity decoupling.** The replaced block (`measure/_measure_intensity.py:107-117`)
  divides by `props.area` and by the same hull `.volume`. The frame is already a DataFrame at
  that point, so dividing by an ndarray is element-wise and positional. Values are identical.
- **KeepSectionLargest via `np.bincount`.** Probe 1 P5: Otsu on the synth plate gives **552**
  objects (the plan's claim). The bincount + `grid.info(include_metadata=True).merge(...)`
  selection equals main's selection (94 labels). Up to **5-way area ties** in one section, so
  the tie-break order is genuinely exercised. The objmap dtype is uint16 and safe for
  `bincount`.
- **Degenerate objects raise no warnings** under `simplefilter("error")`, including building
  the Image (P2; the "PNG file does not have exif data" line is a print from the plate load,
  not a warning), main's MeasureShape/MeasureSize, and the plan's radial profile (P3).
- **Other checks:**
  - `.. list-table::` is present in `MeasureShape.__doc__` (P4), so Task 6's anchor works.
  - `_class_section` matches the plan's edit target (`docs/source/_extensions/measurements_ref.py:55-69`).
  - `_load_extension` exists (`tests/unit/docs/test_measurements_ref_extension.py:23`).
  - `test_version_is_0_19_0` is at `tests/unit/sdk_/test_norm_migration.py:89-90`. Line 90
    is the only use of `import phenotypic` (line 9), so deleting the test leaves an unused
    import, which the ruff `--fix` in Task 6 Step 11 removes.
  - `tests/migration/_runner.golden_path`/`run_scenario`/`FrameGolden.save` and
    `_scenarios.build_scenarios` exist as used.
  - The migration input `reference_measurements.parquet` already uses `Size_Area`.
  - The `_Plain` test enum is excluded from the first-party enum sweeps
    (`tests/unit/schema/test_measurement_info_format.py:19`).
  - `measure/` is outside the tune annotation gate (`tests/unit/tune/_annotation_introspect.py:32`).
  - The `measurement-info-size` label matches `_section_label`.
  - `myst_parser` is enabled, so Task 9's `{versionchanged}` fence works.
