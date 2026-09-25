# Final whole-branch review: size measures consolidation

**Scope:** `git diff 81d19ec66..cd1fe146` (23 commits, 136 files). All code was read at
`cd1fe146` through `git show`/`git archive`, never from the working tree.
**Spec:** `specs/2026-09-24-size-measures-consolidation/design.md`, with §10 treated as decided.
**Prior reviews read, and not repeated here:** plan-review, phase1-impl-test-review,
phase2-code-review, and the p1-fixes, c2, c3 and c4 notes.
Reviewer: final gate, 2026-09-25.

**Counts:** CRITICAL 0 · HIGH 1 · MEDIUM 2 · LOW 8. `[USER]` marks a design fork: MEDIUM-1.

**Probes.** The orchestrator ran these in the working tree at `3b82bd70`, which is `cd1fe146`
plus the simplify commit. That commit changes no import, field, schema or serialization.
Output is quoted from the orchestrator's verbatim relay.

- `scratchpad/final_review_probe.py`, exit 0:
  - `angular_bins {'default': 360, 'minimum': 8, 'maximum': 3600}`,
    `trim_proportion {'default': 0.2, 'minimum': 0.0, 'exclusiveMaximum': 0.5}` and
    `plateau_tolerance {'default': 0.01, 'exclusiveMinimum': 0.0, 'exclusiveMaximum': 1.0}`.
    All three carry docstring-derived descriptions.
  - `op roundtrip equal: True`, `pipeline roundtrip equal: True`, and `legacy-shaped load ok: True`.
  - The GUI registry lists all three params with defaults and descriptions:
    `form renders; mentions all three fields: True`.
  - `fresh import phenotypic loads: []`.
  - `fresh import phenotypic.measure loads: ['scipy.ndimage', 'scipy.spatial', 'scipy.stats', 'skimage.measure']`.
  The same probe on main (`gate-worktrees/size-main`, HEAD `81d19ec6`) printed the identical
  list: `['scipy.ndimage', 'scipy.spatial', 'scipy.stats', 'skimage.measure']`. The branch adds no
  import cost.
- `tests/unit/ci/test_startup_imports.py` and `test_deferred_imports.py`: `236 passed`.

So the public API surface (focus item 3) is clean: `model_json_schema`, the JSON round trip and
the builder param form all check out. The only related text issue is the description wording
in LOW-2.

---

## HIGH

### HIGH-1: Spec §9's golden recapture, differential re-run and end-of-implementation checks are not on the branch (plan Task 10 is still open)

**Where:** `tests/migration/_goldens/measure.MeasureShape.parquet`,
`measure.MeasureSize.parquet`, `measure.MeasureIntensity.parquet` and
`refine.KeepSectionLargest.parquet`. None of the four is in `git diff --stat 81d19ec66..cd1fe146`.
The plan's Task 10 (`plans/.../plan.md:2051-2124`) has every box unchecked.

**Evidence:**
- **Goldens.** `measure.MeasureShape.parquet` at `cd1fe146` still contains the literal
  `Shape_Area`: `git grep` reports "Binary file ... measure.MeasureShape.parquet matches". Spec §9,
  and the user's decision recorded in §10 ("then recapture all four"), require the recapture.
  `tests/migration` is outside `testpaths` (`pyproject.toml:219`), so CI will not flag this.
  The goldens are simply wrong for 0.20.0 from the day of the merge.
- **Differential proof.** The only recorded run of `diff_migration_scenarios.py` (D3, commit
  message of `1decc582`) was taken at `1decc582`, which is **before** the SHAPE flip
  (`34c87594`) and the all-outlines change (`b9f9c261`). At that commit `MeasureShape` still emitted
  `Shape_MeanRadius`/`Shape_MedianRadius`/`Shape_MaxRadius` under their old names. So the
  script's `RENAMES`/`EDT_SUCCESSORS` path (`diff_migration_scenarios.py:84-100`), which checks
  exactly the per-object EDT successors, passed by "column exists unchanged". It has never
  run against the post-flip columns. The unit test
  `tests/unit/measure/test_size_consolidation_equivalence.py:88-110` covers the synth plate
  (96 objects) at unit level. The differential covers the 552-object migration inputs, and has not.
- **Not evidenced at `cd1fe146`:** a docs build after the p1/p2 desc and note changes (the
  recorded build is of `3c91e880`, p1-fixes-notes:78); `mypy`; `ruff` over the changed paths; the full
  sharded regression. Nothing on the branch records them, so they may simply be pending.

**Fix:** execute Task 10 at the final tip, in this order:
1. Run `diff_migration_scenarios.py` against main and the tip.
2. Recapture the four goldens and commit them with the per-drift commit message A6 requires.
3. Run the logic-validation script, mypy and ruff, the docs build (then read the SIZE/SHAPE,
   MeasureSize/MeasureShape and measurements-reference HTML), and the full regression.

See MEDIUM-2 for a contradiction in the Task 10 text that will trip whoever executes it.

---

## MEDIUM

### MEDIUM-1 [USER]: Resuming a pre-0.20.0 run after upgrading silently reuses old-column stores, and the output mixes `Shape_Area` and `Size_Area` rows

**Where:** `src/phenotypic/_cli/_cli_failure_tracker.py:347-381` (`work_id_for_image`),
`:211-272` (`processing_configuration_digest_from_values`), and
`src/phenotypic/_cli/_cli_completion.py:288-355` (`valid_image_success`).

**Evidence (code reading; not probed):**
- The continuation identity is `pipeline_fingerprint = file_sha256(config.pipeline_json)`
  (`:367`), taken over the user's pipeline file bytes. The SLURM path copies those same bytes into
  `worklists/<gen>/pipeline.json` (`_cli_execution_strategies.py:960-969`).
- The configuration digest carries no library version and no measurement-semantics revision.
  The only semantics revision, `PROCESS_LAYER_SEMANTICS_REVISION`, is scoped to `--mode process`
  (`:239-262`).
- `valid_image_success` checks the record's `work_id` and that its artifacts exist. It does not
  look at which columns the store's embedded table carries.
- So consider a user who starts `python -m phenotypic` under 0.19.x, is interrupted, upgrades,
  and re-runs the same command (the documented recovery: "Run the same command again after an
  interruption"). Every image already finished is reused with `Shape_Area`/`Shape_MaxRadius`/…,
  and every new image gets `Size_*`.
- Finalization projects each store onto its own recorded columns (root CLAUDE.md, P7 Task 4). So
  `measurements.csv` then holds both names, each NaN on the other half of the images, inside one
  output folder that looks like a single consistent run.
- With a `MeasureShape`-only pipeline it is worse: the new half has **no** area column at all.

Spec §6 says mixing old and new stores is "documented, not handled". The only user-facing text
is "stores written by earlier versions keep their old column names" (`schema/_change_notes.py:35-36`).
That sentence does not warn that a resumed run is one of those mixtures.

**Fix (choose one):**
- (a) Recommended, text only: add one sentence to `SIZE_SHAPE_SPLIT_NOTE` and to the PR
  description: "A run started before 0.20.0 must be re-run with `--overwrite`, not resumed.
  Resuming reuses the images it already finished, with their old column names."
- (b) Fold a measurement-schema revision into the base digest payload. That cold-starts every
  in-flight `full`/`measure` continuation, which the payload comments (`:241-245`) deliberately
  avoided for process-only changes. Here it would buy correctness, not just caution.
- (c) Accept it as covered by §6 and do nothing.

This is a design fork because (b) changes continuation semantics for every in-flight run, and
(a) changes the §7 note text.

### MEDIUM-2: Plan Task 10 contradicts itself on which goldens to recapture, and its stop rule would halt a correct run

**Where:** `plans/2026-09-24-size-measures-consolidation/plan.md:2058-2079` and `:2108-2118`.

**Evidence:**
- Step 1 says to set `wanted` to **four** scenarios (A6, the user's decision). Its next paragraph
  says "Expected: exactly the two parquet files are modified. **`measure.MeasureIntensity.parquet`
  must not appear**; if it does, stop, because the decoupling changed values."
- Step 6 then `git add`s two files, and its commit message says "Recaptured only these two;
  MeasureIntensity is unchanged".
- Spec §9 and A6 record that the MeasureIntensity golden was **already red on main** (float32/float64
  drift plus the `.volume` ConvexDensity cause). So a correct four-golden recapture *will* modify
  `measure.MeasureIntensity.parquet`. An executor who follows the stop rule halts on a correct
  run. One who follows Step 6 commits two of the four and leaves the other two stale.

**Fix:** rewrite Step 1's "Expected" to "exactly the four files", and drop the MeasureIntensity
stop rule. The differential in Step 0 is what proves that Intensity values are unchanged. Rewrite
Step 6's `git add` and commit message for four files, naming the pre-existing drifts separately
(A6).

---

## LOW

- **LOW-1: The stale "one center" claim survives outside the desc that p1 LOW-2 fixed.**
  `src/phenotypic/measure/_measure_size.py:26-28` says "a family of five radii that are all
  measured from one center inside the colony (inscribed, median, …)". `schema/_size.py:12-13`
  says "a family of radii measured from one center inside the colony". Spec §4.1 says InscribedRadius
  "is the exact distance from the centre to the nearest edge".
  - After A3 and p1 LOW-2, InscribedRadius is the EDT maximum at the deepest pixel, not a
    distance from the plateau centroid.
  - For a ring colony the centroid lies in the hole, so "inside the colony" is not always true
    either. The corrected `INSCRIBED_RADIUS` desc (`_size.py:66-74`) says so; the class and
    enum docstrings do not.
  - The change note's "all measured from one center inside the colony"
    (`_change_notes.py:53-54`) covers only the four signature radii, which is right apart from
    "inside".

  Fix: "…axis lengths, the inscribed radius, and four radii measured from one centre (the
  distance-transform peak)". Mirror the correction into spec §4.1.

- **LOW-2: The `angular_bins` field description misstates what more bins do.**
  `_measure_size.py:39-42`: "More bins resolve narrower protrusions; 360 … resolves any runner
  wider than about 1/57 of the colony radius." This text is the field's `model_json_schema()`
  description and its GUI tooltip.
  - A runner narrower than one bin is never missed. Its tip vertex falls in some bin, and max-per-bin
    keeps it (`:146-148`), so MaxRadius always sees it.
  - What the bin count changes is the runner's **weight**: a sub-bin runner occupies a whole bin
    (1/K of the signature), so it overweights Mean/Median/RobustMean. The runner's angular width is
    set by its distance from the centre, not by the colony radius.
  - Past about 8R bins, a small colony's outline also leaves bins empty, and they are
    interpolated.

  Fix: "Number of equal angular directions sampled. A protrusion narrower than one bin still
  fills a whole bin, so more bins weight a thin runner closer to its true angular width; on a
  small colony (about 8R outline vertices) extra bins are interpolated."

- **LOW-3: `SHAPE.SOLIDITY`'s desc still implies Solidity ≤ 1.** At `schema/_shape.py:48-51`,
  this desc is published into every run's README. Spec §2 item 3 says Solidity can slightly exceed 1
  "stated so nobody 'fixes' it later". The explanation page (`measurement_metrics_biological_meaning.md:42`)
  and the reference-free scorer (`tune/score/_reference_free_scorer.py:63`) already say so; the
  column's own description does not. Fix: append "Can slightly exceed 1, because the hull passes
  through pixel centres."

- **LOW-4: `MEDIAN_BOUNDARY_DIST` points to InscribedRadius for "radial extent".**
  `schema/_shape.py:80-81`: "See Size_InscribedRadius and Size_RobustMeanRadius for the colony's
  radial extent." The InscribedRadius desc (`_size.py:69-72`) says explicitly that it is *not*
  the overall extent. Fix: "See Size_RobustMeanRadius and Size_MaxRadius for the colony's radial
  extent."

- **LOW-5: The change note does not say that the EDT successors differ for touching and
  border colonies.** The rename table in `schema/_change_notes.py:38-50` pairs `Shape_MaxRadius →
  Size_InscribedRadius` and `Shape_Mean/MedianRadius → Shape_*BoundaryDist`. It then says "Compare old data
  against the successor". Spec §3.1/§4.1 and the descs say these values changed on purpose for
  colonies touching another label or the image border (for example, a full-width 10-row band at the
  top edge reports 5, where it reported 10 before). A user following the note's own instruction
  will see unexplained differences on exactly those colonies. Fix: add "(identical except for
  colonies touching another colony or the image border, which now measure to that edge)" after
  "the successor". This touches §7 text; it is a factual completion, not a fork.

- **LOW-6: Spec and RESUME text are stale against the shipped branch.**
  - `design.md:3` still says "Status: … awaiting spec review".
  - `design.md` §7 and §9 still say to rename `test_version_is_0_19_0` in
    `test_norm_migration.py`. The branch instead deleted it there and pinned the version in
    `tests/unit/schema/test_change_note.py:16-17`. That is equivalent, but the spec says otherwise.
  - `plans/.../RESUME.md:4-6` says "No `src/` code has been changed yet … starting with Task 1".
  - `RESUME.md:21-23` repeats the "all measured from one centre" claim from LOW-1.

  Fix: update the status line and §7/§9. Either delete RESUME.md (a session hand-off note;
  no other plan folder on main has one) or mark it historical at the top.

- **LOW-7: The committed helper scripts hard-code this worktree's path.**
  `plans/.../submit_phase_gate.sh:15` sets
  `REPO=/bigdata/exfab/anguy344/PhenoTypic/.claude/worktrees/size-measures-consolidation`. The
  worktree disappears after merge, so the script is dead on `main`. Other committed sbatch
  files use `/bigdata/exfab/anguy344/...` log paths, so the user-specific prefix has precedent;
  a branch-worktree path does not.

  The p2 HIGH-1 advice to note the `\b`-suffix blind spot in the `size_rename.pl` header
  (`size_rename.pl:1-7`) was not applied either.

  Fix: derive `REPO` from `git rev-parse --show-toplevel`, and add the one-line caveat to
  `size_rename.pl`.

- **LOW-8: The history is not bisect-clean.** `cb0a52ca` leaves three `test_linear_softplus.py`
  tests red, and they are fixed only in `68e20230` (p2 HIGH-1). Between `34c87594` (the SHAPE flip) and
  `4a531dcc`/`cb0a52ca`, the bundled CSVs, GUI captures and roughly 57 test files still spell the
  retired names. Plan A5 moved prefabs and GUI defaults ahead of the flip, but not the tests. Eight
  of the 23 commits contain only spec, plan, RESUME or report documents.

  Fix: squash-merge, or at least squash `cb0a52ca`+`68e20230`. Say in the PR description that
  intermediate commits are not individually green.

---

## Verified (no finding)

- **§3.1 columns.** SIZE (`schema/_size.py:27-119`) has exactly the 12 members, in spec
  order. `_operate` (`_measure_size.py:198-224`) fills each from the source §3.1 names:
  - Area and IntegratedIntensity: `_calculate_sum`;
  - Perimeter, BboxArea and the axes: `props.*`;
  - ConvexArea: `convex_hull_area(props.coords)[1]`, which is `hull.volume` or NaN on `QhullError`
    (`_object_geometry.py:22-41`);
  - InscribedRadius: `object_edt(...).max()`;
  - the four signature radii: median, mean, `trim_mean` and max of one signature
    (`:188-195`).

  The moved members carry no `tier=`.
- **§3.2.** SHAPE keeps the eight form descriptors plus the two BoundaryDist members at
  class tier 2 (`_shape.py:26-83`). Feret keeps `tier=1`. `MeasureShape._operate`
  (`_measure_shape.py:112-149`) emits no §3.1 column. Its NaN-initialised frame also
  fixes main's `np.zeros` default for the Qhull-failure Feret path.
- **§4.1 mechanics.**
  - The plateau is 8-connected, at `structure=np.ones((3,3))` (`:125-128`).
  - The centre is the plateau centroid on the unfilled mask's EDT.
  - Contours are all pooled on the hole-filled padded crop with `fully_connected="high"`, and the
    pad is undone by `- 1.0` (`:132-139`).
  - Binning keeps the maximum per bin, and empty bins get circular interpolation (`:145-164`).
  - `binary_fill_holes`' default 4-connected background is the correct dual of 8-connected
    foreground.
- **§4.4 descs.**
  - The InscribedRadius caveat is present, with the 100×20 → 10 example.
  - Median/Mean/Max each carry the "not the retired … column" clarification.
  - RobustMean covers the compact body and elongation.
  - SIZE.AREA carries `bio_desc` and `image="shape/area.png"` verbatim from main's
    `SHAPE.AREA`.
  - No new member authors `bio_desc`.
- **§5.**
  - A grep over `src/` finds no measurer running another measurer. The only
    `Measure*().measure(` hits are `MeasureBounds` inside image accessors (`_grid_image_handler.py:421`,
    `_objects_accessor.py:710`), which are not measurers.
  - `MeasureIntensity` uses `props.area` and `convex_hull_area`.
  - `KeepSectionLargest` uses `np.bincount` on the uint16 objmap (`_image_data_manager.py:138`),
    keeping main's merge order.
  - The `Size_Area == props.area` invariant is pinned by `test_measure_size.py:42`.
- **§6 and §7.**
  - The rename table and the model-output trap are in the note.
  - The note renders from one hook on three surfaces: `append_rst_to_doc` (`_measurement_info.py:610-613`),
    the enum docstrings via `append_change_note`, and `measurements_ref._class_section`.
  - No release note leaked into a `desc`.
  - `__version__ = "0.20.0"`.
- **§8.**
  - Every prefab that measures shape also measures size, pinned over `prefab.__all__` by
    `tests/unit/prefab/test_prefab_measures_size.py`.
  - The five GUI defaults now spell `str(SIZE.AREA)`.
  - `schema/CLAUDE.md:79-84` is updated.
  - A repo-wide `git grep` at `cd1fe146` for every retired `Shape_*` name and `SHAPE.<moved>`
    member finds only:
    - the intended exclusions (change note, SIZE descs, the toy-SHAPE doctest at
      `_measurement_info.py:278-317`, the equivalence and change-note tests,
      `test_measurement_join_migration_run.py`);
    - the FEATURES.md ledger row;
    - binary goldens (HIGH-1).
  - Every docs and scripts file that configures `MeasureShape` and reads a size column also
    configures `MeasureSize`.
- **Import cost.**
  - `_object_geometry.py` imports `scipy.ndimage`/`scipy.spatial`. `_measure_size.py` adds
    module-level `scipy.stats.trim_mean`, `scipy.ndimage.binary_fill_holes` and
    `skimage.measure.find_contours`.
  - All three families were already imported at module level by `phenotypic.measure` on main:
    `_measure_shape.py` imported `scipy.spatial`/`ndimage`, `_measure_symzones.py:26` imported
    `skimage.measure.find_contours`, and `scipy.stats` came in via `detect/_sine_peak_detector.py:15`.
  - `refine/_keep_section_largest.py` **drops** its module-level `from phenotypic.measure import
    MeasureSize`, so refine's import graph shrinks.
  - `schema/_change_notes.py` imports only stdlib `inspect`.
- **Cross-platform.** There are no path, `rawpy` or OS-specific constructs. `np.bincount` on
  uint16 and `np.maximum.at` behave the same on Windows.
- **Diff hygiene.** There is no debug file, and no scratch or `/tmp` path under `src/` or `tests/`.
  The `.testmondata` blob predates the branch. The `_assets/measurements/shape/area.png` change is
  the regenerated `Size_Area` title (`scripts/make_measurement_example_images.py:50`).
- **The differential script's isolation argument** (`diff_migration_scenarios.py:33-41`)
  is sound. The nearest non-object pixel is 8-adjacent to the object, so for an isolated,
  non-border object it is background, inside the padded crop, and equal to main's whole-image
  EDT target.
