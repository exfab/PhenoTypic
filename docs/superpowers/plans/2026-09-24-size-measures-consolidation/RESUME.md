# RESUME — size measures consolidation

**Branch:** `claude/size-measures-consolidation` (cut from `main` at `81d19ec66`)
**State (2026-09-24):** the spec and the implementation plan are written, committed and pushed.
**No `src/` code has been changed yet.** The next step is to execute the plan, starting
with Task 1.

## Read these, in this order

1. `docs/superpowers/specs/2026-09-24-size-measures-consolidation/design.md`: the spec
   (what we're building and why; §10 is the decisions log).
2. `docs/superpowers/plans/2026-09-24-size-measures-consolidation/plan.md`: the 10-task
   plan, with full code, tests, mutation proofs and commands.
3. `docs/superpowers/logic_validation_scripts/2026-09-24-size-measures-consolidation/radial_invariants.py`:
   re-derives every geometric number the spec and plan quote. Run it with
   `uv run --no-project --with numpy --with scipy python <path>`; it should report 0 failures.

## What this change is, in one paragraph

`MeasureSize` becomes the single emitter of colony size: Area, Perimeter, ConvexArea,
BboxArea, Major/MinorAxisLength, and a radius family of InscribedRadius, MedianRadius,
MeanRadius, RobustMeanRadius and MaxRadius, all measured from one centre inside the
colony. `MeasureShape` keeps the form descriptors (Circularity, Compactness, Solidity,
Extent, Eccentricity, Orientation, Min/MaxFeretDiameter) plus `MeanBoundaryDist` and
`MedianBoundaryDist` (the old, misnamed Mean/MedianRadius values). This is a hard break
with no aliases, released as a minor bump to **0.20.0**, with a highlighted
`.. versionchanged:: 0.20.0` note rendered from one `MeasurementInfo.change_note()` hook.

## Decisions the user made (do not re-litigate)

| Question | Decision |
|---|---|
| "One source" | The size columns are **removed** from Shape |
| What moves | Radii, perimeter, hull and box areas, ellipse axes. **Feret stays in Shape** |
| Unmerged branch `shape-radial-measures` | **Port** its radial-signature work into SIZE. It is a read-only source (`git show shape-radial-measures:<path>`); **never merge, rebase or cherry-pick it** (1,321 commits stale) |
| Convex area | **`ConvexHull.volume`** wherever scipy supplies it. The branch's `regionprops.area_convex` swap is **not** ported |
| Back-compat | **Hard break**, no alias machinery |
| Radius names | `InscribedRadius` (name kept; its desc must carry the elongated-colony caveat), `MedianRadius`, `MeanRadius` (plain mean), `RobustMeanRadius` (20% trimmed), `MaxRadius` (the branch's ReachRadius) |
| Old Mean/Median "radius" (edge-distance stats) | Stay in **Shape**, renamed `MeanBoundaryDist` and `MedianBoundaryDist` |
| Release | **Minor bump to 0.20.0**; highlighted note in the class docs **and** the measurement docs |

## Still open, and needed before implementing

- **Execution method** has not been chosen. The recommendation to the user was
  **subagent-driven** (`superpowers:subagent-driven-development`), per the user's global
  CLAUDE.md orchestration rule and because each task depends on the previous one's
  baseline. The alternative is native (`superpowers:executing-plans`). **Ask the user
  before starting.**
- The user has not yet explicitly confirmed one change added while writing the plan: the
  per-object EDT now counts the image border as an edge (a 10-row band on the top edge
  reports InscribedRadius 5; it was 10). This is recorded in spec §4.1 and pinned by a
  Task 3 test. Mention it when asking about the execution method.

## Traps that will bite a fresh session

- **Task 1 must run on an unmodified `src/`.** It captures main's Shape and Intensity
  output as `tests/unit/measure/_golden/size_consolidation_baseline.parquet`. Every
  equivalence test in Tasks 3–5 compares against it.
- **Never run `scripts/capture_migration_goldens.py`.** It rewrites all 142 goldens plus
  the frozen inputs. Task 10 gives a snippet that recaptures exactly
  `measure.MeasureShape` and `measure.MeasureSize`. `measure.MeasureIntensity.parquet`
  must stay byte-unchanged.
- **Same-name trap:** the new `Size_MedianRadius`, `Size_MeanRadius` and `Size_MaxRadius`
  do **not** hold the values of the retired `Shape_MedianRadius`, `Shape_MeanRadius` and
  `Shape_MaxRadius`. The successors are `Shape_MedianBoundaryDist`,
  `Shape_MeanBoundaryDist` and `Size_InscribedRadius`. The rename script in plan Task 7
  encodes this; do not "simplify" it to a prefix swap.
- **Rename-script exclusions** (plan Task 7): never run it on `schema/_change_notes.py`,
  `_size.py`, `_shape.py`, the equivalence and shape tests, `test_change_note.py`, the
  baseline parquet, `docs/superpowers/**` or `_golden*` fixtures. They spell retired names
  on purpose.
- **`bio_desc`:** never author it. The only allowed move is `SHAPE.AREA`'s existing
  `bio_desc` and `image` going verbatim onto `SIZE.AREA`.
- **Change notes never go into `Entry.desc`.** Descs are published into every run's
  README.
- Use `uv run` for everything, and pass explicit paths to `ruff check --fix`. Use the
  `run-phenotypic-test` skill for anything beyond a focused file run; the full suite is a
  Slurm job. GUI tests need `QT_QPA_PLATFORM=offscreen`.

## Facts established this session (verified, with evidence)

- On main, `Shape_MeanRadius` and `Shape_MedianRadius` are EDT mean and median: 0.335R and
  0.294R on a disk of radius 100 (script check 01).
- Main's `MeasureShape` runs the EDT over the whole labelled objmap, so touching colonies
  report 20/21 instead of 10/11 (script check 02).
- On `load_synth_yeast_plate()` (96 colonies), the per-object EDT maximum equals main's
  whole-image EDT maximum for **all 96**: no colony touches another or the border. This
  is why the Task 3–5 equivalence tests can demand a 1e-10 match.
- For an ideal 100×20 rectangle, the radius family is 10.0 / 14.1 / 21.0 / 16.2 / 50.9
  (script check 04).
- The consumer map (blast radius) is summarised in spec §8. The main items:
  - the 7 prefabs lack `MeasureSize`;
  - `MeasureIntensity` runs `MeasureShape` internally;
  - the GUI analysis tab has 5 `on="Shape_Area"` defaults;
  - the bundled `data/meas/*.csv` files use Shape headers;
  - about 60 test files use `Shape_Area` as sample data.
- `tests/unit/sdk_/test_norm_migration.py::test_version_is_0_19_0` pins the version. Plan
  Task 6 deletes it and adds `test_version_is_0_20_0` in `tests/unit/schema/test_change_note.py`.

## Commits on this branch

- `e906f4be7` docs(spec): size measures consolidation
- `92b3c9da3` docs(plan): implementation plan (plus a spec border-edge note and script check 06)
- this RESUME
