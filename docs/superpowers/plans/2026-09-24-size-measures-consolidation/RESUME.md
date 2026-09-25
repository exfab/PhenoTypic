# RESUME — size measures consolidation

**Branch:** `claude/size-measures-consolidation` (cut from `main` at `81d19ec66`)
**State (2026-09-25):** plan Tasks 1-9 are implemented (phases 1 and 2), both phase
reviews' fixes have landed, the simplify pass is done (`3b82bd70`), and the final
whole-branch review's fixes (`reports/.../final-review.md` MEDIUM-1/2, LOW-1 to LOW-7) are
applied on top of it. **Plan Task 10 is DONE (2026-09-25):**
- **Differential, tip `ad890516` vs main:** PASS, 27 checks.
- **Goldens:** four recaptured in `13821bba`; all four scenarios pass.
- **radial_invariants.py:** 0 failures.
- **Docs build (Slurm):** the note renders on all 5 surfaces, and the enum pages no longer
  show the blockquote.
- **mypy:** 429 errors at the tip vs 440 on main.
- **Full sharded regression at `13821bba`, compared with main by name:**
  - 13,989 tests, 0 regressions.
  - 84 failures, all pre-existing: migration goldens for other operations, tune, and
    FilFinder smoke.
  - 3 newly passing.
- **e2e:** scatter + analysis 10/11. The one setup error, a live-server start timeout,
  passes 3 of 3 on rerun at both the tip and main.

What remains is opening the PR. LOW-8 (history not bisect-clean) is handled in the PR
description, not by rewriting history: squash-merge.

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
BboxArea, Major/MinorAxisLength, InscribedRadius (the EDT maximum), and four radii,
MedianRadius, MeanRadius, RobustMeanRadius and MaxRadius, measured from one centre (the
centroid of the distance-transform peak plateau). `MeasureShape` keeps the form descriptors (Circularity, Compactness, Solidity,
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

## Update 2026-09-24 (second session): execution started

- Execution method is **orchestrated clusters** (`execute-plan-orchestration`), and the user
  accepted the border-edge change. The cluster table is in plan.md, under "Execution clusters".
- Task 1 is done: `2004cfac` (baseline parquet, 96 x 30, captured from unmodified `src/`).
- Port source `shape-radial-measures` is now on origin at `5cad1dfa5`. Fetch it and read it by SHA.
- The pre-dispatch plan review is in
  `docs/superpowers/reports/2026-09-24-size-measures-consolidation/plan-review.md`. Its
  binding amendments A1-A10 are at the top of plan.md. The user's decisions are recorded in
  spec §10: zero objects keep the raise; the differential proof comes first, then all four
  goldens are recaptured; both §4.1 rewordings are accepted.
- Pre-change migration-golden status on unmodified `src/` is recorded in plan amendment A6.

The section below is kept for history; its questions are now answered.

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
  the frozen inputs. Task 10 gives a snippet that recaptures exactly four goldens:
  `measure.MeasureShape`, `measure.MeasureSize`, `measure.MeasureIntensity` and
  `refine.KeepSectionLargest` (A6). Intensity and KeepSectionLargest were already red on
  main, so they do change; the differential run first is what proves this branch changed
  no Intensity value and no KeepSectionLargest selection.
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

From `git log --oneline 81d19ec66..3b82bd70`, oldest first:

- `e906f4be` docs(spec): size measures consolidation — MeasureSize as the single source of colony size
- `92b3c9da` docs(plan): implementation plan for the size measures consolidation
- `2f13b9ee` docs(plan): RESUME for the size measures consolidation hand-off
- `2004cfac` test(measure): capture pre-consolidation Shape/Intensity baseline on the synth plate (Task 1)
- `64e81791` docs(plan): apply the pre-dispatch plan review and the user's decisions
- `4d51d6b7` feat(measure): shared convex-hull-area and per-object EDT helpers (Task 2)
- `e3acb345` feat(measure): MeasureSize emits size magnitudes and a five-member radius family (Task 3)
- `b6f14301` refactor: MeasureIntensity and KeepSectionLargest read regionprops instead of running measurers (Task 4)
- `1decc582` refactor: point prefabs and GUI analysis defaults at MeasureSize before the Shape flip (Task 4b)
- `34c87594` feat(measure)!: MeasureShape emits form descriptors only; size magnitudes live in MeasureSize (Task 5)
- `99adf692` feat(schema): highlighted 0.20.0 change note for the size/shape split; bump to 0.20.0 (Task 6)
- `3c91e880` docs(plan): Slurm phase-gate submitter and docs-build job for the size consolidation
- `b9f9c261` fix(measure): the radial signature samples every outline of the label, 8-connected (phase-1 review HIGH-1)
- `d4d6bcfa` fix: phase-1 review follow-ups (hole tests, enum docstring dedent, desc and test tidy)
- `d34d8c01` test(measure): orientation-zone golden compares serialization without the running version stamp
- `c5bd2894` docs(plan): submit_phase_gate.sh takes a PARTITION override
- `01f76c4e` docs(plan): commit the size rename script (plan amendment A7)
- `3b70d667` docs: document MeasureSize as the source of colony size; highlight the 0.20.0 rename (Task 9)
- `4a531dcc` refactor: point bundled data, scripts and docstrings at the Size columns (Task 7)
- `cb0a52ca` test: sweep retired Shape_* column names to their Size/Shape successors (Task 8)
- `68e20230` test(analysis): sweep the suffixed Shape_Area_stderr/_std_pool fixture names (Task 8)
- `c8caf935` fix: phase-2 review follow-ups (bundled hull columns, BoundaryDist wording, model-header trap)
- `cd1fe146` docs(report): phase-2 code review of the consumer migration
- `3b82bd70` refactor: simplify pass over the size consolidation (no behaviour change)

After these: the final-review report and its fixes, then Task 10.
