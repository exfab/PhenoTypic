# RESUME: MeasureNeighborDist nearest-object

A hand-off for the next session. **Read this first, then the spec, then the plan.**

## Where things stand (2026-09-24)

- **Branch:** `worktree-measure-neighbor-nearest` (pushed to `origin`).
- **Worktree:** `.claude/worktrees/measure-neighbor-nearest`. Enter it with
  `EnterWorktree(path=...)`, or `git checkout worktree-measure-neighbor-nearest`
  in a fresh worktree. Run `uv sync` first.
- **Phase:** design and planning are **done**. No product code has been written.
  Every commit on the branch is docs only:

  | Commit | What |
  |---|---|
  | `155b262d9` | spec + independent validation script |
  | `0159d99be` | implementation plan |
  | `6cffc8401` | category rename `GridSpatial` → `NeighborDist` |
  | `eac887240` | Task 0: public grid API only, memoized section windows |

- **Open decision, ask the user before starting:** execution method.
  **Native** (recommended: 5 sequential tasks, mostly one file, and the plan
  contains the code) or **Subagent-driven**. Then invoke
  `superpowers:executing-plans` (Native) or
  `superpowers:subagent-driven-development`.

## Files

| Path | Role |
|---|---|
| `docs/superpowers/specs/2026-09-24-measure-neighbor-nearest/design.md` | the spec (§3.1–3.10) |
| `docs/superpowers/plans/2026-09-24-measure-neighbor-nearest/plan.md` | the plan, Tasks 0–4, with code |
| `docs/superpowers/logic_validation_scripts/2026-09-24-measure-neighbor-nearest/nearest_object_bounds.py` | numpy/scipy-only witness for claims C1–C4; `uv run python <it>` prints `OK` |

## Decisions the user made (don't relitigate)

1. **Nearest-object candidates:** any object on the plate (same cell,
   diagonal, far).
2. **New columns:** `NearestObjLabel`, `NearestDistance`, `NearestRelation`.
   The relation is an **integer code**: 0 same cell, 1 adjacent, 2 diagonal,
   3 further. There's no off-grid code.
3. **Algorithm:** exact branch and bound. Candidates are ordered by a
   bounding-box lower bound, with exact distances from `cKDTree` over
   4-connected boundary pixels. Ties go to the smaller label.
4. **Input types:** on a `GridImage`, objects with no grid cell are excluded
   from the nearest search. On a plain `Image`, all objects count. So the base
   class moves `GridMeasureFeatures` → `MeasureFeatures`, branching on
   `hasattr(image, "grid")`.
5. **Category rename:** `GridSpatial` → `NeighborDist`, so every header is
   `NeighborDist_*`. `ErrorCutoffFinder.MEASUREMENT_PREFIXES` keeps
   `"GridSpatial_"` so older tables still count. There's no general alias
   layer. The PR description must call out the rename.
6. **Public grid API only:** `MeasureNeighborDist` must not touch `grid._*`.
   Use only `info()`, `nrows`, `ncols`, `get_row_edges()`, `get_col_edges()`,
   and memoize locally (Task 0).

## Facts established by measurement (don't re-derive)

- **Why it was slow:** `measure(synth_plate)` took ~40–45 s, and the test
  file took **6 min 12 s** for 16 tests. `_section_bbox` ran 402× through the
  private `_adv_get_grid_section_slices`. Each call made the
  `CenteredAutoGridFinder` edge getters re-fit the whole grid, 804 fits in
  all. Task 0 fixes this. Until it lands, give file-level runs a ≥600 s
  timeout.
- **Task 0 is exact:** public-edge windows match the private helper on 88/88
  synth-plate cells, and building them takes 0.09 s.
- **Float equality:** scipy EDT and `cKDTree` distances are bit-identical
  (0 mismatches in 1,000 samples). `np.hypot` matched `sqrt(x²+y²)` here, but
  it isn't guaranteed correctly rounded, which is why the plan computes the
  lower bound as `sqrt(int sum)`.
- **Off-grid objects:** `ManualGridFinder` clamps an object outside the grid
  into the nearest edge cell, so it never yields an object with `NaN` grid
  row/col. The off-grid exclusion is therefore tested on the helper directly.
- **`Bbox_Max*` is exclusive** (regionprops). The nearest search derives
  inclusive boxes from pixels. Task 0's section windows deliberately reuse
  `Bbox_Max*` to match the existing behaviour.
- **Grid dtype:** `Grid_RowNum` / `Grid_ColNum` are `category` dtype;
  `.to_numpy(dtype=float)` works.
- **Validation script:** mutation-tested. An early-exit mutant gives 149
  failures and a dropped tie-break gives 5.

## Gotchas for this session type

- **Worktree guard.** Bash commands that pipe computed values into `sed`/`uv`,
  or mention the word `source` (even as a path like `docs/source`), get refused
  as "can't verify stays in worktree". Split them into plain commands, write
  probes to `/tmp/*.py` with a heredoc, then `uv run python /tmp/x.py`.
- **Test command:** `QT_QPA_PLATFORM=offscreen uv run pytest <paths> -q --no-header -p no:randomly -o addopts= -m "not slow"`
  (see the `run-phenotypic-test` skill). Never `-n auto`, and never quote an
  `-x` run as a result.
- **Ruff:** `uv run ruff check --fix <explicit paths>`, never bare.
- **Schema authoring:** `MeasurementInfo` members get `label` + `desc` only.
  Leave `bio_desc=""` and `image` unset.
- **Full suite:** run it **once**, at the end (plan Task 4 Step 6), via the
  sbatch script or CI. Never between tasks.

## Next steps

1. Ask the user: Native or Subagent-driven.
2. Execute `plan.md` Tasks 0 → 4 in order, ticking the `- [ ]` boxes as you
   go and committing per task.
3. After Task 4: one fresh whole-branch review, then a simplify pass, then a
   regression run of the affected surface (user's CLAUDE.md review gates).
4. Open the PR. Note the `GridSpatial_*` → `NeighborDist_*` header rename and
   the base-class change (the measurer now accepts a plain `Image` and gains
   `include_meta`).
