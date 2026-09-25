# Phase 2 review — texture-column-naming (commits 3d9ade1f9, 8e80fcfd3)

Reviewer: implementation-test-reviewer. Scope: `git diff d8a078eb1..HEAD -- src tests`.
D8′ and D9 are treated as settled and not relitigated.
Stopped early at the coordinator's turn limit. Checks I did not finish are marked
**UNVERIFIED**. I mutated no code, and `git status` is clean.

## Summary

- CRITICAL: 0
- IMPORTANT: 2
- MINOR: 4
- UNVERIFIED: 3 areas (listed at the end)

The merge rewrite is correct for every measurer shipped in `phenotypic.measure`.
The only columns measurers share with image-info are `Bbox_*` and `Grid_*`, and
both copies come from the same code, so they are bit-identical:
`ObjectsAccessor.info` calls `MeasureBounds` (`_objects_accessor.py:710`), and
`MeasureGridLinRegStats` reads `image.grid.info()` (`_measure_grid_linreg_stats.py:66,68`).
I enumerated every concrete `MeasureFeatures` in `phenotypic.measure` on the
detected synth plate. The only sharers are `MeasureBounds` (the 10 `Bbox_*` columns)
and `MeasureGridLinRegStats` (the `Bbox_*` and `Grid_*` columns).

The new raise does have one real trigger. A `GridFinder` that runs as a
measurement, whether the user put it in `meas` or the nrows/ncols preset injected
it, can disagree with the image's own `grid_finder`. The old code handled that
disagreement by silently dropping colonies in an inner join. That was worse, but
it did not crash, and one documented CLI combination now fails on every image.

---

## IMPORTANT

### I-1 A `GridFinder` in the run order that disagrees with `image.grid_finder` now raises on every image. One trigger is a documented CLI combination.

- `src/phenotypic/_core/_pipeline_parts/_image_pipeline_core.py:1384-1403` (`_build_measurement_run_order`)
- `src/phenotypic/_core/_pipeline_parts/_image_pipeline_core.py:1586-1599` (new conflict raise)
- `src/phenotypic/_cli/_cli_process_single.py:264-275` together with `_cli_utils.resolve_grid_shape`

**Mechanism.** Image-info (`image.grid.info()`) assigns grid cells with the
**image's** `grid_finder`. Any `GridFinder` in the measurement run order assigns
them with **its own** fit and emits the same `Grid_RowNum/ColNum/RowMajorIdx/ColMajorIdx`
columns. There are two ways a finder gets into the run order:

1. **Preset injection.** `_build_measurement_run_order` prepends
   `CenteredAutoGridFinder(nrows=self._nrows, ncols=self._ncols)` from the
   pipeline preset. It ignores the image's grid shape and finder.
   The CLI's `--nrows/--ncols` help says it "Overrides any pipeline-level preset",
   but `resolve_grid_shape` only passes the override to the image
   (`read_kwargs["nrows"]`). The pipeline's `_nrows` is left alone, so the
   injected finder still uses the preset shape.
2. **An explicit `GridFinder` in `meas`**, e.g. `AutoGridFinder`, while the image
   keeps the default `CenteredAutoGridFinder`
   (`_grid_image_handler.py:92-97`). This is a supported configuration:
   `_build_measurement_run_order` checks for it explicitly, and
   `test_image_pipeline.py:409` covers it.

**Verified by probe** on `load_synth_yeast_plate()` + `OtsuDetector` (552 objects):

| Scenario | Before (`d8a078eb1`) | Now |
|---|---|---|
| `meas={"g": AutoGridFinder(8,12), "size": MeasureSize()}` on a default GridImage | 408 rows: 144 colonies silently dropped by the inner join on grid keys | `ValueError ... ['Grid_ColNum', 'Grid_RowMajorIdx', 'Grid_ColMajorIdx']` |
| `ImagePipeline(meas=[MeasureSize()], nrows=16, ncols=24)` on an 8×12 GridImage (the CLI `--nrows` override case, reversed) | **2 rows** | `ValueError ... ['Grid_RowNum', 'Grid_ColNum', ...]` |
| Preset equal to the image's shape (8×12) | 552 rows | 552 rows (OK) |

**Impact.** Raising is scientifically better than dropping 26–99 % of colonies
without a word, so I am not asking for the raise to be reverted. But:

- (a) A CLI run with `--nrows/--ncols` different from a pipeline preset used to
  produce a thin frame. Now every image fails with a message about "two
  measurements" that names no measurer and no grid finder.
- (b) The root defect is the injected finder: it is redundant when the image is
  a `GridImage`, because image-info already emits `Grid_*` from the image's own
  finder, and `GridFinder.measure` refuses a non-`GridImage`
  (`abc_/_grid_measure.py:262-270`). So the injection can only duplicate or
  contradict. Before, it was harmless-looking data loss. Now it is a hard failure.
- (c) No test covers either scenario.

**Fix.**
1. Pass the image into `_build_measurement_run_order`. Skip the injection when
   `isinstance(image, GridImage)`, or inject `image.grid_finder` itself. Either
   way the preset can no longer contradict the image. Alternatively, have the
   CLI set `pipeline.nrows/ncols` from the resolved shape.
2. Make the conflict message actionable. Thread the measurer keys into the merge
   (for example, merge a list of `(key, frame)` pairs, with `"image_info"` for
   the last one), and say which two producers disagree. When every conflicting
   column is `Grid_*`, add: "a GridFinder in the measurement list disagrees with
   the image's grid_finder".
3. Add regression tests for both rows of the table above. Assert the CLI-override
   case either succeeds with 552 rows (after fix 1) or raises the specific message.

### I-2 The merge error cannot tell the user which measurers collided

- `_image_pipeline_core.py:1594-1598`

In a production CLI failure, the exception text is the only diagnostic. It is
wrapped as a per-image scientific error and names only columns. For the texture
case (a same-scale pair built by nested mutation, see M-2) and for I-1, the user
cannot tell which of N measurers is responsible. `measure()` has the keys
(`meas_to_run`) but does not pass them to `_merge_on_object_labels`.

**Fix:** as in I-1 fix 2, pass producer names and include both in the message.
Also include the count of disagreeing rows and one example label, which is cheap
and makes the message easy to act on.

---

## MINOR

### M-1 The dtype of a surviving shared column depends on measurer order
`_image_pipeline_core.py:1599` keeps the **left** copy. With the preset injection
(first frame), `Grid_RowNum` is `category`, as confirmed by the probe. With
`MeasureGridLinRegStats` listed first it is `int64`. With only image-info
supplying it, it is the ordered categorical. Two pipelines, or two versions of
one pipeline, can therefore write different Parquet dtypes for the same column.

**UNVERIFIED:** whether cross-store aggregation (`master_measurements.parquet`
concatenation, polars mirror) tolerates `category` vs `int64` for `Grid_RowNum`
when stores from before and after the upgrade are mixed. The
`MEASUREMENT_HEADER_REVISION` bump forces re-measurement within one run, which
mitigates this within a run but not across runs combined by hand or in the GUI.

**Fix:** always keep image-info's dtype for info-block columns. For example,
after the merge, cast shared `Grid_*`/`Bbox_*` columns to the dtype of the
image-info frame, which is last in the list. Or pin a canonical dtype and add a
test asserting it for each measurer order.

### M-2 The same-scale pre-check is bypassed by nested attribute mutation
`_refuse_same_scale_texture` (`_image_pipeline_core.py:305-348`) runs only when
`meas` is constructed or assigned. `pipe.meas["t2"].scale = 5` passes
`MeasureTexture`'s own `validate_assignment` but never re-runs the pipeline
validator. I found no such mutation in `src`: rg for `.meas[..] =`,
`_meas[..] =`, `.update` and `.pop` returned nothing, and tune rebuilds only `ops`
(`tune/_evaluation/_builder.py:197-213,384`). So this is a notebook-only hole.
The merge then raises (values differ) or silently keeps one copy (identical
config), which is acceptable.

**Fix:** document the limitation in the validator docstring, or re-run the check
at the top of `measure()`. It is O(n) and cheap.

### M-3 `_columns_agree` puts both copies in the same row, but only because of the inner join
`_image_pipeline_core.py:1577-1591`. This is correct as written: after the merge
both copies sit in the same row. But `_columns_agree`'s docstring says the inputs
are "aligned row-for-row", and that holds only because of the merge. If a
measurer ever returns duplicate `Object_Label` rows, the inner join multiplies
rows and the comparison still passes. Nothing asserts label uniqueness per frame.

**Fix:** `assert df[OBJECT.LABEL].is_unique` per frame, or raise a clear
`ValueError`. This is cheap, and it makes the "one row per object" docstring
claim enforced rather than assumed.

### M-4 Old NaN-bearing shared columns used to leak `*_merged` columns; the change removes them silently
In the old code, a shared column containing any NaN failed `np.all(x == x)` and
survived as `<col>_merged`. The new code dedupes it. That is a fix, but any
stored table or downstream consumer that referenced a `*_merged` column would
now miss it. `rg "_merged"` in consumers is **UNVERIFIED**. Worth a one-line
mention in the CHANGELOG.

---

## Revision (item c)

- `MEASUREMENT_HEADER_REVISION` is in the base payload
  (`_cli_failure_tracker.py:227,257`), and `per_image_config_digest` is aliased to
  the same function (`_cli_identity.py:90`). This matches the CLAUDE.md paragraph.
- Migrate derives generations with `per_image_config=None`
  (`_cli_migrate.py:714-716`, `_cli_migrate_state.py:516-520`), so migrate is
  unaffected by the bump. Good.
- The only equality check on the generation that I found is
  `_local_epoch_ownership` (`_cli_execution_strategies.py:84-104`), which raises
  `"Local lifecycle epoch is stale"`. **UNVERIFIED:** whether a local run
  resumed across the upgrade rewrites the processing state with the new
  generation before this fence is reached. If it does not, the user sees a
  cryptic "epoch is stale" error instead of a cold start. I believe this is the
  same path a pipeline edit takes (inherited behaviour, as the CLAUDE.md
  paragraph argues), but I did not confirm it by running a resume. Suggested
  test: build a state file with a revision-0 generation, run the local strategy
  once, and assert a clean cold start, not `RuntimeError`.
- GUI `_runs_registry.py:209` keys runs by `processing_generation`.
  **UNVERIFIED:** whether an upgraded rerun of the same output directory shows as
  a second run or replaces the first.

## Test quality (item d)

**UNVERIFIED.** I did not reach
`tests/unit/core/test_measurement_merge.py`,
`tests/unit/cli/test_work_id_semantics_revision.py`, or the new
`test_measure_texture.py` / `test_dynamic_headers.py` cases, and I did not run
them. From the I-1 probe, the known gap is that no test exercises a
`Grid_*` conflict from a `GridFinder` measurement (explicit or preset-injected),
and none exercises a preset whose shape differs from the image's. Both are the
realistic production triggers of the new raise.

## Unverified checklist
1. Cross-store aggregation with a mismatched `Grid_RowNum` dtype (M-1).
2. Local resume across the revision bump: clean cold start vs "epoch is stale" (c).
3. New test files: can each fail, and are there false greens (d).
