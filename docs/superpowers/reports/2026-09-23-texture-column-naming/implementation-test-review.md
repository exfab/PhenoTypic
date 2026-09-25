# Implementation + test review: texture-column-naming

Reviewed: `git diff 3b7c1543e..HEAD -- src tests` (a7aae8d0b, d8a078eb1), worktree `.worktrees/texture-column-naming`.
The review stopped at the coordinator's turn limit. Checks I did not finish are marked **UNVERIFIED**.

## Summary

The emit/parse pair, the validator and the prefab narrowing match the plan. Every plan Task 1–6 item I checked is present.

What I ran:
- The touched test files: **205 passed**. Command: `QT_QPA_PLATFORM=offscreen uv run pytest -p no:cacheprovider -q` over the 9 touched test files.
- The `MeasureTexture` doctest: 1 collected, and it passes.
- `ruff check` on the changed files: clean.

What I found:
- No definite defect in the diffed code.
- Two IMPORTANT findings:
  - (I1) The recommended multi-scale pattern has a silent row-loss trap that D8 describes as "harmless".
  - (I2) No test ties `scale` to the GLCM distance.
- One IMPORTANT (UNVERIFIED) concern: continuation mixes old and new column names across the upgrade.

## Mutation spot-checks (all reverted; `git status` clean afterwards)

| Mutation | Result |
|---|---|
| M1 `get_headers` made angle-outer / feature-inner | RED: `test_average_column_is_the_mean_of_its_four_directions`, `test_texture_get_headers_emits_scale_direction_feature_order` |
| M5 validator accepts multi-element lists (`len(scale) == 0`) | RED: 3 tests (multi-element refused, assignment, legacy JSON `[3, 4]`) |
| M4 legacy branch deleted from `member_for_header` | RED: 5 tests (schema legacy, GUI metadata, QC layout x2, split_measurements) |
| extra: `ge=1` dropped from the `scale` Field | RED: `test_non_positive_scale_is_refused[0,-1,[0]]` |
| **extra: `distance=scale` -> `distance=5` in `_compute_haralick`** | **GREEN: all 37 tests in `test_measure_texture.py` + `test_image_pipeline.py` pass.** See I2. |

## CRITICAL

None.

## IMPORTANT

### I1. Same-scale duplicate measurers silently drop every row, not "harmless-but-ugly" (D8 premise is wrong)

`_core/_pipeline_parts/_image_pipeline_core.py:1487-1494` has two problems:
- **Every same-named column becomes a merge key.** The check `np.all(df[col] == df[col])` compares the incoming frame with *itself*.
- **The merge is an inner join.** It runs `new_df.merge(df, on=cols_to_merge_on)`.

So two `MeasureTexture` with the same `scale` but different `quant_lvl` or `enhance` do this:
- They emit identical column names with different values.
- All 65 texture columns become join keys.
- The inner join matches nothing.

**Observed:** I ran `OtsuDetector` + `SmallObjectRemover(min_size=200)` on `load_synth_yeast_plate()`, which gives 96 objects with no NaN. With `meas=[MeasureTexture(scale=5, quant_lvl=8), MeasureTexture(scale=5, quant_lvl=32)]`, `apply_and_measure` returns **0 rows** and raises no error. That is silent loss of the whole table, including other measurers' columns.

With NaN rows (the unfiltered plate, where 456/552 objects are NaN), you get 65 `_merged` columns instead. That is what D8 describes.

This change makes "repeated measurers" the documented multi-scale idiom (`_measure_texture.py:41-44`, the prefab docstrings). A user comparing quantization levels at one scale is a plausible configuration.

D8 (no guard) is user-settled, but the fact it rests on is wrong, so the decision should go back to the user with the corrected fact. Options, cheapest first:
- (a) State in the `MeasureTexture` docstring that two measurers must differ in `scale`, and that duplicates corrupt the measurement table.
- (b) Add a construction-time check in `ImagePipeline` that no two `MeasureTexture` entries share a `scale`: raise `ValueError` naming both keys.
- (c) Fix the self-comparison at `:1490` to compare `new_df[col]` with `df[col]`. It is a pre-existing bug and is out of scope per the plan.

Also add a test that pins whichever option is chosen.

### I2. No test proves `scale` reaches the GLCM (`_measure_texture.py:251`)

The mutation `distance=5` (ignore `scale`) passes every test:
- `test_measure_emits_exactly_one_scale_of_columns` checks column names only, and those come from `TEXTURE.get_headers(scale)`, not from the computation.
- `test_multi_scale_texture_via_repeated_measurers` checks header sets only.

So a regression that labels `Texture_02px-...` columns with distance-5 values would ship green. This is exactly the mislabelling risk the new naming exists to prevent.

**Fix:** add a value test in `tests/unit/measure/test_measure_texture.py`. Recompute one finite object directly with `mahotas.features.haralick(MeasureTexture._quantize_arr(fg, 8), distance=2, ignore_zeros=True, return_mean=False)`. Here `fg` is the object crop from `img.gray.foreground()[props[idx].slice]`, with other labels zeroed. Then `assert_allclose` the result's `.T.ravel()` against `table.iloc[idx][TEXTURE.get_headers(2)[:-13]]` at `rtol=1e-12`.

A cheaper alternative is to assert that `scale=1` and `scale=5` give different `avg-Contrast` vectors on finite rows. The `distance=5` mutation must turn it red.

### I3 (UNVERIFIED). Continuation across the upgrade mixes old and new column spellings in one run

The work id is keyed on `pipeline_fingerprint = file_sha256(pipeline.json)` (`_cli/_cli_failure_tracker.py:367`) and `WORK_ID_SCHEMA_VERSION = 1` (`:39`). Neither changes when the installed code starts emitting different column names.

If a user re-runs the same command after upgrading:
- The per-image stores completed before the upgrade are reused. Their embedded tables have `Texture_Contrast-avg-scale05`.
- Images processed after the upgrade emit `Texture_05px-avg-Contrast`.
- The aggregated `master_measurements.parquet` / `measurements.*` would then carry both spellings for the same features, each half NaN.

There is a precedent for fencing this kind of change: `PROCESS_LAYER_SEMANTICS_REVISION` fences a change in what an output means.

I did not check whether aggregation reconciles or refuses this, or whether a phenotypic-version check elsewhere invalidates continuation.

**Fix:** decide explicitly. Either add a measurement-schema revision to the full/measure digest, scoped to pipelines containing `MeasureTexture` if a global bump is too costly. Or document "finish or restart runs in flight before upgrading" in the changelog. Add a test for the chosen behaviour.

## MINOR

- **m1. The pydantic lax mode accepts odd `scale` values** (`_measure_texture.py:121,138-145`).
  - `scale=True` -> `1`, `[True]` -> `1`, `"05"` -> `5`, `b"5"` -> `5`, `5.0` -> `5`, `np.array([5])` -> `5`.
  - This is pre-existing lax behaviour and harmless, but `True` meaning distance 1 is surprising.
  - Fix: reject `bool` in the before-validator, or set `strict=True` on the Field. Strict mode would break `"scale": 5.0` JSON, which is probably acceptable.
- **m2. Non-list sequences get pydantic's generic error, not the D7 guidance.** Affected inputs: `np.array([3, 4])`, `{5}`. Fix: widen the check to include `np.ndarray`, or accept any non-str `Sequence`.
- **m3. `$` accepts a trailing newline** (`schema/_texture.py:14-15,22-23`). `member_for_header("Texture_05px-avg-Contrast\n")` returns `CONTRAST`, while the static scheme matches exactly. The legacy pattern already did this. Fix: `\Z`, or `pattern.fullmatch`.
- **m4. The D7 blast radius is wider than the plan lists.** The plan names three paths: recompile/finalize, QC recompute and QC rebuild. A legacy `"scale": [3, 4]` also breaks every other `ImagePipeline.from_json` consumer:
  - `_cli/_cli_validation.py:49,480`
  - `_cli/_cli_checkpoint_handler.py:589`
  - `_cli/_cli_staged_strategy.py:79`
  - `_cli/_cli_stage2_token.py:153`
  - `tune/__main__.py:267`
  - `_gui/analysis/_recipe_state.py:985`
  - `_gui/builder/_callbacks.py:6895`

  Separately, `_gui/results_viewer/_scatter_tab/_grouping.py:59-66` constructs `cls(**params)`. That call fails with a debug log only, so on such a run every texture column falls into "Unattributed" instead of "MeasureTexture". This is the accepted D7 cost; list it in the changelog.
- **m5. A prefab gap.** No test pins `texture_scale` narrowing: a prefab given `texture_scale=[3, 5]` raising, and a prefab given `5` producing `Texture_05px-...`. Low risk, because the prefabs forward straight to `MeasureTexture`.
- **m6. The `_merged` assertion is weak.** In `test_multi_scale_texture_via_repeated_measurers`, `assert not [... "_merged" ...]` can only fire on a NaN collision. The header-set assertions carry that test. Consider also asserting that the row count equals a single-measurer run's: that would catch the I1 row-loss class of bug if the scales ever collided.
- **m7. Angle labels are still unguarded.** The order guard catches only a feature/angle transposition, as the plan accepted. A `deg045` column holding the 90-degree value would pass. This is out of scope per the plan; noted for completeness.
- **m8. `matrix_name` is unused in `get_headers`.** It is documented as such; fine.
- **m9 (UNVERIFIED).** `uv run mypy` on `_measure_texture.py` + `_texture.py` reports 2 errors, including a `_operate` staticmethod-vs-method override note at `_measure_texture.py:147`. These look pre-existing in the base-class signature. I did not diff them against `3b7c1543e`.
- **m10 (UNVERIFIED).** I did not check whether the Task 6 changelog notes exist: the new names, `scale` as an int, the D7 refusal, and prefab `texture_scale`.

## Verified OK

- **Header order.** `get_headers` is feature-outer x angle-inner for the 52 directional names, then the 13 averages. This matches the positional fill at `_measure_texture.py:216-219,265,276-279`.
- **The strict recognizer.** It rejects every over-padded or under-padded form in the negative list. New and legacy patterns cannot overlap. `member_for_header` falls through correctly.
- **The validator.** `validate_assignment=True` makes `m.scale = [7]` coerce and `m.scale = 0` raise. `from_json` of legacy `[5]` loads, and `to_json` writes `5`.
- **The JSON schema** is now `{"type": "integer", "minimum": 1}`.
- **Tune `infer_search_space`** walks `get_ops()`, which excludes `meas`, so `scale` does not become a tune knob. This resolves plan-review M5.
- **No remaining list reads.** No `src/` or `scripts/` code reads `.scale` as a list or passes a list scale. The `docs` hits (`prefab_pipelines_guide.md:28`, `measurement_metrics_biological_meaning.md:98`, `image_pipeline_methods.rst:120,127`) use `MeasureTexture()` only.
- **Two call sites break on TEXTURE's `scale` argument, both pre-existing, neither a regression:**
  - `_cli/_cli_output_manager.py:553` (`_collect_feature_headers`) calls `info.get_headers()` with no argument, which raises TypeError for TEXTURE. It is swallowed with a debug log, and the function has no `src` callers.
  - `_scatter_tab/_grouping.py` handles the TypeError with a category-prefix fallback.

## Counts

CRITICAL 0 / IMPORTANT 3 (one UNVERIFIED) / MINOR 10
