# Plan review: texture-column-naming

Reviewed: `docs/superpowers/plans/2026-09-23-texture-column-naming/plan.md` (commit c56b521a8).
Scope: worktree `.worktrees/texture-column-naming` only. User-settled decisions (D1 format, D3, D4, D6, D7) were not relitigated.
The review was cut short by the coordinator's turn limit. Items I did not finish checking are marked **UNVERIFIED**. No code experiments were run, so pydantic lax-mode claims below come from the pydantic v2 docs and my own knowledge, not from executing code here.

## Verdict

**Feasible with concerns.** No CRITICAL findings. The call-site inventory and line numbers are accurate. The main gaps are:
- one proposed test cannot fail under the mutation the plan uses to prove it;
- D1 claims the recognizer is strict about padding, and the proposed regex is not.

## Verified call sites (line numbers correct)

- `schema/_texture.py:11-13` holds the regex, `:147-157` `member_for_header`, and `:159-175` `get_headers`. All confirmed.
- `measure/_measure_texture.py`:
  - `:96` is `scale: List[int] = [5]`.
  - `:101-114` is the before-validator.
  - `:140-144` is the loop. `:143` discards the result of `meas.merge(...)`, so the D6 claim that `scale=[5,10]` only ever emitted scale 5 is **confirmed**.
  - The order-dependent code is at `:190-194` (`[:-13]`/`[-13:]`), `:240` (`texture_statistics.T.ravel()`) and `:251` (stride of 4). Confirmed.
- Prefabs: `_grid_section_pipeline.py:91,133,199-204`, `_heavy_round_peaks_pipeline.py:101,202,274-279`, `_round_peaks_pipeline.py:52,104,129-134`. Confirmed. The `List` import at `_round_peaks_pipeline.py:3` is used only at `:104`, so it becomes orphaned (confirmed). `_filamentous_fungi_pipeline.py:130`, `_heavy_otsu_pipeline.py:80` and `_heavy_watershed_pipeline.py:76` are already `int`. Confirmed.
- Tests: `test_image_pipeline.py:50` is the only place in `src/` + `tests/` that constructs MeasureTexture with a list (`scale=[3, 4]`), apart from `test_pipeline_serialization.py:142` (`[3, 5, 7]`). Only these two sites break under D7, and the plan lists both. `test_scatter_grouping.py:14,44` and `tests/gui/results_viewer/colony_view/test_grid.py:45-48,58,73` are confirmed as literal pins.
- No code reads `.scale` as a list outside `_measure_texture.py` itself. I checked with rg for MeasureTexture / texture_scale / "scale" across `src` and `tests`. The only other hit is the prose example in `schema/CLAUDE.md` (`self.scale[0]`), which Task 6 already covers.
- `from_json` path: `ImagePipeline.from_json` -> `_deserialize_operations` -> `op_class.model_validate(params)` (`_core/_pipeline_parts/_serializable_pipeline.py:552`). The before-validator therefore runs on `"scale": [5]`, so the legacy-JSON load in D7 works as designed. `BaseOperation.model_config` has `validate_assignment=True` (`abc_/_base_operation.py:175-179`), so `op.scale = [5]` is coerced too.
- List keying: `_make_unique` (`_image_pipeline_core.py:780-810`) produces `MeasureTexture`, `MeasureTexture_1`. Confirmed.
- `split_measurements` groups by producer **class** through `owns_header` (`util/_measurement_outputs.py:110-131`), so the Task 3 claim that both scales land under one `MeasureTexture` group holds with no edit.
- The two "unknown external header" classifiers are as described. Both use `_EXTERNAL_QUALIFIED_HEADER` (a two-segment PascalCase fullmatch) followed by a sweep of `owns_header` over `schema.__all__`:
  - `_gui/shell/_metadata_context.py:89,92-120`
  - `analysis/qc/_expected_vs_detected.py:76-110`

  Without D3, a legacy column falls through to the metadata namespace. D3 is justified.
- Tune gate scope: only `detect/` + `enhance/` (`tests/unit/tune/test_annotation_coverage.py:3`). Confirmed.

## IMPORTANT

### I1. The order guard's stated mutation proof is a false green

The plan (Task 2) proposes a check per object: `Texture_05px-avg-F == mean(Texture_05px-deg{000,045,090,135}-F)`. Its mutation proof is "swap two angles in `get_headers`".

- **Why the stated mutation does not work.** Swapping two angle labels *within* a feature leaves the same four positions under the same feature name. The mean over four angles is symmetric, and the avg at `_measure_texture.py:251` is computed by position, not by name. So the test stays **green** under that mutation.
- **What it does catch.** Only a feature/angle *transposition* in `get_headers`, for example an angle-outer loop. The name-based mean then picks positions 0/13/26/39, which hold different features, and the test goes red.
- **What nothing catches.** Mislabelled angles (a `deg045` column holding the 90-degree value) remain unguarded, as they are today.
- **NaN false failure.** An object that fails Haralick gets `np.full((4,13), nan)` (`:211,238`), and NaN != NaN, so a naive `==` fails on it.

**Fix:**
1. State the mutation that actually goes red: make the `get_headers` loops angle-outer / feature-inner. Or permute the feature order in the directional block only.
2. Compare with `np.testing.assert_allclose(..., equal_nan=True)`.
3. If angle labels matter, add a directional guard on a synthetic anisotropic object, such as horizontal stripes: contrast at `deg000` ~ 0 while `deg090` is large. It needs its own mutation (swap `deg000`/`deg090` -> red).
4. **UNVERIFIED:** that mahotas returns directions in 0/45/90/135 order. Check `mahotas.features.haralick` before writing the directional test.

### I2. D1 says the recognizer is strict about padding; the proposed regex is not

D1 says: "strict on padding (scale `\d{2,}`, angle `\d{3}`) so parse(emit(x)) == x is the only accepted spelling". But:
- `\d{2,}` accepts `Texture_005px-avg-Contrast` and `Texture_00px-avg-Contrast` (scale 0 is invalid under `ge=1`). Neither is ever emitted.
- `deg\d{3}` accepts `deg030` and `deg999`.
- The Task 1 negative list tests only *under*-padding (`5px`, `deg0`, `deg45`), never *over*-padding.

**Fix, either:**
- tighten the pattern to `(?P<scale>0[1-9]|[1-9]\d+)px-(?:deg(?P<angle>000|045|090|135)|avg)-...` and add negatives `Texture_005px-...`, `Texture_00px-...` and `Texture_05px-deg030-...`; or
- reword D1 to "accepts every emitted spelling" and drop the "only accepted spelling" claim.

The tighter pattern also protects the D1 sort-order promise.

### I3. Docs outside the text sweep mention MeasureTexture (UNVERIFIED contents)

rg finds MeasureTexture in:
- `docs/source/explanation/prefab_pipelines_guide.md`
- `docs/source/explanation/measurement_metrics_biological_meaning.md`
- `docs/source/api_reference/core/image_pipeline_methods.rst`

The plan's Text/docstrings table lists none of them. If any shows `scale=[...]`, a `-scale05` column name or a list `texture_scale`, it goes stale. If it is doctested, it breaks. **Fix:** grep these three files for scale / -deg / -avg and add hits to Task 6.

## MINOR

- **M1. Markdown-escaped regex in the plan table.** The regex at plan line 54 contains a backslash-escaped pipe before `avg` (a markdown table escape). Copied literally, it matches a literal pipe character, and no avg column is recognized. The round-trip test would catch it. Still, write the regex in a fenced code block outside the table.
- **M2. Missing import.** `Field` is not imported in `_measure_texture.py`; `:9` imports only `field_validator`. Add it. `from __future__ import annotations` (`:1`) is fine with pydantic here. Keep `import functools`: `@functools.cache` at `:19` still needs it after the `partial` is removed.
- **M3. Validator edge cases are unspecified.**
  - `[]` behaviour: the D7 message, or pydantic's "valid integer" error?
  - A tuple `(5,)`: probably should behave like `[5]`.
  - `True`: pydantic v2 lax `int` accepts bool -> 1 (**UNVERIFIED**, from docs, not run). Add strict mode or a guard if that matters.

  Suggested shape: if the value is a list or tuple, return its only element when it has length 1, and otherwise raise ValueError with the D7 message. Pydantic wraps the error as "Value error, MeasureTexture measures one scale; ...", so the substring assertion in Task 2 works.
- **M4. D8 misdescribes `_merge_on_object_labels`.** D8 says it "merges on columns whose values are equal". The code at `_image_pipeline_core.py:1487-1490` compares `df[col_new_df] == df[col_other_df]`, which is the **same frame against itself**. So every same-named column is used as a merge key unless it contains NaN, and NaN columns get a `_merged` suffix. For two different scales the column sets are disjoint apart from `Object_Label`, so Task 3's "no `_merged` columns" assertion is valid. Correct the D8 wording. The self-comparison is a latent pre-existing bug and is out of scope.
- **M5. Tune auto-space may change (UNVERIFIED).** `infer_search_space` (`tune/_search_space/_infer.py:726-744`) walks `pipeline.get_ops()`. I did not check whether `get_ops()` includes `meas`. If it does, `scale` changes from an excluded List to an int field with ge=1 and no upper bound. That would produce a new unbounded-heuristic IntRange knob, changing auto-space output for any pipeline with MeasureTexture. Check this, and pin it or add a TuneSpec/exclusion if needed.
- **M6. Test key renames (UNVERIFIED).** The rename of keys in `test_image_pipeline.py:50` (`MeasureTexture3`/`MeasureTexture4`) needs a grep of that file for other "MeasureTexture" key lookups or benchmark assertions.
- **M7. Legacy-JSON test coverage.** Also assert that a legacy `"scale": [3, 4]` payload through `ImagePipeline.from_json` raises. That is the documented D7 cost, and the test pins the error message users will see in recompile/QC. Also assert that `to_json` re-emits `"scale": 5` (an int).
- **M8. Unused capture groups.** The scale/angle named groups are unused by `member_for_header`. They are harmless; drop them or use them in the tightened I2 pattern.
- **M9. Prefab changelog.** A user passing `texture_scale=[3, 5]` to a prefab now gets a ValidationError, as it would with MeasureTexture itself. Add this to the changelog note in Task 6.

## Regex / ownership overlap analysis (item 2)

- New and legacy patterns cannot overlap:
  - Legacy must end `-scale` followed by digits.
  - New must have digits plus `px-` right after `Texture_`, and must end in an alphanumeric label.
- No other schema can claim a new-format header:
  - Static schemes match `member.value` exactly (`_measurement_info.py:387-396`), and no member value has this shape.
  - metric_qualified schemes anchor on their own category prefix (`parse_qualified_header`, `:611-630`). No other category is "Texture" (TEXTURE is the only one; rg for TEXTURE across src confirms).
- The new pattern cannot claim a non-texture column because cat must equal "Texture". A growth fit on a texture column becomes `LogGrowthModel_05px-avg-Contrast_r` (`metric_token` strips only the category and whitespace, `util/_measurement_outputs.py:230,256-268`). The growth schema owns it, not TEXTURE.
- `TextureGray_...` stays unrecognized under both patterns, as the plan intends.

## GUI / schema-shape (item 5)

- The builder derives widgets from `model_fields` annotations (`_gui/_operation_registry.py:386-410`) and detects lists by their `list` origin (`_gui/_param_forms.py:73-88`, `_operation_registry.py:545-555`). int falls back to a numeric widget automatically.
- The builder loads files through `ImagePipeline.from_json` (`_gui/builder/_callbacks.py:6895`), so a stored `[5]` is coerced to 5 before any form is rendered.
- Builder state is a memory store (`_gui/builder/_layout.py:3208`), so no persisted pre-upgrade list value survives a reload.
- No test pins MeasureTexture's JSON schema or its scale widget. rg found no "scale" param assertions in `tests/unit/gui` other than `test_scatter_grouping.py:14`.
- Run-console / results-viewer QC recompute paths re-read pipeline.json through from_json, and the plan already names them. **UNVERIFIED:** whether the run console renders op params anywhere independently of from_json.

## Unnecessary complexity (item 4)

Little to cut. The two-regex approach is minimal given D3. Dropping `functools.partial` is right. The optional README-generator change should stay optional or be dropped. Keep the named capture groups only if they are used (M8).

## Counts

CRITICAL 0 / IMPORTANT 3 / MINOR 9
