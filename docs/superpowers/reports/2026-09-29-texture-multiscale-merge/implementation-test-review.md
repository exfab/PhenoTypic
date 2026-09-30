# Implementation + test review: MeasureTexture multi-scale merge fix

Date: 2026-09-29
Scope: uncommitted worktree change in `.claude/worktrees/elegant-spence-83a8c9`
(`src/phenotypic/measure/_measure_texture.py`, `src/phenotypic/schema/_texture.py`,
new `tests/unit/measure/test_measure_texture.py`).
Reviewer mode: analysis only. The only edits were temporary mutations to
`_measure_texture.py`. Each was restored from a byte copy, and the file's SHA-1
(`65c1fac4…`) was checked against the original after every restore. The final
`git status --short` and `git diff --stat` match the starting state.

## Summary

The fix is correct and complete for the case it targets, which is distinct scales.
`meas = meas.merge(..., on=OBJECT.LABEL, how="outer")` now keeps every scale. Rows
line up by label, the label dtype stays `int64` (rows that failed Haralick with NaN
do not turn it into float), and the public `measure()` wrapper, `include_meta=True`,
`ImagePipeline.measure`, `split_measurements`, `generate_output_key`, and
`TEXTURE.member_for_header` / `_TEXTURE_HEADER_RE` all handle the 131-column frame.
I ran each of those paths to check this (see Evidence). Removing the `len > 1`
guard does not change behaviour.

The fix does expose one latent problem that the bug had been hiding. **Duplicate
scale values now produce pandas `_x`/`_y`-suffixed columns that no schema code
recognises**. Before the fix this input happened to produce the correct 65 columns.
This is a real behaviour regression, but the input is unusual.

The two tests fail when the bug is put back and pass on the fix, so they are real
tests. One regression class survives both of them: the `scale` value never reaching
mahotas. The second test's docstring also overstates what it proves.

Severity counts:
- Critical (introduced): 0
- Medium (introduced/exposed by this change): 1 (duplicate scales)
- Low (test weakness): 2 (scale→distance never pinned; test-2 docstring overclaims)
- Low (docstring accuracy): 1 (new sentence is false for duplicate scales)
- Pre-existing (not introduced): 4 (listed separately at the end)

## Evidence gathered

All commands were run with `uv run` from the worktree. Probe scripts were kept in
`/tmp/texrev/`, outside the repo.

| Probe | Result |
|---|---|
| `MeasureTexture(scale=[5,10]).measure(img)` | shape `(2, 131)`, `Object_Label` int64 `[1, 2]`, RangeIndex |
| `scale=[10,5]` | columns come out in the order given (scale10 block first); the test only covers ascending order |
| tiny 2x2 object (Haralick fails → NaN) with `[5,10]` | 3 rows, labels `[1,2,3]` int64, only row 3 all-NaN; outer merge keeps it |
| objmap labels `{7, 2}` (7 comes first in raster order) | `objects.labels == [2, 7]`; single-scale and multi-scale row order both `[2, 7]` |
| zero objects, `[5,10]` and `[5]` | both raise `OperationFailedError(NoObjectsError)`, the same as before the fix |
| `include_meta=True` with `[5,10]` | shape `(2, 144)`; the meta merge on `Object_Label` works |
| `ImagePipeline(meas=[MeasureSize(), MeasureTexture(scale=[5,10])]).measure(img)` | `(2, 158)`, 65 scale05 + 65 scale10 columns |
| `split_measurements(frame)` | `{'MeasureTexture': (2, 131)}`: every scale10 column is recognised |
| `generate_output_key(frame)` | 131 rows |
| `TEXTURE.member_for_header` on scale10 and scale100 headers | both resolve |
| `scale=(5,10)` tuple / JSON round trip | `[5, 10]` / `[5, 10]` |
| `scale=[5,5]` | **`(2, 131)`, 130 columns end in `_x`/`_y`**; `generate_output_key` → 1 row; `split_measurements` → `{}` |
| `scale=[5,5,5]` | 196 columns: 65 `_x`, 65 `_y`, 65 unsuffixed |
| `scale=[5,10,5]` | 196 columns: 65 `_x`, 65 `_y` |
| Independent oracle: direct `mahotas.features.haralick(q, distance=10, ignore_zeros=True, return_mean=False).T.ravel()` on object 1's crop vs frame's scale10 deg-block row 0 | **bit-exact match** (max abs diff 0.0) |
| Focused regression: `test_measure_texture.py`, `util/test_measurement_outputs.py`, `schema/test_dynamic_headers.py`, `schema/test_change_note.py`, `schema/test_classification.py`, `docs/test_measurements_ref_extension.py`, `core/test_pipeline_serialization.py`, `core/test_image_pipeline.py::test_pipeline_on_image` | **116 passed** |
| `ruff check` on the three changed files | clean |

### Mutation testing

| Mutation | Test 1 | Test 2 | Verdict |
|---|---|---|---|
| M1: reintroduce the bug (`meas.merge(...)` without assignment) | FAIL (missing 65 scale10 columns) | FAIL (`KeyError` scale10 not in index) | **killed**; the tests can fail |
| M2: `pd.concat([meas, other.drop(columns=LABEL)], axis=1)` (positional, no label join) | pass | pass | survives, but it is equivalent here: every scale iterates the same `image.objects.labels` in the same order. Not a defect. |
| M3: `distance=scale` → `distance=1` in `_compute_haralick` | pass | pass | **survives: real gap** (see L1) |
| M5: positional concat of a row-reversed later-scale frame | pass | FAIL | killed. Test 2 does catch row misalignment. |

## Critical Issues (High Confidence)

None.

## Likely Issues (Medium Confidence)

### M1 (introduced / exposed). Duplicate scale values now produce unrecognisable `_x`/`_y` columns

`src/phenotypic/measure/_measure_texture.py:140-142`, together with the
`scale` field validator at lines 101-114.

Before the fix, `scale=[5, 5]` discarded the second merge and so returned the
correct 65 `scale05` columns by accident. Now that the merge result is assigned,
pandas sees 65 overlapping non-key columns and applies the default suffixes
`("_x", "_y")`:

- `scale=[5,5]` → 130 columns such as `Texture_AngularSecondMoment-deg000-scale05_x`.
- `_TEXTURE_HEADER_RE` (`schema/_texture.py:11-13`) is anchored with `$`, so none of
  them match. `generate_output_key` describes only `Object_Label`, and
  `split_measurements` returns `{}`. In other words, the texture block vanishes from
  per-producer splits and from the column key, with no error.
- With three or more entries (`[5,5,5]`, `[5,10,5]`) the result is a mix of suffixed
  and unsuffixed copies.

Impact: silent schema drift in the deliverables for any user or config that lists a
scale twice, for example a hand-edited JSON pipeline or a concatenated list of
scales. No shipped prefab or tune space does this. All prefab defaults are a single
`int`, and no tune/GUI code references `MeasureTexture.scale`. That keeps the
severity at Medium rather than High.

Fix: reject or de-duplicate in the existing validator, which is also the project's
stated place for guards ("put input normalization and guards in a
`field_validator`"). Rejecting is more explicit than silently de-duplicating:

```python
@field_validator("scale", mode="before")
@classmethod
def _coerce_scale_to_list(cls, scale):
    if not hasattr(scale, "__getitem__"):
        scale = [scale]
    scale = list(scale)
    if not scale:
        raise ValueError("scale must contain at least one pixel offset")
    if len(set(scale)) != len(scale):
        raise ValueError(f"scale values must be unique, got {scale}")
    return scale
```

As a belt-and-braces measure, the merge itself can refuse overlap:
`meas.merge(..., on=OBJECT.LABEL, how="outer", validate="one_to_one", suffixes=(None, None))`.
With `suffixes=(None, None)`, pandas raises on overlapping columns instead of
renaming them.

Test to add (it fails on the current code):

```python
def test_duplicate_scales_are_rejected():
    with pytest.raises(pydantic.ValidationError):
        MeasureTexture(scale=[5, 5])
```

## Test Weaknesses (Low)

### L1. Nothing pins that a scale's columns hold that scale's texture

Mutation M3 (`distance=1` hard-coded) passes both tests. Test 1 checks only column
names. Test 2 compares the multi-scale run against single-scale runs, and both go
through the same mutated `_compute_haralick`, so they agree. A regression where
`scale` stops reaching mahotas, or where every scale is computed at the first
scale's distance, would leave both tests green while shipping `scale10` columns
that contain scale-1 or scale-5 values.

This gap existed before the change (there were no texture tests at all). It
matters here because the purpose of the fix is "scale10 columns exist *and mean
scale 10*".

Two proposed assertions, both checked against M3 (under M3 the blocks are
`allclose` and the oracle is off by up to 10.0; on the fixed code the blocks differ
by up to 17.7 and the oracle matches with diff 0.0):

```python
def test_scale10_columns_are_distance_10_haralick(textured_pair_image):
    import mahotas as mh
    frame = MeasureTexture(scale=[5, 10]).measure(textured_pair_image)
    fg = textured_pair_image.gray.foreground()[5:35, 5:35]
    q = np.clip(np.floor(fg * 32), 0, 31).astype(np.uint8)   # quant_lvl=32 default
    expected = mh.features.haralick(q, distance=10, ignore_zeros=True,
                                    return_mean=False).T.ravel()
    got = frame.loc[frame[OBJECT.LABEL] == 1,
                    TEXTURE.get_headers(10)[:52]].to_numpy().ravel()
    np.testing.assert_array_equal(got, expected)  # same ops, same order: bit-exact
```

A cheaper alternative is
`assert not np.allclose(frame[scale05].to_numpy(), frame[scale10].to_numpy())`.
Note that the oracle depends on the fixture layout: object 1's crop is exactly
`[5:35, 5:35]` and has no neighbour pixels, so no masking is needed. Put that in a
comment next to the oracle. Bit-exact equality is justified because the oracle runs
the same numpy/mahotas operations in the same order on the same input.

### L2. Test 2's docstring claims more than the test can show

`tests/unit/measure/test_measure_texture.py:42-43` says the merge "must align each
scale's rows by Object_Label, not just append columns". M2, a positional `concat`
that ignores labels entirely, passes this test, and it is equivalent by
construction because every scale iterates the same `image.objects.labels`. What the
test *does* catch is a row permutation within a scale (M5 killed). Suggested
wording: "each scale's block, rows matched by `Object_Label`, equals that scale
measured on its own." This is only a wording fix. The test itself is sound.

Also worth noting: the existing `tests/unit/core/test_image_pipeline.py:50`
(`MeasureTexture(scale=[3, 4], quant_lvl=8)`) is the test that should have caught
this bug. It stayed green because it only compares `pipe.measure(pipe.apply(x))`
against `pipe.apply_and_measure(x)`, and both were equally wrong. That shows a
self-consistency test cannot stand in for a column-presence assertion. No change is
required now that the new test exists.

## Maintainability Concerns

### D1 (introduced, low). The new TEXTURE docstring sentence is inaccurate for duplicate scales

`src/phenotypic/schema/_texture.py:37-39`: "Each scale value … writes its own full
set of these columns". This holds for distinct scales and is exactly the behaviour
now shipped. For `[5, 5]` it is false: the result is suffixed columns that do not
follow the documented pattern. If M1 is fixed by rejecting duplicates, the sentence
becomes accurate as written. If it is fixed by de-duplicating, change it to "each
distinct scale value". The `scale05`/`scale10` spelling matches
`get_headers` (`{scale:02d}`).

Optional: `MeasurementInfo.change_note()` (`schema/_measurement_info.py:376-387`)
is the project's hook for "a release changes its public columns".
`MeasureTexture(scale=[a, b, …])` output gains 65 columns per extra scale. Users
comparing against earlier multi-scale runs will see new columns, and earlier
deliverables silently lack them. A `.. versionchanged::` note on TEXTURE would
record that. `tests/unit/schema/test_change_note.py:100-102` currently asserts that
TEXTURE carries *no* 0.20.0 note, so a note would have to use the next version or
that test would need updating. This is the maintainer's call and does not block
the change.

## Test Coverage Gaps (beyond L1)

- Non-ascending scale order (`[10, 5]`) produces the columns in input order. This
  was verified by probe but is not tested. It is cheap to add as a parametrisation
  of test 1.
- A NaN-row object (Haralick failure) within a multi-scale run: the probe shows the
  outer merge keeps it and the label stays int64. A test would pin
  `how="outer"`/dtype behaviour in case someone later changes the join to `inner`
  (which, with identical label sets, would still behave the same, so this is
  low-value).
- The fixture and test conventions look fine: seeded `default_rng(0)`, no
  filesystem, sized so a scale-10 offset stays inside each object, and the
  `isna().all().any()` guard makes sure `assert_frame_equal` cannot pass on
  NaN == NaN. Ruff is clean.

## Recommended Changes (prioritised)

1. **(M1 + D1)** Reject duplicate (and empty) `scale` lists in
   `_coerce_scale_to_list`, and add `test_duplicate_scales_are_rejected`.
   Optionally pass `suffixes=(None, None)` to the merge so any future overlap
   raises instead of being renamed.
2. **(L1)** Add the mahotas oracle test (or at least the scale05 ≠ scale10
   assertion), so the columns are pinned to their distance.
3. **(L2)** Reword test 2's docstring.
4. Optional: add a TEXTURE change note for the restored multi-scale columns.

## Pre-existing issues (not introduced by this change)

- **P1. No lower bound on `scale`** (`_measure_texture.py:96`). `scale=[0]` returns
  finite, degenerate values (a pixel co-occurring with itself) under header
  `scale00`. `scale=[-3]` returns all-NaN with no warning (`warn=False` default)
  under header `…-scale-3`, which `_TEXTURE_HEADER_RE` does not recognise.
  `scale=[40]` on 30-px objects is all-NaN with no warning. Fix: `List[PositiveInt]`
  (or `conint(ge=1)`) in the same validator as M1.
- **P2. `scale=[]`** raises a bare `IndexError` wrapped in `OperationFailedError` at
  measure time rather than a `ValidationError` at construction. The M1 fix above
  covers it.
- **P3. Docstring drift in `MeasureTexture`.** The `_operate` docstring
  (`:127-128`) says rows are "indexed by object labels", but labels are a column
  and the index is a RangeIndex. The class `Returns:` block (`:59-65`) calls the
  key "Label" and shows headers without the `Texture_` prefix
  (`Contrast-deg000-scale05`). `_compute_haralick`'s `Returns:` (`:171-174`) says
  it returns a `dict`, but it returns a DataFrame.
- **P4. `ImagePipeline` merge helper** (`_core/_pipeline_parts/_image_pipeline_core.py:1488-1495`)
  compares `df[col] == df[col]` (a frame against itself) where it presumably meant
  `new_df[col] == df[col]`. This holds for any shared column except NaN. It does
  not interact with this change, because texture column names are unique across
  measurers, and I record it only because the brief asked about pipeline merge
  code.
- Zero-object images raise `NoObjectsError` for single-scale and multi-scale runs
  alike. This is unchanged by the fix, and I noted it only because the brief asked.
