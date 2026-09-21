# Review: shared ΔE2000 medoid (`git diff 14c5f2758 HEAD`)

Branch `feat/calibrate-color-rpcc`, commits `f3abc5461` and `e9b86b5a4`. Plan:
`docs/superpowers/plans/2026-09-21-in-frame-checker-color-correction/shared-medoid.md`.

Status: COMPLETE.

**Verdict.** All three requirements hold in the code. The move is verbatim. Requirement 2 holds:
the only pixels that reach `candidate_medoid` are the label's own pixels. Three
problems remain:

- Requirement 3's warning is invisible outside pytest (F1).
- Memory regresses for large objects, and the plan does not mention it (F2).
- The new medoid tests cannot detect a MeasureColor estimator other than `candidate_medoid`, so
  requirement 1 has no guard on the MeasureColor side (F3).

Severity counts: High 0, Medium 3, Low 5.

Scripts used for the runs are in `/tmp`: `legacy_paths.py`, `default_filters.py`, `edges.py`,
and `mutplug/mutplug.py`. No repo file was edited except this report.

---

## Confirmed findings (by running)

### F1: MEDIUM. Users never see the legacy-field warning in a script or on the CLI, so requirement 3 is only half met

`src/phenotypic/measure/_measure_color.py:108-113` raises the warning as `DeprecationWarning` with `stacklevel=2`.
With `stacklevel=2` the warning is attributed to pydantic's frame, not to user
code. I recorded the attributed file for each load path:

| Load path | Attributed to |
|---|---|
| `from_json` and `ImagePipeline.from_json` | `site-packages/pydantic/main.py:716` |
| Constructor kwargs | `site-packages/pydantic/main.py:250` |
| `model_validate_json` | `site-packages/pydantic/main.py:766` |

Python's default filters show a `DeprecationWarning` only when it is attributed to
`__main__`. Outside pytest, the warning is therefore dropped.

- **Confirmed by running.** `/tmp/default_filters.py` loads a legacy
  `{"class":"MeasureColor","params":{"medoid_max_pixels":300,"random_seed":3}}`
  through `MeasureColor.from_json` and through `MeasureColor(medoid_max_pixels=1)`. I ran it with plain
  `uv run python` and `PYTHONWARNINGS` unset. **No warning is printed.** With
  `-W default::DeprecationWarning`, both warnings appear, attributed to
  `pydantic/main.py`.
- **Failure scenario.** A user reruns a saved pipeline that set `random_seed=3` or
  `medoid_max_pixels=5000` with `python -m phenotypic`. The fields are silently
  discarded and the medoid columns change, and nothing tells the user. The plan says
  "drops them with a `DeprecationWarning`", and the class docstring at `_measure_color.py:71-72`
  promises a warning. Both assume the user sees it.
- **Why the tests don't catch it.** pytest turns on `DeprecationWarning` display, and
  `pytest.warns` captures a warning no matter which frame it is attributed to.
- **Fix.** Use `FutureWarning`, which the stdlib shows by default and intends for deprecations
  aimed at end users. Alternatively, also `logger.warning(...)`. Then add a test that
  runs a subprocess with default filters, or at least asserts the category is not
  `DeprecationWarning`.

### F2: MEDIUM. Peak memory is about 17 KB per object pixel, against a flat ~30 MB before, and the plan does not mention it

`src/phenotypic/util/_robust_color_stats.py:212-219` scores `chunk_size=64` candidates
against all N pixels per block. colour's ΔE2000 allocates many `(64, N)` float64
temporaries per block. MeasureColor (`_measure_color.py:176`) cannot change
`chunk_size`.

- **Confirmed by running** (`/tmp/edges.py`, `tracemalloc`, unimodal cloud):

  | Object pixels | `candidate_medoid` | old `medoid_ciede2000(max_pixels=1000)` |
  |---|---|---|
  | 20 000 | 0.4 s, 0.34 GB peak | 0.1 s, 0.03 GB peak |
  | 100 000 | 2.2 s, 1.69 GB peak | 0.1 s, 0.03 GB peak |

- **Failure scenario.** The largest objects are exactly where this bites: a merged lawn, a large filamentous colony, or
  a whole-plate object from a failed detection. At 10^6 pixels, extrapolating linearly
  gives about 17 GB for a single object, and `--njobs` workers each pay it at the same time.
  The result is an OOM kill on a SLURM allocation that ran the old code comfortably. The plan's
  "What changes for users" table covers time only. It also gives 1.18 s at 40k pixels, while I measured 2.2 s
  at 100k. Time is linear, but memory is not bounded at all.
- **Fix.** Size the chunk from a byte budget, e.g.
  `chunk = max(1, min(64, budget_bytes // (N * 8 * ~30)))`. Or put a hard pixel cap, with
  deterministic decimation, in front of MeasureColor's call. Either way, add the memory row
  to the plan's user-facing section.

### F3: MEDIUM (test weakness). The medoid tests pass with an estimator that is not `candidate_medoid`, so requirement 1 is unguarded for MeasureColor

`tests/unit/measure/test_measure_color.py`, `test_medoid_uses_only_the_objects_own_pixels`
and `test_medoid_is_deterministic`.

- **Confirmed by running.** I used an in-process mutation plugin (`/tmp/mutplug/mutplug.py`, loaded with
  `-p mutplug`) and ran the whole test file under each mutation:

  | Mutation of MeasureColor | Result |
  |---|---|
  | none (baseline) | 10 passed |
  | `bbox`: whole bounding box (background + neighbour) to the Lab row | **`uses_only_own_pixels` FAILS** |
  | `anyobj`: every labelled pixel in the bbox (neighbour, no background) | **`uses_only_own_pixels` FAILS** |
  | `subsample`: revert to seeded `medoid_ciede2000(max_pixels=1000)` | **both new medoid tests FAIL** |
  | `k1`: `candidate_medoid(px, k=1)`, i.e. the pixel nearest the geometric median | **10 passed** |
  | `nearest_gm`: pixel nearest MeasureColor's own loose (50 / 1e-4) Weiszfeld centre, no ΔE2000 at all | **10 passed** |

  The requirement-2 and anti-subsample guards work. On the ring fixture, isotropic
  Gaussian noise puts the exhaustive ΔE2000 medoid on the same pixel as the one nearest the geometric
  median. The "equals the exhaustive medoid" assertion therefore cannot tell the
  shared estimator from a trivial one, and MeasureColor could quietly diverge from
  `CalibrateColorRpcc`'s estimator without any test failing.
  `test_candidate_medoid_is_exported_from_util` checks identity between util and
  `_checker_measure` only, not what MeasureColor calls.
- **Fix.** Make two test additions:
  1. Add a spy test. Monkeypatch `phenotypic.measure._measure_color.candidate_medoid` with a
     wrapper that records `(lab_px, k)`. Assert it was called once per label, with `k ==
     op.medoid_candidates`, and with a pixel multiset equal to `lab[objmap == L]`. This also
     pins requirement 2 directly.
  2. Add a fixture where the exhaustive medoid is **not** the pixel nearest the geometric median, such as
     the skewed or bimodal 900/300 cloud already used in `tests/unit/correction/test_checker_measure.py:81`.
     Assert MeasureColor's medoid equals `candidate_medoid(obj_px)` and also
     differs from the nearest-geometric-median pixel. The second assertion is the control that proves the fixture discriminates.

---

## Confirmed findings (by reading)

### F4: LOW. The two "shared" `medoid_candidates` fields validate differently

- `src/phenotypic/correction/_color_correction/_calibrate_color_rpcc.py:159`:
  `medoid_candidates: int = DEFAULT_MEDOID_CANDIDATES`, with no `ge=1`.
- `_measure_color.py:94`: `Field(default=..., ge=1)`.

**Scenario.** `CalibrateColorRpcc(medoid_candidates=0)` constructs without error. It then raises
a plain `ValueError` from `candidate_medoid` at `apply` time, on the first tile, and
not a `ValidationError` at construction. The field was not changed by this diff, but
the change's stated goal is one shared estimator with one knob.

**Fix.** Declare a shared annotated type, e.g.
`MedoidCandidates = Annotated[int, Field(ge=1)]`, and use it in both classes.

### F5: LOW. The legacy-load test covers only `model_validate(dict)`

`test_legacy_medoid_fields_still_load` (`test_measure_color.py`, new block) never
goes through `from_json` or `ImagePipeline.from_json`, which are the paths a saved
pipeline actually takes. I confirmed by running that both paths work today
(`/tmp/legacy_paths.py`: `from_json`, constructor kwargs, `model_validate_json`, and
`ImagePipeline.from_json` with a legacy `meas` entry all load and reset
`medoid_candidates` to 256). If the class were ever loaded through `model_construct` or a
custom `from_json` that strips unknown keys first, nothing would notice.

**Fix.** Parametrize the test over `from_json(legacy_json)` and an
`ImagePipeline.from_json` payload.

### F6: LOW. MeasureColor imports a private helper across packages

`_measure_color.py:26-30` imports `_delta_e` and `candidate_medoid` from
`phenotypic.util._robust_color_stats`. `candidate_medoid` is now exported from
`phenotypic.util`, which is what the lines just above import from. `_delta_e` is
underscore-private in a private module. The project rule is that only `__init__` exports are
public. This is within the library, so it is not a user-facing breach, but it couples `measure/` to a
private name.

**Fix.** Either import `candidate_medoid` and `DEFAULT_MEDOID_CANDIDATES` from `phenotypic.util`,
or add a small public `delta_e2000_from(center, points)` helper.

---

## Suspicions (not confirmed)

### S1: LOW. The foreign-pixel assertion in `test_medoid_uses_only_the_objects_own_pixels` is likely vacuous

The final `assert not np.any(np.all(np.isclose(foreign_lab, measured ...)))` can only
fail if the measured medoid *is* a blue pixel. In both leakage mutations (`bbox`,
`anyobj`), 314 blue pixels sit among about 4 400 tan pixels, so the medoid stays tan. The failure
almost certainly came from the preceding `assert_allclose`.

I did not confirm which line failed. Either way, the test's leakage guard rests on the exhaustive-equality assertion, and the
foreign-pixel line adds a false sense of a second check.

### S2: INFO. The medoid's candidate seed is not the reported `GeoMedian`

`candidate_medoid` seeds with its own Weiszfeld run (`GEOMEDIAN_MAX_ITER=200`,
`GEOMEDIAN_TOL=1e-6`, `_robust_color_stats.py:25-26`). MeasureColor's reported
`*GeoMedian` columns use `geomedian_max_iter=50` and `tol=1e-4`. This is correct for requirement 1,
because calibration and colony use the same seed. However, the new `ColorLab` `desc` text ("candidate pixels nearest
the object's L*a*b* geometric median") and the `medoid_candidates` docstring could be read
as the reported geometric median. It affects only which pixels are candidates, not the answer on
unimodal clouds.

---

## Test able-to-fail audit

| Test | Can it fail if the requirement breaks? | Evidence |
|---|---|---|
| `test_medoid_uses_only_the_objects_own_pixels` | Yes, for leakage (bbox, neighbour) and for the random subsample. **No** for a different deterministic estimator (F3). | Ran mutations |
| `test_medoid_is_deterministic` | Yes, against the seeded subsample, because the pixel-shuffle half fails it. By design it cannot catch a different deterministic estimator. | Ran mutation `subsample` |
| `test_legacy_medoid_fields_still_load` | Yes if no warning is raised or the fields are kept (`pytest.warns` and `not hasattr`). **Cannot** detect that the warning is invisible to users (F1). Covers the dict path only (F5). | Reading, plus F1 run |
| `test_current_fields_load_without_a_deprecation_warning` | Yes if the validator warned unconditionally, via `simplefilter("error")`. | Reading |
| `test_serialization_roundtrip` (updated) | Yes if `medoid_candidates` is not serialized, because 64 would become 256. | Reading |
| `util/test_robust_color_stats.py::test_candidate_medoid_is_exported_from_util` | Yes if `_checker_measure` held its own copy (`is` identity). Says nothing about MeasureColor's call (F3). | Reading |

---

## Checked and clean

- **The move is verbatim.** Code extracted from `git show 14c5f2758` and from the new
  `util/_robust_color_stats.py` is identical after CR stripping, for `_delta_e`, `MedoidResult`,
  `candidate_medoid` and all four constants. The only difference is two blank separator lines between the
  constants. (The worktree files are CRLF while the index is LF, `git ls-files --eol`.
  That comes from the checkout, not this change.) Confirmed by running.
- **Lazy imports.** `colour` is still imported inside `_delta_e`, and
  `tests/unit/ci/test_deferred_imports.py` registers the new site.
  `_checker_measure` and `_calibrate_color_rpcc` import from the util module, and
  `test_checker_measure.py` imports `candidate_medoid` through the `_checker_measure`
  re-export, which works. No old-location import is left broken. Confirmed by reading; the ci suite was
  already run by the caller.
- **Requirement 2 in MeasureColor** (`_measure_color.py:148-165`). `labels` comes from
  `np.unique(objmap)` minus 0, so every iterated label has at least one pixel, and `find_objects` never
  returns `None` for it. A label gap is only skipped. `objmap[sl] == label` removes both background and
  neighbours. The HSV row receives the same masked slice, and that code is unchanged in the diff.
  Confirmed by running: label gap {1, 3, 7}, a one-pixel object (medoid equals the pixel, spreads 0.0,
  hex non-empty), and a two-pixel object all measure without warnings.
- **Empty input.** Unreachable from MeasureColor. It is still guarded: `candidate_medoid` returns
  index −1 and NaN Lab, the `_delta_e` guard yields `np.empty(0)`, `delta_e2000_spread` gives NaNs,
  and `lab_to_srgb_hex(NaN)` gives `""`, which is covered by the existing test. Confirmed by running (`candidate_medoid(empty)`)
  and by reading.
- **Requirement 2 in calibration.** `measure_tile` (`_checker_measure.py:218-220`) feeds
  exactly the reshaped core-box patch. Confirmed by reading.
- **Requirement 1.** Both call sites call the same `phenotypic.util._robust_color_stats.candidate_medoid`
  with the default k=256. Confirmed by reading. There is no test for the MeasureColor side (F3).
- **Legacy load paths.** `from_json`, constructor kwargs, `model_validate_json` and
  `ImagePipeline.from_json` all load. `model_validate(instance)` passes the instance through without a warning,
  which is correct. Confirmed by running.
- **pytest `filterwarnings`** (`pyproject.toml:231-233`) holds only
  `ignore::SyntaxWarning:mahotas`. There is no `error` escalation, so a stray legacy load in another
  test would show in the summary, not error. Confirmed by reading.
- **No leftover references** to `medoid_max_pixels` or `random_seed` for MeasureColor in
  `docs/source` or `src/phenotypic/_gui`. Confirmed by grep.

## Not checked

- `tests/migration` golden drift. The plan says it will move and is already red.
- Whether the tune annotation-coverage gate or the GUI builder schema cache has
  `MeasureColor` field names baked in outside `src/phenotypic/_gui` and `docs/source`.
- Calibration end to end on a real checker frame after the move. I checked the call site by reading only.
- Windows and Linux. The memory figures in F2 are from macOS `tracemalloc`, which counts numpy
  allocations. Real RSS will be at least as high.
- Behaviour of the widened (4k) pass on real sectored colonies: time about 4× the unwidened pass, not measured.
- mypy, ruff and the affected-surface run. The caller already did these.

---

## Disposition (coordinator, 2026-09-21)

Applied test-first in `d8aea8f8e..752d35d86`. Afterwards, correction, measure,
util, schema, ci and the two smoke files gave 1295 passed. The only failures
are the 3 pre-existing `FilFinderDetector` smoke cases. mypy is unchanged at
10 errors.

| ID | Outcome |
|---|---|
| F1 | Fixed. The warning is now a `FutureWarning`. A subprocess test with default filters asserts that it reaches stderr. |
| F2 | Fixed up to about 1M object pixels. The block size is derived from a 256 MiB budget at a measured 265 B per candidate-pixel pair, and the result is bit-identical for every block size (tested). Peak at 100k px fell from 1615 MiB to 256 MiB. **Residual:** above about 1M px the block floors at a single `(1, N)` row, about 1 GB for a 4-megapixel object. Bounding that as well would mean chunking along the pixel axis too. |
| F3 | Fixed. A spy pins the call site: one call per label, the configured `k`, and exactly `objmap == L` pixels. A two-lobe fixture pins the estimator, with a control showing that the pixel nearest the geometric median differs. Both fail under call-site mutations (`k=1`, nearest-geometric-median pick, whole bounding box). |
| F4 | Fixed. `MedoidCandidates = Annotated[int, Field(ge=1)]` is used by both operations. |
| F5 | Fixed. The legacy-load test is parametrised over dict, `MeasureColor.from_json` and `ImagePipeline.from_json`. |
| F6 | Fixed. The medoid is imported via `phenotypic.util`; the deltas use a function-local `colour` import, which the deferred-imports allow-list registers. |
| S1 | Removed the vacuous assertion. |
| S2 | Docstring and `desc` now state that the candidate centre is not the reported `*GeoMedian`. |

**Found during the gate, not caused by this change:**
`tests/unit/tune/test_annotation_coverage.py::test_uncovered_is_subset_of_allowlist`
fails on 24 numeric fields from the original `CalibrateColorRpcc` feature
commit: `CheckerLattice.*`, `ColumnLattice.*`, `CheckerRoi.*`, `QcLimits.*`,
`QcRecord.roi_index`, `CalibrateColorRpcc.degree` and `min_patches`. Neither
`medoid_candidates` field is listed.
