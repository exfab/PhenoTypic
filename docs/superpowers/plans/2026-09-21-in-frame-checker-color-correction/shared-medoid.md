# One ΔE2000 medoid for MeasureColor and CalibrateColorRpcc — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `MeasureColor` and `CalibrateColorRpcc` one ΔE2000 medoid: the deterministic `candidate_medoid`, computed only on the target object's own pixels. Then the colour a correction is calibrated on and the colour a colony is later measured as are the same statistic.

**Decision (2026-09-21, user):** option B. Move `candidate_medoid` into `phenotypic.util` and use it in both places. `MeasureColor` currently uses `medoid_ciede2000`, which picks the medoid from a seeded random subsample of at most 1000 pixels. Re-drawing that seed moves the answer by about 0.12 ΔE2000 on average and 0.81 at worst (measured in the `candidate_medoid` docstring). `candidate_medoid` scores the 256 pixels nearest the Lab geometric median against **every** pixel, with no randomness.

**Architecture:**
- `candidate_medoid` and `MedoidResult`, plus their constants, move from `correction/_color_correction/_checker_measure.py` to `util/_robust_color_stats.py`, next to `medoid_ciede2000`. `_checker_measure` imports them back, so every existing import path keeps working.
- `MeasureColor._robust_lab_row` calls `candidate_medoid` on `lab[sl][submask]`, the pixels of this label only. It already excludes neighbours that share the bounding box; a new test pins that.
- The calibration already passes only each tile's core-box pixels. A new test pins that the value reaching the fit is that tile's medoid pixel.

**Tech Stack:** Python 3.11+, numpy, colour-science, pydantic v2, pytest, `uv`.

**Spec:** `docs/superpowers/specs/2026-09-21-in-frame-checker-color-correction/README.md` (Measurement; Stage E). The `MeasureColor` robust-colour design is `docs/superpowers/specs/2026-06-10-robust-lab-color-measures-design.md`, where it exists; read it before Task 2.

## What changes for users (read before approving)

- **`MeasureColor`'s medoid columns change** (`L*Medoid`, `a*Medoid`, `b*Medoid`, `MedoidColorHex`, and the three `DeltaE2000*FromMedoid` spreads).
  - **Objects over 1000 px:** the value moves, because the result no longer depends on a random subsample. The measured seed-to-seed variation of the old method, about 0.12 ΔE2000 on average and 0.81 at worst, is the order of magnitude to expect.
  - **Objects up to 1000 px:** both methods are exhaustive over the object, so on unimodal pixel clouds they agree. On a strongly bimodal object they can differ (see `CANDIDATE_EDGE_FRACTION`).
- **Speed per object** (measured on this machine, unimodal Lab cloud):

  | object pixels | old `medoid_ciede2000` | `candidate_medoid` | ratio |
  |---|---|---|---|
  | 400 | 19.8 ms | 12.5 ms | 0.6× |
  | 900 | 91.0 ms | 26.2 ms | 0.3× |
  | 4 000 | 110.7 ms | 114.7 ms | 1.0× |
  | 10 000 | 111.7 ms | 290.4 ms | 2.6× |
  | 40 000 | 115.4 ms | 1 184 ms | 10.3× |

  Cost is linear in object size, and the old method was flat above 1000 px. Small and medium colonies get faster; very large ones get slower. That trade fits the project's "accuracy over speed" rule, but it is a real cost on plates of large colonies.
- **Fields.** `MeasureColor.medoid_max_pixels` and `MeasureColor.random_seed` stop meaning anything, and `medoid_candidates: int = 256` replaces them.
  - Saved pipelines that set the old fields must still load. A `mode="before"` validator drops them with a `DeprecationWarning`, because `extra="forbid"` would otherwise reject them.
- **`phenotypic.util.medoid_ciede2000` stays exported and unchanged.** It is public API. It is simply no longer used by `MeasureColor`.
- **`tests/migration` goldens:** any scenario that measures colour will move. That suite is already red, and `tests/CLAUDE.md` forbids regenerating goldens to go green. Do not touch them. Record the expected drift in the PR description.

## Global Constraints

- `uv run` only. Tests: `QT_QPA_PLATFORM=offscreen uv run pytest <paths> -p no:cacheprovider -q -o addopts="" -n 4`. Never use `-x`, and never run the full `tests/unit` suite here.
- Preserve each file's line endings (CRLF where the working copy is CRLF). Check `grep -c $'\r' f` against `wc -l < f`.
- **Lazy imports:** `colour` is imported inside the functions that use it. `tests/unit/ci/test_deferred_imports.py` lists, for each module, the functions allowed to import `colour`. It must list `candidate_medoid`'s helper once the helper moves.
- Operation parameters follow the **`adding-an-operation`** skill (field declaration, tune annotation-coverage gate). Use it for `medoid_candidates`.
- `MeasurementInfo` members: edit `desc` only, never `bio_desc` (root `CLAUDE.md`, Gotchas).
- `uv run ruff check <explicit paths>` only.

**DAG:** 1 → 2 → 3 → 4. Task 2 imports what Task 1 moves; Task 3 is independent of Task 2 but shares the review test file with nothing else, so it may run after 1 in parallel with 2.

---

### Task 1: Move `candidate_medoid` into `phenotypic.util`

**Files:**
- Modify: `src/phenotypic/util/_robust_color_stats.py`. Add `MedoidResult`, `candidate_medoid`, `_delta_e`, and the constants `DEFAULT_MEDOID_CANDIDATES`, `CANDIDATE_EDGE_FRACTION`, `GEOMEDIAN_MAX_ITER` and `GEOMEDIAN_TOL`, **moved verbatim** with their docstrings and comments.
- Modify: `src/phenotypic/util/__init__.py`. Export `candidate_medoid` and `MedoidResult`, and add them to `__all__`.
- Modify: `src/phenotypic/correction/_color_correction/_checker_measure.py`. Delete the moved definitions and import them from `...util._robust_color_stats`. `impurity` and `robust_shift` still need `_delta_e`, so import it too.
- Modify: `tests/unit/ci/test_deferred_imports.py:146`. Add `"_delta_e"` to the `util/_robust_color_stats.py` entry's allowed-`colour` tuple. Remove it from the `_checker_measure` entry if one exists there.
- Test: `tests/unit/util/test_robust_color_stats.py`.

**Interfaces:**
- Produces `phenotypic.util.candidate_medoid(lab_points, k=256, chunk_size=64) -> MedoidResult(index, lab, rank, total_delta_e, widened)`, with a signature and behaviour identical to today's.

- [ ] **Step 1:** Add `test_candidate_medoid_is_exported_from_util` to `test_robust_color_stats.py`. It checks `from phenotypic.util import candidate_medoid, MedoidResult`, and that `candidate_medoid is _checker_measure.candidate_medoid`, so there is one implementation and not a copy. Run it and confirm it fails with an ImportError.
- [ ] **Step 2:** Move the code. Run `tests/unit/util/test_robust_color_stats.py tests/unit/correction/test_checker_measure.py tests/unit/ci/test_deferred_imports.py tests/unit/ci/test_startup_imports.py`. Everything passes, and `test_checker_measure.py` is unchanged.
- [ ] **Step 3:** Commit: `refactor(util): move candidate_medoid to phenotypic.util for shared use`.

---

### Task 2: `MeasureColor` uses `candidate_medoid` on the object's own pixels

**Files:**
- Modify: `src/phenotypic/measure/_measure_color.py`. This covers the imports, `Args:` lines 53–55, fields 77–78, a new `mode="before"` validator, and `_robust_lab_row` (around line 136).
- Modify: `src/phenotypic/schema/_color_lab.py:26-33`. Change `desc` only: say that the medoid is found deterministically among the candidates nearest the object's Lab geometric median, scored against every object pixel.
- Test: `tests/unit/measure/test_measure_color.py`. Update lines 50–53 and add three tests.

**Interfaces:**
- Consumes: `phenotypic.util.candidate_medoid` (Task 1).
- Produces: `MeasureColor.medoid_candidates: int = 256`. The fields `medoid_max_pixels` and `random_seed` are removed but still accepted on load, with a `DeprecationWarning`.

- [ ] **Step 1: Write the tests first.**
  - `test_medoid_uses_only_the_objects_own_pixels`: two labels whose bounding boxes overlap, where label 2's pixels are a very different Lab colour and sit inside label 1's bbox. Assert that label 1's medoid equals the **exhaustive** ΔE2000 medoid of label 1's pixels alone. Write the exhaustive medoid inline in the test, independent of `candidate_medoid`. Also assert that it is not any pixel of label 2.
  - `test_medoid_is_deterministic`: two fresh `MeasureColor()` runs on the same image give identical medoid columns.
  - `test_legacy_medoid_fields_still_load`: `MeasureColor.model_validate({"medoid_max_pixels": 300, "random_seed": 3})` warns `DeprecationWarning` and builds with `medoid_candidates == 256`.
  - Replace lines 50–53 (the `medoid_max_pixels`/`random_seed` round-trip) with a `medoid_candidates=64` round-trip.

  Run them and confirm the new tests fail.
- [ ] **Step 2: Swap the estimator.** In `_robust_lab_row`:

  ```python
          result = candidate_medoid(lab_px, k=self.medoid_candidates)
          medoid = result.lab
          deltas = (
              _delta_e(np.broadcast_to(medoid, lab_px.shape), lab_px)
              if lab_px.shape[0] else np.empty(0)
          )
  ```

  Import `candidate_medoid` and `_delta_e` from `phenotypic.util._robust_color_stats`. `lab_px` is already `lab[sl][submask]`, this label's pixels only. Do not widen it.
- [ ] **Step 3: Fields.** Delete `medoid_max_pixels` and `random_seed`. Add `medoid_candidates: int = 256`, with an `Args:` entry that explains the candidate set, the automatic widening, and why it is deterministic, following the `adding-an-operation` skill for tuning annotation. Add:

  ```python
      @model_validator(mode="before")
      @classmethod
      def _drop_legacy_medoid_fields(cls, data: Any) -> Any:
          """Accept pipelines saved before the medoid became deterministic.

          ``medoid_max_pixels`` and ``random_seed`` configured a random
          subsample that no longer exists.  Dropping them (with a warning)
          keeps old JSON loadable under ``extra="forbid"``.
          """
          if isinstance(data, dict):
              legacy = [k for k in ("medoid_max_pixels", "random_seed") if k in data]
              if legacy:
                  warnings.warn(
                          f"MeasureColor ignores {', '.join(legacy)}: the ΔE2000 "
                          "medoid is now deterministic (candidate_medoid). Use "
                          "medoid_candidates to size the candidate set.",
                          DeprecationWarning, stacklevel=2,
                  )
                  data = {k: v for k, v in data.items() if k not in legacy}
          return data
  ```
- [ ] **Step 4:** Update the class docstring's description of the medoid. Update the `ColorLab` `desc` strings, touching `desc` only.
- [ ] **Step 5:** Run `tests/unit/measure tests/unit/util tests/unit/schema tests/smoke/test_serialization.py tests/smoke/test_operation.py -k "Color or color or medoid or schema"`. The only acceptable failures are the known `FilFinderDetector` smoke tests.
- [ ] **Step 6:** Commit: `feat(measure): MeasureColor uses the deterministic candidate medoid on object pixels`.

---

### Task 3: Pin that the calibration fits each tile's medoid pixel

**Files:**
- Test: `tests/unit/correction/test_calibrate_color_rpcc_review.py`. `test_the_fit_consumes_each_tiles_medoid_pixel` is written but uncommitted.

The main assertion already passes, and mutation-tests correctly: handing the fit the core mean or the per-channel median both fail it. Its control assertion is too loose. The control currently requires the core mean to lie farther from the medoid than the *farthest* clean pixel, and with 6/255 noise the farthest clean pixel is itself 8.5 ΔE out.

- [ ] **Step 1:** Replace the control with a comparison of means. The core mean must lie farther from the medoid than the **clean-only** mean does:
  `ΔE(mean(all core), medoid) > ΔE(mean(clean core), medoid) + ΔE(mean(clean core), medoid)`.

  The clean mean's distance to the medoid is the noise floor of a mean. The occluder must at least double it. Run the test and confirm it passes.
- [ ] **Step 2:** Mutation check. Revert the core to its mean and to its per-channel median in `measure_tile`. Both must fail the test. Restore.
- [ ] **Step 3:** Add one assertion that the tile's pixels are its core box only: `seen[name]` must be one of `core_srgb`'s rows, the object's own pixels.
- [ ] **Step 4:** Commit: `test(color): pin that the fit consumes each tile's medoid pixel`.

---

### Task 4: Gate

- [ ] Run `tests/unit/correction tests/unit/measure tests/unit/util tests/unit/schema tests/unit/ci tests/smoke/test_operation.py tests/smoke/test_serialization.py` once. The only acceptable failures are the 3 known `FilFinderDetector` smoke cases.
- [ ] Run ruff on the changed paths, and `uv run mypy src/phenotypic/util src/phenotypic/measure/_measure_color.py src/phenotypic/correction/_color_correction/`. There must be no new errors.
- [ ] Search `docs/source` for text describing `MeasureColor`'s medoid as subsampled or seeded (`grep -rn "medoid" docs/source`), and correct it.
- [ ] Have a reviewer check whether the tests prove the change; its report goes to `docs/superpowers/reports/2026-09-21-in-frame-checker-color-correction/shared-medoid-review.md`.
