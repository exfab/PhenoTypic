# Core fixes gate review — `1d6bffbe`, `39fe65ed`, `24b6fece`

Scope: `git diff 7f1e0021..24b6fece -- src tests CLAUDE.md` (colour-config propagation,
metadata-literal allowlist, single-channel normalisation). The evidence comes from reading the
code, from the probe `.scratch/review_core_fixes_probe.py` (run by the orchestrator, output quoted
verbatim below), from a baseline run of the two new test files (59 passed), and from three
mutation runs.

## Verdict: **fix first**

Two defects need fixing before this merges. Both fixes are small.

1. **F1.** A single-channel integer input whose dtype is not `uint8` or `uint16` is divided by
   its own dtype's maximum. The result is a near-zero image that looks correct and raises
   nothing, with only a misleading bit-depth warning.
2. **F2.** The loaders rebuild stored state through the user-facing constructor. As a result,
   a stored float gray layer outside `[0, 1]` now **cannot be loaded or migrated**. This also
   turns the documented `PadImage(constant_value=255)` usage on gray-only images into a hard
   failure at the next crop or store reload.

The `uint8`/`uint16` normalisation and the colour-config propagation are correct, and their
tests are sound.

---

## Findings (most severe first)

### F1 — HIGH — wide or signed integer single-channel input is silently scaled to ~0

- **Where:** `src/phenotypic/_core/_image_parts/_image_data_manager.py`, in
  `normalize_integer_matrix` (`out /= np.float32(np.iinfo(arr.dtype).max)`) and in
  `_handle_array_input`'s `single_channel` branch.
- **Mechanism:** every `np.integer` dtype is divided by `iinfo(dtype).max`. That divisor is
  right only for `uint8` and `uint16`. For `int64` it is 9.2e18, for `int32` it is 2.1e9 and
  for `uint32` it is 4.3e9. Meanwhile `_infer_bit_depth` warns "unknown dtype … Defaulting to
  16", so the image is *labelled* 16-bit while the data were divided by 2^63−1.
- **Failure scenario:** a user builds `Image(np.array([[...]]))` from a literal, or
  `Image(np.random.randint(0, 256, (H, W)))`. Both produce int64 by default. A 32-bit integer
  TIFF read through `ski.io.imread` produces an int32 or uint32 array. The gray layer comes
  out around 1e-17 or 1e-6. Detection may still run, but `Intensity_*`,
  `Size_IntegratedIntensity` and every enhancer with an absolute parameter are meaningless,
  and nothing raises. Before this change the raw integers were kept: wrong scale, but
  obviously so.
- **Evidence (probe, verbatim):**
  ```
  [dtype int64 100/200] OK: bit_depth=16 gray=float32[1.08e-17,2.17e-17] ...
  [dtype int32 10000/30000] OK: bit_depth=16 gray=float32[4.66e-06,1.4e-05] ...
  [dtype uint32 10000/30000] OK: bit_depth=16 gray=float32[2.33e-06,6.98e-06] ...
  [dtype int16 100/200] OK: bit_depth=16 gray=float32[0.00305,0.0061] ...
  [python-literal list->np.array int64] OK: bit_depth=16 gray=float32[0,2.76e-17] ...
  [int16 with negative] RAISES ValueError: Single-channel float array contains values outside [0, 1] range. Min: -0.00015259254723787308 ...
  [uint32 retain_original exact] OK: dtype=uint32 equal=False max=1  warnings=[..., 'invalid value encountered in cast']
  ```
  Three further problems follow from the same branch:
  - `_retain_original` is **not exact** for uint32. `4294967295` is not representable in
    float32, and the cast overflows. The probe recovered `max=1` from a 4294967295 colony.
  - A negative signed input is refused with a message about a *float* array that tells the
    user to "pass the integer array", which is exactly what they passed.
  - Mutation **M-C** (divide by `255 if uint8 else 65535`) **survived** the new suite, so no
    test covers any dtype other than uint8 or uint16.
- **Fix:** in `_handle_array_input`, accept `uint8` and `uint16` as they are. For any other
  integer dtype:
  - If `arr.min() >= 0` and `arr.max() <= 2**bit_depth - 1`, where `bit_depth` is explicit or
    inferred (16), cast to `uint8` or `uint16` first. Then normalise, and record that dtype as
    `_gray_source_dtype`.
  - Otherwise raise an integer-specific `ValueError` that names the dtype and the observed
    range.

  Add parametrized tests for int64 literals, int32, uint32, int16 (including negatives), int8
  and bool, and pin each outcome (normalised value or refusal message). The RGB path has the
  same pre-existing behaviour, because `rgb2gray` → `img_as_float` also divides int64 RGB by
  2^63−1. It is out of scope here but worth a follow-up issue.

### F2 — MEDIUM-HIGH — loaders refuse stored float gray outside [0, 1]: no load, migrate or crop

- **Where:**
  - `_image_io_handler.py`: `_load_from_store` (`img = cls(arr=matrix_data, **kwargs)`,
    gray-only branch), `_load_v2_grouped`, `_load_legacy_flat_group`, and `load_pickle`
    (`target_class(arr=loaded["_data.gray"], ...)`).
  - Every one of these routes stored state through `_handle_array_input` →
    `_require_unit_range`.
  - Crops go through the same check: `ImageHandler.__getitem__` and
    `ImageGridHandler.__getitem__` both call `Image(arr=self.gray[key], …)`.
- **Failure scenarios:**
  1. **`PadImage(constant_value=255)` on a gray-only image.** The docstring recommends this
     value, "Use … 255 for white borders matching bright-agar backgrounds".
     `_image_padder.py:237-249` writes the raw 255 into the float `gray` and `detect_mat`. The
     next crop fails, whether it comes from `CropImage` after pad-and-rotate,
     `grid.grid[i]` or `objects[i]`. So does every `load_zarr` of the checkpoint store. In a
     staged GPU run that means Stage 2 (`_cli_staged_workers.py:456`) and Stage 3
     (`:516,:543`), plus `--mode measure`, recompile and the GUI. Before `24b6fece`, an
     8-bit gray image had raw-integer gray, so 255 was in range and the pipeline ran.
     - RGB images already failed at load before this change, through the gray setter's
       `assert 0 <= value <= 1` (`_grayscale_accessor.py:128-131`).
     - So the defect sits in PadImage and is pre-existing. **What is new is that it now
       reaches gray-only inputs, the population this change makes usable.**
  2. **Pre-fix stores, HDF files and pickles of a 2-D float input outside `[0, 1]`.**
     Construction accepted these before the fix; an example is a 32-bit float TIFF scan in
     counts.
     - `Image.load_zarr` now refuses them, as do `--mode measure` and recompile.
     - **`--mode migrate` refuses them too.** `_load_hdf5_for_migration` →
       `_load_from_hdf5_group` → `cls(arr=matrix_data)`, so a legacy full-run tree with such
       images can no longer be migrated. That makes it unreadable by every other mode, which
       refuse unconverted trees.
     - The migrate claim is inferred from the code path; it uses the same mechanism as the
       probed `load_zarr` case.
     - Process-mode `--layer gray` stores of such inputs are also refused by `Image.imread`.
       Process output is documented as valid CLI input.
     - I cannot tell whether such trees exist in the wild. PhenoTypic's own RGB-input stores
       are unaffected, because their gray comes from `rgb2gray`. Gray-only float-TIFF
       datasets would be affected.
- **Evidence (probe, verbatim):**
  ```
  [gray-only PadImage(255) -> crop] RAISES ValueError: Single-channel float array contains values outside [0, 1] range. Min: 0.3921568691730499, Max: 255.0 ...
  [gray-only PadImage(255) -> save2zarr -> load_zarr] RAISES ValueError: ... Max: 255.0 ...
  [RGB PadImage(255) -> save2zarr -> load_zarr] RAISES AssertionError: gray values must be between 0 and 1
  [legacy float-0..255 gray store -> load_zarr] RAISES ValueError: ... Min: 100.0, Max: 200.0 ... or pass the integer array, which is normalis
  ```
  On the store path the refusal message is also misleading: it tells the user to pass an
  integer array to a call that was given a file path.
- **Fix:**
  - **(a)** Restore stored state without the user-input range check. For example, a private
    keyword or helper (`_restore_gray(arr)`) that runs `_set_from_array` directly. Integer
    legacy layers are still normalised, and a float stored gray is taken as written, perhaps
    with a warning when it lies outside `[0, 1]`.
  - **(b)** Crops copy the source's float gray as it is. They should slice `_data` instead of
    re-validating it as user input.
  - **(c)** Fix `PadImage` so that a constant fill is expressed per layer: integer RGB keeps
    `constant_value`, and float `gray`/`detect_mat` get `constant_value / (2**bit_depth - 1)`,
    or the docstring gives the value in `[0, 1]` units.
  - At minimum, do (a), and do (c) before gray-only inputs are advertised as supported.
  - Tests: PadImage(255) on a gray-only `uint8` image → crop → `save2zarr` → `load_zarr`;
    and `load_zarr` of a store with float gray in 0..255.

### F3 — MEDIUM — `bool` single-channel input still yields bool gray/detect_mat

- **Where:** `_handle_array_input`: `np.issubdtype(np.bool_, np.integer)` is False, so a bool
  array is neither normalised nor range-checked. `ImageData.__setattr__` coerces only
  floating arrays.
- **Evidence:**
  `[bool 2-D] OK: bit_depth=16 gray=bool[0,1] detect=bool[0,1]  warnings=['Input image has unknown dtype ...']`.
- **Impact:** this behaviour is pre-existing. But the new docs now state "float32 in `[0, 1]`
  for **every** input" (`_core/CLAUDE.md`, the `ImageData` docstring). After the
  `InoculumDetector` workaround was removed, `InoculumDetector` and every other consumer that
  assumes float32 receives bool. The old workaround would also have failed on bool, since
  `np.iinfo(bool)` raises, so this is not a regression; it is a hole in the new contract.
- **Fix:** cast bool to float32 (0.0/1.0) in `_handle_array_input`, or refuse it. Add bool to
  the F1 parametrization.

### F4 — LOW — `_gray_source_dtype` is not restored on load, and is lost in crops

- **Where:**
  - `_load_from_store` restores `_original` but not `_gray_source_dtype`.
  - Crops build from float gray, so their `_gray_source_dtype` is `None`.
- **Evidence:**
  ```
  [post-fix gray-only save->load_zarr round trip + original] OK: gray_equal=True gray_dtype=float32 original_ok=True src_dtype_after_load=None re-retain dtype=float32
  [crop of int-sourced gray keeps source dtype for original?] OK: crop original dtype=float32
  ```
- **Impact:** latent only. Today both `_retain_original` call sites
  (`_cli_process_single.py:316`, `_cli_staged_workers.py:345`) act on a freshly `imread`
  image. A future caller that re-retains after a reload would silently switch the stored
  `original` series from uint8/uint16 to float32.
- **Fix:** in `_load_from_store`, when the image is gray-only and the restored `_original` is
  integer, set `img._gray_source_dtype = img._original.dtype`. Alternatively, document
  `_retain_original` as valid only on a freshly decoded image.

### F5 — LOW — the stale `_gray_source_dtype` reset is untested (M-A survived)

`_handle_array_input` resets `self._gray_source_dtype = None` before branching. Deleting that
line (**M-A**) left all 59 tests green. Scenario: an image built from `uint8` gray is then
`set_image(float_gray)`, and `_retain_original` rebuilds `rint(gray*255).astype(uint8)` from
float data. There is no in-tree caller today; `ColorDenoise` calls `set_image` on RGB only. Add
a test: `Image(uint8)` → `set_image(float32 in [0,1])` → `_retain_original()` → assert the
dtype is float32.

### F6 — LOW — NaN passes the single-channel range check

`_require_unit_range` uses `float(arr.min())`, and with NaN present both comparisons are
False. Probe: `[float NaN 2-D] OK: ... gray=float32[nan,nan]`. The RGB check behaves the same
way, so this is pre-existing in kind. Refuse non-finite values with `np.isfinite(arr).all()`,
or document that NaN passes.

### F7 — LOW — the docs and change notes over-claim slightly

- "Stores/HDF/pickles written before this hold integer layers that are normalised on load"
  (`_core/CLAUDE.md`). `Image.load_layer_zarr` (the GUI tile path, `_image_io_handler.py:2015`)
  and the GUI's direct Viv chunk reads still return the raw legacy integers. That is correct
  for a raw reader, but the sentence should name the exceptions.
- The INTENSITY/TEXTURE/SIZE notes describe the divisor as "255 or 65535". That holds only
  once F1 is fixed.
- Only SIZE's note says "Segmentation can change too". Shape, BBOX extents and zone radii and
  areas (`_zone_segmentation.py`, which fits on `detect_mat` and `gray`) can also change for
  single-channel runs, but their schema classes carry no note. These columns change only
  through segmentation; that the zone fits are sensitive to scale is unverified.
- `set_image(other_image)` now adopts `other`'s colour configuration. Probe:
  `a after set_image(b): illuminant=D65 gamma=GammaEncoding_sRGB` for a D50/linear `a`. This
  is a public-API behaviour change, but the `set_image` docstring does not mention it.

### F8 — LOW, known — `load_pickle` still resets colour config

Probe: `[load_pickle keeps D50/linear (known out of scope)] OK: illuminant=D65 gamma=GammaEncoding_sRGB`.
The commit records it as out of scope. It is now the only constructor path that drops the
config, which makes it more surprising. File it as a tracked issue.

---

## Tests that pass without proving the behaviour, and surviving mutations

Baseline: `59 passed in 1.21s` (the two new files).

| Mutation | Result | Reading |
|---|---|---|
| **M-A**: delete the `self._gray_source_dtype = None` reset in `_handle_array_input` | **survived** (59 passed) | Genuine gap. See F5. |
| **M-B**: `_retain_original` uses `scaled.astype(dtype)` (no `np.rint`) | **survived** | **Equivalent mutant**, not a weak test. The exhaustive `test_retained_original_is_exact_for_every_uint16_value` shows that in IEEE float32, `(float32(x)/65535)*65535` rounds back to exactly `x` for every `uint16` `x`, so `rint` is a defensive no-op for `uint8`/`uint16`. Keep `rint`, which matters if `gray` was modified before retention. The test is still the right anchor. |
| **M-C**: divisor `255 if uint8 else 65535` instead of `iinfo(dtype).max` | **survived** | Genuine gap: no test covers any integer dtype other than `uint8`/`uint16`. See F1. |

Other observations:

- `test_imread_of_a_legacy_integer_store_is_normalised` asserts only `gray`, which is correct
  for `imread`. Legacy normalisation of `detect_mat` is covered by the `load_zarr`, HDF and
  pickle tests, all of which assert both layers through `_assert_unit_float`. Removing
  `normalize_integer_matrix` from any one loader writes raw 100–200 values into the float32
  buffer and fails the range assertion. I reasoned this from the code and did not run it as
  a mutation.
- `test_crop_keeps_detect_mode` checks only the inherited `detect_mat` slice. The probe
  confirmed that `crop.detect_mat.reset()` under LabL recomputes identically, so the crop
  carries the D50/linear config. That assertion is worth adding to the test, because a crop
  whose mode is right but whose config is wrong would pass today.
- `test_grid_image_copy_does_not_share_the_grid_finder` cannot tell
  `model_copy(deep=True)` from a shallow `model_copy()`. The in-tree finders are flat
  pydantic models, so the two are equivalent today.
- **Missing tests** (each tied to a finding above):
  - non-`uint8`/`uint16` integer dtypes and bool (F1, F3);
  - PadImage(255) on gray-only → crop / `load_zarr` (F2);
  - `load_zarr` of a stored float gray outside `[0,1]` (F2);
  - `set_image` stale dtype (F5);
  - NaN (F6).

---

## Verified OK

- **uint8/uint16 normalisation.** `HxW` and `HxWx1` both give float32 `[0,1]` by the dtype
  maximum. `bit_depth` is inferred before normalisation, and an explicit `bit_depth` stays a
  label only. `HxWx1` float now stays float; previously it went through
  `_convert_float_array_to_int` to integer gray. All covered by the new tests.
- **No double normalisation.** Only integer dtypes are normalised, and a post-fix store holds
  float32 `gray`/`detect_mat` (`_write_store_part` writes `self.gray[:]` as is). Probe:
  post-fix gray-only `save2zarr`→`load_zarr` `gray_equal=True`, `imread` of the store
  `equal=True`, pickle round trip `equal=True`.
- **`_original` round trip.**
  - Exact for every `uint16` value; uses the source dtype rather than the `bit_depth` label.
  - Carried through `copy()` and `GridImage(Image(uint8))` (probe `dtype=uint8 equal=True`)
    and through `GridImage.imread` of a 16-bit PNG (probe `dtype=uint16 equal=True`).
  - The store's `original` series stays integer and is restored by `load_zarr`
    (`original_ok=True`).
- **Legacy integer layers are normalised on load** by `load_zarr`, `imread(store)`, HDF v1
  flat, HDF v2 grouped and `load_pickle`.
- **Detection modes:** every `compute` and `compute_from_rgb` returns float32. RGB input is
  byte-identical, which `test_rgb_input_is_unchanged` asserts.
- **Store rendering:** `build_omero` emits no window for float series
  (`sdk_/ngff_.py:1302`), so a float32 gray-only process store does not render black.
- **Process-mode fence:** `PROCESS_LAYER_SEMANTICS_REVISION` is 4, and its test is updated.
  The full and measure continuation is fenced only by the change notes, as the user decided
  (D1-3). The residual risk stands: a resumed single-channel run mixes raw-unit and
  normalised rows silently.
- **Colour config:**
  - Covered by the new tests (`MeasureColor` via a copy equals in place): `copy()`,
    `Image(other)`, `GridImage(image)`, crops of `Image` and `GridImage`, grid sections,
    `objects[i]`, `CropImage`.
  - Explicit arguments win, including an explicit `"D65"` or `'sRGB'` over a D50/linear
    source (probe `gamma=GammaEncoding_sRGB`), and an invalid explicit value is still refused.
  - `load_zarr` restores D50/linear (probe). The colour accessor holds no cache, so adopting
    the config after construction is safe.
- **`grid_finder` deep copy:** `GridImage.copy()` and `GridImage(grid)` no longer alias the
  finder. The only in-tree caller that passes a finder explicitly is `ImageCropper`, which
  passes `arr` as a plain crop `Image` with no `grid_finder` attribute, so it is unaffected.
  No caller relies on the finder being shared.
- **Removed workarounds:** with `uint8`/`uint16` input normalised, the `InoculumDetector`
  int→float step and the `SubtractBlank` integer refusal are redundant. The rewritten
  `SubtractBlank` test checks values in both polarities.
- **Published columns whose values change for single-channel `uint8`/`uint16` input:**
  - `Size_IntegratedIntensity`: scale only.
  - `Intensity_*` and `Texture_*`: newly produced; previously the measurer raised.
  - `BBOX` intensity-weighted centroids: unchanged, because they are scale-invariant ratios.
  - Every other column changes only through segmentation (see F7).
