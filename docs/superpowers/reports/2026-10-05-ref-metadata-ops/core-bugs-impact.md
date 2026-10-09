# Core bugs: impact analysis and fix options

**Date:** 2026-10-05. **Tree:** `worktree-ref-metadata-ops` at `25276e40`.
**Scope:** analysis only. Nothing under `src/` or `tests/` was edited.
**Inputs:** `phase1-review.md` F1/F2, spec §12 "Pre-existing core issues".
**Evidence:** `file:line` against this tree, plus two executed probes:
`.scratch/core_bugs_probe.py` (behaviour) and `.scratch/gray_census_plugin.py`
(which tests build single-channel integer images). Probe results are quoted
verbatim in §3.

---

## Summary

| | Bug 1: single-channel integer gray | Bug 2: `Image.copy()` drops colour config |
|---|---|---|
| Recommendation | **Option A**: normalise at construction | **Option B2**: carry config in `_set_from_class_instance`, sentinel defaults |
| User-visible? | Yes, for single-channel integer inputs only | Yes, for non-default (D50 / linear) images in notebooks only; the CLI never sets them |
| Fencing | Change note on `INTENSITY` + `SIZE`; bump `PROCESS_LAYER_SEMANTICS_REVISION` 3→4; work-id fence for single-channel inputs (decision D1-3) | None needed |
| RGB inputs | Byte-identical | Byte-identical |
| Migration goldens | Unaffected (RGB uint8, D65/sRGB inputs only, `tests/migration/_inputs.py:80-123`) | Unaffected |

---

## 1. Bug 1: single-channel inputs break the [0, 1] float contract

### 1.1 The contract and where it breaks

The contract is stated in five places:

- `ImageData` docstring, `_image_data_manager.py:22-29`: gray and detect_mat are
  normalized `[0, 1]`.
- `Grayscale` docstring and `vmin`/`vmax`, `_grayscale_accessor.py:20-21,145-153`.
- `DetectionMode.compute`, `_detection_mode.py:41`: "A 2-D float32 array normalised to [0, 1]".
- `DetectMatAccessor.vmin`/`vmax`, `_detect_mat_accessor.py:129-137`.
- The CI gate `tests/unit/enhance/test_detect_mat_invariant.py`, which checks only
  the RGB synthetic plate.

It breaks at three points:

1. **`_handle_array_input`, `_image_data_manager.py:286-294`.** It converts only
   `arr.ndim == 3` floats, and those go to *integer* RGB. A 2-D integer array goes
   through unchanged. So does a 2-D float array, including values outside [0, 1].
2. **`_set_from_matrix`, `:389`.** It runs `self._data.gray = matrix`. The `ImageData.__setattr__`
   coercion (`:43-58`) downcasts only floats, by design ("Non-float arrays ... pass
   through untouched").
3. **`GrayDetectionMode.compute`, `_gray_mode.py:30-32`.** It returns `image._data.gray.copy()`,
   so detect_mat inherits the integer dtype.

By contrast, the RGB path is normalised: `_set_from_rgb` (`:404-411`) calls
`rgb2gray`, which divides by the **dtype max**. Every RGB-derived mode divides by
the dtype max as well (`normalize_rgb_bitdepth`, `sdk_/funcs_.py:286-288`).

### 1.2 Every path a single-channel input takes

| Entry | Code | `gray` | `detect_mat` | `rgb` |
|---|---|---|---|---|
| `Image(arr=<2-D uint8/16>)` | `_handle_array_input` → `_set_from_array` → `_set_from_matrix` | raw int (uint8 0–255 / uint16 0–65535) | same raw int (gray mode) | empty `(0,3)` |
| `Image(arr=<H,W,1>)` | `_set_from_array` `GRAYSCALE_SINGLE_CHANNEL` → `arr[:,:,0]` (`:388-389`) | raw int | raw int | empty |
| `Image(arr=<2-D float>)` | float, no range check | float32, **any range** | same | empty |
| `Image(arr=<3-D float>)` | `_convert_float_array_to_int` (`:461-502`) | rgb2gray [0,1] | [0,1] | int; **refuses** outside [0,1]; truncates (`astype`), does not round |
| `Image.imread(png/jpg/tif)` | `ski.io.imread` (`_image_io_handler.py:787`) → `cls(arr=…)` (`:796`). JPEG forces `bit_depth=8` (`:793-794`) | as the decoded dtype: 8-bit PNG/JPEG→uint8, 16-bit PNG/TIFF→uint16, float TIFF→float (any range) | same | empty |
| `Image.imread(RAW)` | rawpy `postprocess(output_bps=16, sRGB)` (`:771-784`) | always 3-channel → RGB path, never affected | | |
| `Image.imread(<store>)` | `_imread_store` → `read_ngff_image_spec` → `cls(arr=spec.array)` (`:883-891`) | primary series dtype: a PhenoTypic gray-only store holds raw int today (§3, B1.10) | same | empty |
| `Image.load_zarr(store)` | `_load_from_store` (`:1735-1748`) → `cls(arr=gray)`, then `detect_mat[:] = stored` | stored dtype | **stored array written into the freshly computed buffer** (`_detect_mat_accessor.py:126`, in-place) | empty |
| `load_pickle` | `target_class(arr=loaded["_data.gray"])` (`:2184-2189`) | stored dtype | stored | |
| `GridImage(...)` | same chain via `ImageGridHandler.__init__` (`_grid_image_handler.py:88`) | same | same | same |
| Crop `img[...]` | `ImageHandler.__getitem__` (`_image_handler.py:104-108`): `self.__class__(arr=self.gray[key])`. `bit_depth` is re-inferred from dtype | inherits | copied | |
| CLI | `read_kwargs` carries `bit_depth`/`detect_mode` (`_cli_execution_strategies.py:472-474,596-598`, `_cli_staged_strategy.py:91-93`, `_cli_process_single.py:863-865,1002-1004`) → `Image.imread` / `GridImage.imread` | as imread | as imread | |

`bit_depth` only labels the data. Nothing in the gray path scales by it.
`_infer_bit_depth` (`:297-321`) returns 16 for **any** float input, so a
float crop of an 8-bit image is relabelled 16-bit.

### 1.3 What breaks today (consumers that assume [0, 1])

These failures happen **today** on single-channel integer inputs, before any fix.
They are the reason to fix the core rather than patch each consumer.

| Consumer | Mechanism | Effect today |
|---|---|---|
| **Every enhancer that writes `detect_mat[:] = <float [0,1]>`** (40+ files; `grep "detect_mat\[:\] *="` → 56 sites) | `DetectMatAccessor.__setitem__` assigns **into the existing buffer** (`_detect_mat_accessor.py:126`). A uint8 buffer truncates every float in [0,1) to 0 | Binary 0/1 output (B1.11) |
| `SubtractGaussian` (`enhance/_subtract_gaussian.py:94-106`) | `preserve_range=True` by default (`:89`), so the background is 0–255; `clip(d - bg, 0, 1)` is then truncated into uint8 | 0/1 (B1.11). The review's "skimage rescales" explanation applies only with `preserve_range=False` |
| `MeasureTexture` (`measure/_measure_texture.py:184-185`) | `raise ValueError("Foreground array must be normalized between 0 and 1")` | **Raises** on every single-channel integer image (Q3; float control OK, Q3b) |
| `MeasureIntensity` (`measure/_measure_intensity.py:65`) | `_calculate_q1` → `scipy.ndimage.labeled_comprehension(..., out_dtype=<input dtype>, default=NaN)` (`abc_/_measure_features.py:1207,1244`) | **Raises** `cannot convert float NaN to integer` on every integer gray (probe 3). Float and RGB inputs give identical values (0.5877) |
| `MeasureSize` `Size_IntegratedIntensity` | sum of gray | Silently in counts: 80000 vs 313.73 for the same plate as float (Q2) |
| `UserThreshold` (default `threshold=0.5`, `TuneSpec(0.0, 1.0)`, `detect/_user_threshold.py:85`) and `FilFinderDetector.threshold` (`_filfinder_detector.py:202,291`) | absolute thresholds on a [0,1] scale | The whole frame is foreground, and `clear_border` then removes it: **0 objects** (Q5; float control: 1 object, 0.098) |
| `OtsuDetector` (`detect/_otsu_detector.py:86-92`) and others that use `>= threshold_*` on integers | On integer input `threshold_otsu` returns a bin value (100 for an agar=100 / colony=200 plate, Q6), and `>=` then includes the agar | **0 objects** on the two-level probe plate (Q4; float and RGB controls: 1 object). On real scans this becomes a one-bin bias, not a total failure. So Otsu is scale-free in value but not in dtype |
| `Grayscale.__setitem__` (`_grayscale_accessor.py:122-128`) | asserts the value is in [0,1], then writes into a uint8 buffer | Any corrector writing gray (`DenoiseBlockMatch`, `correction/_denoise_block_match.py:255`) is truncated to 0/1 |
| `VisuShrink` / wavelet correctors (`correction/_visushrink_corrector.py:171-182`) | `denoise_wavelet` returns float [0,1] and **replaces** `_data.gray` | The scale silently changes mid-pipeline, from 0–255 to [0,1] |
| `SubtractBlank` | now refuses (`enhance/_subtract_blank.py:215-221`) | Refusal stands in for the missing fix |
| `InoculumDetector` | **local workaround** (`detect/_inoculum_detector.py:157-167`): replaces the buffer with `dm/iinfo.max` | Correct, but duplicates the core's job |
| Viewer / plots | `_accessor_mpl_handler.py:180-191,285-288`, `napari_/_layers.py:35`, `viv_viewer.js:265-268` pick limits by dtype | Display is fine |

Scale-free consumers (in value; see the Otsu row for the integer `>=` caveat):
Otsu-family / Li / Yen / Triangle / Isodata / Mean / Minimum thresholds; `RankOtsu` (`img_as_ubyte`); Canny with the
default `use_quantiles=True`; Chan–Vese (normalises internally); `MeasureBounds`
`centroid_weighted`; colony-ness in zone segmentation (self-normalised,
`measure/_zone_segmentation.py:1221-1224`); `Intensity_CoefficientVarianceIntensity`.

### 1.4 Options

- **A: normalise at construction.** In `_handle_array_input`, convert a 2-D (or
  `H,W,1`) integer array to `float32 / iinfo(dtype).max` **after** `bit_depth` is
  inferred from the original dtype. This is the same divisor `rgb2gray` and
  `normalize_rgb_bitdepth` use. gray and detect_mat are then float32 [0,1] for
  every input. Required companions:
  - A1 `_retain_original` (`_image_data_manager.py:370-374`) snapshots
    `gray[:]` for gray-only images. It must keep the **integer** original, or the
    store's `original` series silently becomes float32 (4× larger and no longer
    "decoded pixels"). Reconstruct it with
    `np.rint(gray * maxval).astype(uint{bit_depth})`. This is exact: float32 has a
    24-bit mantissa, so the error is ≤ 65535·2⁻²⁴ ≈ 0.004 < 0.5. Alternatively,
    snapshot the raw matrix at construction.
  - A2 a legacy-store shim in `_load_from_store` (`_image_io_handler.py:1745`)
    and `_load_v2_grouped`/`_load_legacy_flat_group` (`:1617-1632,1840-1849`). Without it, an **integer** stored
    `detect_mat` is written in place into the float buffer and comes back as
    0–255 floats. Normalise integer stored `detect_mat` by dtype max. The gray is
    normalised by the constructor automatically. Pickles are covered by the
    constructor.
  - A3 the crop and `__getitem__` paths (`_image_handler.py:104-108`,
    `_grid_image_handler.py:265-268`) must pass `bit_depth=self.bit_depth`.
    Otherwise `_infer_bit_depth(float)` relabels an 8-bit single-channel crop as 16-bit.
  - A4 delete the workarounds that become dead: `InoculumDetector` `:157-167`, and
    `SubtractBlank` `:215-221` with its test `test_refuses_a_single_channel_integer_pair`.
- **B: normalise only in `GrayDetectionMode.compute`.** It returns
  `gray / iinfo.max` for integer gray, so detect_mat becomes float32 [0,1] while gray stays raw int.
- **B′: B, plus a dtype-preserving detect_mat setter.** This is not a separate
  option. B alone leaves the gray-side failures from §1.3 in place.

### 1.5 Impact per option

| | **A (construction)** | **B (detect mode only)** |
|---|---|---|
| Files changed (src) | `_image_data_manager.py` (`_handle_array_input`, `_retain_original`), `_image_io_handler.py` (3 legacy loaders), `_image_handler.py` + `_grid_image_handler.py` (`__getitem__` bit_depth), `_inoculum_detector.py`, `_subtract_blank.py`, `_cli_failure_tracker.py` (revision), `schema/_change_notes.py` + `_intensity.py` + `_size.py` | `_gray_mode.py`, `_image_io_handler.py` (detect_mat shim), `_inoculum_detector.py`, `_subtract_blank.py`, `_cli_failure_tracker.py` |
| `detect_mat` for single-channel int | float32 [0,1] | float32 [0,1] |
| `gray` for single-channel int | float32 [0,1] | raw int (contract still violated) |
| **Published measurement columns** (single-channel int inputs only) | **Only `Size_IntegratedIntensity` changes value** (`measure/_measure_size.py:263-265`; Q2: 80000 → 313.73, i.e. ÷255). `Intensity_*` and `Texture_*` **go from raising to producing values**: `MeasureIntensity` raises today on every integer gray (Q1/probe 3, `_calculate_q1` → `labeled_comprehension` with an integer `out_dtype` and a NaN default, `abc_/_measure_features.py:1207,1244`), and `MeasureTexture` raises (Q3). Their new values equal those of the equivalent float or RGB input (probe 3: 0.5877 for all three). Zone/symzones intensity statistics on gray are rescaled where they are emitted | unchanged: `Size_IntegratedIntensity` stays in counts, while `MeasureIntensity` and `MeasureTexture` **still raise** |
| RGB inputs | byte-identical | byte-identical |
| 2-D float inputs already in [0,1] | identical | identical |
| Enhancers/detectors | all §1.3 failures fixed, because the buffer is float | detect_mat-writing enhancers fixed; gray writers (`DenoiseBlockMatch`, `VisuShrink`) still truncate/rescale |
| OME-Zarr store (single-channel input) | `gray` series uint8/uint16 → **float32**; its `omero` block disappears (`ngff_.build_omero` returns `{}` for floats, `sdk_/ngff_.py:1304-1305`); `detect_mat` series values change; `original` series unchanged with A1 | `gray` unchanged; `detect_mat` series dtype/values change |
| Store bytes | change for single-channel inputs only | change for single-channel inputs only |
| `--mode process --layer gray` (single-channel input) | tiff: uint8 → float32 TIFF; zarr: dtype change → **bump `PROCESS_LAYER_SEMANTICS_REVISION` 3→4** (`_cli_failure_tracker.py:209`) | gray unchanged; `--layer detect_mat` changes → bump too |
| Full / measure continuation | no revision constant covers it. See D1-3 | same issue, for detect_mat-derived objmaps |
| GUI | Viv picks the domain by dtype (`viv_viewer.js:265-268`): Float32 → [0,1], correct | unchanged |
| Tests | see §3 census | fewer: gray assertions untouched |

### 1.6 2-D float outside [0, 1]

Today a 2-D float passes through at **any range** (B1.4, B1.5). The 3-D float
path **refuses** outside [0,1] (`_convert_float_array_to_int`, `:487-491`,
B1.6). `normalize_rgb_bitdepth` instead *guesses* (`sdk_/funcs_.py:290-300`: max ≤1 → as-is,
≤255 → /255, ≤65535 → /65535, else raise).

**Recommendation: refuse, using the same error as the 3-D path.** A float array
carries no scale, so guessing 0–255 vs 0–65535 from the observed max is the
heuristic the integer fix exists to remove. Keep in-range floats as float32
rather than quantising them (gray is float anyway). This changes behaviour in
three places:
- `test_a_float_single_channel_pair_outside_the_unit_range_is_refused`
  (`tests/unit/enhance/test_subtract_blank.py:251-259`). It builds out-of-range
  float images, so the refusal moves from `SubtractBlank` to `Image(...)`.
  Update its expected exception.
- A user with a float TIFF scaled 0–65535 now gets an error at `imread`
  instead of a silently wrong analysis. Name the fix in the message (divide
  by the scale, or pass an integer array).
- The 3-D path truncates (`astype`) where it should round. That is a minor
  separate issue; leave it out of scope.

### 1.7 Decisions for the user

- **D1-1 A or B.** Recommend **A**. Only A honours the documented contract on
  every layer. Only A fixes `MeasureTexture`, the gray-writing correctors,
  `MeasureIntensity`, absolute thresholds, and the cross-input unit mismatch (an
  RGB scan's gray-derived columns are in [0,1] while a gray scan's are in counts).
  B leaves gray in counts, so `MeasureIntensity` and `MeasureTexture` keep
  raising. The published-output cost of A is smaller than it first appears.
  `MeasureIntensity` and `MeasureTexture` never produced values for these
  inputs, so the only previously published column that changes value is
  `Size_IntegratedIntensity` (÷255 or ÷65535). On top of that, segmentation
  changes wherever a detector or enhancer was being truncated (Q4, Q5, B1.11).
  A single-channel integer run through the default detectors produced 0
  objects on the probe plate, so in practice few trustworthy single-channel
  results can exist to be invalidated.
- **D1-2 Out-of-range 2-D float: refuse (recommended), clip, or rescale by heuristic.**
- **D1-3 Fencing full/measure continuation.** There is no full-mode semantics
  revision. The base payload deliberately excludes one, because it would
  cold-start every in-flight run (`_cli_failure_tracker.py:233-271` comments). Choices:
  - (a) Follow the **`SIZE_SHAPE_SPLIT_NOTE` precedent** (`schema/_change_notes.py:30-40`).
    Ship a `.. versionchanged::` change note on `SIZE` (`Size_IntegratedIntensity`)
    and `INTENSITY`/`TEXTURE` ("now produced for single-channel integer inputs") saying
    a single-channel run started before the fix must be re-run with
    `--overwrite`, not resumed. No code fence. Cheapest, and consistent with
    how the last public-column change was handled.
  - (b) Mirror **`RAW_DECODE_REVISION`** (`:211-219,328-329`): fold a
    `GRAY_NORMALIZATION_REVISION` into `compute_work_id` only for single-channel
    inputs. That requires a header read (`_cli/_cli_input_headers.py` already
    maps headers to the decoded channel count) inside a function both work-id
    producers call. Precise, but more code, and a header read on the work-id path.
  - (c) A universal bump. It cold-starts every in-flight run, including RGB runs
    whose outputs do not change. Not recommended.

  Recommend **(a)**, plus the process-mode revision bump, which is mandatory under
  either option. Upgrade to (b) only if single-channel runs are known to be in
  flight.
- **D1-4 Changelog.** There is no root CHANGELOG. The repo's mechanism is
  `MeasurementInfo.change_note()` → `schema/_change_notes.py` (rendered in measurer docs,
  the enum page and the Measurements reference). `SIZE` already has one
  (`schema/_size.py:25`), so its `change_note` must return both notes concatenated.

---

## 2. Bug 2: `Image.copy()` drops colour configuration

### 2.1 Mechanism

`copy()` is `self.__class__(self)` (`_image_handler.py:694-704`).
`Image.__init__` defaults are `gamma=SRGB, illuminant="D65"`
(`_core/_image.py:54-60`). `ImageColorSpace.__init__` stores them **before**
`super().__init__` → `set_image` (`_image_color_handler.py:80-94`).
`_set_from_class_instance` (`_image_data_manager.py:332-368`) copies `_data.*`
(arrays and `detect_mode`), `protected` (including bit_depth and name), `public`,
`imported`, the journal and `_original`. It never touches colour state.

### 2.2 Full inventory of per-instance state set by the constructors

| Attribute | Set at | Kept by `copy()`? |
|---|---|---|
| `_data.rgb/gray/detect_mat/sparse_object_map` | `ImageDataManager` | yes (`.copy()` each) |
| `_data.detect_mode` | `ImageData` | yes (`_data.__dict__` loop) |
| `_metadata.protected` (name, type, bit_depth) | `:159-171` | yes (deepcopy) |
| `_metadata.public`, `imported`, `provenance_journal` | | yes |
| `_metadata.private` (UUID) | | **fresh, intentional** (`:356-358`) |
| `_original` | `:172` | yes |
| `gamma` | `_image_color_handler.py:90` | **no → SRGB** |
| `illuminant` | `:91` | **no → D65** |
| `_observer` | `:93` (hard-coded, never assigned anywhere else) | reset to the same constant, so no observable loss today |
| `_accessors.*` | `_image_handler.py:76-82`, `_image_objects_handler.py:47` | rebuilt, correct (they bind `self`) |
| `GridImage._grid_finder` | `_grid_image_handler.py:90-99` | **shared by reference** (`hasattr(arr,"grid_finder")`): see 2.3 |
| `GridImage` type metadata | `:101` | reset to GRID (correct) |

### 2.3 Related defects on the same path

- **Crops lose the config too.** `ImageHandler.__getitem__` (`_image_handler.py:104-108`)
  and `ImageGridHandler.__getitem__` (`_grid_image_handler.py:265-268`) build
  `Image(arr=…)` with defaults. `ImageHandler.__getitem__` additionally **drops
  `detect_mode`**. The grid version propagates it. `objects[i]`
  (`_objects_accessor.py:219`) and grid sections (`_grid_accessor.py:531-534`) go
  through these. `ImageCropper` on a `GridImage` passes
  `illuminant=cropped.illuminant` (`correction/_image_cropper.py:166-167`): the
  value **after** the crop reset it. So it propagates D65 while appearing to
  preserve the config.
- **`GridImage.copy()` aliases `grid_finder`.** Changing `copy.nrows` mutates the
  original's finder (`GridAccessor.nrows` setter writes
  `grid_finder.nrows`, `_grid_accessor.py:100,129`). See B2.6.
- Workarounds that already pass config explicitly (and become redundant):
  `ImagePadder` (`correction/_image_padder.py:278-279`),
  `CalibrateColorRpcc` (`_calibrate_color_rpcc.py:403-405`).

### 2.4 Who calls `copy()` (blast radius)

- `ImageOperation._apply_to_single_image` `inplace=False` (`abc_/_image_operation.py:505-508`):
  every `op.apply(image)` call.
- `ImagePipeline.apply` and `apply_with_intermediates` default `inplace=False`
  (`_image_pipeline_core.py:968-988,997-1038`). Its intermediate saves are at `:1063,1066`.
  **A whole default `pipeline.apply(d50_image)` runs under D65/sRGB** (B2.4).
- `apply_child` defaults to `inplace=False` (`_core/_provenance.py:542`). Every
  `CompositeDetector`/`CompositeEnhance`/filamentous branch therefore runs on a copy.
- Internal copies: `InoculumDetector:159`, `TwoKFilamentousDetector:189`,
  `FilamentousFungiDetector:418,440`, `_canonical_zone_measure.py:280`,
  `_cli_staged_workers.py:177` (the Stage-2 prefix probe), the napari pipeline viewer,
  GUI tune overlays.
- Consumers of the lost state: `color.Lab/XYZ/xy` accessors
  (`_cielab_accessor.py:74-75`, `_xyz_accessor.py`), Lab detect modes
  (`_lab_channel_modes.py:41-63`), `MeasureColor` (`measure/_measure_color.py:159,234,246`:
  `ColorLab_*`, `ColorXYZ_*`, `Colorxy_*` columns), `ColorCheckerProfile`, and the
  SubtractBlank freshness check (F2's false `StaleDetectMatError`).

**The CLI is unaffected.** No CLI flag sets illuminant or gamma (`grep illuminant
src/phenotypic/_cli` → nothing), so every CLI image is D65/sRGB. Stores restore
the stored value (`_image_io_handler.py:1712-1733`), which is D65/sRGB for CLI
stores. The visible population is **notebook/SDK users who construct or
`load_zarr` a D50 or linear image**. For them, `ColorLab_*`/XYZ columns and
Lab-mode detection silently use the wrong configuration. The size of the error
(B2.5): `MeasureColor` on a D50/linear image measured in place vs. on its
copy differs by **15.95** in `ColorLab_L*GeoMedian` and **23.28** in
`ColorLab_a*GeoMedian`. B2.7: `load_zarr` restores D50/linear correctly, and the
next `copy()` loses it again.

**Does any caller rely on the reset?** No such caller was found. Every site that
builds a derived image either passes the source config explicitly (padder, rpcc,
cropper) or is affected by the bug. `ColorCheckerProfile` (`_color_checker_profile.py:507-511`)
sets `illuminant=self.target_illuminant` **deliberately**. It builds a fresh
`Image(arr=…)` for that, not a copy, so it is unaffected. No test asserts a
copy's D65/sRGB default (`grep` of `tests/` for copy plus illuminant/gamma: none).
`test_subtract_blank.py:218-231` uses `inplace=True` to *avoid* the bug, and keeps
passing.

### 2.5 Options

- **B1: fix `copy()` only.** `self.__class__(self, gamma=self.gamma,
  illuminant=self.illuminant)`, then `_observer`, then a fresh `grid_finder` for GridImage.
  Smallest change. It leaves `Image(other)`, crops and grid sections dropping the
  config.
- **B2: carry config in the copy constructor** (recommended). In
  `_set_from_class_instance`, copy `gamma`/`illuminant`/`_observer` from the
  source **unless the caller passed them explicitly**. That needs sentinel
  defaults in `Image.__init__`, `GridImage.__init__`, and
  `ImageColorSpace.__init__` (`_UNSET`, resolving to SRGB/D65 for array input).
  `None` cannot serve as the sentinel: `gamma=None` already means LINEAR
  (`_image_color_handler.py:81`) and `illuminant=None` is refused. One change
  then covers `copy()`, `Image(other)`, `GridImage(image)` (documented usage,
  `abc_/_grid_object_detector.py:162`), ImageCropper/Padder re-wrapping and grid
  sections. Separately, fix the two `__getitem__`s to pass
  `gamma`/`illuminant`/`bit_depth` and `detect_mode` (ImageHandler).
- **B3: explicit `__copy__`/`copy()` via `copy.deepcopy`-style cloning** instead
  of the constructor. More invasive, and diverges from `Image(other)`. Not recommended.

| | B1 | **B2** |
|---|---|---|
| Files | `_image_handler.py` (+ `_grid_image_handler.py` for the finder) | `_image_color_handler.py`, `_image.py`, `_grid_image.py` signatures, `_image_data_manager.py:_set_from_class_instance`, both `__getitem__`s, `_grid_image_handler.py` (finder copy) |
| User-visible | `op.apply`/pipelines on D50/linear images become correct | same + `Image(img)`, crops, grid sections, `objects[i]` |
| Signature change | none | the default *values* stay; only the sentinel object changes (`help()` shows `<unset>`). `model_json_schema` is not involved (Image is not pydantic) |
| CLI / store bytes | none | none |
| Tests to update | none expected | none expected (no test asserts the reset); add regression tests for copy, `Image(other)`, crop, grid copy isolation |

**GridImage finder:** use `copy.deepcopy(arr.grid_finder)` (finders are pydantic
operations, so `model_copy(deep=True)`) so that a copy's `nrows` change cannot
reach the original. Decision D2-2 below.

### 2.6 Decisions for the user

- **D2-1 B1 or B2.** Recommend **B2**. The same loss happens on four paths (copy,
  `Image(other)`, crop, grid section). Fixing only `copy()` leaves
  `ImageCropper(GridImage)` and `objects[i]` wrong while appearing fixed.
- **D2-2 Fix GridImage `grid_finder` aliasing in the same change?** Recommend yes.
  It is the same constructor path. It is currently harmless only because no
  in-tree caller mutates a copy's grid shape. Any fix there is still a
  behaviour change for a user who relied on the alias.
- **D2-3 Should `Image(other, illuminant="D50")` override the source?** Recommend
  yes: explicit wins. Only B2 can express this.

---

## 3. Probe and census results

### 3.1 Behaviour probes

Main ran `.scratch/core_bugs_probe.py`, `core_bugs_probe2.py` and
`core_bugs_probe3.py` at `25276e40`. The key lines, verbatim:

```
[B1.1 Image(2-D uint8)] bit_depth=8 gray=uint8[100.000,200.000] detect_mat=uint8[100.000,200.000] rgb_empty=True
[B1.2 Image(2-D uint16)] bit_depth=16 gray=uint16[10000.000,30000.000] detect_mat=uint16[10000.000,30000.000] rgb_empty=True
[B1.4 Image(2-D float32 0..200)] bit_depth=16 gray=float32[2.000,200.000] detect_mat=float32[2.000,200.000] rgb_empty=True
[B1.5 Image(2-D float32 negative)] bit_depth=16 gray=float32[-0.500,0.500] detect_mat=float32[-0.500,0.500] rgb_empty=True
[B1.6 Image(3-D float32 0..2) -- RGB out of range] RAISED ValueError: Float array contains values outside [0, 1] range. Min: 0.5, Max: 2.0
[B1.8 Image(RGB uint8) reference] bit_depth=8 gray=float32[0.392,0.784] detect_mat=float32[0.392,0.784] rgb_empty=False
    g8.png: bit_depth=8 gray=uint8[100.000,200.000] ...
    g16.png / g16.tif: bit_depth=16 gray=uint16[10000.000,30000.000] ...
    gf.tif: bit_depth=16 gray=float32[2.000,1000.000] ...
    g8.jpg: bit_depth=8 gray=uint8[88.000,205.000] ...
[B1.10 save2zarr/load_zarr/imread(store) of 2-D uint8] stored={'gray': 'uint8', 'detect_mat': 'uint8'} gray_has_omero=True
    load_zarr: bit_depth=8 gray=uint8[100.000,200.000] detect_mat=uint8[100.000,200.000]
    imread(store): bit_depth=8 gray=uint8[100.000,200.000] detect_mat=uint8[100.000,200.000]
[B1.11 SubtractGaussian on 2-D uint8 (default preserve_range=True)] ... detect_mat=uint8[0.000,1.000] unique=[0 1] n_unique=2
[B1.11b SubtractGaussian on same plate as RGB uint8 (control)] ... detect_mat=float32[0.000,0.296] n_unique=65
[B1.16 gray setter on uint8 image (gray[:] = 0.5)] gray[0,0]=0 dtype=uint8
[Q1/probe3 uint8 2-D] MeasureIntensity RAISED ValueError: cannot convert float NaN to integer (_calculate_q1 -> labeled_comprehension)
[probe3 float/255 2-D] ok Intensity_MeanIntensity 0.5877 ; [rgb uint8] ok 0.5877
[Q2 MeasureSize seeded] (uint8, float/255) Size_IntegratedIntensity (80000.0, 313.7255); Size_Area (400.0, 400.0)
[Q3 MeasureTexture seeded uint8] RAISED ValueError: Foreground array must be normalized between 0 and 1.
[Q3b MeasureTexture seeded float] float control ok shape=(1, 66)
[Q4 OtsuDetector] uint8: num_objects=0 | float/255: num_objects=1 mask_frac=0.098 | rgb uint8: num_objects=1 mask_frac=0.098
[Q5 UserThreshold(0.5)] uint8: num_objects=0 | float/255: num_objects=1 mask_frac=0.098
[Q6 Otsu diag] otsu_t=100

[B2.1 copy() D50/Linear/LabL]
    orig: illum=D50 gamma=LINEAR mode=LabL bit_depth=8
    copy: illum=D65 gamma=SRGB   mode=LabL bit_depth=8
[B2.2] Image(img): illum=D65 gamma=SRGB | Image(img, D50, None): illum=D50 gamma=LINEAR
[B2.3 crop img[0:32,0:32] D50/Linear/LabL] illum=D65 gamma=SRGB mode=gray name=p_crop
[B2.4 pipeline.apply(inplace=False) result config] illum=D65 gamma=SRGB
[B2.5 MeasureColor Lab: D50 orig (inplace) vs D50 via copy] max|diff| ColorLab_L*GeoMedian 15.95, ColorLab_a*GeoMedian 23.28
[B2.6 GridImage copy] illum=D65 gamma=SRGB grid_finder_is_same_object=True orig.nrows after copy.nrows=4 -> 4
[B2.7 load_zarr D50 store then copy] loaded: D50/LINEAR ; loaded.copy(): D65/SRGB
[B2.8 copy] public_kept=True original_kept=True private_equal=False journal_equal=True
```

B2.3 also confirms that `ImageHandler.__getitem__` drops `detect_mode`
(`LabL` → `gray`). Its `detect_mat` was still copied bit-for-bit
(`detect_mat_equal=True`), so the crop's `detect_mat` and its declared mode now
disagree: the next `reset()` would recompute gray.

### 3.2 Test census (Bug 1)

**Scope: the 36 candidate test files only, not repo-wide.** These are the files
that mention uint8/uint16 and read gray/detect_mat. Slurm job 29429272 ran
`.scratch/gray_census_plugin.py`, which records tests that pass a 2-D integer
(or out-of-range 2-D float) matrix into `_set_from_matrix`. Result:
`1021 passed, 24 skipped, 18 deselected`. Verbatim census:

```
float_out_of_range: 2 tests in 1 files
       2  tests/unit/enhance/test_subtract_blank.py
int:uint16: 5 tests in 3 files
       2  tests/unit/enhance/test_subtract_blank.py
       2  tests/unit/core/test_image_dtype_conversion.py
       1  tests/unit/sdk_/test_imread_store.py
int:uint8: 49 tests in 14 files
      14  tests/unit/detect/test_inoculum_detector.py
       9  tests/unit/detect/test_mad_hysteresis_detector.py
       5  tests/unit/core/test_accessor_numpy_interface.py
       4  tests/unit/sdk_/test_integrity_validation.py
       4  tests/unit/core/test_image_dtype_conversion.py
       3  tests/unit/enhance/test_subtract_blank.py
       3  tests/unit/core/test_image_pickle.py
       1  tests/unit/core/test_image_provenance_original_zarr.py
       1  tests/unit/detect/test_filfinder_detector.py
       1  tests/unit/sdk_/mixin/test_input_layer_mixin.py
       1  tests/unit/grid/test_grid_image.py
       1  tests/unit/correction/test_color_denoise.py
       1  tests/unit/core/test_save_intermediate_zarr.py
       1  tests/unit/correction/test_image_padder.py
```

So **54 tests in 16 files** build a single-channel integer image, plus 2 that
build an out-of-range float image. Building one is not the same as asserting
the integer behaviour. Below is my estimate of which tests would actually go
red, from reading their assertions. It is an **estimate, not a run**.

| Likely to need updating | Under A | Under B |
|---|---|---|
| `test_image_dtype_conversion.py:317,326`: `array_equal(img.gray[:], uint8/uint16 input)` | yes, assert against `input / iinfo.max` | no |
| `test_grid_image.py:121`: `array_equal(grid_image.gray[:], uint8_gray)` | yes | no |
| `test_imread_store.py:227`: `array_equal(loaded.gray[:], pixels)` (uint16) | yes | no |
| `test_image_provenance_original_zarr.py:132`: original == decoded pixels | only if A1 (integer original) is skipped. With A1 it must stay green, and it is the guard for A1 | no |
| `test_subtract_blank.py::test_refuses_a_single_channel_integer_pair` (×2) | rewrite as a positive single-channel test (the refusal is deleted) | same |
| `test_subtract_blank.py::test_a_float_single_channel_pair_outside_the_unit_range_is_refused` (×2) | only under D1-2 "refuse": the exception moves to `Image(...)` | only if D1-2 is also adopted |
| `test_image_pickle.py` (3): round-trip `loaded == img` | no: both sides normalise | no |
| `test_inoculum_detector.py` (14), `test_mad_hysteresis_detector.py` (9): MAD/GMM are relative | expected green; verify | expected green |
| Others (accessor numpy, integrity hashes, filfinder, input-layer mixin, color_denoise, padder, intermediate zarr) | expected green: they assert shapes, hashes before/after, or self-consistency | expected green |

**Estimate: about 6–8 test updates under A, about 2–4 under B**, before the new
regression tests are added. Either way, the authoritative count is the
per-phase affected-surface run after the change (the repo's `CLAUDE.md` testing
table). The full unit suite was not censused.

### 3.3 Not verified

- No probe ran the CLI end to end on a single-channel input. The CLI claims in
  §1.2 come from code reading (`read_kwargs` → `imread`).
- `--mode process --layer gray` single-channel output dtype comes from code
  reading (`_cli_process_only.py:169-171`, accessor `imsave`).
- The zone and symzones intensity columns were not probed.
- The `DenoiseBlockMatch` truncation is inferred from B1.16 (the gray setter
  truncates 0.5 → 0 into a uint8 buffer), not probed directly (bm3d).
