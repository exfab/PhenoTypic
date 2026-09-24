# Phase D adherence review: input headers, metadata join, post columns

- **Reviewer:** independent reviewer (Claude), analysis only. I did not write this code.
- **Commit reviewed:** `origin/claude/modest-mccarthy-jz0ylw` at `5177ae5`, checked out
  detached. Scope is Phase D only: `2297245` (Task 10), `b16c525` (Task 11), `8191fa5`
  (Task 12). Phases A to C and Phase E (`8738e39`, `5177ae5`) are out of scope except
  where they interact with Phase D.
- **Read:** spec `docs/superpowers/specs/2026-09-24-cli-preflight/design.md` (§0, §2, §7,
  §8, §9, §10.5, decisions D1 to D14), plan Tasks 10 to 12, `header-behavior.md`,
  `phase-gates.md`, root `CLAUDE.md`, `src/phenotypic/_cli/CLAUDE.md`.
- **Method:** code reading against the spec; end-to-end CLI probes on real files (a
  `--dry-run` to see the preflight verdict, then the same command with
  `--skip-validation` to see whether the run actually succeeds); per-check probes against
  `Image.imread`; a comparison of the GUI and CLI metadata analyses; and 34 revert
  proofs, each restored with `git checkout -- <file>` (the tree was verified clean
  afterwards: `git status --porcelain` empty). Probe scripts were kept in the session
  scratchpad, not in the repository; the reproduction steps are written out below.
- **Test commands:** `QT_QPA_PLATFORM=offscreen uv run pytest <files> -q --no-header -o
  addopts= -m "not slow" -p no:randomly` with at most `-n 4`. The full suite was not run.

## Verdict

**Changes required.** The Phase D design is largely implemented as written, the
severity rule and mode scoping are applied correctly, the header pass decodes no pixels
and imports nothing heavy, and most hunks are pinned by a test that fails when the hunk
is reverted. However, the preflight refuses at least three kinds of valid run, which
violates the change's governing rule that the preflight must never block a run that
would succeed (spec §0, "Severity follows reach", and D3):

- D1 (Blocking): channel-stacked TIFFs (ImageJ/Fiji composites, OME-TIFF `CYX`, any
  single-series `(3,H,W)` stack) decode to RGB, but the header reader calls them
  grayscale, so `PF-RGB-OP-GRAY` and `PF-DETECT-MODE-GRAY` refuse runs that complete.
- D2 (Major): `PF-META-NO-KEYS` / `PF-META-DUP-KEYS` refuse a metadata CSV that joins on
  metadata the images themselves restore (review R6's case, handled in §8 but not §9).
- D3 (Major): `PF-BIT-DEPTH` refuses `--bit-depth 16` over JPEG inputs, although
  `Image.imread` ignores the flag for JPEG and records the correct depth.

Seven revert proofs survive (see the table); four of them guard behavior whose
regression would itself be a false refusal.

## Findings

### Blocking

#### D1. Channel-stacked TIFFs are misread as grayscale, so RGB checks refuse valid runs

**Evidence.**

- `_tiff_header` reads only `tif.pages[0].samplesperpixel`
  (`src/phenotypic/_cli/_cli_input_headers.py:138-149`), and returns `channels=1` when it
  is 1 (`:145-146`). `header-behavior.md:19` and `:29` generalize one probe ("3-page
  grayscale TIFF ... first page only") into the rule "multi-page -> the first page's
  count". That probe wrote three separate series
  (`tests/unit/cli/test_preflight_input_headers.py:63-67`, one `writer.write` per page).
  A single series of three pages behaves differently.
- Verified by probe at `5177ae5` (header vs `Image.imread`, 32x32 uint8):

  | File written with `tifffile.imwrite` | Header `channels` | `Image.imread` result |
  |---|---|---|
  | `(3,H,W)`, `imagej=True, axes="CYX"` (Fiji composite) | 1 | `(32, 32, 3)` RGB |
  | `(3,H,W)`, `ome=True, axes="CYX"` (OME-TIFF) | 1 | `(32, 32, 3)` RGB |
  | `(3,H,W)`, `photometric="minisblack"` | 1 | `(32, 32, 3)` RGB |
  | `(4,H,W)`, `photometric="minisblack"` | 1 | `(32, 32, 3)` RGB |

- End to end, two Fiji-style composites of `load_synth_yeast_plate()` in one dataset,
  `--image-type Image --njobs 1 --force-local`:
  - pipeline `OtsuDetector` + `MeasureColor` + `MeasureSize`, `--dry-run`: exit 1,
    `✗ Error [PF-RGB-OP-GRAY]: meas:c read(s) RGB, but these inputs are grayscale (2 of 2
    input(s))`. The same command with `--skip-validation`: `Completed: 2/2, Failed: 0`,
    exit 0, 1104 rows with 15 `Color*` columns carrying real values (for example
    `ColorLab_L*GeoMedian` 91.72).
  - pipeline `OtsuDetector` + `MeasureSize` with `--detect-mode red`, `--dry-run`: exit 1,
    `✗ Error [PF-DETECT-MODE-GRAY]`. The same with `--skip-validation`: `Completed: 2/2`,
    exit 0.
- The Validate button in the GUI runs the same `--dry-run`, so it refuses these runs too.

**Impact.** A user whose plates were saved by Fiji or a microscope as a channel stack
cannot run any colour measurer or colour detection mode without `--skip-validation`,
which also disables every other check. The error text tells them their colour images
are grayscale, which is false. The same root cause also hides the opposite case: a
single-series `(2,H,W)` or `(5,H,W)` stack reports one channel, so `PF-CHANNELS` never
fires for it (unverified whether `imread` refuses such a stack).

**Fix.** Predict from the first *series*, not the first page, since that is what
`skimage.io.imread` returns: `tif.series[0].shape` and `.axes` are metadata-only in
tifffile. A leading channel/sample axis of length 3 or 4 maps to `channels=3`; length 2
or 5 and more maps to `raw_channels`. Add the four rows above to `CASES` in
`test_preflight_input_headers.py` so `test_the_header_prediction_matches_imread` pins
them, and correct `header-behavior.md:19,29`. Until then, a safe interim is to report
`channels=None` (unknown) for any multi-page TIFF, which can only weaken a check.

### Major

#### D2. Metadata key findings refuse CSVs keyed on metadata the images restore

**Evidence.**

- `check_metadata_join` makes `PF-META-NO-KEYS` and `PF-META-DUP-KEYS` errors whenever
  `unverified_join_columns` is empty (`src/phenotypic/_cli/_cli_preflight.py:1084`,
  `:1086-1100`). The source key frame holds only `Metadata_ImageName`,
  `Metadata_FileSuffix` and `Metadata_Dataset`
  (`src/phenotypic/_cli/_metadata_preflight.py:56-77`), and a bare metadata label such as
  `Strain` is never "unverified" because it has no underscore (`:100-107`).
- The production join intersects the CSV with the *measurement* frame's columns
  (`src/phenotypic/_cli/_metadata_join.py`, `prepare_metadata_join_keys`), and that frame
  carries every public metadata key the image restored on read
  (`_image_io_handler.py:797-824`) or a custom operation set. The spec recognizes exactly
  this incompleteness for post columns (§8, review R6) and makes the post check depend on
  it (`_metadata_set_is_complete`, `_cli_preflight.py:1184-1196`), but §9 and the
  implementation do not apply the same rule to the join.
- Verified end to end. Two PNGs written by `Image.rgb.imsave` with
  `image.metadata["Strain"]` set to `WT` and `mut` (the header reader correctly reports
  `carries_phenotypic_metadata=True` for both), and `strain_media.csv` =
  `Strain,Media / WT,YPD / mut,SC`:
  - `--metadata strain_media.csv --dry-run`: exit 1, `✗ Error [PF-META-NO-KEYS]:
    strain_media.csv shares no column with the images' ImageName, FileSuffix or Dataset,
    so without a measurement-level key nothing would join`.
  - The same with `--skip-validation`: exit 0, and `deliverables/measurements.parquet`
    carries `Metadata_Media` correctly joined (`p0`/`WT`/`YPD`, `p1`/`mut`/`SC`).
- By the same mechanism, a CSV keyed on `ImageName + Strain` whose rows differ only by
  `Strain` looks duplicated to the source frame and draws a `PF-META-DUP-KEYS` error
  (inferred from the code path above; not run end to end).

**Impact.** A false refusal for any run whose inputs carry PhenoTypic metadata (a
PhenoTypic TIFF/PNG export, or a PhenoTypic store, including process-mode output, which
`CLAUDE.md` states is valid input to a full run) or whose pipeline contains a custom
operation that sets metadata, when the CSV is keyed on that metadata. The implementation
follows the letter of §9, so this is a spec gap carried into code, not a silent deviation.

**Fix.** Apply §8's completeness rule to §9: when `_metadata_set_is_complete(context)` is
false, emit `PF-META-NO-KEYS` and `PF-META-DUP-KEYS` as warnings, and say why ("the
images restore PhenoTypic metadata, which may supply these keys"). Add the PNG scenario
above as a test. Record the decision beside D12 in the spec.

#### D3. `PF-BIT-DEPTH` refuses `--bit-depth 16` over JPEGs, which `imread` handles correctly

**Evidence.**

- `check_bit_depth` flags every input whose header depth differs from `--bit-depth`
  (`src/phenotypic/_cli/_cli_preflight.py:981-999`, condition at `:993`), and the header
  reader reports JPEG as 8 bits (`_cli_input_headers.py:109`). Its message says
  `Image.imread` "would silently mislabel" the data.
- `Image.imread` overrides the requested depth for JPEG: `if suffix in
  IO.JPEG_FILE_EXTENSIONS: bit_depth = 8` (`_image_io_handler.py:793-794`). Nothing is
  mislabelled. `header-behavior.md` has no JPEG row, so the rule was generalized from TIFF
  and PNG probes only.
- Verified end to end on two JPEGs: `--bit-depth 16 --dry-run` exits 1 with
  `✗ Error [PF-BIT-DEPTH] ... (2 of 2 input(s))`; the same command with
  `--skip-validation` completes 2/2 and every row records `Metadata_BitDepth = 8`.

**Impact.** A false refusal. The flag is pointless for JPEG, but the run it refuses is
correct, and the stated reason is untrue. A related design observation, not a separate
finding: for integer TIFF and PNG inputs, `--bit-depth` changes nothing when it equals the
dtype's depth and is refused (for a whole-run mismatch) when it differs, so the option is
now either a no-op or an error for integer inputs. A `uint16` file holding 8-bit values
(0 to 255) with `--bit-depth 8` is plausibly intentional; the probe shows `imread` keeps
the `uint16` array and records 8. Whether that is ever a legitimate use should be decided
explicitly in `header-behavior.md` rather than implied by the error.

**Fix.** Exclude `IO.JPEG_FILE_EXTENSIONS` from `check_bit_depth` (or report them in a
warning saying the flag is ignored for JPEG), add a JPEG row to `header-behavior.md`, and
add a JPEG case to `test_a_bit_depth_that_contradicts_the_data`.

### Minor

#### D4. Any qualified attribute column softens the metadata key errors, including ones production treats as attributes

**Evidence.** `unverified_measurement_join_columns` counts every non-source column with
an underscore that is not a metadata member (`_metadata_preflight.py:100-107`), so
`Strain_ID` or `Plate_Barcode` counts as a possible measurement key. Production keeps a
column raw (a potential key) only when it already exists in the measurement frame or is a
known schema header (`sdk_/_metadata_helpers.py:308-340`); `Strain_ID` is neither, so it
is prefixed and joined as an attribute. Verified with `analyze_metadata_join`: a CSV
`Strain_ID,Media` yields `join_columns=()` with `unverified=('Strain_ID',)`, so
`PF-META-NO-KEYS` is a warning although production logs "no columns in common ...
skipping join" (`_cli_output_manager.py:336-341`); a CSV `ImageName,Plate_Barcode` with a
repeated `ImageName` yields a `PF-META-DUP-KEYS` warning although production fans the rows
out.

**Impact.** Missed errors (never a false refusal). The rule is the GUI's, reused as §9
specifies, and is documented as conservative, so this is a precision gap, not a deviation.

**Fix.** Treat a column as unverified only when it is a known non-metadata schema header
(the `known_schema_headers` test `external_metadata_preserved_columns` already applies) or
when an in-scope operation is custom (the §8 completeness predicate). Keep the GUI and CLI
on the same helper.

#### D5. Spec and plan deviations that are not recorded in the spec

**Evidence.**

1. §8 (`design.md:529`, `:555`) and plan Task 12 (`plan.md:515`) specify
   `PostMeasurement.required_columns() -> tuple[str, ...]`. The implementation is
   `preflight_columns(available) -> (missing, produced)`
   (`abc_/_post_measurement.py:72-92`), plus a `_preflight_reads_metadata_only` class
   variable. The new shape is better (each op resolves with its own rules and reports
   what it adds), and the commit message says so, but the spec still describes the old
   API. The test file is still named `test_required_columns.py`.
2. §7 (`design.md:487`) says a store's channel count comes from `project_ngff_axes`. The
   implementation reads only the PhenoTypic `series` block and reports third-party stores
   as unknown (`_cli_input_headers.py:152-167`). That is safe (it can only weaken a
   check) and is explained in the module docstring, but not in the spec.
3. §9 (`design.md:585`) and plan Task 11 Step 5 say the GUI's `build_metadata_preflight`
   delegates the analysis to `analyze_metadata_join`; the commit message repeats that it
   "delegates". It does not call `analyze_metadata_join`. It reuses
   `source_join_key_frame` and `unverified_measurement_join_columns` and still calls
   `prepare_metadata_join_keys` itself, after its own `normalize_metadata_input_columns`
   (`_gui/run_console/_request_safety.py:402-434`). The duplicated source-key projection
   is removed, which is the spec's stated purpose. I compared the two analyses on seven
   CSV shapes (plain, per-well, grid layout, qualified attribute, legacy
   `Metadata_ImageName`, duplicate keys, duplicate keys with a qualified attribute): join
   columns, unverified columns and duplicate counts agree on all seven.
4. §7 says the header pass records "channel count, dtype, and shape"; `InputHeader` has
   no shape field. Nothing consumes one, so this is cosmetic.

**Impact.** A later reader of the spec will look for methods and behavior that do not
exist, and the GUI path can drift from the CLI path because the two still compose the
analysis separately.

**Fix.** Add "As implemented" notes to §7, §8 and §9, as §10.2 already does, and either
route the GUI through `analyze_metadata_join` or correct the commit-level claim in the
plan's task notes.

#### D6. Revert proofs that survive

**Evidence.** Seven mutations survive the Phase D tests (full table below): M3, M11, M14,
M19, M22, M23, M31. Four of them guard behavior whose regression would be a false refusal
or a wrong severity:

- **M14:** keying stem groups by stem alone, not `(dataset, stem)`
  (`_cli_preflight.py:1034`), survives. `plate1/a.png` beside `plate2/a.png` is the
  ordinary layout of a multi-plate run, and that regression would refuse it as a
  `PF-STEM-COLLISION` error. No test puts one stem in two datasets. (Current behavior
  verified correct: no finding.)
- **M11:** refusing a 4-sample TIFF (`_cli_input_headers.py:147`) survives, because
  `CASES` has no RGBA TIFF. That regression would refuse RGBA TIFFs with `PF-CHANNELS`.
  (Current behavior verified correct: `(channels, raw) = (3, None)`, `imread` RGB.)
- **M22 and M23:** ignoring the PhenoTypic key in a TIFF `ImageDescription` or a PNG text
  chunk (`_cli_input_headers.py:143`, `:108`) survives. Plan Task 12 Step 2 required a
  test that "any input's header carries PhenoTypic metadata" softens `PF-POST-COLUMN`;
  only the store and custom-class cases are tested
  (`test_preflight_post_columns.py:72-103`). That regression turns the R6 warning into a
  false `PF-POST-COLUMN` error. (Current behavior verified correct for PNG and TIFF:
  `carries=True`, severity `warning`.)

The other three: **M3** (the startup parse reverted to pandas) survives because
`test_skip_validation_still_refuses_an_unreadable_csv` uses an unterminated quote that
both parsers reject, so nothing pins the switch to the shared reader; a CSV with a ragged
long row separates them (Polars `ComputeError`, pandas accepts). **M31** (exact-name
lookup instead of `resolve_metadata_column` in `missing_metadata_columns`) survives
because no test uses a legacy or alias spelling. **M19** (the zero-byte guard removed)
survives because `test_an_empty_file_is_an_unreadable_header`
(`test_preflight_input_headers.py:116-119`) asserts `"empty" in header.error`, and the
`tmp_path` directory name contains the test name, which contains "empty"; the assertion
is vacuous. (With the guard removed the file is still an error, so the behavior is
preserved; only the message check is hollow.)

**Fix.** Add a two-dataset same-stem negative case, an RGBA TIFF row in `CASES`, PNG and
TIFF "carries PhenoTypic metadata" cases for `check_post_columns`, a ragged-row CSV for the
startup parse, a legacy-alias case for `missing_metadata_columns`, and assert on the
message text only (for example `header.error.startswith("the file is empty")`).

#### D7. The write tripwire does not exercise the metadata read path

**Evidence.** `test_the_preflight_writes_nothing`
(`tests/unit/cli/test_cli_preflight_core.py`) builds its context without `metadata_csv`,
so `check_metadata_join` and `_metadata_csv_headers` return before reading anything
(`_cli_preflight.py:1068-1069`, `:1169-1170`). The plan's global constraint says every
task that adds a check re-runs the tripwire; Task 11 did, but its check never ran inside
it.

**Impact.** A future write in `read_metadata_csv`, `analyze_metadata_join` or
`prepare_metadata_join_keys` would pass the tripwire.

**Fix.** Pass a small metadata CSV (written before the tripwire arms) in the tripwire's
full-mode context.

### Nit

#### D8. A second metadata parser remains on the CLI path

`_snapshot_metadata_csv` still validates the bytes with `pd.read_csv`
(`src/phenotypic/phenotypicCLI.py:719`). Because the Polars startup parse runs first and
is the stricter of the two on every case I tried (empty file, header only, duplicate
header, ragged long and short rows, Latin-1 bytes, BOM), the pandas parse can no longer
refuse a CSV the shared reader accepted; it is redundant rather than harmful. §10.5's
"one reader" goal would be met fully by calling `read_metadata_csv` on the payload there
too. Also note that the startup parse runs in `process` mode, which ignores `--metadata`,
so a CSV with a ragged long row (accepted by the old pandas parse) now refuses a process
run that would never read it.

#### D9. `PF-HEADER-UNREADABLE` wording for a Zarr v2 store

A third-party NGFF 0.4 (Zarr v2) store has no root `zarr.json`, so `_store_header`
raises `FileNotFoundError` (`_cli_input_headers.py:161`) and the finding says the header is
unreadable. The severity is right (`imread` refuses v2 stores, `ngff_.py:872-892`), but
the message could name the real cause, as `ngff_._zarr_v2_marker` already can.

## Revert-proof table

Each row: the hunk was replaced (first occurrence only), the named tests were run before
and after, and the file was restored with `git checkout -- <file>`. `P` =
`src/phenotypic/_cli/_cli_preflight.py`, `H` = `_cli_input_headers.py`, `MJ` =
`_metadata_join.py`, `MP` = `_metadata_preflight.py`; test files are
`test_preflight_input_headers.py` (TH), `test_metadata_preflight.py` (TM),
`test_preflight_post_columns.py` (TP), `tests/unit/post/test_required_columns.py` (TR).

| # | Finding / behavior | Hunk reverted | Test | Before | After |
|---|---|---|---|---|---|
| M1 | F22 one reader | `MJ` `read_metadata_csv`: drop `infer_schema_length=None` | TM reader + worker tests | 2 passed | 2 failed (killed) |
| M2 | F22 worker path | `_embedded_measurement_tables.py`: back to `pl.read_csv` | TM worker test | 1 passed | 1 failed (killed) |
| M3 | R22 startup parse uses the shared reader | `phenotypicCLI.py`: back to `pd.read_csv` | TM `test_skip_validation_still_refuses_an_unreadable_csv` | 1 passed | 1 passed (**survived**) |
| M4 | R1 plate maps are not refused | `P:1084` key severity always `error` | TM per-well + grid layout | 2 passed | 2 failed (killed) |
| M5 | NO-KEYS / DUP-KEYS are errors without a measurement key | `P:1084` always `warning` | TM no-keys + dup-keys | 2 passed | 2 failed (killed) |
| M6 | metadata check scoped to full mode | `P` drop the `mode != "full"` guard | TM out-of-scope modes | 2 passed | 2 failed (killed) |
| M7 | unmatched images listed | `MP` unmatched listing disabled | TM unmatched | 1 passed | 1 failed (killed) |
| M8 | 16-bit PNG depth from IHDR | `H:109` PNG bits fixed to 8 | TH | 39 passed | 1 failed (killed) |
| M9 | palette PNG decodes to RGB | `H:35` drop `"P"` | TH | 39 passed | 1 failed (killed) |
| M10 | LA PNG refused | `H:36` empty two-channel set | TH | 39 passed | 2 failed (killed) |
| M11 | 4-sample TIFF is RGB | `H:147` `samples == 3` only | TH | 39 passed | 39 passed (**survived**) |
| M12 | severity follows reach | `P` `_severity_for` always `error` | TH | 39 passed | 3 failed (killed) |
| M13 | F28 stem collisions reported | `P` `clashes = {}` | TH stem test | 1 passed | 1 failed (killed) |
| M14 | stems grouped per dataset | `P:1034` group by stem alone | TH | 39 passed | 39 passed (**survived**) |
| M15 | detect-mode out of scope in measure | `P` drop measure guard | TH scoping test | 1 passed | 1 failed (killed) |
| M16 | D10 RGB check uses `operations_in_scope` | `P` walk the whole tree | TH process-mode test | 1 passed | 1 failed (killed) |
| M17 | bit depth flagged | `P:993` never flag | TH bit-depth test | 1 passed | 1 failed (killed) |
| M18 | unreadable headers flagged | `P` `affected = []` | TH subset test | 1 passed | 1 failed (killed) |
| M19 | zero-byte message | `H:87-88` guard removed | TH empty-file test | 1 passed | 1 passed (**survived**, vacuous assertion) |
| M20 | store channels from `series` | `H:166` `channels = None` | TH store test | 1 passed | 1 failed (killed) |
| M21 | store carries metadata | `H:167` `carries=False` | TP store softening | 1 passed | 1 failed (killed) |
| M22 | TIFF key detected | `H:143` `carries = False` | TH + TP | 46 passed | 46 passed (**survived**) |
| M23 | PNG key detected | `H:108` `carries = False` | TH + TP | 46 passed | 46 passed (**survived**) |
| M24 | custom op softens | `P` custom-module test disabled | TP custom op | 1 passed | 1 failed (killed) |
| M26 | produced columns credited | `P:1239` removed | TP known sources | 1 passed | 1 failed (killed) |
| M27 | `Metadata_Dataset` credited | `P` removed | TP known sources | 1 passed | 1 failed (killed) |
| M28 | `--metadata` headers credited | `P:1218` removed | TP known sources | 1 passed | 1 failed (killed) |
| M29 | metadata-only op errors when complete | `_append_string.py` flag `False` | TP error test | 1 passed | 1 failed (killed) |
| M30 | R6 JoinMetadata frame spelling | `_join_metadata.py` use `self.on` | TR spelling test | 1 passed | 1 failed (killed) |
| M31 | ops' own resolution rules | `post/_utils.py` exact-name lookup | TR + TP | 15 passed | 15 passed (**survived**) |
| M32 | intrinsic set pinned to a real run | `P` drop the suffix line | TP intrinsic pin | 1 passed | 1 failed (killed) |
| M33 | orphans reported | `P` orphans disabled | TM orphans | 1 passed | 1 failed (killed) |
| M36 | RAW without rawpy | `P` `find_spec` treated as present | TH raw test | 1 passed | 1 failed (killed) |
| M38 | post check out of scope in process | `P` drop `MODE_SLOTS` guard | TP process test | 1 passed | 1 failed (killed) |

Summary: 34 mutations, 27 killed, 7 survived (M3, M11, M14, M19, M22, M23, M31; see D6).

## Regression spot checks

All run at `5177ae5` after every revert was restored.

- Phase D test files plus `tests/unit/gui/run_console/test_request_safety.py`: **87
  passed** (`-n 4`).
- `tests/unit/ci/test_startup_imports.py`, `test_deferred_imports.py`,
  `tests/unit/cli/test_cli_preflight_core.py` (lazy-entry guards and the write
  tripwire): **251 passed**.
- Importing `_cli_input_headers`, `_metadata_preflight` and `_cli_preflight` in a fresh
  process loads none of PIL, tifffile, polars, pandas, torch, skimage, zarr, cv2 or
  matplotlib.
- No pixel decode: with `PIL.ImageFile.ImageFile.load`, `tifffile.TiffPage.asarray` and
  `tifffile.TiffFile.asarray` patched to raise, `read_input_headers` read 31 probe files
  (PNG, JPEG, TIFF of every kind above) with zero decode attempts.
- Cross-dataset same stem (`plate1/a.png`, `plate2/a.png`): no `PF-STEM-COLLISION`.
- RGBA TIFF: header `(channels, raw) = (3, None)`, `imread` RGB.
- PhenoTypic PNG and TIFF exports with public metadata: `carries_phenotypic_metadata`
  is true, and `PF-POST-COLUMN` for `AppendString(column="Strain")` is a warning, as §8
  requires. A PhenoTypic JPEG export does not restore its metadata on `imread` in this
  environment (exifread 3.5.1 installed), so the header's `False` for JPEG is accurate.
- Palette PNG with transparency, `I;16` PNG, and float32 RGB TIFF: header agrees with
  `imread` (3, 1 and 3 channels; bits 8, 16 and unknown).
- The phase gate recorded in `phase-gates.md` ("Phase D": 6364 passed, 12 napari and 3
  pytest-qt environmental failures) was not re-run; the full suite was not run.

## Verified as correct

- **Severity rule.** `_reach_finding` and `_severity_for` (`_cli_preflight.py:574-577`,
  `:895-908`) give an error only when every input is affected and list the affected
  inputs otherwise; `PF-STEM-COLLISION` is always an error, as §7 and the probe in
  `header-behavior.md` justify.
- **Mode scoping.** Header, channel, detect-mode, bit-depth, RAW and stem checks return
  nothing in `measure` mode; only `PF-RGB-OP-GRAY` runs there, over the stores' recorded
  series. `check_rgb_ops_on_gray` iterates `operations_in_scope`, so `MeasureColor` in
  `meas` produces nothing under `process`. The metadata check runs only in `full`; the
  post check runs only where `post` is in `MODE_SLOTS`. `--sample` does not narrow the
  checks (R26).
- **Decoded-channel table** for single-page and multi-series files matches `imread`:
  RGBA, palette and CMYK map to 3; LA and 2-sample TIFF and 5-sample TIFF map to refused
  counts, which `imread` does refuse. The exception is D1.
- **16-bit PNG depth** from the IHDR byte (`_cli_input_headers.py:119-129`) is correct and
  necessary, since Pillow reports 16-bit RGB PNG as mode `RGB`.
- **Stores** are read from the root `zarr.json` only; `series` is a name-keyed dict
  (`ngff_.py:584`), so the `"rgb" in series` test is sound, and `zarr.open_array` is never
  called (pinned by a test).
- **Shared metadata reader.** Every caller §10.5 names uses `read_metadata_csv`: the
  worker (`_embedded_measurement_tables.py`), `join_metadata`
  (`_cli_output_manager.py`), the GUI preflight (`_request_safety.py:402-404`), the CLI
  preflight, and the startup parse, which stays outside `--skip-validation`
  (`phenotypicCLI.py`, pinned by a test, though see M3). Open question 4 is answered in
  the commit message with a reason I could not fault: finalization re-joins
  `deliverables/metadata.csv` rather than reading per-store metadata tables.
- **GUI and CLI metadata analyses agree** on seven CSV shapes (D5 item 3), and
  `test_request_safety.py` passes unchanged. The moved
  `unverified_measurement_join_columns` carries the R38 docstring explaining that its
  `"_" in column` test is a qualification test, not a `Metadata_` prefix test, so the
  schema-ownership rule is respected.
- **Plate maps (R1).** A per-well map keyed on `ImageName + Grid_RowNum + Grid_ColNum` and a
  layout keyed on grid position alone produce only warnings.
- **Post columns.** Each post op answers through its own resolution rules;
  `JoinMetadata` reports keys in the frame's spelling (R6); `ExpandMetadata` and
  `MergeMetadata` credit their prefixed outputs, so a later op finds them. The known set
  (intrinsic headers read back from `insert_metadata` and pinned against a real read and
  measure, in-scope measurer headers, `--metadata` headers, `Metadata_Dataset`) matches a
  real run's master: the only master columns outside it are `Bbox_*` and `Object_Label`,
  which the metadata-only ops cannot address. No shipped operation outside `_core` writes
  image metadata, and all five shipped post ops override `preflight_columns`, so the
  "provably complete" error case holds for built-in pipelines. `JoinMetadata` misses stay
  warnings, as §8 requires.
- **No `Metadata_` prefix tests** were introduced; membership is a set test on canonical
  headers, and metadata ownership goes through `metadata_member_for_header`.
