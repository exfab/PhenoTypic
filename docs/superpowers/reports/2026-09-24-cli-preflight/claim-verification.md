# Empirical verification of 8 spec claims

Repo: `/home/user/PhenoTypic` (no tracked file modified). All probes were run with `uv run python`.
Scratch root: `$S/` (called `$S` below). The probe scripts are committed at
`docs/superpowers/plans/2026-09-24-cli-preflight/baseline_probes/`; they ran against `81d19ec`.
Inputs: `$S/in_rgb/tiny.png` (64x64x3 uint8) and `$S/in_gray/tiny.png` (64x64 uint8).

> **Reading the verdicts.** Each verdict tests the claim *as phrased in the row*. Rows 2a and
> 2b are phrased as "the sentinel survives", so REFUTED there means the output directory **was
> deleted**: the ordering defect is confirmed, not disproved. Row 4 is a genuine correction to
> the pre-spec survey, which had said a grayscale image silently falls back to gray; the CLI
> in fact fails each such image with `ValueError`.

| # | Claim | Verdict |
|---|---|---|
| 1a | A custom op outside `phenotypic` fails `from_json` in a fresh process | VERIFIED |
| 1b | `PHENOTYPIC_PRELOAD_MODULES` + `preload_custom_operation_modules()` resolves it | VERIFIED **only for a self-registering module** (one that sets `phenotypic.<Name> = cls`). A module that only defines the class is REFUTED: it still fails |
| 1c | The CLI `--dry-run` with `PHENOTYPIC_PRELOAD_MODULES` set validates | REFUTED. It fails with "Pipeline loading failed" even with a self-registering module, because the main CLI process never calls the preload |
| 2a | Invalid pipeline JSON with `--overwrite` leaves `sentinel.txt` | REFUTED. The whole output dir is deleted **before** the pipeline is validated |
| 2b | A valid pipeline with `--overwrite --dry-run` leaves `sentinel.txt` | REFUTED. The dry run deletes the whole output dir |
| 3 | A no-detector pipeline on a plain `Image` raises | VERIFIED: `RuntimeError` wrapping `NoObjectsError` (with meas), or a bare `NoObjectsError` (empty meas) |
| 4 | `Image(gray, detect_mode=<rgb mode>)` silently falls back | REFUTED. There is no such kwarg (`TypeError`). `set_detect_mode` raises `ValueError`, and that is the path the CLI takes |
| 5 | `.nef`/`.dng` enter the first (`skimage`) branch | VERIFIED. rawpy is installed, but its demosaic branch is dead code |
| 6 | `MeasureColor` on a grayscale Image raises | VERIFIED: `OperationFailedError` wrapping `AttributeError` |
| 7 | A post-op failure in CLI finalization is logged at WARNING and the master is returned | VERIFIED. Also, one failing post op discards the output of every post op |
| 8 | Duplicate `ops` keys: `from_json` silently keeps the last | VERIFIED. The surviving op keeps the **first** occurrence's position in execution order |

---

## 1. Custom op resolution

**Mechanism (read):** `_serializable_pipeline.py:628` `_find_class_in_phenotypic(class_name)` looks the **class name only** up in the following places: `_LEGACY_CLASS_ALIASES`, then `hasattr(phenotypic, name)`, then a fixed list of `phenotypic.*` submodules. The JSON records `"class": "MyThreshDetector"` and no module path. `_cli_preload.py` only runs `importlib.import_module` on each name listed in the env var. So a preloaded module helps only if its import side effect attaches the class to `phenotypic` (see `tests/_fakes/register_fake_gpu.py`: `phenotypic.FakeGpuDetector = FakeGpuDetector`).

Probe files:
- `$S/mods/my_custom_ops.py` defines `MyThreshDetector(ObjectDetector)` and does not register it.
- `$S/mods/my_custom_ops_reg.py` does `phenotypic.MyThreshDetector = MyThreshDetector`.
- `$S/p1/build.py` writes the pipeline JSON.
- `$S/p1/load.py` loads the JSON in a fresh process, optionally calling `preload_custom_operation_modules()` first.

Side finding: `to_json('custom.json')` wrote **`custom.json.pht-pipe`**, because `ensure_typed_json_suffix` appends the typed suffix.

```
== fresh process, no import, no preload
FAIL AttributeError Class 'MyThreshDetector' not found in phenotypic namespace. Make sure it's properly imported in phenotypic.__init__.py
== PYTHONPATH set, PRELOAD=my_custom_ops (non-registering)
preloaded: my_custom_ops my_custom_ops in sys.modules: True
FAIL AttributeError Class 'MyThreshDetector' not found in phenotypic namespace. ...
== PRELOAD=my_custom_ops_reg (self-registering)
preloaded: my_custom_ops_reg my_custom_ops in sys.modules: True
OK ops: {'det': 'my_custom_ops.MyThreshDetector'}
```

CLI run: `PYTHONPATH=$S/mods PHENOTYPIC_PRELOAD_MODULES=<mod> uv run python -m phenotypic --pipeline custom.json.pht-pipe --input in_rgb --output out --dry-run`. The result was the same for `<mod>` = empty, `my_custom_ops` and `my_custom_ops_reg`:
```
✓ Execution configuration validated
✗ Pipeline loading failed:
  - Failed to load pipeline: AttributeError: Class 'MyThreshDetector' not found
in phenotypic namespace. Make sure it's properly imported in phenotypic.__init__.py
exit=1
```
No output dir was created. **Why:** `preload_custom_operation_modules` is called only from `_cli_finalize_fanout.py:809`, `_cli_staged_slurm_worker.py:616`, `_cli_checkpoint_handler.py:501` and `_gui/shell/_app.py:277`. `phenotypicCLI.py` and `__main__.py` never call it, so `validate_pipeline` (`_cli_validation.py:48`) runs without the preload.

Control: I imported `my_custom_ops_reg` in-process, then called `phenotypic_cli()` with the same argv. The output was `✓ Pipeline loaded successfully` and `✓ Configuration validation passed`. This confirms that only the missing preload in the main process blocks the run.

## 2. Overwrite ordering

Valid pipeline: `ImagePipeline(ops={'det': OtsuDetector()}, meas={'size': MeasureSize()}).to_json(...)`, written as `$S/p2/valid.json.pht-pipe`. Invalid pipeline: `$S/p2/bad.json` containing `{not json`.

A. `--pipeline bad.json --input in_rgb --output outA --overwrite`, with `outA/sentinel.txt` present:
```
Overwriting existing output directory: .../p2/outA
Scanning .../in_rgb...
Found 1 images in 1 dataset(s)
✓ Execution configuration validated
✗ Pipeline loading failed:
  - Failed to load pipeline: ValueError: Invalid JSON data: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)
exit=1
ls: cannot access '.../p2/outA': No such file or directory
```
B. The valid pipeline with `--overwrite --dry-run` printed `Overwriting existing output directory: .../p2/outB`, then the dry-run estimate, then `exit=0`. Afterwards `outB` did not exist: sentinel.txt was deleted.

C. Control: a valid pipeline with `--dry-run` and no `--overwrite` gave `Error: Output directory already contains files`, `exit=1`, and the sentinel survived.

Code: `phenotypicCLI.py:2393-2399` runs `shutil.rmtree(output_dir)`. This is before `validate_pipeline` (about line 2517) and before `if config.dry_run:` (line 2547). Only the unstageable-GPU probe (about line 2228) runs before the rmtree, and it swallows every other exception.

## 3. No-detector pipeline on a plain Image

Probe: `$S/p3/probe3.py`, using `Image(load_synth_yeast_plate().rgb[:].copy())`.

Caveat: `load_synth_yeast_plate()` returns a GridImage whose **objmap is already populated** (96 objects). Wrapping the GridImage itself would carry its objects over, so the probe starts from `rgb` only.
```
== enh-only ops (BlurGauss) + MeasureSize, fresh Image (0 objects)
  RAISED RuntimeError : [MeasureSize] (step 1/1, key='size'): The operation: MeasureSize failed on _root_image: plain. <class 'phenotypic.sdk_.exceptions_.NoObjectsError'>: No objects currently in image: "plain". Apply a `Detector` to the Image object first or access image-wide information using Image.props.
    (raised at _image_pipeline_core.py:1218 in measure)
== empty ops + MeasureSize, fresh Image      -> same RuntimeError
== empty ops + MeasureSize, Image w/ copied objmap (96 objects)
  RESULT DataFrame (96, 16)   # no detector is needed if objmap is pre-populated
== enh-only ops + empty meas, fresh Image
  RAISED NoObjectsError : No objects currently in image: "plain". ...
    File "_image_pipeline_core.py", line 1272, in _get_image_info
      else image.objects.info(include_metadata=include_metadata))
    File "_image_objects_handler.py", line 78, in objects
== empty ops + empty meas, fresh Image      -> same bare NoObjectsError
```
With `meas`, the error is wrapped in `RuntimeError`. With empty `meas`, a bare `NoObjectsError` is raised from `_get_image_info`. The CLI's `validate_pipeline` also rejects empty ops together with empty meas ("Pipeline has no operations or measurements").

Side finding: one column key in the result frame is the enum `OBJECT.LABEL`, not the string `'Object_Label'`: `columns: [..., <OBJECT.LABEL: 'Object_Label'>, 'Bbox_CenterRR', ...]`.

## 4. Grayscale with an RGB-requiring detect mode

`requires_rgb=True` holds for every mode except `gray`: `HsvS, HsvV, InvS, LabA, LabB, LabL, MinRGB, blue, green, red`.

Probe: `$S/p4/probe4.py` and `probe4b.py`.
```
Image(gray, detect_mode=m)  -> TypeError Image.__init__() got an unexpected keyword argument 'detect_mode'   (all 10 modes)
Image.imread(gray.png, detect_mode='red') -> TypeError Image.__init__() got an unexpected keyword argument 'detect_mode'
Image(gray).set_detect_mode(m) -> ValueError : Cannot use detect_mode 'm': image has no RGB data.   (all 10 modes)
```
`Image.__init__(arr, name, bit_depth, gamma, illuminant)` has no `detect_mode` parameter.

The silent-fallback branch at `_image_data_manager.py:430` (`mode.requires_rgb and not has_rgb` falls back to a copy of gray) exists, but I found no public way to reach it that succeeds. The closest path is `Image(rgb).set_detect_mode('red')` followed by `set_image(gray)`. That path **raises `IndexError: too many indices for array: array is 2-dimensional, but 3 were indexed`**. The cause is that `_set_from_matrix` then calls `detect_mat.reset()` (`_detect_mat_accessor.py:165`), which recomputes the mode without the RGB guard. This is a separate bug.

CLI trace: all three worker paths read the file in gray mode and then call `set_detect_mode`:
- `_cli_process_single.py:277` pops `detect_mode`, line 278 calls `imread`, and lines 329-330 call `image.set_detect_mode(...)`. This runs after the initial store checkpoint is saved at line 316.
- `_cli_process_only.py:318-321`
- `_cli_staged_workers.py:360`

A CLI run on the grayscale PNG therefore hits the **ValueError**, not a silent fallback. Empirical run: `--pipeline valid --input in_gray --output out_red --detect-mode red --njobs 1`:
```
Processing failed for in_gray/tiny.png:
ValueError: Cannot use detect_mode 'red': image has no RGB data.
phenotypic._cli._cli_failure_tracker.PerImageScientificError: Cannot use detect_mode 'red': image has no RGB data.
Completed: 0/1   Failed: 1   Success rate: 0.0%
exit=1
```
The failure is per image, the run still finalizes (report and README are written), and `--dry-run` validation does not catch it.

## 5. RAW routing

`IO.ACCEPTED_FILE_EXTENSIONS = PNG + JPEG + TIFF + RAW_FILE_EXTENSIONS` (`sdk_/constants_.py:96-101`). So `_image_io_handler.py:732` `if suffix in IO.ACCEPTED_FILE_EXTENSIONS` always catches `.cr2/.cr3/.nef/.arw/.dng`, and the `elif suffix in IO.RAW_FILE_EXTENSIONS and rawpy is not None` demosaic branch at line 736 is unreachable.

Probe: `$S/p5/probe5.py` monkeypatches `skimage.io.imread` and `rawpy.imread` to record calls, then calls `Image.imread` on empty files.
```
rawpy installed: True
'.dng' in ACCEPTED: True | '.nef' in ACCEPTED: True
.dng: skimage calls=[('skimage.io.imread', '.../empty.dng')] rawpy calls=1 -> returned Image shape (8, 8, 3)
.nef / .NEF / .cr2: identical
```
The single `rawpy.imread` call is **metadata only**. Its call stack is `imread:796 <- _extract_raw_metadata:599`, the fallback used because exiftool is not on PATH (`exiftool on PATH: None`), and its exception is swallowed. The pixels always come from skimage.

Tiny TIFF saved with a `.dng` suffix (uint16 RGB, values 0..50800, written by tifffile):
```
imageio plugin: imageio.plugins.pillow PillowPlugin   (skimage 0.25.2)
tifffile: uint16 0 50800 | skimage: uint8 0 198
Image.imread(TIFF-as-.dng): (16, 16, 3) uint8 bit_depth 8
```
skimage/imageio routes `.dng` to Pillow, which **silently down-converts 16-bit to 8-bit**. This was not tested on a real camera DNG or NEF: none was available, and producing one cheaply is not possible. For a real DNG, Pillow would read whatever TIFF IFD it understands, likely the preview image, rather than demosaiced sensor data.

## 6. MeasureColor and CalibrateColorRpcc on grayscale

Probe: `$S/p6/probe6.py` builds `Image(load_synth_yeast_plate().gray[:])`, which has `rgb` empty and `bit_depth` 16, then runs `OtsuDetector`, which finds 559 objects.
```
MeasureColor().measure(gray image)
  RAISED OperationFailedError : The operation: MeasureColor failed on _root_image: grayplate. <class 'AttributeError'>: XYZ conversion is not available for grayscale images.
    (abc_/_measure_features.py:463)
Pipeline Otsu + MeasureColor .apply_and_measure
  RAISED RuntimeError : [MeasureColor] (step 1/1, key='color'): The operation: MeasureColor failed on _root_image: g2. <class 'AttributeError'>: XYZ conversion is not available for grayscale images.
```
CalibrateColorRpcc construction:
```
CalibrateColorRpcc()          -> ValidationError: rois Field required
CalibrateColorRpcc(rois=[])   -> ValidationError: rois List should have at least 1 item after validation, not 0
CalibrateColorRpcc(rois=[CheckerRoi(row=(0,50), col=(0,50))]) -> OK
```
`.apply(gray Image)` with a ROI and no checker present raises the following exception chain:
```
builtins.RuntimeError | CalibrateColorRpcc failed on image g3:
builtins.Exception    | CalibrateColorRpcc failed on image g3: No array form found. Either arr image was 2-D and had no array form. Set a multi-channel image or use a FormatConverter
phenotypic.sdk_.exceptions_.NoArrayError | No array form found. ...
```
It fails on the `image.rgb` access at `_calibrate_color_rpcc.py:477`. There is no dedicated grayscale guard.

## 7. Post-op failure swallowed in finalization

Function: `_cli_output_manager.py:873` `_apply_post_to_master(master_df, pipeline)`. Its only caller is `finalize_post_master_outputs` (line 1041, call at line 1174). Probe: `$S/p7/probe7.py`, with master `pl.DataFrame({Metadata_ImageName, Object_Label, Size_Area})`.
```
== AppendString(column='Metadata_DoesNotExist')
  direct op.apply RAISED KeyError : "Column 'Metadata_DoesNotExist' not found in DataFrame. Available columns: [...]"
  _apply_post_to_master returned; identical to master: True | is same object: True
  LOG: WARNING phenotypic._cli._cli_output_manager: Post-measurement transform raised during aggregation; seeding clean master into measurements.{csv,parquet} instead
  exc_info last line: KeyError: "Column 'Metadata_DoesNotExist' not found ..."
== ExpandMetadata(column='Metadata_DoesNotExist', labels=['A','B'])  -> identical (KeyError, WARNING, master returned)
== AppendString(column='DoesNotExist') [bare label is auto-prefixed to Metadata_DoesNotExist] -> identical
```
The failure is all-or-nothing:
```
ok-alone result: ['a_ok', 'b_ok']
ok+bad via _apply_post_to_master: ['a', 'b']   # the successful first op's output is discarded too
```
The only exception that is not swallowed is a `ValueError` matching `_is_metadata_integrity_error` (line 110: messages starting "Metadata columns normalizing to " or "Metadata aliases ... conflicting non-null values").

## 8. JSON duplicate keys

`$S/p8/dup.json` has `pipe_cfgs` = `det: OtsuDetector`, `blur: BlurGauss`, `det: TriangleDetector`.
```
ops: [('det', 'TriangleDetector'), ('blur', 'BlurGauss')]
warnings raised: []
validate_pipeline: (True, None)
```
`from_json` uses a plain `json.loads` (`_serializable_pipeline.py:279` at `81d19ec`; this report first said `:283`, which is the `_deserialize_pipeline_config` call below it, corrected per review R31) with no `object_pairs_hook`, so the last duplicate silently wins. Because a Python dict keeps a key's first insertion position, **TriangleDetector now runs before BlurGauss**, although it appears after it in the file. The CLI validator accepts the file.
