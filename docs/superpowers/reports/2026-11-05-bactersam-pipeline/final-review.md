# Final review: BacterSamPipeline (feat/bactersam-prefab, 251e364fa..dfd90cf3d)

Reviewer: one fresh-context reviewer, read-only. Single pass over the whole diff
(8 files, +572/-7). Nothing was edited and nothing was committed.

## What I verified by running it

- **The fixture matches the run.** I compared the run's
  `deliverables/pipeline.json.pht-pipe` key by key with
  `tests/unit/prefab/fixtures/bactersam_reference_pipeline.json`. `pipe_cfgs`, `meas`,
  `nrows`, `ncols`, `post`, `filters` and `model` are identical. The fixture drops
  only `name`, `desc` and `plots`, as the plan says.
- **The defaults match the run with no normalisation.** The raw
  `BacterSamPipeline().to_json()` equals the raw fixture in `pipe_cfgs`, `meas`,
  `nrows` and `ncols`, **including key order**. No default differs from the run.
- These pass: `tests/unit/prefab/test_bactersam_pipeline.py`,
  `tests/unit/prefab/test_prefab_measures_size.py`, and
  `tests/gui/builder/test_state_roundtrip.py::TestPrefabRoundTrip`
  (39 passed, 1 xfailed). The prefab's doctest also passes
  (`--doctest-modules src/phenotypic/prefab/_bactersam_pipeline.py`).
- **Imports.** Importing `Sam2` and `_checkpoint_manager` after
  `phenotypic.abc_/correction/enhance/measure/refine` adds **no third-party
  module**. It adds only `phenotypic.detect.nn*` and the `phenotypic.detect`
  modules, which the other prefabs already load. That took about 160 ms on this
  node.
- **Preflight requirements, per op:** only `('GridLargest',)` reports
  `grid_image=True`, and only `('meas:MeasureColor',)` reports `rgb_input=True`.
  `MeasureNeighborDist` works on a plain `Image` (it checks `hasattr(image, "grid")`).
  `check_grid_preset` covers the pipeline's `nrows`/`ncols` preset as well.
- `tests/unit/ci/test_startup_imports.py` and
  `tests/smoke/test_serialization.py::test_prefab_roundtrip` also pass
  (141 passed in 382 s, exit 0). This finished after the report was first handed
  back.

## Strengths

- The op and measurer keys are the run's own keys, so the run JSON works as a
  golden fixture directly. Keeping four `MeasureTexture` measurers (D2), rather
  than one with four scales, keeps the `measurements_by_feature/` layout
  identical. That is the right call for "reproduce the run".
- Fixing `input_layer="detect_mat"` (D5) and not exposing it is well argued: it
  stops a kwarg from silently disabling two ops. The comment at the call site
  says why.
- Construction is keyword-only (tested). Construction loads no torch, and the
  test runs in a fresh subprocess, so another test cannot have imported torch
  first and made it pass.
- The test that colour correction and denoising are absent walks the whole tree,
  not only the top level.
- The builder xfail is strict, with `raises=KeyError`. It fails loudly once the
  builder learns GPU detectors, and fails on any other error. That is the right
  shape for a known gap.
- The docstring layout (Steps / Best For / Consider Also, Args in `__init__`)
  matches the local precedent in `_heavy_otsu_pipeline.py`.

## Issues

### Critical (Must Fix)

None found.

### Important (Should Fix)

1. **The golden test cannot detect a change in op order, and op order is the
   pipeline's semantics.**
   - File: `tests/unit/prefab/test_bactersam_pipeline.py:679-690` (`_sections`
     plus `test_default_config_equals_the_reference_run`), and
     `test_json_round_trip_keeps_class_and_config`.
   - Problem: `_sections` returns plain dicts, and Python `dict ==` ignores key
     order. Execution order *is* the key order of `pipe_cfgs`. I built a
     reference from the fixture with the ops reordered to
     `Crop, Blur, RemoveBG, BorderFilter, Detector, CircularityFilter,
     GridLargest, SmallObjectFilter` and `meas` reversed. The test's comparison
     still reported **equal (True)**.
   - Why it matters: this test is the plan's "decisive equivalence proof" (Task 3
     ruling). Swapping `SmallObjectFilter` and `GridLargest`, for example, changes
     which object survives in each grid section. The test would stay green, and
     so would every other test in the file. The only checks on order are the
     doctest (index 3 is `Detector`) and `gpu_path == ("Detector",)`. The current
     code is in the correct order (verified above); the guard is what is missing.
   - Fix: compare order as well, for example
     ```python
     def _sections(pipeline):
         data = json.loads(pipeline.to_json())
         return {
             "pipe_cfgs": list(data["pipe_cfgs"].items()),
             "meas": list(data["meas"].items()),
             "nrows": data["nrows"],
             "ncols": data["ncols"],
         }
     ```
     Or add `assert list(BacterSamPipeline().get_ops()) == list(reference.get_ops())`
     and the same for `meas`. Then check that the test fails on the reordered
     reference above.

2. **The GUI builder's "Load prefab" list now shows an entry that always fails.**
   - Files: `src/phenotypic/_gui/builder/_modal_browser.py:159` (one card per name
     in `phenotypic.prefab.__all__`) and `_gui/builder/_callbacks.py:7075-7096`
     (`click_prefab_card`).
   - Problem: the BacterSamPipeline card is listed like every other prefab.
     Clicking it raises `KeyError("Operation 'Sam2' not found in registry")`
     (`_gui/builder/_state.py:1227`). The user then sees an error toast built from
     that internal message. The note in the guide is in the docs, not in the GUI.
     The implementer's ruling records this cost, and the strict xfail pins it.
   - Why it matters: a user browsing prefabs in the GUI hits a dead end that
     looks like a bug, and nothing on screen tells them to use the CLI or Python
     instead.
   - Fix (executor's choice; each is small):
     - (a) In `_prefab_page`, render a prefab whose ops include a `GpuDetector` as
       a disabled card with the caption "Uses a GPU detector; load from Python or
       the CLI". `find_gpu_detectors` already exists for this.
     - (b) In `click_prefab_card`, catch this specific case and show that message
       instead of the KeyError text.

     Either one touches `_gui/` and so needs a `FEATURES.md` row (the
     `features-md-gate` job). If the executor keeps the ruling instead, record
     that this was accepted deliberately.

### Minor (Nice to Have)

1. **The `sam_min_mask_region_area` description does not match what upstream
   SAM2 does.** `_bactersam_pipeline.py` Args says "SAM2's own minimum mask area,
   in pixels, applied before the pipeline's filters". Upstream
   `sam2/automatic_mask_generator.py:95-97` describes a postprocessing step that
   removes *disconnected regions and holes* smaller than that area. The prefab
   inherits the wording from `Sam2`'s own docstring (`detect/nn/_sam2.py:164`),
   so the root fix belongs there. That is out of scope here; at most, soften the
   prefab's wording.
2. **The `border_size: int` annotation is narrower than the op's field**, which is
   `int | float | None`, where a value in (0, 1) means a fraction. That is
   harmless at runtime because nothing enforces the annotation. Widen it, or
   leave it as a deliberate simplification.
3. **`texture_scales` is not validated.** With a duplicated scale such as
   `(10, 10)`, a quick probe with two `MeasureTexture(scale=[10])` raised no error
   and produced no extra columns. The cost is a silently doubled texture
   computation. A one-line check in `__init__` (`len(set(texture_scales)) ==
   len(texture_scales)`, positive ints) would make the mistake obvious. Low value.
4. **Neither the docstring nor the guide says that the CLI needs
   `--image-type GridImage`.** Preflight refuses a plain `Image` with a clear
   remedy (`_cli_preflight.py:105`), so this is polish. One clause in the guide's
   "When to choose" paragraph would save the user that round trip.
5. **`benchmark` and `verbose` are not in `test_kwargs_reach_their_ops`.** Both
   are trivial pass-throughs to `super().__init__`; add them for completeness if
   wanted.
6. **The `Device`/`Sam2ModelSize` import is from a private module**
   (`phenotypic.detect.nn._helper._checkpoint_manager`). It is an import between
   internal modules, and the same pattern already appears in four sibling
   detector modules. The public-API rule concerns users, not internal modules.
   That file's own module docstring names it the single source of truth to
   import from. Acceptable as is.

## Review Focus checklist

1. **A plain `Image`.** Covered. `KeepSectionLargest` reports `grid_image`, and
   `check_grid_image` plus `check_grid_preset` refuse before any processing. The
   test checks the op-level declaration, not the CLI scope. The implementer's
   ruling on that is sound, because the CLI preflight tests own the scoping.
2. **Grayscale input.** Covered (`meas:MeasureColor` reports `rgb_input`). `Sam2`
   on `detect_mat` does not demand RGB.
3. **CLI staging.** Covered (`gpu_path == ("Detector",)`). `Sam2` is top level,
   and no `meas`/`post`/`filters`/`model` entry holds a GPU detector.
4. **JSON round trip.** `PrefabPipeline.from_json` re-tags the class, and the test
   asserts both the type and the config. The config comparison is blind to order
   (Important 1).
5. **Kwargs that never reach their op.** Every op kwarg is parametrised and
   reaches its op. `nrows`/`ncols` and `texture_scales` are tested separately.
   Only `benchmark`/`verbose` are untested (Minor 5).

**Docstring and guide claims I checked.** All of these hold:

- The CLI's three-stage split.
- The `--mode process --layer rgb` route: `ColorCorrector`, `CalibrateColorRpcc`
  and `ColorDenoise` are all `ImageCorrector`s, so they change `rgb`, and an
  `rgb` export carries their effect. Process stores are valid `--input`.
- The builder limitation: `OperationRegistry` imports `phenotypic.detect`, not
  `.nn`.
- The `crop_*` meaning (pixels removed from each edge, `_image_cropper.py:46-53`).
- The `crop_n_layers=1` gives four crops claim.
- The `sam_device="auto"` fallback.

**Other places that enumerate prefabs.**

- `tests/smoke/test_serialization.py` and `tests/unit/prefab/test_prefab_measures_size.py`
  iterate `__all__`. Both only construct and serialise; neither applies the
  pipeline, so no SAM2 download. Both pass.
- `tests/unit/core/test_pipeline_serialization.py:869` and
  `tests/integration/gui/builder/test_legacy_pipeline_json.py:37` use hand-written
  lists, so nothing there breaks. They simply do not include the new prefab.
- In `src/`, only the builder's modal lists prefabs (Important 2).

## Declined to judge

- **Whether SAM2 output on CPU, or on another GPU, matches the run.** That is a
  property of the real-image smoke job (Task 3), which runs outside the repo; I
  did not see its results.
- **The doctest uses no `load_synth_yeast_plate()`.** The project rule asks for
  runnable examples on that plate, but running SAM2 needs weights, and the plan
  explicitly allows an example that only constructs the pipeline.
- **Args placed in `__init__` rather than the class docstring.** It contradicts
  `docstring_style.md` ("Args appear in the class docstring"), but it matches
  every existing prefab. Fixing it here alone would make this prefab inconsistent
  with its siblings.
- **Whether the builder should learn GPU detectors** (registering
  `phenotypic.detect.nn`). That is a broader GUI change outside this branch.
- **The fixture's `pytest.warns(UserWarning, match="saved with phenotypic version")`.**
  It is slightly brittle if that warning text ever changes, but it is correct
  today, and it documents that normalisation happens.
- **The biological wording in Best For** ("small early colonies", "agar
  texture"). These are not biological claims in the `bio_desc` sense, and I was
  told not to propose any.

## Assessment

**Ready to merge: With fixes**

**Reasoning:** The prefab reproduces the run's configuration exactly, including op
order, and the Review Focus items hold. But the golden test that is meant to
prove that equivalence is blind to op order, and should be tightened before
merge. The GUI picker's dead-end entry should be either handled or accepted
explicitly as a ruling.
