# C2 notes: decouple and consumers (Task 4, Task 4b, A6 differential script)

## Task 4

- `MeasureIntensity._operate` no longer imports or runs `MeasureShape`.
  - `Density` divides by `regionprops.area`, as main's `Shape_Area` did.
  - `ConvexDensity` divides by `convex_hull_area(coords)[1]` (`ConvexHull.volume`), as main's `Shape_ConvexArea` did.
- `KeepSectionLargest._operate` counts pixels with `np.bincount(objmap)[labels]`.
  - It merges onto `image.grid.info(include_metadata=True)` in the same order as `MeasureFeatures.measure(include_meta=True)`, so `idxmax` breaks ties as before.
  - The docstring line now reads "Counts each object's pixels" (A10/L2).

## Task 4b

- `MeasureSize()` is added to the six prefabs:
  - It is the first `meas` entry in each list.
  - `GridSectionPipeline`'s dict gets `"MeasureSize"` just before `"MeasureShape"`.
  - Each docstring's `Measurements:` line now starts with `MeasureSize`. `FilamentousFungiPipeline` has a bullet list there, so it gets a new first bullet instead.
- Both `abc_/_prefab_pipeline.py` code-block examples now include `MeasureSize()`.
- The five GUI analysis defaults are `str(SIZE.AREA)`, and `SIZE` is added to the schema import.
- The `meas` comment in the prefab test reads "normalised to a dict" (L7).

### Ledger conclusion (A9)

- **FEATURES.md needs one row.** The `features-md-gate` job in `.github/workflows/gui-checks.yml` fails a PR that touches `src/phenotypic/_gui/` without modifying `FEATURES.md`. A changed default is not new chrome, so the row goes in the "Refactor log (no user-visible changes)" section, which exists for this case. Its user-visible column says that an added filter/edge/model card now pre-selects `Size_Area`.
- **WORKFLOWS.md and the screenshots are unchanged.**
  - `_capture_analysis` never adds a card from a dropdown, and `PIPELINE_DOC` spells `"on"` explicitly (Task 7 Step 6 handles that).
  - `_capture_build_pipeline` adds a `MeasureSize` node but never opens its inspector.
  - So the three new `MeasureSize` form fields appear in no screenshot. They render through the generic param form.

### Wider-surface risk (flagged, not run in C2)

`RoundPeaksPipeline` backs the `simple_pipeline_json` fixture in 23 CLI test files (`tests/unit/cli`, `tests/integration/cli`) and in `tests/unit/sdk_/_migration_fixtures.py`. With `MeasureSize` added, those runs emit the extra `Size_*` columns and also compute the radial signatures. These files belong in the C3 phase-gate surface.

`tests/e2e/gui/test_analysis_app.py` adds `LinearLagModel` and `TukeyOutlierRemover` through the UI. Its fixture data carries only `Shape_Area`, so the new default `on=Size_Area` names a column the fixture lacks until Task 8 sweeps it. The test asserts only on class names and revisions, but A9 schedules it for Task 10 anyway.

## A6 differential script

`docs/superpowers/plans/2026-09-24-size-measures-consolidation/diff_migration_scenarios.py`
has two subcommands:

- **capture** runs the four scenarios through `tests.migration._runner.run_scenario`. It saves the results and each scenario's input objmap, then writes a manifest containing:
  - the git HEAD and dirty paths;
  - the sha256 of the measured `src` files, plus the frozen inputs and harness modules;
  - the library versions.
- **compare** applies the per-scenario rules. It fails, rather than skips, when anything is missing. It also fails when:
  - the two captures ran identical src;
  - the frozen inputs differ;
  - the two sides' input objmaps differ.

The EDT exclusion uses 8-connectivity contact with another label, or any pixel on the image border. The docstring proves that an object with no such contact gets an exact per-object EDT, and that an excluded object can only decrease (tip <= main).

## Results

As recorded in the commit message of `1decc582`:

- **D3**, main `81d19ec6` against this tip: PASS, 32 checks. KeepSectionLargest and
  MeasureIntensity are identical (dtype checked), and every moved column matches on all
  552 objects.
- **D4**, the no-op guard, comparing two copies of one capture: FAIL on "byte-identical src
  files", as designed.
- Red on the pre-change code: 6 prefab cases and the GUI default test.
- G4b: 8 passed.
- Surfaces: `tests/unit/prefab` + `gui/analysis` 71 passed; the prefab/analysis importers
  253 passed.
