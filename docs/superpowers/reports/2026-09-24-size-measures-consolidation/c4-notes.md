# C4 notes: Task 7 Steps 5-9 (src/, scripts/, bundled data)

Implementer notes for the orchestrator. The orchestrator ran every test, lint and mypy
result; the implementer ran only read-only commands and the committed rename script.

## Mechanical rewrite (Step 5)

The target list is in the session scratchpad at `c4_targets.txt`. It was built with
`grep -rlIE --exclude-dir=__pycache__ --include='*.py' --include='*.csv' --include='*.md'`
over `src scripts`, excluding `schema/_{change_notes,size,shape,measurement_info}.py` and
`_gui/FEATURES.md`. The script ran on 17 files:

- scripts: `capture_gui_tutorial_screenshots.py`, `make_measurement_example_images.py`
- data: `data/meas/{area_meas,all_meas}.csv` (header line only)
- docstrings/prose: `analysis/filter/_{mad,tukey}_outlier.py`,
  `analysis/_{linear_cap_and_lag,linear_lag,log_growth}_model.py`,
  `schema/_{linear_cap_and_lag,linear_lag,log_growth}_model.py`,
  `_cli/_cli_output_manager.py`, `_gui/_shared/_measurement_tint.py`,
  `sdk_/_metadata_helpers.py` (doctest; an arbitrary column name),
  `util/_measurement_outputs.py`, `schema/CLAUDE.md`

I read every rewritten docstring. All still read correctly. The growth-model
"stripped, e.g. ``Size_Area`` → ``Area``" example holds: `metric_token` strips the
`Size` category prefix, and the emitted `..._Area_v` names are unchanged.

None of the rewrites was in executable `src/` code, so there were no `SIZE`/`SHAPE`
import fixes in `src/`.

## Hand-fixes (Step 6)

- `schema/CLAUDE.md:179`: `SHAPE.get_headers()  # ['Shape_Circularity', 'Shape_MinFeretDiameter', ...]`.
  **Deviation from A7's literal** `['Shape_Circularity', 'Shape_Compactness', ...]`:
  `get_headers()` returns enumeration order, and `_shape.py` declares CIRCULARITY
  then MIN_FERET_DIAMETER, so the A7 text would be a wrong example.
- `schema/CLAUDE.md:8` and `:131`: switched to `Size_Area` by the script. They read
  correctly, because `SIZE` uses the default static header scheme, so `Size_Area` is a
  valid example of "the enum value is the category-prefixed header" and of the
  **static** scheme.
- `scripts/capture_gui_tutorial_screenshots.py` `_capture_scatter`: binds
  `[str(SHAPE.SOLIDITY), str(SHAPE.CIRCULARITY)]` unconditionally, and the comment
  was rewritten. The "verification run" is the real migrated run from
  `verify_scatter_fixture.py` (commit `36fe7bf4`). It predates 0.20.0 and configures
  only MeasureShape, so it has no `Size_*` column, while Solidity and Circularity keep
  their `Shape_*` names in both runs. The function's local import is already
  `from phenotypic.schema import SHAPE`.
  The recipe defaults at :189 and :195 now read `"on": "Size_Area"`. `PIPELINE_DOC`
  configures `MeasureSize` (:174), so they resolve.
- `scripts/make_measurement_example_images.py`: `_shape_area` → `_size_area` (the
  definition and its call). `dest = _OUT / "shape" / "area.png"` is kept.
- `analysis/_error_cutoffs.py` `MEASUREMENT_PREFIXES`: left as it is (a prefix list).

## Bundled CSVs

Both files change only their header line. Neither file carries `Shape_MaxRadius`.
Both carry `Shape_MedianRadius`/`Shape_MeanRadius`, now renamed to
`Shape_MedianBoundaryDist`/`Shape_MeanBoundaryDist`. Main computed those columns as
the per-label mean/median of `distance_transform_edt(image.objmap[:])`
(`git show 81d19ec66:src/phenotypic/measure/_measure_shape.py:130-137`). That is a
distance to the boundary, which is what the new names denote, so the rename is
value-correct in meaning. It follows the branch precedent `387cef18` ("header-only
rename with no value change").
**Caveat:** main's EDT ran over the whole objmap. For colonies that touched in the
source images, the stored values carry the inflation the new per-object EDT fixes
(mutation M5.1). The rename preserves those old values; it does not recompute them.
Area/Perimeter/ConvexArea/BboxArea/axis lengths move to `Size_*` with main's values.
MeasureSize now emits the same quantities (the equivalence test), apart from the
pre-existing ConvexArea `.volume` drift noted in A6.
No `Size_*` header existed before the rename, so it creates no duplicate columns.

## Observations for the orchestrator / other clusters

- **The shipped asset `_assets/measurements/shape/area.png` has "Shape_Area = 1430 px"
  drawn into its pixels.** The script's title now says Size_Area, but the PNG changes
  only if it is regenerated (`uv run python scripts/make_measurement_example_images.py`).
  I did not run it; it writes into `src/_assets`.
- **Tutorial screenshots (WORKFLOWS `scatter`, `_capture_scatter`, page
  `tutorials/gui/19_scatter.md`):** the scatter axes change from Perimeter × Area to
  Solidity × Circularity. When regenerated, `scatter/02_plot_settings.png` and the later
  shots (03-05) will show different axes. The committed PNGs still show the old
  `Shape_Perimeter`/`Shape_Area` axes. The tutorial prose names no axis, so no text
  change is needed. I did not regenerate them.
- Tests that read the renamed CSVs and spell retired names (for C5):
  `tests/unit/core/test_pipeline_analyze.py` (15 hits; outside the Step 8 surface),
  `tests/unit/analysis/test_log_growth_model.py` (30), `tests/unit/analysis/test_icc.py` (4).
- `docs/source/how_to/notebooks/growth_curves.ipynb` (C6) already spells `Size_Area`
  against `all_meas`, so it depends on this CSV rename landing.
- `schema/_measurement_info.py` generic examples (:353, :432, :460, :546, :567) still
  say `Shape_Area`, a header that no longer exists. They were left alone under the A7
  exclusion (the file's toy doctest). A hand edit of those five prose lines, not the
  script, would be safe if wanted.
