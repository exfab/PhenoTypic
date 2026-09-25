# Phase-2 code review: consumer migration (Tasks 7–9)

**Scope:** `1decc582` (prefabs, GUI defaults, differential script), `4a531dcc` (src docstrings,
bundled CSVs, scripts, example image), `3b70d667` (user docs), `cb0a52ca` (57 swept test files).
HEAD `cb0a52ca`. Reviewer: phase-2 deep gate, 2026-09-25.

**Counts:** CRITICAL 0 · HIGH 1 · MEDIUM 3 · LOW 8. `[USER]` marks a design fork: MEDIUM-1 and MEDIUM-3.

Commands the orchestrator ran for this review; output quoted verbatim where cited:
- the three `test_linear_softplus.py` node ids below: **3 failed**;
- `probe_measure_cost.py` (scratchpad), for the timing under Informational.

---

## HIGH

### HIGH-1 — The sweep left three `test_linear_softplus.py` tests red: the `\b` regex skips suffixed names

**Where:** `tests/unit/analysis/test_linear_softplus.py:361-362`, `:395-396`, `:438`.

**Evidence.** `size_rename.pl` matches `\bShape_Area\b`. In `"Shape_Area_stderr"` the character
after `Area` is `_`, a word character, so there is no boundary there and the literal is not
rewritten. The same lines' `on="Shape_Area"` *was* rewritten to `on="Size_Area"`. The code under test
derives the helper columns from `on`
(`src/phenotypic/analysis/abc_/_linear_softplus_base.py:225-227`, `:359`, `:369`:
`f"{self.on}_stderr"`, `f"{self.on}_std_pool"`), so the test frames now carry helper columns the
model never looks up.

The orchestrator ran them, and all three fail:
- `TestWeighting::test_stderr_floor_quantile_lifts_sub_quantile_sigma` fails with `assert None is not None`.
  `_resolve_y_stderr` finds no `Size_Area_stderr` and returns `None` (`:234-235`).
- `TestWeighting::test_stderr_floor_disabled_via_subclass` fails with `assert None is not None`.
- `TestWeighting::test_mixed_singleton_pool_broadcasts` fails with
  `AssertionError: assert 'Shape_Area_std_pool' in Index([... 'Size_Area_stderr', 'Size_Area_std_pool'], ...)`.

The orchestrator's repo-wide grep for the suffixed form `Shape_(Area|…)_[A-Za-z0-9]` finds hits
only in this file. The prefix-joined form finds nothing. The collision audit could not catch this
case, because it looked for successor names that already existed, not for partial rewrites.

**Fix.**
- In the two `_resolve_y_stderr` tests, derive the names from the model:
  `f"{m.on}_stderr"` / `f"{m.on}_std_pool"`, so that the next rename cannot desynchronise them.
- In `test_mixed_singleton_pool_broadcasts`, use `pool_col = f"{m.on}_std_pool"`.
- If literals are preferred, spell them `Size_Area_stderr` / `Size_Area_std_pool`.
- Add one line to plan Task 8 or to the `size_rename.pl` header: the script does not rewrite a retired
  name followed by `_suffix`. After a run, grep for `Shape_(…)_[A-Za-z0-9]`.

The commit message says the swept surface "runs in the phase-2 Slurm gate". That gate should
show these three failures; they are not flaky.

---

## MEDIUM

### MEDIUM-1 [USER] — The bundled CSVs publish a hull *perimeter* under the header `Size_ConvexArea`

**Where:** `src/phenotypic/data/meas/all_meas.csv` and `area_meas.csv`, header columns 117
(`Size_ConvexArea`) and 121 (`Shape_Solidity`). The files are served by `load_meas()` /
`load_area_meas()` (`data/_sample_image_data.py:78-95`).

**Evidence.** The values were computed before `d846ca4a` (the `ConvexHull.area` → `.volume` fix).
The row-2 data field 22 is a quoted `"(1, 1)"`, so a naive split shifts by one. After aligning for it:

| row | Size_Area | Size_Perimeter | Size_ConvexArea | Shape_Solidity |
|---|---|---|---|---|
| 1 | 2019 | 168.37 | **160.01** | **12.62** |
| 2 | 2233 | 177.78 | **168.62** | **13.24** |
| 3 | 2319 | 182.27 | **172.24** | **13.46** |

Across **all 129 rows of both files**, `Size_ConvexArea < Size_Area` and `Shape_Solidity > 1`.
`ConvexArea ≈ 0.95 × Perimeter` is the hull's perimeter, and Solidity ≈ 12.6 is Area / hull-perimeter.

Before 0.20.0 this was already wrong, under `Shape_ConvexArea`. Now it is published under a
`Size_` header whose `desc` says "computed from the convex hull … slightly smaller than the pixel
count" (`schema/_size.py:47-50`). The change note's rename table also says
`Shape_ConvexArea → Size_ConvexArea`, which invites users to trust the column. The C4 notes call
this "the pre-existing ConvexArea `.volume` drift", which understates it: the value is about 12× off
and is a different quantity.

The retained EDT columns, by contrast, are a real but small concern. They hold whole-objmap EDT
values (for example, MeanBoundaryDist 8.50 against the R/3 = 8.45 of a disk of area 2019). These
differ from the new per-object definition only for touching or border colonies, which are rare on
these gridded plates. No action is needed beyond the note below.

**Fix (choose one):**
- (a) Recommended: set `Size_ConvexArea` and `Shape_Solidity` to empty (NaN) in both CSVs. Neither
  can be recomputed without the source images. The consumers (`test_icc`, `test_log_growth_model`,
  `test_pipeline_analyze`, and the `correct_edge_effects` / `fit_logistic_growth` notebooks) read
  only `Size_Area` and metadata; confirm with
  `grep -rn "ConvexArea\|Solidity" tests docs/source/how_to`.
- (b) Drop the two columns.
- (c) Keep the values and state in the `load_meas` / `load_area_meas` docstrings that
  `Size_ConvexArea` / `Shape_Solidity` predate the `.volume` fix and are not valid.

Whichever is chosen, one sentence in the loader docstrings should say the example data predates
0.20.0: its `Shape_*BoundaryDist` values used the whole-plate EDT.

### MEDIUM-2 — The MeanBoundaryDist interpretation is backwards for uniformly thin colonies

**Where:** `docs/source/explanation/measurement_metrics_biological_meaning.md:43` (added in
`3b70d667`), paraphrasing `src/phenotypic/schema/_shape.py:70-72`, the `MEAN_BOUNDARY_DIST` desc,
which is published into every run's README.

**Claim:** "High values relative to InscribedRadius indicate a compact, convex colony; low values
indicate a thin or filamentous one."

**Evidence (closed form).** Take the mean EDT divided by the inscribed radius:
- a disk (and any tangential polygon: square, triangle) gives exactly **1/3**, since
  ∫₀ᴿ(R−r)2πr dr / πR² = R/3;
- an infinite strip of half-width w (EDT = w − |y|) gives a mean of w/2 and an inscribed radius of w,
  so the ratio is **1/2**.

So a uniformly thin, elongated colony (a streak, or a filament of constant width) scores *higher*
than a compact disk. The ratio falls only when a compact core sets the inscribed radius and thin
appendages add low-distance area (runners, spurs, a hyphal fringe). As written, the sentence is
right for "core plus runners" and wrong for "thin all over".

**Fix.** In both the desc and the doc row, reword to something like: "For an ideal disk it is 1/3 of
Size_InscribedRadius. Thin appendages on a compact body (runners, a hyphal fringe) pull the ratio
below 1/3; a uniformly thin, elongated colony raises it toward 1/2."

An agent may author `desc`, but not `bio_desc` (root CLAUDE.md), so this fix is in scope. It is a
factual correction, not a design fork. If `logic_validation_scripts/.../radial_invariants.py` is
extended, add a strip check next to the disk check.

### MEDIUM-3 [USER] — The same-name trap also reaches metric-qualified model columns, and the change note does not say so

**Where:** `src/phenotypic/schema/_change_notes.py` (`SIZE_SHAPE_SPLIT_NOTE`) and spec §6/§7.

**Evidence.** `metric_token` strips the longest category prefix (`util/_measurement_outputs.py:259-266`;
`tests/unit/util/test_metric_token.py::test_every_category_strips_to_the_remainder` pins it for
every category). A growth model fit on the retired `Shape_MaxRadius` (inscribed radius) emitted
`LogGrowthModel_MaxRadius_r`. The same model fit on the new `Size_MaxRadius` (reach) emits the
**identical** header, `LogGrowthModel_MaxRadius_r`, with a different meaning. The same holds for
`MeanRadius` and `MedianRadius`, and for `LinearLagModel` / `LinearCapAndLagModel` / `DoubleSoftplus`
headers. A user comparing an old `analysis.parquet` against a new one sees no rename at all. The
note's "compare old data against the successor" covers measurement columns only.

(Bare-label `RemoveByFeature` is **not** affected. `feature="MeasureShape", value="MaxRadius"` now
raises `ValueError` from `_resolve_column`, which is loud. `MeasureSize` had no radius columns before
0.20.0, so no saved `feature="MeasureSize", value="MaxRadius"` exists.)

**Fix.** Add one sentence to the note's "Same name, different value" paragraph:
"Model outputs are named by the stripped label, so `<Model>_MaxRadius_*`, `<Model>_MeanRadius_*`
and `<Model>_MedianRadius_*` fitted before and after 0.20.0 share a name but not a meaning."
This changes the §7 note text, which is why it is tagged `[USER]`.

---

## LOW

- **LOW-1 — `test_standalone_bundle.py`'s fixture pipeline no longer emits its data column.**
  `tests/unit/gui/analysis/test_standalone_bundle.py:64,70-71`. The data is `Size_Area`, but the
  seeded pipeline is `meas=[MeasureShape()]`. No assertion reads measurer attribution (the tests
  check ops and `columns_for`), so nothing is wrong today, but the fixture now describes a run that
  cannot exist. Fix: `meas=[MeasureSize()]`, or both measurers.

- **LOW-2 — `test_metric_token.py::test_strips_known_category` lost its cross-category example.**
  `tests/unit/util/test_metric_token.py:9-10`. Before the sweep it asserted one `Shape_` and one
  `Size_` strip. Now both lines are `Size_`. `test_every_category_strips_to_the_remainder` still
  covers it, so nothing is lost functionally. Fix: make line 9 `metric_token("Shape_Circularity") == "Circularity"`.

- **LOW-3 — Generic prose in `_measurement_info.py` still names a header that no longer exists.**
  `src/phenotypic/schema/_measurement_info.py:353, 432, 460, 546, 567` (`'Shape_Area'` as the
  example). These lie outside the toy-`SHAPE` doctest (`:278-317`) that A7 excluded. C4 flagged them
  as safe to edit by hand. Fix: change them to `Size_Area` (the static-scheme example, as in
  `schema/CLAUDE.md:8`), and leave `:278-317` alone.

- **LOW-4 — The `MeasureFeatures` docstring shows a two-column `MeasureSize` output.**
  `src/phenotypic/abc_/_measure_features.py:237-242` and `:440-443` show `Area  IntegratedIntensity`
  as the whole `MeasureSize` frame. That was already non-runnable prose, but it now contradicts the
  thirteen-column measurer. Fix: add `...` after `IntegratedIntensity`, or list a few `Size_*` headers.

- **LOW-5 — The doc table gives Solidity a range of "0–1".**
  `measurement_metrics_biological_meaning.md:39`. Spec §2 item 3 says Solidity can slightly exceed 1,
  because the hull runs through pixel centres. The new ConvexArea row on the same page says so too.
  This predates the change, but the page was rewritten in this phase. Fix: "≈0–1 (can slightly
  exceed 1)".

- **LOW-6 — The 07 notebook's MeasureSize highlights omit the radius family.**
  `docs/source/tutorials/notebooks/07_measuring_and_exporting.ipynb:115-118`. The list gives Area,
  IntegratedIntensity and the axes, but not `Size_RobustMeanRadius` / `Size_MaxRadius`, which are
  the reason `MeasureSize` grew. Optional: add one bullet.

- **LOW-7 — The drift bench scripts lost their size coverage.**
  `scripts/bench_fungi_pipeline_drift.py:51` and `scripts/bench_layer_dtype_drift.py:51` configure
  `MeasureShape()` without `MeasureSize()`. They compare every emitted column across dtype and
  pipeline variants, so area, perimeter, axes and radii silently dropped out of their comparison. The
  spec §8 consumer list missed them because they name no retired column. Fix: add
  `"size": MeasureSize()`.

- **LOW-8 — The committed scatter tutorial screenshots show the old axes.**
  `_capture_scatter` now binds `SHAPE.SOLIDITY × SHAPE.CIRCULARITY`
  (`scripts/capture_gui_tutorial_screenshots.py:1345`). The committed
  `docs/source/_static/gui_images/scatter/02-05` still show Perimeter × Area. `19_scatter.md` names
  no axis, so no text changes. Fix: regenerate at the next capture run. A ledger gate does not catch
  this.

---

## Informational (no action)

- **Prefab cost (item 6), measured.** One run on the orchestrator's node, median of 3, synthetic
  plate, Otsu, 552 objects:
  `MeasureSize=0.866s MeasureShape=0.585s MeasureIntensity=0.545s per-object Size=1.57ms`.
  - Adding `MeasureSize` to a prefab raises the Size+Shape measurement cost from 0.59 s to about
    1.45 s per plate, roughly +0.9 s per image.
  - Both measurers now build a per-object EDT and a `ConvexHull` for the same objects, which is
    duplicated work.
  - On a full pipeline that is dominated by detection and enhancement (the Heavy* prefabs, Texture),
    this is modest. I consider it acceptable for the default path.
  - A shared per-image geometry cache is a possible later optimisation, out of scope here.
  - The second probe plate (`load_plate_72hr`) crashed in the probe itself: it returns an ndarray,
    not an `Image`. No number was obtained for it.
- **No fixture pins a prefab's `meas` list.**
  - `tests/integration/gui/builder/test_legacy_pipeline_json.py` and
    `tests/gui/builder/test_state_roundtrip.py:795` compare a prefab against its own round-trip, so
    they are self-relative.
  - `tests/fixtures/builder_dag/legacy_popover_pipeline.json` has `MeasureSize` with no params, which
    loads with the new defaults.
  - No JSON or YAML fixture under `tests/`, `src/` or `docs/` names `MeasureShape` or a retired column.
- **C4 notes misstatement (notes only).** `c4-notes.md` says `growth_curves.ipynb` "spells
  `Size_Area` against `all_meas`". It actually measures with `MeasureSize()` (`:62`, `:91`), so it
  never depended on the CSV rename.

## Verified clean

- **Same-name mapping.** Every `Shape_MaxRadius` / `MeanRadius` / `MedianRadius` rewrite maps
  correctly:
  - `tests/unit/plotting/test_plot_meas_time_series.py:245,262,276,377` → `Shape_MeanBoundaryDist`;
  - `tests/unit/plotting/test_pipeline_bindings.py:94` → `Shape_MedianBoundaryDist`;
  - the bundled CSVs → `Shape_*BoundaryDist`, and neither carries `Shape_MaxRadius`;
  - `diff_migration_scenarios.py:84-96` uses the table as given.
  - No prose outside `_size.py` / `_change_notes.py` equates a `Size_*Radius` with a retired column.
    `test_edge_correction.py:588-610` uses a bare `MeanRadius` as a synthetic column, which is fine.
- **No dynamic header construction reads retired names.** There is no `f"Shape_{…}"`, and no
  `getattr(SHAPE, …)` or `SHAPE[...]` anywhere in src or scripts. `_error_cutoffs.MEASUREMENT_PREFIXES`
  is a prefix list, and `Shape_` there stays valid. The tune scorer, QC, dashboards, plotting and
  GUI defaults read `SIZE.AREA` / `SHAPE.<retained>` through the schema. Only the five analysis
  defaults carried literals, and those are fixed.
- **No internal hot path instantiates `MeasureSize` or `MeasureShape`.** `KeepSectionLargest` and
  `MeasureIntensity` are decoupled. `RemoveByFeature` does so only when a user configures it (plan L5).
- **The sweep's other semantic cases are sound:**
  - `test_scatter_grouping.py`: `MEAS` gains `MeasureSize`, and the negative assertion is kept;
  - `test_cli_output_manager.py:350-351,412`: `Shape_Circularity` / `Shape_Solidity` keep the
    MeasureShape split alive;
  - `test_grid.py:55`;
  - `test_recipe_state_load_warnings.py:539/580`: original and current `on` still differ.
  - No duplicate dict keys or parametrize ids remain: HEAD holds no swept file with both an old name
    and its successor in one literal set, apart from the two that were fixed.
- **Excluded test files.** `test_measurement_join_migration_run.py`, the equivalence test and the
  change-note test were left untouched, as intended.
- **Docs.**
  - The Size table's rows paraphrase the `SIZE` descs accurately. The InscribedRadius elongation
    caveat is kept.
  - The `{ref}` target `measurement-info-size` resolves (`_extensions/measurements_ref.py:49-52`).
  - Every notebook that reads a `Size_*` column either configures `MeasureSize`
    (`07`, `06`, `measure_colony_size_intensity`, `growth_curves`) or reads `load_meas()`, whose
    CSV headers were renamed (`correct_edge_effects`, `fit_logistic_growth`).
    `linear_softplus_model` mentions `Size_Area` only in prose.
  - `schema/CLAUDE.md:179` uses enumeration order: CIRCULARITY, then MIN_FERET_DIAMETER.
