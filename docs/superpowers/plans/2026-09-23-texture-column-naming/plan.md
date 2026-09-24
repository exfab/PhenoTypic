# Plan: MeasureTexture column naming — `Texture_{scale:02d}px-{deg{angle:03d}|avg}-{Feature}`

Worktree: `.worktrees/texture-column-naming` · branch `feat/texture-column-naming`

> Supersedes the first draft of this file (target `Texture{scale}px_…`), which broke the
> `Texture_` category prefix. The format below keeps the prefix, so every prefix-based
> consumer is unaffected; the change concentrates in the schema's emit/parse pair.

## Goal

| | Current | New |
|---|---|---|
| directional | `Texture_Contrast-deg000-scale05` | `Texture_05px-deg000-Contrast` |
| average | `Texture_Contrast-avg-scale05` | `Texture_05px-avg-Contrast` |

Grouping moves from *feature-first* to *scale → direction → feature*, so a scale's
columns read as one block and the feature label ends the name.

## Decisions (open for review before Task 1)

| # | Decision | Rationale |
|---|---|---|
| D1 | **Zero-padded, same widths as today**: scale `{scale:02d}` (`05px`, `10px`), angle `{angle:03d}` (`deg000`, `deg045`, `deg090`, `deg135`). | User decision: plain `sorted()` (results-viewer dropdowns, REMBI catalog) then orders scales and angles numerically. `:02d` is a *minimum* width, so scales ≥ 100 emit 3 digits and sort out of order past 99 — same limitation as today. The recognizer is strict on padding (`\d{2,}` scale, `\d{3}` angle) so `parse(emit(x)) == x` is the only accepted spelling. |
| D2 | **Unit suffix `px` on the scale** (`Texture_05px-…`). | User decision: tells readers the scale is a pixel offset. The literal `px` also makes the new pattern unambiguous against any bare-digit token. |
| D3 | **Recognize both formats, emit new only.** | *Required, not optional* — see "Why D3 is required" below. |
| D4 | **No `--mode migrate` rewrite** of stored columns. | Migrate is provenance-only; D3 keeps old tables valid. |
| D5 | `category()` stays `"Texture"`; `get_headers(scale, matrix_name=None)` signature unchanged. | Every prefix consumer derives `Texture_` from `category()`. |
| D6 | **One scale per `MeasureTexture`.** `scale: int = 5` (validity bound `ge=1`). Multi-scale = several `MeasureTexture` entries in one pipeline, e.g. `meas=[MeasureTexture(scale=3), MeasureTexture(scale=5)]`. | User decision. This removes the multi-scale loop instead of fixing it. The loop was broken anyway: `measure/_measure_texture.py:143` discards the `merge` result, so `scale=[5, 10]` has only ever emitted scale 5. |
| D7 | **Legacy list input:** a **one-element** list (`[5]`) is coerced to `5`; a **multi-element** list raises `ValidationError`: "MeasureTexture measures one scale; add one MeasureTexture per scale." | Every saved `pipeline.json` today serializes `"scale": [5]` (the field was `List[int]`). That JSON is re-read by recompile/finalize (`phenotypicCLI.py:3202,3310`), the results-viewer QC recompute (`_gui/results_viewer/_app.py:438`, degrades to a warning) and the QC rebuild (`_qc_tab/_rebuild.py:380`, raises). Rejecting `[5]` would break every existing run folder. **Cost:** a run folder whose pipeline used `scale=[3, 4]` stops loading in those three paths. Alternative: coerce a multi-element list to its first element with a warning, which is exactly what those runs produced. Rejected here because "one scale" should be unambiguous. |
| D8 | **Two measurers with the same scale are not guarded.** | Same-scale duplicates emit identical columns. `_merge_on_object_labels` (`_image_pipeline_core.py:1480-1497`) merges on columns whose values are equal, and adds `_merged` suffixes where they differ (e.g. NaN). This is user error with harmless-but-ugly output. The docstring says "one per distinct scale". A guard would need pipeline-core changes; out of scope unless requested. |

## Why D3 is required — legacy stores

If `TEXTURE.owns_header` stops recognizing `Texture_Contrast-deg000-scale05`, nothing raises.
Old columns fall through two "unknown external header" classifiers whose PascalCase regex
`^[A-Z][A-Za-z0-9]*_[A-Z][A-Za-z0-9]*$` rejects `-`, and are then **treated as metadata**:

- `_gui/shell/_metadata_context.py:89,109-120` (`_is_gui_metadata_column`) → the analysis
  GUI (`_gui/analysis/_callbacks.py:586,930`) renames them `Metadata_Texture_…`.
- `analysis/qc/_expected_vs_detected.py:76-110` (`_is_layout_metadata_column`) → same for QC layout.
- `util/_measurement_outputs.py:119,128` (`split_measurements`) → they become context columns
  in every per-measurer split (also CLI output, `_cli/_cli_output_manager.py:1376`).
- `util/_measurement_outputs.py:279-284` (`_describe_column`) → descriptions drop out of output keys.

The same cascade hits **new-format** columns if the regex isn't updated, so the tests in
Tasks 1 and 3 guard both directions.

## Call-site inventory

### Must change (code)

| Site | Change |
|---|---|
| `schema/_texture.py:11-13` `_TEXTURE_HEADER_RE` | Two patterns: new `^(?P<cat>[A-Za-z0-9]+)_(?P<scale>\d{2,})px-(?:deg(?P<angle>\d{3})\|avg)-(?P<label>[A-Za-z0-9]+)$`; legacy kept as `_LEGACY_TEXTURE_HEADER_RE`. Unambiguous: legacy ends `-scale\d+`, new ends in a label. |
| `schema/_texture.py:147-157` `member_for_header` | Try new, then legacy; same `cat`/`label` lookup. |
| `schema/_texture.py:159-175` `get_headers` | Emit `f"{cat}_{scale:02d}px-deg{angle:03d}-{label}"` / `f"{cat}_{scale:02d}px-avg-{label}"`. **Keep order exactly**: feature-outer × angle-inner (52), then 13 averages in feature order. |
| `measure/_measure_texture.py:96` | `scale: List[int] = [5]` → `scale: int = Field(5, ge=1)` (D6). |
| `measure/_measure_texture.py:101-114` | Replace `_coerce_scale_to_list` with a `mode="before"` validator: 1-element list → int; longer list → `ValueError` with the D7 message. |
| `measure/_measure_texture.py:140-144` | Delete the multi-scale loop; `return self._compute_haralick(scale=self.scale, …)` directly (drop the `functools.partial` if it no longer earns its keep). |
| `measure/_measure_texture.py:33-47,60-66` | Docstring: "at one pixel-offset scale"; `scale` arg is a single int; add "for several scales, add one `MeasureTexture` per scale" with a runnable doctest example of two entries; Returns examples → `Texture_05px-deg000-Contrast` / `Texture_05px-avg-Contrast`. |
| `prefab/_grid_section_pipeline.py:91,133,199-204` | `texture_scale: int \| list[int]` → `int`; docstring. |
| `prefab/_heavy_round_peaks_pipeline.py:101,202,274-279` | Same. |
| `prefab/_round_peaks_pipeline.py:52,104,129-134` | Same (`int \| List[int]` → `int`; drop unused `List` import if orphaned). |
| `prefab/_filamentous_fungi_pipeline.py:130`, `_heavy_otsu_pipeline.py:80`, `_heavy_watershed_pipeline.py:76` | Already `int` — no change. |

### Order-dependent, no edit (guard with a test)

`measure/_measure_texture.py:190-194` slices `[:-13]`/`[-13:]`; `:240` fills
`texture_statistics.T.ravel()` (feature-major, angle-minor); `:251` averages in strides of 4.
If `get_headers` order changes, values land under the wrong names **silently**.

### Recognized automatically once the regex lands (no edit)

`util/_measurement_outputs.py:119,128,279`; `_gui/shell/_metadata_context.py:92-120`;
`analysis/qc/_expected_vs_detected.py:79-110`; `_cli/_cli_output_manager.py:1376`.

### Safe — `Texture_` prefix kept or no parsing (no edit)

- `_gui/results_viewer/_scatter_tab/_grouping.py:68-72` — prefix fallback for `TEXTURE`.
- `_gui/results_viewer/colony_view/_grid.py:98-140,284`, `_scatter_tab/_layout.py:309` —
  `_MEASUREMENT_PREFIXES` → `"Texture_"`.
- `analysis/_error_cutoffs.py:34-42,122-127` — bare `"Texture"` prefix.
- `util/_measurement_outputs.py:259-270` `metric_token` — a growth fit on a texture column now
  yields e.g. `LogGrowthModel_05px-avg-Contrast_r` (was `…_Contrast-avg-scale05_r`);
  `parse_qualified_header` anchors on the member-label suffix, so a leading digit is fine. Cosmetic.
- `sdk_/_rembi_manifest.py:94` — catalog now lists `05px-avg-Contrast` etc.; cosmetic ordering.
- `refine/_remove_by_feature.py:163` — bare-label suffix match (`col.split("_",1)[-1]`) never matched
  texture labels before (`Contrast-deg000-scale05`) and still doesn't (`05px-avg-Contrast`); full names work.
  **User-config note:** a saved `RemoveByFeature(value="Texture_Contrast-avg-scale05")` no longer
  matches new runs — changelog item.
- `sdk_/_metadata_helpers.py:111-157`, `_gui/run_console/_request_safety.py:383`,
  `_gui/results_viewer/_measurement_source.py`, `_measurement_routes.py:91`,
  `sdk_/_measurement_tables.py`, plotting, QC/ICC/SetAnalyzer/post, prefabs — no texture parsing.
- No `isidentifier`, `DataFrame.query/eval`, natsort, polars regex selectors, or Dash ids/CSS built
  from column names anywhere in `src/` — a leading digit after `Texture_` is safe.

### Text / docstrings

| Site | Update |
|---|---|
| `schema/_texture.py:8-10` | Comment on `{scale:02d}` / `\d{2,}` → describe the two patterns. |
| `schema/_texture.py:18-36` | Class docstring format lines. Renders in the API reference. |
| `schema/_texture.py:149,162-163` | `member_for_header` / `get_headers` docstrings; state the ordering contract in `get_headers`. |
| `measure/_measure_texture.py:60-66` | Returns-section examples (already wrong today — no `Texture_` prefix). |
| `schema/_measurement_info.py:382` | `header_scheme` doc: "`-deg/-scale` suffix scheme". |
| `schema/CLAUDE.md:136-137` | `texture` scheme line. |
| `analysis/_error_cutoffs.py:30-31` | "matrix/scale suffix" wording. |
| `_cli/_cli_readme_generator.py:224-231` | *Optional:* README lists `Texture_Contrast` (never a real column, pre-existing). Could print the pattern when `header_scheme() == "texture"`. |

### Tests

| Site | Change |
|---|---|
| `tests/unit/schema/test_dynamic_headers.py:46-65` | **Breaks** (`"-avg-scale" in h` → `StopIteration`). Rewrite; drop the `:02d` rationale in the >2-digit test. |
| `tests/unit/gui/results_viewer/test_scatter_grouping.py:40-44` | Passes via prefix but pins the old spelling → new spelling. |
| `tests/gui/results_viewer/colony_view/test_grid.py:45-48,58,73` | Same; its docstring claims it matches `get_headers`. |
| `tests/unit/util/test_measurement_outputs.py:133-176` | No literal; relies on `get_headers()[0]` being AngularSecondMoment — still true. Leave. |
| `tests/unit/analysis/test_error_cutoffs.py:170` | `TextureGray_Contrast` (older legacy). Leave. |
| `tests/unit/core/test_image_pipeline.py:50` | `MeasureTexture(scale=[3, 4], quant_lvl=8)` → would raise under D7. Replace with two entries, `"MeasureTexture3": MeasureTexture(scale=3, …)`, `"MeasureTexture4": MeasureTexture(scale=4, …)` — which also exercises repeated measurers through `measure()`/`_merge_on_object_labels`. |
| `tests/unit/core/test_pipeline_serialization.py:137-148` | `test_list_parameters` uses `scale=[3, 5, 7]` as its list-param example. Replace with two `MeasureTexture` entries round-tripping (`scale` 3 and 5, keys deduped `MeasureTexture` / `MeasureTexture_1`); if list-param coverage matters, point the test at another list-typed field. |
| `tests/unit/gui/results_viewer/test_scatter_grouping.py:14` | `"params": {"scale": [5]}` → `{"scale": 5}` (the `[5]` form keeps working via D7, but new pipeline.json writes an int). |

### Data / fixtures (leave, documented)

- `src/phenotypic/data/meas/all_meas.csv` — 105 `TextureGray_…-scale03/04` columns, an even older
  prefix that `TEXTURE` recognizes under neither format; no code reads it by texture name.
- `tests/migration/_goldens/measure.MeasureTexture.parquet` — `TextureGray_…`; already stale,
  Linux-only, not in CI `testpaths`. Out of scope.
- `docs/source/measurements_ref/` is generated from `rst_table()` at build time and gitignored.
- `docs/superpowers/**` historical specs/plans — leave.
  `specs/2026-08-12-phenotypic-mcp-server/03-tool-catalog.md:157` (draft, no code) has a stale
  example — optional one-line fix.

## Tasks (TDD)

1. **Schema emit + recognize** (`schema/_texture.py`, `test_dynamic_headers.py`)
   - Red first: exact expected headers for `get_headers(5)` — `[0] == "Texture_05px-deg000-AngularSecondMoment"`,
     `[52] == "Texture_05px-avg-AngularSecondMoment"`, length 65.
   - Round-trip: every header of `get_headers(s)` for s ∈ {1, 5, 10, 100, 250} resolves to the
     member whose label it ends in.
   - Legacy: `Texture_Contrast-deg000-scale05` / `…-avg-scale05` → `TEXTURE.CONTRAST`.
   - Negatives: `Texture_Contrast`, `Texture_05px-Contrast`, `Texture_05px-deg000` (no label), `Texture_05px-avg-Nope`,
     `Shape_05px-avg-Contrast`, `TextureGray_Contrast-deg000-scale05`, `Texture_05-deg000-Contrast` (missing `px`), `Texture_5px-deg0-Contrast` and `Texture_05px-deg45-Contrast` (unpadded).
   - Mutation proof: break the new regex (e.g. `deg\d{2}`) → round-trip red; drop the legacy
     branch → legacy test red.
   - Update the docstrings and comment in the first three text rows.
2. **Producer: single scale** (`measure/_measure_texture.py`, new `tests/unit/measure/test_measure_texture.py`)
   - Red first:
     - `MeasureTexture(scale=5).scale == 5` (an int);
     - `MeasureTexture(scale=[5]).scale == 5`;
     - `MeasureTexture(scale=[3, 4])` raises `ValidationError` naming "one MeasureTexture per scale";
     - `scale=0` raises.
   - `MeasureTexture(scale=5).measure(image)` on `load_synth_yeast_plate()` plus a detector returns
     exactly `{Object_Label} ∪ get_headers(5)` (66 columns).
   - **Order guard:** per object, `Texture_05px-avg-F == mean(Texture_05px-deg{000,045,090,135}-F)`
     for every feature F. Mutation-prove by swapping two angles in `get_headers`.
   - Every emitted column satisfies `TEXTURE.owns_header`.
   - **Legacy JSON:** a pipeline JSON fragment with `"scale": [5]` loads via `ImagePipeline.from_json`.
   - Implement the field/validator change, delete the loop, update the docstrings.
3. **Multi-scale via repeated measurers** (`tests/unit/core/test_image_pipeline.py`,
   `test_pipeline_serialization.py`)
   - `ImagePipeline(meas=[MeasureTexture(scale=3), MeasureTexture(scale=5)])`: `measure()` output
     contains both `get_headers(3)` and `get_headers(5)`, and no `_merged` columns.
   - `to_json`/`from_json` round-trips both entries with their scales.
   - Update the two existing list-scale tests (see the Tests table).
   - `split_measurements` puts both scales' columns under the one `MeasureTexture` group. It groups
     by class, so this should hold with no edit; pin it.
4. **Prefabs** — narrow `texture_scale` to `int` in `_grid_section_pipeline.py`,
   `_heavy_round_peaks_pipeline.py` and `_round_peaks_pipeline.py`; update their docstrings.
   Run the prefab tests (`tests/unit/prefab`, if present) plus the serialization smoke test.
5. **Downstream recognition guards** (tests only)
   - `_is_gui_metadata_column` and `_is_layout_metadata_column` return False for both
     `Texture_05px-avg-Contrast` and `Texture_Contrast-avg-scale05`. Mutation-prove by deleting the
     legacy branch.
   - `split_measurements` puts new-format texture columns under `MeasureTexture`.
   - Update `test_scatter_grouping.py` and `test_grid.py` literals/docstrings.
6. **Text sweep** — `_measurement_info.py:382`, `schema/CLAUDE.md:136-137` (also its
   `MeasureTexture` example at `:162-170`, which shows `scale: List[int] = [5]` /
   `self.scale[0]`), `_error_cutoffs.py:30-31`; optional README generator + MCP spec line.
   Changelog notes: new column names; `scale` is a single int; multi-element `scale` lists in old
   `pipeline.json` no longer load (D7).

## Verification

- Per task: touched test files only (`uv run pytest <files>`).
- End, once: the affected surface via the `run-phenotypic-test` skill with
  `QT_QPA_PLATFORM=offscreen` — `tests/unit/schema tests/unit/measure tests/unit/util
  tests/unit/analysis tests/unit/gui/results_viewer tests/gui/results_viewer
  tests/unit/core/test_image_pipeline.py tests/unit/core/test_pipeline_serialization.py
  tests/unit/prefab tests/smoke/test_serialization.py`, plus
  importers of `_metadata_context` / `_expected_vs_detected` (derive with grep).
- `uv run ruff check --fix <changed paths>`; `uv run mypy src/phenotypic`.

## Out of scope

Rewriting stored tables; the `TextureGray_` sample CSV and migration golden; renaming other
measurers' columns; a pipeline-level guard against two same-scale `MeasureTexture` entries (D8).
The tune annotation-coverage gate covers only `detect/` + `enhance/`, so `scale` becoming a
numeric `int` field does not enter it.
