# Plan: MeasureTexture column naming — `Texture_{scale}px-{deg{angle}|avg}-{Feature}`

Worktree: `.worktrees/texture-column-naming` · branch `feat/texture-column-naming`

> Supersedes the first draft of this file (target `Texture{scale}px_…`), which broke the
> `Texture_` category prefix. The format below keeps the prefix, so every prefix-based
> consumer is unaffected; the change concentrates in the schema's emit/parse pair.

## Goal

| | Current | New |
|---|---|---|
| directional | `Texture_Contrast-deg000-scale05` | `Texture_5px-deg0-Contrast` |
| average | `Texture_Contrast-avg-scale05` | `Texture_5px-avg-Contrast` |

Grouping moves from *feature-first* to *scale → direction → feature*, so a scale's
columns read as one block and the feature label ends the name.

## Decisions (open for review before Task 1)

| # | Decision | Rationale |
|---|---|---|
| D1 | **No zero-padding**: `Texture_5px-deg0-…`, `deg45`, `deg135`. | Literal reading of the template. Trade-off: plain `sorted()` (results-viewer dropdowns, REMBI catalog) orders `Texture_10px-…` before `Texture_5px-…` and `deg135` before `deg45`. Padding (`Texture_05px-deg000-…`) would fix lexical sort. |
| D2 | **Unit suffix `px` on the scale** (`Texture_5px-…`). | User decision: tells readers the scale is a pixel offset. The literal `px` also makes the new pattern unambiguous against any bare-digit token. |
| D3 | **Recognize both formats, emit new only.** | *Required, not optional* — see "Why D3 is required" below. |
| D4 | **No `--mode migrate` rewrite** of stored columns. | Migrate is provenance-only; D3 keeps old tables valid. |
| D5 | `category()` stays `"Texture"`; `get_headers(scale, matrix_name=None)` signature unchanged. | Every prefix consumer derives `Texture_` from `category()`. |
| D6 | **Fix the multi-scale merge bug** in the same change (Task 2). | Pre-existing: `measure/_measure_texture.py:143` discards the `merge` result, so `scale=[5, 10]` silently returns only scale 5. The new name exists to make scale legible; shipping it while multi-scale is broken would be odd. Separable if preferred. |

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
| `schema/_texture.py:11-13` `_TEXTURE_HEADER_RE` | Two patterns: new `^(?P<cat>[A-Za-z0-9]+)_(?P<scale>\d+)px-(?:deg(?P<angle>\d+)\|avg)-(?P<label>[A-Za-z0-9]+)$`; legacy kept as `_LEGACY_TEXTURE_HEADER_RE`. Unambiguous: legacy ends `-scale\d+`, new ends in a label. |
| `schema/_texture.py:147-157` `member_for_header` | Try new, then legacy; same `cat`/`label` lookup. |
| `schema/_texture.py:159-175` `get_headers` | Emit `f"{cat}_{scale}px-deg{angle}-{label}"` / `f"{cat}_{scale}px-avg-{label}"`. **Keep order exactly**: feature-outer × angle-inner (52), then 13 averages in feature order. |
| `measure/_measure_texture.py:140-144` | Assign the merge result (D6). |

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
  yields e.g. `LogGrowthModel_5px-avg-Contrast_r` (was `…_Contrast-avg-scale05_r`);
  `parse_qualified_header` anchors on the member-label suffix, so a leading digit is fine. Cosmetic.
- `sdk_/_rembi_manifest.py:94` — catalog now lists `5px-avg-Contrast` etc.; cosmetic ordering.
- `refine/_remove_by_feature.py:163` — bare-label suffix match (`col.split("_",1)[-1]`) never matched
  texture labels before (`Contrast-deg000-scale05`) and still doesn't (`5px-avg-Contrast`); full names work.
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
   - Red first: exact expected headers for `get_headers(5)` — `[0] == "Texture_5px-deg0-AngularSecondMoment"`,
     `[52] == "Texture_5px-avg-AngularSecondMoment"`, length 65.
   - Round-trip: every header of `get_headers(s)` for s ∈ {1, 5, 10, 100, 250} resolves to the
     member whose label it ends in.
   - Legacy: `Texture_Contrast-deg000-scale05` / `…-avg-scale05` → `TEXTURE.CONTRAST`.
   - Negatives: `Texture_Contrast`, `Texture_5px-Contrast`, `Texture_5px-deg0`, `Texture_5px-avg-Nope`,
     `Shape_5px-avg-Contrast`, `TextureGray_Contrast-deg000-scale05`, `Texture_5-deg0-Contrast` (missing `px`).
   - Mutation proof: break the new regex (e.g. `deg\d{3}`) → round-trip red; drop the legacy
     branch → legacy test red.
   - Update the docstrings and comment in the first three text rows.
2. **Producer** (`measure/_measure_texture.py`, new `tests/unit/measure/test_measure_texture.py`)
   - Red first on `load_synth_yeast_plate()` + a detector: `MeasureTexture(scale=[5, 10])` returns
     exactly `{Object_Label} ∪ get_headers(5) ∪ get_headers(10)` (131 columns) — fails today (D6).
   - **Order guard:** per object, `Texture_5px-avg-F == mean(Texture_5px-deg{0,45,90,135}-F)` for every
     feature F. Mutation-prove by swapping two angles in `get_headers`.
   - Every emitted column satisfies `TEXTURE.owns_header`.
   - Fix the merge; update the Returns docstring.
3. **Downstream recognition guards** (tests only)
   - `_is_gui_metadata_column` and `_is_layout_metadata_column` return False for both
     `Texture_5px-avg-Contrast` and `Texture_Contrast-avg-scale05`. Mutation-prove by deleting the
     legacy branch.
   - `split_measurements` puts new-format texture columns under `MeasureTexture`.
   - Update `test_scatter_grouping.py` and `test_grid.py` literals/docstrings.
4. **Text sweep** — `_measurement_info.py:382`, `schema/CLAUDE.md:136-137`,
   `_error_cutoffs.py:30-31`; optional README generator + MCP spec line.

## Verification

- Per task: touched test files only (`uv run pytest <files>`).
- End, once: the affected surface via the `run-phenotypic-test` skill with
  `QT_QPA_PLATFORM=offscreen` — `tests/unit/schema tests/unit/measure tests/unit/util
  tests/unit/analysis tests/unit/gui/results_viewer tests/gui/results_viewer
  tests/unit/core/test_image_pipeline.py tests/unit/core/test_pipeline_serialization.py`, plus
  importers of `_metadata_context` / `_expected_vs_detected` (derive with grep).
- `uv run ruff check --fix <changed paths>`; `uv run mypy src/phenotypic`.

## Out of scope

Rewriting stored tables; the `TextureGray_` sample CSV and migration golden; renaming other
measurers' columns.
