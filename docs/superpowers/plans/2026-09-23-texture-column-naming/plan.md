# Plan: MeasureTexture column naming — `Texture{scale}px_{Feature}-deg{angle}|avg`

Worktree: `.worktrees/texture-column-naming` · branch `feat/texture-column-naming`

## Goal

Change emitted texture columns from `Texture_<Feature>-deg000-scale05` /
`Texture_<Feature>-avg-scale05` to `Texture{scale}px_<Feature>-deg{angle}` /
`Texture{scale}px_<Feature>-avg`, e.g. `Texture5px_Contrast-deg0`, `Texture5px_Contrast-avg`.

## Decisions (defaults chosen without further input — override before Task 1 if wrong)

| # | Decision | Why |
|---|----------|-----|
| D1 | **No zero-padding**: `Texture5px`, `deg0/45/90/135`. | Literal reading of the template. Padding existed only to make `scale` sortable/fixed-width; not requested. |
| D2 | **Recognize both** old and new headers in `TEXTURE.member_for_header`; **emit new only**. | Existing OME-Zarr stores / parquet tables still resolve to `TEXTURE` members (descriptions, producer grouping, GUI). Cost: one extra regex alternative. |
| D3 | **No `--mode migrate` column rewrite.** | Migrate is provenance-only by contract; renaming stored measurement columns is a separate, larger decision. Old columns stay valid via D2. |
| D4 | `category()` stays `"Texture"`; the scale token is part of the *emitted* prefix only. | `category()` is a class-level constant used by prefix derivation and the schema docs; per-run scale cannot live there. |
| D5 | `matrix_name` arg of `get_headers` is already unused in the name — leave as is. | Out of scope. |

## The real risk: prefix-based consumers

The convention "column = `Category_Label`" no longer holds: the prefix is now `Texture5px_`,
which does **not** start with `Texture_`. Every `startswith("Texture_")` consumer silently
stops matching (no error). Known sites:

- `src/phenotypic/_gui/results_viewer/colony_view/_grid.py` — `_MEASUREMENT_PREFIXES`
  derives `"Texture_"`; used at `_grid.py:284` and `_scatter_tab/_layout.py:309` to exclude
  measurement columns from axis pickers. **Would start offering texture columns as grid axes.**
- `src/phenotypic/analysis/_error_cutoffs.py` — uses bare `"Texture"` (no `_`): already fine; add a test.
- `src/phenotypic/util/_measurement_outputs.py` — uses `owns_header`, so fine once D2 lands.
- `src/phenotypic/analysis/qc/_expected_vs_detected.py:87` — `owns_header`; fine.

Fix for the GUI sites: replace prefix-string matching with `MeasurementInfo.owns_header`
(schema ownership, per the project rule against string-prefix semantics). Do a repo-wide
`grep -rn 'Texture'` for anything I missed (Task 5).

## Tasks (TDD; each ends with its own focused tests)

1. **Schema emit + recognize** — `schema/_texture.py`
   - New `_TEXTURE_HEADER_RE` accepting new form
     `^Texture(?P<scale>\d+)px_(?P<label>[^-]+)-(?:deg(?P<angle>\d+)|avg)$` **and** the legacy form.
   - `get_headers(scale, matrix_name=None)` emits new names; keep ordering (all `deg` blocks per
     feature, then all `avg`) because `_measure_texture.py:190-256` slices by position.
   - Add round-trip test `parse(emit(member, scale, angle)) == (member, scale, angle)` for
     scales 1, 5, 10, 100, 250 and angles 0/45/90/135/avg; assert legacy headers still resolve;
     assert `Texture5px_Contrast` (no suffix) and `Texture_Contrast-deg0` (mixed) are rejected.
   - Rewrite `tests/unit/schema/test_dynamic_headers.py` texture tests.
   - Prove the round-trip test can fail: temporarily break the regex, confirm red.
2. **Producer** — `measure/_measure_texture.py`: update docstring (lines ~63-65, example names);
   verify multi-scale merge (`meas.merge(..., on=OBJECT.LABEL)`) has no column collisions —
   distinct `Texture5px_`/`Texture10px_` prefixes make this strictly safer. Add a
   `MeasureTexture(scale=[5, 10])` test on `load_synth_yeast_plate()` asserting the exact column
   set and that every column is owned by `TEXTURE`.
3. **GUI prefix consumers** — `_grid.py` / `_scatter_tab/_layout.py`: exclude via `owns_header`
   (or a derived prefix set that includes the `Texture` family), update
   `test_measurement_prefixes.py`, `test_scatter_grouping.py`, `tests/gui/.../test_grid.py`,
   and the `FEATURES.md` row at line 424 (CI-gated ledger — see `gui-tutorial-capture` skill).
4. **Analysis prefix check** — `_error_cutoffs.py`: add a test with `Texture5px_Contrast-avg`
   to pin that `"Texture"` bare-prefix matching still selects it.
5. **Sweep** — grep `Texture_`, `-avg-scale`, `-deg0`, `scale0` across `src tests docs/source`
   (exclude `docs/build`, `docs/_build`, superpowers historical specs/plans — they document past
   decisions). Update `schema/CLAUDE.md` (the "texture" scheme line ~136), `schema/_texture.py`
   class docstring, `docs/source/measurements_ref/measurements/index.rst`, and decide on
   `data/meas/all_meas.csv` (packaged sample data used by ICC tests: leave as legacy since D2
   keeps it valid, but confirm no test asserts new-format names against it).
6. **Provenance/schema-version check** — grep for any measurement-column manifest or
   `measurement_columns` schema version that should be bumped, and any golden fixtures
   containing texture headers (`tests/**/*.json|*.parquet|*.csv`).

## Verification

- Per task: the touched test files only (`uv run pytest <files>`).
- End: affected surface once, `QT_QPA_PLATFORM=offscreen`, via the `run-phenotypic-test` skill
  (schema, measure, analysis, gui/results_viewer, ci startup-import guards).
- `uv run ruff check --fix <changed paths>` (explicit paths only) and `uv run mypy src/phenotypic`.

## Out of scope

Rewriting stored tables (D3); changing `category()`; renaming other measurers' columns.
