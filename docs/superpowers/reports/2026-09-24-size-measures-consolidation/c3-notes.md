# C3 notes: Task 5 (Shape flip) and Task 6 (change note, 0.20.0)

Implementer notes for the orchestrator. All test, lint, mypy and mutation results
below were run by the orchestrator and relayed verbatim; none were run by the
implementer.

## Task 5

### Files
- `src/phenotypic/schema/_shape.py`: 9 Entries deleted, `MEAN_BOUNDARY_DIST` and
  `MEDIAN_BOUNDARY_DIST` added (plan descs, with no "Formerly reported as", per A4),
  class docstring replaced, and the `tier()` comment names the Feret diameters. Only
  `MIN_/MAX_FERET_DIAMETER` carry `tier=1` (A10).
- `src/phenotypic/measure/_measure_shape.py`: replaced per plan Step 4.
  `_calculate_feret_diameters` is byte-identical to main's.
- `src/phenotypic/schema/CLAUDE.md`: the straddler paragraph (Step 5).
- Tests: `tests/unit/measure/test_measure_shape.py` (new), and the
  `test_size_consolidation_equivalence.py` extension, plus the four schema/util
  substitutions (Step 1).

### Deviations from the plan text
1. **L4 port, adapted.** `test_merged_edt_would_fail_this_test` was ported from
   `5cad1dfa5:tests/unit/measure/test_measure_shape.py:65-81`. The fixture's objmap
   comes from a shared `_split_rectangle_objmap()` helper, so the control and the
   fixture cannot drift apart. The branch control asserted only the merged
   **max** (20/21). On this branch the max is `Size_InscribedRadius`, so it no longer
   lives in `MeasureShape`. The column the Shape test guards is the **mean**, so I
   added one assertion: the merged EDT's label-1 mean is not `4.6951 ± 1e-3`. Without
   it, the control would only prove that the fixture exercises a column this file
   does not test.
2. **`test_size_members_are_all_direct_phenotype` not added.** The plan asks for it
   in `test_classification.py`. The existing `test_tier1_primary_enums`
   (`tests/unit/schema/test_classification.py:151-156`) already asserts
   `resolved_kind == "primary"` and `resolved_tier == 1` for every `SIZE` member, so
   the new test would duplicate it exactly.

### Red run: which new tests were not demonstrated red
`test_emits_exactly_the_shape_schema` and
`test_degenerate_objects_give_nan_hull_measures_without_warning` passed on main's
code, because they are schema-driven or depend on members that survive. So did the
public-api, dynamic-headers and measurement-outputs substitutions. The first two were
proven able to fail by mutations M5.2–M5.4. The three substitutions only swap a
retired example member for a surviving one, so nothing in them guards new behaviour.

### Mutations (orchestrator's harness)
| id | mutation | node | result |
|---|---|---|---|
| M5.1 | whole-objmap EDT indexed by label | `test_touching_labels_do_not_inflate_boundary_distances` | KILLED |
| M5.1 control | same | `test_retained_shape_columns_keep_mains_values` | survived (expected: no synth colony touches another) |
| M5.2 | drop `Shape_Extent` from the frame | `test_emits_exactly_the_shape_schema` | KILLED |
| M5.3 | `np.zeros` instead of NaN fill | `test_degenerate_objects_give_nan_hull_measures_without_warning` | KILLED |
| M5.4 | unguarded `numer / denom` (RuntimeWarning) | same | KILLED |

### Step 8 classification (`src/`, `scripts/`)
Grep: `SHAPE\.(retired)\b` plus `\bShape_(retired)\b`, `-rnIE --exclude-dir=__pycache__`.
I also grepped for indirect lookups (`SHAPE[`, `getattr(SHAPE`, `value="Area"` and
the like) and found none.

- **Executable, fixed now:** none. The only executable hits were in
  `_measure_shape.py`, which Task 5 itself rewrites.
- **Executable, C4-owned (left):** `scripts/capture_gui_tutorial_screenshots.py:1342`
  (`SHAPE.PERIMETER`/`SHAPE.AREA`, an AttributeError once run) and `:189`, `:195`
  (`"on": "Shape_Area"` recipe defaults, which fail at run time on a new run). Task 7
  Step 6 covers all three.
- **Toy doctest (leave, A7/M1):** `src/phenotypic/schema/_measurement_info.py:289-317`
  (a local `class SHAPE`) and its generic `{category}_{label}` examples at :353, :418,
  :446, :532 and :553.
- **CSV data (C4 renames):** `src/phenotypic/data/meas/area_meas.csv:1` and
  `all_meas.csv:1`.
- **Docstring/prose (C4, Task 7 Step 5):** `util/_measurement_outputs.py:263`;
  `schema/_{linear_lag,linear_cap_and_lag,log_growth}_model.py:13-14`;
  `analysis/_{linear_lag,linear_cap_and_lag,log_growth}_model.py`;
  `analysis/filter/_{mad,tukey}_outlier.py`; `_cli/_cli_output_manager.py:312`;
  `_gui/_shared/_measurement_tint.py:177`; `sdk_/_metadata_helpers.py:783` (a doctest
  that uses `Shape_Area` only as an arbitrary column name, so it still passes);
  `scripts/make_measurement_example_images.py:28,50`.
- **schema/CLAUDE.md** `:8`, `:130`, `:178`: example headers. `:178` is the A7
  hand-fix (C4). `:8` and `:130` use `Shape_Area` as a generic header-format example,
  and that header no longer exists. C4 should decide whether to switch them to
  `Shape_Circularity`.
- **Deliberate (leave):** `schema/_size.py:79,89,109` ("not the value the retired …
  carried", spec §4.4) and the `_gui/FEATURES.md:821` ledger row.

## Task 6

### Files
- `src/phenotypic/schema/_change_notes.py` (new): `SIZE_SHAPE_SPLIT_NOTE`, the plan
  Step 3 text verbatim. It has only stdlib dependencies, per the schema import rule.
- `src/phenotypic/schema/_measurement_info.py`: the `change_note()` classmethod
  (after `rembi_module`), the `append_rst_to_doc` body, and a sentence in its
  docstring.
- `src/phenotypic/schema/_size.py`, `_shape.py`: the `change_note` override, plus
  `X.__doc__ = f"{X.__doc__}\n\n{X.change_note()}"` at module bottom.
- `docs/source/_extensions/measurements_ref.py`: `_class_section` emits the note
  between the heading and the table, and its docstring was updated.
- `src/phenotypic/__init__.py`: `0.20.0`.
- Tests: `tests/unit/schema/test_change_note.py` (new), the
  `test_measurements_ref_extension.py` extension, and the deletion of
  `test_version_is_0_19_0`.

### Observations (not fixed; for the reviewer)
- **An `append_rst_to_doc` override bypasses the hook.** `QUALITY_CHECK`
  (`schema/_quality_check.py:36-64`) overrides `append_rst_to_doc` with its own
  renderer and never calls `change_note()`. That is harmless today, since its note is
  `""`. A future note on an enum that overrides the renderer would silently render on
  the reference page and the enum page, but not in the measurer's class docs.
- **The note ends up in GUI-visible docstrings.** `_gui/_operation_registry.py:273,315`
  passes the raw `cls.__doc__` to the builder, so `MeasureSize`/`MeasureShape` now
  carry the RST note there, as they already carried the RST table. I checked
  `parse_param_descriptions` (`sdk_/_docstring_params.py:41`): the note sits at
  column 0 after `See Also:`, so it cannot extend an `Args:` block.
- The Step 10 docs build (orchestrator, Slurm) is the only check that the simple
  table inside the `versionchanged` body renders as a `<table>`. The unit tests check
  placement, not rendering.
