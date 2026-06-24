# Analysis subpackage reorganization + Edge Correction GUI section — design

**Date:** 2026-06-23
**Branch:** `refactor/output-and-docs`
**Status:** approved (design); ready for implementation plan

## Goal

Reorganize `src/phenotypic/analysis/` so that, mirroring the existing
`analysis/qc/` subpackage:

1. The outlier **filters** move into a new `analysis/filter/` subpackage.
2. The **edge corrector** moves into a new `analysis/edge/` subpackage and gains a
   new abstract intermediate base class `EdgeCorrection` (alongside `ModelFitter`
   / `QualityCheck` in `abc_`), its own registry category `"Edge Correction"`,
   **and a dedicated GUI section** in the analysis sub-app.
3. The non-class **helper** modules move into a new private `analysis/_helper/`
   subpackage.

The public API (`from phenotypic.analysis import X`) and serialized
`pipeline.json` files must keep working unchanged.

## Decisions (from brainstorming + review)

- **Filters = only the two outlier removers**: `MADOutlierRemover`,
  `TukeyOutlierRemover`. `EdgeCorrector` does **not** go into `filter/`.
- **Models stay at top level**: `LogGrowthModel`, `DoubleSoftplus`,
  `LinearSoftplus`. `ErrorCutoffFinder` also stays top-level (public class, not a
  helper).
- **Hard cutover**: update every internal + test reference; no re-export shims.
- **EdgeCorrection = abstract template base** (not a thin marker): factors the
  strategy-agnostic edge-correction config + topology up; `EdgeCorrector`
  implements the concrete capping strategy.
- **`EdgeCorrection` lives in `abc_` and is NOT re-exported from
  `analysis/__init__.py`** — only from `analysis/abc_/__init__.py`. This keeps it
  invisible to the registry's `inspect.getmembers(phenotypic.analysis)` walk, so
  the abstract base is never registered as an operation (no exclusion-tuple edit
  needed).
- **GUI gets a full dedicated "Edge Correction" section** (not just a registry
  label) — a section stack mirroring the filter stack, with add/remove/edit/
  preview wiring, FEATURES.md row, and test updates.

## Why the moves are safe (the two name-resolution seams)

- **Pipeline (de)serialization** — `_serializable_pipeline.py`'s
  `_find_class_in_phenotypic` resolves analyzer classes **by name** by walking
  the `phenotypic.analysis` namespace. As long as `analysis/__init__.py` keeps
  re-exporting every public class, old `pipeline.json` files round-trip
  unchanged. Class **names** do not change. `OperationInfo.module` (stored as
  `obj.__module__`) will change for moved classes, but no test asserts on the
  module path of an analysis class.
- **GUI registry** — `gui/_operation_registry.py::_discover_analyzers` walks
  `inspect.getmembers(phenotypic.analysis)` (the public module object), relying
  on the `__init__` re-exports, not submodule paths.

## Target package structure

```
analysis/
  __init__.py              # re-exports unchanged → public API identical
  abc_/
    __init__.py            # + EdgeCorrection
    _set_analyzer.py
    _model_fitter.py
    _quality_check.py
    _edge_correction.py    # NEW — EdgeCorrection(SetAnalyzer, ABC) intermediate
    _linear_softplus_base.py
  qc/                      # unchanged
  filter/                  # NEW
    __init__.py            # exports MADOutlierRemover, TukeyOutlierRemover
    _mad_outlier.py        # git mv
    _tukey_outlier.py      # git mv
  edge/                    # NEW
    __init__.py            # exports EdgeCorrector
    _edge_correction.py    # git mv; EdgeCorrector now subclasses EdgeCorrection
  _helper/                 # NEW (private)
    __init__.py            # re-exports public error-report funcs
    _qc_math.py            # git mv
    _error_report.py       # git mv
    _inoculum_prior.py     # git mv
  _double_softplus.py      # stays (model)
  _linear_softplus.py      # stays (model)
  _log_growth_model.py     # stays (model)
  _error_cutoffs.py        # stays (ErrorCutoffFinder)
```

## The `EdgeCorrection` intermediate (abstract template base)

New `abc_/_edge_correction.py`, mirroring `ModelFitter`/`QualityCheck` (both
`SetAnalyzer` subclasses that give their family its own GUI category).

Factor **up** into `EdgeCorrection(SetAnalyzer, ABC)` (strategy-agnostic):
- Grid-layout config fields + validators: `nrows`, `ncols`, `connectivity`,
  `time_label`; `_validate_connectivity`, `_validate_grid_dim`.
- Pure grid-topology helper: `_surrounded_positions` (geometry).
- One abstract correction hook (e.g. `_correct_group`) the concrete strategy
  implements.

Keep **down** in `EdgeCorrector(EdgeCorrection)` (`edge/_edge_correction.py`):
- `top_n`, `pvalue` + `_validate_top_n`; `_perm_test`; the capping algorithm
  (`analyze`/`_apply2group_func`/`_calculate_group_stats`); plotting
  (`show`/`_show_collapsed`/`_show_individual`/`results`);
  `_measurement_infoclass = EDGE_CORRECTION`; `_original_data` PrivateAttr.

`EdgeCorrector` remains a `SetAnalyzer` subclass (not a `ModelFitter`), so every
`issubclass(x, SetAnalyzer)` / `not issubclass(x, ModelFitter)` pipeline branch
keeps treating it as a filter-chain analyzer. The exact method-by-method split is
finalized in the implementation plan.

## Registry category

`gui/_operation_registry.py::_discover_analyzers` — add one branch (ordered after
the `ModelFitter` check, before the `else: "Filter"` fallback):

```python
elif issubclass(obj, EdgeCorrection):
    category = "Edge Correction"
```

Import `EdgeCorrection` from `phenotypic.analysis.abc_`. The string
`"Edge Correction"` must match `_choices_for_category("Edge Correction")`
character-for-character (the layout lookup is exact-match → silent empty dropdown
on mismatch). `EdgeCorrection` is abstract and not re-exported at the
`analysis` top level, so it never appears in the walk and needs no exclusion.

## GUI "Edge Correction" section — the architecturally significant part

**Key constraint:** the pipeline keeps **all** non-model `SetAnalyzer` instances
in a single `pipeline._filters` dict (`get_filters()`/`set_filters()`). There is
no separate "edge" slot in `pipeline.json`. So an `EdgeCorrector` lands in the
same dict as the outlier filters, and `pipeline.json` keeps its existing
`filters` key (no schema change, no new serialization slot, no migration).

Therefore the GUI must **partition `get_filters()` by registry category** at
every read/write site, splitting it into a "filter" stack (category `"Filter"`)
and an "edge" stack (category `"Edge Correction"`), both backed by the one dict.

**Single-source the partition + index mapping.** Introduce one helper, e.g.
`_filter_items_for(pipeline, category) -> list[tuple[str, SetAnalyzer]]`, that
returns the ordered `(key, instance)` sublist of `get_filters()` whose registry
category matches. Every stack build, remove, edit, and preview path uses it so a
stack's *local* index → *global* dict key mapping is computed identically
everywhere. **Removal/edit must never `pop(index)` against the full
`list(get_filters().items())`** (the Explore-flagged aliasing bug) — instead:
take the category sublist, read the key at `index`, then `del`/reassign that key
in the full dict and `set_filters(full_dict)`. Add/remove/edit on either stack
re-renders **both** stacks (shared-dict reindex).

### GUI surfaces to add (mirroring `filter`)

`gui/analysis/_ids.py`:
- `SectionKind = Literal["post","filter","edge"]`;
  `InstantiationKind = Literal["post","filter","model","edge"]`;
  `PlotSectionKind = Literal["filter","model","edge"]`.
- New `ANALYSIS_EDGE_STACK`, `ANALYSIS_EDGE_ADD_DROPDOWN`; `edge_section_id()`;
  add all three to `__all__`. (`section_remove_button_id`/`preview_button_id`/
  `plot_slot_id`/`plot_param_id` already take the widened Literals — no new fns.)

`gui/analysis/_layout.py`:
- `_build_edge_panel` sibling of `_build_filter_panel`
  (`_choices_for_category("Edge Correction")`, `section_label="edge"`,
  `ANALYSIS_EDGE_ADD_DROPDOWN`, `ANALYSIS_EDGE_STACK`, threads
  `columns_provider`); insert into `build_app_layout` between filter and model.
- `build_section_stack`: add `elif kind == "edge":` reading the edge sublist via
  the partition helper; **narrow the `"filter"` arm** to the filter sublist; add
  the `edge_section_id` arm and fire the plot-controls branch for `"edge"`.
- `pipeline_header_children`: stop counting edge correctors as filters — emit a
  separate edge chip and subtract from the filter count.

`gui/analysis/_callbacks.py`:
- Move `EdgeCorrector` defaults out of `_FILTER_DEFAULTS` into `_EDGE_DEFAULTS`;
  `_KIND_DEFAULTS["edge"]`, `_KIND_MODULES["edge"] = ModulePath.ANALYSIS`.
- New `_add_edge` callback (writes into the shared `get_filters()` dict, rebuilds
  edge + filter stacks).
- `_remove_section`: add `elif kind == "edge":`; use the partition helper for the
  index→key mapping; output + rebuild the edge stack (and filter stack).
- `_on_param_edit` fan-in: add a 4th `Output(ANALYSIS_EDGE_STACK, ...)` and an
  `elif kind == "edge":` rebuild arm.
- `_resolve_preview_node`: add `if kind == "edge":` (edge sublist), narrow the
  `"filter"` arm to the filter sublist.
- `_apply_param_edit`: add `elif kind == "edge":` read + write-back via the
  shared dict using the partition helper.

`gui/analysis/_render.py`, `_plot_controls.py`, `_post_preview.py`,
`_recipe_state.py`, `_app.py`: **no kind-branching changes** — they are
kind-agnostic (introspect the instance / take `kind` as a free str), covered by
widening the Literals.

### CI ledgers (mandatory)
- **`gui/FEATURES.md`**: add an "Edge Correction section stack" row mirroring the
  "Filter section stack" row (`#analysis-edge-stack` +
  `_choices_for_category("Edge Correction")`, `✅ shipping`, resolvable
  `Test ref`); update the pipeline-header-summary row and the
  registry-discovery row (currently names only Filter/Model). The
  `features-md-gate` job rejects any `gui/` PR that doesn't touch FEATURES.md;
  pre-commit validates `Test ref` resolves.
- **`gui/WORKFLOWS.md`**: widen the existing `analysis` workflow row description
  to "post / filter / edge / model" (no new tutorial page required unless we want
  a dedicated edge walkthrough — out of scope).

## Reference updates (non-GUI, hard cutover)

| Location | Change |
|---|---|
| `analysis/__init__.py` | `._mad_outlier`/`._tukey_outlier` → `.filter`; `._edge_correction` → `.edge`; `._error_report` → `._helper._error_report` |
| `abc_/__init__.py` | add `from ._edge_correction import EdgeCorrection` + `__all__` |
| `abc_/_linear_softplus_base.py` | `phenotypic.analysis._inoculum_prior` → `..._helper._inoculum_prior` |
| `filter/_mad_outlier.py`, `filter/_tukey_outlier.py` | `from . import _qc_math` → `from .._helper import _qc_math` |
| `qc/_max_modz.py`, `qc/_relative_mad.py`, `qc/_tukey_fraction.py` | `analysis._qc_math` → `analysis._helper._qc_math` (imports) |
| `tests/unit/analysis/test_max_modz.py`, `test_relative_mad.py`, `test_tukey_fraction.py` | `._qc_math` → `._helper._qc_math` |
| `tests/unit/analysis/test_log_growth_model.py:554` | `._tukey_outlier` → `.filter._tukey_outlier` |

### Missed surfaces caught by plan review (must also change)
- **`_qc_math.py` own module docstring** (lines 4–5): Sphinx `:class:` refs to
  `phenotypic.analysis._mad_outlier.MADOutlierRemover` /
  `._tukey_outlier.TukeyOutlierRemover` → `...filter._mad_outlier` /
  `...filter._tukey_outlier`.
- **`_qc_math.py` own doctest imports** (`>>> from phenotypic.analysis._qc_math
  import ...`, ~6 instances) → `..._helper._qc_math`.
- **`qc/_max_modz.py:59`** Sphinx `:func:` docref to
  `phenotypic.analysis._qc_math.modified_z_scores` → `..._helper._qc_math...`.
- **Doctest `>>> from phenotypic.analysis._qc_math import ...` lines** inside
  `_max_modz.py`, `_relative_mad.py`, `_tukey_fraction.py` (not collected by the
  current pytest run — no `--doctest-modules` — but fixed for correctness).

### Explicitly NOT changing
- `_error_cutoffs` consumers (`_cli/_cli_error_outputs.py`,
  `gui/results_viewer/_error_tab/_callbacks.py`,
  `tests/unit/analysis/test_error_cutoffs.py`) — `_error_cutoffs.py` stays
  top-level.
- All `from phenotypic.analysis import X` consumers — public API unchanged.
- `_image_pipeline_core.py`, `_serializable_pipeline.py`,
  `util/_measurement_outputs.py`, `gui/analysis/_render.py`/`_plot_controls.py`,
  `sdk_/_qc_recipe/*` — they import `SetAnalyzer`/`ModelFitter`/`QualityCheck`
  from `analysis.abc_`, unaffected; `EdgeCorrection` is additive there.
- `pipeline.json` schema — no new slot; edge correctors keep serializing into
  the `filters` dict. `tests/e2e/gui/test_analysis_app.py` `filters`/`model` key
  asserts stay valid.

## GUI-section test impact (from the surface map)

| File | Change |
|---|---|
| `tests/unit/gui/test_operation_registry.py` | Add `TestEdgeCorrectionCategory` (EdgeCorrector in `"Edge Correction"`, excluded from `"Filter"`/`"Model"`); update existing `EdgeCorrector`-registered asserts (lines ~227–229, 269, 421). |
| `tests/unit/gui/test_param_forms.py:217` | `EdgeCorrector` expected category `"Filter"` → `"Edge Correction"`. |
| `tests/integration/gui/test_analysis_column_dropdowns.py` | Prefix asserts `analysis-filter*` → `analysis-edge*` for the EdgeCorrector fixture. |
| `tests/integration/gui/test_analysis_plot_preview.py` | `_resolve_preview_node(..., "filter", 0)` → `"edge"`; `kinds == {"filter","model"}` → includes `"edge"`. |
| `tests/unit/gui/analysis/test_plot_controls.py` | Optional `"edge-0-*"` keyed cases (kind-agnostic; not strictly breaking). |
| `tests/unit/gui/analysis/test_recipe_state_load_warnings.py` | Verify slot expectations (unresolved edge class still reports slot `"filter"` — shared dict; acceptable). |
| `tests/unit/analysis/test_edge_correction.py` | Update imports to `analysis.edge`; optional `EdgeCorrection` ABC contract test. |

## Verify, don't assume
1. `git mv` every relocated file (preserve history).
2. `pyproject.toml` uses `[tool.setuptools.packages.find]` (`where=["src"]`, no
   explicit list) → `filter/`/`edge/`/`_helper/` auto-discovered. **Confirmed in
   review; no pyproject change.**
3. Docs: add `phenotypic.analysis.filter.rst` + `phenotypic.analysis.edge.rst`
   stubs (mirroring `phenotypic.analysis.qc.rst`) and both to the `Subpackages`
   toctree in `phenotypic.analysis.rst`. Keep private `_helper` out of the public
   toctree. (Regenerate via `sphinx-apidoc` if that produced the existing files.)
4. Final repo-wide grep for `analysis._qc_math`, `analysis._mad_outlier`,
   `analysis._tukey_outlier`, `analysis._edge_correction`,
   `analysis._error_report`, `analysis._inoculum_prior` returns only the new
   locations.

## Verification gates
- `uv run ruff check --fix`
- `uv run mypy src/phenotypic` (watch the widened `Literal` kinds flow through
  `_ids.py`/`_callbacks.py`/`_plot_controls.py`).
- `QT_QPA_PLATFORM=offscreen uv run --group dev --group qt-test --extra gui
  pytest` on: `tests/unit/analysis/`, `tests/unit/gui/`, `tests/integration/gui/
  test_analysis_*`, `tests/e2e/gui/test_analysis_app.py`, and the serialization
  round-trip (`tests/unit/core/test_pipeline_analyze.py`,
  `test_pipeline_qc_serialization.py`).
- `uv run python scripts/capture_gui_tutorial_screenshots.py` if any visible
  chrome changed (the new edge panel does) — commit refreshed PNGs.

## Out of scope (YAGNI)
- Moving the model classes into an `analysis/model/` subpackage.
- Renaming any class or measurement column.
- Re-export shims at old private paths.
- A new pipeline-level "edge" serialization slot (edge correctors stay in the
  `filters` dict).
- A dedicated edge-correction tutorial page in WORKFLOWS.md.
