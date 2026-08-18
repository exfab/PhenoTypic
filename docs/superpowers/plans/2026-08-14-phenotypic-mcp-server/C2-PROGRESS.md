# C2 progress log — Tasks 6 and 7

Append-only. Written by the C2 agent so its state survives a lost session.

## Status

- **Started:** 2026-08-18, from `9c764780b` on `feat/mcp-server`.
- **T6 (split `_space.py`):** in progress.
- **T7 (fold the four tune modules):** BLOCKED pending team-lead answer on the
  import-purity resolution (see *T7 blocker* below). Question sent before T6
  began so the answer can arrive in parallel.

## T7 Step 1 finding — `_command.py` (asked for explicitly by the brief)

`src/phenotypic/gui/tune/_command.py` **exists**, 382 lines, and is **Dash-free**
(no `dash`, `dash_bootstrap_components`, `flask`, `werkzeug` import anywhere).
Its upward imports are exactly two, both resolvable by resolution-order option 1:

- `:12` `from phenotypic.gui.shell._sandbox import SandboxRoot` → `_services.sandbox`
- `:13-17` `from phenotypic.gui.tune._run_argv import tune_run_argv,
  tune_run_argv_from_tail, tune_run_tail` → `_services.argv` (Task 8's home,
  imported directly rather than through the 15-line shim)

The plan's four-module list is therefore correct and no module is dropped.
**No drift-register entry needed for `_command.py`.**

## T7 blocker — the GUI-import allowlist

`tests/unit/services/test_import_purity.py::test_service_module_does_not_import_gui`
(added by C1, not present in the plan's Task 1 text) forbids every `_services`
module from importing `phenotypic.gui.*` off-allowlist. Task 7's sources reach up
in four places that survive option 1:

| Reach | Call sites | Cost to promote |
|---|---|---|
| `_validation.py:7` → `gui.tune._domain_editor.grid_feasibility` | 1 (`preflight_issues`) | 11 pure lines; `_validation` is its only production consumer |
| `_setup_authoring.py:27` → `gui.shell._source_context.sandbox_fingerprint` | 2 (`:368`, `:392`) | 3 lines, `SandboxRoot`-only |
| `_setup_authoring.py:20` → `gui._config.tune_presets_dir` | 1 (`:522`) | 6 lines + 3 `SANDBOX_*` constants out of a 1025-line module |
| `_setup_authoring.py:21` → `gui.shell._metadata_context.resolve_metadata_csv` | 1 (`:470`) | 596-line module, transitively `_source_context` → `_classifier` |

Recommendation sent to the team lead: promote the first three (the third mirrors
Task 2's `IMAGE_EXTS` move exactly), allowlist only `_metadata_context`. Awaiting
the decision; T7's move will not land before it arrives.

---

## T6 — split `gui/tune/_space.py` (COMPLETE)

### What landed

| File | State |
|---|---|
| `src/phenotypic/_services/tune_spec.py` | **new**, 330 lines, Dash-free, GUI-free |
| `src/phenotypic/gui/tune/_space_view.py` | **new**, the Dash half; imports the pure half |
| `src/phenotypic/gui/tune/_space.py` | 625 → 82 lines, a shim over both halves |
| `tests/unit/services/test_space_split.py` | **new**, 6 tests |
| `tests/unit/services/test_lazy_gui_packages.py` | strict-xfail wrapper removed (see below) |

### Exported surface of `_services/tune_spec.py` after T6

Public: `apply_space_edits`, `space_to_spec` (both in `__all__`).
Private, imported by name elsewhere: `_apply_edits`, `_build_search_space`,
`_default_qc_scorer`, `_editable_knobs`, `_is_tuning_spec`, `_load_space_source`,
`_recover_typed_choices`, `_try_load_pipeline`, `_try_load_spec`. Plus the
`_RunRootLike` Protocol (see deviation D2).

### Deviations from the plan text

**D1 — the shim resolves the view half lazily (PEP 562), not eagerly.**
The plan's Task 6 Step 3 sketch does
`from phenotypic.gui.tune._space_view import _knob_form, build_space_view,
setup_knob_forms` at module level. That would keep `dash` on
`phenotypic.gui.tune._space`'s import path, so the strict xfail in
`test_lazy_gui_packages.py` would have kept xfailing and the split would have
been cosmetic for every consumer that only wants `space_to_spec` (e.g.
`_setup_authoring.py:28`). The shim now mirrors the `__getattr__` pattern
Task 2.5 put in `gui/tune/__init__.py`, forwarding a `_VIEW_NAMES` frozenset.
Verified by mutation M5: making the shim eager reintroduces the failure.

**D2 — `_load_space_source` types its argument as a Protocol, not `TuneRunRoot`.**
`_space.py:40` had `if TYPE_CHECKING: from phenotypic.gui.tune._run_root import
TuneRunRoot`. `ast.walk` in the purity gate visits nodes inside
`if TYPE_CHECKING:` too, so that import fails
`test_service_module_does_not_import_gui` even though it never executes.
Replaced with a structural `_RunRootLike` Protocol declaring the single
attribute the function uses (`path`). The view half keeps the real
`TuneRunRoot` annotation.

**D3 — the plan's third test was not written as specified** (the brief already
directed this). `assert "phenotypic.gui" not in inspect.getsource(tune_spec)`
is a source-text grep; the new module's docstring has to explain why it must
not import the GUI, so that assertion fails on correct code. Replaced with the
AST check from `test_argv_promotion.py`. **Mutation M3 proves this was not
pedantry:** `from phenotypic import gui` evades a substring check and is caught
by the AST one.

### Mutations run (all six restored afterwards; `git diff` verified clean)

| # | Mutation | Result |
|---|---|---|
| M1 | `import dash` in `_services/tune_spec.py` | **FAIL** ×2 — `test_pure_half_is_importable_without_dash`, `test_import_purity::test_service_module_imports_no_dash[...tune_spec]` (`dragged ['dash','flask','werkzeug']`) |
| M2 | `from phenotypic.gui.tune import _ids` in the pure half | **FAIL** ×2 — `test_pure_half_does_not_import_the_gui`, `test_import_purity::test_service_module_does_not_import_gui[...tune_spec]` |
| M3 | `from phenotypic import gui` in the pure half (aliased form) | **FAIL** — `test_pure_half_does_not_import_the_gui`. **`test_import_purity::test_service_module_does_not_import_gui` PASSED — a hole in the tier-wide gate, reported separately.** |
| M4 | shim redefines `space_to_spec` instead of re-exporting | **FAIL** — `test_shim_reexports_the_same_objects` |
| M5 | shim imports `_space_view` eagerly | **FAIL** — `test_lazy_gui_packages::…[phenotypic.gui.tune._space]` (`dragged ['dash','dash_bootstrap_components','flask','werkzeug']`) |
| M6 | drop `build_space_view` from the shim's `_VIEW_NAMES` | **FAIL** — `test_legacy_import_path_still_works` (ImportError) |
| M7 | view half stops importing the pure half, redeclares the helpers | **FIRST RUN: FALSE GREEN.** `assert "_services.tune_spec" in inspect.getsource(_space_view)` passed on the docstring mention alone — the plan's own grep idiom, reproduced in my test. Test rewritten to assert on parsed imports; M7 re-run **FAILS** correctly. |

M7 is the reason the mutation step is not optional: the test as first written
would have shipped green while proving nothing.

### Gate results at T6

- `tests/unit/services` + `tests/unit/gui/tune`: **180 passed**, 0 xfailed
  (services baseline was 40 passed / 1 xfailed; the xfail is now a plain pass
  and 6 new tests were added).
- `uv run --no-sync mypy src/phenotypic`: **421 errors in 125 files** — exactly
  the baseline, no new errors.
- `uv run --no-sync ruff check <the five changed paths>`: clean.

---

## T7 part 1 — fold `_command.py` and `_export.py` (COMPLETE)

Done ahead of the team lead's answer because **all three options on the table
move these two modules unchanged** — neither touches the contested upward
reaches. `_validation.py` and `_setup_authoring.py` are the only ones that do,
and they are still held.

| File | State |
|---|---|
| `src/phenotypic/_services/tune_spec.py` | 330 → 832 lines; two new sections with banners |
| `src/phenotypic/gui/tune/_command.py` | 382 → 31 lines, shim |
| `src/phenotypic/gui/tune/_export.py` | 109 → 28 lines, shim |
| `tests/unit/services/test_shim_equivalence.py` | +4 tests |

### Exported surface of `_services/tune_spec.py` after T7 part 1

`__all__` (13): `PreparedPipelineExport`, `ValidatedTuneCommand`,
`apply_space_edits`, `build_tune_command`, `export_best_from_run`,
`export_pareto_pipeline`, `export_winning_pipeline`, `prepare_best_from_run`,
`publish_prepared_export`, `render_launch_command`, `render_tokens`,
`space_to_spec`, `storage_url_preflight_issue`.

Also module-level and re-exported by a shim or imported by name elsewhere:
`DEFAULT_STORAGE_ENV`, `ExecutionTarget`, `StorageMode`, `_RunRootLike`,
`_apply_edits`, `_build_search_space`, `_default_qc_scorer`, `_editable_knobs`,
`_is_tuning_spec`, `_load_space_source`, `_params_from_best_params_payload`,
`_recover_typed_choices`, `_try_load_pipeline`, `_try_load_spec`, plus the
module-private `_ENV_NAME`, `_INLINE_PASSWORD_ISSUE`, `_PORTABLE_PREFIX`,
`_resolve_existing`, `_resolve_output`, `_storage_tokens`.

Shim surfaces were AST-derived from every `from <module> import ...` in `src/`
and `tests/`, per the X2/X3 standing instruction — not from the plan's list.
That is how `_params_from_best_params_payload` and `render_tokens` got into the
`_export` / `_command` shims.

### How the upward imports were resolved (resolution-order option 1 throughout)

- `_command.py:12` `gui.shell._sandbox.SandboxRoot` → `_services.sandbox`
- `_command.py:13-17` `gui.tune._run_argv.{tune_run_argv,
  tune_run_argv_from_tail, tune_run_tail}` → `_services.argv` **directly**, not
  through the 15-line shim
- `_export.py` had no upward imports at all

**No allowlist entry was added.** `GUI_IMPORT_ALLOWLIST` still holds exactly one
entry (`_services.runs -> gui.shell._classifier`).

### Deviation

**D4 — the merge is verbatim, including module-level import cost.** `_export.py`
imported `ImagePipeline`, `TuningSpec`, `build_pipeline` and five `sdk_` helpers
at module level; `tune_spec.py` previously deferred all of those into function
bodies. Keeping the deferrals would have meant rewriting the moved code, which
is the "a move that quietly takes a behaviour change with it" the review
protocol names. So the imports moved up to module level and the four now-dead
local imports (`resolve_tuning_spec_path`/`resolve_pipeline_config_path`,
`TuningSpec`, `ImagePipeline`, `TuningSpec` in `space_to_spec`) were dropped
rather than left shadowing identical objects. Verified: `phenotypic.tune._spec`,
`phenotypic.tune._evaluation` and both shims are **optuna-free**, so
`test_space_module_does_not_import_optuna` still holds, and
`phenotypic.tune.TuningSpec is phenotypic.tune._spec.TuningSpec` is `True`.

### Mutations run (all restored; `git status` verified)

| # | Mutation | Result |
|---|---|---|
| M8 | `_export` shim redefines `export_best_from_run` | **FAIL** ×2 — `test_export_is_one_function`, `test_export_shim_reexports_every_public_name` |
| M9 | drop `render_tokens` from the `_command` shim | **FAIL** — `test_command_shim_reexports_every_public_name` (AttributeError) |
| M10 | pure half imports `SandboxRoot` through `gui.shell._sandbox` instead of `_services.sandbox` | **FAIL** ×2 — tier gate + `test_pure_half_does_not_import_the_gui`. This is precisely the mistake the brief warned about, and both guards catch it. |
| M11 | pure half imports the argv builders through `gui.tune._run_argv` instead of `_services.argv` | **FAIL** — tier gate (`'phenotypic.gui.tune._run_argv'`) |

### Gates at T7 part 1

- `tests/unit/gui` + `tests/integration/gui` + `tests/unit/services`:
  **1780 passed, 3 skipped, 2 deselected** (= 1776 at T6 + the 4 new shim tests).
- `uv run --no-sync mypy src/phenotypic`: **421 errors in 125 files** — baseline.
- `uv run --no-sync ruff check <the four changed paths>`: clean.

## Still open

`_validation.py` and `_setup_authoring.py` are **not** folded. Both depend on
the team-lead decision recorded under *T7 blocker* above. Nothing else in T6/T7
remains.
