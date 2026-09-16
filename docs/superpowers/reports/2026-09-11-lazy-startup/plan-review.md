# Plan review — `2026-09-11-lazy-startup`

**Reviewer:** `xander-local:plan-reviewer` (opus), read-only. Saved by the controller: the reviewer has no write tool.
**Date:** 2026-09-11 · **Plan reviewed at:** `ecbeff049`

**Verdict: Not ready.** The design is sound and most of it verifies clean, but three of the four tier guards cannot reach GREEN as written (Tasks 3, 5 and 6 would each stall at their own "Expected: all pass" step), one mutation proof does not fail, two existing tests break, and the ruff gate regresses by 34 findings. All six fixes are concrete and small; with them applied the plan is ready.

Root cause of the three tier failures: the evidence probe `probes/post_change_closure.py` is unsound in two ways (C5). Every tier claim derived from it needed re-deriving from a runtime trace.

**Method.** Tasks 1–6 were emulated mechanically onto a copy of `src/` in `/tmp/lazyrev/full/src`, applying the plan's own quoted blocks by plan line number and asserting each "Remove Lx" anchor before editing (every assert passed). The sweep, the tier probes, the plan's new-test logic and the affected existing tests then ran against that copy with `PYTHONPATH`. The repo working tree was never written to.

## Critical

### C1. Tier 4 fails: `from phenotypic import Image` still loads plotly and `matplotlib.pyplot`
Plan: Global Constraints → "Unchanged modules"; spec Amendment A P3; Task 3 Step 6.

With all six tasks applied, `from phenotypic import Image` loads `matplotlib`, `matplotlib.pyplot`, `pandas`, `plotly`, `polars`, `pyarrow`, `scipy`, `skimage`. Chain for `matplotlib.pyplot`:

```
phenotypic/_core/_image_parts/plot_accessor/_diagnostics_plotter.py:7(<module>)
 <- phenotypic/plotting/_image_plots.py:17(_diagnostics_figure)
 <- phenotypic/plotting/_image_plots.py:52(PlotDiagnostics)
 <- phenotypic/plotting/_image_plots.py:35(<module>)
 <- phenotypic/plotting/__init__.py:3(<module>)
 <- phenotypic/_core/_pipeline_parts/_image_pipeline_core.py:41(<module>)
 <- phenotypic/_core/_pipeline_parts/__init__.py:1(<module>)
 <- phenotypic/_core/__init__.py:1(<module>)
 <- phenotypic/__init__.py:55(__getattr__)
```

`plotly` and `polars` arrive on the same chain (`_diagnostics_plotter.py:17` and `:16`). The trigger is a decorator evaluated at class-creation time: `@_diagnostics_figure("fig_intensity_histogram")` (`_image_plots.py:52`) calls a function whose body imports `DiagnosticsPlotter`. A static import walker cannot see it, which is why P3 concluded the module is off every guarded path. It is on the `Image`, CLI and hub paths.

**Fix.** Remove `_diagnostics_plotter.py` from "Unchanged modules" and add a Task 3 row. Every deferred name in it is used only inside functions or in annotations (AST over runtime `Load` positions, covering nested functions, lambdas, class bodies, decorators, defaults and `AnnAssign`), so the row is a plain point-of-use move.

### C2. Tier 2 fails: `python -m phenotypic --help` still imports the whole core
Plan: Task 5 Step 4.2 and Step 6.

```
CHAIN phenotypic._core:
 phenotypic/__init__.py:55(__getattr__)
 <- phenotypic/_cli/_cli_validation.py:17(<module>)
 <- phenotypic/_cli/_cli_interactive.py:17(<module>)
 <- phenotypic/phenotypicCLI.py:171(<module>)
 <- phenotypic/__main__.py:11(<module>)
```

`_cli_interactive` is in the plan's "16 light imports stay at module level" set, but it imports `_cli_validation`, which does `from phenotypic import ImagePipeline`. Seven `_cli/*.py` modules do module-level `from phenotypic import …`.

**Fix (verified).** Add two `_CLI_RUNTIME_IMPORTS` entries (`_cli_interactive`: `execute_dry_run`, `get_sample_datasets`; `_cli_validation`: `validate_execution_config`, `validate_pipeline`) and delete their module-level statements. With exactly that change: `exit 0 ; help ok True ; LOADED: []`.

### C3. Tier 3's forbidden set is unreachable, and its exception procedure forbids the only available exceptions
Plan: Task 6 Steps 2 and 7.

Composing the hub with no request loads `matplotlib`, `matplotlib.pyplot`, `pandas`, `plotly`, `polars`, `pyarrow`, `scipy`, `skimage` — every one through a phenotypic-side module-level import (`_gui/analysis/_callbacks.py:25-26`, `_gui/_operation_registry.py:18`, `_cli/_metadata_join.py:7`, `_gui/_schema_cache.py:23`, `_gui/run_console/_request_safety.py:15`). The plan's Step 7 rule ("only a purely third-party chain may be allowed") would force deferring pandas/polars/scipy/skimage across `_gui/analysis`, the registry, the results viewer and the run console — contradicting the spec's Non-goals and D5.

**Fix.** Restore the spec's wording ("unless the post-change measurement shows a shell-side module-level chain needs one") and pre-populate the allowed map from the measurement. Keep the spec's minimum (colour, numba, h5py, mahotas, cv2, bm3d, and pyplot once C1 lands) as the asserted set; none of those six is loaded before the first request.

### C4. The builder mount change breaks two existing integration tests
`tests/integration/gui/test_smoke_shell.py::test_dispatcher_threads_script_root` and `::test_explicit_url_prefix_preserves_script_root_through_dispatcher` do `dispatcher.mounts["/builder"].register_blueprint(bp)` (lines 222 and 403) → `AttributeError: '_SessionProxy' object has no attribute 'register_blueprint'`. Result: `2 failed, 197 passed, 4 skipped`.

**Fix.** Name both in Task 6 Step 9 and resolve through the session: `dispatcher.mounts["/builder"]._session.get().server`.

### C5. The plan's evidence probe is unsound; two of its conclusions are wrong
`probes/post_change_closure.py`: (1) `planned_drop()` drops every edge whose importer is `phenotypic` itself and never follows `from phenotypic import Image/GridImage/ImagePipeline` from other modules, hiding the seven `_cli` modules and the registry (C2, half of C3); (2) it drops the matplotlib/plotly edges of `_diagnostics_plotter.py`, modelling a deferral the plan does not perform (C1). It also cannot see imports executed at class-definition time.

**Fix.** Re-derive light/heavy splits with a runtime trace, not the static closure.

## Important

- **I1. Mutation M2 does not fail.** Once `abc_/__init__` stops importing the core, the grid-handler imports are import-order-safe on their own: with both restored to module level the sweep is `total 85 ok 85`. The move is a weight optimisation, not a cycle fix. M2's guard becomes the deferral checker only; spec B1 should label the `_grid_image_handler` rows as weight moves and the `abc_` lazy re-export as the single cycle fix. M3 is a genuine sweep mutation and does fail.
- **I2. Task 2 Step 4's expected failure list is short by one.** The Step-3-only state fails nine modules; `phenotypic.analysis._helper` is missing from the plan's list of eight.
- **I3. The lazy `__init__`s regress ruff by 34 findings** (33 × E402 + 1 × F401): `sdk_/__init__.py` 0 → 13, `abc_/__init__.py` 0 → 20, `phenotypicCLI.py` 0 → 1. Cause: P8 places the lazy block before the eager imports. Fix by `# noqa: E402` on the eager blocks (smallest, keeps the gate meaningful).
- **I4. The preload runs before the CLI's own validation.** As written it is the first statement of `phenotypic_cli`, so `migrate`, `recompile`, `--dry-run` and even a usage error pay the full import. Either accept and say so in the plan, or move it after validation and change the abort test's invocation.

## Minor

- **M-a.** Three edited modules lack `from __future__ import annotations`; the rule's justification should say "every module with a `TYPE_CHECKING` entry has it".
- **M-b.** `phenotypic.plotting` stops being an attribute of the package; it is public in the docs. Add `"plotting"` to `_LAZY_SUBPACKAGES` (not to `__all__`).
- **M-c.** The sweep's `refs` exclusion never matches under `src/`.
- **M-d.** `test_detect_mode_choices_match_the_detection_mode_registry` reads a global registry in-process; latent flake if any test ever registers a custom mode.
- **M-e.** Sphinx `autodoc_typehints = "both"` with `TYPE_CHECKING`-only names may change resolved annotations; flag for the Task 8 docs gate rather than assuming 0 new warnings.
- **M-f.** `get_registry()` is an unlocked singleton; after the change a first `/builder/` and a first `/analysis/` request can call `discover()` concurrently (benign duplicate work).

## Checks out (with evidence)

1. **Line numbers and quoted text.** Every anchor the emulation touched was asserted before editing; all passed. Tasks 3 and 4: **all 33 rows** checked, not a sample.
2. **Function names and leaked uses.** Every function named in an "Import inside" column exists and is unambiguous. An AST pass over every runtime `Load` found no runtime use outside the listed functions, except `_pio` at `_accessor_dash_handler.py:30`, which Task 3 removes anyway. No deferred name appears in a pydantic field annotation.
3. **Cycles.** `total 85 ok 85` at HEAD, at T2 and after all six tasks. Targets = 75 packages + 10 entry modules, over the ≥70 floor.
4. **`sdk_`/`abc_` lazy maps.** The `sdk_` map is exactly complete (31 names bound, 31 mapped, empty set difference both ways). No `import *`, no `inspect.getmembers`/`dir()` consumer. Sphinx autosummary uses `__all__` + `safe_getattr`, which `__getattr__` serves. The tune annotation gate and `tests/unit/test_fixtures.py` are unaffected. `abc_`'s map points `DetectionMode`/`register_detection_mode` at the package whose `__init__` registers the built-ins.
5. **CLI loader / `mock.patch`.** The plan's assertion holds, proved against a throwaway module across seven patch flows and then verbatim on the emulation (`patch test OK`). `all(name in globals)` cannot skip incorrectly. The four-helper list is complete (AST over all 39 top-level functions). `phenotypicCLI.py` has no `from __future__ import annotations`; eager-annotation hazards were checked explicitly and are clear.
6. **`--help` probes.** `runpy.run_module` raises `SystemExit(0)`; `redirect_stdout` captures click's output (12,040 chars, `Usage: phenotypic [OPTIONS]`, contains `--detect-mode`); the launcher's `main(['--help'])` exits 0.
7. **GUI.** `GET /builder/` → 200 and `/builder/_dash-layout` → 200 through `_SessionProxy`; `phenotypic.detect` absent before the request, present after. Nothing references `_core_import_elapsed`; `"Core library loaded"` appears only in `test_startup.py`, which stays green.
8. **Tier feasibility.** Tier 1 `LOADED: []`; tier 2 GUI `LOADED: []`; tier 2 CLI passes after C2's fix. `import dash` pulls `plotly` and `IPython`; click, yaml, pydantic, flask and werkzeug pull nothing heavy.
9. **Tests.** `tests._startup_probe` will import (no `tests/__init__.py`, but the pattern is already used). No basename collisions. Shard coverage needs no manifest edit. The deferral checker run verbatim gives **66 passed** for all 33 rows.
10. **Global risks cleared.** Pickling/joblib, `_find_class_in_phenotypic`, `OperationRegistry.discover()`, `test_lazy_import_lock`, Windows packaging markers, `PHENOTYPIC_DOCS_BUILD`, detection-mode registration.

## Not completed

- The broad operation-test run against the emulation was killed at the reviewer's 590 s limit. The 33 moves are verified statically and by a 197-test GUI/CLI/plotly/tune subset, not by the operation suites. Run them early in Tasks 3–4.
- mypy, `sphinx-build -n`, the e2e builder suite and the full default lanes were not run (M-e is the live risk).
- Mutations M1, M4 and M6 were reasoned through, not executed; M2 and M3 were executed.
- Timing claims were not re-measured.
