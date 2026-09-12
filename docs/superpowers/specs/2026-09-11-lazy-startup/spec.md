# Lazy startup: light entry points for the CLI, GUI, and `import phenotypic`

**Date:** 2026-09-11 · **Branch:** `refactor/private-gui` (follow-up inside PR #218, at the user's request) · **Base:** `d437ce765`

## Objective

Every entry point pays for the entire library before it does anything:

- `import phenotypic` imports 14 subpackages and ~3,300 modules.
- `python -m phenotypic --help` and `phenotypic-gui --help` pay that same cost just to print help.
- The GUI builds sub-apps no one has opened yet.

This change makes the entry points light:

1. The package `__init__` files defer what they re-export.
2. Heavy third-party libraries with few importers are imported where they are used.
3. The CLI help path and the GUI launcher import nothing heavy.
4. The GUI builder is built on first visit.

A deterministic guard keeps it that way.

## Decisions (user-confirmed, 2026-09-10/11)

| ID | Decision |
|----|----------|
| D1 | **Scope: the import path first.** Lazy imports are the main track. GUI on-demand page loading is a small add-on, because the measured ceiling is ~0.2 s: building the eager sub-apps (builder, run console, browse, shell) costs ~0.19 s of the ~2.2 s time-to-servable. |
| D2 | **Guard: the module set, not wall-clock time.** Subprocess tests assert which heavy modules are *absent* from `sys.modules` after each entry point. Wall-clock numbers are measured and reported in the PR, never asserted. |
| D3 | **Approach A: targeted deferral.** PEP 562 lazy package `__init__`s; imports moved to the point of use for the heavy libraries that have few importers; a CLI help path with no core imports; the builder behind the existing `ToolSession` proxy. Rejected: generic `importlib.util.LazyLoader` proxies, which cannot defer `from x import y`, fire on `isinstance`/pickling/pydantic introspection, and weaken typing. Also rejected: an entry-points-only change, which leaves real runs and `from phenotypic import Image` at ~1.4 s. |
| D4 | **Fail fast on broken installs.** Every pipeline-running entry point calls `load_runtime_dependencies()` after parsing its options, so a broken required library fails at run start rather than being recorded as a per-image scientific failure. |
| D5 | **The run console stays eager.** Its `create_app` starts the SLURM observer the shell's run registry depends on (`_gui/run_console/_app.py:121-124`). |
| D6 | **Kernels and algorithms are untouched.** Numba kernels (`@numba.njit` at module scope in two ported-reference modules) are deferred at their *importers*, never inside the kernel modules. |

## Measured baseline (macOS, this checkout, `uv run`)

| Path | Time |
|---|---|
| bare interpreter (`python -c pass`) | 0.03 s |
| `import phenotypic` | 1.44–1.97 s (three runs) |
| `python -m phenotypic --help` | 1.65–1.69 s |
| `phenotypic-gui --help` | 1.84–2.00 s |
| GUI to servable (in-process `create_app`, no request) | 2.24 s = package import 1.79 + `shell._app` module 0.26 + `create_app` 0.19 (builder 0.13; run console, browse and shell ~0.02 each) |
| `OperationRegistry.discover()` on a warm package | 0.01 s (the builder's first-visit cost is importing the operations, not discovery) |

These are the heaviest chains under `import phenotypic`, from `-X importtime`, cumulative at first import:

| Library | Cost | First pulled in by |
|---|---|---|
| colour-science (+ scipy.interpolate, pandas, pyarrow, tqdm) | 0.39 s | `sdk_.colourspace` (module-scope `sRGB_D50` constant, `sdk_/colourspace.py:8-22`) via the `sdk_/__init__` re-export |
| skimage + scipy (FootprintMixin) | 0.21 s | `sdk_.mixin` |
| numba (+ llvmlite, coverage) | 0.10 s | `sdk_.reconnect._tensor_voting` via `detect._filamentous_fungi_detector` |
| h5py | 0.06 s | `sdk_.hdf_` via the `sdk_/__init__` re-export |
| mahotas | 0.04 s | `measure._measure_texture` |
| cv2 | 0.03 s | `enhance._subtract_opening` |
| plotly.express | 0.03 s | the image accessor `_accessor_dash_handler` (module-level `try`) |
| matplotlib.pyplot | 0.04 s | the image accessor `_accessor_mpl_handler` |
| bm3d / bm4d / pywt | ≥ 0.02 s (pywt alone) | `correction._color_denoise` |
| polars | 0.04 s | `util._measurement_outputs` |

The GUI launcher adds dash (0.18 s), whose `dash._jupyter` imports IPython (0.08 s) and prompt_toolkit.

## Design

### 1. Lazy package `__init__`s (PEP 562)

- **`phenotypic/__init__.py`** keeps `__version__`/`__author__`/`__email__` and the `_startup_perf` import as the first statement (the lazy `colour.plotting` stub and `IMPORT_STARTED_AT` must still install first).
  - `Image`, `GridImage`, `ImagePipeline` and the 14 subpackages in `__all__` resolve through a module `__getattr__`, which caches the value with `setattr`.
  - `__dir__` returns `__all__` plus the eager names.
  - Unknown names raise `AttributeError`. This keeps `SerializablePipeline._find_class_in_phenotypic` (`_core/_pipeline_parts/_serializable_pipeline.py:628-678`) correct: it tries `hasattr(phenotypic, name)` and then falls through to `importlib.import_module` on each subpackage.
  - A `TYPE_CHECKING` block imports the real names, for mypy and IDEs.
  - The comment "Import abc_ first … breaking the circular import chain" is removed with the eager order (see B1).
- **`sdk_/__init__.py`** re-exports lazily. Importing any `phenotypic.sdk_.*` leaf (for example `_gui/_config.py:57`) runs this `__init__`, which today imports `colourspace` (0.39 s), `mixin` (0.21 s, skimage/scipy), `hdf_.HDF` (0.06 s, h5py), `slurm` (0.03 s), `_measurement_tables` (pandas) and the rest. Every public name keeps working as `from phenotypic.sdk_ import X` and `phenotypic.sdk_.X`.
- **`abc_/__init__.py`**: `PrefabPipeline` (L35) and `DetectionMode`/`register_detection_mode` (L37) become lazy re-exports. Both lines pull in the core (`_core._image_pipeline`; `_core/__init__` → the whole `_image_parts` handler chain). This is also the cycle fix in B1.
- **The operation subpackages' `__init__`s** (`detect`, `enhance`, `measure`, `refine`, `correction`, `grid`, `analysis`, `post`, `prefab`) **stay eager**. `OperationRegistry.discover()` imports each one explicitly and enumerates it with `inspect.getmembers` (`_gui/_operation_registry.py:188-205, 249, 295`), which sees only attributes that already exist.

### 2. Point-of-use deferral for heavy libraries with few importers

The target set is exported as `phenotypic._startup_perf.DEFERRED_RUNTIME_MODULES = ("colour", "numba", "h5py", "mahotas", "cv2", "bm3d", "plotly", "matplotlib.pyplot")`.

An AST inventory found 42 modules that import one of these libraries at module level at runtime (excluding `TYPE_CHECKING` and `_gui/`). In 35 of them, the name is used only inside functions, so the import moves into those functions. The inventory is in Appendix A.

The other 7 use the name at module, class or decorator scope, and each gets a specific treatment:

| Module | Module-scope use | Treatment |
|---|---|---|
| `sdk_/colourspace.py:8-22` | builds the `sRGB_D50` constant from `colour` | Its three importers (`_core/_image_parts/color_space_accessors/_xyz_conversion.py`, `correction/_color_denoise.py`, `correction/_denoise_block_match.py`) import `sdk_.colourspace` inside the function that uses it; the `sdk_/__init__` re-export becomes lazy. The constant and its numbers are unchanged. |
| `correction/_color_correction/_color_checker_profile.py:124-125`, `_helpers.py:18` | module-scope `colour` use | Reached only through `phenotypic.correction` (eager subpackage). Accepted as eager: the correction subpackage is not on any guarded entry path. If the tier-4 guard shows otherwise, the use moves into a `functools.cache`d accessor. |
| `sdk_/reconnect/_tensor_voting.py:62`, `sdk_/branch_pathfinding/_dijkstra_kernels.py:86,122,162` | `@numba.njit` decorators | **Kernel modules are untouched (D6).** Their importers — `detect/_filamentous_fungi_detector.py`, `detect/_two_k_filamentous_detector.py`, `detect/_filamentous_fungi/__init__.py`, `sdk_/reconnect/_colony_labeling.py`, `_colony_reconnect.py` — import them inside `_operate`/the calling function. |
| `_core/.../_accessor_dash_handler.py:14-32` | plotly `try` import and the `PHENOTYPIC_DOCS_BUILD` renderer switch | `PLOTLY_AVAILABLE = importlib.util.find_spec("plotly") is not None`. `px`/`go` and the docs-build renderer switch move into one cached `_plotly()` helper called on first plot. |
| `_cli/_cli_process_single.py:21-23` | `matplotlib.use("Agg")` | Unchanged: a worker entry module, not a startup path, and it must set the backend before anything plots. |

`_core/_image_parts/_grid_image_handler.py:16-17` imports `CenteredAutoGridFinder` and `MeasureBounds` from the `grid` and `measure` packages. They are used only inside methods (`:96`, `:416`), so the imports move there (this is also the B1 cycle fix, and it removes mahotas from the `Image` path).

scipy, skimage and pandas stay eager: they have 49, 94 and 81 module-level importers, and the core `Image` handler chain needs them.

### 3. CLI help path

- **Module level:** `phenotypicCLI.py` keeps only the standard library, `click` and `yaml` at module level. Its phenotypic imports (lines 157-190: `ImagePipeline`, `available_modes`, the `_cli_*` modules, `_core._provenance`) move into the command body (`phenotypic_cli`, line 1567) or into the helpers that use them.
- **`--detect-mode` choices** (`click.Choice(list(available_modes()))`, line 1390) come from `sorted(typing.get_args(phenotypic.sdk_.typing_.DetectMode))`. `sdk_/typing_.py:50` holds the same 11 names that `available_modes()` returns, which is verified. A drift test pins `set(get_args(DetectMode)) == set(available_modes())`. The help text is byte-identical, because `available_modes()` already returns them sorted.
- **Custom detection modes:** those registered through `PHENOTYPIC_PRELOAD_MODULES` are unaffected. Today they register after click has built its choices too.
- **Preload:** after the options are parsed and validated, and before any image work, the command body calls `load_runtime_dependencies()` (D4).

### 4. GUI launcher and hub

- **Launcher:** `_gui/shell/_launcher.py` imports `create_app` inside `launch_gui` rather than at module level (`:39`), so `phenotypic-gui --help` imports only argparse, `_gui._config` and the launcher helpers.
- **Builder:** in `compose_hub` (`_gui/shell/_app.py:540-550`) the builder becomes a `ToolSession` whose build closure runs `builder.create_app(...)` + `wrap_in_chrome(...)`. It is mounted through `_SessionProxy` at `MOUNT_BUILDER`, the same mechanism as the viewer and analysis app (`_app.py:103-133, 345, 382`).
  - The builder session is **not** registered with the idle-release thread: it is built once on first visit and kept for the life of the process, so no builder server-side state is ever lost to a release.
  - `get_registry()` (`_operation_registry.py:814-824`) is already a lazy singleton, so the analysis app and the QC tab are unaffected.
- **Run console:** stays eager (D5).
- **Browse:** stays eager unless the tier-3 measurement shows it pulls in a module outside the allowed set. In that case it is deferred the same way.
- **Startup reporter:** the launcher no longer records "Core library loaded" (it would report ~0 s), and `_STARTUP_STEPS` goes from 3 to 2 (`_launcher.py:49, 110-120`). The imports are reported through the existing `progress` detail inside "Composing GUI hub". The launcher's `_core_import_elapsed` helper (`_launcher.py:52-68`, its only consumer of `IMPORT_STARTED_AT`) is deleted and the launcher stops passing `import_elapsed`. `StartupReporter`'s API (including its `import_elapsed` parameter, pinned by `tests/unit/gui/shell/test_startup.py`) and `_startup_perf.IMPORT_STARTED_AT` are unchanged.

## Behaviour and error handling

### B1. Import-order independence (new hard requirement)

- **What the probe showed.** With a lazy top-level `__init__`, importing each of 87 packages and entry modules *first* in a fresh interpreter (throwaway PEP 562 stub, no repo edits) gave **75 ok and 12 failures**. Every failure comes from one of three sites that today's eager order hides:

  | Site | Failing first imports | Fix |
  |---|---|---|
  | `_grid_image_handler.py:16` `from phenotypic.grid import CenteredAutoGridFinder` | `phenotypic.grid`, `phenotypic.grid._auto_grid_finder` | move into the method (`:96`) |
  | `_grid_image_handler.py:17` `from phenotypic.measure import MeasureBounds` | `phenotypic.measure`, `phenotypic.measure._measure_size` | move into the method (`:416`) |
  | `_image_pipeline_core.py:34` `from phenotypic.analysis.abc_._model_fitter import ModelFitter` | `phenotypic.analysis` and 7 of its modules, `phenotypic.sdk_._qc_recipe` | break the loop `_model_fitter` → `phenotypic.abc_.plotting` → `abc_/__init__` (L35/L37) → prefab/core → `_image_pipeline_core` → `_model_fitter` by making those two `abc_` re-exports lazy (Design §1). `ImagePipelineCore` is a pydantic model, and its field `model: Optional[ModelFitter] = None` (`:209`) is unchanged. |

- **Other entry forms.** The 20 modules doing module-level `from phenotypic import ImagePipeline` (and 4 each for `Image`/`GridImage`) passed the probe. They stay, guarded by the sweep.
- **Guard:** the import-order sweep in §Tests.

### B2. Where latency moves

- **Bare `import phenotypic`** drops from ~1.4 s to near zero. The cost is paid at the first access to `phenotypic.Image` or to a subpackage.
- **`from phenotypic import Image`** no longer loads colour, h5py, mahotas, plotly or pyplot, which saves about half a second (colour alone is 0.39 s). The cost moves:
  - to the first colour-space conversion (colour);
  - to the first filamentous detector (numba);
  - to the first texture measurement (mahotas);
  - to the first denoise (bm3d) or cv2-backed operation;
  - to the first plot (pyplot/plotly).
- **GUI builder:** the first request to `/builder/` pays for importing every operation (~1 s), and blocks until it is built, like the viewer's first request today.
- **CLI runs and workers** pay the preload at start (D4). Total run time is unchanged apart from libraries a run never needs.

### B3. Broken installs

- **Scope.** Every deferred library is a *required* dependency (`pyproject.toml` `[project].dependencies`), so this concerns broken environments only: a numba/llvmlite mismatch, or a binary that crashes on one node.
- **Pipeline runs.** `load_runtime_dependencies()` imports `DEFERRED_RUNTIME_MODULES` and lets any exception propagate. It is called by:
  - the CLI command body (Design §3);
  - the four `__main__` entry modules that load or run a pipeline: `_cli/_cli_process_single.py`, `_cli_staged_slurm_worker.py`, `_cli_recompile_worker.py`, `_cli_checkpoint_handler.py`. These were found by grepping entry modules for `from_json`/`apply`/`measure`/finalize use. Controller, sentinel, lifecycle, chunk-writer, fan-out and migrate workers have none, and stay light.
- **Interactive use.** In the GUI and notebooks, a broken library surfaces as the error at the page or operation that first uses it.

### B4. Compatibility that must hold

- **Access patterns:**
  - `import phenotypic as pht; pht.detect.OtsuDetector()`;
  - `from phenotypic import Image, ImagePipeline`;
  - `dir(phenotypic)` includes `__all__`;
  - static typing works through the `TYPE_CHECKING` block.
- **Deserialization:** `from_json` class resolution (Design §1); `PHENOTYPIC_PRELOAD_MODULES` custom operations (unchanged).
- **GUI registry:** `OperationRegistry` discovery (the eager subpackage `__init__`s).
- **Pickling:** joblib/loky and SLURM work, because classes keep their defining leaf `__module__`, so unpickling imports that leaf.
- **Unchanged pieces:**
  - numba `cache=True` kernels (the modules are unchanged);
  - `matplotlib.use("Agg")` in the per-image worker;
  - the `PHENOTYPIC_DOCS_BUILD` plotly renderer, which runs at the first plotly use.
- **Thread safety:**
  - Python's per-module import locks serialize concurrent first imports.
  - PEP 562 lookups are idempotent and cached.
  - The builder builds under `ToolSession`'s lock, like the viewer.
  - Build closures must not take any other application lock while importing.

## Tests

All startup guards run the entry point in a **subprocess** (`sys.executable`), because other tests in the same xdist worker would have already populated `sys.modules`. Each guard asserts **absence** of the forbidden set *and* a **positive control** that proves the entry point actually ran. Each subprocess has a 60 s timeout, so a hang fails rather than stalls. The guards live in existing sharded directories, so `.github/pytest-shards.json` is unchanged.

| Tier | Test file | Entry point | Must be absent | Positive control |
|---|---|---|---|---|
| 1 | `tests/unit/ci/test_startup_imports.py` | `import phenotypic` | `HEAVY_STARTUP_MODULES` = scipy, skimage, pandas, pyarrow, polars, colour, numba, h5py, matplotlib, plotly, mahotas, cv2, bm3d, dash | `phenotypic._startup_perf` loaded; `phenotypic.__version__` readable |
| 2 | same | `python -m phenotypic --help`; the `phenotypic-gui` console script `--help` (resolved as in `tests/integration/gui/test_console_script.py`; missing script fails) | `HEAVY_STARTUP_MODULES` | exit 0; the usage line names the program; `click` / `argparse` loaded; `--detect-mode` choices equal `sorted(available_modes())` |
| 3 | `tests/unit/gui/shell/test_hub_startup_imports.py` | `create_app(sandbox, start_idle_thread=False, start_slurm_observer=False)` with no request | at least colour, numba, h5py, mahotas, cv2, bm3d, matplotlib.pyplot. scipy, skimage, pandas, plotly and polars are forbidden too, **unless** the post-change measurement shows a shell-side module-level chain needs one; each allowed exception is listed in the test with its import chain and a one-line justification | `dash` loaded; then `app.server.test_client().get("/builder/")` returns 200 and `phenotypic.detect` is in `sys.modules` afterwards (the builder was built on first visit) |
| 4 | `tests/unit/ci/test_startup_imports.py` | `from phenotypic import Image` | `DEFERRED_RUNTIME_MODULES` | `phenotypic._core._image` loaded; scipy/skimage/pandas allowed |

**Import-order sweep** (`tests/unit/ci/test_startup_imports.py`):

- **Targets:** every package under `src/phenotypic` with an `__init__.py`, discovered from the filesystem (excluding vendored `refs/` trees), plus these entry modules: `_core._image`, `_core._grid_image`, `_core._image_pipeline`, `phenotypicCLI`, `_gui.shell._launcher`, `_gui._operation_registry`, `_cli._cli_process_single`, `_cli._cli_staged_slurm_worker`, `_cli._cli_recompile_worker`, `_cli._cli_checkpoint_handler`.
- **Execution:** each target is imported first in its own subprocess, one parametrized case per target, so xdist spreads the load. A floor assertion requires at least 70 discovered packages.
- **Lane:** it runs in the PR lane (not `slow`). The measured cost before this change is ~95 s serial, ~24 s on 4 workers; lazy imports reduce it.

**Other tests:**

- **Preload:** `load_runtime_dependencies()` imports every name in `DEFERRED_RUNTIME_MODULES`. `DEFERRED_RUNTIME_MODULES` ⊆ `HEAVY_STARTUP_MODULES` ∪ {`matplotlib.pyplot`}. With the function patched to raise, the CLI exits non-zero before any image is processed.
- **Drift:** `set(get_args(DetectMode)) == set(available_modes())`.
- **Builder session:** an integration test builds the hub and checks the builder mount serves through the proxy. The existing composed-hub tests are the regression net: `tests/integration/gui/test_smoke_shell.py`, `test_no_id_collisions.py`, `test_lifecycle.py`, `test_viewer_session.py`, `test_viewer_handoff.py`, `test_scheduler_startup_wiring.py`, `test_results_async_binding.py`, `test_results_snapshot_refresh.py`, `test_tune_mount.py`, `tests/gui/browse/test_hub_mount.py`, `tests/unit/gui/test_apps_build_after_simplification.py`, `tests/unit/gui/shell/test_tune_is_unmounted.py`.

**Mutation proofs** (run once, recorded in the plan's task reports before trusting a pass):

- **M1:** restoring `from .colourspace import …`/`import colour` eagerly in `sdk_/__init__` fails tiers 1, 2 and 4.
- **M2:** moving `CenteredAutoGridFinder` back to module level fails the sweep for `phenotypic.grid`.
- **M3:** restoring the eager `PrefabPipeline` re-export fails the sweep for `phenotypic.analysis`.
- **M4:** making the builder eager again fails tier 3.
- **M5:** removing the `load_runtime_dependencies()` call from the CLI body fails the preload test.
- **M6:** re-importing `create_app` at launcher module level fails tier 2.

## Measurements (reported, not asserted)

- **Script:** `docs/superpowers/plans/2026-09-11-lazy-startup/measure_startup.py`. It drives the shipped code, so it sits beside the plan, not in `logic_validation_scripts/`.
- **What it times,** best of 5, each in a fresh process: bare interpreter; `import phenotypic`; `from phenotypic import Image`; `python -m phenotypic --help`; `phenotypic-gui --help`; hub to servable; first `/builder/` request.
- **Runs:** at the pre-change commit (a throwaway worktree) and at the branch head, on the same machine.
- **Output:** the table goes to `docs/superpowers/reports/2026-09-11-lazy-startup/startup-measurements.md` and the PR description.
- **Probes:** the throwaway probes that produced this spec's evidence are committed beside the plan under `probes/`: the import-time summarizer, the static import-closure walker, the PEP 562 stub import sweep and the AST deferral inventory.

## Regression

- **Per task:** the directly touched test files.
- **End of implementation, once:**
  - the full default lanes (`tests/unit tests/integration tests/gui tests/smoke`, run-phenotypic-test settings);
  - `PLAYWRIGHT=1` e2e: the builder suite (`tests/e2e/gui/builder/`, 10 files) plus the six ci_flaky helper-caller modules;
  - `sphinx-build -n` warnings vs the pre-change commit via `docs/superpowers/logic_validation_scripts/2026-09-10-private-gui-module/compare_findings.py docs` (autosummary imports the lazy modules);
  - mypy (fresh `--cache-dir`) and `ruff check src/phenotypic` finding sets not worse than the pre-change commit (418 and 25 at `BASE_PRE`; compared as sets with `compare_findings.py`).
- **Known pre-existing local failures** (not caused by this change): `tests/unit/test_ngff_schema_fixtures.py::test_schema_matches_recorded_digest[*]` fails on autocrlf checkouts, because the committed LF blobs match `SOURCE.md` but the CRLF working copies don't.

## Documentation in the same change

- **Root `CLAUDE.md` Gotchas:** the package entry points are lazy; imports must work in any order; do not add module-level imports of `DEFERRED_RUNTIME_MODULES` or of the core on entry paths; cite the guard tests.
- **`src/phenotypic/sdk_/CLAUDE.md`:** the `sdk_/__init__` re-exports are lazy; a new heavy re-export goes through `__getattr__`.
- **`src/phenotypic/_gui/FEATURES.md`:** the "Staged startup feedback" row (two stages; builder built on first visit).
- **`src/phenotypic/_gui/CLAUDE.md`:** the builder is a lazily built `ToolSession` mount.

## Acceptance criteria

1. **Tier 1:** in a fresh interpreter, `import phenotypic` loads none of `HEAVY_STARTUP_MODULES`. `phenotypic.Image`, `phenotypic.ImagePipeline` and `phenotypic.detect.OtsuDetector` resolve, and `set(phenotypic.__all__) <= set(dir(phenotypic))`.
2. **Tier 2:** `python -m phenotypic --help` and `phenotypic-gui --help` exit 0 and load none of `HEAVY_STARTUP_MODULES`. The `--detect-mode` choices are unchanged.
3. **Tier 3:** a composed hub with no request loads none of its forbidden set, and every allowed exception is justified in the test. The first `/builder/` request returns 200 and builds the builder. The run console and SLURM observer still start at composition, and the viewer and analysis mounts behave as before.
4. **Tier 4:** `from phenotypic import Image` loads none of `DEFERRED_RUNTIME_MODULES`.
5. **Sweep:** every package (at least 70) and every named entry module imports successfully when imported first in a fresh interpreter.
6. **Preload:** the CLI command body and the four pipeline-running entry modules call `load_runtime_dependencies()` before any image work, and a failing import aborts the run before any image is processed.
7. **Mutations:** M1–M6 each make the named guard fail.
8. **No regression:**
   - the full default lanes pass, apart from the known pre-existing local failures;
   - the builder e2e suite and the ci_flaky helper modules pass;
   - there are 0 new Sphinx warnings;
   - the mypy and ruff (`src/phenotypic`) finding sets are not worse than at `BASE_PRE` (418 / 25).
9. **Measurements:** a before/after table for the seven paths is committed in the report and quoted in the PR.
10. **Documentation:** the updates listed above land in the same change.
11. **No numeric change:** no numba kernel, colour constant value or algorithm output changes; existing numeric and golden tests pass unmodified.

## Non-goals

- Deferring scipy, skimage or pandas (49/94/81 module-level importers; the core `Image` handler chain needs them).
- Python 3.15's native lazy imports (PEP 810). The project requires `>=3.11, <3.13`.
- Guarding `phenotypic-tune` startup. It benefits from the lazy package `__init__` but gets no tier of its own.
- Deferring the run console, or browse unless the tier-3 measurement requires it.
- Memory footprint. First-use latency in notebooks is accepted (B2).
- Moving the Dash-free modules `_cli/_cli_error_outputs.py` imports out of `_gui` (a non-goal carried from the private-gui spec).

## Amendment A — findings while writing the plan (2026-09-11)

The plan resolves these against the code; they refine the design without changing its decisions.

| # | Spec text | Finding | Change |
|---|---|---|---|
| P1 | Design §3: the CLI's phenotypic imports "move into the command body or into the helpers that use them" | 36 test files import from `phenotypic.phenotypicCLI`. 27 `mock.patch` sites target names on it, 5 of them imported names such as `create_execution_strategy`, and a function-local import would silently bypass those patches. A post-change static closure (`probes/post_change_closure.py`) shows only 9 of the 25 phenotypic import statements stay heavy (24 names). | Those 24 names plus `available_modes` are bound into module globals by `_load_cli_runtime()` using `globals().setdefault`, so an active patch wins. It is called first in `phenotypic_cli`, `_migrate_legacy_success_evidence`, `_regenerate_missing_overlays` and `_handle_recompile_slurm`. A module `__getattr__` serves external access, and a `TYPE_CHECKING` block serves mypy. The 16 light import statements stay at module level. |
| P2 | Design §4 changes only the launcher module | `phenotypic._gui.shell/__init__.py` eagerly imports `_app` (dash) and `_launcher`, and the console script imports that package first. | `shell/__init__.py` gets PEP 562 re-exports. |
| P3 | Appendix A lists `_core/_image_parts/plot_accessor/_diagnostics_plotter.py`, `correction/_color_correction/_color_correction_report.py` and `grid/_grid_fit_report.py` as simple moves | Each imports `sdk_.viz.figures._theme`, which imports plotly by contract, so moving their own plotly/matplotlib imports would not keep plotly out; none is on a guarded path. | Left unchanged. 33 modules are edited. |
| P4 | Design §2: the dash handler's docs-build renderer switch moves into a first-plot helper | Under `PHENOTYPIC_DOCS_BUILD`, today's switch runs at `import phenotypic` and affects every plotly figure the notebook kernel renders, not only accessor figures. | The switch moves to `_startup_perf`. It still runs at `import phenotypic`, and only when the variable is set. |
| P5 | Design §1: "the 14 subpackages" | `phenotypic/__init__.py` imports 15: `abc_` plus 14. | 15. |
| P6 | Design §2 names the `sdk_.colourspace` importers without their use sites | The use sites are `_xyz_conversion.rgb_to_xyz`, `ColorDenoise._operate` and `DenoiseBlockMatch._denoise_channel`. | They are named in the plan's deferral table. |
| P7 | Tests, tier 2 runs the `phenotypic-gui` console script | A console script's `sys.modules` cannot be inspected after it exits. | The guard runs `phenotypic._gui.shell._launcher.main(["--help"])` in a fresh interpreter. The installed script's exit code and help text stay covered by `tests/integration/gui/test_console_script.py`. |
| P8 | Where each lazy `__init__` defines `__getattr__` is unspecified | A cycle that re-enters a package `__init__` before its end would not see a `__getattr__` defined at the bottom. | Each lazy `__init__` defines its lazy map and `__getattr__` before its first eager import. |

## Known risks

- **Cycles in leaf-first imports the sweep does not cover.** The sweep covers every package and the named entry modules, not every leaf module. Mitigation: the `__init__` of the package containing any leaf runs first, so package coverage catches the cycles found so far; the three known sites are fixed at the source.
- **Code that enumerates `phenotypic` with `inspect.getmembers`.** It now triggers lazy loads through `__dir__`. That is correct, but slower, for such callers. None exist in `src/` (only subpackages are enumerated).
- **Builder first-visit latency (~1 s).** It now falls on the first user click rather than on startup. It is the same trade the viewer already makes.

## Appendix A — deferral inventory (AST; runtime module-level imports; `_gui/` and `TYPE_CHECKING` excluded)

`[module-scope use]` marks the 7 modules handled individually in Design §2. The remaining modules move the import into the functions that use it.

- **colour (10):**
  - `_core/_image_parts/color_space_accessors/_chromaticity_xy_accessor.py`, `_cielab_accessor.py`, `_xyz_conversion.py`, `_xyz_d65_accessor.py`;
  - `correction/_color_correction/_color_checker_profile.py` [module-scope use], `_color_correction_report.py`, `_color_corrector.py`, `_helpers.py` [module-scope use];
  - `sdk_/colourspace.py` [module-scope use];
  - `util/_robust_color_stats.py`.
- **numba (2):** `sdk_/branch_pathfinding/_dijkstra_kernels.py` [module-scope use], `sdk_/reconnect/_tensor_voting.py` [module-scope use]. Deferred at the importers (D6).
- **h5py (2):** `_core/_image_parts/_image_io_handler.py`, `sdk_/hdf_.py`.
- **mahotas (1):** `measure/_measure_texture.py` (import after a `filterwarnings` call, `noqa: E402`; the filter moves with it).
- **cv2 (3):** `enhance/_flatten_illumination.py`, `enhance/_subtract_opening.py`, `refine/_extract_colony_core.py`.
- **bm3d (3):** `correction/_color_denoise.py`, `correction/_denoise_block_match.py`, `enhance/_enhance_block_match.py`.
- **plotly (9):**
  - `_core/_image_parts/accessor_abstracts/_image_accessor_base_parents/_accessor_dash_handler.py` [module-scope use];
  - `_core/_image_parts/plot_accessor/_detect_modes_plotter.py`, `_diagnostics_plotter.py`;
  - `analysis/qc/_expected_vs_detected.py`, `_grid_occupancy.py`, `_replicate_agreement.py`;
  - `correction/_color_correction/_color_correction_report.py`;
  - `grid/_grid_fit_report.py`;
  - `sdk_/viz/figures/_theme.py`. It imports plotly by contract (`tests/unit/viz/test_import_rules.py`), so it stays eager; its importers defer it.
- **matplotlib (14):**
  - `_cli/_cli_process_single.py` [module-scope use; unchanged];
  - `_core/_image_parts/_grid_image_handler.py`;
  - `_core/_image_parts/accessor_abstracts/_image_accessor_base_parents/_accessor_mpl_handler.py`;
  - `_core/_image_parts/accessors/_grid_accessor.py`, `_objmap_accessor.py`, `_objmask_accessor.py`;
  - `_core/_image_parts/color_space_accessors/_hsv_accessor.py`;
  - `_core/_image_parts/plot_accessor/_base_plotter.py`, `_diagnostics_plotter.py`;
  - `analysis/abc_/_model_fitter.py`, `analysis/edge/_edge_correction.py`, `analysis/filter/_mad_outlier.py`, `analysis/filter/_tukey_outlier.py`;
  - `sdk_/orientation_fields/_plots.py`.
