# Private GUI module — `phenotypic.gui` → `phenotypic._gui`

**Date:** 2026-09-10 · **Branch:** `refactor/private-gui` · **Base:** `main @ 0117572d9`

## Objective

The GUI stops being an importable public package. It lives at `phenotypic._gui`,
and users start it one way only: the `phenotypic-gui` console script
(`uv run phenotypic-gui …` or `phenotypic-gui …`).

## Decisions (user-confirmed, 2026-09-10)

| ID | Decision |
|----|----------|
| D1 | Remove the hub's module entry (`python -m phenotypic.gui`, i.e. `gui/__main__.py`). **Keep** the five sub-app debug launchers — `builder`, `results_viewer`, `run_console`, `browse`, `analysis` — as `python -m phenotypic._gui.<sub_app>`. They are contributor tools: documented in CLAUDE.md files only, never in user docs. |
| D2 | Hard break. No `phenotypic.gui` compatibility shim. The lazy exports on `gui/__init__.py` (`launch_gui`, `launch_results_viewer`, `create_*_app`, `OperationRegistry`, `GUI_AVAILABLE`, `SandboxRoot`, `BuilderState`) move with the package and become private. |
| D3 | Delete the GUI API reference section (`docs/source/api_reference/gui/`) plus the two scripts that generate pages into it, their shared helper, and its test. |
| D4 | Historical records under `docs/superpowers/` (specs, plans, reports, reviews) are **not** rewritten. They describe the tree as it was. |

## Must not change (runtime paths that merely contain `gui`)

Renaming any of these orphans user state or changes behaviour; none refers to the package:

- the external viewer cache root `<user cache>/phenotypic/gui/viewer_cache` (`results_viewer/_output_root.py:1044`, pinned by `tests/unit/gui/test_viewer_cache_ownership.py`);
- the GUI submitter log directory `<output>/.phenotypic/logs/gui` (`run_console/_slurm.py:438`, `run_console/_slurm_observer.py:908`, the `'gui' in path.parts` test in `run_console/_callbacks.py:795`, the CLI's `gui_logs.name != "gui"` check at `phenotypicCLI.py:948`, and their tests);
- the sandbox directory `.phenotypic-gui` and the thread-name prefix `"phenotypic-gui"`;
- the console-script name `phenotypic-gui`;
- test directory names (`tests/unit/gui`, `tests/integration/gui`, `tests/gui`, `tests/e2e/gui`) and `docs/source/tutorials/gui/`.

## Non-goals

- Moving the Dash-free modules the CLI imports from the GUI (`results_viewer/_curation_labels.py`, `results_viewer/_error_tab/_publication.py`, used by `_cli/_cli_error_outputs.py`) into `sdk_`. The rename carries the imports; the layering fix is a follow-up.
- `phenotypic-tune` / `phenotypic.tune` (a separate package; the GUI's `tune` sub-app moves with `_gui`).

## Acceptance criteria

1. `importlib.util.find_spec("phenotypic.gui") is None` — including no leftover `src/phenotypic/gui/` directory that would resolve as a namespace package.
2. `importlib.util.find_spec("phenotypic._gui.__main__") is None`; each of the five sub-app `__main__` modules resolves.
3. `[project.scripts] phenotypic-gui = "phenotypic._gui.shell._launcher:main"`, and the built wheel ships `phenotypic/_gui/**/*.{css,js,png}`.
4. No tracked file outside `docs/superpowers/` names `phenotypic.gui` or `phenotypic/gui`, except `tests/unit/gui/test_private_package.py`, which must name the removed path to assert it is gone (controller ruling during Task 1).
5. Nothing in tests, scripts, or CI starts the hub with `python -m`; all hub launches go through the console script, and a missing console script **fails** (never skips).
6. User docs (README, `docs/source/**`) name only `phenotypic-gui`.
7. The affected test surface matches its pre-change baseline; the mypy and ruff counts are not worse than 418 / 65.

## Addendum A — CI browser and gui-checks failures (user-requested, 2026-09-10)

Added at the user's request: first "verify and fix [the Playwright missing-browser issue] in this pr as well", then "fix all gui-checks failures". Each root cause was established before any fix; the evidence (CI run IDs, probes) is in the execution ledger and summarised here.

### Findings

| # | Symptom in CI | Root cause | Disposition |
|---|---|---|---|
| A1 | `run-pytest.yml`: `BrowserType.launch: Executable doesn't exist … chromium_headless_shell-1217` in the four `tests/gui/**/*_browser.py` modules (run 34462064512) | That run's commit predates `f35397c00` (sharded PR suite), and its single job never installed a browser. On `main`, the `gui-browser` shard installs Chromium and owns all four modules (run 34506368123: 2781 passed, 0 missing-browser errors) | Already fixed by the sharding. **Add a guard**: nothing checks that a browser-fixture test lands in a `playwright: true` shard |
| A2 | `gui-checks` e2e: both tests in `test_analysis_app.py` time out clicking `#analysis-model-dropdown`, which Playwright resolves to `<button disabled>` | The e2e helper `publish_coherent_terminal_evidence` writes only a manifest. Since `f4ba81004` (`resolve_run_state`), an output with no processing state resolves to `incomplete`, and `output_mutations_disabled` renders every persistent control disabled. The product is right; the helper no longer models the completed run its docstring promises | **Fix the helper at its source**, for every full-run fixture that calls it |
| A3 | `gui-checks` smoke-capture: `TypeError: publish_aggregate_snapshot() missing 1 required keyword-only argument: 'source_work_ids'` (`scripts/capture_gui_tutorial_screenshots.py:444`) | `eadf0fdf5` made `source_work_ids` required so the type checker would find every caller; `scripts/` is not type-checked, so this caller was missed | **Fix the call** |
| A4 | `gui-checks` e2e: `test_colony_shared_camera` ×2 and `test_scatter_tab` ×1 (older runs) | Fixed on `main` by `d68a7792a` (2026-09-04); all of them pass in run 34506368188 | No change |

### Decisions

| ID | Decision |
|----|----------|
| DA1 | The browser guard lives in `tests/unit/ci/test_pytest_shard_manifest.py`. A browser test is a module that requests a pytest-playwright fixture (`page`, `browser`, `browser_name`, `browser_type`, `launch_browser`, `new_context`, `playwright`) or imports `playwright`. The bare name `context` does not count: two unrelated test modules define a `context` fixture of their own. |
| DA2 | The completed-run publisher lives in `tests/_output_layout.py`, importable without `PLAYWRIGHT=1`. It is store-preserving: it reuses a store the fixture already wrote, and promotes a minimal one only where none exists. The e2e `publish_coherent_terminal_evidence` delegates to it. `_build_sandbox`'s placeholder output (a zero-byte master no viewer can bind) keeps a manifest-only helper. |
| DA3 | The capture script passes `source_work_ids` from the finished run's authorized success set — the derivation `sdk_/_hdf_to_zarr.py` uses for a finished tree. |
| DA4 | Out of scope, recorded as follow-ups: (a) the capture script's tutorial analysis chain — the `Metadata_StrainID` KeyError has never failed a check; its `metadata.csv` keys `plate_001.tif` against extension-less image names, the CLI run passes no `--metadata`, and `LogGrowthModel` uses a date string as its time label, so a real fix changes the rendered tutorial screenshots; (b) the nightly Windows collection error in `tests/unit/test_fixtures.py`; (c) the macOS-only Viv e2e timeouts. |

### Acceptance criteria

8. A test fails when a sharded test module that requests a pytest-playwright fixture is assigned to a shard without `"playwright": true`, or when `run-pytest.yml` stops installing Chromium for those shards — each shown by mutation.
9. Publication over a fixture's outputs resolves to `completion == "complete"` with mutations enabled, leaves the fixture's `deliverables/` bytes and any existing store unchanged, and fails loudly on a missing core file or a miscounted fixture — shown by a non-browser test that can fail.
10. With `PLAYWRIGHT=1`, both previously failing `test_analysis_app.py` tests pass, and the other e2e modules that call the helper have no new failures against Task 2's local e2e run.
11. The capture script's dataset → CLI → seeding path completes without the `TypeError`, and its aggregate proof certifies exactly the run's authorized success set.

## Known risk

Other worktrees have uncommitted edits under `src/phenotypic/gui/`: `.claude/worktrees/renaming-refactor` (20 files), `.worktrees/smart-qc-gui-wt` (16 committed + 5 dirty), codex `cf79` (20 dirty). Reviving any of them after this merge means rebasing across the move. `git mv` keeps rename detection, so most hunks should carry across.
