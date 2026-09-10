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
- the GUI submitter log directory `<output>/.phenotypic/logs/gui` (`run_console/_slurm.py:438`, `run_console/_slurm_observer.py:908`, the `'gui' in path.parts` test in `run_console/_callbacks.py:795`, and their tests);
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
4. No tracked file outside `docs/superpowers/` names `phenotypic.gui` or `phenotypic/gui`.
5. Nothing in tests, scripts, or CI starts the hub with `python -m`; all hub launches go through the console script, and a missing console script **fails** (never skips).
6. User docs (README, `docs/source/**`) name only `phenotypic-gui`.
7. The affected test surface matches its pre-change baseline; the mypy and ruff counts are not worse than 418 / 65.

## Known risk

Other worktrees have uncommitted edits under `src/phenotypic/gui/`: `.claude/worktrees/renaming-refactor` (20 files), `.worktrees/smart-qc-gui-wt` (16 committed + 5 dirty), codex `cf79` (20 dirty). Reviving any of them after this merge means rebasing across the move. `git mv` keeps rename detection, so most hunks should carry across.
