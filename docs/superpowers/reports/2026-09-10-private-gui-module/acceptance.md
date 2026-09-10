# Acceptance — private GUI module

**Spec:** `docs/superpowers/specs/2026-09-10-private-gui-module/spec.md` · **Checked at:** `4cbec8a88` (Task 4 Steps 1–4 at `da352a50e` / `b16182b9b`, the last code commits) · **Date:** 2026-09-10

Every check that expects an empty result has a positive control run on `main` (`0117572d9`). An empty grep is evidence only if the same pattern demonstrably matches something.

## Criteria

| # | Criterion | Result | Evidence |
|---|---|---|---|
| 1 | `phenotypic.gui` does not resolve, including as a namespace package | PASS | `src/phenotypic/gui` absent on disk; `find_spec("phenotypic.gui")` → `None` |
| 2 | Hub `__main__` gone; the five sub-app launchers resolve | PASS | `find_spec("phenotypic._gui.__main__")` → `None`; `analysis`, `browse`, `builder`, `results_viewer`, `run_console` `__main__` all resolve |
| 3 | Console script targets the private launcher; the wheel ships `_gui` assets | PASS | `pyproject.toml:145` `phenotypic-gui = "phenotypic._gui.shell._launcher:main"`; `:253-255` `_gui/**/*.{css,js,png}`; `test_package_contents.py -m slow` → 5 passed |
| 4 | No tracked file outside `docs/superpowers/` names the old package, except the guard test (amended wording) | PASS | Only `tests/unit/gui/test_private_package.py:4,21,22`; the same pattern matches 482 files on `main` |
| 5 | Nothing launches the hub with `python -m`; a missing console script fails, never skips | PASS | Token scan of tests/scripts/.github/src: 0 hub launches, 2 sub-app launches found (the scan demonstrably matches); 3 `sysconfig.get_path("scripts")` lookups, 0 `PATH` fallbacks; `pytest.fail` at `test_console_script.py:34`, `RuntimeError` at `capture_gui_tutorial_screenshots.py:545` and `tests/e2e/gui/conftest.py:219` |
| 6 | User docs name only `phenotypic-gui` | PASS | No `python -m phenotypic._?gui` / `phenotypic[./]_gui` / `phenotypic[./]gui` in README.md or docs/source (21 files match on `main`); `phenotypic-gui` named in README (4), `gui_hub.md` (6), `getting_started.rst` (4) |
| 7 | Test surface matches its baseline; mypy and ruff not worse | PASS | See Task 4 below |

## "Must not change" runtime paths

PASS. Sorted quoted-`"gui"` literal lines across `*.py`, `main` vs HEAD: only 7 removals — the 2 deleted generator lines, plus the 5 planned package-path hand edits — and 0 additions. Literal counts identical on `main`/HEAD: `".phenotypic-gui"` 2/2, `"phenotypic-gui"` 3/3, `gui_logs.name != "gui"` 1/1, `_EXTERNAL_VIEWER_CACHE_SUBDIR` 3/3.

## Task 4 gates

| Step | Gate | Result |
|---|---|---|
| 1 | Test surface, node by node vs `baseline.xml` | PASS — `3006 passed, 16 skipped, 3 xfailed, 29 warnings`, no failures. Only-in-baseline = the 12 tests of the deleted `test_reference_generators.py`; only-in-final = the 9 tests of `test_private_package.py`; 0 status changes |
| 2 | mypy / ruff finding sets vs `main` (`compare_findings.py`) | PASS — mypy 418 = 418 and ruff 65 = 65, identical multisets. mypy ran with a fresh cache on both sides: the branch's incremental cache hid one pre-existing finding |
| 3 | Wheel contents | PASS — 5 passed |
| 4 | Docs build with `sphinx-build -n`, warnings vs `main` (`compare_findings.py docs`) | PASS — exit 0, `build succeeded, 2340 warnings` (`main`: 2386); 0 new, 46 gone (the deleted GUI API reference pages); 0 log lines naming the old package |

## Pre-existing findings surfaced along the way (not caused by this branch)

- `src/phenotypic/_cli/_cli_finalize_run.py:19`: a `TYPE_CHECKING` import of `CommitGuard` from `phenotypic.sdk_._publication_guard`, a module that does not exist. `CommitGuard` is defined in `phenotypic.sdk_._atomic_io`.
- `tests/unit/test_ome_zarr_invariants.py:269`: a docstring with an invalid escape sequence (`SyntaxWarning`).
- `tests/e2e/gui/conftest.py`: the whole suite skips unless `PLAYWRIGHT=1`, a skip-when-unconfigured pattern.
- macOS-local e2e: 4 Viv/GL tests time out; 1 builder-preview test collides on a shared temp path under `-n 4`.
