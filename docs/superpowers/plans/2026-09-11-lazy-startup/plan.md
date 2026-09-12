# Lazy Startup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `import phenotypic`, `phenotypic --help`, `phenotypic-gui --help`, the composed GUI hub and `from phenotypic import Image` load only what they need. The pieces:

- lazy package `__init__`s;
- point-of-use imports of the heavy libraries that few modules import;
- a core-free CLI help path;
- a GUI builder built on its first request.

Deterministic subprocess guards keep all of it that way.

**Architecture:**

- **Lazy re-exports.** PEP 562 `__getattr__` re-exports in `phenotypic`, `phenotypic.sdk_`, `phenotypic.abc_` and `phenotypic._gui.shell`.
- **Import cycles.** Three cycles that today's eager order hides are fixed at their source.
- **Point-of-use imports.** Each deferred import moves into the function that uses it; annotation-only names stay under `TYPE_CHECKING`.
- **CLI.** `phenotypicCLI.py` binds its nine heavy import statements through `_load_cli_runtime()` (with `setdefault`, so `mock.patch` keeps working).
- **GUI.** The hub mounts the builder through the existing `ToolSession`/`_SessionProxy` pair.
- **Preload.** Pipeline-running entry points call `load_runtime_dependencies()`, so broken installs still fail before the first image.

**Tech Stack:** Python 3.11–3.12, uv, pytest (+xdist, pytest-timeout), click, argparse, Dash/Flask/Werkzeug, pydantic v2, AST-based static tests.

**Spec:** `docs/superpowers/specs/2026-09-11-lazy-startup/spec.md` (read it with Amendment A, which records P1–P8).

## Global Constraints

- **Tooling.** `uv` only — never bare `python`/`pip`. Run `ruff check --fix` only with explicit paths.
- **Pytest invocations.**
  - Always set `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg`, use `-o addopts= -m "not slow" -p no:cacheprovider`, and pass an explicit `-n`.
  - Never use `-x` for a measurement (`run-phenotypic-test` skill).
  - E2E needs `PLAYWRIGHT=1`.
- **Failing tests.** A test that cannot run must fail, not skip.
- **Commits.** Each task's implementer commits its own work.
  - Stage by explicit path, never `git add -A`/`git commit -a`, and check `git diff --cached --stat` before committing.
  - End every commit message with the `Co-Authored-By:` trailer your environment specifies (it names the model that wrote the commit), followed by `Claude-Session: https://claude.ai/code/session_017uC78YpUE6Wh24wuAzWFe6`.
- **Sequencing.** Tasks run strictly one at a time: one checkout, one git index. Do not push.
- **Line endings.** The working tree is CRLF (`core.autocrlf=true`).
  - A multi-line exact-match edit that your editor cannot apply must go through a `uv run python -c` script that reads bytes, asserts the old block occurs exactly once, and writes back with the file's own line endings.
  - `git diff --numstat` must never show a whole-file line-ending flip.
- **Shell.** The shell is zsh:
  - quote globs;
  - unquoted `$var` is not word-split, so use arrays;
  - a backtick inside double quotes starts command substitution.

  macOS `grep -E` has no `\s`/`\b`/`\d`, and `$` fails on CRLF lines. Show a positive control before trusting an empty grep.
- **No numeric change.**
  - Never edit `src/phenotypic/sdk_/reconnect/_tensor_voting.py`, `src/phenotypic/sdk_/branch_pathfinding/_dijkstra_kernels.py`, `src/phenotypic/sdk_/colourspace.py`, `src/phenotypic/sdk_/hdf_.py`, or anything under `docs/superpowers/**/refs/`.
  - Never change an algorithm, a constant's value, a public name or an `__all__` list.
- **Unchanged modules** (spec Amendment A P3):
  - `correction/_color_correction/_color_correction_report.py`
  - `grid/_grid_fit_report.py`
  - `sdk_/viz/figures/_theme.py`
  - `correction/_color_correction/_color_checker_profile.py`
  - `correction/_color_correction/_helpers.py`
  - `_cli/_cli_process_single.py`'s `matplotlib.use("Agg")` block
- **GUI docs.** A change under `src/phenotypic/_gui/` needs `src/phenotypic/_gui/FEATURES.md` modified in the same PR (Task 6 does it).

## Execution DAG, shapes, models, gates

```
T0 baseline (orchestrator)
 └─► T1 startup constants + probe helper      [Leaf,     sonnet]
      └─► T2 lazy package inits + cycle fixes  [Keystone, opus]
           └─► T3 point-of-use: image core     [Sweep,    sonnet]
                └─► T4 point-of-use: operations [Sweep,   sonnet]
                     └─► ◆ PHASE 1 GATE (independent review + full default lanes + mypy/ruff compare)
                          └─► T5 CLI help path + preload [Seam, opus]
                               └─► T6 GUI launcher + lazy builder [Seam, opus]
                                    └─► ◆ PHASE 2 GATE (independent review + CLI/GUI surface + builder e2e)
                                         └─► T7 docs + mutation proofs + measurements [Leaf, opus]
                                              └─► T8 final review, /simplify, regression, finish (orchestrator)
```

- **File overlap.** Every task edits `tests/unit/ci/test_startup_imports.py` or files downstream of T2, so nothing runs in parallel.
- **Per-task review.** Each implementation task gets a task reviewer (opus) before the next task starts.
- **Phase gates.** Each gate is an independent reviewer (`xander-local:implementation-test-reviewer`, opus) over the phase's commit range. Findings go through one fix dispatch and a scoped re-review before the next phase.

## Shared facts for implementers

- **Heavy import sets** (introduced by T1, in `src/phenotypic/_startup_perf.py`):
  - `HEAVY_STARTUP_MODULES` = bm3d, colour, cv2, dash, h5py, mahotas, matplotlib, numba, pandas, plotly, polars, pyarrow, scipy, skimage.
  - `DEFERRED_RUNTIME_MODULES` = bm3d, colour, cv2, h5py, mahotas, matplotlib.pyplot, numba, plotly.
- **Probe helper.** Every entry-point guard runs through `tests._startup_probe.run_startup_probe(body)` in a fresh interpreter, because other tests in the same xdist worker have already filled `sys.modules`.
- **Measured baseline (Task 0, `startup-before.json`).** `import phenotypic` 1.572 s; `from phenotypic import Image` 1.568 s; CLI help 1.612 s; GUI help 1.797 s; hub to servable 2.008 s; hub + first `/builder/` 2.051 s; bare interpreter 0.011 s.
- **Do not trust `probes/post_change_closure.py` for new claims.** It is static: it cannot see `from phenotypic import Image` reaching the core through the lazy `__getattr__`, nor imports executed at class-creation time. Both blind spots produced wrong tier claims that the plan review caught (C1, C2, C5). Re-derive any new light/heavy claim with a runtime trace in a fresh interpreter.
- **Point-of-use rule** (T3, T4). For each row of the task's deferral table:
  1. Delete the module-level import statement(s) in the **Remove** column. If an import statement also binds names not in the table, keep those names at module level.
  2. For each name listed under **TYPE_CHECKING**, add it to the module's `if TYPE_CHECKING:` block. Create the block after the remaining imports if none exists, and add `TYPE_CHECKING` to the module's `from typing import …` line or add `from typing import TYPE_CHECKING`. These names appear only in annotations. Every module with a `TYPE_CHECKING` entry has `from __future__ import annotations`, so its annotations are strings at runtime. (Three edited modules lack that import — `_chromaticity_xy_accessor.py`, `_cielab_accessor.py`, `_xyz_d65_accessor.py` — and none of them has a `TYPE_CHECKING` entry.)
  3. For each function in **Import inside**, add the import statement for the names listed against it as the **first statement after the function's docstring**, using the same import form as the removed line.
  4. The function names come from an AST scan of the current tree. If a named function does not exist, stop and report `NEEDS_CONTEXT` with the file's function list.
  - Worked example — `src/phenotypic/_core/_image_parts/accessors/_objmask_accessor.py`. Delete line 8 `import matplotlib.pyplot as plt`. Add `import matplotlib.pyplot as plt` under `if TYPE_CHECKING:`, because `show`'s return annotation uses `plt`. Add `import matplotlib.pyplot as plt` as the first statement after the docstring of `show`.

## Test surface commands

- **Per task:** each task's Step "Test surface" lists its exact command.
- **Phase 1 gate** (full default lanes, once):
  `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest tests/unit tests/integration tests/gui tests/smoke -q --no-header -p no:randomly -p no:cacheprovider -o addopts= -m "not slow" -n 8 -rfE > /tmp/lazy-phase1-lanes.log 2>&1; echo "exit=$?" >> /tmp/lazy-phase1-lanes.log`
- **Known local failures at baseline:** `tests/unit/test_ngff_schema_fixtures.py::test_schema_matches_recorded_digest[image.schema|label.schema|ome.schema|_version.schema]`. They are caused by autocrlf rewriting the vendored bytes.

---

### Task 0: Baseline (orchestrator)

**Files:**
- Create: `docs/superpowers/reports/2026-09-11-lazy-startup/baseline.md`
- Create: `docs/superpowers/reports/2026-09-11-lazy-startup/startup-before.json`

- [ ] **Step 1:** Record `BASE_PRE=$(git rev-parse HEAD)` (the commit carrying this plan) in the SDD ledger.
- [ ] **Step 2:** Build a baseline worktree:

  ```bash
  git worktree add --detach /tmp/pht-lazy-base "$BASE_PRE"
  cd /tmp/pht-lazy-base && uv sync --group dev --group test-qt --group docs --all-extras
  ```

- [ ] **Step 3:** Measure the baseline from the worktree, so its interpreter and source are the ones timed:

  ```bash
  cd /tmp/pht-lazy-base && uv run python /Users/alex/Projects/PhenoTypic/docs/superpowers/plans/2026-09-11-lazy-startup/measure_startup.py --label before --out /Users/alex/Projects/PhenoTypic/docs/superpowers/reports/2026-09-11-lazy-startup/startup-before.json
  ```

- [ ] **Step 4:** Capture the mypy and ruff finding files at the baseline:

  ```bash
  cd /tmp/pht-lazy-base && uv run mypy --cache-dir /tmp/lazy-mypy-cache-before src/phenotypic > /tmp/lazy-mypy-before.txt 2>&1; tail -1 /tmp/lazy-mypy-before.txt
  cd /tmp/pht-lazy-base && uv run ruff check src/phenotypic > /tmp/lazy-ruff-before.txt 2>&1; tail -3 /tmp/lazy-ruff-before.txt
  ```

- [ ] **Step 5:** Run the builder e2e baseline and record its summary line plus every failing test id:

  ```bash
  cd /tmp/pht-lazy-base && PLAYWRIGHT=1 QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest tests/e2e/gui/builder -q -p no:cacheprovider -o addopts= -n 4 > /tmp/lazy-e2e-builder-before.log 2>&1; tail -3 /tmp/lazy-e2e-builder-before.log
  ```

- [ ] **Step 6:** Write `baseline.md` and commit it with `startup-before.json`. `baseline.md` holds:
  - `BASE_PRE`;
  - the seven best-of-5 timings;
  - the mypy summary line;
  - the ruff count;
  - the builder e2e summary and failures.

  Keep `/tmp/pht-lazy-base` until Task 8.

---

### Task 1: Startup constants, preload function, probe helper

**Files:**
- Modify: `src/phenotypic/_startup_perf.py`
- Create: `tests/_startup_probe.py`
- Create: `tests/unit/ci/test_startup_imports.py`

**Interfaces — Produces:**
- `phenotypic._startup_perf.HEAVY_STARTUP_MODULES: tuple[str, ...]`
- `phenotypic._startup_perf.DEFERRED_RUNTIME_MODULES: tuple[str, ...]`
- `phenotypic._startup_perf.load_runtime_dependencies() -> None`
- `tests._startup_probe.run_startup_probe(body: str) -> dict[str, Any]`. `body` must bind a JSON-serialisable `report`.

- [ ] **Step 1: Write the probe helper** `tests/_startup_probe.py`:

```python
"""Run a snippet in a fresh interpreter and read back the report it prints.

A fresh process is the only honest place to ask what an import loads: other tests
in the same xdist worker have already filled ``sys.modules``, so an in-process check
passes or fails by accident. A probe that crashes, hangs or prints no report fails
the calling test.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Prefix of the one stdout line carrying the probe's JSON report.
_REPORT_MARKER = "__PHENOTYPIC_STARTUP_PROBE__="

#: Seconds before a probe is killed; a hang must fail, not stall the suite.
PROBE_TIMEOUT_SECONDS = 60


def run_startup_probe(body: str) -> dict[str, Any]:
    """Execute ``body`` in a fresh interpreter and return the ``report`` it binds.

    Args:
        body: Python source run after ``import json, sys``. It must bind a
            JSON-serialisable name ``report``.

    Returns:
        The decoded ``report``.

    Raises:
        AssertionError: If the interpreter exits non-zero or prints no report.
        subprocess.TimeoutExpired: If the probe runs past
            :data:`PROBE_TIMEOUT_SECONDS`.
    """
    program = f"import json, sys\n{body}\nprint({_REPORT_MARKER!r} + json.dumps(report))\n"
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "MPLBACKEND": "Agg"}
    env.pop("PHENOTYPIC_DOCS_BUILD", None)
    env.pop("PYTEST_CURRENT_TEST", None)
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        timeout=PROBE_TIMEOUT_SECONDS,
        env=env,
        cwd=REPO_ROOT,
    )
    reports = [line[len(_REPORT_MARKER):] for line in result.stdout.splitlines() if line.startswith(_REPORT_MARKER)]
    if result.returncode != 0 or not reports:
        raise AssertionError(
            f"startup probe failed with exit {result.returncode}\n"
            f"--- stdout (tail) ---\n{result.stdout[-2000:]}\n"
            f"--- stderr (tail) ---\n{result.stderr[-4000:]}"
        )
    return json.loads(reports[-1])
```

- [ ] **Step 2: Write the failing tests** `tests/unit/ci/test_startup_imports.py`:

```python
"""Startup-path guards: what each entry point loads, and import-order independence.

Every guard runs its entry point in a fresh interpreter (``tests._startup_probe``)
and pairs the absence it asserts with a positive control, so a probe that silently
imported nothing cannot pass.
"""

from __future__ import annotations

from phenotypic._startup_perf import DEFERRED_RUNTIME_MODULES, HEAVY_STARTUP_MODULES
from tests._startup_probe import run_startup_probe


def test_load_runtime_dependencies_imports_every_deferred_module() -> None:
    """A pipeline run imports the deferred libraries up front, so a broken one fails first."""
    report = run_startup_probe(
        "from phenotypic._startup_perf import DEFERRED_RUNTIME_MODULES, load_runtime_dependencies\n"
        "before = [m for m in DEFERRED_RUNTIME_MODULES if m in sys.modules]\n"
        "load_runtime_dependencies()\n"
        "report = {'before': before, 'missing': [m for m in DEFERRED_RUNTIME_MODULES if m not in sys.modules]}\n"
    )
    assert report["missing"] == []


def test_every_deferred_module_is_watched_at_startup() -> None:
    """A module deferred for startup must also be one the startup guards watch."""
    watched = set(HEAVY_STARTUP_MODULES) | {"matplotlib.pyplot"}
    assert set(DEFERRED_RUNTIME_MODULES) <= watched
```

- [ ] **Step 3: Run the tests.** They must fail:

  ```bash
  QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest tests/unit/ci/test_startup_imports.py -q -o addopts= -p no:cacheprovider -n 0
  ```

  Expected: a collection error, `ImportError: cannot import name 'DEFERRED_RUNTIME_MODULES'`.

- [ ] **Step 4: Implement.** Make these changes in `src/phenotypic/_startup_perf.py`:
  - Add `import warnings` to the imports. Do **not** add `import os`: only Task 3's renderer function uses it, and an unused import fails this task's own ruff step.
  - Replace `__all__ = ["IMPORT_STARTED_AT", "install_lazy_colour_plotting"]` with:

    ```python
    __all__ = [
        "DEFERRED_RUNTIME_MODULES",
        "HEAVY_STARTUP_MODULES",
        "IMPORT_STARTED_AT",
        "install_lazy_colour_plotting",
        "load_runtime_dependencies",
    ]

    #: Heavy third-party modules that no light entry point may load: ``import phenotypic``,
    #: ``phenotypic --help``, ``phenotypic-gui --help`` and the composed GUI hub before any
    #: page is visited. Guarded by ``tests/unit/ci/test_startup_imports.py``.
    HEAVY_STARTUP_MODULES: tuple[str, ...] = (
        "bm3d", "colour", "cv2", "dash", "h5py", "mahotas", "matplotlib",
        "numba", "pandas", "plotly", "polars", "pyarrow", "scipy", "skimage",
    )

    #: Libraries imported at their point of use rather than at module level, so
    #: ``from phenotypic import Image`` does not pay for them.
    DEFERRED_RUNTIME_MODULES: tuple[str, ...] = (
        "bm3d", "colour", "cv2", "h5py", "mahotas", "matplotlib.pyplot", "numba", "plotly",
    )
    ```

  - Append after `install_lazy_colour_plotting` (above the module's final `install_lazy_colour_plotting()` call):

    ```python
    def load_runtime_dependencies() -> None:
        """Import every library in :data:`DEFERRED_RUNTIME_MODULES` now.

        Pipeline-running entry points call this before any image work. The libraries
        are deferred to their point of use for light startup, so without this a broken
        install (a numba/llvmlite mismatch, a binary that crashes on one node) would
        surface in the middle of the first image and be recorded as a per-image
        scientific failure rather than stopping the run at start. Any import error
        propagates.
        """
        warnings.filterwarnings("ignore", category=SyntaxWarning, module="mahotas")
        for module_name in DEFERRED_RUNTIME_MODULES:
            importlib.import_module(module_name)
    ```

- [ ] **Step 5: Run the tests.** Same command as Step 3. Expected: `2 passed`.
- [ ] **Step 6: Lint.** `uv run ruff check src/phenotypic/_startup_perf.py tests/_startup_probe.py tests/unit/ci/test_startup_imports.py` must print no new findings.
- [ ] **Step 7: Commit.** Subject: `feat(startup): name the heavy and deferred module sets, and a fail-fast preload`.

---

### Task 2: Lazy package inits and the three cycle fixes

**Files:**
- Modify: `src/phenotypic/__init__.py`
- Modify: `src/phenotypic/abc_/__init__.py`
- Modify: `src/phenotypic/sdk_/__init__.py`
- Modify: `src/phenotypic/_core/_image_parts/_grid_image_handler.py`
- Modify: `tests/unit/ci/test_startup_imports.py`

**Interfaces:**
- **Consumes:** T1's `HEAVY_STARTUP_MODULES` and `run_startup_probe`.
- **Produces:** `phenotypic.__getattr__`/`__dir__`; `phenotypic.sdk_._LAZY_ATTRS`; `phenotypic.abc_._LAZY_ATTRS`. Import-order independence for every package.

- [ ] **Step 1: Append the tests** to `tests/unit/ci/test_startup_imports.py`.
  - Merge `from pathlib import Path` and `import pytest` into the import block at the top of the file.
  - Import the name `phenotypic` inside the lazy-attribute test, so collection does not import it early.

```python
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"

#: Modules imported first by real entry points (console scripts, SLURM workers, users).
IMPORT_FIRST_ENTRY_MODULES = (
    "phenotypic._core._image",
    "phenotypic._core._grid_image",
    "phenotypic._core._image_pipeline",
    "phenotypic.phenotypicCLI",
    "phenotypic._gui.shell._launcher",
    "phenotypic._gui._operation_registry",
    "phenotypic._cli._cli_process_single",
    "phenotypic._cli._cli_staged_slurm_worker",
    "phenotypic._cli._cli_recompile_worker",
    "phenotypic._cli._cli_checkpoint_handler",
)


def _package_modules() -> list[str]:
    """Every package under ``src/phenotypic`` except the root.

    The ``refs`` guard is a no-op safeguard: the vendored reference trees live under
    ``docs/superpowers/specs/*/refs``, not under ``src/``.
    """
    modules = []
    for init in sorted((SRC_ROOT / "phenotypic").rglob("__init__.py")):
        parts = init.parent.relative_to(SRC_ROOT).parts
        if "refs" in parts or parts == ("phenotypic",):
            continue
        modules.append(".".join(parts))
    return modules


PACKAGE_MODULES = _package_modules()


def test_import_phenotypic_loads_no_heavy_module() -> None:
    """Tier 1: the bare package import pays for nothing it has not been asked for."""
    report = run_startup_probe(
        "import phenotypic\n"
        f"watched = {sorted(HEAVY_STARTUP_MODULES)!r}\n"
        "report = {'loaded': [m for m in watched if m in sys.modules],\n"
        "          'control': 'phenotypic._startup_perf' in sys.modules,\n"
        "          'version': phenotypic.__version__}\n"
    )
    assert report["control"] is True
    assert report["version"]
    assert report["loaded"] == []


def test_the_public_names_still_resolve_from_the_lazy_package() -> None:
    """Attribute access, ``from phenotypic import``, ``dir`` and unknown names keep their contract."""
    import phenotypic
    from phenotypic import Image, ImagePipeline

    assert phenotypic.Image is Image
    assert phenotypic.ImagePipeline is ImagePipeline
    assert phenotypic.detect.OtsuDetector.__name__ == "OtsuDetector"
    assert set(phenotypic.__all__) <= set(dir(phenotypic))
    with pytest.raises(AttributeError):
        phenotypic.NoSuchPhenotypicName  # noqa: B018


def test_package_discovery_found_the_tree() -> None:
    """An empty or broken discovery must not let the sweep pass vacuously."""
    assert len(PACKAGE_MODULES) >= 70, PACKAGE_MODULES


@pytest.mark.parametrize("module_name", [*PACKAGE_MODULES, *IMPORT_FIRST_ENTRY_MODULES])
def test_module_imports_first_in_a_fresh_interpreter(module_name: str) -> None:
    """No import order may be required: each module must import as the first thing a process does."""
    report = run_startup_probe(
        "import importlib\n"
        f"importlib.import_module({module_name!r})\n"
        "report = {'imported': True}\n"
    )
    assert report["imported"] is True
```

- [ ] **Step 2: Run the new tests (RED).**

  ```bash
  QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest tests/unit/ci/test_startup_imports.py -q -o addopts= -p no:cacheprovider -n 8
  ```

  Expected: `test_import_phenotypic_loads_no_heavy_module` fails, and every other test passes. The sweep passes today because the eager order hides the cycles.

- [ ] **Step 3: Rewrite `src/phenotypic/__init__.py`.** Keep the module docstring (lines 1–15) byte-identical. Replace everything after it with:

```python
__version__ = "0.19.0"
__author__ = "Alexander Nguyen"
__email__ = "anguy344@ucr.edu"

# Import first: stamps the import-start time as ``_IMPORT_STARTED_AT`` and installs a
# lazy stub for colour-science's eager-but-unused ``colour.plotting`` submodule before
# anything can import colour. Both happen as import side effects of ``_startup_perf``.
from ._startup_perf import IMPORT_STARTED_AT as _IMPORT_STARTED_AT  # noqa: F401

import importlib as _importlib
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any as _Any

#: Public classes, by the private module that defines each one.
_LAZY_CLASSES: dict[str, str] = {
    "Image": "._core._image",
    "GridImage": "._core._grid_image",
    "ImagePipeline": "._core._image_pipeline",
}

#: Public subpackages. They resolve on first access, so ``import phenotypic`` -- and
#: every console script, which imports this package first -- loads none of them.
_LAZY_SUBPACKAGES: frozenset[str] = frozenset(
    {
        "abc_", "analysis", "correction", "data", "detect", "enhance", "grid", "measure",
        # ``plotting`` is not in ``__all__`` but is public in the docs
        # (``phenotypic.plotting.PlotDiagnostics``), so it resolves here too.
        "plotting",
        "prefab", "refine", "schema", "sdk_", "settings", "tune", "util",
    }
)


def __getattr__(name: str) -> _Any:
    """Import a public class or subpackage on first access and cache it on the package.

    Unknown names raise :class:`AttributeError`;
    ``SerializablePipeline._find_class_in_phenotypic`` relies on that to fall through
    to the subpackages when resolving an operation class by name.
    """
    if name in _LAZY_CLASSES:
        value = getattr(_importlib.import_module(_LAZY_CLASSES[name], __name__), name)
    elif name in _LAZY_SUBPACKAGES:
        value = _importlib.import_module(f".{name}", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if _TYPE_CHECKING:
    from . import (
        abc_,
        analysis,
        correction,
        data,
        detect,
        enhance,
        grid,
        measure,
        prefab,
        refine,
        schema,
        sdk_,
        settings,
        tune,
        util,
    )
    from ._core._grid_image import GridImage
    from ._core._image import Image
    from ._core._image_pipeline import ImagePipeline

__all__ = [
    "Image",  # Class imported from _core
    "GridImage",  # Class imported from _core
    "ImagePipeline",
    "abc_",
    "analysis",
    "data",
    "detect",
    "measure",
    "grid",
    "refine",
    "schema",
    "prefab",
    "correction",
    "enhance",
    "sdk_",
    "util",
    "settings",
    "tune",
]
```

- [ ] **Step 4: Run the tests again and record the cycles.** Same command as Step 2. Expected:
  - `test_import_phenotypic_loads_no_heavy_module` **passes**.
  - Sweep cases **fail** for `phenotypic.grid`, `phenotypic.measure`, `phenotypic.analysis`, `phenotypic.analysis._helper`, `phenotypic.analysis.abc_`, `phenotypic.analysis.edge`, `phenotypic.analysis.filter`, `phenotypic.analysis.qc` and `phenotypic.sdk_._qc_recipe` — nine in all. Each raises `ImportError … partially initialized module` at `_core/_image_parts/_grid_image_handler.py:16`, `:17` or `_core/_pipeline_parts/_image_pipeline_core.py:34`.
  - Every other case passes.
  - Paste the failure list into your report. If any other module fails, or any failure raises at a different site, apply the same rule (move the import to its point of use, or make the re-export lazy) and name every added site in your report.

- [ ] **Step 5: Rewrite the top of `src/phenotypic/abc_/__init__.py`.**
  - Keep the docstring (lines 1–7).
  - Insert this block directly after it, before `from phenotypic.schema import MeasurementInfo`:

```python
import importlib as _importlib
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any as _Any

#: Re-exports that pull in the image core: ``PrefabPipeline`` imports the pipeline, and
#: ``DetectionMode``/``register_detection_mode`` import ``phenotypic._core``, whose
#: ``__init__`` loads the whole image handler chain. Resolving them on first access keeps
#: every ``phenotypic.abc_.*`` import free of the core. It also breaks the cycle
#: ``analysis.abc_._model_fitter`` -> ``abc_.plotting`` -> this ``__init__`` -> pipeline
#: core -> ``_model_fitter``. Defined before the eager imports below, so an import that
#: re-enters this package mid-initialisation still resolves them.
_LAZY_ATTRS: dict[str, str] = {
    "PrefabPipeline": "phenotypic.abc_._prefab_pipeline",
    "DetectionMode": "phenotypic._core._image_parts.detection_modes",
    "register_detection_mode": "phenotypic._core._image_parts.detection_modes",
}


def __getattr__(name: str) -> _Any:
    """Resolve a core-dependent re-export on first access and cache it on the package."""
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if _TYPE_CHECKING:
    from phenotypic._core._image_parts.detection_modes import (
        DetectionMode,
        register_detection_mode,
    )

    from ._prefab_pipeline import PrefabPipeline

```

  - Append `  # noqa: E402` to the first line of every eager import statement that now follows this block. Ruff reports them as module-level imports not at the top of the file, and the lazy block must come first (Amendment A P8); without this the task's lint step fails with 20 findings here and 13 in `sdk_/__init__.py` (plan review I3).
  - Then delete `from ._prefab_pipeline import PrefabPipeline` (currently line 35).
  - Delete the four-line `from phenotypic._core._image_parts.detection_modes import (DetectionMode, register_detection_mode,)` block (currently lines 37–40).
  - Leave `__all__` unchanged.

- [ ] **Step 6: Edit `src/phenotypic/_core/_image_parts/_grid_image_handler.py`.**
  - Delete line 10 `import matplotlib.pyplot as plt`, line 16 `from phenotypic.grid import CenteredAutoGridFinder` and line 17 `from phenotypic.measure import MeasureBounds`.
  - Add `from phenotypic.grid import CenteredAutoGridFinder` as the first statement after the docstring of `ImageGridHandler.__init__`.
  - Add these as the first statements after the docstring of `ImageGridHandler._draw_section_boxes_on_overlay`:

    ```python
    import matplotlib.pyplot as plt

    from phenotypic.measure import MeasureBounds
    ```

- [ ] **Step 7: Run the tests.** Same command as Step 2. Expected: all pass.
- [ ] **Step 8: Make `src/phenotypic/sdk_/__init__.py` lazy for its five heavy submodules.**
  - Keep the docstring (lines 1–14).
  - Replace the block `from . import (colourspace, constants_, exceptions_, napari_, slurm, slurm_,)` (lines 16–23) with the code below.
  - Append `  # noqa: E402` to the first line of every eager import statement that follows the lazy block, for the same reason as `abc_/__init__.py`.
  - Then delete `from .hdf_ import HDF` (line 294) and the whole import blocks `from ._measurement_tables import (…)` (lines 309–322), `from ._metadata_migration import (…)` (lines 323–335) and `from .mixin import (…)` (lines 336–343).
  - Leave `__all__` unchanged.

```python
import importlib as _importlib
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any as _Any

#: Re-exports whose submodules import heavy third-party libraries -- colour-science
#: (``colourspace``), h5py (``hdf_``), pandas (``_measurement_tables``,
#: ``_metadata_migration``), scipy and scikit-image (``mixin``). Importing any
#: ``phenotypic.sdk_.*`` module runs this ``__init__`` first, so these resolve on first
#: access instead. Defined before the eager imports below, so an import that re-enters
#: this package mid-initialisation still resolves them.
_LAZY_ATTRS: dict[str, str] = {
    "colourspace": ".colourspace",
    "HDF": ".hdf_",
    "PreparedEmbeddedMeasurementTable": "._measurement_tables",
    "PreparedImageTables": "._measurement_tables",
    "build_measurement_table_descriptor": "._measurement_tables",
    "build_metadata_table_descriptor": "._measurement_tables",
    "embedded_measurement_columns": "._measurement_tables",
    "read_embedded_measurement_column": "._measurement_tables",
    "read_embedded_measurement_descriptor": "._measurement_tables",
    "replace_embedded_measurement_table": "._measurement_tables",
    "replace_image_tables": "._measurement_tables",
    "write_embedded_measurement_table": "._measurement_tables",
    "write_image_tables": "._measurement_tables",
    "write_metadata_table": "._measurement_tables",
    "MetadataMigrationAuthority": "._metadata_migration",
    "MetadataMigrationReport": "._metadata_migration",
    "MetadataMigrationResult": "._metadata_migration",
    "MetadataMigrationTarget": "._metadata_migration",
    "metadata_migration_authority": "._metadata_migration",
    "migrate_metadata_bundle": "._metadata_migration",
    "migrate_metadata_file": "._metadata_migration",
    "migrate_preflighted_metadata_bundle": "._metadata_migration",
    "preflight_metadata_schema": "._metadata_migration",
    "rollback_metadata_migration": "._metadata_migration",
    "validated_published_metadata_migration_targets": "._metadata_migration",
    "FootprintMixin": ".mixin",
    "GridInferenceMixin": ".mixin",
    "InputLayerMixin": ".mixin",
    "LazyWidgetMixin": ".mixin",
    "NormControlMixin": ".mixin",
    "NormalizedOutputMixin": ".mixin",
}


def __getattr__(name: str) -> _Any:
    """Resolve a heavy re-export on first access and cache it on the package."""
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _importlib.import_module(module_name, __name__)
    value = module if module_name == f".{name}" else getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if _TYPE_CHECKING:
    from . import colourspace
    from ._measurement_tables import (
        PreparedEmbeddedMeasurementTable,
        PreparedImageTables,
        build_measurement_table_descriptor,
        build_metadata_table_descriptor,
        embedded_measurement_columns,
        read_embedded_measurement_column,
        read_embedded_measurement_descriptor,
        replace_embedded_measurement_table,
        replace_image_tables,
        write_embedded_measurement_table,
        write_image_tables,
        write_metadata_table,
    )
    from ._metadata_migration import (
        MetadataMigrationAuthority,
        MetadataMigrationReport,
        MetadataMigrationResult,
        MetadataMigrationTarget,
        metadata_migration_authority,
        migrate_metadata_bundle,
        migrate_metadata_file,
        migrate_preflighted_metadata_bundle,
        preflight_metadata_schema,
        rollback_metadata_migration,
        validated_published_metadata_migration_targets,
    )
    from .hdf_ import HDF
    from .mixin import (
        FootprintMixin,
        GridInferenceMixin,
        InputLayerMixin,
        LazyWidgetMixin,
        NormControlMixin,
        NormalizedOutputMixin,
    )

from . import (
    constants_,
    exceptions_,
    napari_,
    slurm,
    slurm_,
)
```

- [ ] **Step 9: Check that every lazy name resolves.** Expected output: `31 lazy names resolve`.

  ```bash
  QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run python -c "import phenotypic.sdk_ as s; [getattr(s, n) for n in s._LAZY_ATTRS]; print(len(s._LAZY_ATTRS), 'lazy names resolve')"
  ```

- [ ] **Step 10: Test surface.** Expected: no failures.

  ```bash
  QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest tests/unit/ci tests/unit/abc_ tests/unit/sdk_ tests/unit/core tests/unit/tune/test_lazy_import_lock.py tests/unit/gui/test_optional_deps.py -q -o addopts= -m "not slow" -p no:cacheprovider -n 8 -rfE
  ```

  Any failure: rerun that test alone on this commit and at `BASE_PRE` before attributing it, and report both results.

- [ ] **Step 11: Lint and commit.**
  - Run `uv run ruff check` on the five changed files. Expect zero findings: the `# noqa: E402` markers cover the eager imports that now follow each lazy block. Compare against the same per-file runs at `/tmp/pht-lazy-base`, which are also zero.
  - Commit subject: `feat(startup): lazy package entry points, and fix the three cycles the eager order hid`.

---

### Task 3: Point-of-use imports in the image core

**Files:**
- Modify: the 14 files in the table below, plus `src/phenotypic/_startup_perf.py`.
- Create: `tests/unit/ci/test_deferred_imports.py`
- Modify: `tests/unit/ci/test_startup_imports.py`

**Interfaces:**
- **Consumes:** T1's `DEFERRED_RUNTIME_MODULES` and `run_startup_probe`; T2's lazy package inits.
- **Produces:** `phenotypic._startup_perf.configure_docs_build_plotly_renderer() -> bool`; `tests/unit/ci/test_deferred_imports.py::DEFERRED_SITES`.

Deferral table (paths relative to `src/phenotypic/`; apply the point-of-use rule from "Shared facts"):

| File | Remove | TYPE_CHECKING | Import inside |
|---|---|---|---|
| `_core/_image_parts/_image_io_handler.py` | L28 `import h5py` | — | `_load_hdf5_for_migration`: `import h5py` |
| `_core/_image_parts/accessor_abstracts/_image_accessor_base_parents/_accessor_mpl_handler.py` | L5 `import matplotlib.pyplot as plt`, L8 `from matplotlib.patches import Rectangle` | `plt` | `_add_section_boxes`: plt, Rectangle; `_mpl_plot`: plt; `histogram`: plt |
| `_core/_image_parts/accessors/_grid_accessor.py` | L10 `import matplotlib.pyplot as plt` | `plt` | `_build_section_box_shapes`, `show_column_overlay`, `show_row_overlay`: plt |
| `_core/_image_parts/accessors/_objmap_accessor.py` | L12 `import matplotlib.pyplot as plt` | `plt` | `show`: plt |
| `_core/_image_parts/accessors/_objmask_accessor.py` | L8 `import matplotlib.pyplot as plt` | `plt` | `show`: plt |
| `_core/_image_parts/color_space_accessors/_chromaticity_xy_accessor.py` | L1 `import colour` | — | `_subject_arr`: colour |
| `_core/_image_parts/color_space_accessors/_cielab_accessor.py` | L1 `import colour` | — | `_subject_arr`: colour |
| `_core/_image_parts/color_space_accessors/_hsv_accessor.py` | L9 `from matplotlib import pyplot as plt` | `plt` | `histogram`, `show`, `show_objects`: plt |
| `_core/_image_parts/color_space_accessors/_xyz_conversion.py` | L10 `import colour`, L13 `from phenotypic.sdk_.colourspace import sRGB_D50` | — | `rgb_to_xyz`: colour, sRGB_D50 |
| `_core/_image_parts/color_space_accessors/_xyz_d65_accessor.py` | L3 `import colour` | — | `_subject_arr`: colour |
| `_core/_image_parts/plot_accessor/_base_plotter.py` | L6 `import matplotlib.pyplot as plt` | `plt` | `_cleanup_figure`, `_create_colormap`, `_validate_cmap`: plt |
| `_core/_image_parts/plot_accessor/_detect_modes_plotter.py` | L14 `import plotly.graph_objects as go`, L15 `from plotly.subplots import make_subplots` | `go` | `detect_modes`: make_subplots |
| `_core/_image_parts/accessor_abstracts/_image_accessor_base_parents/_accessor_dash_handler.py` | see Step 4 | `go` (already present) | `_plotly_imshow`: `import plotly.express as px` |
| `_core/_image_parts/plot_accessor/_diagnostics_plotter.py` | L7 `import matplotlib.pyplot as plt`, L9 `import plotly.graph_objects as go`, L17 `from phenotypic.sdk_.viz.figures._theme import NAVY, OKABE_ITO` | `plt`, `go` | `plt`: `_diagnostics_matplotlib`, `_plot_background_estimate`, `_plot_gradient_magnitude`, `_plot_local_contrast_map`, `_plot_local_variance_map`, `_plot_noise_autocorrelation`, `_plot_orientation_coherence`; `go`: `_empty_plotly_figure`, `fig_background_estimate`, `fig_contrast_metrics`, `fig_detection_matrix`, `fig_gradient_magnitude`, `fig_intensity_histogram`, `fig_local_contrast_map`, `fig_local_variance`, `fig_noise_autocorrelation`, `fig_orientation_coherence`, `fig_power_spectral_density`, `fig_quality_summary`, `fig_ridge_response`; `NAVY`: `_empty_plotly_figure`, `fig_intensity_histogram`, `fig_power_spectral_density`, `fig_quality_summary`, `fig_ridge_response`; `OKABE_ITO`: `fig_intensity_histogram`, `fig_power_spectral_density`, `fig_ridge_response` |

`_diagnostics_plotter.py` is on the `Image` path even though no static import walker shows it: `plotting/_image_plots.py:52` applies `@_diagnostics_figure(...)`, whose body imports `DiagnosticsPlotter` while the class is being created (plan review C1). Leave its `GridSpec`, scipy, skimage and `phenotypic.util.image_metrics` imports alone — tier 4 watches `matplotlib.pyplot`, not matplotlib core, and polars is not a deferral target.

`_xyz_conversion.rgb_to_xyz` assigns `sRGB_D50.whitepoint`. A function-scope `from phenotypic.sdk_.colourspace import sRGB_D50` binds the same module-level object, so that mutation is unchanged.

- [ ] **Step 1: Write the site checker** `tests/unit/ci/test_deferred_imports.py`, including only the image-core entries for now:

```python
"""Heavy imports stay deferred to the functions that use them, site by site.

``test_startup_imports.py`` guards what each entry point loads. This module guards every
deferral site on its own, so a module-level import cannot creep back into a module no
entry-point guard reaches, and a moved import cannot go missing from a function no other
test calls.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "src" / "phenotypic"

#: Module (relative to ``src/phenotypic``) -> name bound by a deferred import -> the
#: functions that must import it locally. An empty tuple means the name may only appear
#: under ``TYPE_CHECKING`` (or nowhere) at module level.
DEFERRED_SITES: dict[str, dict[str, tuple[str, ...]]] = {
    "_core/_image_parts/_grid_image_handler.py": {
        "CenteredAutoGridFinder": ("__init__",),
        "MeasureBounds": ("_draw_section_boxes_on_overlay",),
        "plt": ("_draw_section_boxes_on_overlay",),
    },
    "_core/_image_parts/_image_io_handler.py": {"h5py": ("_load_hdf5_for_migration",)},
    "_core/_image_parts/accessor_abstracts/_image_accessor_base_parents/_accessor_dash_handler.py": {
        "px": ("_plotly_imshow",),
        "go": (),
        "_pio": (),
    },
    "_core/_image_parts/accessor_abstracts/_image_accessor_base_parents/_accessor_mpl_handler.py": {
        "plt": ("_add_section_boxes", "_mpl_plot", "histogram"),
        "Rectangle": ("_add_section_boxes",),
    },
    "_core/_image_parts/accessors/_grid_accessor.py": {
        "plt": ("_build_section_box_shapes", "show_column_overlay", "show_row_overlay"),
    },
    "_core/_image_parts/accessors/_objmap_accessor.py": {"plt": ("show",)},
    "_core/_image_parts/accessors/_objmask_accessor.py": {"plt": ("show",)},
    "_core/_image_parts/color_space_accessors/_chromaticity_xy_accessor.py": {"colour": ("_subject_arr",)},
    "_core/_image_parts/color_space_accessors/_cielab_accessor.py": {"colour": ("_subject_arr",)},
    "_core/_image_parts/color_space_accessors/_hsv_accessor.py": {"plt": ("histogram", "show", "show_objects")},
    "_core/_image_parts/color_space_accessors/_xyz_conversion.py": {
        "colour": ("rgb_to_xyz",),
        "sRGB_D50": ("rgb_to_xyz",),
    },
    "_core/_image_parts/color_space_accessors/_xyz_d65_accessor.py": {"colour": ("_subject_arr",)},
    "_core/_image_parts/plot_accessor/_base_plotter.py": {
        "plt": ("_cleanup_figure", "_create_colormap", "_validate_cmap"),
    },
    "_core/_image_parts/plot_accessor/_detect_modes_plotter.py": {
        "make_subplots": ("detect_modes",),
        "go": (),
    },
    "_core/_image_parts/plot_accessor/_diagnostics_plotter.py": {
        "plt": (
            "_diagnostics_matplotlib",
            "_plot_background_estimate",
            "_plot_gradient_magnitude",
            "_plot_local_contrast_map",
            "_plot_local_variance_map",
            "_plot_noise_autocorrelation",
            "_plot_orientation_coherence",
        ),
        "go": (
            "_empty_plotly_figure",
            "fig_background_estimate",
            "fig_contrast_metrics",
            "fig_detection_matrix",
            "fig_gradient_magnitude",
            "fig_intensity_histogram",
            "fig_local_contrast_map",
            "fig_local_variance",
            "fig_noise_autocorrelation",
            "fig_orientation_coherence",
            "fig_power_spectral_density",
            "fig_quality_summary",
            "fig_ridge_response",
        ),
        "NAVY": (
            "_empty_plotly_figure",
            "fig_intensity_histogram",
            "fig_power_spectral_density",
            "fig_quality_summary",
            "fig_ridge_response",
        ),
        "OKABE_ITO": ("fig_intensity_histogram", "fig_power_spectral_density", "fig_ridge_response"),
    },
}


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _module_level_bindings(body: list[ast.stmt]) -> dict[str, int]:
    """Names bound by runtime module-level imports (``TYPE_CHECKING`` blocks excluded) -> line."""
    bound: dict[str, int] = {}
    for node in body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound[alias.asname or alias.name.split(".")[0]] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound[alias.asname or alias.name] = node.lineno
        elif isinstance(node, ast.If):
            if not _is_type_checking(node.test):
                bound.update(_module_level_bindings(node.body))
            bound.update(_module_level_bindings(node.orelse))
        elif isinstance(node, ast.Try):
            bound.update(_module_level_bindings(node.body))
            for handler in node.handlers:
                bound.update(_module_level_bindings(handler.body))
            bound.update(_module_level_bindings(node.orelse))
            bound.update(_module_level_bindings(node.finalbody))
    return bound


def _locally_imported_names(function: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Import):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
    return names


def _parse(relative_path: str) -> ast.Module:
    return ast.parse((PACKAGE_ROOT / relative_path).read_text(encoding="utf-8"))


@pytest.mark.parametrize("relative_path", sorted(DEFERRED_SITES))
def test_deferred_names_are_not_imported_at_module_level(relative_path: str) -> None:
    bound = _module_level_bindings(_parse(relative_path).body)
    leaked = {name: bound[name] for name in DEFERRED_SITES[relative_path] if name in bound}
    assert leaked == {}, f"{relative_path}: runtime module-level import of deferred names (name: line) {leaked}"


@pytest.mark.parametrize("relative_path", sorted(DEFERRED_SITES))
def test_each_user_of_a_deferred_name_imports_it_locally(relative_path: str) -> None:
    tree = _parse(relative_path)
    missing = []
    for name, function_names in DEFERRED_SITES[relative_path].items():
        for function_name in function_names:
            candidates = [
                node
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
            ]
            assert candidates, f"{relative_path}: no function named {function_name!r}"
            if not any(name in _locally_imported_names(node) for node in candidates):
                missing.append(f"{function_name} does not import {name}")
    assert missing == [], f"{relative_path}: {missing}"
```

- [ ] **Step 2: Append tier 4 and the docs-renderer guard** to `tests/unit/ci/test_startup_imports.py`:

```python
def test_importing_image_loads_no_deferred_runtime_module() -> None:
    """Tier 4: an Image carries every accessor, and none of them may pay for a plotting or colour library."""
    report = run_startup_probe(
        "from phenotypic import Image\n"
        f"watched = {sorted(DEFERRED_RUNTIME_MODULES)!r}\n"
        "report = {'loaded': [m for m in watched if m in sys.modules],\n"
        "          'control': 'phenotypic._core._image' in sys.modules}\n"
    )
    assert report["control"] is True
    assert report["loaded"] == []


def test_docs_build_still_selects_the_notebook_connected_renderer() -> None:
    """Under PHENOTYPIC_DOCS_BUILD the renderer is still chosen at ``import phenotypic``."""
    report = run_startup_probe(
        "import os\n"
        "os.environ['PHENOTYPIC_DOCS_BUILD'] = '1'\n"
        "import phenotypic\n"
        "import plotly.io\n"
        "report = {'renderer': plotly.io.renderers.default}\n"
    )
    assert report["renderer"] == "notebook_connected"
```

- [ ] **Step 3: Run the new tests (RED).** Expected:
  - tier 4 fails, with colour, h5py, matplotlib.pyplot and plotly loaded;
  - the docs-renderer test passes, because today's dash handler still sets the renderer;
  - the checker fails for every file in the table and passes for `_grid_image_handler.py`, which T2 already moved.

  ```bash
  QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest tests/unit/ci/test_startup_imports.py tests/unit/ci/test_deferred_imports.py -q -o addopts= -p no:cacheprovider -n 8 -rfE
  ```

- [ ] **Step 4: Restructure the dash handler.**
  - In `_accessor_dash_handler.py`, replace the `try:` … `except ImportError:  # pragma: no cover` block and the following `if TYPE_CHECKING:` block (currently lines 14–35) with:

    ```python
    import importlib.util as _importlib_util

    #: Whether plotly is installed, checked without importing it: every ``Image`` carries
    #: these accessors, and plotly loads only when a figure is drawn. Patched to ``False``
    #: by ``tests/unit/core/test_plotly_fallback.py``.
    PLOTLY_AVAILABLE = _importlib_util.find_spec("plotly") is not None

    if TYPE_CHECKING:
        import plotly.graph_objects as go
    ```

  - Add `import plotly.express as px` as the first statement after the docstring of `AccessorDashHandler._plotly_imshow`.
  - In `src/phenotypic/_startup_perf.py`, add `import os` to the imports and `"configure_docs_build_plotly_renderer"` to `__all__`, then define this function above the module's final `install_lazy_colour_plotting()` call:

    ```python
    def configure_docs_build_plotly_renderer() -> bool:
        """Select Plotly's ``notebook_connected`` renderer when building the docs.

        nbsphinx captures cell outputs from the kernel's HTML mimetype, but Plotly's
        default ``plotly_mimetype+notebook`` renderer emits a JSON MIME bundle that
        nbsphinx drops; ``notebook_connected`` swaps that for an HTML+CDN-script bundle,
        so figures survive into the static site. This used to run as a side effect of
        the image accessors importing plotly at ``import phenotypic``. It still runs at
        ``import phenotypic``, but only under ``PHENOTYPIC_DOCS_BUILD``, so no other entry
        point imports plotly.

        Returns:
            ``True`` if the renderer was set.
        """
        if not os.environ.get("PHENOTYPIC_DOCS_BUILD"):
            return False
        import plotly.io as pio

        pio.renderers.default = "notebook_connected"
        return True
    ```

  - Change the module's final line from `install_lazy_colour_plotting()` to two lines:

    ```python
    install_lazy_colour_plotting()
    configure_docs_build_plotly_renderer()
    ```

- [ ] **Step 5: Apply the point-of-use rule** to the other 13 table rows.
- [ ] **Step 6: Run Step 3's command (GREEN).** Expected: all pass.
- [ ] **Step 7: Test surface.** Expected: no failures. Attribute any failure by rerunning it alone here and at `/tmp/pht-lazy-base`.

  ```bash
  QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest tests/unit/core tests/unit/ci tests/unit/plotting tests/unit/sdk_ -q -o addopts= -m "not slow" -p no:cacheprovider -n 8 -rfE
  ```

- [ ] **Step 8: Lint and commit.**
  - Run `uv run ruff check` on every changed file.
  - Commit subject: `perf(core): import colour, h5py, matplotlib and plotly where the image accessors use them`.

---

### Task 4: Point-of-use imports in the operation and analysis modules

**Files:**
- Modify: the 19 files in the table below.
- Modify: `tests/unit/ci/test_deferred_imports.py`

**Interfaces:**
- **Consumes:** T3's `DEFERRED_SITES` and checker.
- **Produces:** nothing new.

Deferral table (paths relative to `src/phenotypic/`; apply the point-of-use rule):

| File | Remove | TYPE_CHECKING | Import inside |
|---|---|---|---|
| `analysis/abc_/_model_fitter.py` | L8 `import matplotlib`, L9 `import matplotlib.pyplot as plt` | `plt` | `_build_plotly_figure`: matplotlib; `show`: matplotlib, plt |
| `analysis/edge/_edge_correction.py` | L8 `import matplotlib.pyplot as plt`, L9 `from matplotlib.figure import Figure` | `plt`, `Figure` | `_show_collapsed`, `_show_individual`: plt |
| `analysis/filter/_mad_outlier.py` | L8 `import matplotlib.pyplot as plt` | `plt` | `_show_collapsed`, `_show_individual`: plt |
| `analysis/filter/_tukey_outlier.py` | L8 `import matplotlib.pyplot as plt` | `plt` | `_show_collapsed`, `_show_individual`: plt |
| `analysis/qc/_expected_vs_detected.py` | L16 `import plotly.graph_objects as go` | `go` | `inspect`: go |
| `analysis/qc/_grid_occupancy.py` | L15 `import plotly.graph_objects as go` | `go` | `inspect`: go |
| `analysis/qc/_replicate_agreement.py` | L18 `import plotly.graph_objects as go` | `go` | `inspect`: go |
| `correction/_color_correction/_color_corrector.py` | L13 `import colour` | — | `_operate`: colour |
| `correction/_color_denoise.py` | L5 `import bm3d`, L16 `from ..sdk_.colourspace import decode_srgb, encode_srgb` | `bm3d` | `_build_profile`, `_denoise_gat`, `_denoise_plain`: bm3d; `_operate`: `from phenotypic.sdk_.colourspace import decode_srgb, encode_srgb` |
| `correction/_denoise_block_match.py` | L5 `import bm3d`, L7 `from bm3d.profiles import BM3DStages`, L16 `from ..sdk_.colourspace import decode_srgb, encode_srgb` | `BM3DStages` | `_denoise_channel`: bm3d and `from phenotypic.sdk_.colourspace import decode_srgb, encode_srgb`; `_convert_stage_arg`: BM3DStages |
| `enhance/_enhance_block_match.py` | L7 `import bm3d`, L8 `from bm3d.profiles import BM3DStages` | any of these names used in an annotation | `_denoise_detect_mat`: bm3d; `_convert_stage_arg`: BM3DStages |
| `enhance/_flatten_illumination.py` | L5 `import cv2` | — | `_filter`: cv2 |
| `enhance/_subtract_opening.py` | L5 `import cv2` | — | `_operate`: cv2 |
| `refine/_extract_colony_core.py` | L11 `import cv2` | — | `_build_ellipse_kernel`, `_extract_single_core`: cv2 |
| `measure/_measure_texture.py` | see Step 3 | — | `_compute_haralick`: `mh = _mahotas()` |
| `detect/_filamentous_fungi_detector.py` | the `from phenotypic.sdk_.reconnect import (…)` block (L32–41) | `ReconnectConfig` | `_operate`: build_reconnect_cost, compute_full_image_app2_gi_cost, filter_mask_by_overlap, markers_from_centroids, partition_by_grid_voronoi, reconnect_fragments_tiled, select_reconnect_fragments; `_reconnect_config`: ReconnectConfig |
| `detect/_two_k_filamentous_detector.py` | the `from phenotypic.sdk_.reconnect import (…)` block (L21–29) | `ReconnectConfig` | `_operate`: build_reconnect_cost, filter_mask_by_overlap, markers_from_centroids, partition_by_grid_voronoi, reconnect_fragments_tiled, select_reconnect_fragments; `_reconnect_config`: ReconnectConfig |
| `sdk_/orientation_fields/_plots.py` | L8–12 (`Axes`, `PathCollection`, `Colormap`, `Normalize`, `Circle`, `Quiver`) | `Axes`, `Colormap`, `Normalize`, `PathCollection`, `Quiver` | `plot_literal_crossing_map`: Circle, Normalize; `plot_literal_crossing_outward_profile`: Normalize; `plot_literal_crossing_population`: Normalize |
| `util/_robust_color_stats.py` | L10 `import colour` | — | `lab_to_srgb_hex`, `medoid_ciede2000`: colour |

- **The two detectors:** the numba kernel modules are **not** edited (D6). Deferring the `sdk_.reconnect` import at these two importers is what keeps numba out of `import phenotypic.detect`.
- **`_enhance_block_match.py`:** its TYPE_CHECKING cell is conditional. Put a name under TYPE_CHECKING only if ruff (F821) or mypy reports it undefined in an annotation after the move.

- [ ] **Step 1: Append the operation entries** to `DEFERRED_SITES`:

```python
    "analysis/abc_/_model_fitter.py": {"matplotlib": ("_build_plotly_figure", "show"), "plt": ("show",)},
    "analysis/edge/_edge_correction.py": {"plt": ("_show_collapsed", "_show_individual"), "Figure": ()},
    "analysis/filter/_mad_outlier.py": {"plt": ("_show_collapsed", "_show_individual")},
    "analysis/filter/_tukey_outlier.py": {"plt": ("_show_collapsed", "_show_individual")},
    "analysis/qc/_expected_vs_detected.py": {"go": ("inspect",)},
    "analysis/qc/_grid_occupancy.py": {"go": ("inspect",)},
    "analysis/qc/_replicate_agreement.py": {"go": ("inspect",)},
    "correction/_color_correction/_color_corrector.py": {"colour": ("_operate",)},
    "correction/_color_denoise.py": {
        "bm3d": ("_build_profile", "_denoise_gat", "_denoise_plain"),
        "decode_srgb": ("_operate",),
        "encode_srgb": ("_operate",),
    },
    "correction/_denoise_block_match.py": {
        "bm3d": ("_denoise_channel",),
        "BM3DStages": ("_convert_stage_arg",),
        "decode_srgb": ("_denoise_channel",),
        "encode_srgb": ("_denoise_channel",),
    },
    "enhance/_enhance_block_match.py": {"bm3d": ("_denoise_detect_mat",), "BM3DStages": ("_convert_stage_arg",)},
    "enhance/_flatten_illumination.py": {"cv2": ("_filter",)},
    "enhance/_subtract_opening.py": {"cv2": ("_operate",)},
    "refine/_extract_colony_core.py": {"cv2": ("_build_ellipse_kernel", "_extract_single_core")},
    "measure/_measure_texture.py": {"mahotas": ("_mahotas",), "mh": ()},
    "detect/_filamentous_fungi_detector.py": {
        "ReconnectConfig": ("_reconnect_config",),
        "build_reconnect_cost": ("_operate",),
        "compute_full_image_app2_gi_cost": ("_operate",),
        "filter_mask_by_overlap": ("_operate",),
        "markers_from_centroids": ("_operate",),
        "partition_by_grid_voronoi": ("_operate",),
        "reconnect_fragments_tiled": ("_operate",),
        "select_reconnect_fragments": ("_operate",),
    },
    "detect/_two_k_filamentous_detector.py": {
        "ReconnectConfig": ("_reconnect_config",),
        "build_reconnect_cost": ("_operate",),
        "filter_mask_by_overlap": ("_operate",),
        "markers_from_centroids": ("_operate",),
        "partition_by_grid_voronoi": ("_operate",),
        "reconnect_fragments_tiled": ("_operate",),
        "select_reconnect_fragments": ("_operate",),
    },
    "sdk_/orientation_fields/_plots.py": {
        "Circle": ("plot_literal_crossing_map",),
        "Normalize": (
            "plot_literal_crossing_map",
            "plot_literal_crossing_outward_profile",
            "plot_literal_crossing_population",
        ),
        "Axes": (),
        "Colormap": (),
        "PathCollection": (),
        "Quiver": (),
    },
    "util/_robust_color_stats.py": {"colour": ("lab_to_srgb_hex", "medoid_ciede2000")},
```

- [ ] **Step 2: Run the checker (RED).** Expected: the new entries fail; T3's entries pass.

  ```bash
  QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest tests/unit/ci/test_deferred_imports.py -q -o addopts= -p no:cacheprovider -n 8 -rfE
  ```

- [ ] **Step 3: Restructure mahotas.** In `measure/_measure_texture.py`:
  - Replace lines 19–22 (the comment, `warnings.filterwarnings(...)` and `import mahotas as mh  # noqa: E402 …`) with:

    ```python
    @functools.cache
    def _mahotas():
        """Import mahotas on first use, silencing its module-load ``SyntaxWarning``.

        Deferred so ``phenotypic.measure`` -- imported by every pipeline -- does not load
        mahotas until a texture is measured.
        """
        warnings.filterwarnings("ignore", category=SyntaxWarning, module="mahotas")
        import mahotas

        return mahotas
    ```

  - In `MeasureTexture._compute_haralick`, add `mh = _mahotas()` as the first statement after its docstring.
  - `functools` and `warnings` are already imported by the module.

- [ ] **Step 4: Apply the point-of-use rule** to the other 18 rows.
- [ ] **Step 5: Run Step 2's command (GREEN).** Expected: all pass.
- [ ] **Step 6: Test surface.** Expected: no failures beyond ones you have attributed as pre-existing by rerunning each alone here and at `/tmp/pht-lazy-base`.

  ```bash
  QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest tests/unit/ci tests/unit/analysis tests/unit/qc tests/unit/correction tests/unit/enhance tests/unit/refine tests/unit/measure tests/unit/detect tests/unit/sdk_ tests/unit/viz tests/unit/plotting -q -o addopts= -m "not slow" -p no:cacheprovider -n 8 -rfE
  ```

- [ ] **Step 7: Lint and commit.**
  - Run `uv run ruff check` on every changed file.
  - Commit subject: `perf(ops): import numba, cv2, bm3d, mahotas and plotting libraries where operations use them`.

---

### ◆ Phase 1 gate (orchestrator)

1. **Independent review.** Dispatch `xander-local:implementation-test-reviewer` (opus) over the T1–T4 commit range with the spec, this plan and the four task reviews. The report goes to `docs/superpowers/reports/2026-09-11-lazy-startup/phase1-review.md`.
2. **Full default lanes.** Run the Phase 1 command in "Test surface commands" once. Attribute every failure by rerunning it alone on HEAD and at `/tmp/pht-lazy-base`. The four schema-digest failures are known.
3. **mypy and ruff.** Run mypy on HEAD with a fresh cache (`--cache-dir /tmp/lazy-mypy-cache-phase1`) to `/tmp/lazy-mypy-phase1.txt`, then compare against the baseline: `uv run python docs/superpowers/logic_validation_scripts/2026-09-10-private-gui-module/compare_findings.py mypy /tmp/lazy-mypy-before.txt /tmp/lazy-mypy-phase1.txt`. Expected exit 0. Do the same for ruff.
4. **Fixes.** One fix dispatch for the review's findings and any regression, then a scoped re-review. Rule on the rest in the ledger.

---

### Task 5: CLI help path and fail-fast preload

**Files:**
- Modify: `src/phenotypic/phenotypicCLI.py`
- Modify: `src/phenotypic/_cli/_cli_process_single.py`, `src/phenotypic/_cli/_cli_staged_slurm_worker.py`, `src/phenotypic/_cli/_cli_recompile_worker.py`, `src/phenotypic/_cli/_cli_checkpoint_handler.py`
- Modify: `tests/unit/ci/test_startup_imports.py`
- Create: `tests/unit/cli/test_cli_runtime_preload.py`

**Interfaces:**
- **Consumes:** T1's `load_runtime_dependencies`, `HEAVY_STARTUP_MODULES` and `run_startup_probe`.
- **Produces:** `phenotypic.phenotypicCLI._CLI_RUNTIME_IMPORTS`, `_load_cli_runtime()` and module `__getattr__`.

- [ ] **Step 1: Append the tier-2 CLI guard and the choices test** to `tests/unit/ci/test_startup_imports.py`:

```python
def test_cli_help_loads_no_heavy_module() -> None:
    """Tier 2: ``python -m phenotypic --help`` prints help without the library behind it."""
    report = run_startup_probe(
        "import contextlib, io, runpy\n"
        "sys.argv = ['phenotypic', '--help']\n"
        "buffer = io.StringIO()\n"
        "code = None\n"
        "with contextlib.redirect_stdout(buffer):\n"
        "    try:\n"
        "        runpy.run_module('phenotypic', run_name='__main__')\n"
        "    except SystemExit as exc:\n"
        "        code = exc.code\n"
        f"watched = {sorted(HEAVY_STARTUP_MODULES)!r}\n"
        "report = {'exit': code, 'help': buffer.getvalue(),\n"
        "          'loaded': [m for m in watched if m in sys.modules],\n"
        "          'control': 'click' in sys.modules}\n"
    )
    assert report["exit"] in (0, None)
    assert "Usage:" in report["help"]
    assert "--detect-mode" in report["help"]
    assert report["control"] is True
    assert report["loaded"] == []


def test_detect_mode_choices_match_the_detection_mode_registry() -> None:
    """The CLI's light choice list and the registry name the same modes, in the order help shows."""
    from typing import get_args

    from phenotypic._core._image_parts.detection_modes import available_modes
    from phenotypic.phenotypicCLI import phenotypic_cli
    from phenotypic.sdk_.typing_ import DetectMode

    # Both sides are read in-process: no test registers a custom detection mode today
    # (a grep for ``register_detection_mode`` in tests/ is empty), so the global registry
    # is stable here.
    option = next(param for param in phenotypic_cli.params if param.name == "detect_mode")
    assert list(option.type.choices) == sorted(available_modes())
    assert set(get_args(DetectMode)) == set(available_modes())
```

- [ ] **Step 2: Write `tests/unit/cli/test_cli_runtime_preload.py`:**

```python
"""Pipeline-running entry points import the deferred libraries before any image work."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "src" / "phenotypic"

PIPELINE_WORKER_ENTRY_MODULES = (
    "_cli/_cli_process_single.py",
    "_cli/_cli_staged_slurm_worker.py",
    "_cli/_cli_recompile_worker.py",
    "_cli/_cli_checkpoint_handler.py",
)


def test_cli_aborts_before_any_output_when_a_runtime_dependency_is_broken(tmp_path: Path, monkeypatch) -> None:
    import phenotypic.phenotypicCLI as cli

    calls: list[str] = []

    def broken_install() -> None:
        calls.append("preload")
        raise ImportError("simulated broken numba install")

    monkeypatch.setattr(cli, "load_runtime_dependencies", broken_install)
    output_dir = tmp_path / "out"
    result = CliRunner().invoke(cli.phenotypic_cli, ["--input", str(tmp_path), "--output", str(output_dir)])

    assert calls == ["preload"]
    assert result.exit_code != 0
    assert isinstance(result.exception, ImportError)
    assert not output_dir.exists()


@pytest.mark.parametrize("relative_path", PIPELINE_WORKER_ENTRY_MODULES)
def test_pipeline_worker_entry_preloads_runtime_dependencies(relative_path: str) -> None:
    tree = ast.parse((PACKAGE_ROOT / relative_path).read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    assert any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "load_runtime_dependencies"
        for node in ast.walk(main)
    ), f"{relative_path}: main() never calls load_runtime_dependencies()"


def test_a_patched_deferred_cli_name_stays_patched_through_the_loader() -> None:
    import phenotypic.phenotypicCLI as cli
    from phenotypic._cli._cli_execution_strategies import create_execution_strategy

    with mock.patch("phenotypic.phenotypicCLI.create_execution_strategy") as patched:
        cli._load_cli_runtime()
        assert cli.create_execution_strategy is patched
    assert cli.create_execution_strategy is create_execution_strategy
```

- [ ] **Step 3: Run the new tests (RED).** Expected failures:
  - tier 2 (heavy modules loaded);
  - the abort test (it calls no preload);
  - four worker AST cases;
  - the patch test (`_load_cli_runtime` missing).

  The choices test passes.

  ```bash
  QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest tests/unit/ci/test_startup_imports.py tests/unit/cli/test_cli_runtime_preload.py -q -o addopts= -p no:cacheprovider -n 8 -rfE -k "cli or detect_mode or preload or patched"
  ```

- [ ] **Step 4: Edit `phenotypicCLI.py`.**
  1. Change `from typing import Any, List, Optional, Sequence, cast` to `from typing import TYPE_CHECKING, Any, List, Optional, Sequence, cast, get_args`, and add `import importlib` beside `import json`.
  2. Delete these module-level import statements:
     - line 157 `from phenotypic import ImagePipeline`;
     - line 158 `…detection_modes import available_modes`;
     - lines 167–170 `_cli_execution_strategies`;
     - line 176 `_core._provenance`;
     - line 182 `_cli_output_manager`;
     - lines 184–192 `_cli_state_management`;
     - lines 193–198 `_cli_staged_resume`;
     - line 208 `_cli_process_only`;
     - lines 209–216 `_cli_recompile_slurm_scripts`;
     - lines 177–180 `_cli_interactive` and lines 219–222 `_cli_validation` — both look light, but `_cli_validation.py:17` does `from phenotypic import ImagePipeline`, which loads the image core through the lazy package (plan review C2).
  3. Add `DetectMode` to the `from phenotypic.sdk_.typing_ import (…)` block (line 252). Add `from phenotypic._startup_perf import load_runtime_dependencies` directly after it.
  4. Insert this block directly before `# Set up logger`:

```python
if TYPE_CHECKING:
    from phenotypic._cli._cli_execution_strategies import (
        create_execution_strategy,
        uses_staged_gpu_strategy,
    )
    from phenotypic._cli._cli_interactive import execute_dry_run, get_sample_datasets
    from phenotypic._cli._cli_output_manager import OutputManager
    from phenotypic._cli._cli_process_only import resolve_process_format
    from phenotypic._cli._cli_recompile_slurm_scripts import (
        TASK_FINALIZE,
        TASK_MEASUREMENTS,
        build_recompile_tasks,
        generate_recompile_slurm_scripts,
        recompile_attempt_dir,
        recompile_task_status_path,
    )
    from phenotypic._cli._cli_staged_resume import (
        build_staged_resume_plan,
        migrate_legacy_stage3_markers,
        pipeline_content_digest,
        reconcile_stage3_publications,
    )
    from phenotypic._cli._cli_state_management import (
        create_initial_state,
        exclude_terminal_failures_for_datasets,
        get_remaining_images_for_datasets,
        load_processing_state,
        save_processing_state,
        update_state_from_events,
        validate_resume_compatibility,
    )
    from phenotypic._cli._cli_validation import validate_execution_config, validate_pipeline
    from phenotypic._core._image_parts.detection_modes import available_modes
    from phenotypic._core._image_pipeline import ImagePipeline
    from phenotypic._core._provenance import pipeline_source_identity

#: Heavy names this module binds on first use. Each of these import statements reaches
#: the image core, pandas or polars, so importing them at module level would make
#: ``phenotypic --help`` pay for the whole library. They are bound into module globals
#: rather than imported locally so that ``mock.patch("phenotypic.phenotypicCLI.<name>")``
#: -- 27 sites in the test suite -- keeps patching what the command body calls.
_CLI_RUNTIME_IMPORTS: dict[str, tuple[str, ...]] = {
    "phenotypic._core._image_pipeline": ("ImagePipeline",),
    "phenotypic._core._image_parts.detection_modes": ("available_modes",),
    "phenotypic._cli._cli_execution_strategies": ("create_execution_strategy", "uses_staged_gpu_strategy"),
    "phenotypic._core._provenance": ("pipeline_source_identity",),
    "phenotypic._cli._cli_output_manager": ("OutputManager",),
    "phenotypic._cli._cli_state_management": (
        "create_initial_state",
        "exclude_terminal_failures_for_datasets",
        "get_remaining_images_for_datasets",
        "load_processing_state",
        "save_processing_state",
        "update_state_from_events",
        "validate_resume_compatibility",
    ),
    "phenotypic._cli._cli_staged_resume": (
        "build_staged_resume_plan",
        "migrate_legacy_stage3_markers",
        "pipeline_content_digest",
        "reconcile_stage3_publications",
    ),
    "phenotypic._cli._cli_process_only": ("resolve_process_format",),
    "phenotypic._cli._cli_recompile_slurm_scripts": (
        "TASK_FINALIZE",
        "TASK_MEASUREMENTS",
        "build_recompile_tasks",
        "generate_recompile_slurm_scripts",
        "recompile_attempt_dir",
        "recompile_task_status_path",
    ),
    "phenotypic._cli._cli_interactive": ("execute_dry_run", "get_sample_datasets"),
    "phenotypic._cli._cli_validation": ("validate_execution_config", "validate_pipeline"),
}

_CLI_RUNTIME_MODULE_BY_NAME: dict[str, str] = {
    name: module_name for module_name, names in _CLI_RUNTIME_IMPORTS.items() for name in names
}


def _load_cli_runtime() -> None:
    """Bind the heavy runtime names into this module's globals.

    ``setdefault`` keeps any value already bound, so an active
    ``mock.patch("phenotypic.phenotypicCLI.<name>")`` stays in force. Cheap after the
    first call: a module whose names are all bound is skipped.
    """
    module_globals = globals()
    for module_name, names in _CLI_RUNTIME_IMPORTS.items():
        if all(name in module_globals for name in names):
            continue
        module = importlib.import_module(module_name)
        for name in names:
            module_globals.setdefault(name, getattr(module, name))


def __getattr__(name: str) -> Any:
    """Serve a heavy runtime name to code outside this module (imports, ``mock.patch``)."""
    if name in _CLI_RUNTIME_MODULE_BY_NAME:
        _load_cli_runtime()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


```

  5. In the `--detect-mode` option, change `type=click.Choice(list(available_modes())),` to `type=click.Choice(sorted(get_args(DetectMode))),`.
  6. In `phenotypic_cli`, add these as the first two statements after its docstring:

     ```python
     load_runtime_dependencies()
     _load_cli_runtime()
     ```

     It is the **first** statement, before mode and option validation, so a broken install
     fails before any parse-dependent work. The cost is that every mode pays the import,
     including `migrate`, `recompile` and a usage error. That is deliberate: a run that
     reaches an image has already paid it, and the abort test pins this placement.

  7. Add `_load_cli_runtime()` as the first statement after the docstring in `_migrate_legacy_success_evidence`, `_regenerate_missing_overlays` and `_handle_recompile_slurm`.
  8. Check that no other top-level function uses a `_CLI_RUNTIME_IMPORTS` name. Expected output: `ok`.

     ```bash
     uv run --no-project python -c "import ast; from pathlib import Path; t=ast.parse(Path('src/phenotypic/phenotypicCLI.py').read_text(encoding='utf-8')); heavy={'ImagePipeline','available_modes','create_execution_strategy','uses_staged_gpu_strategy','pipeline_source_identity','OutputManager','create_initial_state','exclude_terminal_failures_for_datasets','get_remaining_images_for_datasets','load_processing_state','save_processing_state','update_state_from_events','validate_resume_compatibility','build_staged_resume_plan','migrate_legacy_stage3_markers','pipeline_content_digest','reconcile_stage3_publications','resolve_process_format','TASK_FINALIZE','TASK_MEASUREMENTS','build_recompile_tasks','generate_recompile_slurm_scripts','recompile_attempt_dir','recompile_task_status_path','execute_dry_run','get_sample_datasets','validate_execution_config','validate_pipeline'}; ok={'phenotypic_cli','_migrate_legacy_success_evidence','_regenerate_missing_overlays','_handle_recompile_slurm','_load_cli_runtime','__getattr__'}; bad=[d.name for d in t.body if isinstance(d,(ast.FunctionDef,ast.ClassDef)) and d.name not in ok and any(isinstance(x,ast.Name) and x.id in heavy for x in ast.walk(d))]; print(bad or 'ok')"
     ```

- [ ] **Step 5: Add the worker preloads.** In each file below, add `from phenotypic._startup_perf import load_runtime_dependencies` to the module-level imports and insert `load_runtime_dependencies()`:
  - `_cli/_cli_process_single.py`: first statement of `main`, before `attempt_id = attempt_id or uuid4().hex`.
  - `_cli/_cli_staged_slurm_worker.py`: directly after `args = parser.parse_args(argv)`.
  - `_cli/_cli_recompile_worker.py`: first statement of `main`, before `try:`.
  - `_cli/_cli_checkpoint_handler.py`: first statement of `main`, before the `checkpoint: CheckpointType = (` assignment.

- [ ] **Step 6: Run Step 3's command (GREEN).** Expected: all pass.
- [ ] **Step 7: Test surface.** Expected: no failures beyond attributed pre-existing ones.

  ```bash
  QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest tests/unit/cli tests/integration/cli tests/unit/ci tests/integration/gui/test_console_script.py -q -o addopts= -m "not slow" -p no:cacheprovider -n 8 -rfE
  ```

- [ ] **Step 8: Lint and commit.**
  - Run `uv run ruff check` on the six changed source files and the two test files.
  - Commit subject: `perf(cli): phenotypic --help imports nothing heavy; pipeline runs preload what they defer`.

---

### Task 6: GUI launcher, lazy shell package, builder on first request

**Files:**
- Modify: `src/phenotypic/_gui/shell/__init__.py`, `src/phenotypic/_gui/shell/_launcher.py`, `src/phenotypic/_gui/shell/_app.py`
- Modify: `src/phenotypic/_gui/FEATURES.md` (line 583), `src/phenotypic/_gui/CLAUDE.md`
- Modify: `tests/unit/ci/test_startup_imports.py`
- Create: `tests/unit/gui/shell/test_hub_startup_imports.py`

**Interfaces:**
- **Consumes:** `ToolSession` and `_SessionProxy` (`_gui/shell/_session.py`, `_gui/shell/_app.py:103-133`); T1's probe helper.
- **Produces:** `phenotypic._gui.shell.__getattr__`; the builder mounted through `_SessionProxy`.

- [ ] **Step 1: Append the tier-2 GUI guard** to `tests/unit/ci/test_startup_imports.py`:

```python
def test_gui_help_loads_no_heavy_module() -> None:
    """Tier 2: ``phenotypic-gui --help`` prints help without Dash or the library."""
    report = run_startup_probe(
        "import contextlib, io\n"
        "from phenotypic._gui.shell._launcher import main\n"
        "buffer = io.StringIO()\n"
        "code = None\n"
        "with contextlib.redirect_stdout(buffer):\n"
        "    try:\n"
        "        main(['--help'])\n"
        "    except SystemExit as exc:\n"
        "        code = exc.code\n"
        f"watched = {sorted(HEAVY_STARTUP_MODULES)!r}\n"
        "report = {'exit': code, 'help': buffer.getvalue(),\n"
        "          'loaded': [m for m in watched if m in sys.modules],\n"
        "          'control': 'argparse' in sys.modules}\n"
    )
    assert report["exit"] == 0
    assert "phenotypic-gui" in report["help"]
    assert "--url-prefix" in report["help"]
    assert report["control"] is True
    assert report["loaded"] == []
```

- [ ] **Step 2: Write `tests/unit/gui/shell/test_hub_startup_imports.py`:**

```python
"""Tier 3: the composed hub serves before the builder or any heavy library is loaded."""

from __future__ import annotations

from pathlib import Path

from tests._startup_probe import run_startup_probe

#: What the composed hub loads before any request, with the chain that loads it. Measured
#: during the plan review; a shell-side module-level chain may be listed here with its
#: justification (spec, tier 3). None of these is a deferral target.
HUB_ALLOWED_BEFORE_FIRST_REQUEST: dict[str, str] = {
    "plotly": "dash -> plotly, dash's own import; third-party, cannot be cut",
    "pandas": "_gui/analysis/_callbacks.py:25, via compose_hub's eager analysis import",
    "polars": "_gui/analysis/_callbacks.py:26 and _gui/run_console/_request_safety.py:15",
    "pyarrow": "the pandas/polars parquet stack",
    "scipy": "_gui/_operation_registry.py:18 `from phenotypic import ImagePipeline` -> the image core",
    "skimage": "the same chain as scipy",
    "matplotlib": "matplotlib core, not pyplot; the same chain as scipy",
}

#: The spec's minimum: none of these may load before the first request.
HUB_WATCHED_MODULES: tuple[str, ...] = (
    "bm3d", "colour", "cv2", "h5py", "mahotas", "matplotlib.pyplot", "numba",
)


def test_composed_hub_builds_the_builder_on_its_first_request(tmp_path: Path) -> None:
    report = run_startup_probe(
        "from phenotypic._gui.shell._app import create_app\n"
        "from phenotypic._gui.shell._sandbox import SandboxRoot\n"
        f"sandbox = SandboxRoot.from_path({str(tmp_path)!r})\n"
        "app = create_app(sandbox, start_idle_thread=False, start_slurm_observer=False)\n"
        f"watched = {list(HUB_WATCHED_MODULES)!r}\n"
        "loaded_before = [m for m in watched if m in sys.modules]\n"
        "detect_before = 'phenotypic.detect' in sys.modules\n"
        "response = app.server.test_client().get('/builder/')\n"
        "report = {'dash': 'dash' in sys.modules, 'loaded_before': loaded_before,\n"
        "          'detect_before': detect_before, 'status': response.status_code,\n"
        "          'detect_after': 'phenotypic.detect' in sys.modules}\n"
    )
    assert report["dash"] is True
    assert sorted(set(report["loaded_before"]) - set(HUB_ALLOWED_BEFORE_FIRST_REQUEST)) == []
    assert report["detect_before"] is False
    assert report["status"] == 200
    assert report["detect_after"] is True
```

- [ ] **Step 3: Run the new tests (RED).** Expected: both fail. The GUI help test loads dash and heavy libraries; the hub test has `detect_before` True and heavy modules loaded.

  ```bash
  QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest tests/unit/ci/test_startup_imports.py::test_gui_help_loads_no_heavy_module tests/unit/gui/shell/test_hub_startup_imports.py -q -o addopts= -p no:cacheprovider -n 2 -rfE
  ```

- [ ] **Step 4: Replace `src/phenotypic/_gui/shell/__init__.py`.** Keep its docstring (lines 1–14). Replace lines 15–28 with:

```python
from __future__ import annotations

import importlib as _importlib
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any as _Any

#: Public shell names by defining module. The console script imports this package before
#: ``_launcher``, so an eager ``_app`` import here would load Dash -- and through it the
#: sub-apps -- just to print ``phenotypic-gui --help``.
_LAZY_ATTRS: dict[str, str] = {
    "SandboxRoot": "phenotypic._gui.shell._sandbox",
    "ToolSession": "phenotypic._gui.shell._session",
    "create_app": "phenotypic._gui.shell._app",
    "launch_gui": "phenotypic._gui.shell._launcher",
    "main": "phenotypic._gui.shell._launcher",
}


def __getattr__(name: str) -> _Any:
    """Resolve a public shell name on first access and cache it on the package."""
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if _TYPE_CHECKING:
    from phenotypic._gui.shell._app import create_app
    from phenotypic._gui.shell._launcher import launch_gui, main
    from phenotypic._gui.shell._sandbox import SandboxRoot
    from phenotypic._gui.shell._session import ToolSession

__all__ = [
    "SandboxRoot",
    "ToolSession",
    "create_app",
    "launch_gui",
    "main",
]
```

- [ ] **Step 5: Edit `_launcher.py`.**
  1. Delete line 39, `from phenotypic._gui.shell._app import create_app`.
  2. Replace the `_STARTUP_STEPS` comment and value (lines 47–49) with:

     ```python
     #: Number of bar segments the launcher advances: sandbox resolution, hub composition.
     #: The library is no longer imported before ``main()`` (it loads inside composition),
     #: so there is no separate core-load segment. See :class:`StartupReporter`.
     _STARTUP_STEPS = 2
     ```

  3. Delete the `_core_import_elapsed` function (lines 52–68).
  4. Replace the body of `launch_gui` from `if reporter is None:` through the `# ``sandbox``/``app`` are always bound here …` comment with:

```python
    if reporter is None:
        sandbox = SandboxRoot.from_path(root)
        from phenotypic._gui.shell._app import create_app

        app = create_app(sandbox, url_prefix=url_prefix)
    else:
        with reporter:
            with reporter.stage("Resolving sandbox root"):
                sandbox = SandboxRoot.from_path(root)
            with reporter.stage("Composing GUI hub"):
                # Imported here rather than at module level, so ``phenotypic-gui --help``
                # loads neither Dash nor the library; the import is part of what this
                # stage reports.
                from phenotypic._gui.shell._app import create_app

                app = create_app(
                    sandbox,
                    url_prefix=url_prefix,
                    progress=reporter.detail,
                )
        # ``sandbox``/``app`` are always bound here: the ``with`` bodies run
        # unless ``reporter`` raises, in which case we never reach this line.
```

  5. In `main`, delete the argument line `import_elapsed=_core_import_elapsed(),`.

- [ ] **Step 6: Mount the builder lazily in `_app.py`.**
  - Replace the builder block (currently lines 540–550, from `_tick("builder")` through the `wrap_in_chrome(builder_app, …)` call) with:

```python
    # 3. Builder Dash (lazy). ``builder.create_app`` discovers every operation, which
    #    imports the whole operation library, so it is built on the first /builder/
    #    request instead of at startup. Never released: unlike the viewer it holds no
    #    heavy per-output state, and a rebuild would only repeat that import cost.
    def _build_builder() -> dash.Dash:
        builder_app = builder.create_app(
            image_root=sandbox.root,
            url_prefix=join_url_prefix(base_url_prefix, MOUNT_BUILDER),
        )
        wrap_in_chrome(
            builder_app,
            active_tab=SHELL_TAB_BUILDER,
            sandbox=sandbox,
            url_prefix=base_url_prefix,
        )
        return builder_app

    builder_session: ToolSession[dash.Dash] = ToolSession("builder", build=_build_builder)
```

  - In the `DispatcherMiddleware` mount dict, change `MOUNT_BUILDER.rstrip("/"): builder_app.server,` to `MOUNT_BUILDER.rstrip("/"): _SessionProxy(builder_session),`.
  - Do **not** add `builder_session` to the `start_idle_release_thread([...])` list.
  - `get_registry()` (`_gui/_operation_registry.py:814-824`) is an unlocked singleton. After this change a first `/builder/` request and a first `/analysis/` request can call `discover()` concurrently; the duplicate work is benign (last writer wins) and needs no lock, but say so in your report if you see it.
  - Confirm no other `builder_app` reference remains. Expected: no output.

    ```bash
    git grep -n "builder_app" -- src/phenotypic/_gui/shell/_app.py
    ```

- [ ] **Step 7: Run Step 3's command (GREEN).**
  - Expected: both pass.
  - If the hub test fails only because `loaded_before` is non-empty, trace each listed module:
    1. Run `QT_QPA_PLATFORM=offscreen uv run python -X importtime -c "<the probe's first four lines>" 2> /tmp/lazy-hub-importtime.txt`.
    2. Walk the nesting up to the first `phenotypic` module.
  - If a `phenotypic` module imports it at module level, defer it at that module with the point-of-use rule, and add a `DEFERRED_SITES` entry.
  - A shell-side module-level chain may instead be recorded in `HUB_ALLOWED_BEFORE_FIRST_REQUEST` as `"module": "<chain> — <why it stays>"`, which is the spec's rule. Deferring pandas, polars, scipy or skimage across `_gui/analysis`, the operation registry, the results viewer or the run console is out of scope (spec Non-goals and D5).
  - Report every addition.

- [ ] **Step 8: Update the docs for this task.**
  1. In `src/phenotypic/_gui/FEATURES.md` line 583 (the "Staged startup feedback" row), replace the text `core-library load (measured against `phenotypic._IMPORT_STARTED_AT`), sandbox resolution, and hub composition (each of the six sub-apps ticks the bar via `compose_hub(..., progress=...)`)` with `sandbox resolution and hub composition (the hub's imports run inside that stage, and each sub-app built at startup ticks the bar via `compose_hub(..., progress=...)`; the builder is built on its first request)`. Keep the rest of the row byte-identical.
  2. In `src/phenotypic/_gui/CLAUDE.md`, add this bullet as the first item under `## Common gotchas`:

     ```markdown
     - **The builder is built on its first request.** `compose_hub` mounts it through
       `_SessionProxy` over a never-released `ToolSession`, because `builder.create_app`
       discovers every operation, which imports the whole operation library. Do not reach
       for a builder app object at composition time, and do not import `_app` at module level
       from `phenotypic._gui.shell` or its launcher — `tests/unit/gui/shell/test_hub_startup_imports.py`
       and `tests/unit/ci/test_startup_imports.py` fail if either creeps back.
     ```

- [ ] **Step 9: Test surface.**
  - Expected: no failures beyond attributed pre-existing ones.
  - Two tests resolve the builder mount as a Flask app and must go through the session instead: `tests/integration/gui/test_smoke_shell.py::test_dispatcher_threads_script_root` (line 222) and `::test_explicit_url_prefix_preserves_script_root_through_dispatcher` (line 403). Change `builder_flask = dispatcher.mounts["/builder"]` to `builder_flask = dispatcher.mounts["/builder"]._session.get().server`; registering their probe blueprint still works, because `ToolSession.get()` builds the app at that moment (plan review C4).
  - A test that pins the old three-stage launcher sequence ("Core library loaded" recorded by the launcher, or `_STARTUP_STEPS == 3`) is updated to the two-stage sequence and named in your report. `StartupReporter`'s own tests are unchanged.

  ```bash
  QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest tests/unit/gui tests/integration/gui tests/gui tests/unit/ci -q -o addopts= -m "not slow" -p no:cacheprovider -n 8 -rfE
  ```

- [ ] **Step 10: Builder e2e.**
  - Expected: failures only among the baseline's recorded failures (`baseline.md`).
  - Attribute any new failure by rerunning it alone here and at `/tmp/pht-lazy-base`.

  ```bash
  PLAYWRIGHT=1 QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest tests/e2e/gui/builder -q -o addopts= -p no:cacheprovider -n 4 -rfE
  ```

- [ ] **Step 11: FEATURES gate, lint and commit.**
  - `uv run python scripts/check_features_md.py` must exit 0.
  - Run `uv run ruff check` on the three changed source files and the two test files.
  - Commit subject: `perf(gui): phenotypic-gui --help imports nothing heavy; the builder is built on its first request`.

---

### ◆ Phase 2 gate (orchestrator)

1. **Independent review.** Dispatch `xander-local:implementation-test-reviewer` (opus) over the T5–T6 commit range with the spec, this plan and both task reviews. The report goes to `docs/superpowers/reports/2026-09-11-lazy-startup/phase2-review.md`.
2. **Surface run.** Run `tests/unit/cli tests/integration/cli tests/unit/gui tests/integration/gui tests/gui tests/unit/ci` once with the Phase 1 flags, plus the builder e2e suite and the six ci_flaky helper modules:
   - `tests/e2e/gui/test_deliverables_standalone_e2e.py`
   - `tests/e2e/gui/test_filter_offcanvas.py`
   - `tests/e2e/gui/test_heatmap_tab.py`
   - `tests/e2e/gui/test_qc_review_splitter.py`
   - `tests/e2e/gui/test_qc_tab.py`
   - `tests/e2e/gui/test_radial_triage.py`

   Use `PLAYWRIGHT=1 … -n 4`.
3. **Fixes.** One fix dispatch for the review's findings and any regression, then a scoped re-review.

---

### Task 7: Documentation, mutation proofs, measurements

**Files:**
- Modify: `CLAUDE.md`, `src/phenotypic/sdk_/CLAUDE.md`
- Create: `docs/superpowers/reports/2026-09-11-lazy-startup/mutation-proofs.md`
- Create: `docs/superpowers/reports/2026-09-11-lazy-startup/startup-after.json`
- Create: `docs/superpowers/reports/2026-09-11-lazy-startup/startup-measurements.md`

- [ ] **Step 1: Root `CLAUDE.md`.** Add this bullet as the first item under `## Gotchas`:

```markdown
- **Entry points are lazy — keep them that way.** `import phenotypic`, `phenotypic --help`,
  `phenotypic-gui --help` and the composed GUI hub load none of
  `phenotypic._startup_perf.HEAVY_STARTUP_MODULES`, and `from phenotypic import Image` loads
  none of `DEFERRED_RUNTIME_MODULES`. `phenotypic`, `phenotypic.sdk_`, `phenotypic.abc_` and
  `phenotypic._gui.shell` resolve some re-exports through a module `__getattr__`; import
  colour, numba, h5py, mahotas, cv2, bm3d, plotly or `matplotlib.pyplot` inside the function
  that uses it; and keep every package importable as the first import of a fresh process.
  Pipeline-running entry points call `load_runtime_dependencies()`, so a broken install still
  fails before the first image. Guards: `tests/unit/ci/test_startup_imports.py`,
  `tests/unit/ci/test_deferred_imports.py`, `tests/unit/gui/shell/test_hub_startup_imports.py`.
```

- [ ] **Step 2: `src/phenotypic/sdk_/CLAUDE.md`.** Insert this section directly before `## Other Utilities`:

```markdown
## Lazy re-exports

`sdk_/__init__.py` re-exports `colourspace`, `HDF`, the `_measurement_tables` and
`_metadata_migration` names and the mixins through `__getattr__` (`_LAZY_ATTRS`), because
those submodules import colour-science, h5py, pandas and scipy/scikit-image, and every
`phenotypic.sdk_.*` import runs this `__init__` first. A new re-export whose module imports a
heavy library goes into `_LAZY_ATTRS`, never into an eager import at the top of the file;
`tests/unit/ci/test_startup_imports.py` fails otherwise.

```

- [ ] **Step 3: Mutation proofs.** For each mutation below:
  1. Copy the file to `/tmp`.
  2. Apply the one change.
  3. Run the named test and confirm it **fails**.
  4. Restore the copy and confirm `git diff --quiet -- <file>`.
  5. Rerun the test and confirm it passes.

  Record the command, the failing assertion line and the restored pass in `mutation-proofs.md`.

  | ID | File | Mutation | Guard that must fail |
  |---|---|---|---|
  | M1 | `src/phenotypic/sdk_/__init__.py` | add `from . import colourspace` after the eager `from . import (constants_, …)` block | `tests/unit/ci/test_startup_imports.py::test_cli_help_loads_no_heavy_module` and `::test_importing_image_loads_no_deferred_runtime_module` (tier 1 imports no `sdk_`, so it is not expected to fail) |
  | M2 | `src/phenotypic/_core/_image_parts/_grid_image_handler.py` | move `from phenotypic.grid import CenteredAutoGridFinder` back to module level | `tests/unit/ci/test_deferred_imports.py::test_deferred_names_are_not_imported_at_module_level[_core/_image_parts/_grid_image_handler.py]`. **Not** the sweep: once `abc_` stops importing the core, these imports are import-order-safe on their own, so the sweep stays green (plan review I1). |
  | M3 | `src/phenotypic/abc_/__init__.py` | add `from ._prefab_pipeline import PrefabPipeline` after the eager imports | `…::test_module_imports_first_in_a_fresh_interpreter[phenotypic.analysis]` |
  | M4 | `src/phenotypic/_gui/shell/_app.py` | replace `_SessionProxy(builder_session)` with `builder_session.get().server` | `tests/unit/gui/shell/test_hub_startup_imports.py` |
  | M5 | `src/phenotypic/phenotypicCLI.py` | delete the `load_runtime_dependencies()` call in `phenotypic_cli` | `tests/unit/cli/test_cli_runtime_preload.py::test_cli_aborts_before_any_output_when_a_runtime_dependency_is_broken` |
  | M6 | `src/phenotypic/_gui/shell/_launcher.py` | add `from phenotypic._gui.shell._app import create_app` at module level | `tests/unit/ci/test_startup_imports.py::test_gui_help_loads_no_heavy_module` |
  | M7 | `src/phenotypic/enhance/_subtract_opening.py` | add `import cv2` at module level | `tests/unit/ci/test_deferred_imports.py::test_deferred_names_are_not_imported_at_module_level[enhance/_subtract_opening.py]` |

- [ ] **Step 4: After-measurement.** From the main checkout, run `uv run python docs/superpowers/plans/2026-09-11-lazy-startup/measure_startup.py --label after --out docs/superpowers/reports/2026-09-11-lazy-startup/startup-after.json`.
- [ ] **Step 5: Write `startup-measurements.md`.** It holds the before/after table, the machine line from both JSON files and a one-paragraph reading. Generate the table with:

```bash
uv run --no-project python - <<'PY'
import json
from pathlib import Path
root = Path("docs/superpowers/reports/2026-09-11-lazy-startup")
before = json.loads((root / "startup-before.json").read_text(encoding="utf-8"))
after = json.loads((root / "startup-after.json").read_text(encoding="utf-8"))
print(f"| Path | Before (best of {before['repeats']}) | After (best of {after['repeats']}) | Change |")
print("|---|---|---|---|")
for label, b in before["paths"].items():
    a = after["paths"][label]["best_seconds"]
    b = b["best_seconds"]
    print(f"| {label} | {b:.2f} s | {a:.2f} s | {a - b:+.2f} s |")
PY
```

- [ ] **Step 6: Commit.** Stage the two CLAUDE.md files and the three report files. Subject: `docs(startup): lazy entry-point rules, mutation proofs and before/after timings`.

---

### Task 8: Final review, simplify, regression, finish (orchestrator)

- [ ] **Step 1: Final review.** Dispatch the final whole-change review (`xander-local:implementation-test-reviewer`, opus) over `BASE_PRE..HEAD` with the spec (including Amendment A), this plan, both phase reviews and the carry items from the ledger. The report goes to `docs/superpowers/reports/2026-09-11-lazy-startup/implementation-test-review.md`. Then one fix dispatch, a scoped re-review, and rulings on any residuals.
- [ ] **Step 2: Simplify.** Run `/simplify` over `BASE_PRE..HEAD`, apply the behaviour-identical findings, and rerun the directly affected tests.
- [ ] **Step 3: Full regression**, once:
  1. The Phase 1 default-lanes command.
  2. `PLAYWRIGHT=1` builder e2e plus the six ci_flaky helper modules.
  3. `sphinx-build -n` in both `/tmp/pht-lazy-base` and HEAD, compared with `compare_findings.py docs`. Do not assume zero new warnings: `autodoc_typehints = "both"` resolves annotations, and names now bound only under `TYPE_CHECKING` (`plt`, `go`, `Figure`, `BM3DStages`, `ReconnectConfig`, `Axes`, `Colormap`, `Normalize`, `PathCollection`, `Quiver`) may resolve differently. Investigate every new warning (plan review M-e).
  4. mypy with a fresh cache and ruff, compared with `compare_findings.py mypy|ruff` against the Task 0 files.
  5. Attribute every failure by rerunning it alone on HEAD and at `/tmp/pht-lazy-base`.
- [ ] **Step 4: Finish.** Run `git worktree remove /tmp/pht-lazy-base` and delete `/tmp/lazy-*` scratch. Then use `superpowers:finishing-a-development-branch`. The branch already backs PR #218, so pushing the new commits needs the user's go-ahead.

## Known risks

- **Windows sweep.** `run-pytest-full.yml` runs a Windows nightly, which excludes some packages (`rawpy`, `pympler`, `jupyter`). A package whose `__init__` imports one of them unguarded would fail the import-order sweep there. That is a real finding, not noise, so the sweep stays a hard gate.
- **Detection-mode registration.** `phenotypic.abc_` no longer imports `phenotypic._core._image_parts.detection_modes`, so built-in modes register when the image core first loads rather than when `abc_` does. Nothing in `src/` calls `available_modes()` before the core loads. If the full regression shows otherwise, the fix is an explicit core import at that caller.
