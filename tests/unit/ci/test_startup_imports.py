"""Startup-path guards: what each entry point loads, and import-order independence.

Every guard runs its entry point in a fresh interpreter (``tests._startup_probe``)
and pairs the absence it asserts with a positive control, so a probe that silently
imported nothing cannot pass.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

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
    # The control is the half that makes this able to fail: with the deferrals in place,
    # importing `_startup_perf` must load none of the eight, so `missing == []` can only
    # come from the call under test. Before Task 2 made the package lazy, `before` was the
    # full list and this assertion passed with the function deleted.
    assert report["before"] == []
    assert report["missing"] == []


def test_every_deferred_module_is_watched_at_startup() -> None:
    """A module deferred for startup must also be one the startup guards watch."""
    watched = set(HEAVY_STARTUP_MODULES) | {"matplotlib.pyplot"}
    assert set(DEFERRED_RUNTIME_MODULES) <= watched


def test_the_watched_sets_are_the_ones_the_spec_names() -> None:
    """Pin the contents, or every guard asserted against them can be retired silently.

    Tiers 1 and 4 assert ``loaded == []`` over whatever these tuples happen to contain,
    and the subset check above is satisfied by the empty set. Without this test the
    cheapest way to green a failing tier is to delete the offending name from the tuple,
    which passes everything and ships the regression. Changing either set now requires
    changing the spec's own list in the same commit, which is the point.
    """
    assert set(HEAVY_STARTUP_MODULES) == {
        "bm3d",
        "colour",
        "cv2",
        "dash",
        "h5py",
        "mahotas",
        "matplotlib",
        "numba",
        "pandas",
        "plotly",
        "polars",
        "pyarrow",
        "scipy",
        "skimage",
    }
    assert set(DEFERRED_RUNTIME_MODULES) == {
        "bm3d",
        "colour",
        "cv2",
        "h5py",
        "mahotas",
        "matplotlib.pyplot",
        "numba",
        "plotly",
    }


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
    # Plain modules, so package discovery below never finds them. `_startup_perf` is
    # the first thing every entry point imports, and `settings` is in `_LAZY_SUBPACKAGES`.
    "phenotypic._startup_perf",
    "phenotypic.settings",
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


#: Subpackages a user imports directly (``from phenotypic.detect import ...``). Deferring
#: a library at its importer is only worth anything if importing the subpackage stays free
#: of it -- that is the stated purpose of the detector rows, and nothing else asserts it:
#: tiers 1 and 4 never import these at all, and the sweep only checks that they import.
GUARDED_SUBPACKAGES = (
    "phenotypic.abc_",
    "phenotypic.analysis",
    "phenotypic.correction",
    "phenotypic.detect",
    "phenotypic.enhance",
    "phenotypic.grid",
    "phenotypic.measure",
    "phenotypic.plotting",
    "phenotypic.post",
    "phenotypic.refine",
    "phenotypic.schema",
    "phenotypic.sdk_",
    "phenotypic.util",
)

#: Deliberate exceptions, asserted exactly rather than merely allowed, so an exception that
#: stops being true fails as loudly as a new leak. ``correction`` keeps ``colour`` at module
#: level because ``_color_checker_profile.py`` and ``_helpers.py`` are on the spec's
#: unchanged list (spec Amendment A P3).
SUBPACKAGE_EXPECTED_DEFERRALS: dict[str, tuple[str, ...]] = {
    "phenotypic.correction": ("colour",),
}


@pytest.mark.parametrize("package", GUARDED_SUBPACKAGES)
def test_operation_subpackage_loads_no_deferred_runtime_library(package: str) -> None:
    """Tier 5: importing an operation subpackage pays for none of the deferred libraries."""
    report = run_startup_probe(
        "import importlib\n"
        f"importlib.import_module({package!r})\n"
        f"watched = {sorted(DEFERRED_RUNTIME_MODULES)!r}\n"
        "report = {'loaded': sorted(m for m in watched if m in sys.modules),\n"
        f"          'control': {package!r} in sys.modules}}\n"
    )
    assert report["control"] is True
    assert report["loaded"] == sorted(SUBPACKAGE_EXPECTED_DEFERRALS.get(package, ()))


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
    # Not ``sorted(available_modes())``: the claim is that the help text is byte-identical
    # to what the registry call produced, and wrapping the right-hand side in ``sorted()``
    # makes the assertion blind to ``available_modes()``'s own ordering -- which is the
    # property doing the work (``_detection_mode.py`` returns ``tuple(sorted(...))``).
    assert list(option.type.choices) == list(available_modes())
    assert set(get_args(DetectMode)) == set(available_modes())


def test_every_deferred_runtime_module_is_a_hard_dependency() -> None:
    """The required preload set must hold only libraries every install has.

    An extras-only library here would make `phenotypic --help`'s sibling — an actual
    run — fail at startup on any environment without that extra. Such a library
    belongs in ``DEFERRED_OPTIONAL_MODULES``, which is skipped when absent.
    """
    from importlib.metadata import packages_distributions

    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    # Only dependencies with no environment marker: a marker means the dependency is
    # legitimately absent somewhere (the repo already ships `rawpy;sys_platform!='win32'`),
    # and preloading it unconditionally would fail on exactly that platform.
    unconditional = {
        re.split(r"[<>=!\[;]", specifier, maxsplit=1)[0].strip().lower().replace("_", "-")
        for specifier in project.get("dependencies", [])
        if ";" not in specifier
    }
    installed = packages_distributions()

    unresolved, optional_only = [], []
    for module_name in DEFERRED_RUNTIME_MODULES:
        distributions = {
            name.lower().replace("_", "-")
            for name in installed.get(module_name.split(".")[0], ())
        }
        if not distributions:
            unresolved.append(module_name)
        elif not distributions & unconditional:
            optional_only.append(f"{module_name} -> {sorted(distributions)}")

    assert unresolved == [], f"not installed, so this check cannot run: {unresolved}"
    assert optional_only == [], (
        f"conditionally-installed libraries in the required preload set: {optional_only}; "
        "an extra, or a dependency carrying an environment marker, belongs in "
        "DEFERRED_OPTIONAL_MODULES"
    )


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
