"""Startup-path guards: what each entry point loads, and import-order independence.

Every guard runs its entry point in a fresh interpreter (``tests._startup_probe``)
and pairs the absence it asserts with a positive control, so a probe that silently
imported nothing cannot pass.
"""

from __future__ import annotations

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
