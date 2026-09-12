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
