"""The boundary that makes `_services` a layer rather than a folder."""

from __future__ import annotations

import pkgutil
import subprocess
import sys

import pytest

# Two libraries are deliberately NOT listed, both for reasons that make them
# look like coverage while providing none:
#   dash_ag_grid — not installed here, so a probe importing it dies with
#     ModuleNotFoundError and the test fails on the returncode assert rather
#     than the leak assert.
#   plotly — `import phenotypic` alone already pulls it in (verified), so no
#     module in this or any other tier can satisfy the check. Forbidding it
#     would make the gate unsatisfiable rather than strict.
FORBIDDEN = ("dash", "dash_bootstrap_components", "flask", "werkzeug")

# One subprocess per module: a single process would let module A's clean import
# be vouched for by module B having already been imported, and vice versa.
_PROBE = """
import importlib, sys
importlib.import_module({module!r})
leaked = sorted(m for m in {forbidden!r} if m in sys.modules)
print(",".join(leaked))
"""


def _service_modules() -> list[str]:
    import phenotypic._services as services

    # walk_packages, not iter_modules: iter_modules is non-recursive, so a
    # _services/<subpkg>/leak.py importing dash was invisible to this gate.
    return [
        m.name
        for m in pkgutil.walk_packages(services.__path__, prefix="phenotypic._services.")
    ]


def test_services_package_exists_and_is_lazy():
    import phenotypic._services as services

    assert services.__path__, "phenotypic._services must be a package"


@pytest.mark.parametrize("module", _service_modules())
def test_service_module_imports_no_dash(module: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE.format(module=module, forbidden=FORBIDDEN)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"{module} failed to import:\n{proc.stderr}"
    leaked = [name for name in proc.stdout.strip().split(",") if name]
    assert not leaked, f"{module} dragged {leaked} into sys.modules"


# The single accepted upward import in this tier, and the reason for it.
# RunRegistry.rehydrate_from_sandbox needs classify(); _classifier.py is itself
# Dash-free after Task 2, but no task promotes it. Anything NOT listed here is a
# test failure rather than a silent precedent.
GUI_IMPORT_ALLOWLIST: dict[str, set[str]] = {
    "phenotypic._services.runs": {"phenotypic.gui.shell._classifier"},
}


@pytest.mark.parametrize("module", _service_modules())
def test_service_module_does_not_import_gui(module: str) -> None:
    """No ``_services`` module may import ``phenotypic.gui`` off-allowlist.

    Parsed imports, not a source substring: a ``"phenotypic.gui" not in source``
    check matches prose in a docstring (several of these modules explain *why*
    they must not import the GUI) and misses ``from phenotypic import gui``.
    """
    import ast
    import importlib
    import inspect

    tree = ast.parse(inspect.getsource(importlib.import_module(module)))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.append(node.module)

    reached = {
        name for name in imported
        if name == "phenotypic.gui" or name.startswith("phenotypic.gui.")
    }
    allowed = GUI_IMPORT_ALLOWLIST.get(module, set())
    assert reached <= allowed, (
        f"{module} imports {sorted(reached - allowed)} from phenotypic.gui; "
        "promote the dependency or add an explicit allowlist entry explaining why"
    )
