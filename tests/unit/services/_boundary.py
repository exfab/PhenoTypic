"""Shared machinery for the ``_services`` boundary gates.

Three test modules in this package ask the same two questions — *does importing
this module drag a GUI library into ``sys.modules``* and *which*
``phenotypic.gui`` *names do its import statements reach* — and each had grown
its own copy of the answer. Three copies of a gate is three chances for one of
them to be quietly weakened while the others keep passing and vouch for it.

Deliberately not a ``conftest.py``: these are plain callables the tests import
by name, not fixtures, and the parametrize lists below are built at collection
time.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import subprocess
import sys
from types import ModuleType

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


def forbidden_imports_after_importing(module: str) -> list[str]:
    """Import ``module`` in a fresh interpreter; return the ``FORBIDDEN`` leaks.

    Args:
        module: Dotted name to import in the child process.

    Returns:
        The forbidden libraries that ended up in the child's ``sys.modules``,
        sorted. Empty when the import is clean.

    Raises:
        AssertionError: If the child process could not import ``module`` at
            all — a failed import leaks nothing and would otherwise read as a
            pass.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE.format(module=module, forbidden=FORBIDDEN)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"{module} failed to import:\n{proc.stderr}"
    return [name for name in proc.stdout.strip().split(",") if name]


def parsed_import_names(module: ModuleType | str) -> list[str]:
    """Every dotted name ``module``'s import statements name.

    Parsed, never grepped. Modules on both sides of this boundary discuss the
    other side in prose — they have to, to explain why the boundary exists — so
    a source-text check reports a docstring mention as an import and reports an
    aliased ``from phenotypic import gui`` as clean.

    Both the ``from`` target and the names pulled from it are yielded: without
    the names, ``from phenotypic import gui`` yields only ``"phenotypic"`` and
    slips past every caller below.

    Relative imports are skipped. A ``from .x import y`` inside this repo's
    packages can never *name* ``phenotypic.gui``, and folding its ``.x`` /
    ``x.y`` spellings into the result would only add names no caller can
    interpret.

    Args:
        module: An imported module object, or its dotted name.

    Returns:
        Dotted names, in source order, with duplicates preserved.
    """
    if isinstance(module, str):
        module = importlib.import_module(module)
    tree = ast.parse(inspect.getsource(module))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module)
            names.extend(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def gui_modules_reached(module: ModuleType | str) -> set[str]:
    """Every ``phenotypic.gui`` name ``module``'s parsed imports reach.

    Shared by the tier-wide subset gate, the per-entry allowlist equality pin,
    and the two promotion-specific gates, so none of them can drift into
    disagreeing about what "reaches" means.
    """
    return {
        name
        for name in parsed_import_names(module)
        if name == "phenotypic.gui" or name.startswith("phenotypic.gui.")
    }


def shallowest_modules(names: set[str]) -> set[str]:
    """Drop every name that is nested under another name in ``names``.

    ``gui_modules_reached`` yields both ``X`` and ``X.symbol`` for a
    ``from X import symbol``. Only the shallowest name is a module, so this
    reduces a reach set to allowlist grain: how many symbols a caller happens to
    pull from one module is not a fact about how far it reaches.
    """
    return {
        name
        for name in names
        if not any(other != name and name.startswith(f"{other}.") for other in names)
    }
