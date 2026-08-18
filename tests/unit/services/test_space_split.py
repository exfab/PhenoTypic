"""`_space.py` splits into a Dash-free pure half and a Dash view half.

The pure half is what Task 7 folds the tune spec authoring / validation / export
modules into, and what the MCP server will call; the view half is the only part
allowed to know Dash exists. These tests pin the direction of the dependency —
view imports pure, never the reverse — and that the legacy `_space` import path
keeps resolving for the three GUI call sites that still use it.
"""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys


def test_pure_half_is_importable_without_dash():
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import phenotypic._services.tune_spec as t; import sys;"
            " print('dash' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False"


def test_pure_symbols_moved():
    from phenotypic._services.tune_spec import apply_space_edits, space_to_spec

    assert callable(space_to_spec)
    assert callable(apply_space_edits)


def _parsed_imports(module: object) -> list[str]:
    """Every dotted name ``module``'s import statements name.

    Parsed, never grepped. Both halves of this split discuss the other half in
    prose — they have to, to explain why the split exists — so a source-text
    check reports a docstring mention as an import and an aliased
    ``from phenotypic import gui`` as clean.
    """
    tree = ast.parse(inspect.getsource(module))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            names.append(base)
            names += [f"{base}.{alias.name}" for alias in node.names]
    return names


def test_pure_half_does_not_import_the_gui():
    """The pure half may not reach up into any rendering surface.

    The plan specified ``"phenotypic.gui" not in inspect.getsource(tune_spec)``,
    which is a source-text grep: it fails on a module that merely *mentions* the
    GUI in a docstring (this one does) and passes on one that imports it under
    an alias. It would also outlaw the legitimate ``if TYPE_CHECKING:`` block.
    ``tests/unit/services/test_argv_promotion.py::test_argv_module_does_not_import_gui``
    is the correct implementation; this mirrors it.
    """
    from phenotypic._services import tune_spec

    offenders = [
        name
        for name in _parsed_imports(tune_spec)
        if name == "phenotypic.gui" or name.startswith("phenotypic.gui.")
    ]
    assert not offenders, f"_services.tune_spec imports from the GUI: {offenders}"


def test_view_half_imports_the_pure_half():
    """The dependency runs downward, and it is a real import.

    Asserted against the parsed imports for the same reason as above: the view
    half's docstring names ``phenotypic._services.tune_spec``, so a substring
    check stays green after the view stops importing it and re-declares the
    helpers locally — the exact drift this split exists to prevent.
    """
    from phenotypic.gui.tune import _space_view

    assert "phenotypic._services.tune_spec" in _parsed_imports(_space_view)


def test_legacy_import_path_still_works():
    """_setup_authoring.py:28 and three call sites import from _space."""
    from phenotypic.gui.tune._space import (  # noqa: F401
        apply_space_edits,
        build_space_view,
        setup_knob_forms,
        space_to_spec,
    )


def test_shim_reexports_the_same_objects():
    """A double definition would silently diverge; identity forbids it.

    The private names are load-bearing: ``_callbacks.py:2227`` imports
    ``_load_space_source`` and ``tests/unit/gui/tune/test_space.py`` imports
    ``_apply_edits`` and ``_knob_form`` through ``_space``.
    """
    from phenotypic._services import tune_spec
    from phenotypic.gui.tune import _space, _space_view

    for name in (
        "_apply_edits",
        "_build_search_space",
        "_load_space_source",
        "apply_space_edits",
        "space_to_spec",
    ):
        assert getattr(_space, name) is getattr(tune_spec, name), name

    for name in ("_knob_form", "build_space_view", "setup_knob_forms"):
        assert getattr(_space, name) is getattr(_space_view, name), name
