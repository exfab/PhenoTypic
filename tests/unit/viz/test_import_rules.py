"""Enforce the figure-layer import rule by static analysis.

The contract (``abc_/_figure_provider.py``) must import only stdlib at module
level — no UI toolkit and not even plotly (it touches plotly lazily). The theme
(``tools_/viz/figures/_theme.py``) may import plotly but no UI toolkit. UI toolkits
(dash / ipywidgets / panel / bokeh / param) are confined to the shells.

Checking the AST's top-level statements (not nested ``if TYPE_CHECKING:`` blocks
or in-function lazy imports) is what makes "imported at runtime" precise.
"""

from __future__ import annotations

import ast
from pathlib import Path

import phenotypic

_SRC = Path(phenotypic.__file__).parent
_UI_TOOLKITS = {"dash", "ipywidgets", "panel", "bokeh", "param"}


def _toplevel_imports(path: Path) -> set[str]:
    """Top-level (runtime) imported root module names for a source file."""
    tree = ast.parse(path.read_text())
    roots: set[str] = set()
    for node in tree.body:  # only module-body statements run at import time
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_contract_imports_stdlib_only():
    imports = _toplevel_imports(_SRC / "abc_" / "_figure_provider.py")
    # no UI toolkit and not plotly (applied lazily)
    assert not (imports & _UI_TOOLKITS), f"contract imports UI toolkit: {imports}"
    assert "plotly" not in imports, "contract must touch plotly lazily, not at import"


def test_theme_imports_no_ui_toolkit():
    imports = _toplevel_imports(_SRC / "tools_" / "viz" / "figures" / "_theme.py")
    assert not (imports & _UI_TOOLKITS), f"theme imports UI toolkit: {imports}"


def test_notebook_adapter_does_not_import_ipywidgets_at_module_level():
    imports = _toplevel_imports(_SRC / "tools_" / "viz" / "notebook" / "_adapter.py")
    # ipywidgets and IPython are imported lazily inside the builder, not at module top
    assert "ipywidgets" not in imports
    assert "IPython" not in imports
    assert not (imports & _UI_TOOLKITS), imports
