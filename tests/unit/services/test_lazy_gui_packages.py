"""The eager package __init__s are why a content-clean module still drags Dash.

Task 5 promotes RunRegistry, whose _runs_registry.py:59 imports `classify` from
gui.shell._classifier. That single import executes gui/shell/__init__.py. If the
package is eager, _services/runs.py fails the Task 1 purity gate through no fault
of its own content.
"""

from __future__ import annotations

import pytest

from ._boundary import forbidden_imports_after_importing


@pytest.mark.parametrize(
    "module",
    [
        "phenotypic.gui.shell._sandbox",
        "phenotypic.gui.shell._classifier",
        "phenotypic.gui.shell._runs_registry",
        # Was an xfail(strict=True) until Task 6: _space.py used to import dash
        # itself at :33-34, which no package-__init__ laziness could fix. The
        # split moved the pure half to _services/tune_spec and the Dash half to
        # ._space_view, which the shim resolves through PEP 562 __getattr__ --
        # so importing ._space is now genuinely dash-free and the marker is gone
        # rather than rotting into a permanent expected failure.
        "phenotypic.gui.tune._space",
        "phenotypic.gui.tune._run_argv",
    ],
)
def test_submodule_import_does_not_execute_the_dash_app_factory(
    module: str,
) -> None:
    leaked = forbidden_imports_after_importing(module)
    assert not leaked, f"{module} dragged {leaked} in via its package __init__"


@pytest.mark.parametrize(
    ("package", "symbol"),
    [
        ("phenotypic.gui.shell", "create_app"),
        ("phenotypic.gui.shell", "launch_gui"),
        ("phenotypic.gui.shell", "main"),
        ("phenotypic.gui.shell", "SandboxRoot"),
        ("phenotypic.gui.shell", "ToolSession"),
        ("phenotypic.gui.tune", "create_app"),
        ("phenotypic.gui.tune", "TuneRunRoot"),
        ("phenotypic.gui.tune", "TuneRunRootError"),
    ],
)
def test_public_api_is_unchanged(package: str, symbol: str) -> None:
    """Laziness must be invisible: every name still resolves on access."""
    import importlib

    assert getattr(importlib.import_module(package), symbol) is not None


@pytest.mark.parametrize(
    "package", ["phenotypic.gui.shell", "phenotypic.gui.tune"]
)
def test_unknown_attribute_still_raises_attribute_error(package: str) -> None:
    import importlib

    module = importlib.import_module(package)
    with pytest.raises(AttributeError):
        module.definitely_not_a_real_symbol
