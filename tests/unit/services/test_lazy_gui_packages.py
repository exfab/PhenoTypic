"""The eager package __init__s are why a content-clean module still drags Dash.

Task 5 promotes RunRegistry, whose _runs_registry.py:59 imports `classify` from
gui.shell._classifier. That single import executes gui/shell/__init__.py. If the
package is eager, _services/runs.py fails the Task 1 purity gate through no fault
of its own content.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

FORBIDDEN = ("dash", "dash_bootstrap_components", "flask", "werkzeug")

_PROBE = """
import importlib, sys
importlib.import_module({module!r})
print(",".join(sorted(m for m in {forbidden!r} if m in sys.modules)))
"""


@pytest.mark.parametrize(
    "module",
    [
        "phenotypic.gui.shell._sandbox",
        "phenotypic.gui.shell._classifier",
        "phenotypic.gui.shell._runs_registry",
        pytest.param(
            "phenotypic.gui.tune._space",
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "_space.py:33-34 imports dash itself, not via its package "
                    "__init__, so making the package lazy cannot fix it. "
                    "Task 6 splits the module into a pure half "
                    "(_services/tune_spec) and a Dash half (_space_view). "
                    "strict=True means this XPASSes -> FAILS the moment Task 6 "
                    "lands, forcing this marker to be removed rather than rot."
                ),
            ),
        ),
        "phenotypic.gui.tune._run_argv",
    ],
)
def test_submodule_import_does_not_execute_the_dash_app_factory(
    module: str,
) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _PROBE.format(module=module, forbidden=FORBIDDEN),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"{module} failed to import:\n{proc.stderr}"
    leaked = [n for n in proc.stdout.strip().split(",") if n]
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
