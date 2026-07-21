"""Import-boundary tests for :mod:`phenotypic.abc_.plotting`."""

from __future__ import annotations

import subprocess
import sys


def test_plotting_capabilities_are_not_reexported_from_abc_root() -> None:
    import phenotypic.abc_ as abc_root
    import phenotypic.abc_.plotting as plotting

    assert plotting.PhtPlot.__module__.startswith("phenotypic.abc_.plotting")
    assert not hasattr(abc_root, "PhtPlot")
    assert not hasattr(abc_root, "PlotImage")
    assert not hasattr(abc_root, "PlotMeas")
    assert not hasattr(abc_root, "PlotAnalysis")
    assert not hasattr(abc_root, "PlotQc")


def test_plotting_subpackage_adds_no_ui_or_runtime_plotting_imports() -> None:
    script = """
import sys

# The eager phenotypic package currently imports Plotly and Matplotlib for
# unrelated public modules. Clear that baseline so this test isolates imports
# added by phenotypic.abc_.plotting itself.
import phenotypic.abc_

for name in tuple(sys.modules):
    if name == "plotly" or name.startswith("plotly."):
        del sys.modules[name]
    elif name == "matplotlib" or name.startswith("matplotlib."):
        del sys.modules[name]
    elif name == "dash" or name.startswith("dash."):
        del sys.modules[name]
    elif name == "ipywidgets" or name.startswith("ipywidgets."):
        del sys.modules[name]
    elif name == "phenotypic.plotting" or name.startswith("phenotypic.plotting."):
        del sys.modules[name]

import phenotypic.abc_.plotting

forbidden = [
    name
    for name in sys.modules
    if name == "plotly"
    or name.startswith("plotly.")
    or name == "matplotlib"
    or name.startswith("matplotlib.")
    or name == "dash"
    or name.startswith("dash.")
    or name == "ipywidgets"
    or name.startswith("ipywidgets.")
    or name == "phenotypic.plotting"
    or name.startswith("phenotypic.plotting.")
]
if forbidden:
    raise SystemExit(f"forbidden imports: {forbidden}")
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
