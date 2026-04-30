"""Interactive results viewer for PhenoTypic CLI output.

Public entry points:

* :func:`create_app` -- build a configured :class:`dash.Dash` instance
  for an :class:`OutputRoot` (see
  :mod:`phenotypic.gui.results_viewer._output_root`). Useful when an
  embedding application wants to mount the viewer alongside other Dash
  pages.
* :func:`launch_results_viewer` -- one-shot launcher that discovers
  the output root, builds the app, prints a startup banner, and runs
  the server. Mirrors what ``python -m
  phenotypic.gui.results_viewer`` does.
"""

from __future__ import annotations

from phenotypic.gui.results_viewer.__main__ import launch_results_viewer
from phenotypic.gui.results_viewer._app import create_app

__all__ = ["create_app", "launch_results_viewer"]
