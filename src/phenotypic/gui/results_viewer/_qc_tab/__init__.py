"""QC tab for the results viewer.

Per-check :class:`~phenotypic.gui._qc_recipe.QcRecipe`-driven UI: each
entry renders as a card with a Plotly figure, a summary strip, and four
lifecycle controls (edit / enable / duplicate / delete). The cards are
backed by a shared add/edit/duplicate :class:`dbc.Modal` that hosts a
class picker and a :func:`~phenotypic.gui._param_forms.param_form`
parameter editor.

Public surface:

* :func:`build_qc_tab_body` — layout factory called from
  :mod:`phenotypic.gui.results_viewer._layout`.
* :func:`register_qc_callbacks` — registers all callbacks owned by the
  tab.
"""
from __future__ import annotations

from phenotypic.gui.results_viewer._qc_tab._callbacks import (
    register_qc_callbacks,
)
from phenotypic.gui.results_viewer._qc_tab._check_card import build_check_card
from phenotypic.gui.results_viewer._qc_tab._layout import build_qc_tab_body

__all__ = [
    "build_check_card",
    "build_qc_tab_body",
    "register_qc_callbacks",
]
