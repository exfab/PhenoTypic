"""Results-viewer Scatter tab.

The pure data layer lands first: :mod:`._spec` (the configuration object
and the phantom-row predicate), :mod:`._facets` (facet ordering, caps and
the derived frame index) and :mod:`._grouping` (column-to-measurer
attribution). None of them import Dash, so all three are unit-testable
against synthetic frames without booting a server.

:mod:`._ids` holds the tab's component ids and :mod:`._layout` builds its
body -- the toolbar, the configuration popover and the figure surface.
``TAB_SCATTER_ID`` itself lives in ``results_viewer/_ids.py``, because the
top-level ``dbc.Tabs`` is what mounts it.

:mod:`._callbacks` is the Dash wiring, and the only module here that
imports Dash beyond the layout. Its bodies are thin wrappers over pure
functions so the ordering rules the tab depends on -- index before
filter, curation before the phantom filter, fingerprint captured at draw
-- can be asserted against the code that runs rather than a copy of it.
"""

from __future__ import annotations

from phenotypic.gui.results_viewer._scatter_tab._callbacks import (
    register_callbacks,
)
from phenotypic.gui.results_viewer._scatter_tab._layout import (
    build_scatter_tab_body,
)

__all__ = ["build_scatter_tab_body", "register_callbacks"]
