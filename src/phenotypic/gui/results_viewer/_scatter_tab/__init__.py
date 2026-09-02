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

The public factory and callback registrar arrive with the Dash wiring.
"""

from __future__ import annotations
