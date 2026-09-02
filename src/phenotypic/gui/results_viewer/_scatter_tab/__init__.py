"""Results-viewer Scatter tab.

The pure data layer lands first: :mod:`._spec` (the configuration object
and the phantom-row predicate), :mod:`._facets` (facet ordering, caps and
the derived frame index) and :mod:`._grouping` (column-to-measurer
attribution). None of them import Dash, so all three are unit-testable
against synthetic frames without booting a server.

The public factory and callback registrar arrive with the Dash wiring.
"""

from __future__ import annotations
