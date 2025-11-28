"""Grid discovery for plated fungal colonies.

Provides tools to define the row/column layout of arrayed plates (e.g., 96- or 384-spot)
so downstream detection and measurements align colonies to expected wells. Supports
automatic grid inference and manual specification for challenging imaging conditions.
"""

from ._auto_grid_finder import AutoGridFinder
from ._manual_grid_finder import ManualGridFinder

__all__ = [
    "AutoGridFinder",
    "ManualGridFinder"
]
