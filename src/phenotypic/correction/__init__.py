"""Image/grid correction for agar plate captures.

Offers operations that realign grids or correct field-of-view drift so detected colonies
stay anchored to their intended wells or pins. The grid aligner adjusts spacing and
offsets using reference points or heuristics suited to arrayed plate layouts.
"""
from ._grid_aligner import GridAligner

__all__ = ["GridAligner"]
