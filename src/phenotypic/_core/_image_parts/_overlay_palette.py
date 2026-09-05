"""The object-overlay colour palette, free of any plotting dependency.

Lifted out of :class:`AccessorMplHandler` so a caller that needs only the
colours does not have to import the matplotlib accessor chain to get them.
``measure/_measure_symzones.py`` did exactly that: a measure-time path importing
the whole plotting stack to read a 10x3 array of floats.

Kept identical to ``skimage.color.colorlabel.DEFAULT_COLORS`` so overlays drawn
from this palette match the ones skimage draws elsewhere in the library.
"""

from __future__ import annotations

import numpy as np

# Default overlay colors matching skimage.color.colorlabel.DEFAULT_COLORS
OVERLAY_COLORS: np.ndarray = np.array([
    [1.0, 0.0, 0.0],  # red
    [0.0, 0.0, 1.0],  # blue
    [1.0, 1.0, 0.0],  # yellow
    [1.0, 0.0, 1.0],  # magenta
    [0.0, 0.5, 0.0],  # green
    [0.294, 0.0, 0.510],  # indigo
    [1.0, 0.549, 0.0],  # darkorange
    [0.0, 1.0, 1.0],  # cyan
    [1.0, 0.753, 0.796],  # pink
    [0.604, 0.804, 0.196],  # yellowgreen
], dtype=np.float32)
