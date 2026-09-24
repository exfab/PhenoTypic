"""Fixed evidence thresholds shared by the orientation-field consumers.

``MeasureOrientationZones`` and the Method B zone resolver used to declare
these values separately under different names. They are the per-pixel,
per-crossing and per-cell floors that decide whether one orientation estimate
is reliable at all. Ring-level support thresholds are different: they are the
user-facing ``zone_min_crossings`` and ``zone_min_resultant`` fields on
``CanonicalZoneMeasure`` and are never hard-coded.
"""

from __future__ import annotations

import numpy as np

#: ``orientation_field`` returns the dominant image-gradient normal. Fibers run
#: perpendicular to it, so a fiber axis is the orientation plus this offset.
FIBER_AXIS_OFFSET: float = np.pi / 2.0

#: Structure-tensor coherence below which a pixel's orientation is unreliable.
RELIABLE_PIXEL_COHERENCE: float = 0.15

#: Half-width in pixels of the band sampled around each literal ring.
CROSSING_HALF_WIDTH: float = 1.5

#: Doubled-angle resultant below which a pooled axial estimate (one literal
#: crossing, one ring-sector cell, or one local bend window) has no reliable
#: axis.
MIN_AXIAL_RESULTANT: float = 0.15

#: Reliable pixels required before one polar sector contributes an estimate.
MIN_PIXELS_PER_SECTOR: int = 3

#: Default number of equal polar sectors around a colony (10 degrees each).
N_SECTORS: int = 36
