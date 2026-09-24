"""Constants and value types shared by the MeasureOrientationZones modules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from phenotypic.measure._zone_segmentation import (
    ZoneResolution,
    ZoneSegmentation,
)

_VARIANTS = ("Radial", "Mask")
_ZONES = ("Overall", "Dense", "Sparse")
_EPS = 1e-9
_PRIMARY_OUTWARD_METRICS = (
    "OutwardRotationSustainedPeak",
    "OutwardRotationNet",
    "OutwardRotationRate",
    "OutwardRotationConsistency",
)
_DIAGNOSTIC_OUTWARD_METRICS = (
    "OutwardRotationRawPeak",
    "OutwardRotationP90",
    "OutwardRotationP95",
    "OutwardRotationMedianMagnitude",
    "OutwardRotationAbsoluteArea",
    "OutwardRotationTotalVariation",
    "OutwardRotationRateGradient",
    "OutwardRotationRingSupport",
    "OutwardRotationRunSpanSupport",
    "OutwardRotationMedianResultant",
)


@dataclass(frozen=True)
class _ObjectZoneAnalysis:
    """Named results from one object's zone and orientation analysis.

    The zone resolver owns the segmentation geometry. The remaining arrays are
    the orientation evidence consumed by measurements and diagnostic figures.
    Keeping these values named avoids positional tuple unpacking at every call
    site and makes their shared origin explicit.
    """

    prop: object
    resolution: ZoneResolution
    object_mask: np.ndarray
    orientation: np.ndarray
    coherence: np.ndarray
    gradient: np.ndarray
    distance_map: np.ndarray
    center: tuple[float, float]

    @property
    def segmentation(self) -> ZoneSegmentation:
        """Return the geometry owned by the shared resolver result."""
        return self.resolution.segmentation
