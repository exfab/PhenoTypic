"""Shared public Method B parameters for zone-measurement operations."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import field_validator, model_validator

from phenotypic.abc_ import MeasureFeatures
from phenotypic.measure._orientation_zone_segmentation import (
    OrientationChangePointParams,
)


class CanonicalZoneMeasure(MeasureFeatures):
    """Private base declaring the single canonical zone-resolution surface."""

    legacy_mode: bool = False
    outer_zone_percentile: float = 100.0
    sigma_d: float = 1.5
    sigma_i: float = 4.0
    radial_ring_width: float = 8.0
    zone_minimum_segment: int = 4
    zone_min_crossings: int = 3
    zone_min_resultant: float = 0.15
    zone_min_ring_coherence: float = 0.15
    zone_support_weight: float = 4.0
    zone_outer_support_margin: float = 0.0
    zone_maximum_gap: int = 0

    @field_validator(
        "outer_zone_percentile",
        "sigma_d",
        "sigma_i",
        "radial_ring_width",
        "zone_min_resultant",
        "zone_min_ring_coherence",
        "zone_support_weight",
        "zone_outer_support_margin",
        mode="before",
    )
    @classmethod
    def _reject_boolean_floats(cls, value: Any) -> Any:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("numeric Method B parameters cannot be boolean")
        return value

    @field_validator("outer_zone_percentile")
    @classmethod
    def _valid_outer_percentile(cls, value: float) -> float:
        if not np.isfinite(value) or not 0.0 < value <= 100.0:
            raise ValueError("outer_zone_percentile must be finite and in (0, 100]")
        return float(value)

    @field_validator("sigma_d", "sigma_i", "radial_ring_width")
    @classmethod
    def _positive_scales(cls, value: float) -> float:
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("Method B scales must be finite and > 0")
        return float(value)

    @field_validator("zone_minimum_segment", "zone_min_crossings", mode="before")
    @classmethod
    def _positive_integers(cls, value: Any) -> int:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 1
        ):
            raise ValueError("Method B count parameters must be integers >= 1")
        return int(value)

    @field_validator("zone_maximum_gap", mode="before")
    @classmethod
    def _nonnegative_integer(cls, value: Any) -> int:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 0
        ):
            raise ValueError("zone_maximum_gap must be an integer >= 0")
        return int(value)

    @field_validator("zone_min_resultant", "zone_min_ring_coherence")
    @classmethod
    def _valid_support_threshold(cls, value: float) -> float:
        if not np.isfinite(value) or not 0.15 <= value <= 1.0:
            raise ValueError("Method B support thresholds must be in [0.15, 1]")
        return float(value)

    @field_validator("zone_support_weight")
    @classmethod
    def _valid_support_weight(cls, value: float) -> float:
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("zone_support_weight must be finite and >= 0")
        return float(value)

    @field_validator("zone_outer_support_margin")
    @classmethod
    def _valid_support_margin(cls, value: float) -> float:
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("zone_outer_support_margin must be finite and in [0, 1]")
        return float(value)

    @model_validator(mode="after")
    def _canonical_center_is_distance_based(self):
        if not self.legacy_mode and getattr(self, "method", "distance") != "distance":
            raise ValueError("canonical Method B requires method='distance'")
        return self

    def _orientation_change_point_params(self) -> OrientationChangePointParams:
        """Snapshot the shared Method B fields for the pure solver."""
        return OrientationChangePointParams(
            sigma_d=self.sigma_d,
            sigma_i=self.sigma_i,
            ring_width=self.radial_ring_width,
            outer_zone_percentile=self.outer_zone_percentile,
            minimum_segment=self.zone_minimum_segment,
            min_crossings=self.zone_min_crossings,
            min_resultant=self.zone_min_resultant,
            min_ring_coherence=self.zone_min_ring_coherence,
            support_weight=self.zone_support_weight,
            outer_support_margin=self.zone_outer_support_margin,
            maximum_gap=self.zone_maximum_gap,
        )

    @classmethod
    def _migrate_serialized_params(cls, params: dict[str, Any]) -> dict[str, Any]:
        """Preserve legacy behavior for payloads created before this field."""
        migrated = dict(params)
        migrated.setdefault("legacy_mode", True)
        return migrated
