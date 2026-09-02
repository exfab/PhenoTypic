"""Shared public Method B parameters for zone-measurement operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union
import weakref

import numpy as np
from pydantic import PrivateAttr, field_validator, model_validator

from phenotypic.abc_ import MeasureFeatures
from phenotypic.measure._orientation_zone_segmentation import (
    OrientationChangePointParams,
)
from phenotypic.measure._zone_segmentation import detected_center_coordinates
from phenotypic.sdk_.typing_ import OperationField

if TYPE_CHECKING:
    from phenotypic._core._image import Image


class CanonicalZoneMeasure(MeasureFeatures):
    """Private base declaring the single canonical zone-resolution surface."""

    center_detector: Union[OperationField, None] = None  # type: ignore[valid-type]
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

    _center_cache_image_ref: "weakref.ReferenceType[Image] | None" = PrivateAttr(
        default=None
    )
    _center_cache_signature: str | None = PrivateAttr(default=None)
    _center_cache: dict[int, tuple[float, float]] = PrivateAttr(
        default_factory=dict
    )

    @field_validator("center_detector")
    @classmethod
    def _valid_center_detector(cls, value: Any) -> Any:
        if value is None:
            return value
        from phenotypic import ImagePipeline
        from phenotypic.abc_ import ObjectDetector

        if not isinstance(value, (ObjectDetector, ImagePipeline)):
            raise ValueError(
                "center_detector must be an ObjectDetector, ImagePipeline, or None"
            )
        return value

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

    def _canonical_center_for_object(
        self,
        image: "Image",
        object_label: int,
    ) -> tuple[tuple[float, float] | None, bool]:
        """Return an optional detector-selected center for one final object.

        The configured detector is applied once per image-and-parameter
        signature. Its center components are associated with final colonies by
        pixel overlap. ``required`` is true only when the user configured a
        detector, allowing the shared resolver to distinguish an absent
        optional override from a requested detector that found no center.
        """
        if self.legacy_mode or self.center_detector is None:
            return None, False

        signature = self.model_dump_json()
        cached_image = (
            self._center_cache_image_ref()
            if self._center_cache_image_ref is not None
            else None
        )
        if cached_image is not image or self._center_cache_signature != signature:
            from phenotypic import ImagePipeline

            if isinstance(self.center_detector, ImagePipeline):
                center_image = self.center_detector.apply(
                    image, inplace=False, reset=False
                )
            else:
                center_image = self.center_detector.apply(image, inplace=False)
            self._center_cache = detected_center_coordinates(
                np.asarray(image.objmap[:]),
                np.asarray(center_image.objmap[:]),
            )
            self._center_cache_image_ref = weakref.ref(image)
            self._center_cache_signature = signature

        return self._center_cache.get(int(object_label)), True

    @classmethod
    def _migrate_serialized_params(cls, params: dict[str, Any]) -> dict[str, Any]:
        """Preserve legacy behavior for payloads created before this field."""
        migrated = dict(params)
        migrated.setdefault("legacy_mode", True)
        return migrated
