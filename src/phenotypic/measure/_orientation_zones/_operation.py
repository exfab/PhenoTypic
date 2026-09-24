"""Measure absolute and radial-relative hyphal orientation by radial zone."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias

if TYPE_CHECKING:
    # See MeasureSymZones: this satisfies the pydantic mypy plugin's inherited
    # field resolution without changing CanonicalZoneMeasure's runtime type.
    OperationField: TypeAlias = Any

import weakref

import numpy as np
import pandas as pd
from pydantic import PrivateAttr, field_validator, model_validator

from phenotypic.measure._canonical_zone_measure import CanonicalZoneMeasure
from phenotypic.measure._zone_segmentation import (
    ZoneSegmentation,
    expand_slice_around_center,
)
from phenotypic.schema import (
    MeasurementInfo,
    OBJECT,
    ORIENTATION_ZONE_DIAGNOSTIC,
    ORIENTATION_ZONE_PRIMARY,
)
from phenotypic.sdk_._radial_geometry import distance_from_point
from phenotypic.sdk_.orientation_fields import (
    CROSSING_HALF_WIDTH,
    FIBER_AXIS_OFFSET,
    MIN_AXIAL_RESULTANT,
    MIN_PIXELS_PER_SECTOR,
    N_SECTORS,
    RELIABLE_PIXEL_COHERENCE,
    aggregate_literal_crossing_zone,
    axial_change,
    axial_sector_means,
    literal_crossing_ring_profile,
    literal_skeleton_ring_crossings,
    long_range_ring_rotation_profile,
    orientation_field,
    radial_ring_orientation_profile,
    signed_axial_mean,
    signed_radial_relative_field,
)

from ._common import (
    _DIAGNOSTIC_OUTWARD_METRICS,
    _EPS,
    _PRIMARY_OUTWARD_METRICS,
    _VARIANTS,
    _ZONES,
    _ObjectZoneAnalysis,
)
from ._figures import _OrientationZonesFigures

_LITERAL_CROSSING_MIN_POINTS = 3


class MeasureOrientationZones(CanonicalZoneMeasure, _OrientationZonesFigures):
    """Measure absolute and radial-relative hyphal orientation by growth zone.

    Canonical mode computes one P2/P98-scaled structure-tensor field over the
    final detector signal and aggregates coherence-weighted metrics over the
    shared Method B zones. Overall begins at the operational CoreZone boundary
    and ends at the configured target-mask radial percentile; it is independent
    of ``SymmetricRadius``. Absolute concentration, turning, and coherence retain
    both ``Radial`` and raw ``Mask`` variants. Radial-relative tilt and outward
    turning use detected structure and equal-weight reliable angular sectors.
    Their point estimates are count-scale-invariant while reliable-sector
    membership is unchanged; a separate support diagnostic exposes threshold
    crossings. Longer-range rotation summarizes signed radial-relative tilt in
    fixed-width Sholl-style annular bands, compares matching sectors across a
    configurable radial lag, and separately reports the broad Dense-to-Sparse
    change. Emits the
    :class:`~phenotypic.schema.ORIENTATION_ZONE_PRIMARY` columns. Set
    ``include_diagnostics=True`` to also emit the validation, comparator, and
    legacy :class:`~phenotypic.schema.ORIENTATION_ZONE_DIAGNOSTIC` columns.

    Args:
        center_detector: ObjectDetector or ImagePipeline that produces compact
            center regions. The default pipeline resets ``detect_mat`` to
            grayscale and applies ``InoculumDetector(min_diameter=20,
            max_diameter=140, thresh_method="otsu")``. In canonical mode, each
            detected component is associated with the final colony it overlaps
            and its overlap centroid becomes the shared radial origin. Set to
            ``None`` to use the final-mask distance-transform estimator.
            Ignored in legacy mode.
        legacy_mode: Use the historical colony-ness zone partition instead of
            canonical Method B. Defaults to ``False``.
        outer_zone_percentile: Target-mask radial percentile used as the
            canonical SparseZone outer boundary. ``100`` uses full extent.
        sigma_d: Gaussian-derivative scale in pixels, approximately hypha width.
        sigma_i: Structure-tensor integration scale in pixels.
        radial_ring_width: Width in pixels of each center-origin Sholl-style
            annular band.
        zone_minimum_segment: Minimum ring count in every change-point segment.
        zone_min_crossings: Minimum literal crossings for ring support.
        zone_min_resultant: Minimum ring-level axial resultant for support.
        zone_min_ring_coherence: Minimum reliable-pixel mean coherence for
            ring support.
        zone_support_weight: Weight of the Boolean orientation-support feature.
        zone_outer_support_margin: Minimum outer-minus-inner support fraction
            accepted at the first Method B boundary.
        zone_maximum_gap: Largest interior unsupported ring run bridged before
            change-point fitting.
        intensity_source: Image array for the structure tensor and zone
            segmentation in legacy mode. Canonical mode always uses
            ``detect_mat``.
        long_range_lag: Centre-to-centre radial comparison distance in pixels.
            Must be an integer multiple of ``radial_ring_width``.
        outward_peak_window_rings: Odd number of consecutive literal-crossing
            rings used by the sustained-peak rolling median.
        outward_min_run_rings: Minimum contiguous supported-ring count for the
            robust net, rate, and consistency metrics.
        include_diagnostics: Emit validation comparators, quality support, raw
            peak, rate-gradient, and legacy orientation-zone columns. Defaults
            to ``False`` so only primary outward-rotation metrics are returned.
        n_annuli: Number of equal-area annuli in the shared zone segmentation.
        pelt_penalty: PELT penalty controlling core-changepoint sensitivity.
        symmetry_threshold: Minimum angular coverage for symmetric growth.
        n_angular_bins: Number of angular bins for the coverage diagnostic.
        smoothing_window: Moving-average window (annuli) for the coverage test.
        method: Inoculum-centre estimator (``"distance"`` or ``"intensity"``).
        extent_margin: Fractional expansion of the analysis tile past the mask.
        min_samples_per_ring: Minimum pixel count per ring before interpolation.
        tau_core: Colony-ness threshold for the core/dense boundary.
        tau_dense: Colony-ness threshold for the dense/sparse boundary.
        tau_sparse: Colony-ness threshold for the sparse/outside boundary.
        quiver_block: inspect() quiver downsample block size in pixels. Plotting
            parameters follow the measurement parameters in the public schema.

    Examples:
        >>> from phenotypic.data import load_synth_filamentous_plate
        >>> from phenotypic.measure import MeasureOrientationZones
        >>> image = load_synth_filamentous_plate()
        >>> df = MeasureOrientationZones().measure(image)
        >>> 'OrientZones_OutwardRotationRate-Mask-Overall' in df.columns
        True
    """

    _measurement_infoclass: ClassVar[type] = ORIENTATION_ZONE_PRIMARY
    _measurement_infoclasses: ClassVar[list[type]] = [
        ORIENTATION_ZONE_DIAGNOSTIC
    ]

    long_range_lag: float = 16.0
    outward_peak_window_rings: int = 3
    outward_min_run_rings: int = 6
    include_diagnostics: bool = False
    # Plot-only fields stay after every measurement field so constructor,
    # schema, and GUI form order follow the operation's functional parameters.
    quiver_block: int = 12
    # Per-object figure intermediates, populated by _operate. PrivateAttr keeps
    # it out of model_dump()/JSON (mirrors MeasureSymZones' cache pattern).
    _cache: dict = PrivateAttr(default_factory=dict)
    _cache_image_ref: "weakref.ReferenceType[object] | None" = PrivateAttr(
            default=None
    )
    _cache_signature: str | None = PrivateAttr(default=None)

    def get_measurement_infoclasses(
            self,
    ) -> tuple[type[MeasurementInfo], ...]:
        """Return primary and optionally diagnostic orientation schemas."""
        infos = super().get_measurement_infoclasses()
        if self.include_diagnostics:
            return infos
        return tuple(
                info
                for info in infos
                if info is not ORIENTATION_ZONE_DIAGNOSTIC
        )

    @field_validator("long_range_lag")
    @classmethod
    def _positive_radial_scale(cls, value: float) -> float:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                    "radial_ring_width and long_range_lag must be finite and > 0"
            )
        return value

    @field_validator("outward_peak_window_rings", mode="before")
    @classmethod
    def _valid_outward_peak_window(cls, value):
        if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 3
                or value % 2 == 0
        ):
            raise ValueError(
                    "outward_peak_window_rings must be an odd integer >= 3"
            )
        return int(value)

    @field_validator("outward_min_run_rings", mode="before")
    @classmethod
    def _valid_outward_minimum_run(cls, value):
        if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 3
        ):
            raise ValueError("outward_min_run_rings must be an integer >= 3")
        return int(value)

    @model_validator(mode="after")
    def _validate_long_range_scales(self):
        ratio = self.long_range_lag / self.radial_ring_width
        if ratio < 1.0 or not np.isclose(
                ratio,
                round(ratio),
                atol=1e-9,
                rtol=0.0,
        ):
            raise ValueError(
                    "long_range_lag must be an integer multiple of "
                    "radial_ring_width"
            )
        return self

    def _resolve_tile(self, image, seg: ZoneSegmentation, prop, label2section):
        """Return (tile_intensity, obj_mask_tile, centre_rc) for one object.

        Preferred: the object's **grid section** via ``image.grid[idx]`` — an
        object-aware cropped Image (only this object's label survives; the crop
        preserves the complete object, so it is a superset of the object's
        pixels). Verified API: ``image.grid[section_idx]`` returns a cropped
        ``Image``; the crop origin is recovered by the public exact identity
        ``origin = prop.centroid(full) - regionprops(section)[label].centroid``.
        Falls back to the mask-free expanded crop when the image is not a
        GridImage, the section lookup fails, or the section does not cover the
        r_max disk around the centre (crowded/overgrown plate).
        """
        from skimage.measure import regionprops

        min_row, min_col, max_row, max_col = prop.bbox
        object_radius_bound = max(
                np.hypot(
                        row - seg.centroid_global[0], col - seg.centroid_global[1]
                )
                for row, col in (
                    (min_row, min_col),
                    (min_row, max_col),
                    (max_row, min_col),
                    (max_row, max_col),
                )
        )
        r_max = max(
                max(seg.sparse_end_radius, seg.symmetric_radius)
                * (1 + self.extent_margin),
                object_radius_bound + self.radial_ring_width,
        )
        if hasattr(image, "grid") and seg.label in label2section:
            try:
                section = image.grid[label2section[seg.label]]
                sec_props = {
                    p.label: p for p in regionprops(section.objmap[:])
                }
                sp = sec_props.get(seg.label)
                if sp is not None:
                    origin = (
                        prop.centroid[0] - sp.centroid[0],
                        prop.centroid[1] - sp.centroid[1],
                    )
                    centre = (
                        seg.centroid_global[0] - origin[0],
                        seg.centroid_global[1] - origin[1],
                    )
                    H, W = section.objmap[:].shape[:2]
                    if (
                            centre[0] - r_max >= 0
                            and centre[0] + r_max <= H
                            and centre[1] - r_max >= 0
                            and centre[1] + r_max <= W
                    ):
                        tile = np.asarray(
                                getattr(section, self.intensity_source)[:],
                                dtype=np.float64,
                        )
                        return tile, (section.objmap[:] == seg.label), centre
            except (KeyError, IndexError, ValueError, AttributeError):
                pass
        # Fallback: expanded crop on the full plate (non-grid / clipped section).
        hw = image.gray[:].shape[:2]  # 2-tuple; image.shape is (H,W,3) for RGB
        sl = expand_slice_around_center(seg.centroid_global, r_max, hw)
        tile = np.asarray(
                getattr(image, self.intensity_source)[sl], dtype=np.float64
        )
        obj_mask = image.objmap[:][sl] == seg.label
        centre = (
            seg.centroid_global[0] - sl[0].start,
            seg.centroid_global[1] - sl[1].start,
        )
        return tile, obj_mask, centre

    def _zone_bounds(self, seg: ZoneSegmentation):
        if not self.legacy_mode:
            return {
                "Overall": (seg.core_end_radius, seg.sparse_end_radius),
                "Dense"  : (seg.core_end_radius, seg.dense_end_radius),
                "Sparse" : (seg.dense_end_radius, seg.sparse_end_radius),
            }
        return {
            "Overall": (0.0, seg.symmetric_radius),
            "Dense"  : (seg.core_end_radius, seg.dense_end_radius),
            "Sparse" : (seg.dense_end_radius, seg.sparse_end_radius),
        }

    def _orientation_outer_radius(self, seg: ZoneSegmentation) -> float:
        """Return the shared outer selector while preserving legacy bounds."""
        if not self.legacy_mode:
            return float(seg.sparse_end_radius)
        return min(float(seg.sparse_end_radius), float(seg.symmetric_radius))

    def _lacks_zone_evidence(self, seg: ZoneSegmentation) -> bool:
        """True for a canonical failure, which figure rasters must skip.

        Its stand-in orientation field can be ``(1, 1)``, too small for
        ``np.gradient``, and its NaN radii select no pixels anyway. This is the
        same predicate :meth:`_fill_metrics` uses to leave zone metrics empty.
        """
        return not seg.zones_computed and not self.legacy_mode

    def _prep(self, image):
        """Regionprops + label→grid-section map, computed ONCE per image.

        grid.info() is slow on filamentous plates, so never call it per object.
        intensity_image is required so compute_zone_segmentation can read
        prop.centroid_weighted when method="intensity" (else AttributeError).
        """
        from skimage.measure import regionprops
        from phenotypic.schema import GRID

        props = regionprops(
                image.objmap[:],
                intensity_image=image.gray[:].astype(np.float64, copy=False),
        )
        label2section = {}
        if hasattr(image, "grid"):
            info = image.grid.info()
            lab, rmi = str(OBJECT.LABEL), str(GRID.ROW_MAJOR_IDX)
            label2section = dict(
                    zip(info[lab].astype(int), info[rmi].astype(int))
            )
        return props, label2section

    def _analyze_objects(self, image, props, label2section):
        """Yield named zone and orientation evidence for each object.

        This is the single source of truth for the heavy orientation compute,
        reused by
        _operate() (which keeps only compact summaries) and by report()'s
        coherence panel (which recomputes on demand). The full-resolution arrays
        yielded here are consumed and discarded by each caller; nothing full-res
        is retained on the instance. Legacy mode skips tiny objects; canonical
        mode yields their shared missing resolution so diagnostics report code 4.
        """
        for prop in props:
            if prop.area < 10 and self.legacy_mode:
                continue
            resolution = self._resolve_object_zones(image, prop)
            seg = resolution.segmentation
            context = resolution.orientation_context
            if context is None:
                # Canonical failures have no orientation context. Preserve the
                # object's geometry for diagnostics, but keep every field empty
                # so no zone metric can be emitted accidentally.
                if not self.legacy_mode and not seg.zones_computed:
                    obj_mask = np.asarray(seg.obj_mask, dtype=bool)
                    centre = tuple(seg.centroid_rc)
                    dist_map = np.asarray(seg.dist_map, dtype=np.float64)
                    phi = np.zeros(obj_mask.shape, dtype=np.float64)
                    coh = np.zeros(obj_mask.shape, dtype=np.float64)
                    grad = np.zeros(obj_mask.shape, dtype=np.float64)
                else:
                    # Legacy segmentation does not retain tensor evidence, so
                    # compute the historical orientation field on its tile.
                    tile, obj_mask, centre = self._resolve_tile(
                            image, seg, prop, label2section
                    )
                    phi, coh, grad = orientation_field(
                            tile, self.sigma_d, self.sigma_i
                    )
                    dist_map = distance_from_point(tile.shape, centre)
            else:
                obj_mask = context.object_mask
                centre = context.center
                phi = context.phi
                coh = context.coherence
                grad = context.gradient
                dist_map = context.distance_map
            yield _ObjectZoneAnalysis(
                prop=prop,
                resolution=resolution,
                object_mask=obj_mask,
                orientation=phi,
                coherence=coh,
                gradient=grad,
                distance_map=dist_map,
                center=centre,
            )

    def _operate(self, image) -> pd.DataFrame:  # type: ignore[override]
        props, label2section = self._prep(image)
        headers = ORIENTATION_ZONE_PRIMARY.get_headers()
        if self.include_diagnostics:
            headers = [
                *headers,
                *ORIENTATION_ZONE_DIAGNOSTIC.get_headers(),
            ]
        # pre-seed every object's row with NaN so skipped/failed objects still appear
        base: dict[int, dict] = {}
        for prop in props:
            r: dict = {OBJECT.LABEL: prop.label}
            r.update({h: np.nan for h in headers})
            base[prop.label] = r
        self._cache.clear()  # compact per-object figure records only
        # Preserve no-argument notebook figures without retaining a full Image.
        self._cache_image_ref = weakref.ref(image)
        for analysis in self._analyze_objects(image, props, label2section):
            prop = analysis.prop
            seg = analysis.segmentation
            resolution = analysis.resolution
            self._write_zone_resolution_diagnostics(
                base[prop.label], resolution
            )
            (
                outward_rotation,
                per_zone,
                radial_relative,
                long_range,
                ring_profile,
            ) = self._fill_metrics(
                    base[prop.label],
                    seg,
                    analysis.object_mask,
                    analysis.orientation,
                    analysis.coherence,
                    analysis.gradient,
                    analysis.distance_map,
                    analysis.center,
                    resolution=resolution,
            )
            # LEAN CACHE: store compact summaries only — NO full-res tile/phi/coh/
            # grad/dist_map and NO seg dataclass. Bounds memory to O(objects*blocks).
            self._cache[prop.label] = {
                "centroid_global" : tuple(seg.centroid_global),
                "centre"          : analysis.center,
                "radii"           : {
                    "core"      : seg.core_radius,
                    "symmetric" : seg.symmetric_radius,
                    "core_end"  : seg.core_end_radius,
                    "dense_end" : seg.dense_end_radius,
                    "sparse_end": seg.sparse_end_radius,
                },
                "zones_computed"  : seg.zones_computed,
                "zone_resolution" : {
                    "method_code": resolution.method_code,
                    "method_used": resolution.method_used,
                    "failure_reason": resolution.failure_reason,
                    "outer_zone_percentile": self.outer_zone_percentile,
                    "full_extent_radius": (
                        resolution.canonical_result.full_extent_radius
                        if resolution.canonical_result is not None
                        else np.nan
                    ),
                },
                "outward_rotation": outward_rotation,
                "quiver"          : self._downsample_quiver(
                        analysis.orientation,
                        analysis.coherence,
                        self.quiver_block,
                ),  # block-res
                "per_zone"        : per_zone,
                "radial_relative" : radial_relative,
                "long_range"      : long_range,
                "ring_profile"    : ring_profile,
            }
        self._cache_signature = self.model_dump_json()
        return pd.DataFrame(
                [base[p.label] for p in props], columns=[OBJECT.LABEL, *headers]
        )

    def _write_diagnostic(self, row: dict, header: str, value: float) -> None:
        """Write one opt-in diagnostic value."""
        if self.include_diagnostics:
            row[header] = value

    def _write_zone_resolution_diagnostics(self, row: dict, resolution) -> None:
        """Write compact provenance for the shared zone resolver."""
        if not self.include_diagnostics:
            return
        row["OrientZones_ZoneSegmentationMethodCode"] = float(
            resolution.method_code
        )
        result = resolution.canonical_result
        values = {
            "CoreZoneEndRadius": np.nan,
            "DenseRadius": np.nan,
            "OuterRadius": np.nan,
            "FullExtentRadius": np.nan,
            "OuterZonePercentile": np.nan,
            "OuterZoneRetainedMaskFraction": np.nan,
            "ZoneSupportedRingFraction": np.nan,
            "ZoneChangePointObjective": np.nan,
            "ZoneChangePointRingCount": np.nan,
            "ZoneChangePointMinimumSegment": np.nan,
        }
        if result is not None:
            values.update(
                {
                    "CoreZoneEndRadius": result.core_zone_radius,
                    "DenseRadius": result.dense_radius,
                    "OuterRadius": result.outer_radius,
                    "FullExtentRadius": result.full_extent_radius,
                    "OuterZonePercentile": result.requested_percentile,
                    "OuterZoneRetainedMaskFraction": (
                        result.retained_mask_fraction
                    ),
                    "ZoneSupportedRingFraction": result.supported_fraction,
                    "ZoneChangePointObjective": result.objective,
                    "ZoneChangePointRingCount": float(result.ring_count),
                    "ZoneChangePointMinimumSegment": float(
                        self.zone_minimum_segment
                    ),
                }
            )
        for label, value in values.items():
            row[f"OrientZones_{label}"] = value

    def _fill_literal_crossing_metrics(
            self,
            row: dict,
            seg: ZoneSegmentation,
            obj_mask: np.ndarray,
            phi: np.ndarray,
            coherence: np.ndarray,
            dist_map: np.ndarray,
            centre: tuple[float, float],
            *,
            resolution=None,
    ) -> dict[str, dict[str, float]]:
        """Write full-length literal-crossing primary and diagnostic metrics.

        The complete detected object is skeletonized before the inoculum
        selector is applied. Ring centers cover the complete detected radial
        extent and are never trimmed to the symmetric radius.
        """

        primary_by_zone = {
            zone: {metric: np.nan for metric in _PRIMARY_OUTWARD_METRICS}
            for zone in _ZONES
        }

        def _write_missing() -> dict[str, dict[str, float]]:
            for zone in _ZONES:
                for metric in _PRIMARY_OUTWARD_METRICS:
                    row[f"OrientZones_{metric}-Mask-{zone}"] = np.nan
                for metric in _DIAGNOSTIC_OUTWARD_METRICS:
                    self._write_diagnostic(
                            row,
                            f"OrientZones_{metric}-Mask-{zone}",
                            np.nan,
                    )
            return primary_by_zone

        if not seg.zones_computed:
            return _write_missing()
        inner_radius = float(seg.core_end_radius)
        context = (
            resolution.orientation_context
            if resolution is not None
            else None
        )
        if context is not None:
            outer_radius = float(seg.sparse_end_radius)
            profile = context.measurement_profile
            inclusive_outer = np.nextafter(outer_radius, np.inf)
            bounds = {
                "Overall": (inner_radius, inclusive_outer),
                "Dense"  : (inner_radius, float(seg.dense_end_radius)),
                "Sparse" : (float(seg.dense_end_radius), inclusive_outer),
            }
        else:
            outside_core = (
                    obj_mask & np.isfinite(dist_map) & (dist_map >= inner_radius)
            )
            if not outside_core.any():
                return _write_missing()

            object_extent_radius = float(np.max(dist_map[outside_core]))
            radial_span = np.nextafter(object_extent_radius, np.inf) - inner_radius
            n_rings = max(
                    1,
                    int(np.ceil(radial_span / self.radial_ring_width)),
            )
            outer_radius = inner_radius + n_rings * self.radial_ring_width
            radii = (
                    inner_radius
                    + (np.arange(n_rings, dtype=np.float64) + 0.5)
                    * self.radial_ring_width
            )
            selector = self._zone_selector(
                    dist_map,
                    inner_radius,
                    outer_radius,
                    obj_mask,
                    "Mask",
                    include_upper=not self.legacy_mode,
            )
            transform = literal_skeleton_ring_crossings(
                    obj_mask,
                    phi + FIBER_AXIS_OFFSET,
                    coherence,
                    dist_map,
                    centre,
                    radii,
                    selector=selector,
                    minimum_coherence=RELIABLE_PIXEL_COHERENCE,
                    crossing_half_width=CROSSING_HALF_WIDTH,
                    minimum_crossing_resultant=MIN_AXIAL_RESULTANT,
            )
            profile = literal_crossing_ring_profile(
                    transform,
                    minimum_points=_LITERAL_CROSSING_MIN_POINTS,
                    minimum_resultant=MIN_AXIAL_RESULTANT,
            )
            bounds = {
                "Overall": (inner_radius, outer_radius),
                "Dense"  : (
                    inner_radius,
                    min(float(seg.dense_end_radius), outer_radius),
                ),
                "Sparse" : (
                    max(float(seg.dense_end_radius), inner_radius),
                    outer_radius,
                ),
            }
        for zone, (lower, upper) in bounds.items():
            metrics = aggregate_literal_crossing_zone(
                    profile,
                    lower,
                    upper,
                    peak_window_rings=self.outward_peak_window_rings,
                    minimum_run_rings=self.outward_min_run_rings,
            )
            primary_values = {
                "OutwardRotationSustainedPeak": float(
                        np.degrees(metrics.sustained_peak)
                ),
                "OutwardRotationNet"          : float(np.degrees(metrics.net_rotation)),
                "OutwardRotationRate"         : float(
                        np.degrees(metrics.rotation_rate)
                ),
                "OutwardRotationConsistency"  : metrics.consistency,
            }
            for metric, value in primary_values.items():
                row[f"OrientZones_{metric}-Mask-{zone}"] = value
            primary_by_zone[zone] = primary_values

            diagnostic_values = {
                "OutwardRotationRawPeak"        : float(np.degrees(metrics.raw_peak)),
                "OutwardRotationP90"            : float(
                    np.degrees(metrics.percentile_90)),
                "OutwardRotationP95"            : float(
                    np.degrees(metrics.percentile_95)),
                "OutwardRotationMedianMagnitude": float(
                        np.degrees(metrics.median_magnitude)
                ),
                "OutwardRotationAbsoluteArea"   : float(
                        np.degrees(metrics.absolute_area)
                ),
                "OutwardRotationTotalVariation" : float(
                        np.degrees(metrics.total_variation)
                ),
                "OutwardRotationRateGradient"   : float(
                        np.degrees(metrics.rate_gradient)
                ),
                "OutwardRotationRingSupport"    : metrics.ring_support,
                "OutwardRotationRunSpanSupport" : metrics.run_span_support,
                "OutwardRotationMedianResultant": metrics.median_resultant,
            }
            for metric, value in diagnostic_values.items():
                self._write_diagnostic(
                        row,
                        f"OrientZones_{metric}-Mask-{zone}",
                        value,
                )
        return primary_by_zone

    def _fill_metrics(
            self,
            row,
            seg,
            obj_mask,
            phi,
            coh,
            grad,
            dist_map,
            centre,
            *,
            resolution=None,
    ):
        """Write public degree-based zone columns for one object.

        The structure-tensor and axial calculations remain in radians. Angular
        values are converted only at this output/cache boundary so exported
        measurements and diagnostic figures use degrees consistently.
        """
        per_zone = {}
        radial_relative = {}
        outward_rotation = self._fill_literal_crossing_metrics(
                row,
                seg,
                obj_mask,
                phi,
                coh,
                dist_map,
                centre,
                resolution=resolution,
        )
        if seg.zones_computed or self.legacy_mode:
            signed_tilt, _signed_turning, outward_turning, polar_angle = (
                signed_radial_relative_field(phi, centre, dist_map)
            )
        else:
            signed_tilt = np.full(phi.shape, np.nan, dtype=np.float64)
            outward_turning = np.full(phi.shape, np.nan, dtype=np.float64)
            polar_angle = np.full(phi.shape, np.nan, dtype=np.float64)
        absolute_tilt = np.abs(signed_tilt)
        for zone, (r_lo, r_hi) in self._zone_bounds(seg).items():
            zone_ok = seg.zones_computed or (
                self.legacy_mode and zone == "Overall"
            )
            for variant in _VARIANTS:
                if not zone_ok:
                    R = t = cm = direction = np.nan
                else:
                    sel = self._zone_selector(
                            dist_map,
                            r_lo,
                            r_hi,
                            obj_mask,
                            variant,
                            include_upper=(
                                not self.legacy_mode
                                and zone in {"Overall", "Sparse"}
                            ),
                    )
                    R, t, cm = self._aggregate_orientation(phi, coh, grad, sel)
                    direction = self._resultant_direction(phi, coh, sel)
                turning_degrees = float(np.degrees(t))
                per_zone[(variant, zone)] = (
                    R,
                    turning_degrees,
                    cm,
                    direction,
                )  # scalars only
                self._write_diagnostic(
                        row,
                        f"OrientZones_Concentration-{variant}-{zone}",
                        R,
                )
                self._write_diagnostic(
                        row,
                        f"OrientZones_Turning-{variant}-{zone}",
                        turning_degrees,
                )
                self._write_diagnostic(
                        row,
                        f"OrientZones_Coherence-{variant}-{zone}",
                        cm,
                )
            valid_bounds = (
                    np.isfinite(r_lo) and np.isfinite(r_hi) and r_hi > r_lo
            )
            if not zone_ok or not valid_bounds:
                radial_tilt = radial_turning = radial_support = np.nan
            else:
                structure_selector = self._zone_selector(
                        dist_map,
                        r_lo,
                        r_hi,
                        obj_mask,
                        "Mask",
                        include_upper=(
                            not self.legacy_mode
                            and zone in {"Overall", "Sparse"}
                        ),
                )
                radial_tilt, radial_turning, radial_support = (
                    self._aggregate_radial_relative(
                            absolute_tilt,
                            outward_turning,
                            polar_angle,
                            coh,
                            dist_map,
                            structure_selector,
                            N_SECTORS,
                    )
                )
            radial_tilt_degrees = float(np.degrees(radial_tilt))
            radial_turning_degrees = float(np.degrees(radial_turning))
            radial_relative[zone] = (
                radial_tilt_degrees,
                radial_turning_degrees,
                radial_support,
            )
            self._write_diagnostic(
                    row,
                    f"OrientZones_RadialTilt-Mask-{zone}",
                    radial_tilt_degrees,
            )
            self._write_diagnostic(
                    row,
                    f"OrientZones_OutwardTurning-Mask-{zone}",
                    radial_turning_degrees,
            )
            self._write_diagnostic(
                    row,
                    f"OrientZones_RadialSectorSupport-Mask-{zone}",
                    radial_support,
            )
        long_range, ring_profile = self._fill_long_range_metrics(
                row,
                seg,
                obj_mask,
                signed_tilt,
                polar_angle,
                coh,
                dist_map,
        )
        return (
            outward_rotation,
            per_zone,
            radial_relative,
            long_range,
            ring_profile,
        )

    def _fill_long_range_metrics(
            self,
            row,
            seg,
            obj_mask,
            signed_tilt,
            polar_angle,
            coherence,
            dist_map,
    ):
        """Write fixed-lag Sholl-style rotation metrics in public degrees.

        Complete annular bands start at the inferred inoculum boundary. Ring
        pairs are compared at ``long_range_lag`` and assigned to a radial zone
        by their midpoint. The separate Dense-to-Sparse result compares broad
        coherence-weighted sector means rather than individual ring pairs.

        Returns:
            ``(long_range, ring_profile)`` compact dictionaries for figures.
        """
        long_range = {zone: (np.nan, np.nan, np.nan) for zone in _ZONES}
        long_range["DenseToSparse"] = (np.nan, np.nan, np.nan)
        empty = np.empty(0, dtype=np.float64)
        ring_profile = {
            "radii"                 : empty,
            "mean_absolute_tilt"    : empty.copy(),
            "mean_signed_tilt"      : empty.copy(),
            "support"               : empty.copy(),
            "pair_midpoints"        : empty.copy(),
            "mean_absolute_rotation": empty.copy(),
            "mean_signed_rotation"  : empty.copy(),
            "pair_support"          : empty.copy(),
        }

        metric_names = (
            "LongRangeRotation",
            "SignedLongRangeRotation",
            "LongRangeRotationSupport",
        )

        def _write_result(
                name: str, result: tuple[float, float, float]
        ) -> None:
            magnitude, signed, support = result
            magnitude_degrees = float(np.degrees(magnitude))
            signed_degrees = float(np.degrees(signed))
            public = (magnitude_degrees, signed_degrees, support)
            long_range[name] = public
            values = (magnitude_degrees, signed_degrees, support)
            for metric, value in zip(metric_names, values):
                self._write_diagnostic(
                        row,
                        f"OrientZones_{metric}-Mask-{name}",
                        value,
                )

        if not seg.zones_computed:
            for name in (*_ZONES, "DenseToSparse"):
                _write_result(name, (np.nan, np.nan, np.nan))
            return long_range, ring_profile

        inner_radius = float(seg.core_end_radius)
        outer_radius = self._orientation_outer_radius(seg)
        structure_selector = self._zone_selector(
                dist_map,
                inner_radius,
                outer_radius,
                obj_mask,
                "Mask",
                include_upper=not self.legacy_mode,
        )
        ring_centres, ring_sector_tilt, _ring_resultant = (
            radial_ring_orientation_profile(
                    signed_tilt,
                    polar_angle,
                    coherence,
                    dist_map,
                    structure_selector,
                    inner_radius,
                    outer_radius,
                    self.radial_ring_width,
                    N_SECTORS,
                    include_outer=not self.legacy_mode,
            )
        )
        pair_midpoints, signed_rotation = long_range_ring_rotation_profile(
                ring_centres,
                ring_sector_tilt,
                self.long_range_lag,
        )
        long_range_bounds = {
            "Overall": (
                inner_radius,
                np.nextafter(outer_radius, np.inf)
                if not self.legacy_mode
                else outer_radius,
            ),
            "Dense"  : (
                float(seg.core_end_radius),
                min(float(seg.dense_end_radius), outer_radius),
            ),
            "Sparse" : (
                float(seg.dense_end_radius),
                np.nextafter(outer_radius, np.inf)
                if not self.legacy_mode
                else outer_radius,
            ),
        }
        for zone, (lower, upper) in long_range_bounds.items():
            result = self._aggregate_long_range_rotation(
                    pair_midpoints,
                    signed_rotation,
                    lower,
                    upper,
            )
            _write_result(zone, result)

        dense_selector = self._zone_selector(
                dist_map,
                float(seg.core_end_radius),
                float(seg.dense_end_radius),
                obj_mask,
                "Mask",
        )
        sparse_selector = self._zone_selector(
                dist_map,
                float(seg.dense_end_radius),
                outer_radius,
                obj_mask,
                "Mask",
                include_upper=not self.legacy_mode,
        )
        dense_sector_tilt, _dense_resultant = axial_sector_means(
                signed_tilt,
                polar_angle,
                coherence,
                dense_selector,
                N_SECTORS,
        )
        sparse_sector_tilt, _sparse_resultant = axial_sector_means(
                signed_tilt,
                polar_angle,
                coherence,
                sparse_selector,
                N_SECTORS,
        )
        transition = self._aggregate_paired_zone_rotation(
                dense_sector_tilt,
                sparse_sector_tilt,
        )
        _write_result("DenseToSparse", transition)

        def _summarize_cells(
                cells: np.ndarray,
                signed_mean=lambda values: float(np.mean(values)),
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            absolute = np.full(cells.shape[0], np.nan, dtype=np.float64)
            signed = np.full(cells.shape[0], np.nan, dtype=np.float64)
            support = np.zeros(cells.shape[0], dtype=np.float64)
            for index, values in enumerate(cells):
                finite = np.isfinite(values)
                support[index] = float(finite.sum()) / float(values.size)
                if finite.any():
                    absolute[index] = float(np.mean(np.abs(values[finite])))
                    signed[index] = signed_mean(values[finite])
            return absolute, signed, support

        ring_absolute, ring_signed, ring_support = _summarize_cells(
                ring_sector_tilt
        )
        # Rotations use the same directionless rule as the measured
        # SignedLongRangeRotation, so the figure agrees with the table.
        pair_absolute, pair_signed, pair_support = _summarize_cells(
                signed_rotation, signed_mean=signed_axial_mean
        )
        ring_profile = {
            "radii"                 : ring_centres,
            "mean_absolute_tilt"    : np.degrees(ring_absolute),
            "mean_signed_tilt"      : np.degrees(ring_signed),
            "support"               : ring_support,
            "pair_midpoints"        : pair_midpoints,
            "mean_absolute_rotation": np.degrees(pair_absolute),
            "mean_signed_rotation"  : np.degrees(pair_signed),
            "pair_support"          : pair_support,
        }
        return long_range, ring_profile

    # ── operation-private numeric helpers ────────────────────────────

    @staticmethod
    def _zone_selector(
        dist_map,
        r_lo,
        r_hi,
        obj_mask,
        variant,
        *,
        include_upper=False,
    ):
        """Boolean selector for a radial zone on a tile; ``Mask`` also ∩ obj_mask.

        Args:
            dist_map: Per-pixel distance-from-centre map (tile shape).
            r_lo: Inner radius (inclusive) of the zone in pixels.
            r_hi: Outer radius of the zone in pixels.
            obj_mask: Boolean object mask (tile shape) used by the ``Mask`` variant.
            variant: ``"Radial"`` (all tile pixels in the ring) or ``"Mask"``
                (the ring intersected with ``obj_mask``).
            include_upper: Include pixels exactly on ``r_hi``. Internal zone
                boundaries remain half-open; canonical global outer boundaries
                use this option.

        Returns:
            Boolean array (tile shape). All-False when the radius range is invalid
            (non-finite or ``r_hi <= r_lo``).
        """
        if not np.isfinite(r_lo) or not np.isfinite(r_hi) or r_hi <= r_lo:
            return np.zeros(dist_map.shape, dtype=bool)
        upper = np.nextafter(r_hi, np.inf) if include_upper else r_hi
        radial = (dist_map >= r_lo) & (dist_map < upper)
        if variant == "Mask":
            return radial & obj_mask
        return radial

    @staticmethod
    def _aggregate_orientation(phi, coherence, grad_phi, selector, eps=_EPS):
        """Coherence-weighted (R, turning, mean-coherence) over a selector.

        Args:
            phi: Orientation field in radians (tile shape).
            coherence: Structure-tensor coherence in [0, 1] (tile shape).
            grad_phi: Orientation-gradient magnitude in rad/px (tile shape).
            selector: Boolean pixel selector (tile shape).
            eps: Numerical floor for the summed-coherence denominator.

        Returns:
            ``(R, turning, mean_coherence)`` scalars. Returns ``(nan, nan, nan)``
            when the selector is empty or ``sum(coherence) ~ 0``.
        """
        if not selector.any():
            return (np.nan, np.nan, np.nan)
        C = coherence[selector]
        sumC = float(C.sum())
        if sumC < eps:
            return (np.nan, np.nan, np.nan)
        c2 = np.cos(2.0 * phi[selector])
        s2 = np.sin(2.0 * phi[selector])
        Rx = float((C * c2).sum()) / sumC
        Ry = float((C * s2).sum()) / sumC
        R = float(np.hypot(Rx, Ry))
        turning = float((C * grad_phi[selector]).sum()) / sumC
        return (R, turning, float(C.mean()))

    @staticmethod
    def _aggregate_radial_relative(
            absolute_tilt: np.ndarray,
            outward_turning: np.ndarray,
            polar_angle: np.ndarray,
            coherence: np.ndarray,
            dist_map: np.ndarray,
            selector: np.ndarray,
            n_angular_bins: int,
            eps: float = _EPS,
    ) -> tuple[float, float, float]:
        """Aggregate radial-relative metrics with equal occupied-sector weight.

        Pixels are coherence-weighted within fixed polar sectors. Occupied sectors
        are then averaged equally, preventing highly occupied sectors from
        dominating the phenotype. Within a fixed set of reliable sectors,
        multiplying evidence without changing the tilt distributions leaves the
        point estimate unchanged. A support-threshold crossing can add a newly
        reliable sector and therefore change the estimate. Mixed orientations within
        one sector remain pixel-weighted because this method deliberately does not
        identify individual branches.

        Args:
            absolute_tilt: Absolute axial fiber-to-radial difference in radians.
            outward_turning: Radial derivative magnitude in radians per pixel.
            polar_angle: Per-pixel polar position in radians.
            coherence: Structure-tensor coherence in ``[0, 1]``.
            dist_map: Per-pixel distance from the inoculum centre.
            selector: Boolean detected-structure selector for one radial zone.
            n_angular_bins: Number of equal polar sectors around the colony.
            eps: Numerical floor for sector coherence sums.

        Returns:
            ``(mean_absolute_tilt, mean_outward_turning, sector_support)``.
            ``sector_support`` is the fraction of all fixed angular sectors meeting
            the coherence and pixel-support thresholds. The two phenotype values
            are ``NaN`` when no sector is reliable; support is then zero.

        Raises:
            ValueError: If ``n_angular_bins`` is less than one or array shapes differ.
        """
        arrays = (
            absolute_tilt,
            outward_turning,
            polar_angle,
            coherence,
            dist_map,
            selector,
        )
        if any(array.shape != absolute_tilt.shape for array in arrays[1:]):
            raise ValueError(
                    "radial-relative arrays and selector must share one shape"
            )
        if n_angular_bins < 1:
            raise ValueError("n_angular_bins must be >= 1")

        valid = (
                selector
                & (dist_map > eps)
                & np.isfinite(absolute_tilt)
                & np.isfinite(outward_turning)
                & np.isfinite(coherence)
                & (coherence >= RELIABLE_PIXEL_COHERENCE)
        )
        if not valid.any():
            return (np.nan, np.nan, 0.0)

        angle01 = np.mod(polar_angle[valid], 2.0 * np.pi) / (2.0 * np.pi)
        sector_ids = np.minimum(
                (angle01 * n_angular_bins).astype(np.int64),
                n_angular_bins - 1,
        )
        weights = coherence[valid]
        tilts = absolute_tilt[valid]
        turns = outward_turning[valid]
        sector_tilts: list[float] = []
        sector_turns: list[float] = []
        for sector in np.unique(sector_ids):
            chosen = sector_ids == sector
            if int(chosen.sum()) < MIN_PIXELS_PER_SECTOR:
                continue
            sector_weights = weights[chosen]
            weight_sum = float(sector_weights.sum())
            if weight_sum <= eps:
                continue
            sector_tilts.append(
                    float(np.sum(sector_weights * tilts[chosen]) / weight_sum)
            )
            sector_turns.append(
                    float(np.sum(sector_weights * turns[chosen]) / weight_sum)
            )
        if not sector_tilts:
            return (np.nan, np.nan, 0.0)
        sector_support = len(sector_tilts) / float(n_angular_bins)
        return (
            float(np.mean(sector_tilts)),
            float(np.mean(sector_turns)),
            sector_support,
        )

    @staticmethod
    def _aggregate_long_range_rotation(
            pair_midpoints: np.ndarray,
            signed_rotation: np.ndarray,
            lower_radius: float,
            upper_radius: float,
    ) -> tuple[float, float, float]:
        """Aggregate fixed-lag ring rotations whose midpoint lies in one zone.

        Args:
            pair_midpoints: Radial midpoint of each ring pair in pixels.
            signed_rotation: Seam-safe axial changes in radians, shaped
                ``(n_pairs, n_sectors)``.
            lower_radius: Inclusive lower midpoint radius for the zone.
            upper_radius: Exclusive upper midpoint radius for the zone.

        Returns:
            ``(mean_absolute_rotation, mean_signed_rotation, paired_support)``.
            Rotation cells receive equal weight. Support is the fraction of all
            selected pair-sector cells that are reliable. A valid zone with no
            reliable cells returns ``(NaN, NaN, 0)``.

        Raises:
            ValueError: If the pair arrays are inconsistent.
        """
        pair_midpoints = np.asarray(pair_midpoints, dtype=np.float64)
        signed_rotation = np.asarray(signed_rotation, dtype=np.float64)
        if (
                signed_rotation.ndim != 2
                or signed_rotation.shape[0] != pair_midpoints.size
        ):
            raise ValueError("signed_rotation rows must match pair_midpoints")
        if (
                not np.isfinite(lower_radius)
                or not np.isfinite(upper_radius)
                or upper_radius <= lower_radius
        ):
            return (np.nan, np.nan, np.nan)
        selected_rows = (pair_midpoints >= lower_radius) & (
                pair_midpoints < upper_radius
        )
        if not selected_rows.any():
            return (np.nan, np.nan, 0.0)
        chosen = signed_rotation[selected_rows]
        finite = np.isfinite(chosen)
        support = float(finite.sum()) / float(chosen.size)
        if not finite.any():
            return (np.nan, np.nan, support)
        values = chosen[finite]
        return (
            float(np.mean(np.abs(values))),
            signed_axial_mean(values),
            support,
        )

    @staticmethod
    def _aggregate_paired_zone_rotation(
            inner_sector_tilt: np.ndarray,
            outer_sector_tilt: np.ndarray,
    ) -> tuple[float, float, float]:
        """Compare matching sector means between two broad radial zones.

        Args:
            inner_sector_tilt: Signed axial means for the inner zone in radians.
            outer_sector_tilt: Signed axial means for the outer zone in radians.

        Returns:
            ``(mean_absolute_rotation, mean_signed_rotation, paired_support)``.
            Support is the fraction of fixed sectors reliable in both zones.

        Raises:
            ValueError: If the arrays differ in shape or are not one-dimensional.
        """
        inner = np.asarray(inner_sector_tilt, dtype=np.float64)
        outer = np.asarray(outer_sector_tilt, dtype=np.float64)
        if inner.ndim != 1 or outer.shape != inner.shape:
            raise ValueError("paired zone-sector arrays must share one 1-D shape")
        valid = np.isfinite(inner) & np.isfinite(outer)
        support = float(valid.sum()) / float(inner.size) if inner.size else 0.0
        if not valid.any():
            return (np.nan, np.nan, support)
        delta = axial_change(outer[valid], inner[valid])
        return (
            float(np.mean(np.abs(delta))),
            signed_axial_mean(delta),
            support,
        )

    @staticmethod
    def _downsample_quiver(phi, coherence, block):
        """Block-mean the doubled-angle field → (rows, cols, phi_block, coh_block).

        Circular-averages cos2φ/sin2φ (coherence-weighted) and means coherence over
        block×block cells. Returns block-centre coords in the TILE frame plus per-block
        orientation and coherence — a few KB, the only array kept in the lean cache.

        Args:
            phi: Orientation field in radians (tile shape).
            coherence: Structure-tensor coherence in [0, 1] (tile shape).
            block: Block edge length in pixels.

        Returns:
            Tuple ``(rows, cols, phi_block, coh_block)`` of ``(nr, nc)`` arrays:
            block-centre row/col in tile coordinates, per-block orientation (NaN
            where the block coherence is ~0), and per-block mean coherence.
        """
        h, w = phi.shape
        block = max(1, int(block))
        nr, nc = max(h // block, 1), max(w // block, 1)
        rows = np.empty((nr, nc))
        cols = np.empty((nr, nc))
        pb = np.empty((nr, nc))
        cb = np.empty((nr, nc))
        c2, s2 = np.cos(2.0 * phi), np.sin(2.0 * phi)
        for i in range(nr):
            for j in range(nc):
                rsl, csl = (
                    slice(i * block, (i + 1) * block),
                    slice(j * block, (j + 1) * block),
                )
                cc = coherence[rsl, csl]
                rows[i, j], cols[i, j] = (
                    i * block + block / 2,
                    j * block + block / 2,
                )
                cb[i, j] = float(cc.mean())
                wsum = float(cc.sum())
                pb[i, j] = (
                    0.5
                    * np.arctan2(
                            (cc * s2[rsl, csl]).sum(), (cc * c2[rsl, csl]).sum()
                    )
                    if wsum > 1e-12
                    else np.nan
                )
        return rows, cols, pb, cb

    @staticmethod
    def _resultant_direction(phi, coherence, selector):
        """Coherence-weighted mean orientation over a selector (for the inspect glyph).

        Args:
            phi: Orientation field in radians (tile shape).
            coherence: Structure-tensor coherence in [0, 1] (tile shape).
            selector: Boolean pixel selector (tile shape).

        Returns:
            Mean orientation in radians, or NaN when the selector is empty or the
            summed coherence is ~0.
        """
        if not selector.any():
            return np.nan
        C = coherence[selector]
        if float(C.sum()) < _EPS:
            return np.nan
        return 0.5 * np.arctan2(
                float((C * np.sin(2.0 * phi[selector])).sum()),
                float((C * np.cos(2.0 * phi[selector])).sum()),
        )
