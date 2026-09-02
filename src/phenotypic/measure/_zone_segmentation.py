"""Shared mask morphology and radial-zone resolution for one detected object.

The private mask primitive supplies the distance-transform center, PELT core,
symmetry, and expansion measurements to both public measurers. Canonical mode
passes that geometry directly to Method B through ``resolve_zone_segmentation``;
explicit legacy mode extends it through ``compute_zone_segmentation`` and the
historical colony-ness thresholds. Legacy behavior is regression-guarded.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from scipy.ndimage import convolve, distance_transform_edt
from skimage.measure import regionprops

from phenotypic.measure._orientation_zone_segmentation import (
    OrientationAnalysisContext,
    OrientationChangePointParams,
    OrientationZoneResult,
    fit_orientation_zones,
)

# Zone-segmentation constants
_N_ANGULAR_SECTORS = 360
_ZONE_RADIAL_SMOOTHING = 3


@dataclass
class ZoneSegmentation:
    """Intermediate results from the symmetric-radius pipeline for one object."""

    label: int
    bbox_slice: tuple[slice, slice]
    centroid_rc: tuple[float, float]  # local bbox coords
    density_profile: np.ndarray  # for core detection + diagnostics
    annulus_radii: np.ndarray
    core_radius: float
    sholl_counts: np.ndarray  # boundary pixels per annulus
    angular_R_profile: np.ndarray  # per-annulus R̄, NaN where underpopulated
    angular_coverage: np.ndarray  # fraction of n_angular_bins filled
    symmetric_radius: float  # headline metric
    mean_expansion: float
    max_expansion: float
    obj_mask: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=bool))
    dist_map: np.ndarray = field(
            default_factory=lambda: np.zeros((1, 1), dtype=np.float64))
    gray_crop: np.ndarray = field(
            default_factory=lambda: np.zeros((1, 1), dtype=np.float64))
    # Scalar zone radii (concentric circles centred at ``centroid_rc``).
    core_end_radius: float = 0.0
    dense_end_radius: float = 0.0
    sparse_end_radius: float = 0.0
    # Per-angle mask envelope retained for the diagnostic overlay only;
    # does not drive zone segmentation.
    r_outer_full_per_angle: np.ndarray = field(
            default_factory=lambda: np.zeros(_N_ANGULAR_SECTORS, dtype=np.float64))
    core_area: float = 0.0
    dense_area: float = 0.0
    sparse_area: float = 0.0
    # 1D per-ring diagnostics.
    colony_ness_profile: np.ndarray = field(
            default_factory=lambda: np.zeros(1, dtype=np.float64))
    mean_profile: np.ndarray = field(
            default_factory=lambda: np.zeros(1, dtype=np.float64))
    variance_profile: np.ndarray = field(
            default_factory=lambda: np.zeros(1, dtype=np.float64))
    count_profile: np.ndarray = field(
            default_factory=lambda: np.zeros(1, dtype=np.int64))
    I_core: float = 0.0
    I_agar: float = 0.0
    zones_computed: bool = False
    # Plate-frame inoculum centre (frame origin of ``dist_map`` / ``obj_mask`` /
    # ``gray_crop`` is ``(bbox_slice[0].start, bbox_slice[1].start)``).
    centroid_global: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class ZoneSegmentationParams:
    """Parameters controlling the colony-ness → zone-radii pipeline."""
    n_annuli: int = 100
    pelt_penalty: float = 5.0
    symmetry_threshold: float = 4 / 6
    n_angular_bins: int = 6
    smoothing_window: int = 3
    method: str = "distance"
    extent_margin: float = 0.05
    min_samples_per_ring: int = 5
    tau_core: float = 0.9
    tau_dense: float = 0.5
    tau_sparse: float = 0.1
    intensity_source: str = "gray"


@dataclass(frozen=True)
class ZoneResolution:
    """Shared zone segmentation plus optional reusable orientation arrays."""

    segmentation: ZoneSegmentation
    method_code: int
    method_used: str
    failure_reason: str
    canonical_result: OrientationZoneResult | None = None
    orientation_context: OrientationAnalysisContext | None = None


# ── shared pipeline for one object ───────────────────────────────


def distance_from_point(
        shape: tuple[int, int], center_rc: tuple[float, float]
) -> np.ndarray:
    """Euclidean distance from each pixel to a point.

    Args:
        shape: (height, width) of the array.
        center_rc: (row, col) center coordinates.

    Returns:
        Float64 array of distances with the given shape.
    """
    rows, cols = np.indices(shape)
    return np.sqrt((rows - center_rc[0]) ** 2 + (cols - center_rc[1]) ** 2)


def expand_slice_around_center(
        center_global: tuple[float, float],
        r_max: float,
        image_shape: tuple[int, int],
) -> tuple[slice, slice]:
    """Expand a bbox slice to a disk of radius ``r_max`` around a centre.

    Used to build an analysis crop that extends a small margin past the
    farthest mask pixel so the outer annuli in the variance tensor
    sample an agar tail for the baseline-variance reference. The result
    is clipped to the image bounds.

    Args:
        center_global: ``(row, col)`` centre in full-image coordinates.
        r_max: Disk radius in pixels.
        image_shape: ``(H, W)`` of the full image.

    Returns:
        ``(row_slice, col_slice)`` mapping the disk-bounded region to
        plate coordinates.
    """
    h, w = image_shape
    r0 = max(0, int(np.floor(center_global[0] - r_max)))
    r1 = min(h, int(np.ceil(center_global[0] + r_max)) + 1)
    c0 = max(0, int(np.floor(center_global[1] - r_max)))
    c1 = min(w, int(np.ceil(center_global[1] + r_max)) + 1)
    return slice(r0, r1), slice(c0, c1)


def _resolve_target_prop(image: Image, prop):
    """Return the requested object or the largest detected object."""
    if prop is not None:
        return prop
    props = regionprops(
        image.objmap[:],
        intensity_image=image.gray[:].astype(np.float64, copy=False),
    )
    return max(props, key=lambda candidate: candidate.area)


def detected_center_coordinates(
    object_map: np.ndarray,
    center_map: np.ndarray,
) -> dict[int, tuple[float, float]]:
    """Associate detected center components with final colony labels.

    Each positive center component is assigned to the final object with which
    it has the largest pixel overlap. If several center components overlap the
    same object, the component with the greatest overlap wins. Ties are broken
    by the lowest numeric label. The reported coordinate is the centroid of
    the winning component's overlapping pixels, so unrelated detector pixels
    outside the final colony cannot pull the center away.

    Args:
        object_map: Final labeled colony map.
        center_map: Labeled compact-center map produced by ``center_detector``.

    Returns:
        Mapping from final colony label to global ``(row, col)`` coordinate.

    Raises:
        ValueError: If the two label maps do not have the same two-dimensional
            shape.
    """
    objects = np.asarray(object_map)
    centers = np.asarray(center_map)
    if objects.ndim != 2 or centers.ndim != 2 or objects.shape != centers.shape:
        raise ValueError(
            "center_detector output and final objmap must have the same 2-D shape"
        )

    overlapping = (objects > 0) & (centers > 0)
    if not overlapping.any():
        return {}

    center_labels = np.unique(centers[overlapping])
    assignments: dict[int, tuple[int, int, tuple[float, float]]] = {}
    for center_label_value in center_labels:
        center_label = int(center_label_value)
        component_overlap = overlapping & (centers == center_label_value)
        object_labels, counts = np.unique(
            objects[component_overlap], return_counts=True
        )
        maximum = int(counts.max())
        object_label = int(object_labels[counts == maximum].min())
        selected_pixels = np.argwhere(
            component_overlap & (objects == object_label)
        )
        coordinate = (
            float(np.mean(selected_pixels[:, 0])),
            float(np.mean(selected_pixels[:, 1])),
        )
        candidate = (maximum, center_label, coordinate)
        current = assignments.get(object_label)
        if current is None or (-candidate[0], candidate[1]) < (
            -current[0], current[1]
        ):
            assignments[object_label] = candidate

    return {
        object_label: candidate[2]
        for object_label, candidate in assignments.items()
    }


def _compute_mask_morphology(
    image: Image,
    prop=None,
    *,
    params: ZoneSegmentationParams,
    center_global: tuple[float, float] | None = None,
) -> ZoneSegmentation:
    """Compute mask-derived center, PELT core, symmetry, and expansion.

    This primitive deliberately does not calculate the historical
    colony-ness profile or its threshold-derived zone boundaries. Canonical
    Method B uses this result directly; the legacy path extends it below.
    """
    target_prop = _resolve_target_prop(image, prop)
    if target_prop.area < 10:
        empty = np.array([])
        centroid_global = (
            float(target_prop.slice[0].start),
            float(target_prop.slice[1].start),
        )
        return ZoneSegmentation(
            label=target_prop.label,
            bbox_slice=target_prop.slice,
            centroid_rc=(0.0, 0.0),
            density_profile=empty,
            annulus_radii=empty,
            core_radius=0.0,
            sholl_counts=empty,
            angular_R_profile=empty,
            angular_coverage=empty,
            symmetric_radius=0.0,
            mean_expansion=0.0,
            max_expansion=0.0,
            obj_mask=np.zeros((1, 1), dtype=bool),
            dist_map=np.zeros((1, 1), dtype=np.float64),
            gray_crop=np.zeros((1, 1), dtype=np.float64),
            r_outer_full_per_angle=np.zeros(
                _N_ANGULAR_SECTORS, dtype=np.float64
            ),
            zones_computed=False,
            centroid_global=centroid_global,
        )

    slc = target_prop.slice
    objmap_crop = np.asarray(image.objmap[:])[slc]
    gray_crop = np.asarray(image.gray[:])[slc]
    local_mask = objmap_crop == target_prop.label
    if center_global is not None:
        local_cr = (
            float(center_global[0]) - float(slc[0].start),
            float(center_global[1]) - float(slc[1].start),
        )
    elif params.method == "distance":
        dt = np.asarray(
            distance_transform_edt(local_mask), dtype=np.float64
        )
        peak_idx = np.unravel_index(np.argmax(dt), dt.shape)
        local_cr = (float(peak_idx[0]), float(peak_idx[1]))
    else:
        weighted = target_prop.centroid_weighted
        local_cr = (
            weighted[0] - slc[0].start,
            weighted[1] - slc[1].start,
        )
    dist_map = distance_from_point(local_mask.shape, local_cr)
    max_pixel_radius = int(np.max(dist_map[local_mask]))
    effective_annuli = max(6, min(params.n_annuli, max_pixel_radius))
    density_profile, annulus_radii = compute_radial_density_profile(
        local_mask, dist_map, effective_annuli
    )
    core_radius = find_core_radius(
        density_profile, annulus_radii, params.pelt_penalty
    )
    sholl_counts, angular_R_profile, angular_coverage = (
        compute_sholl_angular_profile(
            local_mask,
            dist_map,
            local_cr,
            annulus_radii,
            params.n_angular_bins,
        )
    )
    symmetric_radius = find_symmetric_radius(
        annulus_radii,
        angular_coverage,
        core_radius,
        params.symmetry_threshold,
        params.smoothing_window,
    )
    mean_expansion, max_expansion = compute_radial_expansion(
        local_mask, dist_map, core_radius
    )
    center_global = (
        local_cr[0] + float(slc[0].start),
        local_cr[1] + float(slc[1].start),
    )
    return ZoneSegmentation(
        label=target_prop.label,
        bbox_slice=slc,
        centroid_rc=local_cr,
        density_profile=density_profile,
        annulus_radii=annulus_radii,
        core_radius=core_radius,
        sholl_counts=sholl_counts,
        angular_R_profile=angular_R_profile,
        angular_coverage=angular_coverage,
        symmetric_radius=symmetric_radius,
        mean_expansion=mean_expansion,
        max_expansion=max_expansion,
        obj_mask=local_mask,
        dist_map=dist_map,
        gray_crop=gray_crop,
        r_outer_full_per_angle=per_angle_mask_envelope(
            local_mask, dist_map, local_cr
        ),
        zones_computed=False,
        centroid_global=center_global,
    )


def compute_zone_segmentation(
        image: Image,
        prop=None,
        *,
        params: ZoneSegmentationParams,
) -> ZoneSegmentation:
    """Compute concentric zone geometry (core/dense/sparse radii) for one object.

    Pure relocation of ``MeasureSymZones._compute_intermediates``. Reads
    ``params.<name>`` where the method read ``self.<name>``; calls the relocated
    module-level helpers; also records ``centroid_global`` (plate-frame inoculum
    centre) on the returned record.

    Args:
        image: Detected Image with objmap/objmask.
        prop: Pre-computed RegionProperties object. When ``None`` the largest
            object by area is selected via an internal ``regionprops`` call.
        params: Zone-segmentation parameters.

    Returns:
        ZoneSegmentation with all computed fields populated.
    """
    target_prop = _resolve_target_prop(image, prop)
    base = _compute_mask_morphology(image, target_prop, params=params)
    if target_prop.area < 10 or base.symmetric_radius <= 0:
        return base

    local_mask = base.obj_mask
    dist_map = base.dist_map
    annulus_radii = base.annulus_radii
    symmetric_radius = base.symmetric_radius

    # 9. Expand the analysis crop past the farthest mask pixel by
    # ``extent_margin`` so the outermost annuli see a slice of agar.
    # The mask's role here ends — the ring signal pools every pixel
    # in each annulus regardless of mask membership.
    max_mask_radius = float(np.max(dist_map[local_mask]))
    r_max = max_mask_radius * (1.0 + float(params.extent_margin))
    center_global = base.centroid_global
    image_shape = image.gray[:].shape[:2]
    expanded_slc = expand_slice_around_center(
            center_global, r_max, image_shape,
    )

    # 10. Re-crop arrays on the expanded slice.
    gray_crop_exp = image.gray[:][expanded_slc]
    if params.intensity_source == "detect_mat":
        intensity_crop = image.detect_mat[:][expanded_slc]
    else:
        intensity_crop = gray_crop_exp
    local_mask_exp = image.objmap[:][expanded_slc] == target_prop.label
    local_cr_exp = (
        center_global[0] - float(expanded_slc[0].start),
        center_global[1] - float(expanded_slc[1].start),
    )
    dist_map_exp = distance_from_point(
            intensity_crop.shape, local_cr_exp,
    )

    # 11. Equal-area annulus boundaries on the expanded disk, 0 → r_max.
    # The density-profile annuli ran up to max_mask_radius; here we
    # need a fresh set of centres whose scale matches the ring signal.
    n_annuli = int(annulus_radii.size)
    annulus_boundaries_exp = r_max * np.sqrt(
            np.arange(n_annuli + 1) / n_annuli
    )
    annulus_radii_exp = 0.5 * (
            annulus_boundaries_exp[:-1] + annulus_boundaries_exp[1:]
    )

    # 12. Build the 1D radial profiles (mask-free mean/variance; mask-only
    # count for the envelope cap).
    _theta, r_bin, valid_geom = build_theta_r_maps(
            intensity_crop.shape,
            local_cr_exp,
            dist_map_exp,
            annulus_boundaries_exp,
            n_annuli,
    )
    mean_profile, variance_profile, count_profile = (
        accumulate_radial_profile(
                r_bin, valid_geom, intensity_crop,
                n_annuli, int(params.min_samples_per_ring),
        )
    )
    mask_per_annulus = accumulate_mask_per_annulus(
            r_bin, valid_geom & local_mask_exp, n_annuli,
    )

    # 13. Radial smoothing of the mean profile, then colony-ness
    # normalisation.
    from scipy.ndimage import uniform_filter1d

    mean_profile_smoothed = uniform_filter1d(
            mean_profile, size=_ZONE_RADIAL_SMOOTHING, mode="nearest",
    )
    colony_ness, I_core_val, I_agar_val = compute_colony_ness_profile(
            mean_profile_smoothed, intensity_crop, local_mask_exp,
    )

    # 14. Threshold crossings → scalar zone radii.
    core_end, dense_end, sparse_end = extract_zone_radii(
            colony_ness, mask_per_annulus, annulus_radii_exp,
            float(params.tau_core), float(params.tau_dense),
            float(params.tau_sparse), symmetric_radius,
    )

    # 15. Per-angle mask envelope for the diagnostic overlay only.
    r_outer_full_per_angle = per_angle_mask_envelope(
            local_mask_exp, dist_map_exp, local_cr_exp,
    )

    # 16. Concentric-disk zone areas.
    core_area, dense_area, sparse_area = compute_zone_areas(
            core_end, dense_end, sparse_end,
    )

    return ZoneSegmentation(
            label=target_prop.label,
            bbox_slice=expanded_slc,
            centroid_rc=local_cr_exp,
            density_profile=base.density_profile,
            annulus_radii=annulus_radii,
            core_radius=base.core_radius,
            sholl_counts=base.sholl_counts,
            angular_R_profile=base.angular_R_profile,
            angular_coverage=base.angular_coverage,
            symmetric_radius=symmetric_radius,
            mean_expansion=base.mean_expansion,
            max_expansion=base.max_expansion,
            obj_mask=local_mask_exp,
            dist_map=dist_map_exp,
            gray_crop=gray_crop_exp,
            core_end_radius=core_end,
            dense_end_radius=dense_end,
            sparse_end_radius=sparse_end,
            r_outer_full_per_angle=r_outer_full_per_angle,
            core_area=core_area,
            dense_area=dense_area,
            sparse_area=sparse_area,
            colony_ness_profile=colony_ness,
            mean_profile=mean_profile,
            variance_profile=variance_profile,
            count_profile=count_profile,
            I_core=I_core_val,
            I_agar=I_agar_val,
            zones_computed=True,
            centroid_global=center_global,
    )


def resolve_zone_segmentation(
    image: Image,
    prop,
    *,
    legacy_params: ZoneSegmentationParams,
    canonical_params: OrientationChangePointParams,
    legacy_mode: bool,
    center_global: tuple[float, float] | None = None,
    center_required: bool = False,
) -> ZoneResolution:
    """Resolve one canonical zone geometry for all zone-aware measurers.

    Mask morphology supplies the independent PELT core, symmetry, and
    expansion measurements. Only legacy mode extends that geometry through
    the historical colony-ness thresholds; canonical mode runs Method B
    directly from the mask-derived center and final detection signal.

    Args:
        image: Detected image containing the final object map and signal.
        prop: RegionProperties record for the target object.
        legacy_params: Parameters for the preserved mask and colony-ness path.
        canonical_params: Parameters for Method B.
        legacy_mode: Whether to return the colony-ness zone partition unchanged.
        center_global: Optional detector-selected center in full-image coordinates.
        center_required: Whether a configured detector makes a missing center a
            canonical failure rather than a request for the default estimator.

    Returns:
        Shared segmentation, provenance, and reusable orientation context.
    """
    if legacy_mode:
        base = compute_zone_segmentation(image, prop, params=legacy_params)
        return ZoneResolution(
            segmentation=base,
            method_code=0,
            method_used="legacy_colony_ness",
            failure_reason="none",
        )

    base = _compute_mask_morphology(
        image,
        prop,
        params=legacy_params,
        center_global=center_global,
    )
    if center_required and center_global is None:
        missing = replace(
            base,
            core_radius=float("nan"),
            symmetric_radius=float("nan"),
            mean_expansion=float("nan"),
            max_expansion=float("nan"),
            core_end_radius=float("nan"),
            dense_end_radius=float("nan"),
            sparse_end_radius=float("nan"),
            core_area=float("nan"),
            dense_area=float("nan"),
            sparse_area=float("nan"),
            zones_computed=False,
        )
        return ZoneResolution(
            segmentation=missing,
            method_code=4,
            method_used="missing",
            failure_reason="center_not_found",
        )
    if prop.area < 10:
        missing = replace(
            base,
            core_radius=float("nan"),
            symmetric_radius=float("nan"),
            mean_expansion=float("nan"),
            max_expansion=float("nan"),
            core_end_radius=float("nan"),
            dense_end_radius=float("nan"),
            sparse_end_radius=float("nan"),
            core_area=float("nan"),
            dense_area=float("nan"),
            sparse_area=float("nan"),
            zones_computed=False,
        )
        return ZoneResolution(
            segmentation=missing,
            method_code=4,
            method_used="missing",
            failure_reason="tiny_object",
        )

    # Use one full-image crop for both public measurers. The target mask alone
    # determines the radius; adjacent objects remain excluded from every ring.
    coords = np.asarray(prop.coords, dtype=np.float64)
    if coords.size == 0:
        return ZoneResolution(
            segmentation=replace(
                base,
                core_end_radius=float("nan"),
                dense_end_radius=float("nan"),
                sparse_end_radius=float("nan"),
                core_area=float("nan"),
                dense_area=float("nan"),
                sparse_area=float("nan"),
                zones_computed=False,
            ),
            method_code=4,
            method_used="missing",
            failure_reason="invalid_object_mask",
        )
    radial_extent = float(
        np.max(
            np.hypot(
                coords[:, 0] - base.centroid_global[0],
                coords[:, 1] - base.centroid_global[1],
            )
        )
    )
    analysis_radius = radial_extent + canonical_params.ring_width
    analysis_slice = expand_slice_around_center(
        base.centroid_global,
        analysis_radius,
        image.gray[:].shape[:2],
    )
    signal = np.asarray(image.detect_mat[analysis_slice], dtype=np.float64)
    object_mask = np.asarray(
        image.objmap[:][analysis_slice] == base.label, dtype=bool
    )
    center = (
        base.centroid_global[0] - float(analysis_slice[0].start),
        base.centroid_global[1] - float(analysis_slice[1].start),
    )
    fitted = fit_orientation_zones(
        object_mask,
        signal,
        center,
        canonical_params,
    )
    result = fitted.result
    if result.zones_computed:
        core_area, dense_area, sparse_area = compute_zone_areas(
            result.core_zone_radius,
            result.dense_radius,
            result.outer_radius,
        )
        core_end_radius = result.core_zone_radius
        dense_end_radius = result.dense_radius
        sparse_end_radius = result.outer_radius
    else:
        core_area = dense_area = sparse_area = float("nan")
        core_end_radius = dense_end_radius = sparse_end_radius = float("nan")
    context = fitted.context
    distance_map = (
        context.distance_map
        if context is not None
        else distance_from_point(object_mask.shape, center)
    )
    updated = replace(
        base,
        bbox_slice=analysis_slice,
        centroid_rc=(float(center[0]), float(center[1])),
        obj_mask=object_mask,
        dist_map=distance_map,
        gray_crop=np.asarray(image.gray[analysis_slice], dtype=np.float64),
        core_end_radius=core_end_radius,
        dense_end_radius=dense_end_radius,
        sparse_end_radius=sparse_end_radius,
        r_outer_full_per_angle=per_angle_mask_envelope(
            object_mask, distance_map, center
        ),
        core_area=core_area,
        dense_area=dense_area,
        sparse_area=sparse_area,
        zones_computed=result.zones_computed,
    )
    return ZoneResolution(
        segmentation=updated,
        method_code=result.method_code,
        method_used=result.method_used,
        failure_reason=result.failure_reason,
        canonical_result=result,
        orientation_context=context,
    )


# ── static helpers ───────────────────────────────────────────────


def compute_radial_density_profile(
        obj_mask: np.ndarray,
        dist_map: np.ndarray,
        n_annuli: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute radial density profile using equal-area annuli.

    Args:
        obj_mask: Boolean mask of the object in local coordinates.
        dist_map: Pre-computed Euclidean distance map from the centroid.
        n_annuli: Number of annular bins.

    Returns:
        Tuple of (density, center_radii), each shape (n_annuli,).
    """
    obj_distances = dist_map[obj_mask]
    if len(obj_distances) == 0 or obj_distances.max() == 0:
        return np.zeros(n_annuli), np.zeros(n_annuli)

    max_radius = float(obj_distances.max())
    boundaries = max_radius * np.sqrt(np.arange(n_annuli + 1) / n_annuli)
    center_radii = (boundaries[:-1] + boundaries[1:]) / 2.0

    # Vectorized binning
    bin_indices = np.digitize(obj_distances, boundaries) - 1
    bin_indices = np.clip(bin_indices, 0, n_annuli - 1)
    pixel_counts = np.bincount(bin_indices, minlength=n_annuli)

    # Normalize by geometric area of each annulus
    geometric_areas = np.pi * (boundaries[1:] ** 2 - boundaries[:-1] ** 2)
    geometric_areas = np.maximum(geometric_areas, 1e-10)  # avoid division by zero
    density = pixel_counts.astype(np.float64) / geometric_areas

    return density, center_radii


def find_core_radius(
        density_profile: np.ndarray,
        annulus_radii: np.ndarray,
        pelt_penalty: float,
) -> float:
    """Find the core radius via PELT changepoint detection on the density profile.

    Args:
        density_profile: 1D radial density signal from
            ``compute_radial_density_profile``.
        annulus_radii: Corresponding annulus center radii.
        pelt_penalty: PELT penalty parameter controlling sensitivity.

    Returns:
        Core radius in pixels (0.0 if no changepoint found).
    """
    import ruptures as rpt

    signal = density_profile.reshape(-1, 1)
    if signal.shape[0] < 6:
        return 0.0
    algo = rpt.Pelt(model="l2", min_size=3).fit(signal)
    changepoints = algo.predict(pen=pelt_penalty)
    # changepoints always ends with len(signal). Real ones are all but last.
    real_cps = changepoints[:-1]
    if not real_cps:
        return 0.0
    first_cp_idx = real_cps[0]
    idx = min(first_cp_idx, len(annulus_radii) - 1)
    return float(annulus_radii[idx])


def extract_mask_boundary(obj_mask: np.ndarray) -> np.ndarray:
    """Extract 4-connectivity mask-boundary pixels.

    Boundary pixels are object pixels touching at least one background
    pixel in a 4-neighbourhood (up / down / left / right).

    Args:
        obj_mask: Boolean mask of the object.

    Returns:
        Boolean array the same shape as ``obj_mask``, *True* at
        boundary pixels only.
    """
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int8)
    nbr = convolve(obj_mask.astype(np.int8), kernel, mode="constant", cval=0)
    return obj_mask & (nbr < 4)


def compute_sholl_angular_profile(
        obj_mask: np.ndarray,
        dist_map: np.ndarray,
        centroid_rc: tuple[float, float],
        annulus_radii: np.ndarray,
        n_angular_bins: int,
        min_boundary_per_annulus: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Sholl-count, per-annulus angular R̄, and angular coverage.

    For each equal-area annulus (reconstructed from ``annulus_radii``
    using the inverse of the sqrt construction in
    :func:`compute_radial_density_profile`), this function measures:

    * ``sholl_counts``: number of mask-boundary pixels in the annulus.
    * ``angular_R_profile``: the circular mean resultant length
      ``R̄ = sqrt(mean(cos θ)² + mean(sin θ)²)`` over boundary-pixel
      angles, where θ is the angle from the centroid. Annuli with
      fewer than ``min_boundary_per_annulus`` boundary pixels get
      ``NaN``.
    * ``angular_coverage``: fraction of the ``n_angular_bins`` uniform
      angular bins that contain at least one boundary pixel. Empty
      annuli get coverage 0.

    Args:
        obj_mask: Boolean mask of the object.
        dist_map: Distance-from-centroid map (same shape as
            ``obj_mask``).
        centroid_rc: (row, col) centroid coordinates in local bbox
            space. Provided for signature parity with the plan; angles
            are computed from ``dist_map`` positions relative to this
            centre.
        annulus_radii: Centre radii of the equal-area annuli, matching
            the profile from :func:`compute_radial_density_profile`.
        n_angular_bins: Number of uniform angular bins for the
            coverage diagnostic.
        min_boundary_per_annulus: Minimum boundary pixels required to
            compute a finite ``R̄``; annuli below this threshold are
            marked NaN.

    Returns:
        Tuple ``(sholl_counts, angular_R_profile, angular_coverage)``,
        each shape ``(n_annuli,)``.
    """
    n_annuli = len(annulus_radii)
    empty_counts = np.zeros(n_annuli, dtype=np.int64)
    empty_R = np.full(n_annuli, np.nan, dtype=np.float64)
    empty_coverage = np.zeros(n_annuli, dtype=np.float64)

    if n_annuli == 0 or not obj_mask.any():
        return empty_counts, empty_R, empty_coverage

    # Angular statistics use ALL mask pixels so the signal remains
    # meaningful on dense colonies where mask-boundary pixels only live
    # at the outer envelope. Sholl counts still reflect boundary pixels,
    # exposed separately for diagnostics.
    boundary = extract_mask_boundary(obj_mask)

    mask_coords = np.argwhere(obj_mask)  # (N, 2) row, col
    mask_radii = dist_map[obj_mask]
    dr = mask_coords[:, 0] - centroid_rc[0]
    dc = mask_coords[:, 1] - centroid_rc[1]
    angles = np.arctan2(dr, dc)  # (-pi, pi]

    # 2. Reconstruct annulus boundaries from annulus_radii (inverse of the
    #    equal-area sqrt construction used by compute_radial_density_profile).
    #    boundaries[i] = max_radius * sqrt(i / n_annuli), and
    #    center_radii[i] = (boundaries[i] + boundaries[i+1]) / 2 →
    #    max_radius = 2 * annulus_radii[-1] / (sqrt(1) + sqrt((n-1)/n)).
    if n_annuli == 1:
        max_radius = float(annulus_radii[0]) * np.sqrt(2.0) if annulus_radii[
                                                                   0] > 0 else 0.0
    else:
        denom = np.sqrt(1.0) + np.sqrt((n_annuli - 1) / n_annuli)
        max_radius = float(2.0 * annulus_radii[-1] / denom) if denom > 0 else 0.0

    if max_radius <= 0:
        return empty_counts, empty_R, empty_coverage

    boundaries = max_radius * np.sqrt(np.arange(n_annuli + 1) / n_annuli)

    mask_bin_indices = np.digitize(mask_radii, boundaries) - 1
    mask_bin_indices = np.clip(mask_bin_indices, 0, n_annuli - 1)

    # Sholl count diagnostic uses boundary pixels (matches classical Sholl).
    if boundary.any():
        boundary_radii = dist_map[boundary]
        boundary_bin_indices = np.digitize(boundary_radii, boundaries) - 1
        boundary_bin_indices = np.clip(boundary_bin_indices, 0, n_annuli - 1)
        sholl_counts = np.bincount(
                boundary_bin_indices, minlength=n_annuli,
        ).astype(np.int64)
    else:
        sholl_counts = empty_counts.copy()

    angular_R_profile = np.full(n_annuli, np.nan, dtype=np.float64)
    angular_coverage = np.zeros(n_annuli, dtype=np.float64)

    # Uniform angular-bin edges for coverage
    bin_edges = np.linspace(-np.pi, np.pi, n_angular_bins + 1)

    for k in range(n_annuli):
        mask_k = mask_bin_indices == k
        count_k = int(mask_k.sum())
        if count_k < min_boundary_per_annulus:
            continue

        angles_k = angles[mask_k]
        cos_mean = np.mean(np.cos(angles_k))
        sin_mean = np.mean(np.sin(angles_k))
        angular_R_profile[k] = float(np.sqrt(cos_mean ** 2 + sin_mean ** 2))

        ang_bins = np.digitize(angles_k, bin_edges) - 1
        ang_bins = np.clip(ang_bins, 0, n_angular_bins - 1)
        unique_bins = np.unique(ang_bins)
        angular_coverage[k] = float(len(unique_bins)) / float(n_angular_bins)

    return sholl_counts, angular_R_profile, angular_coverage


def find_symmetric_radius(
        annulus_radii: np.ndarray,
        angular_coverage: np.ndarray,
        core_radius: float,
        threshold: float,
        smoothing_window: int,
) -> float:
    """First radius past ``core_radius`` where smoothed coverage drops below ``threshold``.

    Coverage is the fraction of angular bins occupied by mask pixels
    at a given radius. Growth is considered symmetric as long as
    coverage stays at or above ``threshold`` (e.g., 4/6 = four of
    six 60-degree bins filled).

    NaN-aware: annuli with zero coverage (no mask pixels at all)
    are treated as populated with value 0. Falls back to the outer
    radius of the last annulus when no crossing is found, and falls
    back to the unsmoothed profile when ``smoothing_window`` exceeds
    the annulus count past the core.

    Args:
        annulus_radii: Centre radii of the equal-area annuli.
        angular_coverage: Per-annulus angular coverage fraction
            (0–1). Zero means no mask pixels in the annulus.
        core_radius: Inoculum core radius in pixels.
        threshold: Minimum angular coverage for growth to be
            considered symmetric. Default is 4/6 (~0.667).
        smoothing_window: Moving-average window size (in annuli)
            applied to coverage before the threshold test.

    Returns:
        Radial distance in pixels.
    """
    annulus_radii = np.asarray(annulus_radii, dtype=np.float64)
    angular_coverage = np.asarray(angular_coverage, dtype=np.float64)

    if annulus_radii.size == 0:
        return 0.0

    past_core = annulus_radii > core_radius
    valid_idx = np.where(past_core)[0]

    outer_radius = float(annulus_radii[-1]) if annulus_radii.size > 0 else 0.0

    if valid_idx.size == 0:
        return outer_radius

    values = angular_coverage[valid_idx]
    populated_count = int(valid_idx.size)
    if smoothing_window > populated_count:
        smoothed = values
    else:
        w = max(1, int(smoothing_window))
        kernel = np.ones(w, dtype=np.float64) / float(w)
        smoothed = np.convolve(values, kernel, mode="same")

    crossings = np.where(smoothed < threshold)[0]
    if crossings.size == 0:
        return outer_radius

    # Return the last passing annulus (one before the first failure).
    first_fail = crossings[0]
    if first_fail == 0:
        return core_radius
    last_pass = int(valid_idx[first_fail - 1])
    return float(annulus_radii[last_pass])


def compute_radial_expansion(
        obj_mask: np.ndarray,
        dist_map: np.ndarray,
        core_radius: float,
) -> tuple[float, float]:
    """Mean and max radial distance beyond the inoculum core.

    ``mean_expansion`` averages boundary-pixel distances from the
    centroid and subtracts ``core_radius``. ``max_expansion`` uses the
    maximum mask-pixel distance. Both values are clamped to ``>= 0`` so
    that a rare ``core_radius`` overshooting the actual extent does
    not produce negative output.

    Args:
        obj_mask: Boolean mask of the object in local coordinates.
        dist_map: Distance-from-centroid map (same shape as
            ``obj_mask``).
        core_radius: Inoculum core radius in pixels.

    Returns:
        Tuple ``(mean_expansion, max_expansion)`` in pixels.
    """
    if not obj_mask.any():
        return 0.0, 0.0

    boundary = extract_mask_boundary(obj_mask)
    if boundary.any():
        mean_extent = float(np.mean(dist_map[boundary]))
    else:
        mean_extent = float(np.mean(dist_map[obj_mask]))

    max_extent = float(np.max(dist_map[obj_mask]))

    mean_expansion = max(0.0, mean_extent - float(core_radius))
    max_expansion = max(0.0, max_extent - float(core_radius))
    return mean_expansion, max_expansion


def per_angle_mask_envelope(
        local_mask: np.ndarray,
        dist_map: np.ndarray,
        centroid_rc: tuple[float, float],
) -> np.ndarray:
    """Uncapped per-angle maximum mask radius in 1° sectors.

    Used as a lightweight fallback for the outer-envelope diagnostic
    when the full zone pipeline is skipped (e.g. symmetric radius
    collapsed to zero).

    Args:
        local_mask: Boolean mask of the object in local coordinates.
        dist_map: Distance-from-centroid map (same shape).
        centroid_rc: (row, col) centroid in local coordinates.

    Returns:
        Float64 array of shape ``(_N_ANGULAR_SECTORS,)`` with the
        farthest mask-pixel distance in each 1° angular sector; zero
        where no mask pixels fall in the sector.
    """
    envelope = np.zeros(_N_ANGULAR_SECTORS, dtype=np.float64)
    if not local_mask.any():
        return envelope

    rows, cols = np.indices(local_mask.shape)
    dr = rows - centroid_rc[0]
    dc = cols - centroid_rc[1]
    theta = np.mod(np.degrees(np.arctan2(dr, dc)), 360.0).astype(np.int32)
    # np.maximum.at is an unbuffered reduction that handles duplicate
    # indices correctly, unlike plain fancy indexing.
    np.maximum.at(envelope, theta[local_mask], dist_map[local_mask])
    return envelope


# ── zone segmentation helpers ────────────────────────────────────


def build_theta_r_maps(
        shape: tuple[int, int],
        centroid_rc: tuple[float, float],
        dist_map: np.ndarray,
        annulus_boundaries: np.ndarray,
        n_annuli: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-pixel angular sector and annulus index plus a geometric validity mask.

    The validity selector is purely geometric (annulus index in range
    and distance from centre > 0) — it does **not** depend on the
    object mask. Callers that need a mask-restricted selector for
    envelope counting compose ``valid & local_mask`` themselves.

    Args:
        shape: ``(H, W)`` of the crop.
        centroid_rc: (row, col) centroid in local coordinates.
        dist_map: Distance-from-centroid map (same shape as crop).
        annulus_boundaries: Equal-area annulus boundary radii of
            length ``n_annuli + 1``.
        n_annuli: Number of annular bins.

    Returns:
        Tuple ``(theta, r_bin, valid)`` arrays sharing the crop shape.
        ``theta`` is integer degrees in [0, 360), ``r_bin`` is the
        annulus index in [0, n_annuli), and ``valid`` selects pixels
        with a finite, in-range annulus assignment (mask-free).
    """
    h, w = shape
    rows, cols = np.indices((h, w))
    dr = rows - centroid_rc[0]
    dc = cols - centroid_rc[1]
    theta = np.mod(np.degrees(np.arctan2(dr, dc)), 360.0).astype(np.int16)
    r_bin = np.digitize(dist_map, annulus_boundaries) - 1
    valid = (r_bin >= 0) & (r_bin < n_annuli) & (dist_map > 0)
    return theta, r_bin, valid


def accumulate_radial_profile(
        r_bin: np.ndarray,
        valid: np.ndarray,
        intensity: np.ndarray,
        n_annuli: int,
        min_samples_per_ring: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1D per-ring mean / variance / count profiles.

    Pools all angles of each annulus together so each ring gets one
    scalar summary — mean, variance, and pixel count — derived from
    every valid (i.e. in-range, non-centre) pixel regardless of mask
    membership. Rings with fewer than ``min_samples_per_ring`` pixels
    have their mean and variance linearly interpolated from
    neighbouring rings.

    Args:
        r_bin: Per-pixel annulus index.
        valid: Boolean per-pixel geometric selector (in-range annulus
            and non-zero distance from centre). Not mask-restricted.
        intensity: Per-pixel intensity (gray or detect_mat).
        n_annuli: Number of annular bins.
        min_samples_per_ring: Rings with fewer samples are interpolated.

    Returns:
        Tuple ``(mean_profile, variance_profile, count_profile)``.
        ``mean_profile`` and ``variance_profile`` are length-``n_annuli``
        float64; ``count_profile`` is int64.
    """
    rb = r_bin[valid].astype(np.int32)
    intens = intensity[valid].astype(np.float64)

    count = np.bincount(rb, minlength=n_annuli).astype(np.int64)
    sum_I = np.bincount(rb, weights=intens, minlength=n_annuli)
    sum_I2 = np.bincount(rb, weights=intens * intens, minlength=n_annuli)

    safe_count = np.where(count > 0, count, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, sum_I / safe_count, np.nan)
        variance = np.where(
                count > 0,
                sum_I2 / safe_count - (sum_I / safe_count) ** 2,
                np.nan,
        )
    variance = np.where(
            np.isnan(variance), np.nan, np.maximum(variance, 0.0),
    )

    under = count < max(1, int(min_samples_per_ring))
    mean = np.where(under, np.nan, mean)
    variance = np.where(under, np.nan, variance)

    # Linear-interpolate NaN rings so the segmentation sees a
    # continuous profile. If everything is NaN, return zeros.
    x = np.arange(n_annuli, dtype=np.float64)
    for arr in (mean, variance):
        m = np.isfinite(arr)
        if not m.any():
            arr[:] = 0.0
            continue
        if m.all():
            continue
        arr[~m] = np.interp(x[~m], x[m], arr[m])
    return mean, variance, count


def accumulate_mask_per_annulus(
        r_bin: np.ndarray,
        valid_mask: np.ndarray,
        n_annuli: int,
) -> np.ndarray:
    """1D per-ring mask-pixel count used for the envelope-radius floor.

    Args:
        r_bin: Per-pixel annulus index.
        valid_mask: Boolean per-pixel selector for geometrically-valid
            *mask* pixels (``valid & local_mask``).
        n_annuli: Number of annular bins.

    Returns:
        Int64 array of shape ``(n_annuli,)``.
    """
    rb = r_bin[valid_mask].astype(np.int32)
    return np.bincount(rb, minlength=n_annuli).astype(np.int64)


def compute_colony_ness_profile(
        mean_profile: np.ndarray,
        intensity_crop: np.ndarray,
        local_mask: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Normalise ring-mean intensity into a colony-ness profile.

    ``I_core`` and ``I_agar`` are taken from the 5th and 95th
    percentiles of the expanded intensity crop — mask-free, robust to
    both dark-colony (gray) and bright-colony (detect_mat) conventions.
    The mask is consulted only to decide which percentile corresponds
    to "colony": whichever of ``p5``/``p95`` is closer to the mean
    intensity inside the mask is assigned to ``I_core``.

    The output ``c(r) = clip((mean(r) - I_agar) / (I_core - I_agar), 0, 1)``
    is 1 where the ring is fully colony-composition and 0 where the
    ring is fully agar.

    Args:
        mean_profile: Per-ring mean intensity (length ``n_annuli``).
        intensity_crop: Expanded-crop intensity array.
        local_mask: Boolean mask (expanded-crop shape) of the object
            — used only for direction detection, not for data selection.

    Returns:
        Tuple ``(colony_ness_profile, I_core, I_agar)``.
    """
    flat = intensity_crop.astype(np.float64).ravel()
    p5, p95 = np.percentile(flat, [5.0, 95.0])
    if local_mask.any():
        mask_mean = float(intensity_crop[local_mask].astype(np.float64).mean())
    else:
        mask_mean = float(flat.mean())
    if abs(mask_mean - p5) <= abs(mask_mean - p95):
        I_core = float(p5)
        I_agar = float(p95)
    else:
        I_core = float(p95)
        I_agar = float(p5)
    span = I_core - I_agar
    if abs(span) < 1e-9:
        colony_ness = np.zeros_like(mean_profile, dtype=np.float64)
    else:
        colony_ness = (mean_profile - I_agar) / span
        colony_ness = np.clip(colony_ness, 0.0, 1.0)
    return colony_ness, I_core, I_agar


def extract_zone_radii(
        colony_ness_profile: np.ndarray,
        mask_per_annulus: np.ndarray,
        annulus_radii: np.ndarray,
        tau_core: float,
        tau_dense: float,
        tau_sparse: float,
        symmetric_radius: float,
) -> tuple[float, float, float]:
    """Scalar zone radii from a monotonically-decreasing colony-ness profile.

    Each radius is the last ring whose colony-ness is at or above
    the corresponding threshold, capped outside-in so the nesting
    ``r_core ≤ r_dense_end ≤ r_outer`` holds. The outer radius is
    further capped by the mask envelope (last annulus with any mask
    pixel) and by ``symmetric_radius``.

    Args:
        colony_ness_profile: Length-``n_annuli`` profile with values
            in [0, 1].
        mask_per_annulus: Length-``n_annuli`` mask-pixel count per
            ring; used for the envelope cap.
        annulus_radii: Centre radii of the equal-area annuli
            (length ``n_annuli``).
        tau_core: Colony-ness threshold for the core/dense boundary.
        tau_dense: Colony-ness threshold for the dense/sparse boundary.
        tau_sparse: Colony-ness threshold for the sparse/outside boundary.
        symmetric_radius: Global cap from the angular-coverage analysis.

    Returns:
        Tuple ``(core_end, dense_end, sparse_end)`` as floats.
    """
    n_annuli = int(annulus_radii.size)
    if n_annuli == 0:
        return 0.0, 0.0, 0.0

    has_mask = mask_per_annulus > 0
    last_mask_idx = int(np.where(has_mask)[0].max()) if has_mask.any() else -1
    envelope = (
        float(annulus_radii[last_mask_idx]) if last_mask_idx >= 0 else 0.0
    )

    def _last_above(threshold: float) -> float:
        above = colony_ness_profile >= threshold
        if not above.any():
            return 0.0
        # Last contiguous-from-start index that's above threshold,
        # then extend with non-contiguous matches only if no gaps.
        idx = int(np.where(above)[0].max())
        return float(annulus_radii[idx])

    core_end = _last_above(float(tau_core))
    dense_end = _last_above(float(tau_dense))
    sparse_end = _last_above(float(tau_sparse))

    # Outer cap: envelope ∩ sparse crossing ∩ symmetric radius.
    outer_cap = min(envelope, float(symmetric_radius)) if envelope > 0 else float(
            symmetric_radius)
    sparse_end = min(sparse_end if sparse_end > 0 else envelope, outer_cap)

    # Outside-in nesting clamp.
    dense_end = min(dense_end, sparse_end)
    core_end = min(core_end, dense_end)
    return core_end, dense_end, sparse_end


def compute_zone_areas(
        r_core: float,
        r_dense_end: float,
        r_outer: float,
) -> tuple[float, float, float]:
    """Concentric-disk areas (pixel^2) for the three nested zones.

    Args:
        r_core: Core boundary radius (pixels).
        r_dense_end: Dense boundary radius (pixels).
        r_outer: Outer envelope radius (pixels).

    Returns:
        Tuple ``(core_area, dense_area, sparse_area)`` in pixel^2.
    """
    rc = max(0.0, float(r_core))
    rd = max(rc, float(r_dense_end))
    ro = max(rd, float(r_outer))
    core_area = float(np.pi * rc * rc)
    dense_area = float(np.pi * (rd * rd - rc * rc))
    sparse_area = float(np.pi * (ro * ro - rd * rd))
    return core_area, dense_area, sparse_area
