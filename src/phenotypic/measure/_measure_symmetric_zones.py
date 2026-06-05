"""Mask-based radial expansion and symmetry measurement operator.

Implements :class:`MeasureSymmetricZones`, a branch-free alternative to
:class:`MeasureRadialExpansion` that answers two colony-level questions
directly from the binary object mask: how far has growth progressed past
the inoculum, and out to what radius does that growth remain angularly
uniform?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
import pandas as pd
from pydantic import PrivateAttr
from scipy.ndimage import convolve, distance_transform_edt
from skimage.measure import approximate_polygon, find_contours, regionprops

from phenotypic.abc_ import Control, FigureProvider, MeasureFeatures, figure
from phenotypic.schema import OBJECT
from phenotypic.schema import SYMMETRIC_ZONES

_NEIGHBOR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)

# Zone-segmentation constants
_N_ANGULAR_SECTORS = 360
_ZONE_RADIAL_SMOOTHING = 3

# Diagnostic-overlay constants
_ZONE_POLYGON_STRIDE = 5  # 360 / 5 = 72 vertices per zone polygon
_OBJMAP_POLYGON_TOLERANCE = 0.5  # Douglas-Peucker pixel tolerance

# Okabe-Ito palette constants for diagnostic plots
_OI_NAVY = "#003660"
_OI_ORANGE = "#E69F00"
_OI_SKY = "#56B4E9"
_OI_GREEN = "#009E73"
_OI_BLUE = "#0072B2"
_OI_PURPLE = "#CC79A7"
_OI_VERMILION = "#D55E00"
_OI_GREY = "#BBBBBB"

# Which image array backs the plotly overlay; a select Control bound (by identity)
# to the inspect() figure's ``base_layer`` kwarg for the interactive dashboard.
BASE_LAYER = Control(
        label="Base layer",
        kind="select",
        default="gray",
        options=("rgb", "gray", "detect_mat"),
        help="Image array rendered behind the symmetric-radius overlay.",
)


@dataclass
class _SymmetryIntermediates:
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


class MeasureSymmetricZones(MeasureFeatures, FigureProvider):
    """Measure colony radial expansion and angular symmetry from the object mask alone.

    Quantifies each colony by four scalars derived directly from its binary
    mask and distance-from-inoculum map — no skeletonization, no branch
    tracing, no runner outlier flagging. The headline output is
    ``SymmetricRadius``, the first radius past the inoculum core at which the
    per-annulus circular mean resultant length of mask-boundary pixels drops
    below a tunable symmetry threshold. ``CoreRadius`` (PELT changepoint on
    the radial density profile, identical algorithm to
    :class:`MeasureRadialExpansion`) anchors the measurement; ``MeanExpansion``
    and ``MaxExpansion`` summarise how far growth reached past that core.

    Zone segmentation (core / dense / sparse) uses a 1D per-annulus
    **normalised colony-ness** signal ``c(r)`` computed from the ring-wide
    mean intensity. After calibrating ``I_core`` and ``I_agar`` from
    percentiles of the expanded crop, each ring's mean intensity is mapped
    into ``[0, 1]`` where 1 = pure colony and 0 = pure agar. For a
    roughly circular colony, ``c(r)`` decreases monotonically outward
    (uniform dense core ≈ 1, mixed dense branching ≈ 0.5–0.8, sparse
    branching ≈ 0.1–0.4, agar = 0), so zone boundaries follow directly
    from threshold crossings — no peak-finding. Zones are emitted as
    concentric circles at the three scalar radii. Variance and raw mean
    per ring are retained in the intermediates for diagnostic access but
    do not drive segmentation.

    The mask's only role in this signal is to (1) isolate which colony is
    being analysed, (2) seed the inoculum centre as the peak of the
    in-mask Euclidean distance transform, and (3) detect the
    colony-vs-agar intensity direction so the normalisation handles dark
    colonies (gray convention) and bright colonies (detect_mat
    convention) uniformly. The ring accumulation extends out to
    ``r_max = max_mask_radius × (1 + extent_margin)`` regardless of mask
    boundaries, so background pixels contribute truthfully to the signal.

    Args:
        n_annuli: Number of equal-area annuli used for the radial density
            profile and angular analysis. Defaults to 100.
        pelt_penalty: PELT penalty controlling changepoint sensitivity for
            core detection. Defaults to 5.0.
        symmetry_threshold: Minimum angular coverage (fraction of angular
            bins occupied) for growth to be considered symmetric. Defaults
            to 4/6 (~0.667); with 6 angular bins this means at least 4
            of 6 60-degree sectors must contain mask pixels.
        n_angular_bins: Number of angular bins used to compute the
            per-annulus angular coverage diagnostic. Defaults to 36
            (10 degree resolution).
        smoothing_window: Moving-average window (in annuli) applied to the
            angular R̄ profile before the threshold test. Defaults to 3.
        method: Inoculum centre estimator --- ``"distance"`` uses the peak
            of the Euclidean distance transform, ``"intensity"`` uses the
            intensity-weighted centroid. Defaults to ``"distance"``.
        extent_margin: Fractional expansion of the ring accumulator past
            the farthest mask pixel, so the outer annuli sample a small
            agar tail for the ``I_agar`` reference. Defaults to 0.05
            (5%) — deliberately small because tight plates risk touching
            neighbouring colonies.
        min_samples_per_ring: Minimum pixel count required to compute
            a mean for a given ring; below this the ring's mean is
            filled by linear interpolation from its neighbours before
            normalisation. Defaults to 5.
        tau_core: Colony-ness threshold that marks the core/dense
            boundary. The core zone extends out to the last ring where
            ``c(r) ≥ tau_core``. Defaults to 0.9 (last "≥90% colony"
            ring).
        tau_dense: Colony-ness threshold that marks the dense/sparse
            boundary. The dense zone extends to the last ring where
            ``c(r) ≥ tau_dense``. Defaults to 0.5 (last "majority
            colony" ring).
        tau_sparse: Colony-ness threshold that marks the sparse/outside
            boundary. The sparse zone extends to the last ring where
            ``c(r) ≥ tau_sparse``, capped at the mask envelope. Defaults
            to 0.1.
        intensity_source: Image array used for the mean-intensity
            calculation -- ``"gray"`` uses the grayscale (dark = colony),
            ``"detect_mat"`` uses the detection matrix (bright = colony).
            Direction is auto-detected. Defaults to ``"gray"``.

    Returns:
        pd.DataFrame: Object-level radial symmetry measurements with
        columns:

            - Object_Label: unique object identifier.
            - SymZones_CoreRadius: inoculum core radius (pixels).
            - SymZones_SymmetricRadius: first radius past the core
              where R̄ exceeds the symmetry threshold (pixels).
            - SymZones_MeanExpansion: mean boundary-pixel distance
              beyond the core (pixels, clamped at 0).
            - SymZones_MaxExpansion: maximum mask-pixel distance
              beyond the core (pixels, clamped at 0).
            - SymZones_CoreEndRadius: mean per-angle core boundary
              radius from the bright-fraction outward walk (pixels).
            - SymZones_DenseEndRadius: mean per-angle outer radius
              of the dense branching zone (pixels).
            - SymZones_SparseEndRadius: mean per-angle outer radius
              of the sparse branching zone, capped at the symmetric
              envelope (pixels).
            - SymZones_CoreArea: pixel^2 area of the inoculum core
              zone integrated across the 360-sector polar polygon.
            - SymZones_DenseArea: pixel^2 area of the dense
              branching zone (annular region between core and dense
              boundaries).
            - SymZones_SparseArea: pixel^2 area of the sparse
              branching zone (annular region between dense and outer
              boundaries).

    Best For:
        - Summarising colony-level radial growth with a single symmetry
          figure of merit.
        - Distinguishing uniformly-expanding colonies from those with
          sectors, lopsided growth, or directional bias.
        - Comparing wild-type versus mutant expansion phenotypes when the
          biological question is about the colony envelope, not individual
          hyphae.
        - Time-course assays where runner counts are noisy but colony
          extent is informative.

    Consider Also:
        - :class:`MeasureRadialExpansion` when you need per-branch
          statistics (branch counts, outlier runner detection) rather
          than a single symmetry scalar.
        - :class:`MeasureShape` for general morphological descriptors
          (circularity, eccentricity) that do not require radial analysis.
        - :class:`MeasureBounds` for lightweight bounding-box data
          without any radial pipeline.

    See Also:
        :doc:`/tutorials/notebooks/07_measuring_and_exporting` for a
        walkthrough of measuring and exporting colony data.
    """

    _measurement_infoclass: ClassVar[type] = SYMMETRIC_ZONES

    n_annuli: int = 100
    pelt_penalty: float = 5.0
    symmetry_threshold: float = 4 / 6
    n_angular_bins: int = 6
    smoothing_window: int = 3
    method: Literal["distance", "intensity"] = "distance"
    extent_margin: float = 0.05
    min_samples_per_ring: int = 5
    tau_core: float = 0.9
    tau_dense: float = 0.5
    tau_sparse: float = 0.1
    intensity_source: Literal["gray", "detect_mat"] = "gray"

    # Diagnostic cache populated by ``measure()`` so ``inspect()`` /
    # ``napari()`` can reuse the per-object intermediates. Pure runtime
    # state — never a constructor parameter — so modeled as private attrs.
    __cache_image: "Image | None" = PrivateAttr(default=None)
    __cache_props: "list | None" = PrivateAttr(default=None)
    __cache_intermediates: "dict[int, _SymmetryIntermediates]" = PrivateAttr(
            default_factory=dict)

    # ── shared pipeline for one object ───────────────────────────────

    @staticmethod
    def _distance_from_point(
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

    @staticmethod
    def _expand_slice_around_center(
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

    def _compute_intermediates(
            self,
            image: Image,
            object_label: int | None = None,
            prop=None,
    ) -> _SymmetryIntermediates:
        """Run the full symmetric-radius pipeline for a single object.

        Args:
            image: Detected Image with objmap/objmask.
            object_label: Specific object label to analyse. If *None*,
                the largest object by area is selected.
            prop: Pre-computed RegionProperties object. When provided the
                internal ``regionprops`` call is skipped.

        Returns:
            _SymmetryIntermediates with all computed fields populated.
        """
        # 1. Resolve target prop
        if prop is not None:
            target_prop = prop
        else:
            props = regionprops(
                image.objmap[:],
                intensity_image=image.gray[:].astype(np.float64, copy=False),
        )

            if object_label is not None:
                target_prop = None
                for p in props:
                    if p.label == object_label:
                        target_prop = p
                        break
                if target_prop is None:
                    raise ValueError(
                            f"Object label {object_label} not found in objmap."
                    )
            else:
                target_prop = max(props, key=lambda p: p.area)

        # 2. Early exit for tiny objects (all expansion fields zero, arrays empty)
        if target_prop.area < 10:
            empty = np.array([])
            tiny_mask = np.zeros((1, 1), dtype=bool)
            return _SymmetryIntermediates(
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
                    obj_mask=tiny_mask,
                    dist_map=np.zeros((1, 1), dtype=np.float64),
                    gray_crop=np.zeros((1, 1), dtype=np.float64),
                    core_end_radius=0.0,
                    dense_end_radius=0.0,
                    sparse_end_radius=0.0,
                    r_outer_full_per_angle=np.zeros(
                            _N_ANGULAR_SECTORS, dtype=np.float64),
                    core_area=0.0,
                    dense_area=0.0,
                    sparse_area=0.0,
                    colony_ness_profile=np.zeros(1, dtype=np.float64),
                    mean_profile=np.zeros(1, dtype=np.float64),
                    variance_profile=np.zeros(1, dtype=np.float64),
                    count_profile=np.zeros(1, dtype=np.int64),
                    I_core=0.0,
                    I_agar=0.0,
                    zones_computed=False,
            )

        # 3. Crop to bbox; compute local_mask, centroid, dist_map
        slc = target_prop.slice
        objmap_crop = image.objmap[:][slc]
        gray_crop = image.gray[:][slc]
        local_mask = objmap_crop == target_prop.label

        if self.method == "distance":
            dt = distance_transform_edt(local_mask)
            peak_idx = np.unravel_index(np.argmax(dt), dt.shape)
            local_cr = (float(peak_idx[0]), float(peak_idx[1]))
        else:
            cw = target_prop.centroid_weighted
            local_cr = (cw[0] - slc[0].start, cw[1] - slc[1].start)

        dist_map = self._distance_from_point(local_mask.shape, local_cr)

        # Auto-scale annuli: cap at the pixel-radius from the inoculum
        # centre to the farthest mask edge so annuli never become sub-pixel
        # wide, and floor at 6 (PELT minimum).
        max_pixel_radius = int(np.max(dist_map[local_mask]))
        effective_annuli = max(6, min(self.n_annuli, max_pixel_radius))

        # 4. Radial density profile
        density_profile, annulus_radii = self._compute_radial_density_profile(
                local_mask, dist_map, effective_annuli
        )

        # 5. Core radius via PELT changepoint detection
        core_radius = self._find_core_radius(
                density_profile, annulus_radii, self.pelt_penalty
        )

        # 6. Sholl-like angular profile
        sholl_counts, angular_R_profile, angular_coverage = (
            self._compute_sholl_angular_profile(
                    local_mask, dist_map, local_cr, annulus_radii, self.n_angular_bins,
            )
        )

        # 7. Symmetric radius (first radius where angular coverage drops
        #    below the symmetry threshold past core)
        symmetric_radius = self._find_symmetric_radius(
                annulus_radii,
                angular_coverage,
                core_radius,
                self.symmetry_threshold,
                self.smoothing_window,
        )

        # 8. Mean / max radial expansion past the core
        mean_expansion, max_expansion = self._compute_radial_expansion(
                local_mask, dist_map, core_radius,
        )

        # 9–. Zone segmentation (skip when no symmetric envelope).
        if symmetric_radius <= 0:
            r_outer_full_edge = self._per_angle_mask_envelope(
                    local_mask, dist_map, local_cr,
            )
            return _SymmetryIntermediates(
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
                    core_end_radius=0.0,
                    dense_end_radius=0.0,
                    sparse_end_radius=0.0,
                    r_outer_full_per_angle=r_outer_full_edge,
                    core_area=0.0,
                    dense_area=0.0,
                    sparse_area=0.0,
                    colony_ness_profile=np.zeros(1, dtype=np.float64),
                    mean_profile=np.zeros(1, dtype=np.float64),
                    variance_profile=np.zeros(1, dtype=np.float64),
                    count_profile=np.zeros(1, dtype=np.int64),
                    I_core=0.0,
                    I_agar=0.0,
                    zones_computed=False,
            )

        # 9. Expand the analysis crop past the farthest mask pixel by
        # ``extent_margin`` so the outermost annuli see a slice of agar.
        # The mask's role here ends — the ring signal pools every pixel
        # in each annulus regardless of mask membership.
        max_mask_radius = float(np.max(dist_map[local_mask]))
        r_max = max_mask_radius * (1.0 + float(self.extent_margin))
        center_global = (
            local_cr[0] + float(slc[0].start),
            local_cr[1] + float(slc[1].start),
        )
        image_shape = image.gray[:].shape[:2]
        expanded_slc = self._expand_slice_around_center(
                center_global, r_max, image_shape,
        )

        # 10. Re-crop arrays on the expanded slice.
        gray_crop_exp = image.gray[:][expanded_slc]
        if self.intensity_source == "detect_mat":
            intensity_crop = image.detect_mat[:][expanded_slc]
        else:
            intensity_crop = gray_crop_exp
        local_mask_exp = image.objmap[:][expanded_slc] == target_prop.label
        local_cr_exp = (
            center_global[0] - float(expanded_slc[0].start),
            center_global[1] - float(expanded_slc[1].start),
        )
        dist_map_exp = self._distance_from_point(
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
        _theta, r_bin, valid_geom = self._build_theta_r_maps(
                intensity_crop.shape,
                local_cr_exp,
                dist_map_exp,
                annulus_boundaries_exp,
                n_annuli,
        )
        mean_profile, variance_profile, count_profile = (
            self._accumulate_radial_profile(
                    r_bin, valid_geom, intensity_crop,
                    n_annuli, int(self.min_samples_per_ring),
            )
        )
        mask_per_annulus = self._accumulate_mask_per_annulus(
                r_bin, valid_geom & local_mask_exp, n_annuli,
        )

        # 13. Radial smoothing of the mean profile, then colony-ness
        # normalisation.
        from scipy.ndimage import uniform_filter1d

        mean_profile_smoothed = uniform_filter1d(
                mean_profile, size=_ZONE_RADIAL_SMOOTHING, mode="nearest",
        )
        colony_ness, I_core_val, I_agar_val = self._compute_colony_ness_profile(
                mean_profile_smoothed, intensity_crop, local_mask_exp,
        )

        # 14. Threshold crossings → scalar zone radii.
        core_end, dense_end, sparse_end = self._extract_zone_radii(
                colony_ness, mask_per_annulus, annulus_radii_exp,
                float(self.tau_core), float(self.tau_dense),
                float(self.tau_sparse), symmetric_radius,
        )

        # 15. Per-angle mask envelope for the diagnostic overlay only.
        r_outer_full_per_angle = self._per_angle_mask_envelope(
                local_mask_exp, dist_map_exp, local_cr_exp,
        )

        # 16. Concentric-disk zone areas.
        core_area, dense_area, sparse_area = self._compute_zone_areas(
                core_end, dense_end, sparse_end,
        )

        return _SymmetryIntermediates(
                label=target_prop.label,
                bbox_slice=expanded_slc,
                centroid_rc=local_cr_exp,
                density_profile=density_profile,
                annulus_radii=annulus_radii,
                core_radius=core_radius,
                sholl_counts=sholl_counts,
                angular_R_profile=angular_R_profile,
                angular_coverage=angular_coverage,
                symmetric_radius=symmetric_radius,
                mean_expansion=mean_expansion,
                max_expansion=max_expansion,
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
        )

    # ── MeasureFeatures interface ────────────────────────────────────

    def _operate(self, image: Image) -> pd.DataFrame:
        """Populate the symmetric-radius measurement DataFrame.

        Args:
            image: Detected Image with objmap/objmask.

        Returns:
            pd.DataFrame with one row per detected object and a leading
            ``OBJECT.LABEL`` column followed by the four SYMMETRIC_ZONES
            columns.
        """
        measurements = {
            str(feature): np.full(image.num_objects, np.nan)
            for feature in SYMMETRIC_ZONES
            if feature != SYMMETRIC_ZONES.CATEGORY
        }

        props = regionprops(
                image.objmap[:],
                intensity_image=image.gray[:].astype(np.float64, copy=False),
        )

        # Refresh cache so inspect() can reuse these results
        self.__cache_image = image
        self.__cache_props = props
        self.__cache_intermediates = {}

        # Zone columns get 0.0 for tiny objects (no zones to resolve);
        # the original four columns stay NaN to disambiguate "not measurable"
        # from legitimately-zero values.
        _zero_for_tiny = (
            SYMMETRIC_ZONES.CORE_END_RADIUS,
            SYMMETRIC_ZONES.DENSE_END_RADIUS,
            SYMMETRIC_ZONES.SPARSE_END_RADIUS,
            SYMMETRIC_ZONES.CORE_AREA,
            SYMMETRIC_ZONES.DENSE_AREA,
            SYMMETRIC_ZONES.SPARSE_AREA,
        )

        for idx, prop in enumerate(props):
            if prop.area < 10:
                for feat in _zero_for_tiny:
                    measurements[str(feat)][idx] = 0.0
                continue

            try:
                inter = self._compute_intermediates(image, prop.label, prop=prop)
            except Exception:
                import logging

                logging.getLogger(__name__).debug(
                        "Skipping object label %d", prop.label, exc_info=True,
                )
                continue  # leave NaN

            self.__cache_intermediates[prop.label] = inter

            measurements[str(SYMMETRIC_ZONES.CORE_RADIUS)][idx] = inter.core_radius
            measurements[str(SYMMETRIC_ZONES.SYMMETRIC_RADIUS)][
                idx] = inter.symmetric_radius
            measurements[str(SYMMETRIC_ZONES.MEAN_EXPANSION)][
                idx] = inter.mean_expansion
            measurements[str(SYMMETRIC_ZONES.MAX_EXPANSION)][idx] = inter.max_expansion
            measurements[str(SYMMETRIC_ZONES.CORE_AREA)][idx] = inter.core_area
            measurements[str(SYMMETRIC_ZONES.DENSE_AREA)][idx] = inter.dense_area
            measurements[str(SYMMETRIC_ZONES.SPARSE_AREA)][idx] = inter.sparse_area
            measurements[str(SYMMETRIC_ZONES.CORE_END_RADIUS)][
                idx] = inter.core_end_radius
            measurements[str(SYMMETRIC_ZONES.DENSE_END_RADIUS)][
                idx] = inter.dense_end_radius
            measurements[str(SYMMETRIC_ZONES.SPARSE_END_RADIUS)][
                idx] = inter.sparse_end_radius

        df = pd.DataFrame(measurements)
        df.insert(0, OBJECT.LABEL, image.objects.labels2series())
        return df

    # ── cache helpers ────────────────────────────────────────────────

    def _require_cache_image(self) -> Image:
        """Return the cached image or raise if :meth:`measure` has not run."""
        if self.__cache_image is None:
            raise RuntimeError(
                    "MeasureSymmetricZones: diagnostic cache is empty. "
                    "Call .measure(image) before .inspect()."
            )
        return self.__cache_image

    def _get_local_obj_mask(self, label: int) -> np.ndarray:
        """Local (bbox-cropped) boolean mask for the object with ``label``."""
        inter = self.__cache_intermediates[label]
        return inter.obj_mask

    def _get_local_dist_map(self, label: int) -> np.ndarray:
        """Local distance-from-centroid map for the object with ``label``."""
        inter = self.__cache_intermediates[label]
        return inter.dist_map

    def _get_global_offset(self, label: int) -> tuple[int, int]:
        """Top-left (row, col) offset from the local bbox to plate coordinates."""
        inter = self.__cache_intermediates[label]
        slc = inter.bbox_slice
        return int(slc[0].start), int(slc[1].start)

    # ── static helpers ───────────────────────────────────────────────

    @staticmethod
    def _compute_radial_density_profile(
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

    @staticmethod
    def _find_core_radius(
            density_profile: np.ndarray,
            annulus_radii: np.ndarray,
            pelt_penalty: float,
    ) -> float:
        """Find the core radius via PELT changepoint detection on the density profile.

        Args:
            density_profile: 1D radial density signal from
                ``_compute_radial_density_profile``.
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

    @staticmethod
    def _extract_mask_boundary(obj_mask: np.ndarray) -> np.ndarray:
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

    @staticmethod
    def _compute_sholl_angular_profile(
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
        :meth:`_compute_radial_density_profile`), this function measures:

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
                the profile from :meth:`_compute_radial_density_profile`.
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
        boundary = MeasureSymmetricZones._extract_mask_boundary(obj_mask)

        mask_coords = np.argwhere(obj_mask)  # (N, 2) row, col
        mask_radii = dist_map[obj_mask]
        dr = mask_coords[:, 0] - centroid_rc[0]
        dc = mask_coords[:, 1] - centroid_rc[1]
        angles = np.arctan2(dr, dc)  # (-pi, pi]

        # 2. Reconstruct annulus boundaries from annulus_radii (inverse of the
        #    equal-area sqrt construction used by _compute_radial_density_profile).
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

    @staticmethod
    def _find_symmetric_radius(
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

    @staticmethod
    def _compute_radial_expansion(
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

        boundary = MeasureSymmetricZones._extract_mask_boundary(obj_mask)
        if boundary.any():
            mean_extent = float(np.mean(dist_map[boundary]))
        else:
            mean_extent = float(np.mean(dist_map[obj_mask]))

        max_extent = float(np.max(dist_map[obj_mask]))

        mean_expansion = max(0.0, mean_extent - float(core_radius))
        max_expansion = max(0.0, max_extent - float(core_radius))
        return mean_expansion, max_expansion

    @staticmethod
    def _per_angle_mask_envelope(
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

    @staticmethod
    def _build_theta_r_maps(
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

    @staticmethod
    def _accumulate_radial_profile(
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

    @staticmethod
    def _accumulate_mask_per_annulus(
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

    @staticmethod
    def _compute_colony_ness_profile(
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

    @staticmethod
    def _extract_zone_radii(
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

    @staticmethod
    def _compute_zone_areas(
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

    # ── overlay polygon helpers ──────────────────────────────────────

    @staticmethod
    def _object_polygon_xy(
            local_mask: np.ndarray,
            slc: tuple[slice, slice],
            tolerance: float,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Largest simplified mask contour as plate-coordinate (x, y) arrays.

        The mask is padded by one zero pixel before contour extraction so
        objects that touch the local bbox edge still produce closed
        contours. Without padding, ``find_contours`` returns an open
        curve for edge-touching masks and plotly's ``fill="toself"``
        closes it with a straight last→first segment that slices across
        the object.

        Args:
            local_mask: Boolean mask in local bbox coordinates.
            slc: ``(row_slice, col_slice)`` mapping the local bbox to
                plate coordinates.
            tolerance: Douglas-Peucker tolerance in pixels.

        Returns:
            ``(xs, ys)`` float ndarrays for the simplified polygon in
            plate coordinates, or ``(None, None)`` if no usable contour
            exists (mask empty or contour collapses below 3 vertices).
        """
        if not local_mask.any():
            return None, None

        padded = np.pad(local_mask, 1, mode="constant", constant_values=False)
        contours = find_contours(padded.astype(np.float64), 0.5)
        if not contours:
            return None, None

        contour = max(contours, key=len)
        if contour.shape[0] < 3:
            return None, None

        simplified = approximate_polygon(contour, tolerance=tolerance)
        if simplified.shape[0] < 3:
            return None, None

        # Subtract 1 to undo the padding offset before mapping to plate coords.
        r0 = float(slc[0].start)
        c0 = float(slc[1].start)
        ys = simplified[:, 0] - 1.0 + r0
        xs = simplified[:, 1] - 1.0 + c0
        return xs, ys

    @staticmethod
    def _polar_polygon_xy(
            centroid_xy: tuple[float, float],
            r_per_angle: np.ndarray,
            stride: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Closed polar polygon as plate-coordinate (x, y) arrays.

        ``r_per_angle`` is the per-degree radius array (length 360);
        ``stride`` downsamples it (``stride=5`` → 72 vertices). The
        polygon is closed by repeating the first vertex.

        Args:
            centroid_xy: ``(cx, cy)`` centroid in plate coordinates.
            r_per_angle: (360,) per-degree radii.
            stride: Subsampling stride applied to the angle axis.

        Returns:
            Tuple ``(xs, ys)`` of float64 arrays, both length
            ``ceil(360 / stride) + 1`` (closed).
        """
        cx, cy = centroid_xy
        angles_deg = np.arange(0, _N_ANGULAR_SECTORS, stride, dtype=np.int32)
        theta = np.deg2rad(angles_deg)
        radii = r_per_angle[angles_deg].astype(np.float64)
        xs = cx + radii * np.cos(theta)
        ys = cy + radii * np.sin(theta)
        xs = np.concatenate([xs, xs[:1]])
        ys = np.concatenate([ys, ys[:1]])
        return xs, ys

    @staticmethod
    def _polar_annulus_xy(
            centroid_xy: tuple[float, float],
            r_inner_per_angle: np.ndarray,
            r_outer_per_angle: np.ndarray,
            stride: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Closed polar annulus as plate-coordinate (x, y) arrays.

        Traces the outer ring forward then the inner ring in reverse,
        producing a ring polygon that plotly's ``fill="toself"`` renders
        correctly (the reversed inner winding subtracts the inner disk
        from the outer disk without self-intersection).

        Args:
            centroid_xy: ``(cx, cy)`` centroid in plate coordinates.
            r_inner_per_angle: (360,) per-degree inner radii.
            r_outer_per_angle: (360,) per-degree outer radii.
            stride: Subsampling stride applied to the angle axis.

        Returns:
            Tuple ``(xs, ys)`` of float64 arrays, closed.
        """
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        cx, cy = centroid_xy
        angles_deg = np.arange(0, _N_ANGULAR_SECTORS, stride, dtype=np.int32)
        theta = np.deg2rad(angles_deg)
        r_out = r_outer_per_angle[angles_deg].astype(np.float64)
        r_in = r_inner_per_angle[angles_deg].astype(np.float64)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        outer_x = cx + r_out * cos_t
        outer_y = cy + r_out * sin_t
        inner_x = cx + r_in * cos_t
        inner_y = cy + r_in * sin_t
        # Outer ring forward, inner ring reversed, close back to the first
        # outer vertex. The opposite windings let plotly's non-zero-fill
        # "toself" render the ring (the inner traversal subtracts the hole).
        xs = np.concatenate([outer_x, inner_x[::-1], outer_x[:1]])
        ys = np.concatenate([outer_y, inner_y[::-1], outer_y[:1]])
        return xs, ys

    # ── diagnostics ──────────────────────────────────────────────────

    @figure(
            title="Symmetric-radius overlay",
            primary=True,
            controls={"base_layer": BASE_LAYER},
    )
    def inspect(
            self,
            image: Image | None = None,
            base_layer: Literal["rgb", "gray", "detect_mat"] = "gray",
            *,
            for_save: bool = False,
    ):
        """Plate-level diagnostic overlay for symmetric-radius measurement.

        Args:
            image: Detected Image with objmap/objmask. If *None*, the
                image cached by the most recent :meth:`measure` call is
                reused.
            base_layer: Which image array to use as the plotly background.
            for_save: When *True*, every overlay trace is force-shown
                (no ``visible="legendonly"``) so the figure renders
                meaningfully as a static raster. The CLI's
                ``--save-inspect`` flag passes this. Defaults to *False*
                (interactive Jupyter use, with overlay layers toggleable
                from the legend).

        Returns:
            plotly.graph_objects.Figure with toggleable overlay layers.
            Renders natively in Jupyter via the plotly mime bundle. For
            scroll-to-zoom, call
            ``fig.show(config={"scrollZoom": True})``.
        """
        from phenotypic.tools_._plotly_helpers import _require_plotly

        _require_plotly()

        import plotly.graph_objects as go

        if image is None:
            image = self._require_cache_image()

        props = regionprops(
                image.objmap[:],
                intensity_image=image.gray[:].astype(np.float64, copy=False),
        )
        intermediates_cache: dict[int, _SymmetryIntermediates] = {}
        for prop in props:
            if (
                    self.__cache_image is image
                    and prop.label in self.__cache_intermediates
            ):
                intermediates_cache[prop.label] = self.__cache_intermediates[prop.label]
                continue
            try:
                inter = self._compute_intermediates(image, prop.label, prop=prop)
                intermediates_cache[prop.label] = inter
            except Exception:
                continue

        self.__cache_image = image
        self.__cache_props = props
        self.__cache_intermediates = intermediates_cache

        if not intermediates_cache:
            fig = go.Figure()
            fig.add_annotation(
                    text="No objects found for symmetric-radius analysis.",
                    xref="paper", yref="paper", x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(color=_OI_NAVY),  # family from the phenotypic template
            )
            return fig

        fig = self._build_plate_overview(
                image, intermediates_cache,
                base_layer=base_layer,
        )
        h, w = image.gray[:].shape[:2]
        overview_h = int(900 * h / w)

        fig.update_layout(
                title=dict(
                        text=f"Symmetric Radius -- {len(intermediates_cache)} objects",
                        font=dict(color=_OI_NAVY),  # family from the phenotypic template
                ),
                height=overview_h,
        )

        if for_save:
            for trace in fig.data:
                if getattr(trace, "visible", True) == "legendonly":
                    trace.visible = True
        return fig

    @staticmethod
    def _build_plate_overview(
            image: Image,
            intermediates_cache: dict[int, _SymmetryIntermediates],
            base_layer: str = "gray",
    ):
        """Build a zoomable plotly plate overview with toggleable overlays.

        Args:
            image: Detected Image with objmap/objmask.
            intermediates_cache: Pre-computed intermediates keyed by label.
            base_layer: Which image array to use as background.

        Returns:
            plotly.graph_objects.Figure.
        """
        from phenotypic.tools_._plotly_helpers import (
            plotly_imshow,
            add_plotly_obj_labels,
        )

        if base_layer == "rgb":
            arr = image.rgb[:]
        elif base_layer == "detect_mat":
            arr = image.detect_mat[:]
        else:
            arr = image.gray[:]

        h, w = arr.shape[:2]
        display_w = 900
        display_h = int(display_w * h / w)
        fig = plotly_imshow(
                arr, title="Plate Overview",
                figsize=(display_w // 100, display_h // 100),
        )
        fig.update_coloraxes(showscale=False)
        add_plotly_obj_labels(fig, image)

        MeasureSymmetricZones._add_overlay_traces(
                fig, image, intermediates_cache,
        )
        return fig

    @staticmethod
    def _add_overlay_traces(
            fig,
            image: Image,
            intermediates_cache: dict[int, _SymmetryIntermediates],
    ) -> None:
        """Add toggleable trace layers to the plate overview.

        Layers (all legend-toggleable):
        - **Objmap** — simplified mask contour per object, filled with a
          label-keyed overlay palette.
        - **Zones** — three legend entries grouped under the "Zones"
          heading: **Sparse zone** (sky, annulus r_dense_end..r_outer),
          **Dense zone** (navy, annulus r_core..r_dense_end), and
          **Core zone** (vermilion, disk 0..r_core). ``r_outer`` is capped
          at the symmetric envelope. The legend group is set to toggle
          as a group, so clicking any entry hides/shows all three; the
          entries can also be toggled individually.
        - **Centroids** — inoculum centre markers for every object.
        - **Core radius** — dashed vermilion circles.
        - **Symmetric radius** — purple dashed circles (all objects).
        - **Outer envelope** — solid green per-angle polygon tracing the
          uncapped mask reach per object. Drawn for every object with any
          mask pixels, including those whose symmetry thresholding failed
          (``zones_computed=False``); those objects intentionally have no
          Zones traces.

        Args:
            fig: Plotly figure to modify in-place.
            image: Detected Image (used for objmap overlay).
            intermediates_cache: Per-object intermediates keyed by label.
        """
        import plotly.graph_objects as go

        _N_CIRCLE_PTS = 72
        _theta = np.linspace(0, 2 * np.pi, _N_CIRCLE_PTS, endpoint=True)

        def _circle_xy(cx: float, cy: float, r: float):
            return cx + r * np.cos(_theta), cy + r * np.sin(_theta)

        def _hex_to_rgba(hex_str: str, alpha: float) -> str:
            h = hex_str.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r}, {g}, {b}, {alpha:.3f})"

        # Make grouped legend entries toggle every trace in the group.
        fig.update_layout(legend=dict(groupclick="togglegroup"))

        _FILL_ALPHA = 0.30

        # ── Objmap polygons (label-keyed palette, NaN-separated buckets) ──
        # Use the same overlay palette as skimage-based objmap overlays so
        # the Objmap layer here matches colors elsewhere in phenotypic, and
        # key it on ``label`` (not enumeration index) so neighbouring objects
        # cycle through the whole palette rather than colliding.
        from phenotypic._core._image_parts.accessor_abstracts._image_accessor_base_parents._accessor_mpl_handler import (  # noqa: E501
            AccessorMplHandler,
        )

        _palette_rgb = (AccessorMplHandler._OVERLAY_COLORS * 255).astype(np.int32)
        n_palette = len(_palette_rgb)

        bucket_xs: list[list[float]] = [[] for _ in range(n_palette)]
        bucket_ys: list[list[float]] = [[] for _ in range(n_palette)]
        sorted_inters = sorted(
                intermediates_cache.values(), key=lambda x: x.label,
        )
        for inter in sorted_inters:
            xs, ys = MeasureSymmetricZones._object_polygon_xy(
                    inter.obj_mask, inter.bbox_slice, _OBJMAP_POLYGON_TOLERANCE,
            )
            if xs is None or ys is None:
                continue
            # Labels start at 1; subtract before mod so label 1 → palette 0.
            bi = (int(inter.label) - 1) % n_palette
            bucket_xs[bi].extend(xs.tolist())
            bucket_xs[bi].append(float("nan"))
            bucket_ys[bi].extend(ys.tolist())
            bucket_ys[bi].append(float("nan"))

        first_objmap_drawn = False
        for bi, (bxs, bys) in enumerate(zip(bucket_xs, bucket_ys)):
            if not bxs:
                continue
            r, g, b = _palette_rgb[bi]
            fillcolor = f"rgba({int(r)}, {int(g)}, {int(b)}, {_FILL_ALPHA:.3f})"
            fig.add_trace(go.Scatter(
                    x=bxs, y=bys, mode="lines",
                    line=dict(width=0),
                    fill="toself",
                    fillcolor=fillcolor,
                    legendgroup="objmap",
                    name="Objmap",
                    showlegend=not first_objmap_drawn,
                    visible="legendonly",
                    hoverinfo="skip",
            ))
            first_objmap_drawn = True

        # ── Zone polygons — nested annuli + core disk ───────────────
        # Core: solid disk 0..r_core. Dense: annulus r_core..r_dense_end.
        # Sparse: annulus r_dense_end..r_outer (capped at symmetric radius).
        # Drawn sparse → dense → core so narrower zones render on top of
        # wider ones and the legend-group toggle handles all three at once.
        sparse_xs: list[float] = []
        sparse_ys: list[float] = []
        dense_xs: list[float] = []
        dense_ys: list[float] = []
        zcore_xs: list[float] = []
        zcore_ys: list[float] = []

        for inter in intermediates_cache.values():
            if not inter.zones_computed:
                continue
            slc = inter.bbox_slice
            r0, c0 = int(slc[0].start), int(slc[1].start)
            cxy = (inter.centroid_rc[1] + c0, inter.centroid_rc[0] + r0)

            # Zone radii are scalars; broadcast to per-angle arrays so the
            # existing polar-polygon helpers draw concentric circles.
            r_core_arr = np.full(
                    _N_ANGULAR_SECTORS, inter.core_end_radius, dtype=np.float64,
            )
            r_dense_arr = np.full(
                    _N_ANGULAR_SECTORS, inter.dense_end_radius, dtype=np.float64,
            )
            r_outer_arr = np.full(
                    _N_ANGULAR_SECTORS, inter.sparse_end_radius, dtype=np.float64,
            )

            sxs, sys = MeasureSymmetricZones._polar_annulus_xy(
                    cxy, r_dense_arr, r_outer_arr, _ZONE_POLYGON_STRIDE,
            )
            sparse_xs.extend(sxs.tolist())
            sparse_xs.append(float("nan"))
            sparse_ys.extend(sys.tolist())
            sparse_ys.append(float("nan"))

            dxs, dys = MeasureSymmetricZones._polar_annulus_xy(
                    cxy, r_core_arr, r_dense_arr, _ZONE_POLYGON_STRIDE,
            )
            dense_xs.extend(dxs.tolist())
            dense_xs.append(float("nan"))
            dense_ys.extend(dys.tolist())
            dense_ys.append(float("nan"))

            cxs, cys = MeasureSymmetricZones._polar_polygon_xy(
                    cxy, r_core_arr, _ZONE_POLYGON_STRIDE,
            )
            zcore_xs.extend(cxs.tolist())
            zcore_xs.append(float("nan"))
            zcore_ys.extend(cys.tolist())
            zcore_ys.append(float("nan"))

        # Three legend entries share legendgroup="zones"; groupclick is set
        # to "togglegroup" above so clicking any of them toggles all three at
        # once, while individual toggles are still available.
        zone_layers = [
            (sparse_xs, sparse_ys, _OI_SKY, "Sparse zone"),
            (dense_xs, dense_ys, _OI_NAVY, "Dense zone"),
            (zcore_xs, zcore_ys, _OI_VERMILION, "Core zone"),
        ]
        for zxs, zys, zcolor, zname in zone_layers:
            if not zxs:
                continue
            fig.add_trace(go.Scatter(
                    x=zxs, y=zys, mode="lines",
                    line=dict(width=0),
                    fill="toself",
                    fillcolor=_hex_to_rgba(zcolor, _FILL_ALPHA),
                    legendgroup="zones",
                    legendgrouptitle_text="Zones",
                    name=zname,
                    visible="legendonly",
                    hoverinfo="skip",
            ))

        # ── Centroids ───────────────────────────────────────────────
        cent_x, cent_y = [], []
        for inter in intermediates_cache.values():
            slc = inter.bbox_slice
            r0, c0 = int(slc[0].start), int(slc[1].start)
            cent_x.append(inter.centroid_rc[1] + c0)
            cent_y.append(inter.centroid_rc[0] + r0)
        if cent_x:
            fig.add_trace(go.Scatter(
                    x=cent_x, y=cent_y, mode="markers",
                    marker=dict(
                            color=_OI_ORANGE, size=8, symbol="circle",
                            line=dict(color=_OI_NAVY, width=1),
                    ),
                    name="Centroids",
                    hoverinfo="skip",
            ))

        # ── Core radius circles (all objects) ───────────────────────
        core_x: list[float] = []
        core_y: list[float] = []
        for inter in intermediates_cache.values():
            if inter.core_radius <= 0:
                continue
            slc = inter.bbox_slice
            r0, c0 = int(slc[0].start), int(slc[1].start)
            cx = inter.centroid_rc[1] + c0
            cy = inter.centroid_rc[0] + r0
            xs, ys = _circle_xy(cx, cy, inter.core_radius)
            core_x.extend(xs.tolist())
            core_y.extend(ys.tolist())
            core_x.append(float("nan"))
            core_y.append(float("nan"))
        if core_x:
            fig.add_trace(go.Scattergl(
                    x=core_x, y=core_y, mode="lines",
                    line=dict(color=_OI_VERMILION, width=1.5, dash="dash"),
                    name="Core radius",
                    hoverinfo="skip",
            ))

        # ── Symmetric radius circles (all objects) ──────────────────
        sym_x: list[float] = []
        sym_y: list[float] = []
        for inter in intermediates_cache.values():
            sr = inter.symmetric_radius
            if sr <= 0 or not np.isfinite(sr):
                continue
            slc = inter.bbox_slice
            r0, c0 = int(slc[0].start), int(slc[1].start)
            cx = inter.centroid_rc[1] + c0
            cy = inter.centroid_rc[0] + r0
            xs, ys = _circle_xy(cx, cy, sr)
            sym_x.extend(xs.tolist())
            sym_y.extend(ys.tolist())
            sym_x.append(float("nan"))
            sym_y.append(float("nan"))
        if sym_x:
            fig.add_trace(go.Scattergl(
                    x=sym_x, y=sym_y, mode="lines",
                    line=dict(color=_OI_PURPLE, width=2.5, dash="dash"),
                    name="Symmetric radius",
                    hoverinfo="skip",
            ))

        # ── Outer envelope polygons (per-angle, uncapped mask reach) ──
        env_x: list[float] = []
        env_y: list[float] = []
        for inter in intermediates_cache.values():
            r_env = inter.r_outer_full_per_angle
            if float(r_env.max()) <= 0:
                continue
            slc = inter.bbox_slice
            r0, c0 = int(slc[0].start), int(slc[1].start)
            cxy = (inter.centroid_rc[1] + c0, inter.centroid_rc[0] + r0)
            xs, ys = MeasureSymmetricZones._polar_polygon_xy(
                    cxy, r_env, _ZONE_POLYGON_STRIDE,
            )
            env_x.extend(xs.tolist())
            env_y.extend(ys.tolist())
            env_x.append(float("nan"))
            env_y.append(float("nan"))
        if env_x:
            fig.add_trace(go.Scattergl(
                    x=env_x, y=env_y, mode="lines",
                    line=dict(color=_OI_GREEN, width=1.25),
                    name="Outer envelope",
                    hoverinfo="skip",
            ))


MeasureSymmetricZones.__doc__ = SYMMETRIC_ZONES.append_rst_to_doc(
        MeasureSymmetricZones
)
