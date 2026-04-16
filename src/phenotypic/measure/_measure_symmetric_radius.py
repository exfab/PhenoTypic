"""Mask-based radial expansion and symmetry measurement operator.

Implements :class:`MeasureSymmetricRadius`, a branch-free alternative to
:class:`MeasureRadialExpansion` that answers two colony-level questions
directly from the binary object mask: how far has growth progressed past
the inoculum, and out to what radius does that growth remain angularly
uniform?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
import pandas as pd
from scipy.ndimage import convolve, distance_transform_edt
from skimage.measure import regionprops

from phenotypic.abc_ import MeasureFeatures
from phenotypic.tools_.constants_ import OBJECT
from ..tools_.measurement_info_ import SYMMETRIC_RADIUS

_NEIGHBOR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)

# Zone-segmentation constants
_N_ANGULAR_SECTORS = 360
_SECTOR_HALFWIDTH_DEG = 1
_ZONE_RADIAL_SMOOTHING = 3
_ZONE_ANGULAR_SMOOTHING_DEG = 5

# Okabe-Ito palette constants for diagnostic plots
_OI_NAVY = "#003660"
_OI_ORANGE = "#E69F00"
_OI_SKY = "#56B4E9"
_OI_GREEN = "#009E73"
_OI_BLUE = "#0072B2"
_OI_PURPLE = "#CC79A7"
_OI_VERMILION = "#D55E00"
_OI_GREY = "#BBBBBB"


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
    r_core_per_angle: np.ndarray = field(
        default_factory=lambda: np.zeros(_N_ANGULAR_SECTORS, dtype=np.float64))
    r_dense_end_per_angle: np.ndarray = field(
        default_factory=lambda: np.zeros(_N_ANGULAR_SECTORS, dtype=np.float64))
    r_outer_per_angle: np.ndarray = field(
        default_factory=lambda: np.zeros(_N_ANGULAR_SECTORS, dtype=np.float64))
    core_area: float = 0.0
    dense_area: float = 0.0
    sparse_area: float = 0.0
    bright_fraction_tensor: np.ndarray = field(
        default_factory=lambda: np.zeros((_N_ANGULAR_SECTORS, 1), dtype=np.float64))
    zones_computed: bool = False


class MeasureSymmetricRadius(MeasureFeatures):
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
        tau_core: Minimum bright-pixel fraction required for an annulus
            to be classified as inoculum core in the per-angle outward
            walk. Defaults to 0.9.
        tau_sparse: Minimum bright-pixel fraction required for an annulus
            to be classified as dense branching (the outer edge of the
            dense zone). Defaults to 0.5.
        bright_intensity_fraction: Fraction of the core's median intensity
            used as the bright/background threshold for zone segmentation.
            Defaults to 0.5.
        intensity_source: Image array used for the brightness comparison
            -- ``"gray"`` uses the grayscale, ``"detect_mat"`` uses the
            detection matrix. Defaults to ``"gray"``.

    Returns:
        pd.DataFrame: Object-level radial symmetry measurements with
        columns:

            - ObjectLabel: unique object identifier.
            - SymmetricRadius_CoreRadius: inoculum core radius (pixels).
            - SymmetricRadius_SymmetricRadius: first radius past the core
              where R̄ exceeds the symmetry threshold (pixels).
            - SymmetricRadius_MeanExpansion: mean boundary-pixel distance
              beyond the core (pixels, clamped at 0).
            - SymmetricRadius_MaxExpansion: maximum mask-pixel distance
              beyond the core (pixels, clamped at 0).
            - SymmetricRadius_CoreEndRadius: mean per-angle core boundary
              radius from the bright-fraction outward walk (pixels).
            - SymmetricRadius_DenseEndRadius: mean per-angle outer radius
              of the dense branching zone (pixels).
            - SymmetricRadius_SparseEndRadius: mean per-angle outer radius
              of the sparse branching zone, capped at the symmetric
              envelope (pixels).
            - SymmetricRadius_CoreArea: pixel^2 area of the inoculum core
              zone integrated across the 360-sector polar polygon.
            - SymmetricRadius_DenseArea: pixel^2 area of the dense
              branching zone (annular region between core and dense
              boundaries).
            - SymmetricRadius_SparseArea: pixel^2 area of the sparse
              branching zone (annular region between dense and outer
              boundaries).

    See Also:
        :doc:`/tutorials/notebooks/07_measuring_and_exporting` for a
        walkthrough of measuring and exporting colony data.
    """

    _measurement_info_class = SYMMETRIC_RADIUS

    def __init__(
            self,
            n_annuli: int = 100,
            pelt_penalty: float = 5.0,
            symmetry_threshold: float = 4 / 6,
            n_angular_bins: int = 6,
            smoothing_window: int = 3,
            method: Literal["distance", "intensity"] = "distance",
            tau_core: float = 0.9,
            tau_sparse: float = 0.5,
            bright_intensity_fraction: float = 0.5,
            intensity_source: Literal["gray", "detect_mat"] = "gray",
    ):
        self.n_annuli = n_annuli
        self.pelt_penalty = pelt_penalty
        self.symmetry_threshold = symmetry_threshold
        self.n_angular_bins = n_angular_bins
        self.smoothing_window = smoothing_window
        self.method = method
        self.tau_core = tau_core
        self.tau_sparse = tau_sparse
        self.bright_intensity_fraction = bright_intensity_fraction
        self.intensity_source = intensity_source

        self.__cache_image: Image | None = None
        self.__cache_props: list | None = None
        self.__cache_intermediates: dict[int, _SymmetryIntermediates] = {}

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
            props = regionprops(image.objmap[:], intensity_image=image.gray[:])

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
                    r_core_per_angle=np.zeros(_N_ANGULAR_SECTORS, dtype=np.float64),
                    r_dense_end_per_angle=np.zeros(
                            _N_ANGULAR_SECTORS, dtype=np.float64),
                    r_outer_per_angle=np.zeros(_N_ANGULAR_SECTORS, dtype=np.float64),
                    core_area=0.0,
                    dense_area=0.0,
                    sparse_area=0.0,
                    bright_fraction_tensor=np.zeros(
                            (_N_ANGULAR_SECTORS, 1), dtype=np.float64),
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

        # 9–16. Per-angle zone segmentation (skip when no symmetric envelope).
        if symmetric_radius <= 0:
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
                    r_core_per_angle=np.zeros(_N_ANGULAR_SECTORS, dtype=np.float64),
                    r_dense_end_per_angle=np.zeros(
                            _N_ANGULAR_SECTORS, dtype=np.float64),
                    r_outer_per_angle=np.zeros(_N_ANGULAR_SECTORS, dtype=np.float64),
                    core_area=0.0,
                    dense_area=0.0,
                    sparse_area=0.0,
                    bright_fraction_tensor=np.zeros(
                            (_N_ANGULAR_SECTORS, 1), dtype=np.float64),
                    zones_computed=False,
            )

        # 9. Brightness reference from PELT-detected core
        if self.intensity_source == "detect_mat":
            intensity_crop = image.detect_mat[:][slc]
        else:
            intensity_crop = image.gray[:][slc]
        _, bright_threshold = self._compute_brightness_reference(
                intensity_crop, local_mask, dist_map, core_radius,
                self.bright_intensity_fraction,
        )

        # 10. Per-pixel bright mask
        is_bright = (intensity_crop >= bright_threshold) & local_mask

        # 11. Build the (360, n_annuli) bright-fraction tensor
        n_annuli = int(annulus_radii.size)
        max_radius = float(np.max(dist_map[local_mask]))
        annulus_boundaries = max_radius * np.sqrt(
                np.arange(n_annuli + 1) / n_annuli
        )
        theta, r_bin, valid = self._build_theta_r_maps(
                local_mask, local_cr, dist_map, annulus_boundaries, n_annuli,
        )
        bright_fraction, mask_per_cell = self._accumulate_bright_fraction_tensor(
                theta, r_bin, valid, is_bright, n_annuli,
        )

        # 12. Radial smoothing
        from scipy.ndimage import uniform_filter1d
        bright_fraction = uniform_filter1d(
                bright_fraction, size=_ZONE_RADIAL_SMOOTHING,
                axis=1, mode="nearest",
        )

        # 13. Per-angle outward walk + zone radii (capped at symmetric_radius)
        r_core, r_dense_end, r_outer = self._extract_zone_boundaries(
                bright_fraction, mask_per_cell, annulus_radii,
                self.tau_core, self.tau_sparse, symmetric_radius,
        )

        # 14. Circular angular median filter (re-enforce nesting after smoothing)
        from scipy.ndimage import median_filter
        r_core = median_filter(r_core, size=_ZONE_ANGULAR_SMOOTHING_DEG, mode="wrap")
        r_dense_end = median_filter(
                r_dense_end, size=_ZONE_ANGULAR_SMOOTHING_DEG, mode="wrap")
        r_outer = median_filter(
                r_outer, size=_ZONE_ANGULAR_SMOOTHING_DEG, mode="wrap")
        r_core = np.minimum(r_core, r_dense_end)
        r_dense_end = np.minimum(r_dense_end, r_outer)

        # 15. Polar polygon area integration
        core_area, dense_area, sparse_area = self._compute_zone_areas(
                r_core, r_dense_end, r_outer,
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
                r_core_per_angle=r_core,
                r_dense_end_per_angle=r_dense_end,
                r_outer_per_angle=r_outer,
                core_area=core_area,
                dense_area=dense_area,
                sparse_area=sparse_area,
                bright_fraction_tensor=bright_fraction,
                zones_computed=True,
        )

    # ── MeasureFeatures interface ────────────────────────────────────

    def _operate(self, image: Image) -> pd.DataFrame:
        """Populate the symmetric-radius measurement DataFrame.

        Args:
            image: Detected Image with objmap/objmask.

        Returns:
            pd.DataFrame with one row per detected object and a leading
            ``OBJECT.LABEL`` column followed by the four SYMMETRIC_RADIUS
            columns.
        """
        measurements = {
            str(feature): np.full(image.num_objects, np.nan)
            for feature in SYMMETRIC_RADIUS
            if feature != SYMMETRIC_RADIUS.CATEGORY
        }

        props = regionprops(image.objmap[:], intensity_image=image.gray[:])

        # Refresh cache so inspect() can reuse these results
        self.__cache_image = image
        self.__cache_props = props
        self.__cache_intermediates = {}

        for idx, prop in enumerate(props):
            # Tiny objects can't produce meaningful measurements — leave NaN
            # to disambiguate "not measurable" from legitimately-zero values.
            if prop.area < 10:
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

            measurements[str(SYMMETRIC_RADIUS.CORE_RADIUS)][idx] = inter.core_radius
            measurements[str(SYMMETRIC_RADIUS.SYMMETRIC_RADIUS)][
                idx] = inter.symmetric_radius
            measurements[str(SYMMETRIC_RADIUS.MEAN_EXPANSION)][
                idx] = inter.mean_expansion
            measurements[str(SYMMETRIC_RADIUS.MAX_EXPANSION)][idx] = inter.max_expansion
            measurements[str(SYMMETRIC_RADIUS.CORE_AREA)][idx] = inter.core_area
            measurements[str(SYMMETRIC_RADIUS.DENSE_AREA)][idx] = inter.dense_area
            measurements[str(SYMMETRIC_RADIUS.SPARSE_AREA)][idx] = inter.sparse_area
            measurements[str(SYMMETRIC_RADIUS.CORE_END_RADIUS)][idx] = float(
                    inter.r_core_per_angle.mean())
            measurements[str(SYMMETRIC_RADIUS.DENSE_END_RADIUS)][idx] = float(
                    inter.r_dense_end_per_angle.mean())
            measurements[str(SYMMETRIC_RADIUS.SPARSE_END_RADIUS)][idx] = float(
                    inter.r_outer_per_angle.mean())

        df = pd.DataFrame(measurements)
        df.insert(0, OBJECT.LABEL, image.objects.labels2series())
        return df

    # ── cache helpers ────────────────────────────────────────────────

    def _require_cache_image(self) -> Image:
        """Return the cached image or raise if :meth:`measure` has not run."""
        if self.__cache_image is None:
            raise RuntimeError(
                    "MeasureSymmetricRadius: diagnostic cache is empty. "
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
        boundary = MeasureSymmetricRadius._extract_mask_boundary(obj_mask)

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

        boundary = MeasureSymmetricRadius._extract_mask_boundary(obj_mask)
        if boundary.any():
            mean_extent = float(np.mean(dist_map[boundary]))
        else:
            mean_extent = float(np.mean(dist_map[obj_mask]))

        max_extent = float(np.max(dist_map[obj_mask]))

        mean_expansion = max(0.0, mean_extent - float(core_radius))
        max_expansion = max(0.0, max_extent - float(core_radius))
        return mean_expansion, max_expansion

    # ── zone segmentation helpers ────────────────────────────────────

    @staticmethod
    def _compute_brightness_reference(
            intensity: np.ndarray,
            local_mask: np.ndarray,
            dist_map: np.ndarray,
            core_radius: float,
            fraction: float,
    ) -> tuple[float, float]:
        """Median intensity of the PELT-detected core and the bright threshold.

        Args:
            intensity: Local-bbox intensity crop (gray or detect_mat).
            local_mask: Boolean mask of the object in local coordinates.
            dist_map: Distance-from-centroid map.
            core_radius: PELT-detected core radius (pixels). When 0, the
                full object interior is used as fallback.
            fraction: Multiplier on the reference intensity that defines
                the bright/background threshold.

        Returns:
            Tuple ``(reference_intensity, bright_threshold)``.
        """
        if core_radius > 0:
            core_pixels = intensity[(dist_map < core_radius) & local_mask]
            if core_pixels.size == 0:
                core_pixels = intensity[local_mask]
        else:
            core_pixels = intensity[local_mask]
        reference = float(np.median(core_pixels))
        return reference, float(fraction) * reference

    @staticmethod
    def _build_theta_r_maps(
            local_mask: np.ndarray,
            centroid_rc: tuple[float, float],
            dist_map: np.ndarray,
            annulus_boundaries: np.ndarray,
            n_annuli: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-pixel angular sector and annulus index plus a validity mask.

        Args:
            local_mask: Boolean mask of the object in local coordinates.
            centroid_rc: (row, col) centroid in local coordinates.
            dist_map: Distance-from-centroid map (same shape as mask).
            annulus_boundaries: Equal-area annulus boundary radii of
                length ``n_annuli + 1``.
            n_annuli: Number of annular bins.

        Returns:
            Tuple ``(theta, r_bin, valid)`` arrays sharing the mask shape.
            ``theta`` is integer degrees in [0, 360), ``r_bin`` is the
            annulus index in [0, n_annuli), and ``valid`` selects pixels
            inside the mask with a finite, in-range annulus assignment.
        """
        h, w = local_mask.shape
        rows, cols = np.indices((h, w))
        dr = rows - centroid_rc[0]
        dc = cols - centroid_rc[1]
        theta = np.mod(np.degrees(np.arctan2(dr, dc)), 360.0).astype(np.int16)
        r_bin = np.digitize(dist_map, annulus_boundaries) - 1
        valid = (
                local_mask
                & (r_bin >= 0)
                & (r_bin < n_annuli)
                & (dist_map > 0)
        )
        return theta, r_bin, valid

    @staticmethod
    def _accumulate_bright_fraction_tensor(
            theta: np.ndarray,
            r_bin: np.ndarray,
            valid: np.ndarray,
            is_bright: np.ndarray,
            n_annuli: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """(360, n_annuli) bright-fraction and mask-occupancy tensors.

        Args:
            theta: Per-pixel angular sector (int degrees in [0, 360)).
            r_bin: Per-pixel annulus index.
            valid: Boolean per-pixel selector for in-mask, in-range pixels.
            is_bright: Per-pixel bright/background classification.
            n_annuli: Number of annular bins.

        Returns:
            Tuple ``(bright_fraction, mask_per_cell)``. ``bright_fraction``
            is the per-(angle, annulus) ratio of bright to total pixels
            (NaN where total == 0); ``mask_per_cell`` is the integer count
            of mask pixels per cell, used to derive the outer envelope.
        """
        th = theta[valid].astype(np.int32)
        rb = r_bin[valid].astype(np.int32)
        br = is_bright[valid].astype(np.int32)

        offsets = np.array([-1, 0, 1], dtype=np.int32)
        th3 = np.mod(th[:, None] + offsets[None, :], _N_ANGULAR_SECTORS).ravel()
        rb3 = np.broadcast_to(rb[:, None], (rb.size, 3)).ravel()
        br3 = np.broadcast_to(br[:, None], (br.size, 3)).ravel()

        flat_idx = th3 * n_annuli + rb3
        total = np.bincount(
                flat_idx, minlength=_N_ANGULAR_SECTORS * n_annuli,
        ).reshape(_N_ANGULAR_SECTORS, n_annuli)
        bright = np.bincount(
                flat_idx, weights=br3.astype(np.float64),
                minlength=_N_ANGULAR_SECTORS * n_annuli,
        ).reshape(_N_ANGULAR_SECTORS, n_annuli)

        with np.errstate(invalid="ignore", divide="ignore"):
            bright_fraction = np.where(total > 0, bright / total, np.nan)
        return bright_fraction, total

    @staticmethod
    def _extract_zone_boundaries(
            bright_fraction: np.ndarray,
            mask_per_cell: np.ndarray,
            annulus_radii: np.ndarray,
            tau_core: float,
            tau_sparse: float,
            symmetric_radius: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-angle core / dense / outer radii from the bright-fraction tensor.

        Args:
            bright_fraction: (360, n_annuli) bright-pixel ratio tensor.
            mask_per_cell: (360, n_annuli) mask-occupancy tensor.
            annulus_radii: Centre radii of the equal-area annuli.
            tau_core: Bright-fraction threshold for inoculum core.
            tau_sparse: Bright-fraction threshold for dense branching.
            symmetric_radius: Cap applied to all per-angle radii.

        Returns:
            Tuple of three (360,) float arrays
            ``(r_core, r_dense_end, r_outer)``.
        """
        n_annuli = annulus_radii.size

        def _prefix_pass_count(cond: np.ndarray) -> np.ndarray:
            return np.logical_and.accumulate(cond, axis=1).sum(axis=1)

        def _radii_from_prefix(prefix: np.ndarray) -> np.ndarray:
            out = np.zeros(_N_ANGULAR_SECTORS, dtype=np.float64)
            nonzero = prefix > 0
            out[nonzero] = annulus_radii[
                np.clip(prefix[nonzero] - 1, 0, n_annuli - 1)
            ]
            return out

        cond_core = bright_fraction >= tau_core
        cond_dense = bright_fraction >= tau_sparse
        prefix_core = _prefix_pass_count(cond_core)
        prefix_dense = _prefix_pass_count(cond_dense)
        prefix_outer = _prefix_pass_count(mask_per_cell > 0)

        r_core = np.minimum(_radii_from_prefix(prefix_core), symmetric_radius)
        r_dense_end = np.minimum(_radii_from_prefix(prefix_dense), symmetric_radius)
        r_outer = np.minimum(_radii_from_prefix(prefix_outer), symmetric_radius)
        return r_core, r_dense_end, r_outer

    @staticmethod
    def _compute_zone_areas(
            r_core: np.ndarray,
            r_dense_end: np.ndarray,
            r_outer: np.ndarray,
    ) -> tuple[float, float, float]:
        """Polar polygon areas (pixel^2) for the three nested zones.

        Args:
            r_core: Per-angle core boundary radius (360,).
            r_dense_end: Per-angle dense boundary radius (360,).
            r_outer: Per-angle outer envelope radius (360,).

        Returns:
            Tuple ``(core_area, dense_area, sparse_area)`` in pixel^2.
        """
        delta_theta = 2.0 * np.pi / _N_ANGULAR_SECTORS
        core_area = float(0.5 * delta_theta * np.sum(r_core ** 2))
        dense_area = float(
                0.5 * delta_theta * np.sum(
                        np.maximum(r_dense_end ** 2 - r_core ** 2, 0.0)
                )
        )
        sparse_area = float(
                0.5 * delta_theta * np.sum(
                        np.maximum(r_outer ** 2 - r_dense_end ** 2, 0.0)
                )
        )
        return core_area, dense_area, sparse_area

    # ── diagnostics ──────────────────────────────────────────────────

    def inspect(
            self,
            image: Image | None = None,
            base_layer: Literal["rgb", "gray", "detect_mat"] = "gray",
    ):
        """Plate-level diagnostic overlay for symmetric-radius measurement.

        Args:
            image: Detected Image with objmap/objmask. If *None*, the
                image cached by the most recent :meth:`measure` call is
                reused.
            base_layer: Which image array to use as the plotly background.

        Returns:
            Panel Column layout with a single zoomable plotly figure
            containing toggleable overlay layers.
        """
        from phenotypic.tools_.panel_ import require_panel, ensure_panel_extension

        require_panel()
        ensure_panel_extension()

        import panel as pn
        from phenotypic.tools_._plotly_helpers import _require_plotly

        _require_plotly()

        if image is None:
            image = self._require_cache_image()

        props = regionprops(image.objmap[:], intensity_image=image.gray[:])
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
            return pn.pane.Markdown(
                    "No objects found for symmetric-radius analysis."
            )

        fig = self._build_plate_overview(
                image, intermediates_cache,
                base_layer=base_layer,
        )
        h, w = image.gray[:].shape[:2]
        overview_h = int(900 * h / w)

        header = pn.pane.Markdown(
                f"## Symmetric Radius -- {len(intermediates_cache)} objects",
                styles={"font-family": "'DM Sans', sans-serif", "color": _OI_NAVY},
        )

        return pn.Column(
                header,
                pn.pane.Plotly(
                        fig,
                        config={"scrollZoom": True},
                        sizing_mode="stretch_width",
                        height=overview_h,
                ),
        )

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

        MeasureSymmetricRadius._add_overlay_traces(
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
        - **Objmap** — semi-transparent colored mask of detected objects.
        - **Zones** — painted core / dense / sparse zones per object.
        - **Centroids** — inoculum centre markers for every object.
        - **Core radius** — dashed vermilion circles.
        - **Symmetric radius** — purple dashed circles (all objects).
        - **Outer envelope** — solid green circles.

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

        # ── Objmap overlay ──────────────────────────────────────────
        _ALPHA = 77  # ~0.3
        _PALETTE = [
            (0, 114, 178, _ALPHA),  # blue
            (230, 159, 0, _ALPHA),  # orange
            (0, 158, 115, _ALPHA),  # green
            (204, 121, 167, _ALPHA),  # purple
            (86, 180, 233, _ALPHA),  # sky
            (213, 94, 0, _ALPHA),  # vermilion
        ]
        objmap = image.objmap[:]
        h, w = objmap.shape[:2]
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        labels = np.unique(objmap)
        labels = labels[labels > 0]
        for i, lbl in enumerate(labels):
            color = _PALETTE[i % len(_PALETTE)]
            overlay[objmap == lbl] = color
        fig.add_trace(go.Image(
                z=overlay,
                name="Objmap",
                visible="legendonly",
                hoverinfo="skip",
        ))

        # ── Zones overlay (core / dense / sparse) ───────────────────
        zone_palette = np.array([
            [0, 0, 0, 0],          # 0 outside
            [213, 94, 0, _ALPHA],  # 1 core vermilion
            [0, 54, 96, _ALPHA],   # 2 dense navy
            [86, 180, 233, _ALPHA],  # 3 sparse sky
        ], dtype=np.uint8)

        zone_canvas = np.zeros((h, w, 4), dtype=np.uint8)
        for inter in intermediates_cache.values():
            if not inter.zones_computed:
                continue
            slc = inter.bbox_slice
            mask = inter.obj_mask
            mh, mw = mask.shape
            rows, cols = np.indices((mh, mw))
            dr = rows - inter.centroid_rc[0]
            dc = cols - inter.centroid_rc[1]
            theta_local = np.mod(
                    np.degrees(np.arctan2(dr, dc)), 360.0,
            ).astype(np.int16)
            r_pixel = np.sqrt(dr ** 2 + dc ** 2)

            rc = inter.r_core_per_angle[theta_local]
            rd = inter.r_dense_end_per_angle[theta_local]
            ro = inter.r_outer_per_angle[theta_local]

            zone = np.zeros((mh, mw), dtype=np.uint8)
            zone[mask & (r_pixel <= rc)] = 1
            zone[mask & (r_pixel > rc) & (r_pixel <= rd)] = 2
            zone[mask & (r_pixel > rd) & (r_pixel <= ro)] = 3

            zone_canvas[slc] = zone_palette[zone]

        fig.add_trace(go.Image(
                z=zone_canvas,
                name="Zones",
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

        # ── Outer envelope circles (all objects) ────────────────────
        env_x: list[float] = []
        env_y: list[float] = []
        for inter in intermediates_cache.values():
            outer = float(inter.core_radius) + float(inter.max_expansion)
            if outer <= 0:
                continue
            slc = inter.bbox_slice
            r0, c0 = int(slc[0].start), int(slc[1].start)
            cx = inter.centroid_rc[1] + c0
            cy = inter.centroid_rc[0] + r0
            xs, ys = _circle_xy(cx, cy, outer)
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


MeasureSymmetricRadius.__doc__ = SYMMETRIC_RADIUS.append_rst_to_doc(
        MeasureSymmetricRadius
)
