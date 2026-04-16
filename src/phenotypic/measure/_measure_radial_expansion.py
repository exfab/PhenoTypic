from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
import pandas as pd
from scipy.ndimage import convolve, distance_transform_edt
from skimage.morphology import skeletonize
from skimage.measure import regionprops

from phenotypic.abc_ import MeasureFeatures
from phenotypic.tools_.branch_pathfinding import backtrack_path, run_multisource_dijkstra
from phenotypic.tools_.constants_ import OBJECT
from ..tools_.measurement_info_ import RADIAL_EXPANSION


_NEIGHBOR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)

# Okabe-Ito palette constants for diagnostic plots
_OI_NAVY = "#003660"
_OI_ORANGE = "#E69F00"
_OI_SKY = "#56B4E9"
_OI_GREEN = "#009E73"
_OI_BLUE = "#0072B2"
_OI_PURPLE = "#CC79A7"
_OI_VERMILION = "#D55E00"
_OI_GREY = "#BBBBBB"

# Overlay alpha for zone RGBA canvases (0-255 scale).
_ZONE_ALPHA_U8 = 77  # ≈ 0.3


@dataclass
class _ExpansionIntermediates:
    """Intermediate results from the radial expansion pipeline for one object.

    ``obj_mask``, ``gray_crop``, and ``dist_map`` are intentionally absent:
    the first two can be re-sliced from the cached ``Image`` via
    :meth:`MeasureRadialExpansion._get_local_obj_mask` /
    :meth:`MeasureRadialExpansion._get_local_gray_crop`, and ``dist_map`` is
    cheap to recompute on demand (``_get_local_dist_map``).
    """

    label: int
    bbox_slice: tuple[slice, slice]    # crop offset into the full-image arrays
    centroid_rc: tuple[float, float]   # intensity-weighted centroid (row, col) in LOCAL bbox coords
    density_profile: np.ndarray        # 1D radial density signal per annulus
    annulus_radii: np.ndarray          # center radius of each annulus (px)
    core_radius: float                 # PELT-determined core→branch transition radius
    peripheral_mask: np.ndarray        # obj_mask with core zone removed (bool, local shape)
    skeleton: np.ndarray               # skeletonized peripheral zone (bool, local shape)
    endpoints: np.ndarray              # (N, 2) branch tip coordinates in local coords
    branch_paths: list[tuple[np.ndarray, float]] = field(default_factory=list)
    branch_lengths: np.ndarray = field(default_factory=lambda: np.array([]))
    runner_index: int | None = None        # index into branch_lengths, or None
    runner_threshold: float | None = None  # threshold value used for runner detection


class MeasureRadialExpansion(MeasureFeatures):
    """Measure radial expansion of filamentous fungal colonies with runner detection.

    Decompose each colony into a dense core and peripheral skeleton branches
    using PELT changepoint detection on a radial density profile. Trace
    branches from tips back to the core boundary, measure path lengths, and
    flag anomalously long runners via MAD/IQR outlier statistics.

    Tip→core paths are cost-optimal, traced by multi-source Dijkstra over a
    skeleton-preferring cost surface, so they hug the skeleton almost
    everywhere but may take single-pixel object-interior detours across
    disjoint skeleton junctions.

    Best For:
        - Quantifying radial growth rate of filamentous fungi on solid
          media.
        - Detecting asymmetric runner hyphae that extend far beyond the
          colony body.
        - Distinguishing compact versus invasive colony morphotypes by
          branch-length distributions.
        - Comparing wild-type versus mutant expansion phenotypes across
          time-course assays.
        - Measuring core-to-periphery structure in dense fungal mats.

    Consider Also:
        - :class:`MeasureShape` for general morphological descriptors
          (circularity, eccentricity) that do not require skeleton
          decomposition.
        - :class:`MeasureBounds` for lightweight bounding-box and centroid
          data without radial analysis.

    Returns:
        pd.DataFrame: Object-level radial expansion measurements with
        columns:

            - Label: unique object identifier.
            - CoreRadius: radius of the dense core zone (pixels).
            - NumBranches: number of skeleton branches from core to tips.
            - MeanRadius, MedianRadius: average branch path lengths.
            - MaxBranchLength: longest branch.
            - RobustMeanRadius: mean branch length excluding the runner.
            - RunnerDetected: 1.0 if an outlier runner was found, else 0.0.
            - RunnerLength: path length of the runner branch (NaN if none).

    See Also:
        :doc:`/tutorials/notebooks/07_measuring_and_exporting` for a
        walkthrough of measuring and exporting colony data.
        :doc:`/explanation/measurement_metrics_biological_meaning` for
        interpreting expansion metrics in a biological context.
    """

    _measurement_info_class = RADIAL_EXPANSION

    def __init__(
        self,
        outlier_method: Literal["mad", "iqr", "ellipse"] = "mad",
        outlier_k: float = 3.0,
        n_annuli: int = 100,
        pelt_penalty: float = 5.0,
        skeleton_method: Literal["zhang", "lee"] = "zhang",
        method: Literal["distance", "intensity"] = "distance",
    ):
        self.outlier_method = outlier_method
        self.outlier_k = outlier_k
        self.n_annuli = n_annuli
        self.pelt_penalty = pelt_penalty
        self.skeleton_method = skeleton_method
        self.method = method

        # Populated by _operate(); consumed by inspect() when no image is passed.
        # Underscore-prefixed so SerializablePipeline skips them in to_json().
        self._cache_image: Image | None = None
        self._cache_intermediates: dict[int, _ExpansionIntermediates] = {}
        self._cache_props: list | None = None

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
    ) -> _ExpansionIntermediates:
        """Run the full radial-expansion pipeline for a single object.

        Args:
            image: Detected Image with objmap/objmask.
            object_label: Specific object label to analyse. If *None*,
                the largest object by area is selected.
            prop: Pre-computed RegionProperties object. When provided the
                internal ``regionprops`` call is skipped.

        Returns:
            _ExpansionIntermediates with all computed fields.
        """
        if prop is not None:
            target_prop = prop
        else:
            props = regionprops(image.objmap[:], intensity_image=image.gray[:])

            # Select target object
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

        slc = target_prop.slice

        # Early exit for tiny objects
        if target_prop.area < 10:
            empty = np.array([])
            tiny_mask = np.zeros((1, 1), dtype=bool)
            return _ExpansionIntermediates(
                label=target_prop.label,
                bbox_slice=slc,
                centroid_rc=(0.0, 0.0),
                density_profile=empty,
                annulus_radii=empty,
                core_radius=0.0,
                peripheral_mask=tiny_mask,
                skeleton=tiny_mask,
                endpoints=np.empty((0, 2), dtype=np.int32),
            )

        # Crop to bounding box
        objmap_crop = image.objmap[:][slc]
        local_mask = objmap_crop == target_prop.label

        # Estimate inoculum center in local coordinates
        if self.method == "distance":
            dt = distance_transform_edt(local_mask)
            peak_idx = np.unravel_index(np.argmax(dt), dt.shape)
            local_cr = (float(peak_idx[0]), float(peak_idx[1]))
        else:
            cw = target_prop.centroid_weighted
            local_cr = (cw[0] - slc[0].start, cw[1] - slc[1].start)

        # Local distance map — used only within this call; not cached on
        # the dataclass. Consumers that need it later call
        # _get_local_dist_map(inter) to rebuild it on demand.
        dist_map = self._distance_from_point(local_mask.shape, local_cr)

        # Radial density profile
        density_profile, annulus_radii = self._compute_radial_density_profile(
            local_mask, dist_map, self.n_annuli
        )

        # Core radius via PELT changepoint detection
        core_radius = self._find_core_radius(
            density_profile, annulus_radii, self.pelt_penalty
        )

        # Mask out the core zone
        peripheral_mask = self._mask_core(local_mask, dist_map, core_radius)

        # Short-circuit when peripheral zone is too small
        if peripheral_mask.sum() < 3:
            return _ExpansionIntermediates(
                label=target_prop.label,
                bbox_slice=slc,
                centroid_rc=local_cr,
                density_profile=density_profile,
                annulus_radii=annulus_radii,
                core_radius=core_radius,
                peripheral_mask=peripheral_mask,
                skeleton=np.zeros_like(peripheral_mask),
                endpoints=np.empty((0, 2), dtype=np.int32),
            )

        # Skeletonize peripheral zone
        skeleton = skeletonize(peripheral_mask, method=self.skeleton_method)

        # Find branch endpoints
        endpoints = self._find_skeleton_endpoints(skeleton)

        # Trace branches from endpoints back to the core via Dijkstra
        branch_paths = self._trace_branches_dijkstra(
            local_mask,
            skeleton,
            endpoints,
            dist_map,
            local_cr,
            core_radius,
        )

        branch_lengths = np.array(
            [length for _, length in branch_paths]
        ) if branch_paths else np.array([])

        # Runner detection
        runner_index: int | None = None
        runner_threshold: float | None = None
        if len(branch_lengths) >= 3:
            runner_index, runner_threshold = self._detect_runner(
                branch_lengths, self.outlier_method, self.outlier_k
            )

        return _ExpansionIntermediates(
            label=target_prop.label,
            bbox_slice=slc,
            centroid_rc=local_cr,
            density_profile=density_profile,
            annulus_radii=annulus_radii,
            core_radius=core_radius,
            peripheral_mask=peripheral_mask,
            skeleton=skeleton,
            endpoints=endpoints,
            branch_paths=branch_paths,
            branch_lengths=branch_lengths,
            runner_index=runner_index,
            runner_threshold=runner_threshold,
        )

    # ── cache helpers ────────────────────────────────────────────────

    def _require_cache_image(self) -> Image:
        """Return the cached image or raise if nothing has been measured yet."""
        if self._cache_image is None:
            raise RuntimeError(
                "No cached image available. Call .measure(image) first, or "
                "pass image= explicitly to the method that needs one."
            )
        return self._cache_image

    def _get_local_obj_mask(self, inter: _ExpansionIntermediates) -> np.ndarray:
        """Re-derive the bbox-local boolean object mask from the cached image."""
        image = self._require_cache_image()
        return image.objmap[:][inter.bbox_slice] == inter.label

    def _get_local_gray_crop(self, inter: _ExpansionIntermediates) -> np.ndarray:
        """Re-derive the bbox-local grayscale crop from the cached image."""
        image = self._require_cache_image()
        return image.gray[:][inter.bbox_slice]

    def _get_local_dist_map(self, inter: _ExpansionIntermediates) -> np.ndarray:
        """Recompute the local Euclidean distance-from-centroid map.

        ``dist_map`` is not stored on the dataclass because it is by far
        the largest per-object field (float64, bbox-sized) and is only
        needed inside the zone overlay. This helper rebuilds it from the
        bbox-sliced ``obj_mask`` shape and ``centroid_rc`` on demand.

        For tiny-object short-circuit returns the stored ``peripheral_mask``
        is a ``(1, 1)`` placeholder that does NOT match the re-sliced
        ``obj_mask`` shape; we therefore derive the shape from the cached
        image slice, not from ``peripheral_mask``.
        """
        obj_mask = self._get_local_obj_mask(inter)
        return self._distance_from_point(obj_mask.shape, inter.centroid_rc)

    @staticmethod
    def _get_global_offset(inter: _ExpansionIntermediates) -> tuple[int, int]:
        """Return (row_offset, col_offset) to shift local coords → full-image coords."""
        return inter.bbox_slice[0].start, inter.bbox_slice[1].start

    # ── MeasureFeatures interface ────────────────────────────────────

    def _operate(self, image: Image) -> pd.DataFrame:
        measurements = {
            str(feature): np.full(image.num_objects, np.nan)
            for feature in RADIAL_EXPANSION
            if feature != RADIAL_EXPANSION.CATEGORY
        }

        props = regionprops(image.objmap[:], intensity_image=image.gray[:])

        # Reset cache for this image. Pipeline reuse of a single operator
        # instance across multiple images is "last image wins" by design —
        # inspect() reflects the most recent _operate() call.
        self._cache_image = image
        self._cache_props = props
        self._cache_intermediates = {}

        for idx, prop in enumerate(props):
            try:
                inter = self._compute_intermediates(image, prop.label, prop=prop)
            except Exception:
                import logging
                logging.getLogger(__name__).debug(
                    "Skipping object label %d", prop.label, exc_info=True
                )
                continue  # leave NaN

            self._cache_intermediates[prop.label] = inter

            n = len(inter.branch_lengths)
            if n == 0:
                measurements[str(RADIAL_EXPANSION.CORE_RADIUS)][idx] = inter.core_radius
                measurements[str(RADIAL_EXPANSION.NUM_BRANCHES)][idx] = 0
                measurements[str(RADIAL_EXPANSION.RUNNER_DETECTED)][idx] = 0.0
                continue

            # Compute body lengths (excluding runner)
            if inter.runner_index is not None:
                body_mask = np.ones(n, dtype=bool)
                body_mask[inter.runner_index] = False
                body_lengths = inter.branch_lengths[body_mask]
            else:
                body_lengths = inter.branch_lengths

            measurements[str(RADIAL_EXPANSION.CORE_RADIUS)][idx] = inter.core_radius
            measurements[str(RADIAL_EXPANSION.NUM_BRANCHES)][idx] = n
            measurements[str(RADIAL_EXPANSION.MEAN_RADIUS)][idx] = np.mean(inter.branch_lengths)
            measurements[str(RADIAL_EXPANSION.MEDIAN_RADIUS)][idx] = np.median(inter.branch_lengths)
            measurements[str(RADIAL_EXPANSION.MAX_BRANCH_LENGTH)][idx] = np.max(inter.branch_lengths)

            if len(body_lengths) > 0:
                measurements[str(RADIAL_EXPANSION.ROBUST_MEAN_RADIUS)][idx] = np.mean(body_lengths)

            if inter.runner_index is not None:
                measurements[str(RADIAL_EXPANSION.RUNNER_DETECTED)][idx] = 1.0
                measurements[str(RADIAL_EXPANSION.RUNNER_LENGTH)][idx] = inter.branch_lengths[inter.runner_index]
            else:
                measurements[str(RADIAL_EXPANSION.RUNNER_DETECTED)][idx] = 0.0

        df = pd.DataFrame(measurements)
        df.insert(0, OBJECT.LABEL, image.objects.labels2series())
        return df

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
    def _mask_core(
        obj_mask: np.ndarray,
        dist_map: np.ndarray,
        core_radius: float,
    ) -> np.ndarray:
        """Remove the dense core zone from an object mask.

        Args:
            obj_mask: Boolean mask of the object.
            dist_map: Pre-computed Euclidean distance map from the centroid.
            core_radius: Core radius in pixels.

        Returns:
            Boolean mask with pixels inside the core set to False.
        """
        if core_radius <= 0:
            return obj_mask.copy()
        return obj_mask & (dist_map > core_radius)

    @staticmethod
    def _find_skeleton_endpoints(skeleton: np.ndarray) -> np.ndarray:
        """Identify skeleton branch endpoints (pixels with exactly one 8-connected neighbor).

        Args:
            skeleton: Boolean skeleton image.

        Returns:
            (N, 2) int array of (row, col) endpoint coordinates.
        """
        neighbor_count = convolve(
            skeleton.astype(np.int32), _NEIGHBOR_KERNEL, mode="constant", cval=0
        )
        endpoint_mask = skeleton & (neighbor_count == 1)
        return np.argwhere(endpoint_mask)

    @staticmethod
    def _build_branch_cost_surface(
        local_mask: np.ndarray,
        skeleton: np.ndarray,
        dist_map: np.ndarray,
        core_radius: float,
    ) -> np.ndarray:
        """Skeleton-preferring cost surface for tip->core Dijkstra.

        Skeleton pixels are ~10x cheaper than on-object off-skeleton pixels
        and ~200x cheaper than off-object pixels, so paths hug the skeleton
        except across one-pixel gaps where a single object-interior detour
        is cheaper than a dead-end.

        Args:
            local_mask: Boolean object mask in local bbox coordinates.
            skeleton: Boolean skeleton image in local bbox coordinates.
            dist_map: Euclidean distance from the inoculum centroid
                (local coordinates, unused — kept for signature symmetry
                with callers that may introduce radial penalties later).
            core_radius: PELT-determined core radius in pixels (unused,
                see ``dist_map``).

        Returns:
            Float32 cost surface ready for
            :func:`phenotypic.tools_.branch_pathfinding.run_multisource_dijkstra`.
        """
        del dist_map, core_radius  # reserved for future radial penalties
        affinity = (
            10.0 * skeleton.astype(np.float32)
            + 1.0 * local_mask.astype(np.float32)
        )
        cost = 1.0 / (affinity + 0.05)
        return cost.astype(np.float32)

    @staticmethod
    def _trace_branches_dijkstra(
        local_mask: np.ndarray,
        skeleton: np.ndarray,
        endpoints: np.ndarray,
        dist_map: np.ndarray,
        centroid_rc: tuple[float, float],
        core_radius: float,
    ) -> list[tuple[np.ndarray, float]]:
        """Trace cost-optimal paths from skeleton tips back to the core.

        Uses multi-source Dijkstra seeded at the dense core zone
        (``dist_map <= core_radius`` intersected with ``local_mask``) and
        backtracks the predecessor map from each endpoint. Returns
        Euclidean pixel-space path lengths (diagonal steps = sqrt(2)) ---
        the cost profile is used only internally for optimization and is
        discarded here, so measurement columns remain in pixel units.

        Args:
            local_mask: Boolean object mask in local bbox coordinates.
            skeleton: Boolean skeleton image in local bbox coordinates.
            endpoints: (N, 2) int array of tip coordinates (row, col).
            dist_map: Euclidean distance from the inoculum centroid.
            centroid_rc: (row, col) centroid in local coordinates.
                Used as a single-pixel fallback seed when
                ``core_radius == 0`` leaves the core zone empty.
            core_radius: PELT-determined core radius in pixels.

        Returns:
            List of ``(coords, path_length)`` tuples where ``coords`` is a
            ``(M, 2)`` int32 array ordered tip->core and ``path_length`` is
            the cumulative Euclidean pixel length.
        """
        if len(endpoints) == 0 or not skeleton.any():
            return []

        cost = MeasureRadialExpansion._build_branch_cost_surface(
            local_mask, skeleton, dist_map, core_radius,
        )
        core_labels = (local_mask & (dist_map <= core_radius)).astype(np.int32)
        if core_labels.sum() == 0:
            # core_radius == 0 fallback: single-pixel seed at the centroid.
            cr_r = int(round(centroid_rc[0]))
            cr_c = int(round(centroid_rc[1]))
            cr_r = max(0, min(cr_r, local_mask.shape[0] - 1))
            cr_c = max(0, min(cr_c, local_mask.shape[1] - 1))
            core_labels[cr_r, cr_c] = 1

        dijkstra = run_multisource_dijkstra(cost, core_labels, delta=0.0)

        branches: list[tuple[np.ndarray, float]] = []
        for ep in endpoints:
            r, c = int(ep[0]), int(ep[1])
            result = backtrack_path(
                r, c, dijkstra.predecessor, dijkstra.cost_distance, cost,
            )
            if result is None:
                continue
            coords, _cost_profile = result
            if len(coords) > 1:
                diffs = np.diff(coords, axis=0).astype(np.float64)
                path_length = float(np.sqrt((diffs ** 2).sum(axis=1)).sum())
            else:
                path_length = 0.0
            branches.append((coords.astype(np.int32), path_length))

        return branches

    @staticmethod
    def _detect_runner(
        branch_lengths: np.ndarray,
        method: str,
        k: float,
    ) -> tuple[int | None, float | None]:
        """Detect an outlier runner branch from the branch-length distribution.

        Args:
            branch_lengths: 1D array of branch path lengths.
            method: Outlier method --- ``"mad"``, ``"iqr"``, or
                ``"ellipse"`` (falls back to MAD).
            k: Multiplier for the outlier threshold.

        Returns:
            Tuple of (runner_index, threshold). Both are *None* when
            fewer than 3 branches are present; *runner_index* is *None*
            (with a valid *threshold*) when no branch exceeds the
            threshold.
        """
        if len(branch_lengths) < 3:
            return None, None

        if method == "iqr":
            q1, q3 = np.percentile(branch_lengths, [25, 75])
            iqr = q3 - q1
            threshold = q3 + k * iqr
        else:
            if method == "ellipse":
                import warnings
                warnings.warn(
                    "Ellipse outlier method not yet implemented; falling back to MAD.",
                    stacklevel=2,
                )
            # MAD (default and ellipse fallback)
            median = np.median(branch_lengths)
            mad = np.median(np.abs(branch_lengths - median))
            threshold = median + k * (mad if mad > 0 else 1.0)

        candidates = np.where(branch_lengths > threshold)[0]
        if len(candidates) == 0:
            return None, float(threshold)

        runner_idx = int(candidates[np.argmax(branch_lengths[candidates])])
        return runner_idx, float(threshold)

    # ── diagnostics ──────────────────────────────────────────────────

    def decompose(self, image: Image) -> pd.DataFrame:
        """Return per-branch decomposition table for debugging.

        Produces one row per skeleton branch for every detected object,
        including angle from the intensity-weighted centroid, path length,
        and whether the branch was flagged as a runner.

        Args:
            image: Detected Image with objmap/objmask.

        Returns:
            pd.DataFrame with columns: ObjectLabel, BranchIndex, Angle,
            Length, IsRunner.
        """
        rows: list[dict] = []
        props = regionprops(image.objmap[:], intensity_image=image.gray[:])
        for prop in props:
            try:
                inter = self._compute_intermediates(image, prop.label, prop=prop)
            except Exception:
                import logging
                logging.getLogger(__name__).debug(
                    "Skipping object label %d", prop.label, exc_info=True
                )
                continue
            for i, (path, length) in enumerate(inter.branch_paths):
                tip = path[0] if len(path) > 0 else np.array([np.nan, np.nan])
                angle = float(np.arctan2(
                    tip[0] - inter.centroid_rc[0],
                    tip[1] - inter.centroid_rc[1],
                ))
                rows.append({
                    "ObjectLabel": inter.label,
                    "BranchIndex": i,
                    "Angle": angle,
                    "Length": length,
                    "IsRunner": 1 if inter.runner_index == i else 0,
                })
        if not rows:
            return pd.DataFrame(
                columns=["ObjectLabel", "BranchIndex", "Angle", "Length", "IsRunner"]
            )
        return pd.DataFrame(rows)

    def inspect(
        self,
        image: Image | None = None,
        object_label: int | None = None,
    ):
        """Interactive diagnostic dashboard for radial expansion measurement.

        If *image* is omitted, the dashboard uses the image and intermediates
        cached by the most recent call to :meth:`measure`. If *image* is
        provided and matches the cached instance (by identity), the cache is
        reused; otherwise intermediates are recomputed for the new image
        (and the cache is refreshed as a side effect).

        .. warning::
            Cache reuse is keyed on ``id(image)`` — the Python object
            identity of the ``Image`` instance. If the image's underlying
            arrays (``image.rgb``, ``image.objmap``, etc.) are mutated in
            place between ``.measure()`` and ``.inspect()``, the overlay
            diagnostics will silently reflect the pre-mutation state. To
            force a fresh compute after in-place edits, call
            ``.measure(image)`` again before ``.inspect()``.

        The dashboard renders spatial diagnostics (core/periphery zones,
        skeletons, endpoints, branch traces, core circles) for **all** cached
        objects as toggleable layers on a single full-plate overview. The
        1D panels (radial density profile, branch length distribution) and
        the summary table remain per-object and react to a dropdown selector.

        Args:
            image: Detected Image with objmap/objmask. If ``None``, uses the
                image from the last ``.measure()`` call.
            object_label: Pre-selected object label. If ``None``, the largest
                object by area is selected initially.

        Returns:
            Panel Column layout with the plate overview, an object selector,
            and per-object radial profile, branch distribution, and summary
            panels.

        Raises:
            RuntimeError: If called with ``image=None`` and no prior
                ``.measure()`` has populated the cache.
        """
        from phenotypic.tools_.panel_ import require_panel, ensure_panel_extension

        require_panel()
        ensure_panel_extension()

        import panel as pn
        from phenotypic.tools_._plotly_helpers import _require_plotly

        _require_plotly()

        # Resolve image and cache state
        if image is None:
            image = self._require_cache_image()
            if self._cache_props is None:
                raise RuntimeError(
                    "Cached image is set but cached regionprops are missing. "
                    "This should not happen — call .measure(image) again."
                )
            props: list = self._cache_props
            intermediates_cache = self._cache_intermediates
        elif (
            self._cache_image is not None
            and self._cache_props is not None
            and id(image) == id(self._cache_image)
        ):
            props = self._cache_props
            intermediates_cache = self._cache_intermediates
        else:
            # Fresh image: compute intermediates and populate the cache so
            # subsequent calls to inspect() (with no arg) reuse them.
            props = regionprops(image.objmap[:], intensity_image=image.gray[:])
            intermediates_cache = {}
            for prop in props:
                try:
                    inter = self._compute_intermediates(image, prop.label, prop=prop)
                    intermediates_cache[prop.label] = inter
                except Exception:
                    continue
            self._cache_image = image
            self._cache_props = props
            self._cache_intermediates = intermediates_cache

        if not intermediates_cache:
            return pn.pane.Markdown("No objects found for radial expansion analysis.")

        # Determine default selection
        all_labels = sorted(intermediates_cache.keys())
        if object_label is not None and object_label in intermediates_cache:
            default_label = object_label
        else:
            default_label = max(
                intermediates_cache,
                key=lambda lbl: int(intermediates_cache[lbl].peripheral_mask.size),
            )

        # Build selector widget
        selector = pn.widgets.Select(
            name="Object",
            options={f"Object {lbl}": lbl for lbl in all_labels},
            value=default_label,
        )

        # Build reactive plate overview with full-image diagnostic overlays.
        # The overlays render once (not reactively) since they represent all
        # objects at full plate scale; only the bbox highlight updates when
        # the selector changes.
        overview_fig = self._build_plate_overview(
            image, props, intermediates_cache, default_label,
        )
        self._add_full_image_overlays(
            overview_fig, image, intermediates_cache,
        )
        h, w = image.gray[:].shape[:2]
        overview_h = int(900 * h / w)
        overview_pane = pn.pane.Plotly(
            overview_fig,
            config={"scrollZoom": True},
            sizing_mode="stretch_width",
            height=overview_h,
        )

        instance = self  # capture for closures

        @pn.depends(selector.param.value)
        def update_overview(label):
            fig = instance._update_plate_overview(
                overview_fig, props, intermediates_cache, label,
            )
            overview_pane.object = fig
            return overview_pane

        # Build reactive per-object diagnostic panes
        @pn.depends(selector.param.value)
        def plot_radial_profile(label):
            return instance._plot_radial_profile(intermediates_cache[label])

        @pn.depends(selector.param.value)
        def plot_branch_distribution(label):
            return instance._plot_branch_distribution(intermediates_cache[label])

        @pn.depends(selector.param.value)
        def build_summary(label):
            return instance._build_summary_panel(
                intermediates_cache[label], instance,
            )

        @pn.depends(selector.param.value)
        def object_header(label):
            return pn.pane.Markdown(
                f"### Object {label}",
                styles={"font-family": "'DM Sans', sans-serif", "color": _OI_NAVY},
            )

        header = pn.pane.Markdown(
            f"## Radial Expansion Diagnostics -- {len(all_labels)} objects",
            styles={"font-family": "'DM Sans', sans-serif", "color": _OI_NAVY},
        )

        return pn.Column(
            header,
            update_overview,
            selector,
            object_header,
            pn.Row(plot_radial_profile, plot_branch_distribution, build_summary),
        )

    @staticmethod
    def _build_plate_overview(
        image: Image,
        props: list,
        intermediates_cache: dict[int, _ExpansionIntermediates],
        selected_label: int | None = None,
    ):
        """Build a zoomable plotly plate overview with object bounding boxes.

        Args:
            image: Detected Image with objmap/objmask.
            props: regionprops list for all objects.
            intermediates_cache: Pre-computed intermediates keyed by label.
            selected_label: Label of the currently selected object.

        Returns:
            plotly.graph_objects.Figure with bounding box overlays.
        """
        from phenotypic.tools_._plotly_helpers import (
            plotly_imshow,
            add_plotly_obj_labels,
        )

        h, w = image.gray[:].shape[:2]
        # Scale to a reasonable display width, preserving aspect ratio
        display_w = 900
        display_h = int(display_w * h / w)
        fig = plotly_imshow(
            image.gray[:], title="Plate Overview",
            figsize=(display_w // 100, display_h // 100),
        )
        add_plotly_obj_labels(fig, image)

        MeasureRadialExpansion._add_bbox_shapes(
            fig, props, intermediates_cache, selected_label,
        )
        return fig

    @staticmethod
    def _update_plate_overview(
        fig,
        props: list,
        intermediates_cache: dict[int, _ExpansionIntermediates],
        selected_label: int | None,
    ):
        """Update bounding box highlighting on the plate overview.

        Args:
            fig: Existing plotly figure to update.
            props: regionprops list for all objects.
            intermediates_cache: Pre-computed intermediates keyed by label.
            selected_label: Label of the newly selected object.

        Returns:
            Updated plotly figure with new highlighting.
        """
        # Clear existing bbox shapes and re-add with new selection. We only
        # strip "rect" shapes so the core-radius circles added by the
        # full-image overlay remain intact.
        existing = fig.layout.shapes or ()
        fig.layout.shapes = tuple(
            s for s in existing if getattr(s, "type", None) != "rect"
        )
        MeasureRadialExpansion._add_bbox_shapes(
            fig, props, intermediates_cache, selected_label,
        )
        return fig

    @staticmethod
    def _add_bbox_shapes(
        fig,
        props: list,
        intermediates_cache: dict[int, _ExpansionIntermediates],
        selected_label: int | None,
    ) -> None:
        """Add bounding box rectangles to a plotly figure.

        Preserves any existing shapes (e.g. core-radius circles added by the
        full-image overlay).

        Args:
            fig: Plotly figure to add shapes to (modified in-place).
            props: regionprops list for all objects.
            intermediates_cache: Pre-computed intermediates keyed by label.
            selected_label: Label to highlight with a thicker border.
        """
        existing = list(fig.layout.shapes or ())
        for prop in props:
            label = prop.label
            if label not in intermediates_cache:
                continue
            inter = intermediates_cache[label]

            is_selected = label == selected_label
            has_runner = inter.runner_index is not None

            if is_selected:
                color = _OI_SKY
                width: float = 3
            elif has_runner:
                color = _OI_VERMILION
                width = 1.5
            else:
                color = _OI_GREEN
                width = 1

            # bbox is (min_row, min_col, max_row, max_col)
            min_row, min_col, max_row, max_col = prop.bbox
            existing.append(dict(
                type="rect",
                x0=min_col, y0=min_row,
                x1=max_col, y1=max_row,
                line=dict(color=color, width=width),
            ))

        fig.update_layout(shapes=existing)

    # ── full-image overlay helpers ──────────────────────────────────

    def _add_full_image_overlays(
        self,
        fig,
        image: Image,
        intermediates_cache: dict[int, _ExpansionIntermediates],
    ) -> None:
        """Composite per-object diagnostics onto the plate overview.

        Renders the following layers across ALL cached objects, each as a
        legend-toggleable plotly trace or a shape:

        - **Core / periphery zones** — one full-image RGBA uint8 canvas.
        - **Branch traces** — body / runner / dead-end scatter groups.
        - **Skeleton pixels** — concatenated scatter in global coords.
        - **Endpoints** — scatter in global coords.
        - **Junctions** — convolve-derived per object, concatenated.
        - **Centroids** — per-object global centroid markers.
        - **Core circles** — dashed vermilion shapes at ``core_radius``.
        - **Colony-radius circles** — solid green shapes at
          ``core_radius + robust_mean(branch_lengths)``, i.e. the
          ``RobustMeanRadius`` measurement rendered on the plate.

        Args:
            fig: Plotly figure returned by ``_build_plate_overview``.
                Modified in place.
            image: The (cached or provided) Image the overlays map onto.
            intermediates_cache: Per-object intermediates keyed by label.
        """
        import plotly.graph_objects as go

        if not intermediates_cache:
            return

        h, w = image.gray[:].shape[:2]

        # 1) Zone canvas (navy core, sky periphery), composited in bbox windows
        zone_canvas = np.zeros((h, w, 4), dtype=np.uint8)
        navy_rgba = (0, 54, 96, _ZONE_ALPHA_U8)
        sky_rgba = (86, 180, 233, _ZONE_ALPHA_U8)
        for inter in intermediates_cache.values():
            if inter.peripheral_mask.size <= 1:
                continue  # tiny/empty short-circuit
            slc = inter.bbox_slice
            window = zone_canvas[slc]
            # periphery first so core overwrites the overlap band
            window[inter.peripheral_mask] = sky_rgba
            if inter.core_radius > 0:
                local_mask = self._get_local_obj_mask(inter)
                local_dist = self._get_local_dist_map(inter)
                core_zone = local_mask & (local_dist <= inter.core_radius)
                window[core_zone] = navy_rgba

        fig.add_trace(
            go.Image(
                z=zone_canvas,
                name="Core / Periphery",
                visible="legendonly",
                hoverinfo="skip",
            )
        )

        # 2) Branch traces — three consolidated traces (body / runner / dead-end)
        # concatenated with NaN separators. Using ~3 traces instead of one per
        # branch keeps browsers usable at 384-colony scale.
        body_xs: list[float] = []
        body_ys: list[float] = []
        runner_xs: list[float] = []
        runner_ys: list[float] = []
        dead_xs: list[float] = []
        dead_ys: list[float] = []
        for inter in intermediates_cache.values():
            r0, c0 = self._get_global_offset(inter)
            for i, (coords, length) in enumerate(inter.branch_paths):
                if len(coords) == 0:
                    continue
                xs = (coords[:, 1] + c0).tolist()
                ys = (coords[:, 0] + r0).tolist()
                is_runner = inter.runner_index is not None and i == inter.runner_index
                if length == 0:
                    bucket_x, bucket_y = dead_xs, dead_ys
                elif is_runner:
                    bucket_x, bucket_y = runner_xs, runner_ys
                else:
                    bucket_x, bucket_y = body_xs, body_ys
                bucket_x.extend(xs)
                bucket_y.extend(ys)
                bucket_x.append(float("nan"))
                bucket_y.append(float("nan"))

        for name, color, lw, xs_arr, ys_arr in [
            ("Branches", _OI_NAVY, 2, body_xs, body_ys),
            ("Runner branches", _OI_VERMILION, 3, runner_xs, runner_ys),
            ("Dead-end branches", _OI_GREY, 1.5, dead_xs, dead_ys),
        ]:
            if not xs_arr:
                continue
            fig.add_trace(
                go.Scattergl(
                    x=xs_arr,
                    y=ys_arr,
                    mode="lines",
                    line=dict(color=color, width=lw),
                    name=name,
                    hoverinfo="skip",
                )
            )

        # 3) Skeleton pixels — concatenated scatter across all objects
        skel_rows: list[np.ndarray] = []
        skel_cols: list[np.ndarray] = []
        for inter in intermediates_cache.values():
            r0, c0 = self._get_global_offset(inter)
            coords = np.argwhere(inter.skeleton)
            if coords.size == 0:
                continue
            skel_rows.append(coords[:, 0] + r0)
            skel_cols.append(coords[:, 1] + c0)
        if skel_rows:
            fig.add_trace(
                go.Scattergl(
                    x=np.concatenate(skel_cols),
                    y=np.concatenate(skel_rows),
                    mode="markers",
                    marker=dict(color=_OI_NAVY, size=2),
                    name="Skeleton",
                    visible="legendonly",
                    hoverinfo="skip",
                )
            )

        # 4) Endpoints — concatenated scatter
        ep_rows: list[np.ndarray] = []
        ep_cols: list[np.ndarray] = []
        for inter in intermediates_cache.values():
            if inter.endpoints.size == 0:
                continue
            r0, c0 = self._get_global_offset(inter)
            ep_rows.append(inter.endpoints[:, 0] + r0)
            ep_cols.append(inter.endpoints[:, 1] + c0)
        if ep_rows:
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate(ep_cols),
                    y=np.concatenate(ep_rows),
                    mode="markers",
                    marker=dict(color=_OI_ORANGE, size=7, line=dict(color="white", width=0.5)),
                    name="Endpoints",
                    hoverinfo="skip",
                )
            )

        # 5) Junctions — convolve per object, derived at render time
        jr_rows: list[np.ndarray] = []
        jr_cols: list[np.ndarray] = []
        for inter in intermediates_cache.values():
            if not inter.skeleton.any():
                continue
            r0, c0 = self._get_global_offset(inter)
            neighbor_count = convolve(
                inter.skeleton.astype(np.int32),
                _NEIGHBOR_KERNEL,
                mode="constant",
                cval=0,
            )
            junction_mask = inter.skeleton & (neighbor_count >= 3)
            junctions = np.argwhere(junction_mask)
            if junctions.size == 0:
                continue
            jr_rows.append(junctions[:, 0] + r0)
            jr_cols.append(junctions[:, 1] + c0)
        if jr_rows:
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate(jr_cols),
                    y=np.concatenate(jr_rows),
                    mode="markers",
                    marker=dict(color=_OI_GREEN, size=5, line=dict(color="white", width=0.3)),
                    name="Junctions",
                    visible="legendonly",
                    hoverinfo="skip",
                )
            )

        # 6) Centroids — per-object markers
        cent_rows: list[float] = []
        cent_cols: list[float] = []
        for inter in intermediates_cache.values():
            r0, c0 = self._get_global_offset(inter)
            cent_rows.append(inter.centroid_rc[0] + r0)
            cent_cols.append(inter.centroid_rc[1] + c0)
        if cent_rows:
            fig.add_trace(
                go.Scatter(
                    x=cent_cols,
                    y=cent_rows,
                    mode="markers",
                    marker=dict(
                        color=_OI_ORANGE,
                        size=8,
                        symbol="circle",
                        line=dict(color=_OI_NAVY, width=1),
                    ),
                    name="Centroids",
                    hoverinfo="skip",
                )
            )

        # 7) Core circles — plotly shapes at global centroids
        existing_shapes = list(fig.layout.shapes or ())
        for inter in intermediates_cache.values():
            if inter.core_radius <= 0:
                continue
            r0, c0 = self._get_global_offset(inter)
            cx = inter.centroid_rc[1] + c0
            cy = inter.centroid_rc[0] + r0
            existing_shapes.append(dict(
                type="circle",
                xref="x", yref="y",
                x0=cx - inter.core_radius,
                y0=cy - inter.core_radius,
                x1=cx + inter.core_radius,
                y1=cy + inter.core_radius,
                line=dict(color=_OI_VERMILION, width=1.2, dash="dash"),
            ))

        # 8) Colony-radius circles — the measurement itself, drawn from the
        # intensity-weighted centroid out to core_radius + robust mean branch
        # length (runner branch excluded when one is detected). This is the
        # "MeanRadius" / "RobustMeanRadius" value that ends up in the
        # measurements DataFrame.
        for inter in intermediates_cache.values():
            if len(inter.branch_lengths) == 0:
                continue
            if inter.runner_index is not None and len(inter.branch_lengths) > 1:
                body_mask = np.ones(len(inter.branch_lengths), dtype=bool)
                body_mask[inter.runner_index] = False
                body = inter.branch_lengths[body_mask]
                if body.size == 0:
                    continue
                mean_branch = float(np.mean(body))
            else:
                mean_branch = float(np.mean(inter.branch_lengths))
            radius = inter.core_radius + mean_branch
            r0, c0 = self._get_global_offset(inter)
            cx = inter.centroid_rc[1] + c0
            cy = inter.centroid_rc[0] + r0
            existing_shapes.append(dict(
                type="circle",
                xref="x", yref="y",
                x0=cx - radius,
                y0=cy - radius,
                x1=cx + radius,
                y1=cy + radius,
                line=dict(color=_OI_GREEN, width=1.2, dash="solid"),
            ))

        fig.update_layout(shapes=existing_shapes)

    # ── diagnostic plot helpers ─────────────────────────────────────

    @staticmethod
    def _dashboard_rcparams() -> dict:
        """Return the standard dashboard matplotlib rcParams dict."""
        return {
            "axes.facecolor": "#ffffff",
            "figure.facecolor": "#f5f7fa",
            "axes.edgecolor": "#dde3ed",
            "axes.grid": True,
            "grid.color": "#e8ecf2",
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlecolor": _OI_NAVY,
            "axes.titleweight": "600",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.labelcolor": "#2e3a4e",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.color": "#8892a4",
            "ytick.color": "#8892a4",
            "font.family": "sans-serif",
            "font.sans-serif": ["DM Sans", "Helvetica Neue", "Arial"],
            "axes.prop_cycle": __import__("matplotlib").cycler(color=[
                _OI_NAVY, _OI_ORANGE, _OI_SKY, _OI_GREEN, _OI_BLUE, _OI_PURPLE,
            ]),
        }

    @classmethod
    def _plot_radial_profile(cls, inter: _ExpansionIntermediates):
        """Radial density profile with core radius marker.

        Args:
            inter: Computed intermediates for a single object.

        Returns:
            Panel Matplotlib pane showing the radial density curve.
        """
        import panel as pn
        import matplotlib.pyplot as plt

        with plt.rc_context(cls._dashboard_rcparams()):
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot(inter.annulus_radii, inter.density_profile, color=_OI_NAVY, lw=2)

            if inter.core_radius > 0:
                ax.axvline(
                    inter.core_radius, ls="--", color=_OI_VERMILION, lw=1.5,
                    label=f"Core radius = {inter.core_radius:.1f} px",
                )
                ax.axvspan(0, inter.core_radius, alpha=0.1, color=_OI_NAVY)
                ax.legend(fontsize=7, framealpha=0.8)
            else:
                ax.annotate(
                    "No changepoint detected",
                    xy=(0.5, 0.9), xycoords="axes fraction",
                    ha="center", fontsize=8, color="#8892a4",
                    fontstyle="italic",
                )

            ax.set_xlabel("Radius (px)")
            ax.set_ylabel("Normalized Density")
            ax.set_title("Radial Density Profile")
            fig.tight_layout()
            pane = pn.pane.Matplotlib(fig, tight=True, dpi=100)
            plt.close(fig)
            return pane

    @classmethod
    def _plot_branch_distribution(cls, inter: _ExpansionIntermediates):
        """Branch length distribution with runner threshold.

        Args:
            inter: Computed intermediates for a single object.

        Returns:
            Panel Matplotlib pane showing branch length strip plot.
        """
        import panel as pn
        import matplotlib.pyplot as plt

        with plt.rc_context(cls._dashboard_rcparams()):
            fig, ax = plt.subplots(figsize=(5, 3.5))

            if len(inter.branch_lengths) == 0:
                ax.text(
                    0.5, 0.5, "No branches detected",
                    ha="center", va="center", fontsize=10,
                    color="#8892a4", transform=ax.transAxes,
                )
                ax.set_title("Branch Lengths + Runner Threshold")
                fig.tight_layout()
                pane = pn.pane.Matplotlib(fig, tight=True, dpi=100)
                plt.close(fig)
                return pane

            # Add jitter for visibility
            rng = np.random.default_rng(42)
            jitter = rng.uniform(-0.15, 0.15, size=len(inter.branch_lengths))

            # Color each point: runner in vermilion, rest in navy
            for i, (bl, j) in enumerate(zip(inter.branch_lengths, jitter)):
                is_runner = inter.runner_index is not None and i == inter.runner_index
                color = _OI_VERMILION if is_runner else _OI_NAVY
                size = 60 if is_runner else 35
                ax.scatter(bl, j, color=color, s=size, zorder=3, edgecolors="white", lw=0.5)

            # Runner threshold line
            if inter.runner_threshold is not None:
                ax.axvline(
                    inter.runner_threshold, ls="--", color=_OI_VERMILION, lw=1.5,
                    zorder=2,
                )
                ax.annotate(
                    f"Threshold = {inter.runner_threshold:.1f}",
                    xy=(inter.runner_threshold, 0.2),
                    fontsize=7, color=_OI_VERMILION,
                    ha="left", va="bottom",
                )

            ax.set_xlabel("Path Length (px)")
            ax.set_yticks([])
            ax.set_title("Branch Lengths + Runner Threshold")
            fig.tight_layout()
            pane = pn.pane.Matplotlib(fig, tight=True, dpi=100)
            plt.close(fig)
            return pane

    @staticmethod
    def _build_summary_panel(
        inter: _ExpansionIntermediates,
        instance: MeasureRadialExpansion,
    ):
        """Build an HTML summary table of metrics and parameters.

        Args:
            inter: Computed intermediates for a single object.
            instance: The MeasureRadialExpansion instance with parameters.

        Returns:
            Panel HTML pane with a styled summary table.
        """
        import panel as pn

        n = len(inter.branch_lengths)
        mean_r = f"{np.mean(inter.branch_lengths):.1f}" if n > 0 else "--"
        median_r = f"{np.median(inter.branch_lengths):.1f}" if n > 0 else "--"
        max_bl = f"{np.max(inter.branch_lengths):.1f}" if n > 0 else "--"

        # Robust mean (excluding runner)
        if n > 0 and inter.runner_index is not None:
            body_mask = np.ones(n, dtype=bool)
            body_mask[inter.runner_index] = False
            body_lengths = inter.branch_lengths[body_mask]
            robust_mean = f"{np.mean(body_lengths):.1f}" if len(body_lengths) > 0 else "--"
        elif n > 0:
            robust_mean = f"{np.mean(inter.branch_lengths):.1f}"
        else:
            robust_mean = "--"

        runner_detected = "Yes" if inter.runner_index is not None else "No"
        runner_length = (
            f"{inter.branch_lengths[inter.runner_index]:.1f}"
            if inter.runner_index is not None
            else "--"
        )

        row_style = (
            "padding: 4px 10px; border-bottom: 1px solid #e8ecf2;"
        )
        label_style = (
            "font-family: 'DM Mono', monospace; font-size: 11px; "
            "color: #8892a4; text-transform: uppercase; letter-spacing: 0.08em;"
        )
        value_style = (
            "font-family: 'DM Mono', monospace; font-size: 12px; "
            "color: #003660; font-weight: 500; text-align: right;"
        )
        section_style = (
            "font-family: 'DM Sans', sans-serif; font-size: 11px; "
            "font-weight: 600; color: #003660; padding: 8px 10px 4px 10px; "
            "border-bottom: 2px solid #003660;"
        )

        def _row(label: str, value: str) -> str:
            return (
                f"<tr>"
                f"<td style='{row_style} {label_style}'>{label}</td>"
                f"<td style='{row_style} {value_style}'>{value}</td>"
                f"</tr>"
            )

        def _section(title: str) -> str:
            return (
                f"<tr><td colspan='2' style='{section_style}'>{title}</td></tr>"
            )

        html = (
            "<div style='"
            "background: #ffffff; border: 1px solid #dde3ed; border-radius: 10px; "
            "padding: 12px 0; box-shadow: 0 1px 3px rgba(0,54,96,0.07); "
            "max-width: 320px;'>"
            "<table style='width: 100%; border-collapse: collapse;'>"
            f"{_section('Metrics')}"
            f"{_row('CoreRadius', f'{inter.core_radius:.1f} px')}"
            f"{_row('NumBranches', str(n))}"
            f"{_row('MeanRadius', mean_r)}"
            f"{_row('MedianRadius', median_r)}"
            f"{_row('RobustMeanRadius', robust_mean)}"
            f"{_row('MaxBranchLength', max_bl)}"
            f"{_row('RunnerDetected', runner_detected)}"
            f"{_row('RunnerLength', runner_length)}"
            f"{_section('Parameters')}"
            f"{_row('outlier_method', instance.outlier_method)}"
            f"{_row('outlier_k', str(instance.outlier_k))}"
            f"{_row('pelt_penalty', str(instance.pelt_penalty))}"
            f"{_row('n_annuli', str(instance.n_annuli))}"
            f"{_row('skeleton_method', instance.skeleton_method)}"
            f"{_row('method', instance.method)}"
            "</table></div>"
        )

        return pn.pane.HTML(html)


MeasureRadialExpansion.__doc__ = RADIAL_EXPANSION.append_rst_to_doc(
    MeasureRadialExpansion
)
