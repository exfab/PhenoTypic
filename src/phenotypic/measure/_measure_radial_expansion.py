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


@dataclass
class _ExpansionIntermediates:
    """Intermediate results from the radial expansion pipeline for one object."""

    label: int
    centroid_rc: tuple[float, float]  # intensity-weighted centroid (row, col) in LOCAL bbox coords
    obj_mask: np.ndarray              # local binary mask of the object
    gray_crop: np.ndarray             # local grayscale crop
    density_profile: np.ndarray       # 1D radial density signal per annulus
    annulus_radii: np.ndarray         # center radius of each annulus (px)
    core_radius: float                # PELT-determined core→branch transition radius
    peripheral_mask: np.ndarray       # obj_mask with core zone removed
    skeleton: np.ndarray              # skeletonized peripheral zone (boolean)
    endpoints: np.ndarray             # (N, 2) branch tip coordinates in local coords
    dist_map: np.ndarray = field(default_factory=lambda: np.array([]))  # Euclidean distance from centroid
    branch_paths: list[tuple[np.ndarray, float]] = field(default_factory=list)
    branch_lengths: np.ndarray = field(default_factory=lambda: np.array([]))
    runner_index: int | None = None       # index into branch_lengths, or None
    runner_threshold: float | None = None  # threshold value used for runner detection


class MeasureRadialExpansion(MeasureFeatures):
    """Measure radial expansion of filamentous fungal colonies with runner detection.

    Decompose each colony into a dense core and peripheral skeleton branches
    using PELT changepoint detection on a radial density profile. Trace
    branches from tips back to the core boundary, measure path lengths, and
    flag anomalously long runners via MAD/IQR outlier statistics.

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

        # Early exit for tiny objects
        if target_prop.area < 10:
            empty = np.array([])
            tiny_mask = np.zeros((1, 1), dtype=bool)
            return _ExpansionIntermediates(
                label=target_prop.label,
                centroid_rc=(0.0, 0.0),
                obj_mask=tiny_mask,
                gray_crop=np.zeros((1, 1), dtype=np.float64),
                density_profile=empty,
                annulus_radii=empty,
                core_radius=0.0,
                peripheral_mask=tiny_mask,
                skeleton=tiny_mask,
                endpoints=np.empty((0, 2), dtype=np.int32),
                dist_map=np.zeros((1, 1), dtype=np.float64),
            )

        # Crop to bounding box
        slc = target_prop.slice
        objmap_crop = image.objmap[:][slc]
        gray_crop = image.gray[:][slc]
        local_mask = objmap_crop == target_prop.label

        # Estimate inoculum center in local coordinates
        if self.method == "distance":
            dt = distance_transform_edt(local_mask)
            peak_idx = np.unravel_index(np.argmax(dt), dt.shape)
            local_cr = (float(peak_idx[0]), float(peak_idx[1]))
        else:
            cw = target_prop.centroid_weighted
            local_cr = (cw[0] - slc[0].start, cw[1] - slc[1].start)

        # Pre-compute distance map once
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
                centroid_rc=local_cr,
                obj_mask=local_mask,
                gray_crop=gray_crop,
                density_profile=density_profile,
                annulus_radii=annulus_radii,
                core_radius=core_radius,
                peripheral_mask=peripheral_mask,
                skeleton=np.zeros_like(peripheral_mask),
                endpoints=np.empty((0, 2), dtype=np.int32),
                dist_map=dist_map,
            )

        # Skeletonize peripheral zone
        skeleton = skeletonize(peripheral_mask, method=self.skeleton_method)

        # Find branch endpoints
        endpoints = self._find_skeleton_endpoints(skeleton)

        # Build core-boundary mask (thin ring just outside core radius)
        core_boundary_mask = (
            skeleton
            & (dist_map > core_radius)
            & (dist_map <= core_radius + 3)
        )

        # Trace branches from endpoints back to core boundary
        branch_paths = self._trace_branches_to_core(
            skeleton, endpoints, core_boundary_mask
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
            centroid_rc=local_cr,
            obj_mask=local_mask,
            gray_crop=gray_crop,
            density_profile=density_profile,
            annulus_radii=annulus_radii,
            core_radius=core_radius,
            peripheral_mask=peripheral_mask,
            skeleton=skeleton,
            endpoints=endpoints,
            dist_map=dist_map,
            branch_paths=branch_paths,
            branch_lengths=branch_lengths,
            runner_index=runner_index,
            runner_threshold=runner_threshold,
        )

    # ── MeasureFeatures interface ────────────────────────────────────

    def _operate(self, image: Image) -> pd.DataFrame:
        measurements = {
            str(feature): np.full(image.num_objects, np.nan)
            for feature in RADIAL_EXPANSION
            if feature != RADIAL_EXPANSION.CATEGORY
        }

        props = regionprops(image.objmap[:], intensity_image=image.gray[:])

        for idx, prop in enumerate(props):
            try:
                inter = self._compute_intermediates(image, prop.label, prop=prop)
            except Exception:
                import logging
                logging.getLogger(__name__).debug(
                    "Skipping object label %d", prop.label, exc_info=True
                )
                continue  # leave NaN

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
    def _trace_branches_to_core(
        skeleton: np.ndarray,
        endpoints: np.ndarray,
        core_boundary_mask: np.ndarray,
    ) -> list[tuple[np.ndarray, float]]:
        """Trace skeleton branches from endpoints toward the core boundary.

        Walks each branch path from its tip along the skeleton, stopping
        when the core boundary ring is reached or no more connected pixels
        remain. Returns Euclidean path lengths (diagonal steps = sqrt(2)).

        Args:
            skeleton: Boolean skeleton image.
            endpoints: (N, 2) array of endpoint (row, col) coordinates.
            core_boundary_mask: Boolean mask marking the core boundary ring
                on the skeleton.

        Returns:
            List of (coords, path_length) tuples where *coords* is an
            (M, 2) int32 array of walk coordinates and *path_length* is
            the cumulative Euclidean distance.
        """
        branches: list[tuple[np.ndarray, float]] = []
        for ep in endpoints:
            r, c = int(ep[0]), int(ep[1])
            h, w = skeleton.shape
            visited = np.zeros((h, w), dtype=np.bool_)
            path_r, path_c = [r], [c]
            visited[r, c] = True
            reached_core = False

            while True:
                found = False
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if (
                            0 <= nr < h
                            and 0 <= nc < w
                            and skeleton[nr, nc]
                            and not visited[nr, nc]
                        ):
                            visited[nr, nc] = True
                            path_r.append(nr)
                            path_c.append(nc)
                            r, c = nr, nc
                            found = True
                            if core_boundary_mask[nr, nc]:
                                reached_core = True
                            break
                    if found:
                        break
                if not found or reached_core:
                    break

            coords = np.column_stack([path_r, path_c]).astype(np.int32)
            # Euclidean path length (diagonal = sqrt(2), straight = 1.0)
            if len(coords) > 1:
                diffs = np.diff(coords, axis=0).astype(np.float64)
                segment_lengths = np.sqrt((diffs ** 2).sum(axis=1))
                path_length = float(segment_lengths.sum())
            else:
                path_length = 0.0

            branches.append((coords, path_length))

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
        image: Image,
        object_label: int | None = None,
    ):
        """Interactive diagnostic dashboard for radial expansion measurement.

        Shows a zoomable plotly plate overview with bounding boxes for all
        objects, plus per-object diagnostic plots that update reactively
        when a different object is selected from the dropdown.

        Args:
            image: Detected Image with objmap/objmask.
            object_label: Pre-selected object label. If None, the largest
                object by area is selected initially.

        Returns:
            Panel Column layout with plate overview, object selector, and
            6 diagnostic panels per object.
        """
        from phenotypic.tools_.panel_ import require_panel, ensure_panel_extension

        require_panel()
        ensure_panel_extension()

        import panel as pn
        from phenotypic.tools_._plotly_helpers import _require_plotly

        _require_plotly()

        # Compute intermediates for all objects
        props = regionprops(image.objmap[:], intensity_image=image.gray[:])
        intermediates_cache: dict[int, _ExpansionIntermediates] = {}
        for prop in props:
            try:
                inter = self._compute_intermediates(image, prop.label, prop=prop)
                intermediates_cache[prop.label] = inter
            except Exception:
                continue

        if not intermediates_cache:
            return pn.pane.Markdown("No objects found for radial expansion analysis.")

        # Determine default selection
        all_labels = sorted(intermediates_cache.keys())
        if object_label is not None and object_label in intermediates_cache:
            default_label = object_label
        else:
            default_label = max(
                intermediates_cache,
                key=lambda l: intermediates_cache[l].obj_mask.sum(),
            )

        # Build selector widget
        selector = pn.widgets.Select(
            name="Object",
            options={f"Object {lbl}": lbl for lbl in all_labels},
            value=default_label,
        )

        # Build reactive plate overview
        overview_fig = self._build_plate_overview(
            image, props, intermediates_cache, default_label,
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

        # Build reactive diagnostic panes
        @pn.depends(selector.param.value)
        def plot_radial_profile(label):
            return instance._plot_radial_profile(intermediates_cache[label])

        @pn.depends(selector.param.value)
        def plot_zone_overlay(label):
            return instance._plot_zone_overlay(intermediates_cache[label])

        @pn.depends(selector.param.value)
        def plot_skeleton(label):
            return instance._plot_skeleton(intermediates_cache[label])

        @pn.depends(selector.param.value)
        def plot_branch_traces(label):
            return instance._plot_branch_traces(intermediates_cache[label])

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
            overview_pane,
            selector,
            object_header,
            pn.Row(plot_radial_profile, plot_zone_overlay, plot_skeleton),
            pn.Row(plot_branch_traces, plot_branch_distribution, build_summary),
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
        # Clear existing bbox shapes and re-add with new selection
        fig.layout.shapes = []
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

        Args:
            fig: Plotly figure to add shapes to (modified in-place).
            props: regionprops list for all objects.
            intermediates_cache: Pre-computed intermediates keyed by label.
            selected_label: Label to highlight with a thicker border.
        """
        shapes = []
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
            shapes.append(dict(
                type="rect",
                x0=min_col, y0=min_row,
                x1=max_col, y1=max_row,
                line=dict(color=color, width=width),
            ))

        fig.update_layout(shapes=shapes)

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
    def _plot_zone_overlay(cls, inter: _ExpansionIntermediates):
        """Core/periphery zone overlay on grayscale image.

        Args:
            inter: Computed intermediates for a single object.

        Returns:
            Panel Matplotlib pane showing core and peripheral zones.
        """
        import panel as pn
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        with plt.rc_context(cls._dashboard_rcparams()):
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.imshow(inter.gray_crop, cmap="gray", aspect="equal")

            # Core zone overlay
            if inter.core_radius > 0:
                core_zone = inter.obj_mask & (inter.dist_map <= inter.core_radius)
                core_overlay = np.zeros((*inter.obj_mask.shape, 4))
                navy_rgb = (0 / 255, 54 / 255, 96 / 255)
                core_overlay[core_zone] = (*navy_rgb, 0.3)
                ax.imshow(core_overlay, aspect="equal")

            # Peripheral zone overlay
            periph_overlay = np.zeros((*inter.peripheral_mask.shape, 4))
            sky_rgb = (86 / 255, 180 / 255, 233 / 255)
            periph_overlay[inter.peripheral_mask] = (*sky_rgb, 0.3)
            ax.imshow(periph_overlay, aspect="equal")

            # Centroid marker
            ax.plot(
                inter.centroid_rc[1], inter.centroid_rc[0],
                "o", color=_OI_ORANGE, ms=8, zorder=5,
            )

            # Core boundary circle
            if inter.core_radius > 0:
                circle = Circle(
                    (inter.centroid_rc[1], inter.centroid_rc[0]),
                    inter.core_radius,
                    fill=False, edgecolor=_OI_VERMILION, ls="--", lw=1.5,
                )
                ax.add_patch(circle)

            ax.set_title("Core / Periphery Zones")
            ax.axis("off")
            fig.tight_layout()
            pane = pn.pane.Matplotlib(fig, tight=True, dpi=100)
            plt.close(fig)
            return pane

    @classmethod
    def _plot_skeleton(cls, inter: _ExpansionIntermediates):
        """Skeleton structure with endpoints and junctions.

        Args:
            inter: Computed intermediates for a single object.

        Returns:
            Panel Matplotlib pane showing skeleton, endpoints, and junctions.
        """
        import panel as pn
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        with plt.rc_context(cls._dashboard_rcparams()):
            fig, ax = plt.subplots(figsize=(5, 3.5))

            # Object mask background
            ax.imshow(inter.obj_mask, cmap="gray", alpha=0.3, aspect="equal")

            # Skeleton pixels
            skel_coords = np.argwhere(inter.skeleton)
            if len(skel_coords) > 0:
                ax.scatter(
                    skel_coords[:, 1], skel_coords[:, 0],
                    c=_OI_NAVY, s=1, zorder=2,
                )

            # Endpoints
            if len(inter.endpoints) > 0:
                ax.scatter(
                    inter.endpoints[:, 1], inter.endpoints[:, 0],
                    c=_OI_ORANGE, s=40, zorder=4, label="Endpoints",
                )

            # Junction points (neighbor count >= 3 on skeleton)
            if inter.skeleton.any():
                neighbor_count = convolve(
                    inter.skeleton.astype(np.int32), _NEIGHBOR_KERNEL,
                    mode="constant", cval=0,
                )
                junction_mask = inter.skeleton & (neighbor_count >= 3)
                junctions = np.argwhere(junction_mask)
                if len(junctions) > 0:
                    ax.scatter(
                        junctions[:, 1], junctions[:, 0],
                        c=_OI_GREEN, s=20, zorder=3, label="Junctions",
                    )

            # Core boundary circle
            if inter.core_radius > 0:
                circle = Circle(
                    (inter.centroid_rc[1], inter.centroid_rc[0]),
                    inter.core_radius,
                    fill=False, edgecolor=_OI_VERMILION, ls="--", lw=1.5,
                )
                ax.add_patch(circle)

            ax.set_title("Skeleton Structure")
            ax.axis("off")
            if len(inter.endpoints) > 0:
                ax.legend(fontsize=7, loc="upper right", framealpha=0.8)
            fig.tight_layout()
            pane = pn.pane.Matplotlib(fig, tight=True, dpi=100)
            plt.close(fig)
            return pane

    @classmethod
    def _plot_branch_traces(cls, inter: _ExpansionIntermediates):
        """Branch traces color-coded with runner highlighted.

        Args:
            inter: Computed intermediates for a single object.

        Returns:
            Panel Matplotlib pane showing traced branch paths.
        """
        import panel as pn
        import matplotlib.pyplot as plt

        with plt.rc_context(cls._dashboard_rcparams()):
            fig, ax = plt.subplots(figsize=(5, 3.5))

            # Object mask background
            ax.imshow(inter.obj_mask, cmap="gray", alpha=0.3, aspect="equal")

            color_cycle = [_OI_NAVY, _OI_ORANGE, _OI_SKY, _OI_GREEN, _OI_BLUE, _OI_PURPLE]
            cycle_idx = 0

            for i, (coords, length) in enumerate(inter.branch_paths):
                if len(coords) == 0:
                    continue

                is_runner = inter.runner_index is not None and i == inter.runner_index
                is_dead_end = length == 0

                if is_dead_end:
                    color = _OI_GREY
                    lw = 1
                elif is_runner:
                    color = _OI_VERMILION
                    lw = 3
                else:
                    color = color_cycle[cycle_idx % len(color_cycle)]
                    cycle_idx += 1
                    lw = 2

                ax.plot(coords[:, 1], coords[:, 0], color=color, lw=lw, zorder=2)

                # Tip marker (first coord)
                ax.plot(
                    coords[0, 1], coords[0, 0], "o",
                    color=color, ms=5, zorder=3,
                )
                # End marker (last coord)
                ax.plot(
                    coords[-1, 1], coords[-1, 0], "x",
                    color=color, ms=5, mew=1.5, zorder=3,
                )

            ax.set_title("Branch Traces")
            ax.axis("off")
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
