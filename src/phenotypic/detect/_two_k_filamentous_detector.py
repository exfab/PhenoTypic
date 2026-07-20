"""TwoKFilamentousDetector — two-k hysteresis branches + grid center-fill + Dijkstra."""
from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, Optional, Union

import numpy as np
from pydantic import model_validator
from skimage.filters import threshold_otsu
from typing_extensions import Self

from phenotypic.abc_ import GridObjectDetector
from phenotypic import ImagePipeline
from phenotypic.enhance import (
    ContrastStretching,
    FlattenIllumination,
    SubtractGaussian,
)
from phenotypic.enhance._two_k_phase_kernel import two_k_phase
from phenotypic.detect._inoculum_detector import InoculumDetector
from phenotypic.refine import KeepSectionLargest
from phenotypic.sdk_.reconnect import (
    ReconnectConfig,
    build_reconnect_cost,
    filter_mask_by_overlap,
    markers_from_centroids,
    partition_by_grid_voronoi,
    reconnect_fragments_tiled,
    select_reconnect_fragments,
)
from phenotypic.sdk_.typing_ import OperationField, TuneSpec

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage


class TwoKFilamentousDetector(GridObjectDetector):
    """Detect filamentous fungi via two-k phase hysteresis + grid center-fill + Dijkstra.

    Pipeline: ``branch_base`` (flatten + contrast) prepares detect_mat; ``two_k_phase``
    yields a continuous branch response (binarized by non-zero) and a loose PCT result;
    the grid ``center_detector`` intersected with a background-subtraction body fills the
    inoculum cores; the union is overlap-filtered, grid-Voronoi labeled, and its
    disconnected fragments are reconnected by tiled multi-source Dijkstra over a cost
    surface built from the same loose PCT result (no extra phase-congruency pass).
    """

    # ── scene-derivation multipliers / algorithm constants (mirror FFD) ──
    _GAUSS_SIGMA_PER_R: ClassVar[float] = 1.2
    _TILE_SIZE_PER_R: ClassVar[float] = 4.8
    _TILE_OVERLAP_PER_R: ClassVar[float] = 2.4
    _MAD_WINDOW_PER_W: ClassVar[float] = 2.0
    _PATH_DILATION_PER_W: ClassVar[float] = 0.5
    _SNR_MARGIN_PER_W: ClassVar[float] = 0.5
    _COHERENCE_RADIUS_PER_W: ClassVar[float] = 5.0
    beta: ClassVar[float] = 2.0
    gamma: ClassVar[float] = 1.2
    delta: ClassVar[float] = 1.0
    gauss_n_iter: ClassVar[int] = 2

    # ── branch enhancement ──
    branch_base: Union[OperationField, None] = None                 # -> flatten(300)+stretch(70,99)
    n_orient: Annotated[int, TuneSpec(4, 8)] = 8
    min_wavelength: Annotated[float, TuneSpec(2.0, 10.0)] = 5.0
    k_strict: Annotated[float, TuneSpec(4.0, 8.0)] = 6.0
    k_loose: Annotated[float, TuneSpec(3.5, 6.0)] = 4.5
    seed_thresh: Literal["otsu", "triangle"] = "otsu"
    cand_thresh: Literal["otsu", "triangle"] = "triangle"

    # ── center-fill ──
    center_detector: Union[OperationField, None] = None             # -> InoculumDetector pipeline
    background_subtractor: Union[OperationField, None] = None       # -> SubtractGaussian(gauss_sigma, 2)

    # ── reconnection / scene (mirror FFD) ──
    max_colony_radius_px: Annotated[float, TuneSpec(50.0, 500.0, log=True)] = 250.0
    min_branch_width_px: Annotated[int, TuneSpec(2, 10)] = 3
    reconnection_tolerance: Annotated[float, TuneSpec(1.0, 5.0)] = 2.5
    max_gap_length: Annotated[int, TuneSpec(10, 60)] = 30
    border_margin_px: Annotated[int, TuneSpec(20, 100)] = 50
    frag_reach_px: Annotated[int, TuneSpec(5, 30)] = 10
    gap_crossing_penalty: Annotated[float, TuneSpec(1.0, 10.0)] = 4.0
    reconnect_scope: Literal["branches", "pseudo"] = "branches"
    gauss_sigma: Annotated[Optional[float], TuneSpec(tunable=False)] = None
    tile_size: Annotated[Optional[int], TuneSpec(tunable=False)] = None
    tile_overlap: Annotated[Optional[int], TuneSpec(tunable=False)] = None
    mad_window: Annotated[Optional[int], TuneSpec(tunable=False)] = None
    path_dilation_radius: Annotated[Optional[int], TuneSpec(tunable=False)] = None
    snr_margin: Annotated[Optional[int], TuneSpec(tunable=False)] = None
    coherence_window_radius: Annotated[Optional[int], TuneSpec(tunable=False)] = None

    @staticmethod
    def __build_center_pipe() -> "ImagePipeline":
        return ImagePipeline(ops=[InoculumDetector(), KeepSectionLargest()])

    @model_validator(mode="after")
    def _derive_scene_params(self) -> Self:
        if self.k_loose >= self.k_strict:
            raise ValueError(
                f"k_loose ({self.k_loose}) must be < k_strict ({self.k_strict}): the loose "
                "pass supplies the faint candidates and the strict pass the confident seeds; "
                "inverting them inverts the hysteresis."
            )
        if self.branch_base is None:
            self.branch_base = ImagePipeline(ops=[
                FlattenIllumination(sigma=300.0),
                ContrastStretching(lower_percentile=70, upper_percentile=99),
            ])
        if self.center_detector is None:
            self.center_detector = self.__build_center_pipe()

        R = self.max_colony_radius_px
        w = self.min_branch_width_px
        if self.gauss_sigma is None:
            self.gauss_sigma = self._GAUSS_SIGMA_PER_R * R
        if self.tile_size is None:
            self.tile_size = int(round(self._TILE_SIZE_PER_R * R))
        if self.tile_overlap is None:
            self.tile_overlap = int(round(self._TILE_OVERLAP_PER_R * R))
        if self.tile_size <= self.tile_overlap:
            raise ValueError("tile_size must exceed tile_overlap")
        if self.mad_window is None:
            _mad = int(round(self._MAD_WINDOW_PER_W * w)) + 1
            self.mad_window = _mad + 1 if _mad % 2 == 0 else _mad
        if self.path_dilation_radius is None:
            self.path_dilation_radius = max(1, int(round(self._PATH_DILATION_PER_W * w)))
        if self.snr_margin is None:
            self.snr_margin = max(2, int(round(self._SNR_MARGIN_PER_W * w)))
        if self.coherence_window_radius is None:
            self.coherence_window_radius = int(round(self._COHERENCE_RADIUS_PER_W * w))
        if self.background_subtractor is None:
            self.background_subtractor = SubtractGaussian(
                sigma=self.gauss_sigma, n_iter=self.gauss_n_iter,
            )
        return self

    def _reconnect_config(self) -> ReconnectConfig:
        return ReconnectConfig(
            beta=self.beta, gamma=self.gamma, delta=self.delta,
            coherence_window_radius=self.coherence_window_radius,
            mad_window=self.mad_window, gap_crossing_penalty=self.gap_crossing_penalty,
            border_margin_px=self.border_margin_px, frag_reach_px=self.frag_reach_px,
            tile_size=self.tile_size, tile_overlap=self.tile_overlap,
            max_gap_length=self.max_gap_length, path_dilation_radius=self.path_dilation_radius,
            snr_margin=self.snr_margin, reconnection_tolerance=self.reconnection_tolerance,
        )

    def _fill_centers(self, image: "GridImage", enhanced: "GridImage"):
        """Grid stamps ∩ background-subtraction body -> (center_mask, center_objmap)."""
        if isinstance(self.center_detector, ImagePipeline):
            center_img = self.center_detector.apply(image, inplace=False, reset=False)
        else:
            center_img = self.center_detector.apply(image, inplace=False)
        grid_mask = center_img.objmask[:] > 0

        body_img = self.background_subtractor.apply(enhanced.copy(), inplace=False)
        body = np.asarray(body_img.detect_mat[:], dtype=float)
        body_mask = body > threshold_otsu(body)

        center_mask = grid_mask & body_mask
        return center_mask, center_img.objmap[:]

    def _operate(self, image: "GridImage") -> "GridImage":
        # ── BRANCH: two-k hysteresis on the enhanced base ──
        enhanced = image.copy()
        self.branch_base.apply(enhanced, inplace=True)
        enhanced_arr = np.asarray(enhanced.detect_mat[:], dtype=np.float32)
        enhanced_gray = np.asarray(enhanced.gray[:], dtype=np.float32)

        gated, loose = two_k_phase(
            enhanced_arr, k_strict=self.k_strict, k_loose=self.k_loose,
            seed_thresh=self.seed_thresh, cand_thresh=self.cand_thresh,
            n_orient=self.n_orient, min_wavelength=self.min_wavelength,
        )
        branch_mask = gated > 0

        # ── CENTERS: grid stamps ∩ body ──
        center_mask, center_objmap = self._fill_centers(image, enhanced)
        if center_objmap.max() == 0:
            raise ValueError("No centers detected by center_detector; cannot label colonies.")
        if not center_mask.any():
            raise ValueError(
                "center_detector found wells but none intersect the background-subtracted "
                "colony body (grid ∩ body is empty). Check background_subtractor sigma or the "
                "grid coordinates."
            )

        # ── FILTER + GRID VORONOI ──
        # Union centers FIRST so branch rings (PCT leaves the inoculum core a hole) connect
        # through their cores; THEN keep only components overlapping a center — the analogue of
        # FFD's `inoculum_structure_mask = _filter_mask_by_overlap(overall_objmask, inoculum_objmask)`.
        # This drops stray/agar objects not attached to any well (do NOT keep all objects).
        colony_mask = branch_mask | center_mask
        structure_mask = filter_mask_by_overlap(colony_mask, center_mask)
        markers = markers_from_centroids(center_objmap)
        colony_labels = partition_by_grid_voronoi(markers, structure_mask)
        if colony_labels.max() == 0:
            raise RuntimeError("Voronoi assignment produced empty result.")

        # ── RECONNECT (Dijkstra over the loose PCT cost surface) ──
        central_mask, fragment_labels = select_reconnect_fragments(
            colony_labels, center_mask, colony_mask, structure_mask,
            scope=self.reconnect_scope,
            min_fragment_size=max(1, self.min_branch_width_px),
        )
        cfg = self._reconnect_config()
        unmasked_cost, cost_surface = build_reconnect_cost(
            loose.pc_sum, loose.M, loose.m, loose.orientation,
            enhanced_arr, colony_labels, central_mask, cfg,
        )
        colony_labels = reconnect_fragments_tiled(
            colony_labels, fragment_labels, cost_surface, unmasked_cost,
            loose.pc_sum.astype(np.float32), enhanced_gray, cfg,
        )

        # ── FINAL VORONOI ──
        # Re-partition the overlap-filtered `structure_mask` (NOT the raw branch_mask) together
        # with the reconnected labels, so the final objmap contains ONLY center-overlapping
        # objects — mirrors FFD's `final_mask = (colony_labels > 0) | inoculum_structure_mask`.
        final_mask = (colony_labels > 0) | structure_mask
        colony_labels = partition_by_grid_voronoi(markers, final_mask)

        if colony_labels.dtype != image._OBJMAP_DTYPE:
            colony_labels = colony_labels.astype(image._OBJMAP_DTYPE)
        image.objmap[:] = colony_labels
        return image
