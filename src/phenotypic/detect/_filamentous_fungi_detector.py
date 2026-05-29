from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar, Optional, Union
import numpy as np
import gc

from pydantic import model_validator
from typing_extensions import Self

if TYPE_CHECKING:
    from phenotypic._core._image import Image
    from phenotypic._core._grid_image import GridImage
    from phenotypic.enhance._enhance_features import _PhaseCong3Result

from scipy.ndimage import center_of_mass, label as ndi_label
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import disk, dilation

from phenotypic.abc_ import GridObjectDetector
from phenotypic import ImagePipeline
from phenotypic.enhance import (
    SubtractGaussian,
    ContrastStretching,
    EnhanceFeatures,
)
from phenotypic.tools_.typing_ import OperationField

from phenotypic.detect import HysteresisDetector
from phenotypic.detect._inoculum_detector import InoculumDetector
from phenotypic.refine import KeepSectionLargest

from phenotypic.tools_.branch_pathfinding import (
    _apply_distance_gap_penalty_inplace,
    _apply_border_penalty_inplace,
    _apply_structure_mask_inplace,
    _compute_screening_envelope,
    compute_anisotropy,
    compute_orientation_coherence,
    compute_local_mad_map,
    assemble_composite_cost,
    calibrate_screening_threshold,
    prescreen_fragments,
    run_multisource_dijkstra,
    assign_fragments_to_colonies,
    extract_fragment_paths,
    extract_calibration_branches,
    calibrate_thresholds,
    apply_filter_cascade,
    euclidean_voronoi_assign,
    connectivity_correct_labels
)


class FilamentousFungiDetector(GridObjectDetector):
    """Detect and separate filamentous fungal colonies by two-stage detection with Euclidean Voronoi partition.

    Segment filamentous fungi in two stages: (1) detect compact inoculation
    centres with an ``inoculum_detector``, (2) capture the full hyphal
    structure via phase-congruency-based branch detection and Dijkstra
    reconnection. Filtered centre centroids seed a Euclidean Voronoi
    partition that assigns every fungal pixel to its nearest colony, with
    connectivity-based correction ensuring uniform labelling within
    connected components. For a full comparison see
    :doc:`/explanation/detection_strategies_compared`.

    Args:
        inoculum_detector: ObjectDetector or ImagePipeline that identifies
            compact fungal centres/nuclei. Should produce small, tight
            regions at inoculation points. Default uses an internal
            InoculumDetector + KeepSectionLargest pipeline.

        max_colony_radius_px: Largest colony radius (in pixels) the
            detector should handle. Sizes scene-derived spatial
            parameters (``gauss_sigma``, ``tile_size``, ``tile_overlap``)
            for this worst case. Default 250.
        min_branch_width_px: Narrowest hyphal branch width (in pixels) to
            detect. Sizes signal-scale parameters
            (``pct_min_wavelength``, ``mad_window``,
            ``path_dilation_radius``, ``snr_margin``,
            ``coherence_window_radius``). Default 3.

        ignore_borders: If True, drops objects touching the image border
            during hysteresis-threshold branch detection. Default True.
        edge_noise_threshold: Noise threshold scaling factor for phase
            congruency edge detection. Higher values are stricter
            (reject more pixels as noise; preserve fewer thin hyphae).
            Default 6.0.
        reconnection_tolerance: IQR multiplier for path quality threshold
            calibration. Higher values accept more reconnection paths
            (may bridge genuinely-missing hyphae but risks over-merging).
            Default 2.5.
        max_gap_length: Maximum acceptable length (pixels) of a
            suspicious cost stretch along a reconnection path. Paths
            with longer bad stretches are rejected. Default 30.
        border_margin_px: Border penalty buffer width in pixels.
            Prevents reconnection paths from routing along image
            borders. Default 50.
        frag_reach_px: Maximum 2D distance (pixels) from a fragment's
            boundary to the nearest routable (low-cost) pixel. Fragments
            more isolated than this are dropped before Dijkstra routing,
            since no plausible path could connect them. Default 10.
        gap_crossing_penalty: Distance-gap penalty strength during
            Dijkstra routing. Higher values make paths route around
            low-PCT-energy gaps more aggressively. Default 4.0.

        gauss_sigma: Override for SubtractGaussian sigma. If None
            (default), derived from ``max_colony_radius_px``.
        tile_size: Override for tile side length. If None (default),
            derived from ``max_colony_radius_px``.
        tile_overlap: Override for tile overlap. If None (default),
            derived from ``max_colony_radius_px``.
        pct_min_wavelength: Override for log-Gabor minimum wavelength.
            If None (default), derived from ``min_branch_width_px``.
        mad_window: Override for local MAD window size (must be odd).
            If None (default), derived from ``min_branch_width_px``.
        path_dilation_radius: Override for dilating reconnection paths.
            If None (default), derived from ``min_branch_width_px``.
        snr_margin: Override for SNR background ring radius beyond
            ``path_dilation_radius``. If None (default), derived from
            ``min_branch_width_px``.
        coherence_window_radius: Override for orientation coherence
            computation radius. If None (default), derived from
            ``min_branch_width_px``.

    Returns:
        Image: Input image with ``objmask`` set to a binary fungal mask
        and ``objmap`` set to a labelled colony map where each fungal
        colony receives a unique integer label via Voronoi assignment.

    Raises:
        TypeError: If *inoculum_detector* is not an ObjectDetector or
            ImagePipeline instance.
        ValueError: If no centres are detected, no branch structure is
            detected, or no centres overlap with the branch structure
            after filtering.

    Best For:
        * Filamentous fungal colonies (e.g., *Aspergillus*, *Neurospora*,
          *Trichoderma*) with irregular, spreading hyphal morphologies.
        * Dense plates where neighbouring fungal colonies touch or overlap
          and must be individually labelled.
        * Time-course experiments tracking hyphal extension from compact
          inoculation sites.
        * Grid-based fungal culture plates (GridImage) where one colony per
          well must be quantified.
        * High-throughput fungal phenotyping screens requiring consistent
          separation quality across hundreds of plates.

    Consider Also:
        * :class:`WatershedDetector` when colonies are compact and roughly
          circular (yeast-like morphology).
        * :class:`OtsuDetector` when fungi are well-separated and a simple
          binary mask suffices without individual labelling.
        * :class:`CompositeDetector` when combining multiple detection
          strategies without the two-stage centre-plus-body approach.

    References:
        [1] P. Kovesi, "Image features from phase congruency," *Videre:
        J. Comput. Vis. Res.*, vol. 1, no. 3, pp. 1--26, 1999.

    See Also:
        :doc:`/tutorials/notebooks/10_detecting_filamentous_fungi`
            Dedicated tutorial for filamentous fungi detection workflows.
        :doc:`/how_to/notebooks/choose_detection_algorithm`
            Guide for selecting the right detector for your plate images.
        :doc:`/explanation/detection_strategies_compared`
            In-depth comparison of all detection strategies.
    """

    @staticmethod
    def __build_center_pipe() -> "ImagePipeline":
        """Build the default inoculum-center detection pipeline.

        Constructed lazily (rather than as a class-body attribute) so
        importing this module does not instantiate operations — the
        pydantic v2 migration makes leaf operations uninstantiable until
        their own migration phase completes.
        """
        return ImagePipeline(
                ops=[InoculumDetector(), KeepSectionLargest()]
        )

    # Scene-derivation multipliers (private; override in subclass to tune).
    # Raw param = multiplier * scene knob (rounded to int where required).
    # Declared ``ClassVar`` so they stay class-level constants (not pydantic
    # fields) while remaining subclass-overridable, exactly as before.
    _GAUSS_SIGMA_PER_R: ClassVar[float] = 1.2
    _TILE_SIZE_PER_R: ClassVar[float] = 4.8
    _TILE_OVERLAP_PER_R: ClassVar[float] = 2.4
    _WAVELENGTH_PER_W: ClassVar[float] = 2.0
    _MAD_WINDOW_PER_W: ClassVar[float] = 2.0
    _PATH_DILATION_PER_W: ClassVar[float] = 0.5
    _SNR_MARGIN_PER_W: ClassVar[float] = 0.5
    _COHERENCE_RADIUS_PER_W: ClassVar[float] = 5.0

    # Algorithm internals (hidden from the constructor; override in subclass
    # to tune). ``ClassVar` keeps them out of ``model_fields`` so they are
    # not constructor parameters, matching the pre-migration behaviour.
    beta: ClassVar[float] = 2.0  # anisotropy exponent in composite cost
    gamma: ClassVar[float] = 1.2  # MAD penalty weight in composite cost numerator
    gauss_n_iter: ClassVar[int] = 2  # SubtractGaussian iterations
    delta: ClassVar[float] = 1.0  # Dijkstra radial retreat penalty
    pct_n_orient: ClassVar[int] = 8  # phase congruency angular resolution

    # ── Inoculum detector (None → default pipeline, filled by validator) ──
    # ``OperationField`` preserves the concrete detector/pipeline class
    # across a JSON round-trip; ``| None`` keeps the unset sentinel that
    # ``_derive_scene_params`` replaces with the default pipeline.
    inoculum_detector: Union[OperationField, None] = None

    # ── Scene parameters ──
    max_colony_radius_px: float = 250.0
    min_branch_width_px: int = 3

    # ── Explicit user knobs ──
    ignore_borders: bool = True
    edge_noise_threshold: float = 6.0
    reconnection_tolerance: float = 2.5
    max_gap_length: int = 30
    border_margin_px: int = 50
    frag_reach_px: int = 10
    gap_crossing_penalty: float = 4.0

    # ── Scene-derivation overrides (None → auto-derived by the validator) ──
    gauss_sigma: Optional[float] = None
    tile_size: Optional[int] = None
    tile_overlap: Optional[int] = None
    pct_min_wavelength: Optional[float] = None
    mad_window: Optional[int] = None
    path_dilation_radius: Optional[int] = None
    snr_margin: Optional[int] = None
    coherence_window_radius: Optional[int] = None

    @model_validator(mode="after")
    def _derive_scene_params(self) -> Self:
        """Fill the default pipeline and scene-derived parameters.

        Reproduces the body of the pre-migration ``__init__``:

        * a ``None`` ``inoculum_detector`` is replaced with the lazily
          built default ``InoculumDetector`` + ``KeepSectionLargest``
          pipeline (the field default cannot be a live pipeline because
          operations are uninstantiable at class-definition time);
        * each scene-derivation override left at ``None`` is computed
          from ``max_colony_radius_px`` / ``min_branch_width_px`` using
          the ``_*_PER_*`` multipliers, with ``mad_window`` forced odd.

        Bad ``inoculum_detector`` types are rejected by the field's
        ``ObjectDetector | ImagePipeline | None`` annotation before this
        validator runs.
        """
        if self.inoculum_detector is None:
            self.inoculum_detector = self.__build_center_pipe()

        R = self.max_colony_radius_px
        w = self.min_branch_width_px

        if self.gauss_sigma is None:
            self.gauss_sigma = self._GAUSS_SIGMA_PER_R * R
        if self.tile_size is None:
            self.tile_size = int(round(self._TILE_SIZE_PER_R * R))
        if self.tile_overlap is None:
            self.tile_overlap = int(round(self._TILE_OVERLAP_PER_R * R))
        if self.pct_min_wavelength is None:
            self.pct_min_wavelength = self._WAVELENGTH_PER_W * w
        if self.mad_window is None:
            # mad_window must be odd; +1 on an even 2w keeps it odd.
            _mad_default = int(round(self._MAD_WINDOW_PER_W * w)) + 1
            if _mad_default % 2 == 0:
                _mad_default += 1
            self.mad_window = _mad_default
        if self.path_dilation_radius is None:
            self.path_dilation_radius = max(
                    1, int(round(self._PATH_DILATION_PER_W * w))
            )
        if self.snr_margin is None:
            self.snr_margin = max(2, int(round(self._SNR_MARGIN_PER_W * w)))
        if self.coherence_window_radius is None:
            self.coherence_window_radius = int(
                    round(self._COHERENCE_RADIUS_PER_W * w)
            )
        return self

    def _operate(self, image: 'GridImage') -> 'GridImage':
        """Detect and separate filamentous fungi using grid-based Voronoi partition.

        Algorithm:
        1. Run inoculum_detector to find fungal centers (full labeled regions)
        2. Detect branches via dual-mask pipeline (Gaussian + phase congruency)
        3. Filter centers, create grid markers, Voronoi assign with grid seeds
        4. Identify pseudo-fragments (per-label CCs not overlapping inoculum)
        5. Dijkstra reconnection of pseudo-fragments
        6. Final Voronoi partition with grid markers
        7. Set objmap with assignment results
        """

        from phenotypic import ImagePipeline

        # Validate that detectors are set before operation
        if self.inoculum_detector is None:
            raise ValueError(
                    "inoculum_detector is required but not set. "
                    "Provide a detector when creating FilamentousFungiDetector."
            )

        # ── PHASE 1: INOCULUM DETECTION ─────────────────────────────
        if isinstance(self.inoculum_detector, ImagePipeline):
            inoculum_img = self.inoculum_detector.apply(image, inplace=False,
                                                        reset=False)
        else:
            inoculum_img = self.inoculum_detector.apply(image, inplace=False)
        inoculum_objmask = inoculum_img.objmask[:]

        if inoculum_img.objmap[:].max() == 0:
            raise ValueError(
                    "No centers detected by inoculum_detector. Cannot perform "
                    "separation. Try adjusting inoculum_detector parameters or using a "
                    "different detection strategy."
            )

        self._log_memory_usage("after center detection")

        # ── PHASE 2: BRANCH DETECTION ───────────────────────────────
        # ContrastStretching-enhanced copy for dual-mask detection
        enhanced_work = image.copy()
        ContrastStretching().apply(enhanced_work, inplace=True)
        enhanced_arr = enhanced_work.detect_mat[:]
        enhanced_gray = enhanced_work.gray[:]  # capture before destructive call

        # Mask A: Gauss branches (destructive: modifies enhanced_work in place)
        bg_removed_arr = self._subtract_background(enhanced_work)
        del enhanced_work  # no longer valid after destructive call

        # Mask B: PCT branches
        pct_result = EnhanceFeatures(
                n_orient=self.pct_n_orient,
                min_wavelength=self.pct_min_wavelength,
                k=self.edge_noise_threshold,
        )._phasecong3(enhanced_arr)

        # Overlap filter: keep Gauss labels with any PCT overlap
        fragmented_overall_detect_mat = self._combine_bg_removed_with_pct(
                bg_removed_arr=bg_removed_arr,
                pct_sum=pct_result.pc_sum,
        )

        _fragmented_detect_img = image.copy()
        _fragmented_detect_img.detect_mat[:] = fragmented_overall_detect_mat
        HysteresisDetector(
                low="triangle",
                high="otsu",
                ignore_zeros=False,
                ignore_borders=self.ignore_borders
        ).apply(_fragmented_detect_img, inplace=True)
        overall_objmask = _fragmented_detect_img.objmask[:]
        del _fragmented_detect_img

        self._log_memory_usage("after dual-mask branch detection")

        # ── PHASE 3: CENTER FILTERING + GRID VORONOI ─────────────────

        # The filtered structure that overlaps with the inoculum centers
        inoculum_structure_mask = self._filter_mask_by_overlap(
                mask=overall_objmask, reference_mask=inoculum_objmask,
        )
        overlap_objmap = label(inoculum_structure_mask)

        if overlap_objmap.max() == 0:
            raise ValueError(
                    "No centers overlap with detected branch structure after "
                    "filtering. Check that inoculum_detector picks up the same "
                    "objects captured by the dual-mask branch detection."
            )

        self._log_memory_usage("after overlap filtering")

        centroid_markers = self._create_markers_from_centroids(inoculum_img.objmap[:])

        inoculum_structure_map = self._separate_colonies(centroid_markers,
                                                         inoculum_structure_mask)

        if inoculum_structure_map.max() == 0:
            raise RuntimeError(
                    "Voronoi assignment produced empty result. "
                    "Centroid markers may not overlap any foreground mask pixels."
            )

        self._log_memory_usage(
                "after Voronoi assignment",
                include_process=True,
                include_tracemalloc=True,
        )

        # ── PHASE 4: DIJKSTRA RECONNECTION ──────────────────────────
        colony_labels = inoculum_structure_map

        central_mask, fragment_labels = self._identify_pseudo_fragments(
                colony_labels=colony_labels,
                center_objmask=inoculum_objmask,
        )

        unmasked_cost, cost_surface = self._build_cost_surface(
                pct_result=pct_result,
                enhanced_arr=enhanced_arr,
                colony_labels=colony_labels,
                central_mask=central_mask,
        )

        colony_labels = self._reconnect_fragments_tiled(
                colony_labels=colony_labels,
                fragment_labels=fragment_labels,
                cost_surface=cost_surface,
                unmasked_cost=unmasked_cost,
                pct_energy=pct_result.pc_sum.astype(np.float32),
                grayscale=enhanced_gray,
        )

        self._log_memory_usage(
                "after Dijkstra reconnection",
                include_process=True,
                include_tracemalloc=True,
        )

        # ── PHASE 5: FINAL VORONOI ────────────────────────────────────
        final_mask = (colony_labels > 0) | inoculum_structure_mask
        colony_labels = self._separate_colonies(centroid_markers, final_mask)

        # ── PHASE 6: WRITE RESULT ───────────────────────────────────
        if colony_labels.dtype != image._OBJMAP_DTYPE:
            colony_labels = colony_labels.astype(image._OBJMAP_DTYPE)

        image.objmap[:] = colony_labels

        gc.collect()

        self._log_memory_usage(
                "final cleanup",
                include_process=True,
                include_tracemalloc=True,
        )

        return image

    # ── Phase 2 helpers ─────────────────────────────────────────────

    def _subtract_background(self, enhanced_work: 'Image') -> np.ndarray:
        """Subtracts the background of the input Image class. This potentially deletes
        branches so is combined downstream with the PCT response"""
        return SubtractGaussian(
                sigma=self.gauss_sigma, n_iter=self.gauss_n_iter
        ).apply(enhanced_work, inplace=False).detect_mat[:]

    @staticmethod
    def _combine_bg_removed_with_pct(
            bg_removed_arr: np.ndarray,
            pct_sum: np.ndarray,

    ):
        return np.maximum(
                bg_removed_arr,
                pct_sum,
        ).clip(min=0, max=1)

    # ── Phase 4 helpers ─────────────────────────────────────────────

    @staticmethod
    def _identify_pseudo_fragments(
            colony_labels: np.ndarray,
            center_objmask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Identify pseudo-fragments: per-label CCs that don't overlap inoculum.

        After grid Voronoi, every mask pixel has a label. CCs that overlap
        with the inoculum detection are "central" (main colony mass). CCs
        that don't are pseudo-fragments — blobs assigned to a section by
        proximity but not physically connected to the section's colony body.

        Args:
            colony_labels: Grid Voronoi label map.
            center_objmask: Inoculum detection binary mask.

        Returns:
            (central_mask, fragment_labels) where central_mask is the main
            colony mass and fragment_labels is a labeled map of
            pseudo-fragments.
        """
        foreground = colony_labels > 0
        cc_map, n_cc = ndi_label(foreground)

        if n_cc == 0:
            return (np.zeros_like(foreground),
                    np.zeros(foreground.shape, dtype=np.int32))

        # For each global CC: does it overlap inoculum?
        seeded_ccs = np.unique(cc_map[center_objmask & foreground])
        is_central = np.zeros(n_cc + 1, dtype=bool)
        is_central[seeded_ccs] = True

        central_mask = is_central[cc_map]
        fragment_mask = foreground & ~central_mask

        if fragment_mask.any():
            fragment_labels = label(fragment_mask).astype(np.int32)
        else:
            fragment_labels = np.zeros(foreground.shape, dtype=np.int32)

        return central_mask, fragment_labels

    def _apply_penalties_inplace(
            self,
            cost: np.ndarray,
            pct_energy: np.ndarray,
            colony_labels: np.ndarray,
    ) -> None:
        """Apply distance-gap and border penalties in place.

        Args:
            cost: 2D cost array to penalize (modified in place).
            pct_energy: 2D PCT energy map for gap penalty gating.
            colony_labels: Labeled colony assignment from watershed.
        """
        _apply_distance_gap_penalty_inplace(
                cost, pct_energy, colony_labels, self.gap_crossing_penalty,
        )
        _apply_border_penalty_inplace(cost, self.border_margin_px)

    def _build_cost_surface(
            self,
            pct_result: '_PhaseCong3Result',
            enhanced_arr: np.ndarray,
            colony_labels: np.ndarray,
            central_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build composite cost surface from PCT features.

        Reuses base_cost allocation: copies once for unmasked, then mutates
        the original for the masked surface.

        Args:
            pct_result: Phase congruency result containing M, m, orientation,
                and pc_sum fields.
            enhanced_arr: 2D contrast-stretched detection matrix for MAD
                computation.
            colony_labels: Labeled colony assignment from watershed.
            central_mask: Boolean mask of branch pixels overlapping colonies.

        Returns:
            Tuple of (unmasked_cost, cost_surface) where unmasked_cost is the
            composite cost before colony masking and cost_surface has
            colony/central pixels set to near-zero traversal cost.
        """
        # Anisotropy gives pixel level directional dependence
        anisotropy = compute_anisotropy(pct_result.M, pct_result.m)

        # Coherence is a measure of the length of the structures orientation
        coherence = compute_orientation_coherence(
                pct_result.orientation, self.coherence_window_radius
        )

        # For identifying noisy regions away from inoculum center
        mad = compute_local_mad_map(enhanced_arr, self.mad_window)

        base_cost = assemble_composite_cost(
                pct_result.pc_sum, anisotropy, coherence, mad,
                self.beta, self.gamma,
        )

        # Copy once for unmasked cost, then mutate original for masked
        unmasked_cost = base_cost.copy()
        self._apply_penalties_inplace(
                unmasked_cost, pct_result.pc_sum, colony_labels
        )

        colony_mask = (colony_labels > 0) | central_mask
        _apply_structure_mask_inplace(base_cost, colony_mask.astype(np.int32))
        self._apply_penalties_inplace(
                base_cost, pct_result.pc_sum, colony_labels
        )

        return unmasked_cost, base_cost

    def _reconnect_fragments_tiled(
            self,
            colony_labels: np.ndarray,
            fragment_labels: np.ndarray,
            cost_surface: np.ndarray,
            unmasked_cost: np.ndarray,
            pct_energy: np.ndarray,
            grayscale: np.ndarray,
    ) -> np.ndarray:
        """Generate tiles, process each, merge results into output mask.

        Args:
            colony_labels: Labeled colony assignment from watershed.
            fragment_labels: Labeled array of disconnected branch fragments.
            cost_surface: Masked composite cost surface for Dijkstra.
            unmasked_cost: Unmasked composite cost for quality calibration.
            pct_energy: Float32 (H, W) PCT energy map for quality filtering.
            grayscale: Float32 (H, W) enhanced grayscale for SNR filtering.

        Returns:
            Updated colony labels with reconnected fragments painted in.
        """
        if fragment_labels.max() == 0:
            return colony_labels

        # Prescreen fragments: compute envelope once, share across calibration + screening
        colony_branch_mask = (colony_labels > 0).astype(np.int32)
        min_cost_envelope, _ = _compute_screening_envelope(
                cost_surface, colony_branch_mask, self.frag_reach_px
        )
        tau_screen, _ = calibrate_screening_threshold(
                cost_surface, colony_branch_mask, r_screen=self.frag_reach_px,
                min_cost_envelope=min_cost_envelope,
        )

        screen_result = prescreen_fragments(
                cost_surface, fragment_labels,
                r_screen=self.frag_reach_px,
                tau_screen=tau_screen,
                colony_branch_mask=colony_branch_mask,
                min_cost_envelope=min_cost_envelope,
        )
        screened_frags = screen_result.screened_fragment_labels

        if screened_frags.max() == 0:
            return colony_labels

        # Compute PCT noise ceiling for F5 background masking
        pct_noise_ceil = float(threshold_otsu(pct_energy))

        # Generate tiles
        tiles = self._generate_tiles(
                colony_labels.shape, self.tile_size, self.tile_overlap
        )

        output = colony_labels.copy()

        for row_start, row_end, col_start, col_end in tiles:
            tile_cost = cost_surface[row_start:row_end, col_start:col_end]
            tile_raw = unmasked_cost[row_start:row_end, col_start:col_end]
            tile_colony = output[row_start:row_end, col_start:col_end]
            tile_frags = screened_frags[row_start:row_end, col_start:col_end]
            tile_pct = pct_energy[row_start:row_end, col_start:col_end]
            tile_gray = grayscale[row_start:row_end, col_start:col_end]

            tile_result = self._process_tile(
                    tile_cost, tile_raw, tile_colony, tile_frags,
                    tile_pct, tile_gray, pct_noise_ceil,
            )
            self._merge_tile_into_output(
                    output, tile_result, row_start, col_start
            )

        return output

    @staticmethod
    def _generate_tiles(
            image_shape: tuple[int, int],
            tile_size: int,
            overlap: int,
    ) -> list[tuple[int, int, int, int]]:
        """Generate overlapping tile coordinates covering the full image.

        Args:
            image_shape: (height, width) of the image.
            tile_size: Side length of square tiles.
            overlap: Overlap in pixels between adjacent tiles.

        Returns:
            List of (row_start, row_end, col_start, col_end) tuples.
        """
        H, W = image_shape
        step = tile_size - overlap
        tiles: list[tuple[int, int, int, int]] = []

        row = 0
        while row < H:
            row_end = min(row + tile_size, H)
            col = 0
            while col < W:
                col_end = min(col + tile_size, W)
                tiles.append((row, row_end, col, col_end))
                if col_end == W:
                    break
                col += step
            if row_end == H:
                break
            row += step

        return tiles

    def _process_tile(
            self,
            tile_cost: np.ndarray,
            tile_raw: np.ndarray,
            tile_colony: np.ndarray,
            tile_frags: np.ndarray,
            tile_pct: np.ndarray,
            tile_gray: np.ndarray,
            pct_noise_ceil: float,
    ) -> np.ndarray:
        """Process a single tile: Dijkstra, assign, paths, quality filter, assemble.

        Args:
            tile_cost: Masked cost surface for this tile.
            tile_raw: Unmasked cost surface for quality calibration.
            tile_colony: Colony labels for this tile.
            tile_frags: Fragment labels for this tile.
            tile_pct: PCT energy map for this tile.
            tile_gray: Grayscale image for this tile.
            pct_noise_ceil: PCT energy threshold for F5 background masking.

        Returns:
            Updated tile colony labels with reconnected fragments.
        """
        if tile_frags.max() == 0:
            return tile_colony

        if tile_colony.max() == 0:
            return tile_colony

        # Run Dijkstra from colony boundaries
        dijkstra = run_multisource_dijkstra(
                tile_cost, tile_colony, self.delta
        )

        # Assign fragments to colonies by majority vote
        assignments = assign_fragments_to_colonies(
                tile_frags, dijkstra.colony_id, dijkstra.cost_distance
        )

        # Extract minimum-cost paths from fragments to colonies
        paths, _unconnected = extract_fragment_paths(
                tile_frags, assignments, dijkstra, tile_cost
        )

        if not paths:
            return tile_colony

        # Quality filter: calibrate from colony skeleton branches
        calibration = extract_calibration_branches(
                tile_colony, tile_raw,
                window_cost=self.max_gap_length,
                dilation_radius=self.path_dilation_radius,
                pct_energy=tile_pct,
                grayscale=tile_gray,
                snr_margin=self.snr_margin,
                pct_noise_ceil=pct_noise_ceil,
        )

        # Only apply quality filters if we have calibration data
        if calibration.median_cost_values.size > 0:
            thresholds = calibrate_thresholds(
                    calibration, k=self.reconnection_tolerance
            )
            filter_result = apply_filter_cascade(
                    paths, tile_raw, thresholds,
                    window_cost=self.max_gap_length,
                    dilation_radius=self.path_dilation_radius,
                    pct_energy=tile_pct,
                    grayscale=tile_gray,
                    snr_margin=self.snr_margin,
                    pct_noise_ceil=pct_noise_ceil,
            )
            passed_ids = filter_result.passed_ids
        else:
            # No calibration data: accept all paths
            passed_ids = set(paths.keys())

        # Build result: paint fragment + dilated path with colony ID
        result = tile_colony.copy()
        selem = disk(self.path_dilation_radius)

        # Group path coords by colony for batched dilation
        colony_coords: dict[int, list[np.ndarray]] = {}

        for fid in passed_ids:
            if fid not in paths or fid not in assignments:
                continue
            path = paths[fid]
            cid = assignments[fid].colony_id
            if cid < 0:
                continue

            # Paint fragment pixels
            frag_mask = tile_frags == fid
            result[frag_mask] = cid

            # Collect path coords for batched dilation
            rows = path.coords[:, 0]
            cols = path.coords[:, 1]
            valid = (
                    (rows >= 0) & (rows < result.shape[0])
                    & (cols >= 0) & (cols < result.shape[1])
            )
            colony_coords.setdefault(cid, []).append(
                    path.coords[valid]
            )

        # Single dilation per colony
        for cid, coord_list in colony_coords.items():
            all_coords = np.vstack(coord_list)
            path_mask = np.zeros(result.shape, dtype=np.bool_)
            path_mask[all_coords[:, 0], all_coords[:, 1]] = True
            dilated = dilation(path_mask, selem)
            result[dilated] = cid

        return result

    @staticmethod
    def _merge_tile_into_output(
            output: np.ndarray,
            tile_labels: np.ndarray,
            row_start: int,
            col_start: int,
    ) -> None:
        """Write tile results into global output array.

        Only overwrites pixels that are currently unlabeled (0) in the output,
        preserving existing colony labels from earlier tiles or the watershed.

        Args:
            output: Global output label array (modified in place).
            tile_labels: Processed tile label array.
            row_start: Row offset of this tile in the global image.
            col_start: Column offset of this tile in the global image.
        """
        tile_h, tile_w = tile_labels.shape
        out_slice = output[row_start:row_start + tile_h, col_start:col_start + tile_w]
        new_pixels = (tile_labels > 0) & (out_slice == 0)
        out_slice[new_pixels] = tile_labels[new_pixels]

    # ── Existing static methods (unchanged) ─────────────────────────

    @staticmethod
    def _filter_mask_by_overlap(mask, reference_mask):
        """
        Retain only objects in mask_to_clean that overlap with reference_mask.

        Args:
            mask (np.ndarray): Binary mask to filter (2D boolean or uint8)
            reference_mask (np.ndarray): Binary mask defining valid regions (2D boolean or uint8)

        Returns:
            np.ndarray: Filtered binary mask with same shape as mask_to_clean

        Raises:
            ValueError: If masks don't have compatible spatial overlap
        """
        # Label connected components in mask to clean
        labeled = label(mask)

        # Handle potential size mismatch by finding overlapping region
        min_h = min(mask.shape[0], reference_mask.shape[0])
        min_w = min(mask.shape[1], reference_mask.shape[1])

        # Compute intersection in overlapping region
        intersection = labeled[:min_h, :min_w] * reference_mask[:min_h, :min_w]

        # Find which labels have overlap
        overlapping_labels = np.unique(intersection[intersection > 0])

        # Create output mask retaining only overlapping objects
        max_label = int(labeled.max())
        keep = np.zeros(max_label + 1, dtype=labeled.dtype)
        keep[overlapping_labels] = overlapping_labels

        return keep[labeled].astype(mask.dtype, copy=False)

    @staticmethod
    def _create_markers_from_centroids(objmap: np.ndarray) -> np.ndarray:
        """Create Voronoi seed markers at detected inoculum centroids.

        Args:
            objmap: Labeled integer array where each detected inoculum
                has a unique positive ID (from ``inoculum_img.objmap[:]``).

        Returns:
            2D int32 marker array with one seed per inoculum centroid.
        """
        labels = np.unique(objmap)
        labels = labels[labels > 0]

        markers = np.zeros(objmap.shape, dtype=np.int32)
        for marker_id, lbl in enumerate(labels, start=1):
            com = center_of_mass(objmap == lbl)
            r = min(int(round(com[0])), objmap.shape[0] - 1)
            c = min(int(round(com[1])), objmap.shape[1] - 1)
            markers[r, c] = marker_id

        return markers

    @staticmethod
    def _separate_colonies(
            markers: np.ndarray,
            mask: np.ndarray,
    ) -> np.ndarray:
        """Voronoi-partition mask pixels and correct fragment connectivity."""
        voronoi_map = euclidean_voronoi_assign(
                markers=markers,
                mask=mask,
                restrict_to_seeded_cc=False,
        )
        return connectivity_correct_labels(
                voronoi_labels=voronoi_map,
                mask=mask,
                markers=markers,
        )
