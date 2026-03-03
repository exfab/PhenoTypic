from __future__ import annotations
from typing import TYPE_CHECKING, Union
import numpy as np
import gc

if TYPE_CHECKING:
    from phenotypic._core._image import Image
    from phenotypic._core._image_pipeline import ImagePipeline  # type: ignore
    from phenotypic.enhance._phase_congruency import _PhaseCong3Result

from scipy.ndimage import center_of_mass, distance_transform_edt
from skimage import segmentation
from skimage.measure import label
from skimage.morphology import disk, dilation

from phenotypic.abc_ import ObjectDetector
from phenotypic import ImagePipeline
from phenotypic.enhance import (
    MedianFilter,
    BM3DDenoiser,
    BayesShrinkEnhancer,
    CLAHE,
    GrayOpening,
    GaussianBlur,
    SubtractGaussian,
    ContrastStretching,
    CoherenceEnhancingDiffusion,
    PhaseCongruencyEnhancer,
)

from phenotypic.detect import (
    OtsuDetector,
    RoundPeaksDetector,
    SecondaryOtsuDetector,
    TriangleDetector,
    HysteresisDetector,
)
from phenotypic.refine import MaskOpener, MaskCloser, GridSectionLargest

from phenotypic.detect._filamentous_fungi import (
    compute_anisotropy,
    compute_orientation_coherence,
    compute_local_mad_map,
    assemble_composite_cost,
    apply_structure_mask,
    calibrate_screening_threshold,
    prescreen_fragments,
    run_multisource_dijkstra,
    assign_fragments_to_colonies,
    extract_fragment_paths,
    extract_calibration_branches,
    calibrate_quality_thresholds,
    apply_quality_filters,
)


class FilamentousFungiDetector(ObjectDetector):
    """Detects and separates filamentous fungi using two-stage detection and inoculum-distance watershed.

    FilamentousFungiDetector uses two detection strategies to segment filamentous fungi:
    (1) inoculum_detector identifies compact fungal centers/nuclei, (2) overall_detector captures
    the complete fungal structure including spreading hyphae. The detector filters centers to
    those overlapping with the overall structure, then uses the full inoculum regions as seed
    markers for watershed on an inoculum-distance EDT elevation surface. The EDT gives each
    pixel its Euclidean distance to the nearest inoculum region, so each inoculum sits at
    elevation 0 (deepest basin) and the flood fills outward. Boundaries form where two floods
    meet. This keeps branches assigned to their origin colony: the origin flood reaches a
    branch's base before a neighbor's flood can traverse through the connection zone.

    When ``enable_reconnection=True``, the detector replaces the legacy ``overall_detector``
    path with a dual-mask branch detection pipeline (CED + Gaussian subtraction + phase
    congruency) followed by Dijkstra-based branch reconnection. Fragmented hyphal branches
    that fall outside the initial Voronoi watershed are reconnected to their parent colonies
    via minimum-cost paths through a composite cost surface derived from phase congruency
    features. Path quality is validated against calibration metrics from known-good colony
    skeleton branches.

    Args:
        inoculum_detector: ObjectDetector or ImagePipeline that identifies fungal centers/nuclei.
            Should produce small, compact regions at center points. Examples: OtsuDetector(),
            RoundPeaksDetector(), or preprocessing pipeline with GaussianBlur() + detector.

        overall_detector: ObjectDetector or ImagePipeline that captures complete fungal
            structures including hyphae and spreading edges. Should produce full fungal body
            masks. Examples: TriangleDetector(), CannyDetector(), or custom detector with
            lower threshold than inoculum_detector. Ignored when ``enable_reconnection=True``.

        enable_reconnection: When True, use dual-mask branch detection and Dijkstra-based
            reconnection instead of the legacy ``overall_detector`` path.

        ced_num_iter: Number of CED diffusion iterations.
        ced_sigma: CED noise scale (Gaussian pre-smoothing sigma).
        ced_rho: CED integration scale (structure tensor smoothing sigma).
        ced_C: CED contrast parameter controlling diffusion anisotropy.

        pct_n_orient: Number of orientations for phase congruency computation.
        pct_min_wavelength: Minimum wavelength for log-Gabor filters.
        pct_k: Noise threshold scaling factor for phase congruency.

        gauss_sigma: Sigma for SubtractGaussian background subtraction.
        gauss_n_iter: Number of SubtractGaussian iterations.

        morph_width: Disk radius for morphological open/close operations on branch masks.

        beta: Exponent on anisotropy in the composite cost formula.
        gamma: Weight of MAD penalty in the composite cost numerator.
        r_coherence: Radius for orientation coherence computation.
        mad_window: Window size for local MAD computation (must be odd).

        r_screen: Screening radius for fragment pre-screening.
        delta: Dijkstra radial penalty factor for retreating steps.
        quality_percentile: Percentile for calibrating path quality thresholds.
        path_dilation_radius: Disk radius for dilating reconnection paths.

        tile_size: Side length of square tiles for tiled Dijkstra processing.
        tile_overlap: Overlap in pixels between adjacent tiles.

    Returns:
        Image: Input image with objmask (binary mask) and objmap (labeled fungi) set.
            Each labeled fungus is separated by Voronoi assignment based on filtered
            centers. objmask is True for all fungal pixels within assigned regions.

    Raises:
        TypeError: If inoculum_detector or overall_detector are not ObjectDetector or
            ImagePipeline instances.
        ValueError: If no centers detected, no overall structure detected, or no centers
            overlap with overall structure after filtering.

    **Use cases**

    - **Filamentous fungal colonies:** Irregular spreading structures that require separate
      center and overall detection strategies for accurate segmentation.
    - **Touching/overlapping fungi:** Voronoi assignment using center seeds effectively
      separates filaments in close contact where simple thresholding merges them.
    - **Variable morphology:** Two-stage approach adapts to fungi with varying sizes,
      growth patterns, and hyphae density.
    - **Grid-based fungal cultures:** Works on GridImage with multiple wells containing
      filamentous fungi; can be integrated into ImagePipeline.
    - **High-throughput fungal phenotyping:** Enables batch processing of fungal plate
      images with consistent separation quality.

    **Limitations**

    - Requires two compatible detectors: centers must overlap significantly with overall
      structure, or ValueError is raised. Tuning both detectors is necessary.
    - Voronoi assignment quality depends on center detection accuracy: missing centers
      cause under-segmentation; spurious centers cause over-segmentation.
    - May over-segment if centers are too numerous or too close together.
    - Less suitable for circular, yeast-like colonies; use WatershedDetector instead
      for round morphologies.
    - **Memory per tile**: Dijkstra heap = ``2*tile_H*tile_W`` entries. With
      ``tile_size=1200``, ~35 MB per tile.
    - **Tile boundary artifacts**: Fragments split across tiles may be partially
      reconnected. Overlap (100px) mitigates this.
    - **F3 displacement filter**: No-op when colony branches < ``window_disp`` (40px).
    - **Single-threaded Dijkstra**: ~7s per 1200x1200 tile.

    Examples:
        Detect and separate filamentous fungi with center and overall detection:

        >>> from phenotypic.detect import FilamentousFungiDetector, OtsuDetector, TriangleDetector
        >>> from phenotypic.data import load_synth_yeast_plate
        >>>
        >>> # Create detector: centers detected via OtsuDetector, overall via TriangleDetector
        >>> detector = FilamentousFungiDetector(
        ...     inoculum_detector=OtsuDetector(ignore_zeros=True),
        ...     overall_detector=TriangleDetector(),
        ... )
        >>>
        >>> # Note: load_synth_plate returns circular colonies; example is illustrative
        >>> image = load_synth_yeast_plate()
        >>> result = detector.apply(image)
        >>> num_fungi = result.objmap[:].max()
        >>> print(f"Detected and separated {num_fungi} fungal colonies")

        Integration in processing pipeline with preprocessing and refinement:

        >>> from phenotypic import ImagePipeline
        >>> from phenotypic.enhance import GaussianBlur, CLAHE
        >>> from phenotypic.refine import SmallObjectRemover
        >>> from phenotypic.data import load_synth_yeast_plate
        >>>
        >>> # Build pipeline with enhancement, two-stage fungi detection, and cleanup
        >>> pipeline = ImagePipeline([
        ...     GaussianBlur(sigma=1.5),
        ...     CLAHE(clip_limit=2.0),
        ...     FilamentousFungiDetector(
        ...         inoculum_detector=ImagePipeline([
        ...             GaussianBlur(sigma=0.5),
        ...             OtsuDetector()
        ...         ]),
        ...         overall_detector=TriangleDetector(),
        ...     ),
        ...     SmallObjectRemover(min_size=100)
        ... ])
        >>>
        >>> image = load_synth_yeast_plate()
        >>> result = pipeline.apply(image)
        >>> print(f"Final separated fungi: {result.objmap[:].max()}")
    """
    __overall_pipe = ImagePipeline(
            ops=[
                MedianFilter(),
                BM3DDenoiser(),
                CLAHE(kernel_size=500),
                GrayOpening(),
                BayesShrinkEnhancer(
                        wavelet="db4",
                        mode="hard"
                ),
                OtsuDetector(ignore_zeros=True),
            ]
    )

    __center_pipe = ImagePipeline(
            ops=[
                GaussianBlur(sigma=5),
                SubtractGaussian(sigma=500),
                RoundPeaksDetector(thresh_method="triangle"),
                SecondaryOtsuDetector(ignore_zeros=True),
                MaskOpener(),
                GridSectionLargest(),
            ]
    )

    def __init__(
            self,
            inoculum_detector: Union[ObjectDetector, 'ImagePipeline', None] = None,
            overall_detector: Union[ObjectDetector, 'ImagePipeline', None] = None,
            # Reconnection parameters
            enable_reconnection: bool = False,
            ced_num_iter: int = 50,
            ced_sigma: float = 3.0,
            ced_rho: float = 6.0,
            ced_C: float = 90.0,
            pct_n_orient: int = 8,
            pct_min_wavelength: float = 5.0,
            pct_k: float = 6.0,
            gauss_sigma: float = 300.0,
            gauss_n_iter: int = 2,
            morph_width: int = 5,
            beta: float = 2.0,
            gamma: float = 1.0,
            r_coherence: int = 12,
            mad_window: int = 7,
            r_screen: int = 10,
            delta: float = 1.0,
            quality_percentile: float = 95.0,
            path_dilation_radius: int = 2,
            tile_size: int = 1200,
            tile_overlap: int = 100,
    ):
        super().__init__()

        # Type validation (allow None for serialization/deserialization)
        from phenotypic import ImagePipeline

        if inoculum_detector is not None and not isinstance(inoculum_detector,
                                                            (ObjectDetector,
                                                             ImagePipeline)):
            raise TypeError(
                    "inoculum_detector must be an ObjectDetector or ImagePipeline instance, "
                    f"got {type(inoculum_detector).__name__}"
            )
        if overall_detector is not None and not isinstance(overall_detector,
                                                           (ObjectDetector,
                                                            ImagePipeline)):
            raise TypeError(
                    "overall_detector must be an ObjectDetector or ImagePipeline instance, "
                    f"got {type(overall_detector).__name__}"
            )

        self.inoculum_detector = inoculum_detector if inoculum_detector \
            else self.__center_pipe

        self.overall_detector = overall_detector if overall_detector \
            else self.__overall_pipe

        # Reconnection parameters
        self.enable_reconnection = enable_reconnection
        self.ced_num_iter = ced_num_iter
        self.ced_sigma = ced_sigma
        self.ced_rho = ced_rho
        self.ced_C = ced_C
        self.pct_n_orient = pct_n_orient
        self.pct_min_wavelength = pct_min_wavelength
        self.pct_k = pct_k
        self.gauss_sigma = gauss_sigma
        self.gauss_n_iter = gauss_n_iter
        self.morph_width = morph_width
        self.beta = beta
        self.gamma = gamma
        self.r_coherence = r_coherence
        self.mad_window = mad_window
        self.r_screen = r_screen
        self.delta = delta
        self.quality_percentile = quality_percentile
        self.path_dilation_radius = path_dilation_radius
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap

    def _operate(self, image: 'Image') -> 'Image':
        """Detect and separate filamentous fungi using two-stage detection and inoculum-distance watershed.

        Algorithm:
        1. Run inoculum_detector to find fungal centers (full labeled regions)
        2. Detect branches via dual-mask pipeline (reconnection) or overall_detector (legacy)
        3. Filter centers to keep only those overlapping with overall structure
        4. Use full inoculum regions as watershed markers (masked to overall structure)
        5. Watershed on inoculum-distance EDT assigns each mask pixel to nearest inoculum
        6. Dijkstra reconnection of fragmented branches (reconnection mode only)
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
            center_result = self.inoculum_detector.apply(image, inplace=False,
                                                         reset=False)
        else:
            center_result = self.inoculum_detector.apply(image, inplace=False)
        center_objmask = center_result.objmask[:]
        center_objmap = center_result.objmap[:]

        if center_objmap.max() == 0:
            raise ValueError(
                    "No centers detected by inoculum_detector. Cannot perform "
                    "separation. Try adjusting inoculum_detector parameters or using a "
                    "different detection strategy."
            )

        self._log_memory_usage("after center detection")

        # ── PHASE 2: BRANCH DETECTION ───────────────────────────────
        if self.enable_reconnection:
            # Dual-mask approach on internal copy
            ced_work = image.copy()
            CoherenceEnhancingDiffusion(
                num_iter=self.ced_num_iter, sigma=self.ced_sigma,
                rho=self.ced_rho, C=self.ced_C, dt=0.125
            ).apply(ced_work, inplace=True)
            ced_arr = ced_work.detect_mat[:]

            # Mask A: Gauss branches
            gauss_labels = self._detect_gauss_branches(ced_work)

            # Mask B: PCT branches
            pct_mask, pct_result = self._detect_pct_branches(ced_arr)

            # Overlap filter: keep Gauss labels with any PCT overlap
            branch_labels = self._filter_gauss_by_pct_overlap(gauss_labels, pct_mask)
            overall_objmask = branch_labels > 0

            self._log_memory_usage("after dual-mask branch detection")
        else:
            # Legacy path
            if self.overall_detector is None:
                raise ValueError(
                        "overall_detector is required but not set. "
                        "Provide a detector when creating FilamentousFungiDetector."
                )
            if isinstance(self.overall_detector, ImagePipeline):
                overall_result = self.overall_detector.apply(image, inplace=False,
                                                             reset=False)
            else:
                overall_result = self.overall_detector.apply(image, inplace=False)

            overall_objmask = overall_result.objmask[:]

            if overall_result.num_objects == 0:
                raise ValueError(
                        "No overall structure detected by overall_detector. Cannot create "
                        "assignment mask. Try adjusting overall_detector parameters."
                )

            branch_labels = None

            self._log_memory_usage("after overall detection")

        # ── PHASE 3: CENTER FILTERING + WATERSHED ───────────────────
        filtered_center_objmask = self._filter_centers_by_overlap(
                center_mask=center_objmask, overall_mask=overall_objmask
        )
        overlap_objmap = label(filtered_center_objmask)

        if overlap_objmap.max() == 0:
            raise ValueError(
                    "No centers overlap with overall structure after filtering. "
                    "Check that inoculum_detector and overall_detector are compatible "
                    "(detecting the same objects)."
            )

        self._log_memory_usage("after overlap filtering")

        region_markers = center_objmap.copy()
        region_markers[~overall_objmask] = 0

        colony_labels = self._voronoi_assign(region_markers, overall_objmask)

        if colony_labels.max() == 0:
            raise RuntimeError(
                    "Voronoi assignment produced empty result. "
                    "Marker centroids may not overlap with the overall mask."
            )

        self._log_memory_usage(
                "after Voronoi assignment",
                include_process=True,
                include_tracemalloc=True,
        )

        # ── PHASE 4: DIJKSTRA RECONNECTION ──────────────────────────
        if self.enable_reconnection and branch_labels is not None:
            central_mask, fragment_labels = self._separate_central_and_fragments(
                branch_labels, colony_labels
            )

            raw_cost, cost_surface = self._build_cost_surface(
                pct_result, ced_arr, colony_labels, central_mask
            )

            colony_labels = self._reconnect_fragments_tiled(
                colony_labels, fragment_labels, cost_surface, raw_cost
            )

            self._log_memory_usage(
                    "after Dijkstra reconnection",
                    include_process=True,
                    include_tracemalloc=True,
            )

        # ── PHASE 5: WRITE RESULT ───────────────────────────────────
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

    def _detect_gauss_branches(self, ced_work: 'Image') -> np.ndarray:
        """Apply ContrastStretching, SubtractGaussian, TriangleDetector, and morphology on copy.

        Args:
            ced_work: CED-enhanced image (will not be modified; operates on a copy).

        Returns:
            Labeled array of detected Gaussian branches.
        """
        work = ced_work.copy()
        ContrastStretching().apply(work, inplace=True)
        SubtractGaussian(sigma=self.gauss_sigma, n_iter=self.gauss_n_iter).apply(
            work, inplace=True
        )
        TriangleDetector().apply(work, inplace=True)
        MaskOpener(shape="disk", width=self.morph_width, n_iter=1).apply(
            work, inplace=True
        )
        MaskCloser(shape="disk", width=self.morph_width, n_iter=2).apply(
            work, inplace=True
        )
        MaskOpener(shape="disk", width=self.morph_width, n_iter=1).apply(
            work, inplace=True
        )
        MaskCloser(shape="disk", width=self.morph_width, n_iter=2).apply(
            work, inplace=True
        )
        return work.objmap[:]

    def _detect_pct_branches(
            self, ced_arr: np.ndarray
    ) -> tuple[np.ndarray, '_PhaseCong3Result']:
        """Run phase congruency on CED array and apply hysteresis threshold.

        Args:
            ced_arr: 2D CED-enhanced detection matrix.

        Returns:
            Tuple of (binary_mask, pct_result) where binary_mask is the
            hysteresis-thresholded phase congruency mask and pct_result
            contains the raw phase congruency feature maps.
        """
        from phenotypic._core._image import Image

        pct = PhaseCongruencyEnhancer(
            n_orient=self.pct_n_orient,
            min_wavelength=self.pct_min_wavelength,
            k=self.pct_k,
        )
        pct_result = pct._phasecong3(ced_arr)

        # Create temporary Image from pc_sum for hysteresis detection
        temp = Image(arr=pct_result.pc_sum)
        temp = HysteresisDetector(low="triangle", high="otsu").apply(temp)
        pct_mask = temp.objmask[:]

        return pct_mask, pct_result

    @staticmethod
    def _filter_gauss_by_pct_overlap(
            gauss_labels: np.ndarray, pct_mask: np.ndarray
    ) -> np.ndarray:
        """LUT-based label filtering: keep Gauss labels with any pixel overlapping PCT mask.

        Args:
            gauss_labels: Labeled array of Gaussian branch detections.
            pct_mask: Binary mask from phase congruency hysteresis detection.

        Returns:
            Filtered label array with non-overlapping labels zeroed out.
        """
        intersection = gauss_labels * pct_mask
        overlap_labels = np.unique(intersection[intersection > 0])
        max_label = int(gauss_labels.max())
        if max_label == 0:
            return gauss_labels.copy()
        keep = np.zeros(max_label + 1, dtype=gauss_labels.dtype)
        keep[overlap_labels] = overlap_labels
        return keep[gauss_labels]

    # ── Phase 4 helpers ─────────────────────────────────────────────

    @staticmethod
    def _separate_central_and_fragments(
            branch_labels: np.ndarray, colony_labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Separate branch pixels into central (overlapping colony) and fragments.

        Args:
            branch_labels: Labeled array of detected branches.
            colony_labels: Labeled colony assignment from watershed.

        Returns:
            Tuple of (central_mask, fragment_labels) where central_mask is a
            boolean mask of branch pixels overlapping assigned colonies, and
            fragment_labels is a labeled array of disconnected branch fragments.
        """
        central_mask = (branch_labels > 0) & (colony_labels > 0)
        fragment_mask = (branch_labels > 0) & (colony_labels == 0)

        if np.any(fragment_mask):
            fragment_labels = label(fragment_mask)
        else:
            fragment_labels = np.zeros_like(branch_labels, dtype=np.int32)

        return central_mask, fragment_labels

    def _build_cost_surface(
            self,
            pct_result: '_PhaseCong3Result',
            ced_arr: np.ndarray,
            colony_labels: np.ndarray,
            central_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build composite cost surface from PCT features.

        Args:
            pct_result: Phase congruency result containing M, m, orientation,
                and pc_sum fields.
            ced_arr: 2D CED-enhanced detection matrix for MAD computation.
            colony_labels: Labeled colony assignment from watershed.
            central_mask: Boolean mask of branch pixels overlapping colonies.

        Returns:
            Tuple of (raw_cost, cost_surface) where raw_cost is the unmasked
            composite cost and cost_surface has colony/central pixels set to
            near-zero traversal cost.
        """
        anisotropy = compute_anisotropy(pct_result.M, pct_result.m)
        coherence = compute_orientation_coherence(
            pct_result.orientation, self.r_coherence
        )
        mad = compute_local_mad_map(ced_arr, self.mad_window)

        raw_cost = assemble_composite_cost(
            pct_result.pc_sum, anisotropy, coherence, mad,
            self.beta, self.gamma,
        )

        colony_mask = (colony_labels > 0) | central_mask
        cost_surface = apply_structure_mask(
            raw_cost, colony_mask.astype(np.int32)
        )

        return raw_cost, cost_surface

    def _reconnect_fragments_tiled(
            self,
            colony_labels: np.ndarray,
            fragment_labels: np.ndarray,
            cost_surface: np.ndarray,
            raw_cost: np.ndarray,
    ) -> np.ndarray:
        """Generate tiles, process each, merge results into output mask.

        Args:
            colony_labels: Labeled colony assignment from watershed.
            fragment_labels: Labeled array of disconnected branch fragments.
            cost_surface: Masked composite cost surface for Dijkstra.
            raw_cost: Unmasked composite cost for quality calibration.

        Returns:
            Updated colony labels with reconnected fragments painted in.
        """
        if fragment_labels.max() == 0:
            return colony_labels

        # Prescreen fragments: calibrate threshold from colony boundaries
        colony_branch_mask = (colony_labels > 0).astype(np.int32)
        tau_screen, _ = calibrate_screening_threshold(
            cost_surface, colony_branch_mask, r_screen=self.r_screen
        )

        screen_result = prescreen_fragments(
            cost_surface, fragment_labels,
            r_screen=self.r_screen,
            tau_screen=tau_screen,
            colony_branch_mask=colony_branch_mask,
        )
        screened_frags = screen_result.screened_fragment_labels

        if screened_frags.max() == 0:
            return colony_labels

        # Generate tiles
        tiles = self._generate_tiles(
            colony_labels.shape, self.tile_size, self.tile_overlap
        )

        output = colony_labels.copy()

        for row_start, row_end, col_start, col_end in tiles:
            tile_cost = cost_surface[row_start:row_end, col_start:col_end]
            tile_raw = raw_cost[row_start:row_end, col_start:col_end]
            tile_colony = output[row_start:row_end, col_start:col_end].copy()
            tile_frags = screened_frags[row_start:row_end, col_start:col_end].copy()

            tile_result = self._process_tile(
                tile_cost, tile_raw, tile_colony, tile_frags
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
    ) -> np.ndarray:
        """Process a single tile: Dijkstra, assign, paths, quality filter, assemble.

        Args:
            tile_cost: Masked cost surface for this tile.
            tile_raw: Unmasked cost surface for quality calibration.
            tile_colony: Colony labels for this tile.
            tile_frags: Fragment labels for this tile.

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
            tile_colony, tile_raw
        )

        # Only apply quality filters if we have calibration data
        if calibration.cpl_values.size > 0:
            thresholds = calibrate_quality_thresholds(
                calibration, percentile=self.quality_percentile
            )
            paths_obj: dict[int, object] = dict(paths)
            filter_result = apply_quality_filters(paths_obj, thresholds)
            passed_ids = filter_result.passed_ids
        else:
            # No calibration data: accept all paths
            passed_ids = set(paths.keys())

        # Build result: paint fragment + dilated path with colony ID
        result = tile_colony.copy()
        selem = disk(self.path_dilation_radius)

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

            # Paint dilated path pixels
            path_mask = np.zeros_like(result, dtype=np.bool_)
            rows = path.coords[:, 0]
            cols = path.coords[:, 1]
            # Clip to tile bounds
            valid = (
                (rows >= 0) & (rows < result.shape[0]) &
                (cols >= 0) & (cols < result.shape[1])
            )
            path_mask[rows[valid], cols[valid]] = True
            dilated_path = dilation(path_mask, selem)
            result[dilated_path & (result == 0)] = cid

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
    def _filter_mask_by_overlap(mask_to_clean, reference_mask):
        """
        Retain only objects in mask_to_clean that overlap with reference_mask.

        Args:
            mask_to_clean (np.ndarray): Binary mask to filter (2D boolean or uint8)
            reference_mask (np.ndarray): Binary mask defining valid regions (2D boolean or uint8)

        Returns:
            np.ndarray: Filtered binary mask with same shape as mask_to_clean

        Raises:
            ValueError: If masks don't have compatible spatial overlap
        """
        # Label connected components in mask to clean
        labeled = label(mask_to_clean)

        # Handle potential size mismatch by finding overlapping region
        min_h = min(mask_to_clean.shape[0], reference_mask.shape[0])
        min_w = min(mask_to_clean.shape[1], reference_mask.shape[1])

        # Compute intersection in overlapping region
        intersection = labeled[:min_h, :min_w] * reference_mask[:min_h, :min_w]

        # Find which labels have overlap
        overlapping_labels = np.unique(intersection[intersection > 0])

        # Create output mask retaining only overlapping objects
        max_label = int(labeled.max())
        keep = np.zeros(max_label + 1, dtype=labeled.dtype)
        keep[overlapping_labels] = overlapping_labels
        labeled[:] = keep[labeled]

        return labeled.astype(mask_to_clean.dtype)

    @staticmethod
    def _filter_centers_by_overlap(
            center_mask: np.ndarray, overall_mask: np.ndarray
    ) -> np.ndarray:
        """Filter center mask to keep only centers overlapping with overall structure.

        Args:
            center_mask: Binary mask of detected centers (2D boolean or uint8)
            overall_mask: Binary mask of overall structure (2D boolean or uint8)

        Returns:
            Filtered binary mask containing only overlapping centers

        Raises:
            ValueError: If masks have incompatible shapes
        """

        # Label centers
        labeled_centers = label(center_mask)  # type: ignore
        labeled_centers = labeled_centers.astype(np.uint16)

        # Handle size mismatch by cropping to overlapping region
        min_h: int = min(center_mask.shape[0], overall_mask.shape[0])
        min_w: int = min(center_mask.shape[1], overall_mask.shape[1])

        labeled_centers_crop = labeled_centers[:min_h, :min_w]
        overall_mask_crop = overall_mask[:min_h, :min_w]

        # Find center labels that overlap with overall structure
        overlap_region = (labeled_centers_crop > 0) & (overall_mask_crop > 0)
        overlapping_labels_all = np.unique(labeled_centers_crop[overlap_region])
        overlapping_labels: np.ndarray = overlapping_labels_all[
            overlapping_labels_all > 0]

        # Create filtered binary mask keeping only overlapping centers
        filtered_mask = np.isin(labeled_centers, overlapping_labels)

        return filtered_mask.astype(center_mask.dtype)

    @staticmethod
    def _create_markers_from_centers(center_objmap: np.ndarray) -> np.ndarray:
        """Create assignment markers from center objects using centroids.

        Computes centroid of each labeled center region and places a unique
        marker at that position. Ensures exactly one marker per center object.

        Args:
            center_objmap: Labeled map of center objects (2D integer array)

        Returns:
            Marker array with unique integers at centroid positions (2D int32)

        Note:
            Uses scipy.ndimage.center_of_mass() for accurate centroid computation.
            Rounds to integers for pixel coordinates.
        """

        # Create empty markers array
        markers = np.zeros_like(center_objmap, dtype=np.int32)

        # Get unique center labels (excluding background 0)
        center_labels_all = np.unique(center_objmap)
        center_labels: np.ndarray = center_labels_all[center_labels_all > 0]

        # Place marker at centroid of each center
        for marker_id, label_val in enumerate(center_labels, start=1):
            # Compute centroid
            mask = center_objmap == label_val
            centroid = center_of_mass(mask)

            # Round to integer coordinates
            row = int(round(centroid[0]))
            col = int(round(centroid[1]))

            # Bounds checking to ensure marker is placed inside the image
            if 0 <= row < markers.shape[0] and 0 <= col < markers.shape[1]:
                markers[row, col] = marker_id

        return markers

    @staticmethod
    def _voronoi_assign(markers: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Assign each masked pixel to its nearest inoculum region via distance-weighted watershed.

        Computes a Euclidean distance transform from the inoculum marker regions
        (``markers == 0`` gives all non-inoculum pixels). Each inoculum region
        sits at elevation 0 (deepest basin); the watershed floods outward from
        each inoculum through the mask. Boundaries form where two floods meet
        at equidistant points from their respective inoculums.

        This keeps branches with their origin colony: the origin flood fills
        a branch from its base (low elevation, near inoculum) before a
        neighbor's flood can reach the branch tip through the connection zone.

        Disconnected mask regions with no marker remain unlabeled (0).

        Args:
            markers: 2D integer array with labeled inoculum regions as seeds.
                Non-zero values are colony labels; zero is background.
            mask: Binary mask constraining the watershed flood region
                (overall fungal structure).

        Returns:
            Labeled 2D array where each masked pixel has the label of its
            nearest inoculum region by Euclidean distance, constrained to
            the mask.
        """
        elevation = distance_transform_edt(markers == 0)
        return segmentation.watershed(
                elevation, markers, mask=mask, connectivity=2
        )
