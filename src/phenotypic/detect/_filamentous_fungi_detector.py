from __future__ import annotations
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, Optional, Union
import numpy as np
import gc

from pydantic import model_validator
from typing_extensions import Self

if TYPE_CHECKING:
    from phenotypic._core._image import Image
    from phenotypic._core._grid_image import GridImage
    from phenotypic.sdk_.reconnect import ReconnectConfig

from skimage.measure import label

from phenotypic.abc_ import GridObjectDetector
from phenotypic import ImagePipeline
from phenotypic.enhance import (
    SubtractGaussian,
    ContrastStretching,
    FocusEdgePhase,
)
from phenotypic.sdk_.typing_ import (
    FilamentousFungiReconnectStrategy,
    OperationField,
    TuneSpec,
)

from phenotypic.detect import HysteresisDetector
from phenotypic.detect._inoculum_detector import InoculumDetector
from phenotypic.refine import KeepSectionLargest

# `phenotypic.sdk_.reconnect` is imported inside the two methods that use it,
# not here. Its `_tensor_voting` kernel applies `@numba.njit` at import time, so
# a module-scope import would make `import phenotypic` load numba (and, through
# `numba.misc.coverage_support`, coverage) for every entry point — while only a
# run that actually reaches this detector needs it. `from __future__ import
# annotations` is in force, so the `-> ReconnectConfig` annotation below is a
# string and does not force the import either.


class FilamentousFungiDetector(GridObjectDetector):
    """Detect and individually label filamentous fungal colonies by two-stage inoculum-plus-hyphae detection with Euclidean Voronoi partition.

    First detect compact inoculation centres with ``inoculum_detector``, then
    capture the full hyphal network using phase-congruency edge detection
    combined with Gaussian background subtraction. Disconnected branch
    fragments are reconnected to their parent colonies via quality-filtered
    Dijkstra pathfinding on a composite cost surface derived from phase
    congruency energy, local texture, and orientation coherence. Inoculum
    centroids seed a Euclidean Voronoi partition that assigns every fungal
    pixel to its nearest colony, with connectivity-based correction enforcing
    uniform labelling within each connected component.

    For an algorithm overview and a comparison with other detection strategies,
    see :doc:`/explanation/detection_strategies_compared` and
    :doc:`/explanation/filamentous_fungi_algorithm`.

    Best For:
        - Filamentous fungal colonies (*Aspergillus*, *Neurospora*,
          *Trichoderma*) with irregular, spreading hyphal morphologies.
        - Dense plates where neighbouring fungal colonies touch or overlap
          and must be individually labelled.
        - Time-course experiments tracking hyphal extension radially outward
          from compact inoculation sites.
        - Grid-based fungal culture plates where one colony per well must be
          quantified separately.
        - High-throughput fungal phenotyping screens requiring consistent
          separation quality across hundreds of plates.

    Consider Also:
        - :class:`WatershedDetector` when colonies are compact and roughly
          circular (yeast or bacterial morphology).
        - :class:`OtsuDetector` when fungi are well-separated and a single
          binary mask suffices without individual colony labelling.
        - :class:`CompositeDetector` when combining multiple detection
          strategies is preferred over the two-stage centre-plus-hyphae
          approach.
        - :class:`InoculumDetector` when only the compact inoculation centres
          are needed and full hyphal reconstruction is not required.

    Args:
        inoculum_detector: ObjectDetector or ImagePipeline used to locate
            compact fungal centres. Should produce small, tight regions at
            inoculation points; centroids from this detector seed the final
            Voronoi partition. When ``None`` (default), an internal
            ``InoculumDetector`` + ``KeepSectionLargest`` pipeline is used.
            Default: None.

        # Scene-scale parameters — set these first; derived params follow
        max_colony_radius_px: Expected maximum colony radius in pixels.
            Acts as the master scene knob: proportionally scales
            ``gauss_sigma``, ``tile_size``, and ``tile_overlap`` when those
            are left at ``None``. A reasonable starting point is the radius
            of the largest colony in pixels at your imaging resolution (e.g.
            measure colony extent in your image viewer before setting this).
            Reduce for short-incubation plates or high-well-count formats;
            increase for slow-growing species with extensive radial growth.
            Typical range: 50--400. Default: 250.0.
        min_branch_width_px: Expected narrowest hyphal branch width in
            pixels. Scales signal-detection parameters
            (``pct_min_wavelength``, ``mad_window``, ``path_dilation_radius``,
            ``snr_margin``, ``coherence_window_radius``) when those are left
            at ``None``. Set to the thinnest hyphae visible at your imaging
            resolution; the derived ``pct_min_wavelength``
            (``2 × min_branch_width_px``) is clamped at the Nyquist floor
            of 2 px. Typical range: 2--8. Default: 3.

        # Detection control
        ignore_borders: Drop objects touching the image border during
            hysteresis-threshold branch detection. Enable (default) to avoid
            partial colonies at plate edges; disable when genuine peripheral
            hyphal growth must be retained. Default: True.
        edge_noise_threshold: Noise-suppression multiplier ``k`` for the
            phase congruency detector. Only features whose phase energy
            exceeds the estimated noise mean plus ``k`` standard deviations of
            the noise energy are accepted as real edges. Higher values suppress
            agar texture artefacts at the cost of rejecting weak peripheral
            hyphae; lower values recover fine structure but may pass background
            noise on textured media. Typical range: 2.0--10.0. Default: 6.0.

        # Reconnection quality
        reconnection_tolerance: IQR multiplier for calibrating reconnection
            path quality thresholds from confirmed calibration branches.
            Thresholds are set at median ± ``reconnection_tolerance`` × IQR
            across five quality metrics. Higher values accept more candidate
            paths (permissive); lower values require paths to closely resemble
            calibration branches (conservative). Typical range: 1.5--4.0.
            Default: 2.5.
        max_gap_length: Maximum contiguous stretch of high-cost pixels
            tolerated along a reconnection path, in pixels. Paths containing
            a window worse than the calibrated threshold are rejected as
            routing through bare agar. Increase to bridge longer hyphal gaps;
            decrease to reject longer detours through background. Typical
            range: 10--100. Default: 30.
        border_margin_px: Width of the border penalty ramp applied to
            image-edge pixels in the Dijkstra cost surface. Prevents
            reconnection paths from routing along plate borders instead of
            through genuine hyphal corridors. Set to 0 to disable. Typical
            range: 0--150. Default: 50.
        frag_reach_px: Pre-screening radius in pixels. Fragments whose
            nearest routable pixel exceeds this distance from the colony
            boundary are discarded before Dijkstra, saving computation.
            Fragments within this radius are forwarded for full
            quality-filtered reconnection. Typical range: 5--40. Default: 10.
        gap_crossing_penalty: Scaling factor for the distance-weighted gap
            penalty applied to Dijkstra path costs. Higher values strongly
            penalise traversal of bare agar far from the colony, keeping
            paths near established structure; lower values allow longer
            background detours. Typical range: 1.0--10.0. Default: 4.0.
        reconnect_strategy: Reconnection edge-cost strategy. ``"dijkstra"``
            preserves the legacy destination-only composite-cost recurrence.
            ``"app2_gwdt"`` computes one full-image APP2 grey-weighted
            distance transform from the detected foreground, applies its fixed
            GI lookup, and uses source-faithful endpoint-averaged GI edge
            costs. Default: ``"dijkstra"``.

        # Scene-derivation overrides (leave at None to auto-derive)
        gauss_sigma: Gaussian sigma for background subtraction, in pixels.
            When ``None``, set to ``1.2 × max_colony_radius_px`` (300 px at
            the default radius). Must exceed the largest colony radius so the
            Gaussian estimates only the illumination gradient, not colony
            signal. Typical range: 50--600. Default: None.
        tile_size: Side length of square processing tiles in pixels. When
            ``None``, set to ``int(round(4.8 × max_colony_radius_px))``
            (1200 px at the default radius). Must be large enough to contain
            an entire colony and its satellite fragments within one tile.
            Typical range: 200--3000. Default: None.
        tile_overlap: Overlap between adjacent tiles in pixels. When
            ``None``, set to ``int(round(2.4 × max_colony_radius_px))``
            (600 px at the default radius). Larger overlap ensures fragments
            near tile boundaries are co-located with their parent colony in
            at least one tile. Typical range: 50--1500. Default: None.
        pct_min_wavelength: Minimum log-Gabor filter wavelength in pixels
            for phase congruency detection. When ``None``, set to
            ``2.0 × min_branch_width_px`` (6 px at the default width).
            Must be ≥ 2 (Nyquist). Match to the thinnest hyphae width at
            your imaging resolution. Typical range: 2--20. Default: None.
        mad_window: Side length of the square median-filter kernel for local
            MAD texture computation (must be odd). When ``None``, set to
            ``2 × min_branch_width_px + 1`` forced odd (7 at the default
            width). Should span approximately one branch diameter plus
            background buffer on each side. Typical range: 3--21.
            Default: None.
        path_dilation_radius: Disk radius for dilating accepted reconnection
            paths before painting colony labels. When ``None``, set to
            ``max(1, round(0.5 × min_branch_width_px))`` (2 at the default
            width). Also sets the inner band radius for path quality metrics.
            Match to half the expected hyphal width. Typical range: 1--10.
            Default: None.
        snr_margin: Extra pixel margin beyond ``path_dilation_radius`` that
            forms the background annular ring for local SNR estimation.
            When ``None``, set to ``max(2, round(0.5 × min_branch_width_px))``
            (2 at the default width). Keep narrow on dense hyphal networks
            to avoid sampling adjacent hyphae as background. Typical range:
            1--8. Default: None.
        coherence_window_radius: Radius of the square averaging kernel for
            orientation coherence computation. When ``None``, set to
            ``round(5.0 × min_branch_width_px)`` (15 at the default width).
            Larger radius captures long-range directional consistency;
            reduce for highly curved or heavily branching networks. Typical
            range: 5--50. Default: None.

    Returns:
        Image: Input image with ``objmask`` set to a binary mask of all
        detected fungal pixels and ``objmap`` set to a labelled colony map
        where each fungal colony receives a unique consecutive integer label
        via Voronoi assignment.

    Raises:
        TypeError: If ``inoculum_detector`` is not an ObjectDetector or
            ImagePipeline instance.
        ValueError: If no inoculum centres are detected, or no detected
            centres overlap with the branch structure after filtering.

    References:
        [1] P. Kovesi, "Phase congruency: A low-level image invariant,"
        *Psychol. Res.*, vol. 64, no. 2, pp. 136--148, 2000.

        [2] E. W. Dijkstra, "A note on two problems in connexion with
        graphs," *Numer. Math.*, vol. 1, no. 1, pp. 269--271, 1959.

    See Also:
        :doc:`/tutorials/notebooks/10_detecting_filamentous_fungi`
            Dedicated tutorial for filamentous fungi detection workflows.
        :doc:`/how_to/notebooks/choose_detection_algorithm`
            Guide for selecting the right detector for your plate images.
        :doc:`/explanation/filamentous_fungi_algorithm`
            Algorithm details for the two-stage detection and Voronoi
            partition approach.
        :doc:`/explanation/detection_strategies_compared`
            Comparison of all detection strategies and their failure modes.
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
    # TODO: review bound (unverified vs literature)
    max_colony_radius_px: Annotated[float, TuneSpec(50.0, 500.0, log=True)] = 250.0
    # TODO: review bound (unverified vs literature)
    min_branch_width_px: Annotated[int, TuneSpec(2, 10)] = 3

    # ── Explicit user knobs ──
    # Docstrings document each default + its qualitative direction but not an
    # explicit range; the windows below are derived from the default + domain
    # knowledge (search hints only, never validity bounds).
    ignore_borders: bool = True
    # TODO: review bound (unverified vs literature)
    edge_noise_threshold: Annotated[float, TuneSpec(2.0, 12.0)] = 6.0
    # TODO: review bound (unverified vs literature)
    reconnection_tolerance: Annotated[float, TuneSpec(1.0, 5.0)] = 2.5
    # TODO: review bound (unverified vs literature)
    max_gap_length: Annotated[int, TuneSpec(10, 60)] = 30
    # TODO: review bound (unverified vs literature)
    border_margin_px: Annotated[int, TuneSpec(20, 100)] = 50
    # TODO: review bound (unverified vs literature)
    frag_reach_px: Annotated[int, TuneSpec(5, 30)] = 10
    # TODO: review bound (unverified vs literature)
    gap_crossing_penalty: Annotated[float, TuneSpec(1.0, 10.0)] = 4.0
    reconnect_strategy: FilamentousFungiReconnectStrategy = "dijkstra"
    reconnect_scope: Literal["branches", "pseudo"] = "branches"

    # ── Scene-derivation overrides (None → auto-derived by the validator) ──
    # Auto-derived from max_colony_radius_px / min_branch_width_px when left at
    # None, so they are never independent search targets (decision: tunable=False).
    gauss_sigma: Annotated[Optional[float], TuneSpec(tunable=False)] = None
    tile_size: Annotated[Optional[int], TuneSpec(tunable=False)] = None
    tile_overlap: Annotated[Optional[int], TuneSpec(tunable=False)] = None
    pct_min_wavelength: Annotated[Optional[float], TuneSpec(tunable=False)] = None
    mad_window: Annotated[Optional[int], TuneSpec(tunable=False)] = None
    path_dilation_radius: Annotated[Optional[int], TuneSpec(tunable=False)] = None
    snr_margin: Annotated[Optional[int], TuneSpec(tunable=False)] = None
    coherence_window_radius: Annotated[Optional[int], TuneSpec(tunable=False)] = None

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
        if self.tile_size <= self.tile_overlap:
            raise ValueError(
                    "tile_size must be greater than tile_overlap so "
                    "sliding-window tiles advance by at least one pixel"
            )
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
        from phenotypic.sdk_.reconnect import (
            build_reconnect_cost,
            compute_full_image_app2_gi_cost,
            filter_mask_by_overlap,
            markers_from_centroids,
            partition_by_grid_voronoi,
            reconnect_fragments_tiled,
            select_reconnect_fragments,
        )

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
        pct_result = FocusEdgePhase(
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
        inoculum_structure_mask = filter_mask_by_overlap(
                overall_objmask, inoculum_objmask,
        )
        overlap_objmap = label(inoculum_structure_mask)

        if overlap_objmap.max() == 0:
            raise ValueError(
                    "No centers overlap with detected branch structure after "
                    "filtering. Check that inoculum_detector picks up the same "
                    "objects captured by the dual-mask branch detection."
            )

        self._log_memory_usage("after overlap filtering")

        centroid_markers = markers_from_centroids(inoculum_img.objmap[:])

        inoculum_structure_map = partition_by_grid_voronoi(centroid_markers,
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

        central_mask, fragment_labels = select_reconnect_fragments(
                colony_labels, inoculum_objmask, overall_objmask, inoculum_structure_mask,
                scope=self.reconnect_scope,
                min_fragment_size=max(1, self.min_branch_width_px),
        )

        app2_gi_cost = None
        if self.reconnect_strategy == "app2_gwdt":
            app2_gi_cost = compute_full_image_app2_gi_cost(
                    enhanced_arr,
                    background=~overall_objmask.astype(np.bool_, copy=False),
            )

        unmasked_cost, cost_surface = build_reconnect_cost(
                pct_result.pc_sum,
                pct_result.M,
                pct_result.m,
                pct_result.orientation,
                enhanced_arr,
                colony_labels,
                central_mask,
                self._reconnect_config(),
        )

        colony_labels = reconnect_fragments_tiled(
                colony_labels,
                fragment_labels,
                cost_surface,
                unmasked_cost,
                pct_result.pc_sum.astype(np.float32),
                enhanced_gray,
                self._reconnect_config(),
                app2_gi_cost=app2_gi_cost,
        )

        self._log_memory_usage(
                "after Dijkstra reconnection",
                include_process=True,
                include_tracemalloc=True,
        )

        # ── PHASE 5: FINAL VORONOI ────────────────────────────────────
        final_mask = (colony_labels > 0) | inoculum_structure_mask
        colony_labels = partition_by_grid_voronoi(centroid_markers, final_mask)

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

    def _reconnect_config(self) -> ReconnectConfig:
        """Bundle scene-derived scalars for the sdk_.reconnect functions."""
        from phenotypic.sdk_.reconnect import ReconnectConfig

        return ReconnectConfig(
            beta=self.beta,
            gamma=self.gamma,
            delta=self.delta,
            coherence_window_radius=self.coherence_window_radius,
            mad_window=self.mad_window,
            gap_crossing_penalty=self.gap_crossing_penalty,
            border_margin_px=self.border_margin_px,
            frag_reach_px=self.frag_reach_px,
            tile_size=self.tile_size,
            tile_overlap=self.tile_overlap,
            max_gap_length=self.max_gap_length,
            path_dilation_radius=self.path_dilation_radius,
            snr_margin=self.snr_margin,
            reconnection_tolerance=self.reconnection_tolerance,
        )
