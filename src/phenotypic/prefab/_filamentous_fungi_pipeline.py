from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Union

from phenotypic import ImagePipeline
from phenotypic.abc_ import PrefabPipeline, ObjectDetector
from phenotypic.correction import StableDenoise
from phenotypic.detect import FilamentousFungiDetector
from phenotypic.enhance import HomomorphicFilter
from phenotypic.detect._inoculum_detector import InoculumDetector
from phenotypic.measure import (
    MeasureGridSpatial,
    MeasureIntensity,
    MeasureShape,
    MeasureTexture,
)
from phenotypic.refine import GridSectionLargest

if TYPE_CHECKING:
    pass


class FilamentousFungiPipeline(PrefabPipeline):
    """Ready-to-use pipeline for filamentous fungi detection with StableDenoise denoising and spatial measurements.

    Pipeline Steps:
        1. ``StableDenoise`` -- Variance-stabilized BM3D denoising for Poisson-Gaussian
           noise removal on gray and detect_mat channels.
        2. ``HomomorphicFilter`` -- Illumination normalization via
           frequency-domain filtering on detect_mat.
        3. ``FilamentousFungiDetector`` -- Two-stage detection (inoculum +
           dual-mask reconnection) with Euclidean Voronoi partition and
           Dijkstra branch reconnection (enabled by default).

    Measurements:
        - ``MeasureGridSpatial`` -- Grid-level spatial statistics.
        - ``MeasureShape`` -- Per-colony shape descriptors.
        - ``MeasureIntensity`` -- Per-colony intensity statistics.
        - ``MeasureTexture`` -- Haralick texture features.

    Args:
        bm3d_block_size: BM3D patch size for denoising. Default 8.
        bm3d_stage_arg: BM3D processing mode. ``'all_stages'`` gives best
            quality; ``'hard_thresholding'`` is faster.
        homo_sigma: Gaussian cutoff sigma for the homomorphic filter.
        homo_gamma_low: Gain for low-frequency (illumination) components.
        homo_gamma_high: Gain for high-frequency (reflectance) components.
        inoculum_min_diameter: Smallest expected inoculum diameter in pixels
            for the default InoculumDetector. Ignored when ``inoculum_detector``
            is provided. Default 30.0.
        inoculum_max_diameter: Largest expected inoculum diameter in pixels
            for the default InoculumDetector. Ignored when ``inoculum_detector``
            is provided. Default 100.0.
        inoculum_detector: Custom ObjectDetector or ImagePipeline that
            identifies fungal centers/nuclei. When None, builds a default
            pipeline of ``InoculumDetector`` + ``GridSectionLargest``.
        overall_detector: ObjectDetector or ImagePipeline that captures complete
            fungal structures including hyphae. Ignored when
            ``enable_reconnection=True`` (the default).
        enable_reconnection: When True (default), use dual-mask branch
            detection and Dijkstra-based reconnection instead of the legacy
            ``overall_detector`` path.
        pct_n_orient: Number of orientations for phase congruency computation.
        pct_min_wavelength: Minimum wavelength for log-Gabor filters.
        pct_k: Noise threshold scaling factor for phase congruency.
        gauss_sigma: Sigma for SubtractGaussian background subtraction.
        gauss_n_iter: Number of SubtractGaussian iterations.
        morph_width: Disk radius for morphological open/close operations on
            branch masks.
        beta: Exponent on anisotropy in the composite cost formula.
        gamma: Weight of MAD penalty in the composite cost numerator.
        r_coherence: Radius for orientation coherence computation.
        mad_window: Window size for local MAD computation (must be odd).
        r_screen: Screening radius for fragment pre-screening.
        delta: Dijkstra radial penalty factor for retreating steps.
        quality_k: IQR multiplier for path quality threshold calibration.
        window_cost: Sliding window size in pixels for the windowed cost metric.
        edge_margin: Border penalty width in pixels.
        gap_penalty_alpha: Distance-gap penalty strength.
        snr_margin: Extra radius beyond ``path_dilation_radius`` for the SNR
            background ring.
        path_dilation_radius: Disk radius for dilating reconnection paths.
        tile_size: Side length of square tiles for tiled Dijkstra processing.
        tile_overlap: Overlap in pixels between adjacent tiles.
        texture_scale: Scale parameter for Haralick texture features.
        texture_warn: Whether to warn on texture computation errors.
        benchmark: Enable per-step timing and memory benchmarks.
        verbose: Enable verbose logging during pipeline execution.

    See Also:
        :doc:`/tutorials/notebooks/10_detecting_filamentous_fungi` for a
        visual walkthrough of filamentous fungi detection.
        :doc:`/explanation/filamentous_fungi_algorithm` for the theory
        behind the reconnection algorithm.
        :doc:`/explanation/prefab_pipelines_guide` for guidance on choosing
        a prefab pipeline.
    """

    def __init__(
            self,
            bm3d_block_size: int = 4,
            bm3d_stage_arg: Literal["all_stages", "hard_thresholding"] = "all_stages",
            homo_sigma: float = 300.0,
            homo_gamma_low: float = 0.5,
            homo_gamma_high: float = 1.5,
            inoculum_min_diameter: float = 30.0,
            inoculum_max_diameter: float = 100.0,
            inoculum_detector: Union[ObjectDetector, ImagePipeline, None] = None,
            overall_detector: Union[ObjectDetector, ImagePipeline, None] = None,
            ignore_borders: bool = True,
            enable_reconnection: bool = True,
            pct_n_orient: int = 8,
            pct_min_wavelength: float = 5.0,
            pct_k: float = 6.0,
            gauss_sigma: float = 300.0,
            gauss_n_iter: int = 2,
            morph_width: int = 5,
            beta: float = 2.0,
            gamma: float = 1.2,
            r_coherence: int = 12,
            mad_window: int = 7,
            r_screen: int = 10,
            delta: float = 1.0,
            quality_k: float = 2.5,
            window_cost: int = 30,
            edge_margin: int = 50,
            gap_penalty_alpha: float = 4.0,
            snr_margin: int = 3,
            path_dilation_radius: int = 2,
            tile_size: int = 1200,
            tile_overlap: int = 100,
            texture_scale: int = 5,
            texture_warn: bool = False,
            benchmark: bool = False,
            verbose: bool = False,
    ) -> None:
        if inoculum_detector is None:
            inoculum_detector = ImagePipeline(
                    ops=[
                        InoculumDetector(
                                min_diameter=inoculum_min_diameter,
                                max_diameter=inoculum_max_diameter,
                        ),
                        GridSectionLargest(),
                    ]
            )

        ops = [
            StableDenoise(
                    block_size=bm3d_block_size,
                    stage_arg=bm3d_stage_arg,
            ),
            HomomorphicFilter(
                    sigma=homo_sigma,
                    gamma_low=homo_gamma_low,
                    gamma_high=homo_gamma_high,
            ),
            FilamentousFungiDetector(
                    inoculum_detector=inoculum_detector,
                    overall_detector=overall_detector,
                    ignore_borders=ignore_borders,
                    enable_reconnection=enable_reconnection,
                    pct_n_orient=pct_n_orient,
                    pct_min_wavelength=pct_min_wavelength,
                    pct_k=pct_k,
                    gauss_sigma=gauss_sigma,
                    gauss_n_iter=gauss_n_iter,
                    morph_width=morph_width,
                    beta=beta,
                    gamma=gamma,
                    r_coherence=r_coherence,
                    mad_window=mad_window,
                    r_screen=r_screen,
                    delta=delta,
                    quality_k=quality_k,
                    window_cost=window_cost,
                    edge_margin=edge_margin,
                    gap_penalty_alpha=gap_penalty_alpha,
                    snr_margin=snr_margin,
                    path_dilation_radius=path_dilation_radius,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
            ),
        ]

        meas = [
            MeasureGridSpatial(),
            MeasureShape(),
            MeasureIntensity(),
            MeasureTexture(scale=texture_scale, warn=texture_warn),
        ]

        super().__init__(ops=ops, meas=meas, benchmark=benchmark, verbose=verbose)


__all__ = ("FilamentousFungiPipeline",)
