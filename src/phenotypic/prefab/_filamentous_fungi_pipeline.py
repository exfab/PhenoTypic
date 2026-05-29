from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Union

from phenotypic import ImagePipeline
from phenotypic.abc_ import PrefabPipeline, ObjectDetector
from phenotypic.correction import StableDenoise
from phenotypic.detect import FilamentousFungiDetector
from phenotypic.enhance import FlattenIllumination
from phenotypic.detect._inoculum_detector import InoculumDetector
from phenotypic.measure import (
    MeasureGridSpatial,
    MeasureIntensity,
    MeasureShape,
    MeasureTexture,
)
from phenotypic.refine import KeepSectionLargest

if TYPE_CHECKING:
    pass


class FilamentousFungiPipeline(PrefabPipeline):
    """Ready-to-use pipeline for filamentous fungi detection with StableDenoise denoising and spatial measurements.

    Pipeline Steps:
        1. ``StableDenoise`` -- Variance-stabilized BM3D denoising for Poisson-Gaussian
           noise removal on gray and detect_mat channels.
        2. ``FlattenIllumination`` -- Illumination normalization via
           frequency-domain filtering on detect_mat.
        3. ``FilamentousFungiDetector`` -- Two-stage detection (inoculum +
           dual-mask reconnection) with Euclidean Voronoi partition and
           Dijkstra branch reconnection.

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
            pipeline of ``InoculumDetector`` + ``KeepSectionLargest``.
        max_colony_radius_px: Largest colony radius (in pixels) the
            detector should handle. Sizes scene-derived spatial
            parameters for this worst case. Default 250.
        min_branch_width_px: Narrowest hyphal branch width (in pixels) to
            detect. Sizes signal-scale parameters. Default 3.
        ignore_borders: If True, drops objects touching the image border.
        edge_noise_threshold: Noise threshold scaling factor for phase
            congruency edge detection.
        reconnection_tolerance: IQR multiplier for path quality threshold
            calibration (higher = more permissive).
        max_gap_length: Maximum acceptable length (pixels) of a
            suspicious cost stretch along a reconnection path.
        border_margin_px: Border penalty buffer width in pixels.
        frag_reach_px: Maximum 2D distance (pixels) from a fragment's
            boundary to the nearest routable pixel; fragments more
            isolated than this are dropped before Dijkstra routing.
        gap_crossing_penalty: Distance-gap penalty strength during
            Dijkstra routing.
        gauss_sigma: Override for SubtractGaussian sigma; None auto-derives.
        pct_min_wavelength: Override for log-Gabor minimum wavelength;
            None auto-derives.
        coherence_window_radius: Override for orientation coherence radius;
            None auto-derives.
        mad_window: Override for local MAD window (odd); None auto-derives.
        snr_margin: Override for SNR ring radius; None auto-derives.
        path_dilation_radius: Override for path dilation radius;
            None auto-derives.
        tile_size: Override for tile side length; None auto-derives.
        tile_overlap: Override for tile overlap; None auto-derives.
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
            # Scene parameters
            max_colony_radius_px: float = 250.0,
            min_branch_width_px: int = 3,
            # Explicit user knobs
            ignore_borders: bool = True,
            edge_noise_threshold: float = 6.0,
            reconnection_tolerance: float = 2.5,
            max_gap_length: int = 30,
            border_margin_px: int = 50,
            frag_reach_px: int = 10,
            gap_crossing_penalty: float = 4.0,
            # Scene-derivation overrides (None → auto-derived)
            gauss_sigma: Optional[float] = None,
            pct_min_wavelength: Optional[float] = None,
            coherence_window_radius: Optional[int] = None,
            mad_window: Optional[int] = None,
            snr_margin: Optional[int] = None,
            path_dilation_radius: Optional[int] = None,
            tile_size: Optional[int] = None,
            tile_overlap: Optional[int] = None,
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
                        KeepSectionLargest(),
                    ]
            )

        ops = [
            StableDenoise(
                    block_size=bm3d_block_size,
                    stage_arg=bm3d_stage_arg,
            ),
            FlattenIllumination(
                    sigma=homo_sigma,
                    gamma_low=homo_gamma_low,
                    gamma_high=homo_gamma_high,
            ),
            FilamentousFungiDetector(
                    inoculum_detector=inoculum_detector,
                    max_colony_radius_px=max_colony_radius_px,
                    min_branch_width_px=min_branch_width_px,
                    ignore_borders=ignore_borders,
                    edge_noise_threshold=edge_noise_threshold,
                    reconnection_tolerance=reconnection_tolerance,
                    max_gap_length=max_gap_length,
                    border_margin_px=border_margin_px,
                    frag_reach_px=frag_reach_px,
                    gap_crossing_penalty=gap_crossing_penalty,
                    gauss_sigma=gauss_sigma,
                    pct_min_wavelength=pct_min_wavelength,
                    coherence_window_radius=coherence_window_radius,
                    mad_window=mad_window,
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
