from __future__ import annotations

from typing import List, Literal

from phenotypic.abc_ import PrefabPipeline
from phenotypic.enhance import GaussianBlur
from phenotypic.detect import RoundPeaksDetector
from phenotypic.measure import (
    MeasureShape,
    MeasureIntensity,
    MeasureTexture,
    MeasureColor,
)


class RoundPeaksPipeline(PrefabPipeline):
    """Detect and measure round colonies using lightweight peak-based detection.

    A fast, minimal pipeline that applies Gaussian smoothing and grid-aware
    peak detection for circular colonies. Fewer stages and parameters than
    the Heavy variants, making it the fastest prefab option.

    Steps:
        1. GaussianBlur — smooth noise
        2. RoundPeaksDetector — grid-aware circular colony detection

    Measurements: MeasureShape, MeasureIntensity, MeasureTexture, MeasureColor.

    Best For:
        - Well-separated round colonies on grid plates.
        - High-throughput screening where speed matters.
        - Plates with consistent colony sizes and regular spacing.
        - Quick prototyping before switching to a heavier pipeline.

    Consider Also:
        - :class:`HeavyRoundPeaksPipeline` when additional refinement stages
          are needed for cleaner results.
        - :class:`HeavyOtsuPipeline` for general-purpose detection with more
          robust preprocessing.
        - :class:`FilamentousFungiPipeline` for filamentous organisms.

    See Also:
        :doc:`/tutorials/notebooks/08_using_prefab_pipelines` for a visual
        comparison of prefab pipelines.
        :doc:`/explanation/prefab_pipelines_guide` for guidance on choosing
        a prefab pipeline.

    Args:
        blur_sigma: Gaussian blur sigma. Typical range: 1--5. Default: 5.
        blur_mode: Boundary handling (``'reflect'``, ``'constant'``,
            ``'nearest'``). Default: ``'reflect'``.
        blur_cval: Fill value when ``blur_mode='constant'``. Default: 0.0.
        blur_truncate: Kernel extent in standard deviations. Default: 4.0.
        detector_thresh_method: Thresholding method (``'otsu'``, ``'mean'``,
            ``'local'``, ``'triangle'``, ``'minimum'``, ``'isodata'``).
            Default: ``'otsu'``.
        detector_subtract_background: Normalize background before
            thresholding. Default: ``True``.
        detector_remove_noise: Morphological opening to remove specks.
            Default: ``True``.
        detector_footprint_radius: Radius for morphological operations.
            Default: 5.
        detector_smoothing_sigma: Sigma for grid profile smoothing.
            Default: 2.0.
        detector_min_peak_distance: Minimum grid line spacing. ``None``
            auto-estimates. Default: ``None``.
        detector_peak_prominence: Minimum peak prominence. ``None``
            auto-estimates. Default: ``None``.
        detector_edge_refinement: Refine grid edges using local profiles.
            Default: ``True``.
        texture_scale: Scale(s) for Haralick texture features. Default: 5.
        texture_quant_lvl: Quantization level (8, 16, 32, 64).
            Default: 32.
        texture_enhance: Enhance contrast before texture measurement.
            Default: ``False``.
        texture_warn: Warn on unreliable texture measurements.
            Default: ``False``.
        benchmark: Enable per-step timing. Default: ``False``.
        verbose: Enable verbose logging. Default: ``False``.
    """

    def __init__(
            self,
            *,
            blur_sigma: int = 5,
            blur_mode: str = "reflect",
            blur_cval: float = 0.0,
            blur_truncate: float = 4.0,
            detector_thresh_method: Literal[
                "otsu",
                "mean",
                "local",
                "triangle",
                "minimum",
                "isodata",
            ] = "otsu",
            detector_subtract_background: bool = True,
            detector_remove_noise: bool = True,
            detector_footprint_radius: int = 5,
            detector_smoothing_sigma: float = 2.0,
            detector_min_peak_distance: int | None = None,
            detector_peak_prominence: float | None = None,
            detector_edge_refinement: bool = True,
            texture_scale: int | List[int] = 5,
            texture_quant_lvl: Literal[8, 16, 32, 64] = 32,
            texture_enhance: bool = False,
            texture_warn: bool = False,
            benchmark: bool = False,
            verbose: bool = False,
    ) -> None:
        gaussian = GaussianBlur(
                sigma=blur_sigma,
                mode=blur_mode,
                cval=blur_cval,
                truncate=blur_truncate,
        )

        detector = RoundPeaksDetector(
                thresh_method=detector_thresh_method,
                subtract_background=detector_subtract_background,
                remove_noise=detector_remove_noise,
                footprint_width=detector_footprint_radius,
                smoothing_sigma=detector_smoothing_sigma,
                min_peak_distance=detector_min_peak_distance,
                peak_prominence=detector_peak_prominence,
                edge_refinement=detector_edge_refinement,
        )

        texture_meas = MeasureTexture(
                scale=texture_scale,
                quant_lvl=texture_quant_lvl,
                enhance=texture_enhance,
                warn=texture_warn,
        )

        ops = [gaussian, detector]
        meas = [
            MeasureShape(),
            MeasureIntensity(),
            texture_meas,
            MeasureColor(),
        ]

        super().__init__(ops=ops, meas=meas, benchmark=benchmark, verbose=verbose)


__all__ = ("RoundPeaksPipeline",)
