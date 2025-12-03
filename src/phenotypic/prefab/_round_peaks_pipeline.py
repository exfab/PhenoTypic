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
    """Lightweight pipeline for circular colonies on solid media agar.

    This prefab pipeline provides a streamlined sequence of operations tailored for
    imaging pinned or arrayed fungal colonies on solid media agar. It performs a
    gentle Gaussian blur to suppress scanner and agar noise, followed by a
    grid-aware circular colony detector and a compact set of measurement
    modules. Compared with :class:`HeavyRoundPeaksPipeline`, this variant exposes a
    smaller number of stages and parameters but still allows fine control over
    blur strength, thresholding, grid refinement, and texture scale.

    Operations:
        1. ``GaussianBlur``
        2. ``RoundPeaksDetector``

    Measurements:
        - ``MeasureShape``
        - ``MeasureIntensity``
        - ``MeasureTexture``
        - ``MeasureColor``

    Parameters
    ----------
    blur_sigma : int, optional
        Standard deviation (in pixels) of the Gaussian blur kernel. Lower
        values preserve sharp edges and small colonies, which is useful when
        pins produce tight colonies with fine boundaries. Higher values smooth
        away scanner grain and agar micro-texture but can merge neighboring
        colonies or wash out tiny satellite growth.
    blur_mode : {"reflect", "constant", "nearest"}, optional
        Boundary handling strategy for the blur. ``"reflect"`` (default)
        mirrors intensities at the plate edge and avoids artificial halos,
        which is helpful for colonies close to the border of an agar image.
        ``"constant"`` and ``"nearest"`` can be used for cropped regions but
        may introduce rim artifacts that slightly bias edge intensity and
        downstream detection.
    blur_cval : float, optional
        Constant fill value used when ``blur_mode="constant"``. Setting this
        close to the background agar intensity can stabilize blur at cut edges
        of the plate. A value too bright or too dark may create spurious rims
        that either look like faint colonies or mask real colonies near the
        image boundary.
    blur_truncate : float, optional
        Radius of the Gaussian kernel in standard deviations. Increasing this
        slightly widens the effective blur footprint and further smooths broad
        illumination gradients on the plate, at the cost of speed. For typical
        pinned fungal colonies, the default is usually sufficient; very large
        values can over-smooth diffuse halos or ring-like growth.
    detector_thresh_method : {"otsu", "mean", "local", "triangle", "minimum", "isodata"}, optional
        Thresholding method used inside ``RoundPeaksDetector``. ``"otsu"`` works
        well for plates with reasonably uniform agar and clear contrast between
        colonies and background. ``"local"`` can cope with strong gradients or
        condensation streaks but may be slower. ``"mean"``, ``"triangle"``,
        ``"minimum"``, and ``"isodata"`` offer alternative trade-offs and can
        be useful when colonies are very pale or when agar pigmentation varies
        strongly across the plate.
    detector_subtract_background : bool, optional
        Whether to normalize background intensity before thresholding. Enabling
        this (default) helps make colonies more comparable across plates with
        different agar batches or scanner settings. If disabled, very faint
        colonies on bright agar may be missed, but native shading patterns or
        radial nutrient gradients remain more faithful.
    detector_remove_noise : bool, optional
        Whether to remove small specks using a morphological opening before
        grid inference. This is often beneficial for fungal plates with dust,
        bubbles, or condensation droplets, but if colonies are extremely small
        (early time points or slow-growing strains) an aggressive noise removal
        may erase real colonies.
    detector_footprint_radius : int, optional
        Radius in pixels used for morphological operations in the detector.
        Larger values clean up bigger spurious regions and slightly shrink
        detected colonies. For tightly pinned arrays, too large a radius can
        erode narrow colonies or disconnect wispy hyphal fronts, under-
        estimating colony size.
    detector_smoothing_sigma : float, optional
        Standard deviation of the 1D Gaussian smoothing used when estimating
        row/column intensity profiles for grid detection. Increasing this makes
        grid inference more robust to noisy or streaky agar images but may blur
        subtle deviations in grid regularity (e.g., warped pinning patterns) so
        that mispinned colonies are forced into a regular grid.
    detector_min_peak_distance : int or None, optional
        Minimum allowed distance between inferred grid lines (peaks). When set
        explicitly, this constrains the expected spacing between pinned colony
        rows/columns, helping reject spurious peaks caused by glare or edge
        artifacts. If ``None`` (default), the distance is estimated from the
        data, which is more flexible but can be unstable for plates with many
        missing colonies.
    detector_peak_prominence : float or None, optional
        Minimum prominence required for peaks in the row/column profiles.
        Higher values make the detector ignore weak structures such as barely
        growing colonies or diffuse halos, focusing instead on strong, dense
        colonies. Lower values are more sensitive to early growth but can be
        confused by agar texture or illumination bands.
    detector_edge_refinement : bool, optional
        If ``True``, refines estimated grid edges using local intensity
        profiles. This improves alignment of each pinned colony to its grid
        cell, especially when colonies expand asymmetrically or when the plate
        is slightly shifted during imaging. Disabling refinement speeds up
        analysis but may misplace colonies at the edge of their wells.
    texture_scale : int or list[int], optional
        Spatial scale(s) in pixels at which texture is measured by
        ``MeasureTexture``. Smaller values emphasize fine-scale surface
        roughness (e.g., wrinkling or concentric ring patterns on filamentous
        colonies). Larger values summarize broader patterns such as coarse
        colony zoning or radial banding. Using multiple scales (by passing a
        list) increases feature richness but also computation time.
    texture_quant_lvl : {8, 16, 32, 64}, optional
        Intensity quantization level for texture computation. Higher values
        capture subtler differences in pigmentation and surface texture but
        require more data and are more sensitive to noise. For images of fungal
        colonies with smooth agar backgrounds, 32 is a good balance; 8 or 16
        may be preferable for very low-contrast plates.
    texture_enhance : bool, optional
        If ``True``, enhances contrast before measuring texture. This can make
        faint radial structures or sectoring within colonies more detectable,
        which is useful when subtle phenotypes matter. However, enhancement may
        exaggerate scanner artifacts or agar imperfections, so enabling it can
        bias texture metrics on marginal images.
    texture_warn : bool, optional
        If ``True``, emits warnings when texture measurements may be
        unreliable (for example, very small colonies or extreme intensity
        clipping). This can help identify pins where scanner saturation, agar
        contamination, or segmentation artifacts distort morphology.
    benchmark : bool, optional
        If ``True``, records timing information for each pipeline stage. This
        is primarily useful when optimizing throughput for large plate series
        but adds minor overhead.
    verbose : bool, optional
        If ``True``, logs additional information during pipeline execution.
        This can help debug unexpected detection behavior (e.g., missing rows
        of colonies) or confirm that grid inference behaves sensibly on new
        imaging setups.

    Notes
    -----
    This pipeline is intended for relatively clean, well-aligned images of
    pinned or arrayed circular colonies on agar where only a modest amount of
    preprocessing is needed. For plates with severe background gradients,
    strong vignetting, or highly irregular grids, consider using
    :class:`HeavyRoundPeaksPipeline`, which exposes additional refinement and
    alignment stages.
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
            footprint_radius=detector_footprint_radius,
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
