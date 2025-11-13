from typing import Literal, Optional

from phenotypic.abc_ import PrefabPipeline
from phenotypic import ImagePipeline
from phenotypic.enhance import CLAHE, GaussianBlur, MedianEnhancer, ContrastStretching
from phenotypic.detect import OtsuDetector
from phenotypic.grid import GridApply, MinResidualErrorReducer, GridAlignmentOutlierRemover
from phenotypic.refine import BorderObjectRemover, SmallObjectRemover, LowCircularityRemover
from phenotypic.correction import GridAligner

from phenotypic.measure import MeasureColor, MeasureShape, MeasureIntensity, MeasureTexture


class GridSectionPipeline(PrefabPipeline):
    """
    Provides an image processing pipeline designed for grid-based section-level analysis.

    This class defines a sequence of operations and measurements optimized for processing
    gridded images where each section is analyzed independently. The pipeline includes
    multiple stages of preprocessing, detection, filtering, and alignment steps, followed
    by section-level detection and final measurements.

    Attributes:
        gaussian_sigma (int): Standard deviation for Gaussian kernel in initial smoothing.
        gaussian_mode (str): Mode for handling image boundaries during Gaussian smoothing.
        gaussian_truncate (float): Truncate the Gaussian kernel at this many standard deviations.
        clahe_kernel_size (int | None): Size of kernel for CLAHE. If None, automatically calculated.
        clahe_clip_limit (float): Contrast limit for CLAHE.
        median_mode (str): Boundary mode for median filter.
        median_cval (float): Constant value for median filter when mode is 'constant'.
        otsu_ignore_zeros (bool): Whether to ignore zero pixels in Otsu thresholding.
        otsu_ignore_borders (bool): Whether to ignore border objects in Otsu detection.
        border_remover_size (int | float | None): Size of border region where objects are removed.
        circularity_cutoff (float): Minimum circularity threshold for objects to be retained.
        small_object_min_size (int): Minimum size of objects to retain in first removal step.
        outlier_axis (Optional[int]): Axis for outlier analysis. None for both, 0 for rows, 1 for columns.
        outlier_stddev_multiplier (float): Multiplier for standard deviation in outlier detection.
        outlier_max_coeff_variance (int): Maximum coefficient of variance for outlier analysis.
        aligner_axis (int): Axis for grid alignment (0 for rows, 1 for columns).
        aligner_mode (str): Mode for grid alignment rotation.
        section_blur_sigma (int): Standard deviation for Gaussian kernel in section-level detection.
        section_blur_mode (str): Mode for Gaussian smoothing in section-level detection.
        section_blur_truncate (float): Truncate for Gaussian kernel in section-level detection.
        section_median_mode (str): Boundary mode for median filter in section-level detection.
        section_median_cval (float): Constant value for median filter in section-level detection.
        section_contrast_lower_percentile (int): Lower percentile for contrast stretching in sections.
        section_contrast_upper_percentile (int): Upper percentile for contrast stretching in sections.
        section_otsu_ignore_zeros (bool): Whether to ignore zeros in section-level Otsu detection.
        section_otsu_ignore_borders (bool): Whether to ignore borders in section-level Otsu detection.
        grid_apply_reset_enh_matrix (bool): Whether to reset enh_gray before applying section-level pipeline.
        small_object_min_size_2 (int): Minimum size of objects to retain in second removal step.
        color_white_chroma_max (float): Maximum white chroma value for color measurement.
        color_chroma_min (float): Minimum chroma value for color measurement.
        color_include_XYZ (bool): Whether to include XYZ color space measurements.
        texture_scale (int | list[int]): Scale parameter(s) for Haralick texture features.
        texture_quant_lvl (Literal[8, 16, 32, 64]): Quantization level for texture computation.
        texture_enhance (bool): Whether to enhance image before texture measurement.
        texture_warn (bool): Whether to warn on texture computation errors.
        benchmark (bool): Indicates whether benchmarking is enabled across the pipeline.
    """

    def __init__(self,
                 gaussian_sigma: int = 10,
                 gaussian_mode: str = 'reflect',
                 gaussian_truncate: float = 4.0,
                 clahe_kernel_size: int | None = None,
                 clahe_clip_limit: float = 0.01,
                 median_mode: str = 'nearest',
                 median_cval: float = 0.0,
                 otsu_ignore_zeros: bool = True,
                 otsu_ignore_borders: bool = True,
                 border_remover_size: int | float | None = 50,
                 circularity_cutoff: float = 0.6,
                 small_object_min_size: int = 100,
                 outlier_axis: Optional[int] = None,
                 outlier_stddev_multiplier: float = 1.5,
                 outlier_max_coeff_variance: int = 1,
                 aligner_axis: int = 0,
                 aligner_mode: str = 'edge',
                 section_blur_sigma: int = 5,
                 section_blur_mode: str = 'reflect',
                 section_blur_truncate: float = 4.0,
                 section_median_mode: str = 'nearest',
                 section_median_cval: float = 0.0,
                 section_contrast_lower_percentile: int = 2,
                 section_contrast_upper_percentile: int = 98,
                 section_otsu_ignore_zeros: bool = True,
                 section_otsu_ignore_borders: bool = True,
                 grid_apply_reset_enh_matrix: bool = True,
                 small_object_min_size_2: int = 100,
                 color_white_chroma_max: float = 4.0,
                 color_chroma_min: float = 8.0,
                 color_include_XYZ: bool = False,
                 texture_scale: int | list[int] = 5,
                 texture_quant_lvl: Literal[8, 16, 32, 64] = 32,
                 texture_enhance: bool = False,
                 texture_warn: bool = False,
                 benchmark: bool = False,
                 **kwargs):
        """
        Initializes the GridSectionPipeline with customizable operations and measurements.

        Args:
            gaussian_sigma (int): Standard deviation for Gaussian kernel in initial smoothing.
            gaussian_mode (str): Mode for handling image boundaries during Gaussian smoothing.
            gaussian_truncate (float): Truncate the Gaussian kernel at this many standard deviations.
            clahe_kernel_size (int | None): Size of kernel for CLAHE. If None, automatically calculated.
            clahe_clip_limit (float): Contrast limit for CLAHE.
            median_mode (str): Boundary mode for median filter.
            median_cval (float): Constant value for median filter when mode is 'constant'.
            otsu_ignore_zeros (bool): Whether to ignore zero pixels in Otsu thresholding.
            otsu_ignore_borders (bool): Whether to ignore border objects in Otsu detection.
            border_remover_size (int | float | None): Size of border region where objects are removed.
            circularity_cutoff (float): Minimum circularity threshold for objects to be retained.
            small_object_min_size (int): Minimum size of objects to retain in first removal step.
            outlier_axis (Optional[int]): Axis for outlier analysis. None for both, 0 for rows, 1 for columns.
            outlier_stddev_multiplier (float): Multiplier for standard deviation in outlier detection.
            outlier_max_coeff_variance (int): Maximum coefficient of variance for outlier analysis.
            aligner_axis (int): Axis for grid alignment (0 for rows, 1 for columns).
            aligner_mode (str): Mode for grid alignment rotation.
            section_blur_sigma (int): Standard deviation for Gaussian kernel in section-level detection.
            section_blur_mode (str): Mode for Gaussian smoothing in section-level detection.
            section_blur_truncate (float): Truncate for Gaussian kernel in section-level detection.
            section_median_mode (str): Boundary mode for median filter in section-level detection.
            section_median_cval (float): Constant value for median filter in section-level detection.
            section_contrast_lower_percentile (int): Lower percentile for contrast stretching in sections.
            section_contrast_upper_percentile (int): Upper percentile for contrast stretching in sections.
            section_otsu_ignore_zeros (bool): Whether to ignore zeros in section-level Otsu detection.
            section_otsu_ignore_borders (bool): Whether to ignore borders in section-level Otsu detection.
            grid_apply_reset_enh_matrix (bool): Whether to reset enh_gray before applying section-level pipeline.
            small_object_min_size_2 (int): Minimum size of objects to retain in second removal step.
            color_white_chroma_max (float): Maximum white chroma value for color measurement.
            color_chroma_min (float): Minimum chroma value for color measurement.
            color_include_XYZ (bool): Whether to include XYZ color space measurements.
            texture_scale (int | list[int]): Scale parameter(s) for Haralick texture features.
            texture_quant_lvl (Literal[8, 16, 32, 64]): Quantization level for texture computation.
            texture_enhance (bool): Whether to enhance image before texture measurement.
            texture_warn (bool): Whether to warn on texture computation errors.
            benchmark (bool): Indicates whether benchmarking is enabled across the pipeline.
        """
        ops = {
            'blur'                            : GaussianBlur(sigma=gaussian_sigma, mode=gaussian_mode,
                                                             truncate=gaussian_truncate),
            'clahe'                           : CLAHE(kernel_size=clahe_kernel_size, clip_limit=clahe_clip_limit),
            'median filter'                   : MedianEnhancer(mode=median_mode, cval=median_cval),
            'detection'                       : OtsuDetector(ignore_zeros=otsu_ignore_zeros,
                                                             ignore_borders=otsu_ignore_borders),
            'border_removal'                  : BorderObjectRemover(border_size=border_remover_size),
            'low circularity remover'         : LowCircularityRemover(cutoff=circularity_cutoff),
            'small object remover'            : SmallObjectRemover(min_size=small_object_min_size),
            'Reduce by section residual error': MinResidualErrorReducer(),
            'outlier removal'                 : GridAlignmentOutlierRemover(
                    axis=outlier_axis,
                    stddev_multiplier=outlier_stddev_multiplier,
                    max_coeff_variance=outlier_max_coeff_variance
            ),
            'align'                           : GridAligner(axis=aligner_axis, mode=aligner_mode),
            'section-level detect'            : GridApply(
                    ImagePipeline({
                        'blur'               : GaussianBlur(
                                sigma=section_blur_sigma,
                                mode=section_blur_mode,
                                truncate=section_blur_truncate
                        ),
                        'median filter'      : MedianEnhancer(mode=section_median_mode, cval=section_median_cval),
                        'contrast stretching': ContrastStretching(
                                lower_percentile=section_contrast_lower_percentile,
                                upper_percentile=section_contrast_upper_percentile
                        ),
                        'detection'          : OtsuDetector(
                                ignore_zeros=section_otsu_ignore_zeros,
                                ignore_borders=section_otsu_ignore_borders
                        ),
                    }),
                    reset_enh_matrix=grid_apply_reset_enh_matrix
            ),
            'small object remover 2'          : SmallObjectRemover(min_size=small_object_min_size_2),
            'grid_reduction'                  : MinResidualErrorReducer()
        }
        meas = {
            'MeasureColor'    : MeasureColor(
                    white_chroma_max=color_white_chroma_max,
                    chroma_min=color_chroma_min,
                    include_XYZ=color_include_XYZ
            ),
            'MeasureShape'    : MeasureShape(),
            'MeasureIntensity': MeasureIntensity(),
            'MeasureTexture'  : MeasureTexture(
                    scale=texture_scale,
                    quant_lvl=texture_quant_lvl,
                    enhance=texture_enhance,
                    warn=texture_warn
            )
        }
        super().__init__(ops=ops, meas=meas, benchmark=benchmark, **kwargs)
