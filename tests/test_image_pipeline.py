import logging

from phenotypic import GridImage, ImagePipeline
from phenotypic.correction import GridAligner
from phenotypic.data import load_plate_12hr
from phenotypic.detect import OtsuDetector, WatershedDetector
from phenotypic.enhance import CLAHE, ContrastStretching, GaussianBlur, MedianEnhancer
from phenotypic.measure import MeasureColor, MeasureIntensity, MeasureShape, MeasureTexture
from phenotypic.refine import (BorderObjectRemover, LowCircularityRemover, SmallObjectRemover,
                               ResidualOutlierRemover, MinResidualErrorReducer)
from phenotypic.util import GridApply
from .resources.TestHelper import timeit
from .test_fixtures import plate_grid_images

# Configure logging to see all debug information
logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@timeit
def test_empty_pipeline():
    empty_pipeline = ImagePipeline({})
    assert empty_pipeline.apply(GridImage(load_plate_12hr())).num_objects == 0


@timeit
def test_pipeline_on_image(plate_grid_images):
    pipe = ImagePipeline(
            ops={
                'blur'     : GaussianBlur(sigma=5),
                'detection': OtsuDetector(),
                'remove'   : BorderObjectRemover(50),
            },
            meas={
                'MeasureColor'    : MeasureColor(),
                'MeasureShape'    : MeasureShape(),
                'MeasureIntensity': MeasureIntensity(),
                'MeasureTexture'  : MeasureTexture(scale=[3, 4], quant_lvl=8),
            }
    )
    output = pipe.apply(plate_grid_images)
    output = pipe.measure(output)
    assert output is not None

    compound_output = pipe.apply_and_measure(plate_grid_images, reset=True)

    # Compare with better NaN handling and allow for floating point differences
    import pandas as pd
    import numpy as np

    # Check same shape
    assert output.shape == compound_output.shape, f"Different shapes: {output.shape} vs {compound_output.shape}"

    # Check same columns
    assert set(output.columns) == set(compound_output.columns), "Different columns"

    # Exclude columns that are expected to differ (e.g., UUIDs that change between runs)
    cols_to_skip = {'Metadata_ImageName'}  # UUIDs change between pipeline runs

    # For each column, check if values are close (handling NaNs)
    for col in output.columns:
        if col in cols_to_skip:
            continue

        o_series = output[col]
        c_series = compound_output[col]

        # Handle categorical columns
        if isinstance(o_series.dtype, pd.CategoricalDtype):
            # Convert to underlying codes for comparison
            assert np.array_equal(o_series.cat.codes, c_series.cat.codes), \
                f"Column {col} has different categorical values"
        # Check if both are numeric
        elif pd.api.types.is_numeric_dtype(o_series):
            # Use allclose with NaN handling
            assert np.allclose(o_series.values, c_series.values, equal_nan=True, rtol=1e-10, atol=1e-10), \
                f"Column {col} has different values"
        else:
            # For non-numeric, use equals
            assert o_series.equals(c_series), f"Column {col} has different values"


@timeit
def test_kmarx_pipeline_pickleable(plate_grid_images):
    import pickle

    pipe = ImagePipeline(
            {
                'blur'                            : GaussianBlur(sigma=2),
                'clahe'                           : CLAHE(),
                'median filter'                   : MedianEnhancer(),
                'detection'                       : OtsuDetector(),
                'border_removal'                  : BorderObjectRemover(50),
                'low circularity remover'         : LowCircularityRemover(0.6),
                'small object remover'            : SmallObjectRemover(100),
                'Reduce by section residual error': MinResidualErrorReducer(),
                'outlier removal'                 : ResidualOutlierRemover(),
                'align'                           : GridAligner(),
                'section-level detect'            : GridApply(ImagePipeline({
                    'blur'               : GaussianBlur(sigma=5),
                    'median filter'      : MedianEnhancer(),
                    'contrast stretching': ContrastStretching(),
                    'detection'          : OtsuDetector(),
                }
                )
                ),
                'small object remover 2'          : SmallObjectRemover(100),
                'grid_reduction'                  : MinResidualErrorReducer()
            }
    )
    pickle.dumps(pipe.apply_and_measure)


@timeit
def test_watershed_kmarx_pipeline_pickleable(plate_grid_images):
    import pickle

    kmarx_pipeline = ImagePipeline(
            ops={
                'blur'                            : GaussianBlur(sigma=5),
                'clahe'                           : CLAHE(),
                'median filter'                   : MedianEnhancer(),
                'detection'                       : WatershedDetector(footprint='auto', min_size=100, relabel=True),
                'low circularity remover'         : LowCircularityRemover(0.5),
                'reduce by section residual error': MinResidualErrorReducer(),
                'outlier removal'                 : ResidualOutlierRemover(),
                'align'                           : GridAligner(),
                'grid_reduction'                  : MinResidualErrorReducer(),
            },
            meas={
                'MeasureColor'    : MeasureColor(),
                'MeasureShape'    : MeasureShape(),
                'MeasureIntensity': MeasureIntensity(),
                'MeasureTexture'  : MeasureTexture()
            }
    )
    pickle.dumps(kmarx_pipeline)


@timeit
def test_watershed_kmarx_pipeline_with_measurements_pickleable(plate_grid_images):
    import pickle

    kmarx_pipeline = ImagePipeline(
            ops={
                'blur'                            : GaussianBlur(sigma=5),
                'clahe'                           : CLAHE(),
                'median filter'                   : MedianEnhancer(),
                'detection'                       : WatershedDetector(footprint='auto', min_size=100, relabel=True),
                'low circularity remover'         : LowCircularityRemover(0.5),
                'reduce by section residual error': MinResidualErrorReducer(),
                'outlier removal'                 : ResidualOutlierRemover(),
                'align'                           : GridAligner(),
                'grid_reduction'                  : MinResidualErrorReducer(),
            },
            meas={
                'MeasureColor'    : MeasureColor(),
                'MeasureShape'    : MeasureShape(),
                'MeasureIntensity': MeasureIntensity(),
                'MeasureTexture'  : MeasureTexture()
            }
    )
    pickle.dumps(kmarx_pipeline)
