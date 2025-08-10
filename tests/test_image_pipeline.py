from phenotypic import ImagePipeline
from phenotypic.enhancement import CLAHE, GaussianSmoother, MedianEnhancer, ContrastStretching
from phenotypic.detection import OtsuDetector, WatershedDetector
from phenotypic.grid import GridApply, MinResidualErrorReducer, GridAlignmentOutlierRemover
from phenotypic.objects import BorderObjectRemover, SmallObjectRemover, LowCircularityRemover
from phenotypic.measure import MeasureColor, MeasureShape, MeasureIntensity, MeasureTexture
from phenotypic.morphology import MaskFill
from phenotypic.correction import GridAligner


from phenotypic import GridImage
from phenotypic.data import load_plate_12hr
from .test_fixtures import plate_grid_images
from .resources.TestHelper import timeit

import logging

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
def test_kmarx_pipeline(plate_grid_images):
    kmarx_pipeline = ImagePipeline(
        ops={
            'blur': GaussianSmoother(sigma=5),
            'clahe': CLAHE(),
            'median filter': MedianEnhancer(),
            'detection': WatershedDetector(footprint='auto', min_size=50, relabel=True),
            'border remover': BorderObjectRemover(5),
            'mask_fill': MaskFill(),
            'low circularity remover': LowCircularityRemover(0.5),
            'reduce by section residual error': MinResidualErrorReducer(),
            'outlier removal': GridAlignmentOutlierRemover(),
            'align': GridAligner(),
            'grid_reduction': MinResidualErrorReducer(),
        },
        measurements={
            'MeasureColor': MeasureColor(),
            'MeasureShape': MeasureShape(),
            'MeasureIntensity': MeasureIntensity(),
            'MeasureTexture': MeasureTexture(scale=3),
            'MeasureTexture2': MeasureTexture(scale=4),
        }
    )
    output = kmarx_pipeline.apply(plate_grid_images)
    output = kmarx_pipeline.measure(output)
    assert output is not None

@timeit
def test_kmarx_pipeline_pickleable(plate_grid_images):
    import pickle
    kmarx_pipeline = ImagePipeline(
        {
            'blur': GaussianSmoother(sigma=2),
            'clahe': CLAHE(),
            'median filter': MedianEnhancer(),
            'detection': OtsuDetector(),
            'border_removal': BorderObjectRemover(50),
            'low circularity remover': LowCircularityRemover(0.6),
            'small object remover': SmallObjectRemover(100),
            'Reduce by section residual error': MinResidualErrorReducer(),
            'outlier removal': GridAlignmentOutlierRemover(),
            'align': GridAligner(),
            'section-level detect': GridApply(ImagePipeline({
                'blur': GaussianSmoother(sigma=5),
                'median filter': MedianEnhancer(),
                'contrast stretching': ContrastStretching(),
                'detection': OtsuDetector(),
            }
            )
            ),
            'small object remover 2': SmallObjectRemover(100),
            'grid_reduction': MinResidualErrorReducer()
        }
    )
    pickle.dumps(kmarx_pipeline.apply_and_measure)

@timeit
def test_watershed_kmarx_pipeline_pickleable(plate_grid_images):
    import pickle
    kmarx_pipeline = ImagePipeline(
        ops={
            'blur': GaussianSmoother(sigma=5),
            'clahe': CLAHE(),
            'median filter': MedianEnhancer(),
            'detection': WatershedDetector(footprint='auto', min_size=100, relabel=True),
            'low circularity remover': LowCircularityRemover(0.5),
            'reduce by section residual error': MinResidualErrorReducer(),
            'outlier removal': GridAlignmentOutlierRemover(),
            'align': GridAligner(),
            'grid_reduction': MinResidualErrorReducer(),
        },
        measurements={
            'MeasureColor': MeasureColor(),
            'MeasureShape': MeasureShape(),
            'MeasureIntensity': MeasureIntensity(),
            'MeasureTexture': MeasureTexture()
        }
    )
    pickle.dumps(kmarx_pipeline)

@timeit
def test_watershed_kmarx_pipeline_with_measurements_pickleable(plate_grid_images):
    import pickle
    kmarx_pipeline = ImagePipeline(
        ops={
            'blur': GaussianSmoother(sigma=5),
            'clahe': CLAHE(),
            'median filter': MedianEnhancer(),
            'detection': WatershedDetector(footprint='auto', min_size=100, relabel=True),
            'low circularity remover': LowCircularityRemover(0.5),
            'reduce by section residual error': MinResidualErrorReducer(),
            'outlier removal': GridAlignmentOutlierRemover(),
            'align': GridAligner(),
            'grid_reduction': MinResidualErrorReducer(),
        },
        measurements={
            'MeasureColor': MeasureColor(),
            'MeasureShape': MeasureShape(),
            'MeasureIntensity': MeasureIntensity(),
            'MeasureTexture': MeasureTexture()
        }
    )
    pickle.dumps(kmarx_pipeline)