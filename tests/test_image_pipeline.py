from phenotypic.pipeline import ImagePipeline
from phenotypic.enhancement import CLAHE, GaussianSmoother, MedianEnhancer, ContrastStretching
from phenotypic.detection import OtsuDetector
from phenotypic.grid import GridAligner, GridApply, MinResidualErrorReducer, LinRegResidualOutlierRemover
from phenotypic.objects import BorderObjectRemover, SmallObjectRemover, LowCircularityRemover

from phenotypic import GridImage
from phenotypic.data import load_plate_12hr
from .test_fixtures import plate_grid_images


def test_empty_pipeline():
    empty_pipeline = ImagePipeline({})
    assert empty_pipeline.apply(GridImage(load_plate_12hr())).num_objects == 0


def test_kmarx_pipeline(plate_grid_images):
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
            'outlier removal': LinRegResidualOutlierRemover(),
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
    output = kmarx_pipeline.apply(plate_grid_images)
    assert output is not None

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
            'outlier removal': LinRegResidualOutlierRemover(),
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
    pickle.dumps(kmarx_pipeline)