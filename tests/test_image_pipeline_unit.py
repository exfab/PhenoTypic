"""Lightweight unit tests for ImagePipelineCore and ImagePipelineBatch
These tests use stub classes to avoid heavy dependencies and long runtimes.
Run together with the integration tests already present.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from phenotypic import ImagePipeline, Image, ImageSet
from phenotypic.abstract import MeasureFeatures, ObjectDetector
from phenotypic.data import load_plate_12hr, load_plate_72hr
from phenotypic.detection import OtsuDetector
from phenotypic.enhancement import GaussianSmoother
from phenotypic.measure import MeasureShape
from phenotypic.objedit import BorderObjectRemover
from .resources.TestHelper import timeit
from .test_fixtures import temp_hdf5_file


class SumObjects(MeasureFeatures):
    def _operate(self, image: Image) -> pd.DataFrame:
        return pd.DataFrame({'Sum': [image.array[:].sum()]}, index=image.objects.labels2series())


class DetectFull(ObjectDetector):
    def _operate(self, image: Image) -> Image:
        image.objmask[:] = 1
        return image


# ---------------------------------------------------------------------------
# Helper to build ImageSetCore with dummy images
# ---------------------------------------------------------------------------


def _make_imageset(tmp_path: Path):
    images = [
        Image(load_plate_12hr(), name='12hr'),
        Image(load_plate_72hr(), name='72hr'),
    ]
    return ImageSet(
            name="iset",
            src=images,
            outpath=tmp_path,
            overwrite=True)


def _make_dummy_imageset(tmp_path: Path):
    images = [
        Image(np.full(shape=(2, 2, 3), fill_value=1), name='image1'),
        Image(np.full(shape=(3, 3, 3), fill_value=1), name='image2'),
    ]
    return ImageSet(
            name='iset',
            src=images,
            outpath=tmp_path,
            overwrite=True,
    )


# ---------------------------------------------------------------------------
# Tests for ImagePipelineCore
# ---------------------------------------------------------------------------

@timeit
def test_core_apply_and_measure():
    img = Image(load_plate_12hr(), name='12hr')
    pipe = ImagePipeline(ops=[OtsuDetector(), BorderObjectRemover(border_size=1)], meas=[SumObjects()])

    df = pipe.apply_and_measure(img)
    assert not df.empty


# ---------------------------------------------------------------------------
# Tests for ImagePipelineBatch (single worker to keep CI light)
# ---------------------------------------------------------------------------

@timeit
def test_batch_apply_and_measure(temp_hdf5_file):
    imageset = _make_imageset(temp_hdf5_file)
    pipe = ImagePipeline(ops=[GaussianSmoother(), OtsuDetector()], meas=[MeasureShape()], verbose=False,
                         njobs=2)

    df = pipe.apply_and_measure(imageset)
    assert df.empty is False, 'No measurements'

    alt_df = imageset.get_measurement()
    assert df.equals(alt_df), 'ImageSet.get_measurements() is different from results'
