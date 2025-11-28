"""Lightweight unit tests for ImagePipelineCore and ImagePipelineBatch
These tests use stub classes to avoid heavy dependencies and long runtimes.
Run together with the integration tests already present.
"""

from pathlib import Path

import pandas as pd
import pytest

from phenotypic import Image, ImagePipeline
from phenotypic.abc_ import MeasureFeatures, ObjectDetector
from phenotypic.data import load_plate_12hr
from phenotypic.detect import OtsuDetector
from phenotypic.refine import BorderObjectRemover
from .resources.TestHelper import timeit
from .test_fixtures import temp_hdf5_file


class SumObjects(MeasureFeatures):
    def _operate(self, image: Image) -> pd.DataFrame:
        labels = image.objects.labels2series()
        return pd.DataFrame({
            labels.name: labels.values,
            'Sum'      : SumObjects._calculate_sum(array=image.gray[:], objmap=image.objmap[:])
        })


class DetectFull(ObjectDetector):
    def _operate(self, image: Image) -> Image:
        image.objmask[5:10, 5:10] = 1
        return image


# ---------------------------------------------------------------------------
# Helper to build ImageSetCore with dummy images
# ---------------------------------------------------------------------------


def _make_imageset(tmp_path: Path):
    from phenotypic.data import load_synthetic_colony
    from phenotypic import ImageSet

    image1 = load_synthetic_colony(mode='Image')
    image1.name = 'synth1'
    image2 = load_synthetic_colony(mode='Image')
    image2.name = 'synth2'
    images = [image1, image2]
    imset = ImageSet(
            name="iset",
            outpath=tmp_path,
            overwrite=False)
    imset.import_images(images)
    return imset


# ---------------------------------------------------------------------------
# Tests for ImagePipelineCore
# ---------------------------------------------------------------------------

@timeit
def test_core_apply_and_measure():
    img = Image(load_plate_12hr(), name='12hr')
    pipe = ImagePipeline(ops=[DetectFull()],
                         meas=[SumObjects()], )

    df = pipe.apply_and_measure(img)
    assert not df.empty


# ---------------------------------------------------------------------------
# Tests for ImagePipelineBatch (single worker to keep CI light)
# ---------------------------------------------------------------------------

@timeit
@pytest.mark.skip(reason="no way of currently testing this")
def test_batch_apply_and_measure(temp_hdf5_file):
    """
    Tests the batch application and measurement of image processing and measurement
    pipelines. This function validates if the pipeline's `apply_and_measure`
    method works correctly by comparing the generated DataFrame to the one retrieved
    via `ImageSet.get_measurement()`.

    Args:
        temp_hdf5_file: Temporary HDF5 file used to create an ImageSet for testing.

    Raises:
        AssertionError: If the resulting DataFrame from `apply_and_measure` is either
            empty or not equal to the alternative DataFrame retrieved from
            `ImageSet.get_measurement()`.
    """
    imageset = _make_imageset(temp_hdf5_file)
    pipe = ImagePipeline(
            ops=[DetectFull()],
            meas=[SumObjects()],
            verbose=False,
            njobs=2)

    df = pipe.apply_and_measure(imageset)
    assert df.empty is False, 'No measurements from batch apply_and_measure'

    alt_df = imageset.get_measurement()
    assert df.equals(alt_df), 'ImageSet.get_measurements() is different from results'


@timeit
@pytest.mark.skip(reason="no way of currently testing this")
def test_batch_apply_and_measure_repeated(temp_hdf5_file):
    """
    Tests the batch application and measurement functionality of the image
    pipeline and ensures consistency across repeated runs.

    Args:
        temp_hdf5_file: Temporary HDF5 file where the dataset is stored. Used
            as the input source for creating the image set.

    Raises:
        AssertionError: If the resulting dataframe from the batch application
            is empty or if the results from two successive batch applications
            are not identical.
    """
    imageset = _make_imageset(temp_hdf5_file)
    pipe = ImagePipeline(
            ops=[DetectFull()],
            meas=[SumObjects()],
            verbose=False,
            njobs=2)

    df = pipe.apply_and_measure(imageset)
    assert df.empty is False, 'No measurements from batch apply_and_measure'

    df2 = pipe.apply_and_measure(imageset)
    assert df2.equals(df), 'apply_and_measure run 2 is different from run 1 results'
