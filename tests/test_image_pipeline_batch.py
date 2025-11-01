"""Lightweight unit tests for ImagePipelineCore and ImagePipelineBatch
These tests use stub classes to avoid heavy dependencies and long runtimes.
Run together with the integration tests already present.
"""

from pathlib import Path

import pandas as pd

from phenotypic import Image, ImagePipeline, ImageSet
from phenotypic.ABC_ import MeasureFeatures, ObjectDetector
from phenotypic.data import load_plate_12hr
from phenotypic.detection import OtsuDetector
from phenotypic.objedit import BorderObjectRemover
from .resources.TestHelper import timeit
from .test_fixtures import temp_hdf5_file


class SumObjects(MeasureFeatures):
    def _operate(self, image: Image) -> pd.DataFrame:
        labels = image.objects.labels2series()
        return pd.DataFrame({
            labels.name: labels.values,
            'Sum'      : SumObjects._calculate_sum(array=image.matrix[:], labels=image.objmap[:])
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
def test_batch_apply_and_measure(temp_hdf5_file):
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
