"""Lightweight unit tests for ImagePipelineCore and ImagePipelineBatch
These tests use stub classes to avoid heavy dependencies and long runtimes.
Run together with the integration tests already present.
"""

from pathlib import Path

import h5py
import pandas as pd
import numpy as np

from phenotypic import ImagePipeline, Image, ImageSet
from phenotypic.abstract import MeasureFeatures, ObjectDetector
from phenotypic.objects import BorderObjectRemover
from phenotypic.data import load_plate_12hr, load_plate_72hr
from phenotypic.detection import OtsuDetector
from phenotypic.measure import MeasureShape
from .resources.TestHelper import timeit


class SumObjects(MeasureFeatures):
    def _operate(self, image: Image) -> pd.DataFrame:
        return pd.DataFrame({'Sum': image.array[:].sum()}, index=image.objects.labels2series())


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
        image_list=images,
        out_path=tmp_path / "iset.h5",
        overwrite=True)


def _make_dummy_imageset(tmp_path: Path):
    images = [
        Image(np.full(shape=(2, 2, 3), fill_value=1), name='image1'),
        Image(np.full(shape=(3, 3, 3), fill_value=1), name='image2'),
    ]
    return ImageSet(
        name='iset',
        image_list=images,
        out_path=tmp_path / 'iset.h5',
        overwrite=True,
    )


# ---------------------------------------------------------------------------
# Tests for ImagePipelineCore
# ---------------------------------------------------------------------------

@timeit
def test_core_apply_and_measure():
    img = Image(load_plate_12hr(), name='12hr')
    pipe = ImagePipeline(ops=[OtsuDetector(), BorderObjectRemover(border_size=1)], measurements=[SumObjects()])

    df = pipe.apply_and_measure(img)
    assert not df.empty


# ---------------------------------------------------------------------------
# Tests for ImagePipelineBatch (single worker to keep CI light)
# ---------------------------------------------------------------------------

@timeit
def test_batch_apply_and_measure(tmp_path):
    imageset = _make_dummy_imageset(tmp_path)
    pipe = ImagePipeline(ops=[DetectFull()], measurements=[SumObjects()], verbose=False)

    df = pipe.apply_and_measure(imageset, num_workers=1, verbose=False)
    print(df)
    assert all([x in df.loc[:, 'Sum'] for x in [12, 27]]), "runtime aggregated sum of objects should be 4 and 9"

    alt_df = imageset.get_measurement()
    print(alt_df)
    assert all([x in alt_df.loc[:, 'Sum'] for x in [12, 27]]), "post-runtime aggregated sum of objects should be 4 and 9"

    # Verify images and measurements got written to HDF5
    with h5py.File(imageset._out_path, "r", libver="latest", swmr=True) as h5:
        grp = h5[str(imageset._hdf5_images_group_key)]
        assert len(grp) == len(imageset.get_image_names())
        for name in imageset.get_image_names():
            assert name in grp
            # Check if measurements were stored in HDF5 format (not pandas format)
            if name in grp and "measurements" in grp[name]:
                meas_group = grp[name]["measurements"]
                assert "values" in meas_group
                assert "columns" in meas_group
                assert "index" in meas_group
