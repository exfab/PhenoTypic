"""Lightweight unit tests for ImagePipelineCore and ImagePipelineBatch
These tests use stub classes to avoid heavy dependencies and long runtimes.
Run together with the integration tests already present.
"""

from pathlib import Path

import h5py
import pandas as pd

from phenotypic import ImagePipeline, Image, ImageSet
from phenotypic.data import load_plate_12hr, load_plate_72hr
from phenotypic.detection import OtsuDetector
from phenotypic.measure import MeasureShape


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


# ---------------------------------------------------------------------------
# Tests for ImagePipelineCore
# ---------------------------------------------------------------------------

def test_core_apply_and_measure():
    img = Image(load_plate_12hr(), name='12hr')
    pipe = ImagePipeline(ops=[OtsuDetector()], measurements=[MeasureShape()])

    df = pipe.apply_and_measure(img)
    assert not df.empty


# ---------------------------------------------------------------------------
# Tests for ImagePipelineBatch (single worker to keep CI light)
# ---------------------------------------------------------------------------

def test_batch_apply_and_measure(tmp_path):
    imageset = _make_imageset(tmp_path)
    pipe = ImagePipeline(ops=[OtsuDetector()], measurements=[MeasureShape()], verbose=False)

    df = pipe.apply_and_measure(imageset, num_workers=1, verbose=False)

    # Verify images and measurements got written to HDF5
    with h5py.File(imageset._out_path, "r", libver="latest", swmr=True) as h5:
        grp = h5[str(imageset._hdf5_image_group_key)]
        assert len(grp) == len(imageset.get_image_names())
        for name in imageset.get_image_names():
            assert name in grp
            # Check if measurements were stored in HDF5 format (not pandas format)
            if name in grp and "measurements" in grp[name]:
                meas_group = grp[name]["measurements"]
                assert "values" in meas_group
                assert "columns" in meas_group
                assert "index" in meas_group
