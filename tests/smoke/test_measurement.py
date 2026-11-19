import pytest

import pandas as pd

from phenotypic.abc_ import MeasureEmbedding
from phenotypic.data import load_synth_yeast_plate
from phenotypic.refine import MaskOpening

from unit.test_fixtures import _image_measurements
from unit.resources.TestHelper import timeit

# An embedder's default construction loads real model weights (DINOv2-base,
# ~350 MB, network-dependent), so it is left out here as in
# tests/unit/util/test_output_headers.py. The stub-backbone tests in
# tests/unit/measure/test_measure_dino_embedding.py cover the measurer. The
# filter lives here, not in walk_package_for_measurements, whose other
# consumer (test_serialization.py) round-trips embedders without loading them.
_default_built_measurements = [
    (qualname, obj)
    for qualname, obj in _image_measurements
    if not issubclass(obj, MeasureEmbedding)
]


@pytest.fixture(scope="session")
def mask_opened_image():
    """Session-scoped image with MaskOpening pre-applied."""
    image = load_synth_yeast_plate()
    MaskOpening().apply(image, inplace=True)
    return image


@pytest.mark.smoke
@pytest.mark.parametrize("qualname,obj", _default_built_measurements)
@timeit
def test_measurement(qualname, obj, mask_opened_image):
    """The goal of this test is to ensure that all operations are callable with basic functionality,
    and return a valid dataframe object. This does not check for accuracy"""
    image = mask_opened_image.copy()
    assert isinstance(obj().measure(image), pd.DataFrame)
