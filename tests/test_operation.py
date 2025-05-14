import pickle, pytest

from phenotypic.abstract import ImageOperation

from phenotypic.data import load_plate_12hr
from phenotypic.detection import OtsuDetector
from phenotypic.
from .test_fixtures import _public


@pytest.mark.parametrize("qualname,obj", _public)
def test_operation(qualname, obj):

    if issubclass(ImageOperation):

