import pytest

from phenotypic.abstract import ImageOperation

import phenotypic
from phenotypic.data import load_plate_12hr
from phenotypic.detection import OtsuDetector

from .test_fixtures import _public


# TODO: Finish writing this test
# @pytest.mark.parametrize("qualname,obj", _public)
# def test_operation(qualname, obj):
#     """The goal of this test is to ensure that all operations are callable with basic functionality,
#      and return a valid Image object."""
#     image = phenotypic.Image(load_plate_12hr())
#     image = OtsuDetector().apply(image)
#     if issubclass(obj, ImageOperation):
#         assert obj().apply(image).isempty() is False
#
