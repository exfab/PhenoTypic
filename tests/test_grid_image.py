import numpy as np
import pytest
from phenotypic import Image, GridImage
from phenotypic.grid import OptimalCenterGridFinder
from phenotypic.detection import OtsuDetector
from phenotypic.util.exceptions_ import IllegalAssignmentError

from .resources.TestHelper import timeit

from .test_fixtures import plate_grid_images_with_detection, sample_image_array

@timeit
def test_blank_gridimage_initialization():
    # Test default initialization
    grid_image = GridImage()
    assert grid_image is not None
    assert isinstance(grid_image._grid_setter, OptimalCenterGridFinder)

@timeit
def test_gridimage_initialization(sample_image_array):
    # Test custom initialization with image and grid setter
    input_image = sample_image_array
    grid_image = GridImage(input_image=input_image)
    assert grid_image.isempty() is False

    grid_setter = OptimalCenterGridFinder(nrows=10, ncols=10)
    grid_image = GridImage(input_image=input_image, grid_finder=grid_setter)
    assert grid_image._grid_setter == grid_setter


@timeit
def test_grid_accessor_default_property():
    grid_image = GridImage()
    grid_accessor = grid_image.grid
    assert grid_accessor is not None
    assert grid_accessor.nrows == 8
    assert grid_accessor.ncols == 12


@timeit
def test_grid_property_assignment_error():
    grid_image = GridImage()
    with pytest.raises(IllegalAssignmentError):
        grid_image.grid = "some value"


@timeit
def test_image_grid_section_retrieval(plate_grid_images_with_detection):
    grid_image = plate_grid_images_with_detection
    sub_image = grid_image[10:20, 10:20]
    assert isinstance(sub_image, Image)
    assert sub_image.shape[:2] == (10, 10)


@timeit
def test_grid_show_overlay(plate_grid_images_with_detection):
    grid_image = plate_grid_images_with_detection
    fig, ax = grid_image.show_overlay(annotate=False)
    assert fig is not None
    assert ax is not None


@timeit
def test_optimal_grid_setter_defaults():
    grid_image = GridImage()
    grid_setter = grid_image._grid_setter
    assert isinstance(grid_setter, OptimalCenterGridFinder)
    assert grid_setter.nrows == 8
    assert grid_setter.ncols == 12
