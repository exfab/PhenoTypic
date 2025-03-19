import numpy as np
import pytest
from phenoscope import Image, GridImage
from phenoscope.grid import OptimalCenterGridFinder
from phenoscope.util.exceptions_ import IllegalAssignmentError

from .resources.TestHelper import timeit

@timeit
def test_gridimage_initialization():
    # Test default initialization
    grid_image = GridImage()
    assert grid_image is not None
    assert isinstance(grid_image._grid_setter, OptimalCenterGridFinder)

    # Test custom initialization with image and grid setter
    input_image = np.zeros((100, 100))
    grid_setter = OptimalCenterGridFinder(nrows=10, ncols=10)
    grid_image = GridImage(input_image=input_image, grid_setter=grid_setter)
    assert grid_image._grid_setter == grid_setter

@timeit
def test_grid_accessor_property():
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
def test_image_grid_section_retrieval():
    input_image = np.random.rand(100, 100)
    grid_image = GridImage(input_image=input_image)
    sub_image = grid_image[10:20, 10:20]
    assert isinstance(sub_image, Image)
    assert sub_image.shape == (10, 10)

@timeit
def test_grid_show_overlay():
    input_image = np.random.rand(100, 100)
    grid_image = GridImage(input_image=input_image)
    fig, ax = grid_image.show_overlay(show_gridlines=True, annotate=False)
    assert fig is not None
    assert ax is not None

@timeit
def test_optimal_grid_setter_defaults():
    grid_image = GridImage()
    grid_setter = grid_image._grid_setter
    assert isinstance(grid_setter, OptimalCenterGridFinder)
    assert grid_setter.nrows == 8
    assert grid_setter.ncols == 12
