import pandas as pd
import pytest

import numpy as np
import skimage

from phenoscope.data import (
    load_colony_12_hr,
    load_colony_72hr,
    load_plate_12hr,
    load_plate_72hr,
)

from phenoscope import Image

from .resources.TestHelper import timeit
DEBUG = False


@pytest.fixture(scope='session')
def sample_image_arrays():
    """Fixture that returns (image_array, input_schema,schema)"""
    return [
        (load_colony_12_hr(), None, 'RGB'),  # Test Auto Formatter
        (load_colony_72hr(), 'RGB', 'RGB'),
        (load_plate_12hr(), 'RGB', 'RGB'),
        (load_plate_72hr(), 'RGB', 'RGB'),
        (np.full(shape=(100, 100), fill_value=0), None, 'Grayscale'),  # Black Image
        (np.full(shape=(100, 100), fill_value=0), 'Grayscale', 'Grayscale'),  # Black Image
        (np.full(shape=(100, 100), fill_value=1), 'Grayscale', 'Grayscale'),  # White Image
    ]

def print_inputs(image_array, input_schema, true_schema):
    if DEBUG:
        print(
            f'input_type:{type(image_array)} input_schema:{input_schema} schema:{true_schema}'
        )

@timeit
def test_empty_image():
    empty_image = Image()
    assert empty_image is not None
    assert empty_image.isempty() is True

@timeit
def test_set_image_from_array(sample_image_arrays):
    """
    Tests the functionality of setting an image from a given array and validating its attributes.

    This function tests several properties and behaviors of the `Image` class, ensuring
    that an image array can be successfully set to an instance of the `Image` class. It
    checks for successful creation, verifies that the image object is not empty after
    setting, and ensures the shape of the image matches the input image array.

    Args:
        sample_image_arrays (list[tuple]): A list of tuples, where each tuple contains
            an image array, input schema details, and schema details. Each tuple in the
            input provides the data needed for testing the behavior of the `set_image`
            method.
    """
    for image, input_schema, true_schema in sample_image_arrays:
        print_inputs(image, input_schema, true_schema)
        phenoscope_image = Image()
        phenoscope_image.set_image(image, input_schema)
        assert phenoscope_image is not None
        assert phenoscope_image.isempty() is False
        assert phenoscope_image.shape == image.shape

@timeit
def test_set_image_from_image(sample_image_arrays):
    """
    Tests the functionality of setting an image using an image object, ensuring that
    deep copying and data integrity are maintained during the process. The test
    verifies a variety of attributes including shape, array, matrices, and object
    masks of the resulting Image instance.

    Args:
        sample_image_arrays (list): A list of tuples where each tuple contains an
            image mock (numpy.ndarray), an input schema, and a corresponding schema.
    """
    for image, input_schema, true_schema in sample_image_arrays:
        print_inputs(image, input_schema, true_schema)
        phenoscope_image = Image()
        phenoscope_image.set_image(Image(input_image=image, input_schema=input_schema))

        phenoscope_image_2 = Image()
        phenoscope_image_2.set_image(phenoscope_image)
        assert phenoscope_image_2 is not None
        assert phenoscope_image_2.isempty() is False
        assert phenoscope_image_2.shape == image.shape
        if true_schema != 'Grayscale':
            assert np.array_equal(phenoscope_image_2.array[:], phenoscope_image.array[:])
        assert np.array_equal(phenoscope_image_2.matrix[:], phenoscope_image.matrix[:])
        assert np.array_equal(phenoscope_image_2.enh_matrix[:], phenoscope_image.enh_matrix[:])
        assert np.array_equal(phenoscope_image_2.omask[:], phenoscope_image.omask[:])
        assert np.array_equal(phenoscope_image_2.omap[:], phenoscope_image.omap[:])

@timeit
def test_image_construct_from_array(sample_image_arrays):
    for image, input_schema, true_schema in sample_image_arrays:
        print_inputs(image, input_schema, true_schema)
        phenoscope_image = Image(input_image=image, input_schema=input_schema)
        assert phenoscope_image is not None
        assert phenoscope_image.isempty() is False
        assert phenoscope_image.shape == image.shape

@timeit
def test_image_array_access(sample_image_arrays):
    for image, input_schema, true_schema in sample_image_arrays:
        print_inputs(image, input_schema, true_schema)
        phenoscope_image = Image(input_image=image, input_schema=true_schema)
        if true_schema != 'Grayscale':
            assert np.array_equal(phenoscope_image.array[:], image)

@timeit
def test_image_matrix_access(sample_image_arrays):
    for image, input_schema, true_schema in sample_image_arrays:
        print_inputs(image, input_schema, true_schema)
        ps_image = Image(input_image=image, input_schema=input_schema)
        if input_schema == 'RGB':
            assert np.array_equal(ps_image.matrix[:], skimage.color.rgb2gray(image))
        elif input_schema == 'Grayscale':
            assert np.array_equal(ps_image.matrix[:], image)

@timeit
def test_image_det_matrix_access(sample_image_arrays):
    for image, input_schema, true_schema in sample_image_arrays:
        print_inputs(image, input_schema, true_schema)
        ps_image = Image(input_image=image, input_schema=true_schema)
        assert np.array_equal(ps_image.enh_matrix[:], ps_image.matrix[:])

        ps_image.enh_matrix[:10, :10] = 0
        ps_image.enh_matrix[-10:, -10:] = 1
        assert not np.array_equal(ps_image.enh_matrix[:], ps_image.matrix[:])

@timeit
def test_image_object_mask_access(sample_image_arrays):
    for image, input_schema, true_schema in sample_image_arrays:
        print_inputs(image, input_schema, true_schema)
        ps_image = Image(input_image=image, input_schema=true_schema)

        # When no objects in image
        assert np.array_equal(ps_image.omask[:], np.full(shape=ps_image.matrix.shape, fill_value=True))

        ps_image.omask[:10, :10] = 0
        ps_image.omask[-10:, -10:] = 1

        assert not np.array_equal(ps_image.omask[:], np.full(shape=ps_image.matrix.shape, fill_value=True))

@timeit
def test_image_object_map_access(sample_image_arrays):
    for image, input_schema, true_schema in sample_image_arrays:
        print_inputs(image, input_schema, true_schema)
        ps_image = Image(input_image=image, input_schema=true_schema)

        # When no objects in image
        assert np.array_equal(ps_image.omap[:], np.full(shape=ps_image.matrix.shape, fill_value=1, dtype=np.uint32))
        assert ps_image.num_objects == 0

        ps_image.omap[:10, :10] = 1
        ps_image.omap[-10:, -10:] = 2

        assert not np.array_equal(ps_image.omap[:], np.full(shape=ps_image.matrix.shape, fill_value=1, dtype=np.uint32))
        assert ps_image.num_objects > 0
        assert ps_image.objects.num_objects > 0

@timeit
def test_image_copy(sample_image_arrays):
    for image, input_schema, true_schema in sample_image_arrays:
        print_inputs(image, input_schema, true_schema)
        ps_image = Image(input_image=image, input_schema=true_schema)
        ps_image_copy = ps_image.copy()
        assert ps_image_copy is not ps_image
        assert ps_image_copy.isempty() is False

        assert ps_image._private_metadata != ps_image_copy._private_metadata
        assert ps_image._protected_metadata == ps_image_copy._protected_metadata
        assert ps_image._public_metadata == ps_image_copy._public_metadata

        if true_schema != 'Grayscale':
            assert np.array_equal(ps_image.array[:], ps_image.array[:])
        assert np.array_equal(ps_image.matrix[:], ps_image_copy.matrix[:])
        assert np.array_equal(ps_image.enh_matrix[:], ps_image_copy.enh_matrix[:])
        assert np.array_equal(ps_image.omask[:], ps_image_copy.omask[:])
        assert np.array_equal(ps_image.omap[:], ps_image_copy.omap[:])

@timeit
def test_slicing(sample_image_arrays):
    for image, input_schema, true_schema in sample_image_arrays:
        print_inputs(image, input_schema, true_schema)
        ps_image = Image(input_image=image, input_schema=true_schema)
        row_slice, col_slice = 10, 10
        sliced_ps_image = ps_image[:row_slice, :col_slice]
        if true_schema != 'Grayscale':
            assert np.array_equal(sliced_ps_image.array[:], ps_image.array[:row_slice, :col_slice])
        assert np.array_equal(sliced_ps_image.matrix[:], ps_image.matrix[:row_slice, :col_slice])
        assert np.array_equal(sliced_ps_image.enh_matrix[:], ps_image.enh_matrix[:row_slice, :col_slice])
        assert np.array_equal(sliced_ps_image.omask[:], ps_image.omask[:row_slice, :col_slice])
        assert np.array_equal(sliced_ps_image.omap[:], ps_image.omap[:row_slice, :col_slice])

@timeit
def test_image_object_size_label_consistency(sample_image_arrays):
    for image, input_schema, true_schema in sample_image_arrays:
        print_inputs(image, input_schema, true_schema)
        ps_image = Image(input_image=image, input_schema=true_schema)
        assert ps_image.num_objects == 0

        ps_image.omap[:10, :10] = 1
        ps_image.omap[-10:, -10:] = 2

        assert ps_image.num_objects == 2
        assert ps_image.num_objects == ps_image.objects.num_objects
        assert ps_image.num_objects == len(ps_image.objects.labels)

@timeit
def test_image_object_label_consistency_with_skimage(sample_image_arrays):
    for image, input_schema, true_schema in sample_image_arrays:
        print_inputs(image, input_schema, true_schema)
        ps_image = Image(input_image=image, input_schema=true_schema)

        ps_image.omap[:10, :10] = 1
        ps_image.omap[-10:, -10:] = 2

        assert ps_image.objects.label_table().equals(pd.Series(skimage.measure.regionprops_table(ps_image.omap[:], properties=['label'])['label']))