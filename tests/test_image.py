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


def print_inputs(image_array, input_schema, schema):
    if DEBUG:
        print(
            f'input_type:{type(image_array)} input_schema:{input_schema} schema:{schema}'
        )


def test_empty_image():
    empty_image = Image()
    assert empty_image is not None
    assert empty_image.isempty() is True


def test_set_image(sample_image_arrays):
    for image, input_schema, schema in sample_image_arrays:
        print_inputs(image, input_schema, schema)
        phenoscope_image = Image()
        phenoscope_image.set_image(image, input_schema)
        assert phenoscope_image is not None
        assert phenoscope_image.isempty() is False
        assert phenoscope_image.shape == image.shape


def test_image_construct_from_array(sample_image_arrays):
    for image, input_schema, schema in sample_image_arrays:
        print_inputs(image, input_schema, schema)
        phenoscope_image = Image(input_image=image, input_schema=input_schema)
        assert phenoscope_image is not None
        assert phenoscope_image.isempty() is False
        assert phenoscope_image.shape == image.shape


def test_image_array_access(sample_image_arrays):
    for image, input_schema, schema in sample_image_arrays:
        print_inputs(image, input_schema, schema)
        phenoscope_image = Image(input_image=image, input_schema=schema)
        if schema != 'Grayscale':
            assert np.array_equal(phenoscope_image.array[:], image)


def test_image_matrix_access(sample_image_arrays):
    for image, input_schema, schema in sample_image_arrays:
        print_inputs(image, input_schema, schema)
        ps_image = Image(input_image=image, input_schema=input_schema)
        if input_schema == 'RGB':
            assert np.array_equal(ps_image.matrix[:], skimage.color.rgb2gray(image))
        elif input_schema == 'Grayscale':
            assert np.array_equal(ps_image.matrix[:], image)


def test_image_det_matrix_access(sample_image_arrays):
    for image, input_schema, schema in sample_image_arrays:
        print_inputs(image, input_schema, schema)
        ps_image = Image(input_image=image, input_schema=schema)
        assert np.array_equal(ps_image.det_matrix[:], ps_image.matrix[:])

        ps_image.det_matrix[:10, :10] = 0
        ps_image.det_matrix[-10:, -10:] = 1
        assert not np.array_equal(ps_image.det_matrix[:], ps_image.matrix[:])


def test_image_object_mask_access(sample_image_arrays):
    for image, input_schema, schema in sample_image_arrays:
        print_inputs(image, input_schema, schema)
        ps_image = Image(input_image=image, input_schema=schema)

        # When no objects in image
        assert np.array_equal(ps_image.obj_mask[:], np.full(shape=ps_image.matrix.shape, fill_value=True))

        ps_image.obj_mask[:10, :10] = 0
        ps_image.obj_mask[-10:, -10:] = 1

        assert not np.array_equal(ps_image.obj_mask[:], np.full(shape=ps_image.matrix.shape, fill_value=True))


def test_image_object_map_access(sample_image_arrays):
    for image, input_schema, schema in sample_image_arrays:
        print_inputs(image, input_schema, schema)
        ps_image = Image(input_image=image, input_schema=schema)

        # When no objects in image
        assert np.array_equal(ps_image.obj_map[:], np.full(shape=ps_image.matrix.shape, fill_value=1, dtype=np.uint32))
        assert ps_image.objects.num_objects == 0

        ps_image.obj_map[:10, :10] = 1
        ps_image.obj_map[-10:, -10:] = 2

        assert not np.array_equal(ps_image.obj_map[:], np.full(shape=ps_image.matrix.shape, fill_value=1, dtype=np.uint32))
        assert ps_image.objects.num_objects > 0

def test_image_copy(sample_image_arrays):
    for image, input_schema, schema in sample_image_arrays:
        print_inputs(image, input_schema, schema)
        ps_image = Image(input_image=image, input_schema=schema)
        ps_image_copy = ps_image.copy()
        assert ps_image_copy is not ps_image
        assert ps_image_copy.isempty() is False

        assert ps_image._private_metadata != ps_image_copy._private_metadata
        assert ps_image._protected_metadata == ps_image_copy._protected_metadata
        assert ps_image._public_metadata == ps_image_copy._public_metadata

        if schema != 'Grayscale':
            assert np.array_equal(ps_image.array[:], ps_image.array[:])
        assert np.array_equal(ps_image.matrix[:], ps_image_copy.matrix[:])
        assert np.array_equal(ps_image.det_matrix[:], ps_image_copy.det_matrix[:])
        assert np.array_equal(ps_image.obj_mask[:], ps_image_copy.obj_mask[:])
        assert np.array_equal(ps_image.obj_map[:], ps_image_copy.obj_map[:])

def test_slicing(sample_image_arrays):
    for image, input_schema, schema in sample_image_arrays:
        print_inputs(image, input_schema, schema)
        ps_image = Image(input_image=image, input_schema=schema)
        row_slice, col_slice = 10, 10
        sliced_ps_image = ps_image[:row_slice,:col_slice]
        if schema != 'Grayscale':
            assert np.array_equal(sliced_ps_image.array[:], ps_image.array[:row_slice,:col_slice])
        assert np.array_equal(sliced_ps_image.matrix[:], ps_image.matrix[:row_slice,:col_slice])
        assert np.array_equal(sliced_ps_image.det_matrix[:], ps_image.det_matrix[:row_slice,:col_slice])
        assert np.array_equal(sliced_ps_image.obj_mask[:], ps_image.obj_mask[:row_slice,:col_slice])
        assert np.array_equal(sliced_ps_image.obj_map[:], ps_image.obj_map[:row_slice,:col_slice])
