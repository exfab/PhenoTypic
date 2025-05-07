import pandas as pd

import numpy as np
import skimage

import phenotypic

from .resources.TestHelper import timeit

from .test_fixtures import sample_image_array_with_imformat


@timeit
def test_empty_image():
    empty_image = phenotypic.Image()
    assert empty_image is not None
    assert empty_image.isempty() is True


@timeit
def test_set_image_from_array(sample_image_array_with_imformat):
    """
    Tests the functionality of setting an image from an array and verifies the
    image properties such as shape, non-emptiness, and proper initialization.

    Args:
        sample_images: A tuple containing the input image array, input image
            format, and the expected true image format.
    """
    input_image, input_imformat, true_imformat = sample_image_array_with_imformat
    phenotypic_image = phenotypic.Image()
    phenotypic_image.set_image(input_image, input_imformat)
    assert phenotypic_image is not None
    assert phenotypic_image.isempty() is False
    assert phenotypic_image.shape == input_image.shape


@timeit
def test_set_image_from_image(sample_image_array_with_imformat):
    """
    Tests the `set_image` method of the `Image` class from the `phenotypic` package. The function
    validates that an image can be set from another `Image` instance or raw input data, with
    properties and states intact.

    Args:
        sample_image_inputs: A tuple containing the following:
            input_image: The input image as a NumPy array.
            input_imformat: The format of the input image as a string.
            true_imformat: The expected image format as a string.
    """
    input_image, input_imformat, true_imformat = sample_image_array_with_imformat
    phenotypic_image = phenotypic.Image()
    phenotypic_image.set_image(phenotypic.Image(input_image=input_image, imformat=input_imformat))

    phenotypic_image_2 = phenotypic.Image()
    phenotypic_image_2.set_image(phenotypic_image)
    assert phenotypic_image_2 is not None
    assert phenotypic_image_2.isempty() is False
    assert phenotypic_image_2.shape == input_image.shape
    if true_imformat != 'Grayscale':
        assert np.array_equal(phenotypic_image_2.array[:], phenotypic_image.array[:])
    assert np.array_equal(phenotypic_image_2.matrix[:], phenotypic_image.matrix[:])
    assert np.array_equal(phenotypic_image_2.enh_matrix[:], phenotypic_image.enh_matrix[:])
    assert np.array_equal(phenotypic_image_2.objmask[:], phenotypic_image.objmask[:])
    assert np.array_equal(phenotypic_image_2.objmap[:], phenotypic_image.objmap[:])


@timeit
def test_image_construct_from_array(sample_image_array_with_imformat):
    input_image, input_imformat, true_imformat = sample_image_array_with_imformat
    phenotypic_image = phenotypic.Image(input_image=input_image, imformat=input_imformat)
    assert phenotypic_image is not None
    assert phenotypic_image.isempty() is False
    assert phenotypic_image.shape == input_image.shape


@timeit
def test_image_array_access(sample_image_array_with_imformat):
    input_image, input_imformat, true_imformat = sample_image_array_with_imformat
    phenotypic_image = phenotypic.Image(input_image=input_image, imformat=input_imformat)
    if true_imformat != 'Grayscale':
        assert np.array_equal(phenotypic_image.array[:], input_image)


@timeit
def test_image_matrix_access(sample_image_array_with_imformat):
    input_image, input_imformat, true_imformat = sample_image_array_with_imformat
    ps_image = phenotypic.Image(input_image=input_image, imformat=input_imformat)
    if input_imformat == 'RGB':
        assert np.array_equal(ps_image.matrix[:], skimage.color.rgb2gray(input_image))
    elif input_imformat == 'Grayscale':
        assert np.array_equal(ps_image.matrix[:], input_image)


@timeit
def test_image_det_matrix_access(sample_image_array_with_imformat):
    input_image, input_imformat, true_imformat = sample_image_array_with_imformat
    ps_image = phenotypic.Image(input_image=input_image, imformat=input_imformat)
    assert np.array_equal(ps_image.enh_matrix[:], ps_image.matrix[:])

    ps_image.enh_matrix[:10, :10] = 0
    ps_image.enh_matrix[-10:, -10:] = 1
    assert not np.array_equal(ps_image.enh_matrix[:], ps_image.matrix[:])


@timeit
def test_image_object_mask_access(sample_image_array_with_imformat):
    input_image, input_imformat, true_imformat = sample_image_array_with_imformat
    ps_image = phenotypic.Image(input_image=input_image, imformat=input_imformat)

    # When no objects in image
    assert np.array_equal(ps_image.objmask[:], np.full(shape=ps_image.matrix.shape, fill_value=True))

    ps_image.objmask[:10, :10] = 0
    ps_image.objmask[-10:, -10:] = 1

    assert not np.array_equal(ps_image.objmask[:], np.full(shape=ps_image.matrix.shape, fill_value=True))


@timeit
def test_image_object_map_access(sample_image_array_with_imformat):
    input_image, input_imformat, true_imformat = sample_image_array_with_imformat
    ps_image = phenotypic.Image(input_image=input_image, imformat=input_imformat)

    # When no objects in image
    assert np.array_equal(ps_image.objmap[:], np.full(shape=ps_image.matrix.shape, fill_value=1, dtype=np.uint32))
    assert ps_image.num_objects == 0

    ps_image.objmap[:10, :10] = 1
    ps_image.objmap[-10:, -10:] = 2

    assert not np.array_equal(ps_image.objmap[:], np.full(shape=ps_image.matrix.shape, fill_value=1, dtype=np.uint32))
    assert ps_image.num_objects > 0
    assert ps_image.objects.num_objects > 0


@timeit
def test_image_copy(sample_image_array_with_imformat):
    input_image, input_imformat, true_imformat = sample_image_array_with_imformat
    ps_image = phenotypic.Image(input_image=input_image, imformat=input_imformat)
    ps_image_copy = ps_image.copy()
    assert ps_image_copy is not ps_image
    assert ps_image_copy.isempty() is False

    assert ps_image._metadata.private != ps_image_copy._metadata.private
    assert ps_image._metadata.protected == ps_image_copy._metadata.protected
    assert ps_image._metadata.public == ps_image_copy._metadata.public

    if true_imformat != 'Grayscale':
        assert np.array_equal(ps_image.array[:], ps_image.array[:])
    assert np.array_equal(ps_image.matrix[:], ps_image_copy.matrix[:])
    assert np.array_equal(ps_image.enh_matrix[:], ps_image_copy.enh_matrix[:])
    assert np.array_equal(ps_image.objmask[:], ps_image_copy.objmask[:])
    assert np.array_equal(ps_image.objmap[:], ps_image_copy.objmap[:])


@timeit
def test_slicing(sample_image_array_with_imformat):
    input_image, input_imformat, true_imformat = sample_image_array_with_imformat
    ps_image = phenotypic.Image(input_image=input_image, imformat=input_imformat)
    row_slice, col_slice = 10, 10
    sliced_ps_image = ps_image[:row_slice, :col_slice]
    if true_imformat != 'Grayscale':
        assert np.array_equal(sliced_ps_image.array[:], ps_image.array[:row_slice, :col_slice])
    assert np.array_equal(sliced_ps_image.matrix[:], ps_image.matrix[:row_slice, :col_slice])
    assert np.array_equal(sliced_ps_image.enh_matrix[:], ps_image.enh_matrix[:row_slice, :col_slice])
    assert np.array_equal(sliced_ps_image.objmask[:], ps_image.objmask[:row_slice, :col_slice])
    assert np.array_equal(sliced_ps_image.objmap[:], ps_image.objmap[:row_slice, :col_slice])


@timeit
def test_image_object_size_label_consistency(sample_image_array_with_imformat):
    input_image, input_imformat, true_imformat = sample_image_array_with_imformat
    ps_image = phenotypic.Image(input_image=input_image, imformat=input_imformat)
    assert ps_image.num_objects == 0

    ps_image.objmap[:10, :10] = 1
    ps_image.objmap[-10:, -10:] = 2

    assert ps_image.num_objects == 2
    assert ps_image.num_objects == ps_image.objects.num_objects
    assert ps_image.num_objects == len(ps_image.objects.labels)


@timeit
def test_image_object_label_consistency_with_skimage(sample_image_array_with_imformat):
    input_image, input_imformat, true_imformat = sample_image_array_with_imformat
    ps_image = phenotypic.Image(input_image=input_image, imformat=input_imformat)

    ps_image.objmap[:10, :10] = 1
    ps_image.objmap[-10:, -10:] = 2

    assert ps_image.objects.get_labels_series().equals(
        pd.Series(skimage.measure.regionprops_table(ps_image.objmap[:], properties=['label'])['label'])
    )
