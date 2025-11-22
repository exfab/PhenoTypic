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
    Tests the functionality to set an image from an array into the `phenotypic.Image` object and ensures
    that the image attributes are assigned correctly. This is particularly relevant for processing
    images of microbe colonies on solid media agar, as correct initialization ensures subsequent
    processing will reflect the colony morphology accurately.

    Args:
        sample_image_array_with_imformat: A tuple containing the input image as a NumPy array, the
            expected image format of the input, and the true image format. The input image determines
            the initial representation of the microbial colony image. Accurate representation of
            microbial colonies depends on the dimensionality and format of the input image.
    """
    input_image, input_imformat, true_imformat = sample_image_array_with_imformat
    phenotypic_image = phenotypic.Image()
    phenotypic_image.set_image(input_image)
    assert phenotypic_image is not None
    assert phenotypic_image.isempty() is False
    assert phenotypic_image.shape == input_image.shape


@timeit
def test_set_image_from_image(sample_image_array_with_imformat):
    """
    Tests the `set_image` method of the `Image` class from the `phenotypic` package. The function
    validates that an image can be set from another `Image` instance or raw arr data, with
    properties and states intact.

    Args:
        sample_image_array_with_imformat: A tuple containing the following:
            arr: The arr image as a NumPy array.
            input_imformat: The format of the arr image as a string.
            true_imformat: The expected image format as a string.
    """
    input_image, input_imformat, true_imformat = sample_image_array_with_imformat
    phenotypic_image = phenotypic.Image()
    phenotypic_image.set_image(phenotypic.Image(arr=input_image))

    phenotypic_image_2 = phenotypic.Image()
    phenotypic_image_2.set_image(phenotypic_image)
    assert phenotypic_image_2 is not None
    assert phenotypic_image_2.isempty() is False
    assert phenotypic_image_2.shape == input_image.shape
    if not phenotypic_image.rgb.isempty():
        assert np.array_equal(phenotypic_image_2.rgb[:], phenotypic_image.rgb[:])
    assert np.array_equal(phenotypic_image_2.gray[:], phenotypic_image.gray[:])
    assert np.array_equal(phenotypic_image_2.enh_gray[:], phenotypic_image.enh_gray[:])
    assert np.array_equal(phenotypic_image_2.objmask[:], phenotypic_image.objmask[:])
    assert np.array_equal(phenotypic_image_2.objmap[:], phenotypic_image.objmap[:])


@timeit
def test_image_construct_from_array(sample_image_array_with_imformat):
    input_image, input_imformat, true_imformat = sample_image_array_with_imformat
    phenotypic_image = phenotypic.Image(arr=input_image)
    assert phenotypic_image is not None
    assert phenotypic_image.isempty() is False
    assert phenotypic_image.shape == input_image.shape


@timeit
def test_image_array_access(sample_image_array_with_imformat):
    input_image, input_imformat, true_imformat = sample_image_array_with_imformat
    phenotypic_image = phenotypic.Image(arr=input_image)
    if not phenotypic_image.rgb.isempty():
        assert np.array_equal(phenotypic_image.rgb[:], input_image)


@timeit
def test_image_matrix_access(sample_image_array_with_imformat):
    input_image, input_imformat, true_imformat = sample_image_array_with_imformat
    ps_image = phenotypic.Image(arr=input_image)
    if not ps_image.rgb.isempty():
        assert np.array_equal(ps_image.gray[:], skimage.color.rgb2gray(input_image)), \
            f'Image.gray and skimage.color.rgb2gray do not match at {np.unique(ps_image.gray[:] != skimage.color.rgb2gray(input_image), return_counts=True)}'
        # assert np.allclose(ps_image.gray[:], skimage.color.rgb2gray(arr), atol=1.0 / np.finfo(ps_image.gray[:].dtype).max),\
        #     f'Image.gray and skimage.color.rgb2gray do not match at {np.unique(ps_image.gray[:] != skimage.color.rgb2gray(arr), return_counts=True)}'
    else:
        assert np.array_equal(ps_image.gray[:], input_image)


@timeit
def test_image_matrix_change(sample_image_array_with_imformat):
    input_image, input_imformat, true_imformat = sample_image_array_with_imformat
    ps_image = phenotypic.Image(arr=input_image)
    ps_image.gray[10:10, 10:10] = 0
    if not ps_image.rgb.isempty():
        altered_image = skimage.color.rgb2gray(input_image)
        altered_image[10:10, 10:10] = 0

        assert np.allclose(ps_image.gray[:], altered_image, atol=1.0/np.finfo(ps_image.gray[:].dtype).max), \
            f'Image.gray and skimage.color.rgb2gray do not match at {np.unique(ps_image.gray[:] != altered_image, return_counts=True)}'

        assert np.array_equal(ps_image.rgb[:],
                              input_image), 'Image.rgb was altered and color information was changed'

    else:
        altered_image = input_image.copy()
        altered_image[10:10, 10:10] = 0
        assert np.array_equal(ps_image.gray[:], altered_image), \
            f'Image.gray and arr do not match at {np.unique(ps_image.gray[:] != altered_image, return_counts=True)}'


@timeit
def test_image_det_matrix_access(sample_image_array_with_imformat):
    input_image, input_imformat, true_imformat = sample_image_array_with_imformat
    ps_image = phenotypic.Image(arr=input_image)
    assert np.array_equal(ps_image.enh_gray[:], ps_image.gray[:])

    ps_image.enh_gray[:10, :10] = 0
    ps_image.enh_gray[-10:, -10:] = 1
    assert not np.array_equal(ps_image.enh_gray[:], ps_image.gray[:])


@timeit
def test_image_object_mask_access(sample_image_array_with_imformat):
    input_image, input_imformat, true_imformat = sample_image_array_with_imformat
    ps_image = phenotypic.Image(arr=input_image)

    # When no objects in _root_image
    assert np.array_equal(ps_image.objmask[:], np.full(shape=ps_image.gray.shape, fill_value=False))

    ps_image.objmask[:10, :10] = 0
    ps_image.objmask[-10:, -10:] = 1

    assert not np.array_equal(ps_image.objmask[:], np.full(shape=ps_image.gray.shape, fill_value=False))


@timeit
def test_image_object_map_access(sample_image_array_with_imformat):
    input_image, input_imformat, true_imformat = sample_image_array_with_imformat
    ps_image = phenotypic.Image(arr=input_image)

    # When no objects in _root_image
    assert np.array_equal(ps_image.objmap[:], np.full(shape=ps_image.gray.shape, fill_value=0, dtype=np.uint32))
    assert ps_image.num_objects == 0

    ps_image.objmap[:10, :10] = 1
    ps_image.objmap[-10:, -10:] = 2

    assert not np.array_equal(ps_image.objmap[:], np.full(shape=ps_image.gray.shape, fill_value=0, dtype=np.uint32))
    assert ps_image.num_objects > 0
    assert ps_image.objects.num_objects > 0


@timeit
def test_image_copy(sample_image_array_with_imformat):
    input_image, input_imformat, true_imformat = sample_image_array_with_imformat
    ps_image = phenotypic.Image(arr=input_image)
    ps_image_copy = ps_image.copy()
    assert ps_image_copy is not ps_image
    assert ps_image_copy.isempty() is False

    assert ps_image._metadata.private != ps_image_copy._metadata.private
    assert ps_image._metadata.protected == ps_image_copy._metadata.protected
    assert ps_image._metadata.public == ps_image_copy._metadata.public

    if not ps_image.rgb.isempty():
        assert np.array_equal(ps_image.rgb[:], ps_image.rgb[:])
    assert np.array_equal(ps_image.gray[:], ps_image_copy.gray[:])
    assert np.array_equal(ps_image.enh_gray[:], ps_image_copy.enh_gray[:])
    assert np.array_equal(ps_image.objmask[:], ps_image_copy.objmask[:])
    assert np.array_equal(ps_image.objmap[:], ps_image_copy.objmap[:])


@timeit
def test_slicing(sample_image_array_with_imformat):
    input_image, input_imformat, true_imformat = sample_image_array_with_imformat
    ps_image = phenotypic.Image(arr=input_image)
    row_slice, col_slice = 10, 10
    sliced_ps_image = ps_image[:row_slice, :col_slice]
    if not ps_image.rgb.isempty():
        assert np.array_equal(sliced_ps_image.rgb[:], ps_image.rgb[:row_slice, :col_slice])
    assert np.array_equal(sliced_ps_image.gray[:], ps_image.gray[:row_slice, :col_slice])
    assert np.array_equal(sliced_ps_image.enh_gray[:], ps_image.enh_gray[:row_slice, :col_slice])
    assert np.array_equal(sliced_ps_image.objmask[:], ps_image.objmask[:row_slice, :col_slice])
    assert np.array_equal(sliced_ps_image.objmap[:], ps_image.objmap[:row_slice, :col_slice])


@timeit
def test_image_object_size_label_consistency(sample_image_array_with_imformat):
    input_image, input_imformat, true_imformat = sample_image_array_with_imformat
    ps_image = phenotypic.Image(arr=input_image)
    assert ps_image.num_objects == 0

    ps_image.objmap[:10, :10] = 1
    ps_image.objmap[-10:, -10:] = 2

    assert ps_image.num_objects == 2
    assert ps_image.num_objects == ps_image.objects.num_objects
    assert ps_image.num_objects == len(ps_image.objects.labels)


@timeit
def test_image_object_label_consistency_with_skimage(sample_image_array_with_imformat):
    input_image, input_imformat, true_imformat = sample_image_array_with_imformat
    ps_image = phenotypic.Image(arr=input_image)

    ps_image.objmap[:10, :10] = 1
    ps_image.objmap[-10:, -10:] = 2

    assert ps_image.objects.labels2series().equals(
            pd.Series(skimage.measure.regionprops_table(ps_image.objmap[:], properties=['label'])['label']),
    )


@timeit
def test_rgb_imsave_jpg(tmp_path):
    out = tmp_path/"out.jpg"

    image = phenotypic.data.load_colony(mode='Image')
    image.rgb.imsave(out)
    assert out.exists(), f"RGB JPEG file was not created at {out}"


@timeit
def test_rgb_imsave_png(tmp_path):
    out = tmp_path/"out.png"
    image = phenotypic.data.load_colony(mode='Image')
    image.rgb.imsave(out)
    assert out.exists(), f"RGB PNG file was not created at {out}"


@timeit
def test_rgb_imsave_tiff(tmp_path):
    out = tmp_path/"out.tiff"
    image = phenotypic.data.load_colony(mode='Image')
    image.rgb.imsave(out)
    assert out.exists(), f"RGB TIFF file was not created at {out}"


@timeit
def test_gray_imsave_jpg(tmp_path):
    out = tmp_path/"out_gray.jpg"
    image = phenotypic.data.load_colony(mode='Image')
    image.gray.imsave(out)
    assert out.exists(), f"Gray JPEG file was not created at {out}"


@timeit
def test_gray_imsave_png(tmp_path):
    out = tmp_path/"out_gray.png"
    image = phenotypic.data.load_colony(mode='Image')
    image.gray.imsave(out)
    assert out.exists(), f"Gray PNG file was not created at {out}"


@timeit
def test_gray_imsave_tiff(tmp_path):
    out = tmp_path/"out_gray.tiff"
    image = phenotypic.data.load_colony(mode='Image')
    image.gray.imsave(out)
    assert out.exists(), f"Gray TIFF file was not created at {out}"


@timeit
def test_enh_gray_imsave_jpg(tmp_path):
    out = tmp_path/"out_enh_gray.jpg"
    image = phenotypic.data.load_colony(mode='Image')
    image.enh_gray.imsave(out)
    assert out.exists(), f"Enhanced Gray JPEG file was not created at {out}"


@timeit
def test_enh_gray_imsave_png(tmp_path):
    out = tmp_path/"out_enh_gray.png"
    image = phenotypic.data.load_colony(mode='Image')
    image.enh_gray.imsave(out)
    assert out.exists(), f"Enhanced Gray PNG file was not created at {out}"


@timeit
def test_enh_gray_imsave_tiff(tmp_path):
    out = tmp_path/"out_enh_gray.tiff"
    image = phenotypic.data.load_colony(mode='Image')
    image.enh_gray.imsave(out)
    assert out.exists(), f"Enhanced Gray TIFF file was not created at {out}"
