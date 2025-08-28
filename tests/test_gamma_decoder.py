import pytest
import numpy as np

import phenotypic
from phenotypic.correction import GammaDecoder
from phenotypic.data import load_plate_12hr

from .resources.TestHelper import timeit
from .test_fixtures import sample_image_array_with_imformat


@timeit
def test_gamma_decoder_basic():
    """Test basic functionality of GammaDecoder."""
    # Create an image
    image = phenotypic.Image(load_plate_12hr())

    # Apply GammaDecoder
    decoder = GammaDecoder()
    result = decoder.apply(image)

    # Check that the result is an Image
    assert isinstance(result, phenotypic.Image)

    # Check that the image is not empty
    assert not result.isempty()

    # Check that the _known_gamma_decoding flag is set
    assert result._known_gamma_decoding


@pytest.mark.parametrize("color_profile", ["sRGB", "ITU-R BT.709"])
@timeit
def test_gamma_decoder_color_profiles(color_profile):
    """Test GammaDecoder with different color profiles."""
    # Create an image and set the color profile
    image = phenotypic.Image(load_plate_12hr(), color_profile=color_profile)

    # Apply GammaDecoder
    decoder = GammaDecoder()
    result = decoder.apply(image)

    # Check that the result is an Image
    assert isinstance(result, phenotypic.Image)

    # Check that the image is not empty
    assert not result.isempty()

    # Check that the _known_gamma_decoding flag is set
    assert result._known_gamma_decoding

    # Check that the color profile is preserved
    assert result.color_profile == color_profile


@timeit
def test_gamma_decoder_custom_kwargs():
    """Test GammaDecoder with custom kwargs."""
    # Create an image
    image = phenotypic.Image(load_plate_12hr())

    # Apply GammaDecoder with custom kwargs
    decoder = GammaDecoder(function="ITU-R BT.709")
    result = decoder.apply(image)

    # Check that the result is an Image
    assert isinstance(result, phenotypic.Image)

    # Check that the image is not empty
    assert not result.isempty()

    # Check that the _known_gamma_decoding flag is set
    assert result._known_gamma_decoding


@timeit
def test_gamma_decoder_inplace():
    """Test GammaDecoder with inplace=True."""
    # Create an image
    image = phenotypic.Image(load_plate_12hr())

    # Apply GammaDecoder with inplace=True
    decoder = GammaDecoder()
    result = decoder.apply(image, inplace=True)

    # Check that the result is the same object as the input
    assert result is image

    # Check that the _known_gamma_decoding flag is set
    assert result._known_gamma_decoding


@timeit
def test_gamma_decoder_with_fixture(sample_image_array_with_imformat):
    """Test GammaDecoder with the sample_image_array_with_imformat fixture."""
    # Create an image
    array, input_imformat, true_imformat = sample_image_array_with_imformat
    image = phenotypic.Image(array, imformat=input_imformat)

    # Apply GammaDecoder
    decoder = GammaDecoder()
    if image.imformat.is_matrix() is False:
        result = decoder.apply(image)

        # Check that the result is an Image
        assert isinstance(result, phenotypic.Image)

        # Check that the image is not empty
        assert not result.isempty()

        # Check that the _known_gamma_decoding flag is set
        assert result._known_gamma_decoding

        # Check that the image format is preserved
        # Convert enum to string without the enum class name
