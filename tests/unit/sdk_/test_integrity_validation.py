from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from phenotypic import Image
import phenotypic.settings as settings
from phenotypic.abc_ import ImageEnhancer, ObjectDetector
from phenotypic.enhance import BlurGauss
from phenotypic.sdk_.exceptions_ import OperationIntegrityError


class StaticMaskDetector(ObjectDetector):
    """Detector fixture that writes a fixed mask without reading RGB."""

    mask: ClassVar[np.ndarray] = np.zeros((5, 5), dtype=bool)

    def _operate(self, image: Image) -> Image:
        image.objmask[:] = self.mask
        return image


class RgbMutatingEnhancer(ImageEnhancer):
    """Invalid enhancer fixture that mutates protected RGB data."""

    def _operate(self, image: Image) -> Image:
        image._data.rgb.setflags(write=True)
        image._data.rgb[0, 0, 0] = 255 - image._data.rgb[0, 0, 0]
        return image


class RgbCreatingEnhancer(ImageEnhancer):
    """Invalid enhancer fixture that creates RGB data on grayscale input."""

    def _operate(self, image: Image) -> Image:
        image._data.rgb = np.zeros((*image.gray[:].shape, 3), dtype=np.uint8)
        return image


class RgbRemovingEnhancer(ImageEnhancer):
    """Invalid enhancer fixture that removes existing RGB data."""

    def _operate(self, image: Image) -> Image:
        image._data.rgb = np.empty((0, 3), dtype=np.uint8)
        return image


def test_enhancer_validation_allows_grayscale_image_without_rgb() -> None:
    """Validation treats absent grayscale RGB as stable, not as a type error."""
    image = Image(arr=np.zeros((5, 5), dtype=np.uint8))

    with settings.validation(True):
        result = BlurGauss(sigma=1.0, mode="constant").apply(image)

    assert result.rgb.isempty()
    assert result.detect_mat[:].shape == (5, 5)


def test_detector_validation_allows_grayscale_image_without_rgb() -> None:
    """ObjectDetector integrity checks accept grayscale-only images."""
    image = Image(arr=np.zeros((5, 5), dtype=np.uint8))
    StaticMaskDetector.mask = np.eye(5, dtype=bool)

    with settings.validation(True):
        result = StaticMaskDetector().apply(image)

    assert result.rgb.isempty()
    np.testing.assert_array_equal(result.objmask[:], StaticMaskDetector.mask)


def test_measure_validation_allows_grayscale_image_without_rgb() -> None:
    """Measure integrity checks accept a missing RGB target on grayscale input."""
    from phenotypic.sdk_.funcs_ import validate_measure_integrity

    @validate_measure_integrity("image.rgb")
    def inspect_image(image: Image) -> None:
        return None

    image = Image(arr=np.zeros((5, 5), dtype=np.uint8))

    with settings.validation(True):
        inspect_image(image)

    assert image.rgb.isempty()


def test_rgb_mutation_still_fails_for_rgb_images() -> None:
    """Validation still catches protected RGB mutation when RGB exists."""
    rgb = np.zeros((5, 5, 3), dtype=np.uint8)
    image = Image(arr=rgb)

    with settings.validation(True), pytest.raises(OperationIntegrityError):
        RgbMutatingEnhancer().apply(image)


def test_absent_rgb_created_by_operation_fails_validation() -> None:
    """Validation catches absent-to-present RGB state changes."""
    image = Image(arr=np.zeros((5, 5), dtype=np.uint8))

    with settings.validation(True), pytest.raises(OperationIntegrityError):
        RgbCreatingEnhancer().apply(image)


def test_existing_rgb_removed_by_operation_fails_validation() -> None:
    """Validation catches present-to-absent RGB state changes."""
    image = Image(arr=np.zeros((5, 5, 3), dtype=np.uint8))

    with settings.validation(True), pytest.raises(OperationIntegrityError):
        RgbRemovingEnhancer().apply(image)
