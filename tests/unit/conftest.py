"""Session-scoped shared fixtures for unit tests.

These fixtures load expensive image data once per session and share it across
tests.  Tests that need a mutable copy should call ``.copy()`` on the fixture.
"""

import pytest

import phenotypic
from phenotypic.data import load_synth_yeast_plate, load_plate_12hr


@pytest.fixture(scope="session")
def synth_plate():
    """Pre-loaded synth yeast plate (not detected)."""
    return load_synth_yeast_plate()


@pytest.fixture(scope="session")
def synth_plate_detected():
    """Synth plate with OtsuDetector already applied."""
    from phenotypic.detect import OtsuDetector

    image = load_synth_yeast_plate()
    OtsuDetector().apply(image, inplace=True)
    return image


@pytest.fixture(scope="session")
def synth_grid_image():
    """Synth plate wrapped as GridImage."""
    return phenotypic.GridImage(load_synth_yeast_plate())


@pytest.fixture(scope="session")
def plate_12hr_grid_image():
    """12hr plate as GridImage."""
    return phenotypic.GridImage(load_plate_12hr())


@pytest.fixture(scope="session")
def colony_image():
    """Colony image for imsave tests."""
    return phenotypic.data.load_colony(mode="Image")
