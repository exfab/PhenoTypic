"""``load_bacteria_plate_full`` fetches the bacterial sample plate through pooch.

The tests swap the download for a small 16-bit TIFF, so they check the
loader's wiring (file name, return modes, grid shape) without the network.
The real file's contents are pinned by the SHA256 in the pooch registry.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
import tifffile

from phenotypic import GridImage, Image
from phenotypic.data import _sample_image_data, load_bacteria_plate_full
from phenotypic.data._remote_registry import _REGISTRY

FILENAME = "BacteriaFullPlate.tif"


@pytest.fixture
def fetched(tmp_path, monkeypatch) -> list[str]:
    """Replace the download with a tiny 16-bit RGB TIFF; record requested names."""
    path = tmp_path / FILENAME
    pixels = np.linspace(0, 65535, 48 * 64 * 3).astype(np.uint16).reshape(48, 64, 3)
    tifffile.imwrite(path, pixels, photometric="rgb")
    requested: list[str] = []

    def fake_fetch(filename: str) -> str:
        requested.append(filename)
        return str(path)

    monkeypatch.setattr(_sample_image_data, "fetch_snp", fake_fetch)
    return requested


def test_registry_pins_the_plate_by_sha256():
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", _REGISTRY.registry[FILENAME])


def test_default_is_an_8_by_12_grid_image(fetched):
    plate = load_bacteria_plate_full()
    assert isinstance(plate, GridImage)
    assert (plate.nrows, plate.ncols) == (8, 12)
    assert fetched == [FILENAME]


def test_image_mode_is_a_plain_image(fetched):
    plate = load_bacteria_plate_full(mode="Image")
    assert type(plate) is Image


def test_array_mode_keeps_16_bits(fetched):
    pixels = load_bacteria_plate_full(mode="array")
    assert pixels.dtype == np.uint16
    assert pixels.shape == (48, 64, 3)


def test_image_mode_keeps_16_bits(fetched):
    plate = load_bacteria_plate_full(mode="Image")
    assert plate.bit_depth == 16
    assert plate.rgb[:].dtype == np.uint16


def test_filepath_mode_returns_the_cached_path(fetched):
    assert load_bacteria_plate_full(mode="filepath").endswith(FILENAME)
