"""load_layer_hdf5 reads a single layer from v2 and legacy-flat HDFs."""
import numpy as np
import pytest
from phenotypic import Image


def _make_rgb(h=32, w=48):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_load_layer_from_v2(tmp_path):
    img = Image(arr=_make_rgb())
    path = tmp_path / "v2.h5"
    img.save2hdf5(path)

    rgb = Image.load_layer_hdf5(path, "rgb")
    gray = Image.load_layer_hdf5(path, "gray")
    assert rgb.shape == (32, 48, 3)
    assert gray.shape == (32, 48)


def test_load_layer_from_legacy_flat(tmp_path):
    img = Image(arr=_make_rgb())
    path = tmp_path / "flat.h5"
    img.save_intermediate_layers(path, layers=("rgb", "gray"))

    rgb = Image.load_layer_hdf5(path, "rgb")
    assert rgb.shape == (32, 48, 3)


def test_missing_layer_raises(tmp_path):
    img = Image(arr=np.zeros((16, 16), dtype=np.uint8))  # gray-only, no rgb
    path = tmp_path / "gray_only.h5"
    img.save2hdf5(path)
    with pytest.raises(KeyError):
        Image.load_layer_hdf5(path, "rgb")
