"""load_layer_hdf5 reads a single layer from v2 and legacy-flat HDFs."""
import h5py
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
    from phenotypic._core._image_parts._image_io_handler import (
        _METADATA_SCHEMA_VERSION_ATTR,
        _METADATA_SCHEMA_VERSION_FLAT,
    )

    # The legacy-flat layout is written HERE, not produced by a writer.
    #
    # This used to call `img.save_intermediate_layers(path, layers=("rgb","gray"))`,
    # which Phase 2 Task 2.4 removed in favour of `save_intermediate_zarr`. That
    # left this test as the only consumer of a deleted method -- and it was the
    # only remaining producer of the legacy-flat layout anywhere in the tree, so
    # `load_layer_hdf5`'s legacy-flat READ path would have lost its coverage
    # entirely until Phase 6 deletes this file.
    #
    # Writing the bytes directly is the better pin regardless: it states the
    # on-disk format this reader must keep understanding, instead of inheriting
    # it from a writer that no longer exists. Phase 5 migration depends on that
    # read path.
    img = Image(arr=_make_rgb())
    path = tmp_path / "flat.h5"
    with h5py.File(path, mode="w") as f:
        f.create_dataset("rgb", data=img.rgb[:])
        f.create_dataset("gray", data=img.gray[:])
        f.attrs[_METADATA_SCHEMA_VERSION_ATTR] = _METADATA_SCHEMA_VERSION_FLAT

    rgb = Image.load_layer_hdf5(path, "rgb")
    assert rgb.shape == (32, 48, 3)
    with h5py.File(path, "r") as f:
        assert "schema_version" not in f.attrs
        assert (
            int(f.attrs[_METADATA_SCHEMA_VERSION_ATTR])
            == _METADATA_SCHEMA_VERSION_FLAT
        )


def test_missing_layer_raises(tmp_path):
    img = Image(arr=np.zeros((16, 16), dtype=np.uint8))  # gray-only, no rgb
    path = tmp_path / "gray_only.h5"
    img.save2hdf5(path)
    with pytest.raises(KeyError):
        Image.load_layer_hdf5(path, "rgb")
