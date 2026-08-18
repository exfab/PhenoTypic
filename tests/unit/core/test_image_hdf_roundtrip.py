"""Tests for the v2 grouped HDF5 round-trip layout.

Covers :meth:`Image.save2hdf5` / :meth:`Image.load_hdf5` and the
:class:`GridImage` overrides introduced with ``schema_version=2``. Each test
exercises a specific branch documented in
``/Users/alex/.claude/plans/please-help-me-update-typed-dove.md`` (Phase 1 /
1b).
"""

from __future__ import annotations

import warnings

import h5py
import numpy as np
import pytest

from phenotypic import Image, GridImage
from phenotypic.grid import AutoGridFinder, CenteredAutoGridFinder
from phenotypic.schema import IMAGE
from phenotypic.sdk_.constants_ import GAMMA_ENCODINGS

# Module-level slow marker: full HDF5 schema and back-compat matrix. The
# companion binary-roundtrip suite in tests/unit/core/ stays on the fast lane
# as the smoke check; this file moves to the nightly full lane.
pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_small_rgb(height: int = 24, width: int = 32, dtype=np.uint8) -> np.ndarray:
    """Return a deterministic tiny RGB array cheap enough to round-trip."""
    return np.zeros((height, width, 3), dtype=dtype)


def _make_small_gray(height: int = 24, width: int = 32, dtype=np.uint8) -> np.ndarray:
    """Return a deterministic tiny grayscale array."""
    return np.zeros((height, width), dtype=dtype)


# ===========================================================================
# 1. Core round-trip (Image)
# ===========================================================================


def test_image_roundtrip_bit_depth_16(tmp_path):
    """16-bit images round-trip with bit_depth and layer dtypes preserved."""
    rgb = np.zeros((16, 24, 3), dtype=np.uint16)
    img = Image(arr=rgb, name="sixteen", bit_depth=16)

    path = tmp_path / "bit16.h5"
    img.save2hdf5(path)
    loaded = Image.load_hdf5(path)

    assert loaded.bit_depth == 16
    assert loaded.rgb[:].dtype == np.uint16


def test_image_roundtrip_restores_name_from_constructor_placeholder(tmp_path):
    """A v2 HDF reload replaces the constructor's empty image-name placeholder."""
    img = Image(arr=_make_small_rgb(), name="plate_001")
    path = tmp_path / "named.h5"

    img.save2hdf5(path)
    loaded = Image.load_hdf5(path)

    assert loaded.name == "plate_001"


def test_image_roundtrip_illuminant_gamma(tmp_path):
    """D50 + LINEAR round-trip preserves illuminant and gamma identity."""
    img = Image(
        arr=_make_small_rgb(),
        name="illum_gamma",
        illuminant="D50",
        gamma=GAMMA_ENCODINGS.LINEAR,
    )

    path = tmp_path / "illum_gamma.h5"
    img.save2hdf5(path)
    loaded = Image.load_hdf5(path)

    # gamma round-trips as the GAMMA_ENCODINGS identity (via name lookup).
    assert loaded.gamma is GAMMA_ENCODINGS.LINEAR
    assert loaded.illuminant == "D50"


def test_image_roundtrip_imported_metadata(tmp_path):
    """Imported metadata JSON-encodes mixed types with fidelity."""
    img = Image(arr=_make_small_rgb(), name="imported_meta")

    # Populate imported metadata directly — normally filled by imread().
    img._metadata.imported.clear()
    img._metadata.imported.update({
        "exposure_count" : 7,                                   # int
        "gain"           : 2.5,                                 # float
        "shutter_tuple"  : (1, 200),                            # tuple -> list
        "auto_wb"        : True,                                # bool
        "focus_missing"  : None,                                # None
        "nested"         : {"make": "Acme", "model": "X1"},     # nested dict
        "captured_at"    : np.datetime64("2024-01-15"),         # datetime64
    })

    path = tmp_path / "imported.h5"
    img.save2hdf5(path)
    loaded = Image.load_hdf5(path)

    imported = loaded._metadata.imported
    assert imported["exposure_count"] == 7
    assert isinstance(imported["exposure_count"], int)
    assert imported["gain"] == 2.5
    assert isinstance(imported["gain"], float)
    # Tuples deserialise as lists because JSON has no tuple type — documented.
    assert imported["shutter_tuple"] == [1, 200]
    assert imported["auto_wb"] is True
    assert imported["focus_missing"] is None
    assert imported["nested"] == {"make": "Acme", "model": "X1"}
    # np.datetime64 stored via default=str -> ISO string form.
    assert imported["captured_at"] == "2024-01-15"


def test_image_roundtrip_mangled_legacy_values_fixed(tmp_path):
    """Regression: JSON encoder must not coerce digit-looking strings to int."""
    img = Image(arr=_make_small_rgb(), name="coercion_regression")
    img._metadata.public.update({
        "count": 5,
        "ratio": 0.5,
        "label": "123",  # v1 loader would have turned this into int 123.
    })

    path = tmp_path / "coercion.h5"
    img.save2hdf5(path)
    loaded = Image.load_hdf5(path)

    pub = loaded._metadata.public
    assert pub["count"] == 5
    assert isinstance(pub["count"], int)
    assert pub["ratio"] == 0.5
    assert isinstance(pub["ratio"], float)
    assert pub["label"] == "123"
    assert isinstance(pub["label"], str)


# ===========================================================================
# 2. Core round-trip (GridImage)
# ===========================================================================


def test_grid_image_roundtrip_dimensions(tmp_path):
    """GridImage nrows / ncols are preserved across round-trip."""
    grid_img = GridImage(arr=_make_small_rgb(64, 96), nrows=16, ncols=24)

    path = tmp_path / "grid_dims.h5"
    grid_img.save2hdf5(path)
    loaded = GridImage.load_hdf5(path)

    assert isinstance(loaded, GridImage)
    assert loaded.nrows == 16
    assert loaded.ncols == 24


def test_grid_image_roundtrip_custom_grid_finder(tmp_path):
    """A configured AutoGridFinder round-trips with matching class + params."""
    finder = AutoGridFinder(nrows=16, ncols=24, residual_fraction=0.4)
    grid_img = GridImage(arr=_make_small_rgb(64, 96), grid_finder=finder)

    path = tmp_path / "grid_finder.h5"
    grid_img.save2hdf5(path)
    loaded = GridImage.load_hdf5(path)

    assert isinstance(loaded.grid_finder, AutoGridFinder)
    assert loaded.grid_finder.nrows == 16
    assert loaded.grid_finder.ncols == 24
    assert loaded.grid_finder.residual_fraction == pytest.approx(0.4)


def test_gridimage_roundtrip_default_auto_grid_finder(tmp_path):
    """A GridImage built without an explicit finder round-trips cleanly."""
    grid_img = GridImage(arr=_make_small_rgb(64, 96))  # default CenteredAutoGridFinder

    path = tmp_path / "grid_default.h5"
    grid_img.save2hdf5(path)
    loaded = GridImage.load_hdf5(path)

    assert isinstance(loaded, GridImage)
    assert isinstance(loaded.grid_finder, CenteredAutoGridFinder)
    assert loaded.nrows == grid_img.nrows
    assert loaded.ncols == grid_img.ncols


def test_gridimage_idempotent_resave(tmp_path):
    """Writing the same path twice exercises ``del grid['grid_finder_json']``."""
    grid_img = GridImage(arr=_make_small_rgb(64, 96), nrows=8, ncols=12)
    path = tmp_path / "grid_idempotent.h5"

    grid_img.save2hdf5(path)
    # Second save must not raise (overwrites existing grid_finder_json dataset).
    grid_img.save2hdf5(path)

    loaded = GridImage.load_hdf5(path)
    assert loaded.nrows == 8
    assert loaded.ncols == 12


# ===========================================================================
# 3. Schema markers (direct h5py assertions)
# ===========================================================================


def test_schema_version_marker(tmp_path):
    """Layout and metadata namespace markers describe independent contracts."""
    from phenotypic._core._image_parts._image_io_handler import (
        _METADATA_SCHEMA_VERSION_ATTR,
        _METADATA_SCHEMA_VERSION_FLAT,
    )

    # --- Image ------------------------------------------------------------
    img = Image(arr=_make_small_rgb(), name="schema_img")
    img_path = tmp_path / "schema_img.h5"
    img.save2hdf5(img_path)

    with h5py.File(img_path, "r") as f:
        assert int(f.attrs["schema_version"]) == 2
        assert (
            int(f.attrs[_METADATA_SCHEMA_VERSION_ATTR])
            == _METADATA_SCHEMA_VERSION_FLAT
        )
        saved_class = f.attrs["phenotypic_class"]
        if isinstance(saved_class, bytes):
            saved_class = saved_class.decode()
        assert saved_class == "Image"

        assert "layers" in f
        assert "gray" in f["layers"]
        assert "detect_mat" in f["layers"]
        assert "objmap" in f["layers"]

        assert "metadata" in f
        for section in ("protected", "public", "imported"):
            assert section in f["metadata"]

    # --- GridImage --------------------------------------------------------
    grid_img = GridImage(arr=_make_small_rgb(64, 96), nrows=8, ncols=12)
    grid_path = tmp_path / "schema_grid.h5"
    grid_img.save2hdf5(grid_path)

    with h5py.File(grid_path, "r") as f:
        assert int(f.attrs["schema_version"]) == 2
        assert (
            int(f.attrs[_METADATA_SCHEMA_VERSION_ATTR])
            == _METADATA_SCHEMA_VERSION_FLAT
        )
        saved_class = f.attrs["phenotypic_class"]
        if isinstance(saved_class, bytes):
            saved_class = saved_class.decode()
        assert saved_class == "GridImage"

        assert "grid" in f
        assert "grid_finder_json" in f["grid"]
        assert int(f["grid"].attrs["nrows"]) == 8
        assert int(f["grid"].attrs["ncols"]) == 12


def test_metadata_schema_marker_belongs_to_image_owning_group(tmp_path):
    """Nested image writes mark their group without duplicating the root marker."""
    from phenotypic._core._image_parts._image_io_handler import (
        _METADATA_SCHEMA_VERSION_ATTR,
        _METADATA_SCHEMA_VERSION_FLAT,
    )

    img = Image(arr=_make_small_rgb(), name="nested")
    path = tmp_path / "nested_image_group.h5"

    with h5py.File(path, "w") as f:
        image_group = f.require_group("images/nested")
        img._save_image2hdfgroup(image_group)

        assert _METADATA_SCHEMA_VERSION_ATTR not in f.attrs
        assert (
            int(image_group.attrs[_METADATA_SCHEMA_VERSION_ATTR])
            == _METADATA_SCHEMA_VERSION_FLAT
        )
        assert int(image_group.attrs["schema_version"]) == 2


# ===========================================================================
# 4. Back-compat (v1 legacy layout)
# ===========================================================================


def test_back_compat_legacy_flat_hdf_loads(tmp_path):
    """A hand-built v1 file (no schema_version) still loads via Image.load_hdf5."""
    path = tmp_path / "legacy_v1.h5"

    rgb = np.zeros((12, 16, 3), dtype=np.uint8)
    gray = np.zeros((12, 16), dtype=np.float64)
    detect = np.zeros((12, 16), dtype=np.float64)
    objmap = np.zeros((12, 16), dtype=np.int32)

    with h5py.File(path, "w") as f:
        f.create_dataset("rgb", data=rgb)
        f.create_dataset("gray", data=gray)
        dm = f.create_dataset("detect_mat", data=detect)
        dm.attrs["detect_mode"] = "gray"
        f.create_dataset("objmap", data=objmap)

        prot = f.require_group("protected_metadata")
        prot.attrs["ImageName"] = "legacy_plate"

        pub = f.require_group("public_metadata")
        pub.attrs["experiment"] = "legacy_exp"
        # Note: no schema_version attribute anywhere.

    loaded = Image.load_hdf5(path)

    assert loaded.rgb[:].shape == (12, 16, 3)
    assert loaded._data.detect_mode == "gray"
    # The legacy bare framework key ("ImageName") is remapped on load to the
    # current Metadata_-prefixed key; the stale bare key does not survive.
    assert loaded._metadata.protected.get(IMAGE.IMAGE_NAME) == "legacy_plate"
    assert "ImageName" not in loaded._metadata.protected
    # Arbitrary, non-framework public keys are passed through untouched.
    assert loaded._metadata.public.get("experiment") == "legacy_exp"
    # Legacy files never persisted imported metadata.
    assert loaded._metadata.imported == {}


# ===========================================================================
# 5. Auto-dispatch
# ===========================================================================


def test_auto_dispatch_warning(tmp_path):
    """Loading a GridImage file through Image.load_hdf5 warns but succeeds."""
    grid_img = GridImage(arr=_make_small_rgb(64, 96), nrows=8, ncols=12)
    path = tmp_path / "dispatch.h5"
    grid_img.save2hdf5(path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = Image.load_hdf5(path)

    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert user_warnings, "Expected a UserWarning about GridImage dispatch"
    joined = " ".join(str(w.message) for w in user_warnings)
    assert "GridImage" in joined

    # Plain Image — not silently upcast.
    assert isinstance(loaded, Image)
    assert not isinstance(loaded, GridImage)


def test_no_auto_dispatch_warning_for_gridimage_load(tmp_path):
    """Loading a GridImage file via GridImage.load_hdf5 emits no dispatch warning."""
    grid_img = GridImage(arr=_make_small_rgb(64, 96), nrows=8, ncols=12)
    path = tmp_path / "dispatch_clean.h5"
    grid_img.save2hdf5(path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = GridImage.load_hdf5(path)

    dispatch_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and "GridImage" in str(w.message)
        and "use GridImage.load_hdf5" in str(w.message)
    ]
    assert dispatch_warnings == []
    assert isinstance(loaded, GridImage)


# ===========================================================================
# 6. Edge cases
# ===========================================================================


def test_roundtrip_gray_only_no_rgb(tmp_path):
    """Grayscale-only images take the ``'rgb' not in layers`` branch."""
    gray = np.zeros((16, 24), dtype=np.uint8)
    img = Image(arr=gray, name="gray_only")
    assert img.rgb.isempty()  # guard against fixture drift

    path = tmp_path / "gray_only.h5"
    img.save2hdf5(path)

    # Confirm writer did NOT store an rgb dataset.
    with h5py.File(path, "r") as f:
        assert "rgb" not in f["layers"]

    loaded = Image.load_hdf5(path)
    assert loaded.rgb.isempty()
    assert np.array_equal(loaded.gray[:], img.gray[:])


def test_v2_file_missing_imported_section(tmp_path):
    """Remove ``/metadata/imported`` post-save and verify graceful load."""
    img = Image(arr=_make_small_rgb(), name="no_imported")
    path = tmp_path / "no_imported.h5"
    img.save2hdf5(path)

    with h5py.File(path, "r+") as f:
        del f["metadata"]["imported"]

    loaded = Image.load_hdf5(path)
    assert loaded._metadata.imported == {}


def test_gridimage_corrupt_grid_finder_json_falls_back(tmp_path):
    """Corrupted grid_finder_json emits a warning but loader still succeeds."""
    grid_img = GridImage(arr=_make_small_rgb(64, 96), nrows=8, ncols=12)
    path = tmp_path / "grid_corrupt.h5"
    grid_img.save2hdf5(path)

    # Overwrite the grid_finder_json dataset with invalid JSON.
    with h5py.File(path, "r+") as f:
        del f["grid"]["grid_finder_json"]
        f["grid"].create_dataset(
            "grid_finder_json",
            data="{not json",
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = GridImage.load_hdf5(path)

    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert user_warnings, "Expected a UserWarning about GridFinder deserialization"
    joined = " ".join(str(w.message) for w in user_warnings).lower()
    assert "gridfinder" in joined or "deserialization" in joined

    # Still a usable GridImage with the default CenteredAutoGridFinder.
    assert isinstance(loaded, GridImage)
    assert isinstance(loaded.grid_finder, CenteredAutoGridFinder)


def test_gridimage_missing_grid_subgroup_via_gridimage_loader(tmp_path):
    """Plain Image HDF loaded via GridImage.load_hdf5 falls back to defaults."""
    img = Image(arr=_make_small_rgb(64, 96), name="plain_for_grid")
    path = tmp_path / "plain_via_grid.h5"
    img.save2hdf5(path)

    # save2hdf5 on a plain Image does NOT write /grid/.
    with h5py.File(path, "r") as f:
        assert "grid" not in f

    with warnings.catch_warnings():
        # Filter "use GridImage.load_hdf5" noise — the caller already IS using it.
        warnings.simplefilter("always")
        loaded = GridImage.load_hdf5(path)

    assert isinstance(loaded, GridImage)
    assert isinstance(loaded.grid_finder, CenteredAutoGridFinder)


def test_unknown_gamma_name_falls_back_to_string(tmp_path):
    """A v2 file with an unrecognised gamma name loads without crashing."""
    img = Image(arr=_make_small_rgb(), name="unknown_gamma")
    path = tmp_path / "unknown_gamma.h5"
    img.save2hdf5(path)

    # Mutate the root gamma attribute to a name that isn't in GAMMA_ENCODINGS.
    with h5py.File(path, "r+") as f:
        f.attrs["gamma"] = "weird"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = Image.load_hdf5(path)

    # Loader must coerce unknown names to the constructor-safe SRGB default.
    assert loaded.gamma is GAMMA_ENCODINGS.SRGB
    gamma_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and "weird" in str(w.message)
        and "GAMMA_ENCODINGS.SRGB" in str(w.message)
    ]
    assert gamma_warnings, "Expected a UserWarning naming the unknown gamma"


def test_kwargs_override_root_attrs(tmp_path):
    """Caller-supplied kwargs win over root-attr defaults via ``setdefault``."""
    rgb = np.zeros((16, 24, 3), dtype=np.uint16)
    img = Image(arr=rgb, name="kwargs_win", bit_depth=16)
    path = tmp_path / "kwargs_win.h5"
    img.save2hdf5(path)

    loaded = Image.load_hdf5(path, bit_depth=8)
    assert loaded.bit_depth == 8
