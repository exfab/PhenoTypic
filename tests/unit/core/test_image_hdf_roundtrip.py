"""The public HDF image API is gone; the private migration readers remain.

Phase 6 of the OME-Zarr store change removes ``save2hdf5`` / ``load_hdf5`` /
``load_layer_hdf5`` / ``save_intermediate_layers`` from :class:`Image` and
:class:`GridImage`. What used to be this file's v2-grouped round-trip matrix
is now covered from the other side: ``tests/unit/core/test_image_store_*``
pins the store round trip, and the legacy layouts this file used to write by
hand are frozen as golden fixtures under ``tests/fixtures/legacy_hdf/`` and
read back through ``--mode migrate``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from phenotypic import GridImage, Image


@pytest.mark.parametrize(
    "name", ["save2hdf5", "load_hdf5", "load_layer_hdf5", "save_intermediate_layers"]
)
@pytest.mark.parametrize("cls", [Image, GridImage])
def test_public_hdf_api_is_removed(cls, name: str) -> None:
    assert not hasattr(cls, name)


@pytest.mark.parametrize(
    "name", ["_load_v2_grouped", "_load_legacy_flat_group", "_load_hdf5_for_migration"]
)
def test_private_migration_readers_survive(name: str) -> None:
    """--mode migrate is built on these; Phase 5 breaks without them."""
    assert hasattr(Image, name)


def test_migration_can_still_read_every_golden_fixture(tmp_path) -> None:
    from phenotypic.sdk_._hdf_to_zarr import migrate_hdf_to_zarr
    from phenotypic.sdk_.ngff_ import valid_staged_store

    fixtures = Path(__file__).resolve().parents[2] / "fixtures" / "legacy_hdf"
    for layout in (
        "v1_flat",
        "v2_grouped",
        "v2_enh_gray",
        "v2_grid",
        "v2_image_type",
        "v2_work_id",
    ):
        store = migrate_hdf_to_zarr(
            fixtures / layout / "img.h5", tmp_path / f"{layout}.ome.zarr"
        )
        assert valid_staged_store(store) is True
