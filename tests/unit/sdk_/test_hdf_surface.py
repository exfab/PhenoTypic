"""Pins the HDF keeper list so a future cleanup cannot delete past it."""

from __future__ import annotations

import pytest

from phenotypic.sdk_.hdf_ import HDF, _clear_hdf_consistency_flags, _open_hdf_with_recovery

KEEPERS = [
    "safe_writer",
    "swmr_writer",
    "strict_writer",
    "swmr_reader",
    "reader",
    "get_group",
    "save_array2hdf5",
]

REMOVED = [
    "preallocate_series_layout",
    "save_series_new",
    "save_series_update",
    "save_series_append",
    "load_series",
    "preallocate_frame_layout",
    "save_frame_new",
    "save_frame_update",
    "save_frame_append",
    "load_frame",
    "assert_swmr_on",
    "get_uncompressed_sizes_for_group",
    "close_handle",
]


@pytest.mark.parametrize("name", KEEPERS)
def test_keeper_survives(name: str) -> None:
    assert hasattr(HDF, name), (
        f"{name} has live callers; deleting it breaks test_hdf_open_recovery.py "
        "or the legacy-fixture generator."
    )


@pytest.mark.parametrize("name", REMOVED)
def test_dead_dataframe_layer_is_gone(name: str) -> None:
    assert not hasattr(HDF, name)


def test_recovery_helpers_survive() -> None:
    assert callable(_open_hdf_with_recovery)
    assert callable(_clear_hdf_consistency_flags)
