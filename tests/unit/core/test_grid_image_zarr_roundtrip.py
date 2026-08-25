"""GridImage grid state must round-trip through ``attributes.phenotypic.grid``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from phenotypic import GridImage
from phenotypic.data import load_synth_yeast_plate
from phenotypic.grid import AutoGridFinder
from phenotypic.sdk_.ngff_ import PhenotypicAttr, read_phenotypic_attributes


@pytest.fixture
def pixels() -> np.ndarray:
    """Raw RGB pixels.

    ``load_synth_yeast_plate()`` returns a **GridImage**, and
    ``GridImage.__init__`` takes ``arr.grid_finder`` in preference to both
    ``grid_finder=`` and ``nrows=``/``ncols=`` when the input has one. Passing
    the bare array is the only way for this fixture to control the grid.
    """
    return np.asarray(load_synth_yeast_plate().rgb[:])


@pytest.fixture
def grid(pixels: np.ndarray) -> GridImage:
    return GridImage(pixels, nrows=16, ncols=24)


def test_grid_dimensions_round_trip(grid: GridImage, tmp_path: Path) -> None:
    assert (grid.nrows, grid.ncols) == (16, 24)
    store = grid.save2zarr(tmp_path / "g.ome.zarr")
    back = GridImage.load_zarr(store)
    assert (back.nrows, back.ncols) == (16, 24)


def test_grid_dimensions_are_recorded_in_the_block(
    grid: GridImage, tmp_path: Path
) -> None:
    store = grid.save2zarr(tmp_path / "g.ome.zarr")
    block = read_phenotypic_attributes(store)[PhenotypicAttr.GRID]
    assert block["nrows"] == 16
    assert block["ncols"] == 24


def test_grid_finder_round_trips_by_class_and_params(
    pixels: np.ndarray, tmp_path: Path
) -> None:
    grid = GridImage(pixels, grid_finder=AutoGridFinder(nrows=8, ncols=12))
    assert type(grid.grid_finder).__name__ == "AutoGridFinder"
    store = grid.save2zarr(tmp_path / "g.ome.zarr")
    back = GridImage.load_zarr(store)
    assert type(back.grid_finder).__name__ == "AutoGridFinder"
    assert (back.grid_finder.nrows, back.grid_finder.ncols) == (8, 12)


def test_grid_finder_non_default_params_survive(
    pixels: np.ndarray, tmp_path: Path
) -> None:
    """Class and dimensions alone would pass with every other param dropped."""
    finder = AutoGridFinder(nrows=8, ncols=12, residual_fraction=0.4)
    store = GridImage(pixels, grid_finder=finder).save2zarr(
        tmp_path / "g.ome.zarr"
    )
    assert GridImage.load_zarr(store).grid_finder.residual_fraction == 0.4


def test_grid_block_lives_under_phenotypic_not_ome(
    grid: GridImage, tmp_path: Path
) -> None:
    store = grid.save2zarr(tmp_path / "g.ome.zarr")
    root = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    assert PhenotypicAttr.GRID in root["attributes"]["phenotypic"]
    assert "plate" not in json.dumps(root["attributes"]["ome"])


def test_no_hcs_plate_metadata_is_emitted(grid: GridImage, tmp_path: Path) -> None:
    """HCS would need one image group per well: 16x24 = 384 groups, no benefit."""
    store = grid.save2zarr(tmp_path / "g.ome.zarr")
    groups = [p.name for p in store.iterdir() if p.is_dir()]
    assert set(groups) <= {"OME", "rgb", "gray", "detect_mat"}


def test_corrupt_grid_finder_warns_and_falls_back(
    grid: GridImage, tmp_path: Path
) -> None:
    store = grid.save2zarr(tmp_path / "g.ome.zarr")
    root_path = store / "zarr.json"
    payload = json.loads(root_path.read_text(encoding="utf-8"))
    payload["attributes"]["phenotypic"][PhenotypicAttr.GRID]["grid_finder"] = {
        "class": "NoSuchFinder",
        "params": {},
    }
    root_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.warns(UserWarning, match="GridFinder"):
        back = GridImage.load_zarr(store)
    assert back.grid_finder is not None
    # The stored nrows/ncols still apply -- only the finder payload was lost.
    assert (back.nrows, back.ncols) == (16, 24)


def test_explicit_nrows_lose_to_a_stored_grid_finder(
    grid: GridImage, tmp_path: Path
) -> None:
    """Pins the behaviour the plan's own implementation produces.

    Phase 2 Task 2.3 carries a test named
    ``test_explicit_kwargs_take_priority_over_stored_grid`` asserting
    ``(8, 12)`` here. The ``setdefault`` implementation it ships beside cannot
    produce that: ``GridImage.__init__`` takes ``grid_finder`` in preference to
    ``nrows``/``ncols``, so a restored finder wins over both. The HDF path this
    task mirrors behaves identically -- verified by execution:
    ``GridImage.load_hdf5(path, nrows=8, ncols=12)`` on a 16x24 file returns
    ``(16, 24)``. Raised for a ruling; the implementation follows the plan.
    """
    store = grid.save2zarr(tmp_path / "g.ome.zarr")
    back = GridImage.load_zarr(store, nrows=8, ncols=12)
    assert (back.nrows, back.ncols) == (16, 24)


def test_explicit_nrows_apply_when_the_store_carries_no_finder(
    grid: GridImage, tmp_path: Path
) -> None:
    """With no stored finder, the setdefault path does give kwargs priority."""
    store = grid.save2zarr(tmp_path / "g.ome.zarr")
    root_path = store / "zarr.json"
    payload = json.loads(root_path.read_text(encoding="utf-8"))
    del payload["attributes"]["phenotypic"][PhenotypicAttr.GRID]["grid_finder"]
    root_path.write_text(json.dumps(payload), encoding="utf-8")
    back = GridImage.load_zarr(store, nrows=8, ncols=12)
    assert (back.nrows, back.ncols) == (8, 12)


def test_an_explicit_grid_finder_kwarg_beats_the_stored_one(
    grid: GridImage, tmp_path: Path
) -> None:
    store = grid.save2zarr(tmp_path / "g.ome.zarr")
    back = GridImage.load_zarr(store, grid_finder=AutoGridFinder(nrows=4, ncols=6))
    assert type(back.grid_finder).__name__ == "AutoGridFinder"
    assert (back.nrows, back.ncols) == (4, 6)


def test_grid_explicit_name_and_bit_depth_override_stored_protected_metadata(
    grid: GridImage, tmp_path: Path
) -> None:
    """Grid loading shares the base loader's protected-metadata precedence."""
    source = GridImage(
        np.asarray(grid.rgb[:]).astype(np.uint16) * 257,
        name="stored_grid_A01",
        nrows=grid.nrows,
        ncols=grid.ncols,
        bit_depth=16,
    )
    store = source.save2zarr(tmp_path / "g.ome.zarr")

    back = GridImage.load_zarr(store, name="override_grid_B02", bit_depth=8)

    assert back.name == "override_grid_B02"
    assert back.bit_depth == 8


def test_image_class_records_gridimage(grid: GridImage, tmp_path: Path) -> None:
    store = grid.save2zarr(tmp_path / "g.ome.zarr")
    block = read_phenotypic_attributes(store)
    assert block[PhenotypicAttr.IMAGE_CLASS] == "GridImage"


def test_load_image_from_store_dispatches_to_gridimage(
    grid: GridImage, tmp_path: Path
) -> None:
    from phenotypic.sdk_ import load_image_from_store

    store = grid.save2zarr(tmp_path / "g.ome.zarr")
    back = load_image_from_store(store)
    assert type(back) is GridImage
    assert (back.nrows, back.ncols) == (16, 24)


def test_grid_load_zarr_does_not_warn_about_grid_state(
    grid: GridImage, tmp_path: Path
) -> None:
    import warnings

    store = grid.save2zarr(tmp_path / "g.ome.zarr")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert type(GridImage.load_zarr(store)) is GridImage


def test_grid_image_layers_round_trip_bit_exact(
    grid: GridImage, tmp_path: Path
) -> None:
    """The subclass override must not shadow the base loader's array work."""
    store = grid.save2zarr(tmp_path / "g.ome.zarr")
    back = GridImage.load_zarr(store)
    np.testing.assert_array_equal(back.rgb[:], grid.rgb[:])
    np.testing.assert_array_equal(back.gray[:], grid.gray[:])
    np.testing.assert_array_equal(back.detect_mat[:], grid.detect_mat[:])
    np.testing.assert_array_equal(back.objmap[:], grid.objmap[:])


def test_image_type_metadata_stays_grid(grid: GridImage, tmp_path: Path) -> None:
    """``image_class`` and ``Metadata_ImageType`` are distinct fields."""
    store = grid.save2zarr(tmp_path / "g.ome.zarr")
    block = read_phenotypic_attributes(store)
    assert (
        block[PhenotypicAttr.METADATA]["protected"]["Metadata_ImageType"]
        == "GridImage"
    )
    assert (
        GridImage.load_zarr(store)._metadata.protected["Metadata_ImageType"]
        == "GridImage"
    )


def test_load_image_from_store_dispatches_to_gridimage_on_a_non_grid_image_type(
    grid: GridImage, tmp_path: Path
) -> None:
    """The discriminating direction for the sibling test above.

    There, ``image_class`` and ``Metadata_ImageType`` both read ``GridImage``,
    so dispatching on either field -- or on an enum member that merely happens
    to spell the same string -- passes. Here they disagree: the store is a real
    GridImage whose user-visible type is ``GridSection``.
    """
    from phenotypic.sdk_ import load_image_from_store

    grid._metadata.protected["Metadata_ImageType"] = "GridSection"
    store = grid.save2zarr(tmp_path / "g.ome.zarr")
    block = read_phenotypic_attributes(store)
    assert block[PhenotypicAttr.IMAGE_CLASS] == "GridImage"
    assert block[PhenotypicAttr.METADATA]["protected"]["Metadata_ImageType"] == (
        "GridSection"
    )
    back = load_image_from_store(store)
    assert type(back) is GridImage
    assert (back.nrows, back.ncols) == (16, 24)
    assert back._metadata.protected["Metadata_ImageType"] == "GridSection"
