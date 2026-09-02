"""load_zarr refuses a store that is not a PhenoTypic run bundle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phenotypic import Image
from phenotypic.data import load_synth_yeast_plate
from tests._process_stores import write_process_store


def _processed_store(tmp_path: Path) -> Path:
    return write_process_store(
        tmp_path / "processed.ome.zarr", Image(load_synth_yeast_plate())
    )


def test_load_zarr_refuses_a_processed_store(tmp_path: Path) -> None:
    store = _processed_store(tmp_path)
    with pytest.raises(ValueError, match="image_class"):
        Image.load_zarr(store)


def test_the_error_names_imread_as_the_remedy(tmp_path: Path) -> None:
    """A user who hits this must be told what to call instead."""
    store = _processed_store(tmp_path)
    with pytest.raises(ValueError, match="imread"):
        Image.load_zarr(store)


def test_load_zarr_does_not_return_a_degraded_image(tmp_path: Path) -> None:
    """Regression: load_zarr must not hand back an object for this store.

    The plan predicted a *degraded Image* here -- spec 3.3 claims
    ``_load_from_store`` reads every field with a defaulting ``.get()``.
    Verified false: it subscripts the series mapping bare, at
    ``series["gray"]`` and ``series["detect_mat"]``
    (``_image_io_handler.py:1511,1521``), so before the guard this raised
    ``KeyError: 'detect_mat'`` -- an obscure error rather than a plausible
    wrong object. The guard is what turns either outcome into one that names
    the remedy, so this test accepts only ``ValueError`` and fails on a
    returned object.
    """
    store = _processed_store(tmp_path)
    try:
        result = Image.load_zarr(store)
    except ValueError:
        return
    pytest.fail(
        f"load_zarr returned {type(result).__name__} with "
        f"num_objects={result.num_objects} instead of raising"
    )


def test_a_third_party_store_raises_the_same_guard(tmp_path: Path) -> None:
    """No phenotypic block at all: a clear ValueError, not a bare KeyError.

    ValueError specifically, and not the wider `(ValueError, KeyError)` an
    earlier draft allowed: KeyError is exactly what this test exists to rule
    out, so accepting it would let the guard be placed after
    `require_readable_store`, where it never fires.
    """
    store = tmp_path / "foreign.ome.zarr"
    store.mkdir()
    (store / "zarr.json").write_text(
        json.dumps({
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {"ome": {"version": "0.5"}},
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="imread"):
        Image.load_zarr(store)


def test_a_missing_store_still_raises_file_not_found(tmp_path: Path) -> None:
    """The guard must not turn an absent store into a 'not a bundle' error.

    `read_root_attributes` raises FileNotFoundError on a store with no root
    zarr.json, which is the normal 'interrupted write' signal every staged
    caller depends on. Reading the root before the guard must not swallow it.
    """
    with pytest.raises(FileNotFoundError):
        Image.load_zarr(tmp_path / "absent.ome.zarr")


def test_a_bundle_store_still_loads(tmp_path: Path) -> None:
    """The guard must not fire on the path it is not for."""
    img = Image(load_synth_yeast_plate())
    store = img.save2zarr(tmp_path / "bundle.ome.zarr")
    assert Image.load_zarr(store).gray[:].shape == img.gray[:].shape
