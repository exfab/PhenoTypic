"""Every written store must validate against the vendored NGFF schemas."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from phenotypic import GridImage, Image
from phenotypic.data import load_synth_yeast_plate
from tests._ngff_conformance import assert_store_conforms


def test_a_written_image_store_conforms(tmp_path: Path) -> None:
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    assert_store_conforms(store)


def test_a_written_grid_store_conforms(tmp_path: Path) -> None:
    store = GridImage(load_synth_yeast_plate(), nrows=8, ncols=12).save2zarr(
        tmp_path / "g.ome.zarr"
    )
    assert_store_conforms(store)


def test_an_rgb_less_store_conforms(tmp_path: Path) -> None:
    image = Image(np.asarray(Image(load_synth_yeast_plate()).gray[:]))
    assert_store_conforms(image.save2zarr(tmp_path / "gray.ome.zarr"))


def test_a_single_level_node_preview_store_conforms(tmp_path: Path) -> None:
    """The levels=1 path (builder node previews) must conform too."""
    store = Image(load_synth_yeast_plate()).save_intermediate_zarr(
        tmp_path / "n.ome.zarr", layers=("gray", "detect_mat", "objmap")
    )
    assert_store_conforms(store)


def test_a_remote_ref_resolves_from_the_vendored_copy(tmp_path: Path) -> None:
    """An unregistered $ref raises Unresolvable, which is NOT a ValidationError,
    so the suite would error rather than fail. Verified: all three schemas
    reference _version.schema by absolute URL."""
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    assert_store_conforms(store)  # would raise Unresolvable without the registry


def test_a_wrong_version_string_is_rejected(tmp_path: Path) -> None:
    """Proves the resolved _version.schema enum is actually enforced."""
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    group = store / "gray" / "zarr.json"
    payload = json.loads(group.read_text(encoding="utf-8"))
    payload["attributes"]["ome"]["version"] = "0.4"
    group.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError):
        assert_store_conforms(store)


def test_partial_omero_is_rejected(tmp_path: Path) -> None:
    """Proves the gate has teeth: the exact defect an earlier draft would ship.

    Note what is and is not schema-enforced here. `$defs/omero` requires only
    `channels`, and the channel item has no `required` list -- so a channel
    missing `color` would validate. But `window`, *if present*, requires all
    four of start/min/end/max, which is what this truncation violates. Emitting
    the complete block is PhenoTypic policy (asserted in Phase 1 Task 1.4);
    this test covers the part the schema does enforce.
    """
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    # `rgb`, NOT `gray` (ledger C2). P2/ALGO-2 omits `omero` from every
    # FLOAT series and `gray` is float32, so `build_omero` returns {} for
    # it -- indexing ["omero"] below raised KeyError in the test body,
    # before `assert_store_conforms` was ever reached. `rgb` is the only
    # series that ever carries the block.
    group = store / "rgb" / "zarr.json"
    payload = json.loads(group.read_text(encoding="utf-8"))
    payload["attributes"]["ome"]["omero"]["channels"][0]["window"] = {"max": 255}
    group.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError):
        assert_store_conforms(store)


def test_missing_image_label_is_rejected(tmp_path: Path) -> None:
    """label.schema requires image-label even though the prose says SHOULD."""
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    label_json = store / "rgb" / "labels" / "objmap" / "zarr.json"
    payload = json.loads(label_json.read_text(encoding="utf-8"))
    del payload["attributes"]["ome"]["image-label"]
    label_json.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError):
        assert_store_conforms(store)


def test_a_reordered_series_list_is_rejected(tmp_path: Path) -> None:
    """2.2.3: path order MUST match the Image element order.

    Without this the path-order assertion is satisfied vacuously by the four
    positive tests -- which is exactly how a KeyError in it shipped green once
    already (ledger GEN-47 / GEN-33).
    """
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    ome_json = store / "OME" / "zarr.json"
    payload = json.loads(ome_json.read_text(encoding="utf-8"))
    payload["attributes"]["ome"]["series"].reverse()
    ome_json.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError, match="series order"):
        assert_store_conforms(store)


def test_a_dangling_dataset_path_is_rejected(tmp_path: Path) -> None:
    """A reader follows datasets[].path; a dangling one is a broken store."""
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    group = store / "gray" / "zarr.json"
    payload = json.loads(group.read_text(encoding="utf-8"))
    payload["attributes"]["ome"]["multiscales"][0]["datasets"][0]["path"] = "9"
    group.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Exception):  # zarr raises before our assert
        assert_store_conforms(store)


def test_a_dimension_names_mismatch_is_rejected(tmp_path: Path) -> None:
    """2.1 MUST -- and the only other assertion of it checks the builder."""
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    level = store / "gray" / "0" / "zarr.json"
    payload = json.loads(level.read_text(encoding="utf-8"))
    payload["dimension_names"] = list(reversed(payload["dimension_names"]))
    level.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError):
        assert_store_conforms(store)


def test_a_nested_chunk_key_separator_is_rejected(tmp_path: Path) -> None:
    """Design spec 1.4: the separator MUST be uniform store-wide."""
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    level = store / "gray" / "0" / "zarr.json"
    payload = json.loads(level.read_text(encoding="utf-8"))
    payload["chunk_key_encoding"] = {
        "name": "default",
        "configuration": {"separator": "/"},
    }
    level.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError, match="separator"):
        assert_store_conforms(store)


def test_a_label_less_store_still_conforms(tmp_path: Path) -> None:
    """save_intermediate_zarr(layers=("gray",)) writes one, and it is VALID.

    The reader-level fold turned this from tolerated into FileNotFoundError
    once already (ledger GEN-33).
    """
    image = Image(load_synth_yeast_plate())
    store = image.save_intermediate_zarr(tmp_path / "i.ome.zarr", layers=("gray",))
    assert_store_conforms(store)


def test_missing_series_is_rejected(tmp_path: Path) -> None:
    """ome.schema requires series."""
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    ome_json = store / "OME" / "zarr.json"
    payload = json.loads(ome_json.read_text(encoding="utf-8"))
    del payload["attributes"]["ome"]["series"]
    ome_json.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError):
        assert_store_conforms(store)
