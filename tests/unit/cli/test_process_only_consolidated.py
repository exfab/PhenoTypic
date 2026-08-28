"""Consolidated metadata: one GET to open a store, and safely ignorable."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from phenotypic import Image
from phenotypic.data import load_synth_yeast_plate
from phenotypic.sdk_ import ngff_
from phenotypic.sdk_.ngff_ import PhenotypicAttr
from phenotypic._cli._cli_process_only import write_process_only_layer


def _store(tmp_path: Path) -> Path:
    img = Image(load_synth_yeast_plate())
    out = tmp_path / "p01.ome.zarr"
    write_process_only_layer(img, "rgb", out, fmt="zarr")
    return out


def _root(store: Path) -> dict:
    return json.loads((store / "zarr.json").read_text(encoding="utf-8"))


def test_the_store_is_consolidated(tmp_path: Path) -> None:
    root = _root(_store(tmp_path))
    assert "consolidated_metadata" in root
    assert root["consolidated_metadata"]["metadata"]


def test_consolidation_is_marked_safely_ignorable(tmp_path: Path) -> None:
    """Zarr v3 requires readers to FAIL on an unknown key without this."""
    root = _root(_store(tmp_path))
    assert root["consolidated_metadata"]["must_understand"] is False


def test_the_phenotypic_block_survives_consolidation(tmp_path: Path) -> None:
    """It is a sibling of `attributes`, not nested inside it."""
    store = _store(tmp_path)
    root = _root(store)
    assert "consolidated_metadata" not in root["attributes"]
    block = root["attributes"][PhenotypicAttr.ROOT]
    assert block[PhenotypicAttr.STORE_SCHEMA_VERSION] == ngff_.STORE_SCHEMA_VERSION
    assert block[PhenotypicAttr.SERIES] == {"rgb": "rgb"}
    # And the existing reader still finds it unchanged.
    assert ngff_.read_root_attributes(store)[PhenotypicAttr.ROOT] == block


def test_per_node_metadata_still_exists(tmp_path: Path) -> None:
    """A reader that ignores the key must still be able to walk the tree."""
    store = _store(tmp_path)
    assert (store / "rgb" / "zarr.json").is_file()
    assert (store / "rgb" / "0" / "zarr.json").is_file()
    assert (store / "OME" / "zarr.json").is_file()


def _file_count(store: Path) -> int:
    return len([p for p in store.rglob("*") if p.is_file()])


def test_consolidation_adds_no_files(tmp_path: Path) -> None:
    """Same count as the SAME store written unconsolidated. One fewer GET.

    An earlier version re-asserted `4 + 2 * levels` and stopped there, which
    is the arithmetic `test_a_single_series_store_is_four_files_plus_two_per_
    level` already pins. Repeating it cannot distinguish "consolidation adds
    no files" from "a store has 8 files": both readings are green while only
    one is the claim. The comparison is against an unconsolidated store of
    the same image, so the difference measured is consolidation itself.
    """
    img = Image(load_synth_yeast_plate())
    plain = img._save_store(
        tmp_path / "plain.ome.zarr",
        series=("rgb",),
        write_objmap=False,
        levels=ngff_.pyramid_level_count(*img.shape[:2]),
        work_id=None,
        durable=False,
        write_image_class=False,
        consolidate=False,
    )
    assert "consolidated_metadata" not in _root(plain), (
        "the control store must actually be unconsolidated"
    )

    consolidated = _store(tmp_path)
    assert "consolidated_metadata" in _root(consolidated)
    assert _file_count(consolidated) == _file_count(plain)


def test_a_consolidated_store_still_round_trips_through_imread(
    tmp_path: Path,
) -> None:
    img = Image(load_synth_yeast_plate())
    out = tmp_path / "p01.ome.zarr"
    write_process_only_layer(img, "rgb", out, fmt="zarr")
    assert np.array_equal(Image.imread(out).rgb[:], img.rgb[:])


def test_no_warning_escapes_the_writer(tmp_path: Path, recwarn) -> None:
    """TWO ZarrUserWarnings fire, not one, and both must be caught.

    zarr 3.1.5 emits `Consolidated metadata is currently not part in the Zarr
    format 3 specification` AND `Object at METADATA.ome.xml is not recognized
    as a component of a Zarr hierarchy` -- the latter once per image, which at
    AutoConvertRaw scale is tens of thousands of lines of log. A `message=`
    filter naming only consolidation catches one of them; filtering on the
    ZarrUserWarning class catches both.
    """
    from zarr.errors import ZarrUserWarning

    _store(tmp_path)
    assert [w for w in recwarn if issubclass(w.category, ZarrUserWarning)] == []
