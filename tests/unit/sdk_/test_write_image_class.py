"""write_image_class=False is the only thing that omits image_class."""

from __future__ import annotations

import json
from pathlib import Path

from phenotypic import Image
from phenotypic.data import load_synth_yeast_plate
from phenotypic.sdk_ import ngff_
from phenotypic.sdk_.ngff_ import PhenotypicAttr


def _block(store: Path) -> dict:
    payload = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    return payload["attributes"][PhenotypicAttr.ROOT]


def test_save2zarr_still_writes_image_class(tmp_path: Path) -> None:
    """The default is unchanged; only the process-mode caller opts out."""
    img = Image(load_synth_yeast_plate())
    store = img.save2zarr(tmp_path / "bundle.ome.zarr")
    assert _block(store)[PhenotypicAttr.IMAGE_CLASS] == "Image"


def test_save_store_can_omit_image_class(tmp_path: Path) -> None:
    img = Image(load_synth_yeast_plate())
    store = img._save_store(
        tmp_path / "processed.ome.zarr",
        series=("gray",),
        write_objmap=False,
        levels=ngff_.pyramid_level_count(*img.gray[:].shape[:2]),
        work_id=None,
        durable=False,
        write_image_class=False,
    )
    block = _block(store)
    assert PhenotypicAttr.IMAGE_CLASS not in block
    # Everything else the store needs is still there.
    assert block[PhenotypicAttr.STORE_SCHEMA_VERSION] == ngff_.STORE_SCHEMA_VERSION
    assert block[PhenotypicAttr.SERIES] == {"gray": "gray"}
    assert PhenotypicAttr.LABELS not in block


def test_consolidation_is_off_by_default(tmp_path: Path) -> None:
    """Every existing caller keeps today's unconsolidated store."""
    img = Image(load_synth_yeast_plate())
    store = img.save2zarr(tmp_path / "bundle.ome.zarr")
    root = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    assert "consolidated_metadata" not in root


def test_consolidate_writes_the_block_at_the_final_path(tmp_path: Path) -> None:
    """It is produced inside the .part, so the promoted root already has it.

    The promote is a directory rename, so a root ``zarr.json`` that exists at
    the final path is by construction the complete one -- which is the property
    consolidating *after* the promote would destroy.
    """
    img = Image(load_synth_yeast_plate())
    store = img._save_store(
        tmp_path / "consolidated.ome.zarr",
        series=("rgb",),
        write_objmap=False,
        levels=ngff_.pyramid_level_count(*img.rgb[:].shape[:2]),
        work_id=None,
        durable=False,
        consolidate=True,
    )
    root = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    assert root["consolidated_metadata"]["must_understand"] is False
    assert sorted(root["consolidated_metadata"]["metadata"]) == [
        "OME", "rgb", "rgb/0", "rgb/1",
    ]
    # A sibling of `attributes`, so the phenotypic block is untouched.
    assert "consolidated_metadata" not in root["attributes"]
    assert ngff_.read_root_attributes(store)[PhenotypicAttr.ROOT][
        PhenotypicAttr.SERIES
    ] == {"rgb": "rgb"}


def test_consolidation_leaves_no_stray_part(tmp_path: Path) -> None:
    """The consolidated root is the promoted one, not a second write."""
    img = Image(load_synth_yeast_plate())
    target = tmp_path / "clean.ome.zarr"
    img._save_store(
        target,
        series=("rgb",),
        write_objmap=False,
        levels=2,
        work_id=None,
        durable=False,
        consolidate=True,
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["clean.ome.zarr"]
