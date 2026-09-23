"""figures/ rides the root-last transaction (spec §1, §3 step 2, measure mode)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

from phenotypic import Image
from phenotypic._cli._embedded_measurement_tables import prepare_image_tables
from phenotypic.data import load_synth_yeast_plate
from phenotypic.sdk_ import ngff_
from phenotypic.sdk_._image_figures import (
    StoredFigureBinding,
    StoredFigureFile,
    StoredFigurePage,
    StoredFigures,
    read_image_figures_descriptor,
)
from phenotypic.sdk_._measurement_tables import replace_image_tables


def _figures(tag: bytes = b"one", binding: str = "sym") -> StoredFigures:
    page = StoredFigurePage("default", None, "plotly", {}, (
        StoredFigureFile("plotly-json", "application/vnd.plotly.v1+json",
                         "default.plotly.json", tag),
    ))
    return StoredFigures((StoredFigureBinding(binding, "X", binding, (page,)),), ())


def _tables():
    return prepare_image_tables(pd.DataFrame({"Object_Label": [1]}), None)


@pytest.fixture(scope="module")
def plate() -> Image:
    return Image(load_synth_yeast_plate())


def test_save2zarr_writes_figures_and_an_independent_reader_opens_them(tmp_path, plate):
    import zarr

    store = plate.save2zarr(tmp_path / "p.ome.zarr", figures=_figures())
    descriptor = read_image_figures_descriptor(store)
    assert descriptor["bindings"]["sym"]["pages"][0]["files"][0]["path"] == (
        "figures/sym/default.plotly.json"
    )
    root = zarr.open_group(str(store), mode="r")
    assert isinstance(root["figures"], zarr.Group)
    assert isinstance(root["figures/sym"], zarr.Group)
    ome = json.loads((store / "OME" / "zarr.json").read_text(encoding="utf-8"))
    assert "figures" not in ome["attributes"]["ome"]["series"]
    assert "figures" not in (store / "OME" / "METADATA.ome.xml").read_text(
        encoding="utf-8"
    )


def test_no_figures_means_no_key_and_no_group(tmp_path, plate):
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    assert read_image_figures_descriptor(store) is None
    assert not (store / "figures").exists()


def test_process_writer_carries_figures_inside_the_consolidated_store(tmp_path, plate):
    from phenotypic._cli._cli_process_only import write_process_only_layer

    out = tmp_path / "p.ome.zarr"
    write_process_only_layer(plate, "rgb", out, fmt="zarr", figures=_figures())
    assert (out / "figures/sym/default.plotly.json").read_bytes() == b"one"
    root = json.loads((out / "zarr.json").read_text(encoding="utf-8"))
    assert "figures/sym" in root["consolidated_metadata"]["metadata"]


def test_measure_rebuild_replaces_the_group_and_keeps_pixels_linked(tmp_path, plate):
    store = plate.save2zarr(tmp_path / "p.ome.zarr", figures=_figures(b"old", "gone"))
    pixel = next(
        p for p in (store / "rgb" / "0").rglob("*") if p.is_file() and p.name != "zarr.json"
    )
    pixel_inode = pixel.stat().st_ino
    replace_image_tables(
        store, _tables(), objmap_target=ngff_.objmap_path("rgb"),
        figures=_figures(b"new", "kept"),
    )
    assert not (store / "figures/gone").exists()
    assert (store / "figures/kept/default.plotly.json").read_bytes() == b"new"
    assert list(read_image_figures_descriptor(store)["bindings"]) == ["kept"]
    assert pixel.stat().st_ino == pixel_inode


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="holds a descriptor across a directory rename, which Windows refuses",
)
def test_a_same_name_rebuild_never_writes_through_into_the_live_store(tmp_path, plate):
    """Spec §5: the part's copies are hard links; the new bytes must be new files."""
    store = plate.save2zarr(tmp_path / "p.ome.zarr", figures=_figures(b"old-bytes", "sym"))
    held = os.open(store / "figures/sym/default.plotly.json", os.O_RDONLY)
    try:
        replace_image_tables(
            store, _tables(), objmap_target=ngff_.objmap_path("rgb"),
            figures=_figures(b"new-bytes", "sym"),
        )
        os.lseek(held, 0, os.SEEK_SET)
        assert os.read(held, 32) == b"old-bytes"
    finally:
        os.close(held)
    assert (store / "figures/sym/default.plotly.json").read_bytes() == b"new-bytes"


def test_measure_rebuild_with_no_bindings_removes_key_and_group(tmp_path, plate):
    store = plate.save2zarr(tmp_path / "p.ome.zarr", figures=_figures())
    replace_image_tables(
        store, _tables(), objmap_target=ngff_.objmap_path("rgb"), figures=None
    )
    assert read_image_figures_descriptor(store) is None
    assert not (store / "figures").exists()


def test_measure_rebuild_writes_figures_before_the_root_is_promoted(
    tmp_path, plate, monkeypatch
):
    """Spec §5: judged at the promote, not from the final tree."""
    import hashlib

    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    real_promote = ngff_.promote_store
    seen = {}

    def _spy(part, final, **kwargs):
        root = json.loads((Path(part) / "zarr.json").read_text(encoding="utf-8"))
        entry = root["attributes"]["phenotypic"]["figures"]["bindings"]["sym"]["pages"][0]["files"][0]
        data = (Path(part) / entry["path"]).read_bytes()
        seen["match"] = hashlib.sha256(data).hexdigest() == entry["sha256"]
        return real_promote(part, final, **kwargs)

    monkeypatch.setattr(ngff_, "promote_store", _spy)
    replace_image_tables(
        store, _tables(), objmap_target=ngff_.objmap_path("rgb"), figures=_figures()
    )
    assert seen == {"match": True}


def test_migrates_table_replace_leaves_figures_untouched(tmp_path, plate):
    from phenotypic.sdk_._measurement_tables import replace_embedded_measurement_table

    store = plate.save2zarr(tmp_path / "p.ome.zarr", figures=_figures())
    before = read_image_figures_descriptor(store)
    replace_embedded_measurement_table(
        store, _tables().measurements_payload(), objmap_target=ngff_.objmap_path("rgb")
    )
    assert read_image_figures_descriptor(store) == before
    assert (store / "figures/sym/default.plotly.json").read_bytes() == b"one"
