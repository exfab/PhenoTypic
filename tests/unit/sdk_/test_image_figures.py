"""figures/ group + attributes.phenotypic.figures descriptor (spec §1)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from phenotypic.sdk_ import ngff_
from phenotypic.sdk_._image_figures import (
    StoredFigureBinding,
    StoredFigureFailure,
    StoredFigureFile,
    StoredFigurePage,
    StoredFigures,
    apply_image_figures_attributes,
    read_image_figures_descriptor,
    write_image_figures,
)

_JSON = b'{"data": []}'
_PNG = b"\x89PNG fake"


def _stored() -> StoredFigures:
    page = StoredFigurePage(
        key="default", label=None, backend="plotly", metadata={"plate": 1},
        files=(
            StoredFigureFile("plotly-json", "application/vnd.plotly.v1+json",
                             "default.plotly.json", _JSON),
            StoredFigureFile("png", "image/png", "default.png", _PNG),
        ),
    )
    return StoredFigures(
        bindings=(StoredFigureBinding("sym", "MeasureSymZones", "sym", (page,)),),
        failed=(StoredFigureFailure("orient", None, None, "RuntimeError: boom"),),
    )


def test_writer_lays_out_groups_files_and_a_hash_bound_descriptor(tmp_path: Path):
    fragment = write_image_figures(tmp_path, _stored())
    group = json.loads((tmp_path / "figures" / "zarr.json").read_text(encoding="utf-8"))
    assert group == {"zarr_format": 3, "node_type": "group", "attributes": {}}
    binding_group = (tmp_path / "figures" / "sym" / "zarr.json").read_text(encoding="utf-8")
    assert json.loads(binding_group) == group
    assert (tmp_path / "figures/sym/default.plotly.json").read_bytes() == _JSON

    descriptor = fragment[ngff_.PhenotypicAttr.FIGURES]
    assert descriptor["schema_version"] == 1
    assert descriptor["bindings"]["sym"]["class"] == "MeasureSymZones"
    page = descriptor["bindings"]["sym"]["pages"][0]
    assert page["metadata"] == {"plate": 1}
    assert [f["format"] for f in page["files"]] == ["plotly-json", "png"]
    for entry in page["files"]:
        data = (tmp_path / entry["path"]).read_bytes()
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()
    assert descriptor["failed"] == [
        {"binding": "orient", "page": None, "format": None, "error": "RuntimeError: boom"}
    ]


def test_all_failed_writes_the_key_with_empty_bindings(tmp_path: Path):
    stored = StoredFigures(bindings=(), failed=_stored().failed)
    descriptor = write_image_figures(tmp_path, stored)[ngff_.PhenotypicAttr.FIGURES]
    assert descriptor["bindings"] == {}
    assert (tmp_path / "figures" / "zarr.json").is_file()


def test_apply_sets_and_removes_the_key():
    phenotypic: dict = {"figures": {"stale": True}}
    apply_image_figures_attributes(phenotypic, None)
    assert "figures" not in phenotypic
    apply_image_figures_attributes(phenotypic, {"figures": {"schema_version": 1}})
    assert phenotypic["figures"] == {"schema_version": 1}


def test_reader_returns_none_for_a_pre_feature_store(tmp_path: Path):
    (tmp_path / "zarr.json").write_text(json.dumps(
        {"zarr_format": 3, "node_type": "group",
         "attributes": {"phenotypic": {"store_schema_version": 3}}}
    ), encoding="utf-8")
    assert read_image_figures_descriptor(tmp_path) is None
