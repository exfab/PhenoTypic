"""Pipeline identity and operation checkpoints are durable at the root."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from phenotypic import Image, ImagePipeline
from phenotypic.enhance import BlurGauss
from phenotypic._core._provenance import (
    provenance_success_sink,
    set_provenance_status,
    write_provenance_checkpoint,
)


def _pixels() -> np.ndarray:
    return np.arange(40 * 32 * 3, dtype=np.uint8).reshape(40, 32, 3)


def _journal(store: Path) -> dict:
    payload = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    return payload["attributes"]["phenotypic"]["provenance"]


def test_pipeline_loaded_from_file_sets_resolved_source_identity(tmp_path: Path) -> None:
    pipeline = ImagePipeline(ops={"smooth": BlurGauss(sigma=1.0)})
    source = tmp_path / "nested" / "pipeline.json"
    source.parent.mkdir()
    source.write_text(pipeline.to_json() or "", encoding="utf-8")

    loaded = ImagePipeline.from_json(source.parent / ".." / "nested" / "pipeline.json")
    result = loaded.apply(Image(_pixels()))

    identity = result._metadata.provenance_journal["pipeline"]
    assert identity == {
        "source_path": str(source.resolve()),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    assert result.provenance[0]["pipeline_step_path"] == ["smooth"]


def test_journal_only_checkpoint_and_success_sink_publish_ordered_prefix(
    tmp_path: Path,
) -> None:
    image = Image(_pixels())
    image._metadata.provenance_journal["status"] = "in_progress"
    store = tmp_path / "checkpoint.ome.zarr"

    write_provenance_checkpoint(store, image, journal_only=True)

    payload = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    assert payload == {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "phenotypic": {"provenance": image._metadata.provenance_journal}
        },
    }
    assert not (store / "gray").exists()

    with provenance_success_sink(
        lambda updated: write_provenance_checkpoint(store, updated)
    ):
        BlurGauss(sigma=1.0).apply(image, inplace=True)

    assert [entry["operation_name"] for entry in _journal(store)["operations"]] == [
        "BlurGauss"
    ]
    assert _journal(store)["status"] == "in_progress"

    set_provenance_status(image, "failed")
    write_provenance_checkpoint(store, image)
    assert _journal(store)["status"] == "failed"


def test_success_sink_updates_only_root_attributes_on_a_full_store(
    tmp_path: Path,
) -> None:
    image = Image(_pixels())
    image._metadata.provenance_journal["status"] = "in_progress"
    store = image.save2zarr(tmp_path / "full.ome.zarr")
    ome_before = (store / "OME" / "zarr.json").read_bytes()

    with provenance_success_sink(
        lambda updated: write_provenance_checkpoint(store, updated)
    ):
        BlurGauss(sigma=1.0).apply(image, inplace=True)

    assert len(_journal(store)["operations"]) == 1
    assert (store / "OME" / "zarr.json").read_bytes() == ome_before
