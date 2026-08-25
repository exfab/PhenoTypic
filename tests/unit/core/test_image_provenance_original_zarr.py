"""Provenance and retained-source data live in the root OME-Zarr store."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr

from phenotypic import Image
from phenotypic.abc_ import ImageCorrector
from phenotypic.enhance import BlurGauss
from phenotypic.sdk_.ngff_ import (
    PhenotypicAttr,
    STORE_SCHEMA_VERSION,
    read_phenotypic_attributes,
)
from tests._ngff_conformance import assert_store_conforms


def _root_payload(store: Path) -> dict:
    return json.loads((store / "zarr.json").read_text(encoding="utf-8"))


class _ZeroPixelsCorrector(ImageCorrector):
    """Replace processed pixels while retained decoded pixels stay untouched."""

    def _operate(self, image: Image) -> Image:
        image.set_image(np.zeros_like(image.rgb[:]))
        return image


def test_operation_journal_round_trips_only_through_root_phenotypic_namespace(
    tmp_path: Path,
) -> None:
    source = Image(np.arange(48 * 32 * 3, dtype=np.uint8).reshape(48, 32, 3))
    result = BlurGauss(sigma=1.25).apply(source)
    result._metadata.provenance_journal.update(
        {
            "status": "in_progress",
            "pipeline": {
                "source_path": "/resolved/pipeline.json",
                "sha256": "a" * 64,
            },
            "retry_base_length": 0,
        }
    )

    store = result.save2zarr(tmp_path / "journal.ome.zarr")

    root = _root_payload(store)
    stored = root["attributes"][PhenotypicAttr.ROOT]["provenance"]
    assert stored == result._metadata.provenance_journal
    assert stored["schema_version"] == 1
    assert stored["status"] == "in_progress"
    assert "provenance" not in root["attributes"]["ome"]
    assert STORE_SCHEMA_VERSION == 3
    for group_json in store.glob("*/zarr.json"):
        payload = json.loads(group_json.read_text(encoding="utf-8"))
        assert "provenance" not in payload.get("attributes", {}).get("ome", {})

    restored = Image.load_zarr(store)
    assert restored._metadata.provenance_journal == stored
    assert [entry["operation_name"] for entry in restored.provenance] == [
        "BlurGauss"
    ]


def test_direct_save_does_not_invent_an_original_series(tmp_path: Path) -> None:
    image = Image(np.zeros((48, 32, 3), dtype=np.uint8))

    store = image.save2zarr(tmp_path / "direct.ome.zarr")

    assert "original" not in read_phenotypic_attributes(store)[
        PhenotypicAttr.SERIES
    ]
    assert "original" not in _root_payload(store)["attributes"]["ome"]
    assert not (store / "original").exists()


def test_retained_rgb_original_is_a_full_registered_pyramid_and_repromotes(
    tmp_path: Path,
) -> None:
    pixels = np.arange(1025 * 7 * 3, dtype=np.uint16).reshape(1025, 7, 3)
    image = Image(pixels)
    image._retain_original()
    processed = _ZeroPixelsCorrector().apply(image)

    store = processed.save2zarr(tmp_path / "with-original.ome.zarr")

    block = read_phenotypic_attributes(store)
    declared = _root_payload(store / "OME")["attributes"]["ome"]["series"]
    assert declared == ["rgb", "gray", "detect_mat", "original"]
    assert list(block[PhenotypicAttr.SERIES].values()) == declared
    levels = block[PhenotypicAttr.PYRAMID]["levels"]
    assert levels > 1
    assert sorted(
        path.name for path in (store / "original").iterdir() if path.name.isdigit()
    ) == [str(index) for index in range(levels)]

    level0 = np.asarray(zarr.open_array(store=str(store / "original" / "0"), mode="r"))
    np.testing.assert_array_equal(np.moveaxis(level0, 0, -1), pixels)
    processed_level0 = np.asarray(
        zarr.open_array(store=str(store / "rgb" / "0"), mode="r")
    )
    assert not np.array_equal(np.moveaxis(processed_level0, 0, -1), pixels)
    level0_meta = _root_payload(store / "original" / "0")
    original_ome = _root_payload(store / "original")["attributes"]["ome"]
    assert level0_meta["dimension_names"] == ["c", "y", "x"]
    assert [axis["name"] for axis in original_ome["multiscales"][0]["axes"]] == [
        "c",
        "y",
        "x",
    ]
    assert [channel["label"] for channel in original_ome["omero"]["channels"]] == [
        "R",
        "G",
        "B",
    ]
    xml = (store / "OME" / "METADATA.ome.xml").read_text(encoding="utf-8")
    assert xml.count("<Image ") == 4
    assert 'Name="original"' in xml
    assert_store_conforms(store)

    restored = Image.load_zarr(store)
    promoted = restored.save2zarr(tmp_path / "repromoted.ome.zarr")
    promoted_original = np.asarray(
        zarr.open_array(store=str(promoted / "original" / "0"), mode="r")
    )
    np.testing.assert_array_equal(np.moveaxis(promoted_original, 0, -1), pixels)
    assert_store_conforms(promoted)
