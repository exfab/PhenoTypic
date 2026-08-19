"""The attributes.phenotypic block is the sole source of truth on read."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phenotypic.sdk_ import ngff_
from phenotypic.sdk_.ngff_ import PhenotypicAttr


def _sections() -> dict[str, dict]:
    return {
        "protected": {
            "Metadata_ImageName": "plate_01",
            "Metadata_ImageType": "Grid",
            "Metadata_BitDepth": 16,
        },
        "public": {"Metadata_Strain": "BY4741"},
        "imported": {"TIFF:XResolution": 300.0},
    }


def test_primary_series_prefers_rgb() -> None:
    assert ngff_.primary_series(["rgb", "gray", "detect_mat"]) == "rgb"


def test_primary_series_falls_back_to_gray() -> None:
    assert ngff_.primary_series(["gray", "detect_mat"]) == "gray"


def test_objmap_path_is_relative_to_the_primary_series() -> None:
    assert ngff_.objmap_path("gray") == "gray/labels/objmap"
    assert ngff_.objmap_path("rgb") == "rgb/labels/objmap"


def test_series_and_labels_are_separate_keys() -> None:
    block = ngff_.build_phenotypic_attributes(
        image_class="GridImage",
        series_names=["rgb", "gray", "detect_mat"],
        pyramid_levels=4,
        metadata_sections=_sections(),
        detect_mode="gray",
        illuminant="D65",
        gamma="sRGB",
        grid={"nrows": 8, "ncols": 12, "grid_finder": {"class": "X", "params": {}}},
        work_id="w-1",
    )
    assert set(block[PhenotypicAttr.SERIES]) == {"rgb", "gray", "detect_mat"}
    assert block[PhenotypicAttr.LABELS] == {"objmap": "rgb/labels/objmap"}
    assert PhenotypicAttr.SERIES != PhenotypicAttr.LABELS


def test_has_labels_false_omits_the_labels_key_entirely() -> None:
    """Not an empty mapping -- absent. Ledger C3.

    A builder preview store (`save_intermediate_zarr`) writes no objmap. An
    earlier draft emitted `labels` unconditionally, so the preview DECLARED
    `labels.objmap = "gray/labels/objmap"` for a group that was never written
    and `assert_store_conforms` FileNotFoundError'd walking to it. The
    downstream guard added for that tested for an EMPTY mapping, so emitting
    `{}` here would leave the same defect in place with the guard looking
    correct. Absence is the contract.
    """
    block = ngff_.build_phenotypic_attributes(
        image_class="Image",
        series_names=["gray"],
        pyramid_levels=1,
        metadata_sections=_sections(),
        detect_mode="gray",
        illuminant=None,
        gamma=None,
        has_labels=False,
    )
    assert PhenotypicAttr.LABELS not in block


def test_has_labels_defaults_to_true_so_existing_callers_are_unchanged() -> None:
    block = ngff_.build_phenotypic_attributes(
        image_class="Image",
        series_names=["gray", "detect_mat"],
        pyramid_levels=1,
        metadata_sections=_sections(),
        detect_mode="gray",
        illuminant=None,
        gamma=None,
    )
    assert block[PhenotypicAttr.LABELS] == {"objmap": "gray/labels/objmap"}


def test_two_version_markers_are_both_present_and_distinct() -> None:
    block = ngff_.build_phenotypic_attributes(
        image_class="Image",
        series_names=["gray", "detect_mat"],
        pyramid_levels=1,
        metadata_sections=_sections(),
        detect_mode="gray",
        illuminant=None,
        gamma=None,
    )
    assert block[PhenotypicAttr.STORE_SCHEMA_VERSION] == 3
    assert block[PhenotypicAttr.METADATA_SCHEMA_VERSION] == 2


def test_image_class_and_image_type_stay_distinct() -> None:
    """A GridSection is not a GridImage; collapsing them loses information."""
    sections = _sections()
    sections["protected"]["Metadata_ImageType"] = "GridSection"
    block = ngff_.build_phenotypic_attributes(
        image_class="Image",
        series_names=["gray", "detect_mat"],
        pyramid_levels=1,
        metadata_sections=sections,
        detect_mode="gray",
        illuminant=None,
        gamma=None,
    )
    assert block[PhenotypicAttr.IMAGE_CLASS] == "Image"
    assert (
        block[PhenotypicAttr.METADATA]["protected"]["Metadata_ImageType"]
        == "GridSection"
    )


def test_downsample_methods_pins_the_actual_values() -> None:
    """One literal assertion, or nothing pins "mean"/"nearest" anywhere.

    Both other tests now compare a produced value against the constant the
    producer reads, so they can no longer fail on a wrong value (ledger GEN-43).
    """
    assert ngff_.DOWNSAMPLE_METHODS == {
        "image": ("mean", "2x block mean over an edge-replicated pad"),
        "label": ("nearest", "2x nearest-neighbour (top-left of each block)"),
    }
    assert ngff_.DOWNSAMPLE_KINDS == {"image": "mean", "label": "nearest"}


def test_pyramid_block_records_levels_stop_and_downsample_methods() -> None:
    block = ngff_.build_phenotypic_attributes(
        image_class="Image",
        series_names=["gray", "detect_mat"],
        pyramid_levels=4,
        metadata_sections=_sections(),
        detect_mode="gray",
        illuminant=None,
        gamma=None,
    )
    pyramid = block[PhenotypicAttr.PYRAMID]
    assert pyramid == {
        "levels": 4,
        "stop_px": 512,
        "downsample": ngff_.DOWNSAMPLE_KINDS,
    }


def test_work_id_is_a_constructor_argument_not_a_patch() -> None:
    block = ngff_.build_phenotypic_attributes(
        image_class="Image",
        series_names=["gray", "detect_mat"],
        pyramid_levels=1,
        metadata_sections=_sections(),
        detect_mode="gray",
        illuminant=None,
        gamma=None,
        work_id="abc123",
    )
    assert block[PhenotypicAttr.WORK_ID] == "abc123"


def test_arbitrary_metadata_keys_are_stored_verbatim() -> None:
    """Real images carry Metadata_PlateNum (member=None) and bare public keys.

    A write-time canonicality gate would abort save2zarr on most production
    runs. See OPEN-QUESTIONS D3.
    """
    sections = _sections()
    sections["public"]["Metadata_PlateNum"] = 3
    sections["public"]["MyColumn"] = "x"
    block = ngff_.build_phenotypic_attributes(
        image_class="Image",
        series_names=["gray", "detect_mat"],
        pyramid_levels=1,
        metadata_sections=sections,
        detect_mode="gray",
        illuminant=None,
        gamma=None,
    )
    stored = block[PhenotypicAttr.METADATA]["public"]
    assert stored["Metadata_PlateNum"] == 3
    assert stored["MyColumn"] == "x"


def test_block_is_json_serialisable() -> None:
    block = ngff_.build_phenotypic_attributes(
        image_class="GridImage",
        series_names=["rgb", "gray", "detect_mat"],
        pyramid_levels=4,
        metadata_sections=_sections(),
        detect_mode="gray",
        illuminant="D65",
        gamma="sRGB",
        grid={"nrows": 8, "ncols": 12, "grid_finder": None},
    )
    assert json.loads(json.dumps(block)) == block


def test_read_phenotypic_attributes_round_trips(tmp_path: Path) -> None:
    block = ngff_.build_phenotypic_attributes(
        image_class="Image",
        series_names=["gray", "detect_mat"],
        pyramid_levels=1,
        metadata_sections=_sections(),
        detect_mode="gray",
        illuminant=None,
        gamma=None,
    )
    store = tmp_path / "x.ome.zarr"
    store.mkdir()
    (store / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {
                    "ome": {"version": "0.5", "bioformats2raw.layout": 3},
                    "phenotypic": block,
                },
            }
        ),
        encoding="utf-8",
    )
    assert ngff_.read_phenotypic_attributes(store) == block


def test_read_phenotypic_attributes_raises_on_a_missing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ngff_.read_phenotypic_attributes(tmp_path / "absent.ome.zarr")
