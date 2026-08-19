"""The write-only OME projection. Never read back; validated on write."""

from __future__ import annotations

import re

import numpy as np
import pytest

from phenotypic.sdk_ import ngff_


def test_multiscales_scale_comes_from_actual_level_shapes() -> None:
    shapes = ngff_.pyramid_level_shapes((1025, 7), 3)
    block = ngff_.build_multiscales(series="gray", level_shapes=shapes, name="plate")
    scales = [
        transform["scale"]
        for dataset in block["multiscales"][0]["datasets"]
        for transform in dataset["coordinateTransformations"]
        if transform["type"] == "scale"
    ]
    assert scales[0] == pytest.approx([1.0, 1.0])
    assert scales[1] == pytest.approx([1025 / 513, 7 / 4])
    assert scales[1][0] != pytest.approx(2.0)


def test_multiscales_axes_are_ordered_channel_then_space() -> None:
    shapes = ngff_.pyramid_level_shapes((3, 1024, 1024), 2)
    block = ngff_.build_multiscales(series="rgb", level_shapes=shapes)
    axes = block["multiscales"][0]["axes"]
    assert [axis["name"] for axis in axes] == ["c", "y", "x"]
    assert [axis["type"] for axis in axes] == ["channel", "space", "space"]


def test_multiscales_dataset_paths_are_level_indices() -> None:
    shapes = ngff_.pyramid_level_shapes((2048, 2048), 3)
    block = ngff_.build_multiscales(series="gray", level_shapes=shapes)
    assert [d["path"] for d in block["multiscales"][0]["datasets"]] == ["0", "1", "2"]


def test_omero_emits_every_required_channel_field() -> None:
    """NGFF is conditionally strict: partial omero fails the conformance gate."""
    block = ngff_.build_omero(
        series="rgb", dtype=np.dtype("uint16"), bit_depth=16, name="plate"
    )
    channels = block["omero"]["channels"]
    assert len(channels) == 3
    for channel in channels:
        assert re.fullmatch(r"[0-9A-F]{6}", channel["color"]), channel
        assert set(channel["window"]) == {"min", "max", "start", "end"}
        assert channel["window"]["max"] == 65535
        assert channel["window"]["end"] == 65535
        for key in ("label", "active", "family", "coefficient", "inverted"):
            assert key in channel


def test_omero_window_max_tracks_bit_depth() -> None:
    block = ngff_.build_omero(
        series="rgb", dtype=np.dtype("uint8"), bit_depth=8, name=None
    )
    assert block["omero"]["channels"][0]["window"]["max"] == 255


@pytest.mark.parametrize("series", ["gray", "detect_mat"])
def test_omero_is_omitted_for_every_float_series(series: str) -> None:
    """A float layer in [0,1] under a bit-depth window renders near-black.

    `gray` matters most: it is the PRIMARY series in every rgb-less store, so
    it is the layer an external reader opens by default. Verified by execution
    that it is float32 in [0.545, 0.955] while bit_depth is 8 -- identical to
    detect_mat, which is why keying on the series NAME missed it (ALGO-2).
    """
    assert (
        ngff_.build_omero(
            series=series, dtype=np.dtype("float32"), bit_depth=8, name=None
        )
        == {}
    )


def test_omero_is_keyed_on_dtype_not_on_the_series_name() -> None:
    """Self-maintaining: a future float layer needs no list entry, and an
    integer `gray` would get its block back automatically if the deferred
    conversion ever lands."""
    assert (
        ngff_.build_omero(series="gray", dtype=np.dtype("uint8"), bit_depth=8, name=None)
        != {}
    )
    assert (
        ngff_.build_omero(
            series="rgb", dtype=np.dtype("float32"), bit_depth=8, name=None
        )
        == {}
    )


def test_image_label_is_always_emitted_with_version_and_source() -> None:
    block = ngff_.build_image_label()
    assert block["image-label"]["version"] == "0.5"
    assert block["image-label"]["source"] == {"image": "../../"}


def test_image_label_colors_is_background_only() -> None:
    """`colors` is optional -- $defs/image-label has no `required` list -- and
    nothing in PhenoTypic reads it (P1)."""
    block = ngff_.build_image_label()
    assert block["image-label"]["colors"] == [{"label-value": 0, "rgba": [0, 0, 0, 0]}]


def test_image_label_takes_no_label_values() -> None:
    """It must not depend on array contents; that is what keeps it constant-size."""
    import inspect

    assert inspect.signature(ngff_.build_image_label).parameters == {}


def test_properties_is_never_emitted() -> None:
    """Locked decision #10: parquet stays the only measurement surface."""
    assert "properties" not in ngff_.build_image_label()["image-label"]


def test_image_label_is_constant_size_regardless_of_colony_count() -> None:
    """Drops the ~60 KB per-plate JSON the spec's OQ9 budgeted for."""
    import json

    assert len(json.dumps(ngff_.build_image_label())) < 500


def _xml_kwargs() -> dict:
    return {
        "series_names": ["rgb", "gray", "detect_mat"],
        "series_shapes": {
            "rgb": (3, 64, 48),
            "gray": (64, 48),
            "detect_mat": (64, 48),
        },
        "series_dtypes": {
            "rgb": np.dtype("uint8"),
            "gray": np.dtype("float32"),
            "detect_mat": np.dtype("float32"),
        },
        "metadata_sections": {
            "protected": {"Metadata_ImageName": "plate_01"},
            "public": {},
            "imported": {"TIFF:XResolution": 300.0},
        },
    }


def test_ome_xml_names_every_series_in_order() -> None:
    xml = ngff_.build_ome_xml(**_xml_kwargs())
    assert xml.count("<Image ") == 3
    assert xml.index("rgb") < xml.index("gray") < xml.index("detect_mat")


def test_map_annotation_namespaces_are_rembi_module_VALUES() -> None:
    """Not `str(enum)`, which is the Python repr since 3.11 (ledger ALGO-10).

    ome.xsd cannot catch this -- Annotation/@Namespace is xsd:anyURI, which
    accepts 'REMBI_MODULE.IMAGE_DATA' happily. Only an explicit assertion does.
    """
    xml = ngff_.build_ome_xml(**_xml_kwargs())
    assert "REMBI_MODULE" not in xml, "a Python-internal name leaked into OME-XML"
    assert 'Namespace="ImageData"' in xml


def test_control_characters_in_imported_metadata_do_not_break_the_document() -> None:
    r"""Real EXIF carries NUL-padded strings; XML 1.0 forbids them outright.

    `xml.sax.saxutils.escape` handles only & < >, so a NUL survives it and the
    document is not well-formed -- and `build_ome_xml` is pure string
    formatting, so nothing raises. This project's inputs are DSLR/raw captures
    read through `exiftool -json -n`, and `_normalize_metadata_value` decodes
    bytes with errors="replace", which fixes invalid UTF-8 but leaves \x00
    intact. Ledger ALGO-R2B-11.
    """
    from tests._ngff_conformance import assert_ome_xml_valid

    kwargs = _xml_kwargs()
    kwargs["metadata_sections"] = {
        "imported": {"EXIF:Make": "Canon\x00\x00 EOS", "EXIF:\x0bBad": "ok"}
    }
    xml = ngff_.build_ome_xml(**kwargs)
    # assert_ome_xml_valid catches XMLSchemaException, which covers a
    # well-formedness failure as well as a schema violation -- so this one call
    # is the whole gate. (Do not add a bare `ElementTree.fromstring` probe: the
    # stdlib parser has no billion-laughs guard, and this text comes from user
    # image files.)
    assert_ome_xml_valid(xml)
    assert "\x00" not in xml and "\x0b" not in xml


def test_ome_xml_validates_against_the_vendored_xsd() -> None:
    """The whole point of bioformats2raw.layout: 3 (ALGO-1).

    An earlier draft emitted a bare `<Pixels />` with none of its eight required
    attributes and no `<MetadataOnly/>`, plus `<M>` entries directly under
    `<MapAnnotation>` -- three separate violations, none of which the
    `xml.count("<Image ")` assertion could see.
    """
    from tests._ngff_conformance import assert_ome_xml_valid

    assert_ome_xml_valid(ngff_.build_ome_xml(**_xml_kwargs()))


def test_every_pixels_element_is_metadata_only() -> None:
    """§2.2.3: MUST use <MetadataOnly/>, never BinData/BinaryOnly/TiffData."""
    xml = ngff_.build_ome_xml(**_xml_kwargs())
    assert xml.count("<MetadataOnly/>") == 3
    for forbidden in ("<BinData", "<BinaryOnly", "<TiffData"):
        assert forbidden not in xml


def test_pixel_type_follows_the_dtype() -> None:
    xml = ngff_.build_ome_xml(**_xml_kwargs())
    assert 'Type="uint8"' in xml       # rgb
    assert 'Type="float"' in xml       # gray / detect_mat, float32


def test_an_unmapped_dtype_raises_rather_than_degrading() -> None:
    """PixelType is a closed enum; a silent fallback ships an invalid file."""
    kwargs = _xml_kwargs()
    kwargs["series_dtypes"]["gray"] = np.dtype("float16")
    with pytest.raises(ValueError, match="PixelType"):
        ngff_.build_ome_xml(**kwargs)


# NOTE (ledger SIMP-18): the propagation test lives in Phase 2 Task 2.2 as
# `test_a_failed_ome_xml_build_aborts_the_write`, which asserts the same
# propagation PLUS the consequence that matters -- that no store is left behind.
# It is strictly stronger and guards the same regression (someone re-adding
# `except Exception: return None`), so a unit-level twin here would be pure
# duplication. The neighbouring tests are NOT redundant with the XSD and stay:
# the OME content model permits BinData/TiffData (only NGFF 2.2.3 forbids them),
# and the XSD validates `Type` against its enum, not against the array's actual
# dtype -- so a float32 -> "uint8" mapping bug passes XSD cleanly.
