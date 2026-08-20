"""Every written store must validate against the vendored NGFF schemas."""

from __future__ import annotations

import json
import shutil
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
    # `match`, not a bare `pytest.raises(Exception)`. Mutating the
    # `zarr.open_array` call out of the harness left the bare form green:
    # reading `gray/9/zarr.json` for `dimension_names` raises FileNotFoundError
    # a few lines later, so the bare form proved only "something went wrong",
    # not that the dangling path was what went wrong. Both routes name the
    # path, so matching it pins the failure to this defect either way.
    with pytest.raises(Exception, match=r"gray[/\\]9"):
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


# ---------------------------------------------------------------------------
# Reader-level MUSTs the plan's cases left unexercised.
#
# Every assertion below was mutated to `assert True` against the fourteen tests
# above and NOTHING failed -- so each was satisfied vacuously by the positive
# stores, which is the exact shape of the KeyError that shipped green once
# (ledger GEN-47 / GEN-33). One negative case per surviving mutant.
# ---------------------------------------------------------------------------


def test_an_unnamed_extra_image_element_is_rejected(tmp_path: Path) -> None:
    """2.2.3: every multiscales group MUST represent exactly one Image.

    `Name` is use="optional" in ome.xsd, so the order scrape cannot see an
    unnamed <Image> -- only the count assertion can, and nothing reached it.
    """
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    xml_path = store / "OME" / "METADATA.ome.xml"
    extra = (
        '    <Image ID="Image:99">\n'
        '      <Pixels ID="Pixels:99" DimensionOrder="XYZCT" Type="uint8" '
        'SizeX="1" SizeY="1" SizeZ="1" SizeC="1" SizeT="1">\n'
        "        <MetadataOnly/>\n"
        "      </Pixels>\n"
        "    </Image>\n"
    )
    xml = xml_path.read_text(encoding="utf-8").replace(
        "  <StructuredAnnotations>", extra + "  <StructuredAnnotations>"
    )
    xml_path.write_text(xml, encoding="utf-8")
    with pytest.raises(AssertionError, match="exactly one Image"):
        assert_store_conforms(store)


def test_an_invalid_ome_xml_document_is_rejected(tmp_path: Path) -> None:
    """2.2.3 makes OME/METADATA.ome.xml a conditional MUST against ome.xsd.

    `assert_ome_xml_valid` is exercised on `build_ome_xml` output in
    tests/unit/sdk_/test_ngff_projection.py, but nothing made the STORE's
    document invalid -- removing the call from `assert_store_conforms`
    left every test here green.
    """
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    xml_path = store / "OME" / "METADATA.ome.xml"
    xml = xml_path.read_text(encoding="utf-8").replace(' SizeX="', ' Sixe="', 1)
    xml_path.write_text(xml, encoding="utf-8")
    with pytest.raises(AssertionError, match="ome.xsd"):
        assert_store_conforms(store)


def test_an_axis_count_that_does_not_match_the_array_is_rejected(
    tmp_path: Path,
) -> None:
    """2.4: the number of dimensions MUST correspond to the number of axes.

    Three axes over a 2-D array still satisfies `image.schema` (minContains 2 /
    maxContains 3 space axes), so only a reader-level check sees it.

    Which reader-level check is not what one would guess, and it is worth
    recording: the store is rejected by the `dimension_names` assertion, not by
    the `len(array.shape) == len(expected_axes)` one above it. That assertion is
    UNREACHABLE for any store zarr can open. Patching every level's
    `dimension_names` to the new axes -- the only way to get past it -- makes
    `zarr.open_array` raise `ValueError: dimension_names and shape need to have
    the same number of dimensions` from zarr's own v3 metadata validation
    (zarr/core/metadata/v3.py:272); leaving `dimension_names` alone, or deleting
    it, lands on the assertion below instead. Mutating the ndim assertion to
    `assert True` leaves the whole file green.
    """
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    group = store / "gray" / "zarr.json"
    payload = json.loads(group.read_text(encoding="utf-8"))
    multiscale = payload["attributes"]["ome"]["multiscales"][0]
    multiscale["axes"].insert(0, {"name": "c", "type": "channel"})
    group.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError, match=r"\['c', 'y', 'x'\]"):
        assert_store_conforms(store)


def test_a_missing_dimension_names_is_rejected(tmp_path: Path) -> None:
    """2.1 makes it a MUST; Zarr v3 makes it OPTIONAL array metadata.

    So a level missing it is a valid Zarr array and an NGFF violation, and it
    must surface as AssertionError rather than KeyError -- which is the whole
    reason the harness reads it with `.get`. The reordering case above cannot
    show that: it never removes the key.
    """
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    level = store / "gray" / "0" / "zarr.json"
    payload = json.loads(level.read_text(encoding="utf-8"))
    del payload["dimension_names"]
    level.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError):
        assert_store_conforms(store)


def test_a_labels_group_that_does_not_list_objmap_is_rejected(
    tmp_path: Path,
) -> None:
    """2.6: the label MUST be reachable through the `labels` group's list."""
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    labels_json = store / "rgb" / "labels" / "zarr.json"
    payload = json.loads(labels_json.read_text(encoding="utf-8"))
    payload["attributes"]["ome"]["labels"] = ["not_objmap"]
    labels_json.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError, match="not_objmap"):
        assert_store_conforms(store)


def test_a_float_label_dtype_is_rejected(tmp_path: Path) -> None:
    """2.6: label pixels MUST be an integer dtype."""
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    level = store / "rgb" / "labels" / "objmap" / "0" / "zarr.json"
    payload = json.loads(level.read_text(encoding="utf-8"))
    payload["data_type"] = "float32"
    payload["fill_value"] = 0.0
    level.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError, match="integer dtype"):
        assert_store_conforms(store)


def test_a_label_with_fewer_levels_than_its_image_is_rejected(
    tmp_path: Path,
) -> None:
    """2.6: the label's level count MUST match the unlabeled image."""
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    group = store / "rgb" / "labels" / "objmap" / "zarr.json"
    payload = json.loads(group.read_text(encoding="utf-8"))
    payload["attributes"]["ome"]["multiscales"][0]["datasets"].pop()
    group.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError, match="level count"):
        assert_store_conforms(store)


def test_a_missing_ome_group_is_rejected(tmp_path: Path) -> None:
    """The OME/ group is mandatory for this named-series layout.

    The assertion is deliberately an assert rather than an `if is_dir()` guard,
    so that a regression which stopped writing the group cannot pass every
    conformance test in the suite -- but until now nothing reached it.
    """
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    shutil.rmtree(store / "OME")
    with pytest.raises(AssertionError, match="OME/ group is mandatory"):
        assert_store_conforms(store)
