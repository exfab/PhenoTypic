"""``MeasureEmbedding``: the embedding measurer ABC (#281 P4 Task 1, spec E1).

An embedder returns its table's frame from ``measure()`` for notebook use:
``Object_Label`` + ``embedding`` for crops, the ``GRID`` key columns +
``embedding`` for a grid. Each ``embedding`` cell is a 1-D ``float16`` vector.
"""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic.abc_ import MeasureEmbedding, MeasureFeatures
from phenotypic.abc_._measure_embedding import GridCellOutsideImageError
from phenotypic.schema import GRID, OBJECT
from phenotypic.sdk_.exceptions_ import OperationFailedError
from tests._tiny_run import TOY_DIM, TOY_GRID_COL_EDGES, TOY_GRID_ROW_EDGES


def test_grid_region_needs_no_objects(toy_grid_embedder, plate):
    frame = toy_grid_embedder.measure(plate)          # no detection ran
    assert {str(GRID.ROW_NUM), str(GRID.COL_NUM), "embedding"} <= set(frame.columns)


def test_an_embedder_is_a_measurer_with_a_region_dependent_object_need(
    toy_embedder, toy_grid_embedder
):
    assert isinstance(toy_embedder, MeasureFeatures)
    assert isinstance(toy_embedder, MeasureEmbedding)
    assert toy_embedder.requires_objects is True
    assert toy_grid_embedder.requires_objects is False
    assert toy_embedder.payload_key() == "toy-v1/rgb/objects"
    assert toy_grid_embedder.payload_key() == "toy-v1/rgb/grid"


def test_crop_frame_is_one_float16_vector_per_object(toy_embedder, detected_plate):
    frame = toy_embedder.measure(detected_plate)
    assert list(frame.columns) == [str(OBJECT.LABEL), "embedding"]
    assert len(frame) == detected_plate.num_objects
    assert sorted(frame[str(OBJECT.LABEL)]) == detected_plate.objects.labels
    for cell in frame["embedding"]:
        assert isinstance(cell, np.ndarray)
        assert cell.dtype == np.float16
        assert cell.shape == (TOY_DIM,)


def test_crop_vectors_come_from_each_objects_bbox(toy_embedder, detected_plate):
    """The toy vector is the box's coordinate sum, so it names the box embedded."""
    frame = toy_embedder.measure(detected_plate).set_index(str(OBJECT.LABEL))
    for prop in detected_plate.objects.props:
        expected = np.float16(sum(prop.bbox))
        assert frame.loc[prop.label, "embedding"][0] == expected


def test_grid_frame_spells_every_cell_in_row_major_order(toy_grid_embedder, plate):
    frame = toy_grid_embedder.measure(plate)
    assert list(frame.columns) == [
        str(GRID.ROW_MAJOR_IDX),
        str(GRID.ROW_NUM),
        str(GRID.COL_NUM),
        str(GRID.ROW_INTERVAL_START),
        str(GRID.ROW_INTERVAL_END),
        str(GRID.COL_INTERVAL_START),
        str(GRID.COL_INTERVAL_END),
        "embedding",
    ]
    assert frame[str(GRID.ROW_MAJOR_IDX)].tolist() == [0, 1, 2, 3]
    assert frame[str(GRID.ROW_NUM)].tolist() == [0, 0, 1, 1]
    assert frame[str(GRID.COL_NUM)].tolist() == [0, 1, 0, 1]
    last = frame.iloc[3]
    assert (
        last[str(GRID.ROW_INTERVAL_START)],
        last[str(GRID.ROW_INTERVAL_END)],
        last[str(GRID.COL_INTERVAL_START)],
        last[str(GRID.COL_INTERVAL_END)],
    ) == (
        TOY_GRID_ROW_EDGES[1],
        TOY_GRID_ROW_EDGES[2],
        TOY_GRID_COL_EDGES[1],
        TOY_GRID_COL_EDGES[2],
    )
    expected = np.float16(sum((100, 150, 200, 300)))
    assert last["embedding"][0] == expected


def test_a_cell_outside_the_image_fails_naming_the_cell_and_the_size(plate):
    """Spec E3a: a cell past the image edge fails the image rather than being clipped."""
    from phenotypic.grid import ManualGridFinder
    from tests._tiny_run import toy_embedder_class

    embedder = toy_embedder_class()(
        regions="grid",
        grid=ManualGridFinder(row_edges=[0, 300, 700], col_edges=[0, 400]),
    )
    with pytest.raises(
        OperationFailedError, match=r"cell \(1, 0\).*image is 600 x 800"
    ) as excinfo:
        embedder.measure(plate)
    assert excinfo.value.err_type is GridCellOutsideImageError


def test_a_grid_embedder_needs_a_grid_and_an_object_embedder_refuses_one():
    from phenotypic.grid import ManualGridFinder
    from tests._tiny_run import toy_embedder_class

    toy = toy_embedder_class()
    with pytest.raises(ValueError, match="regions='grid' needs a grid"):
        toy(regions="grid")
    with pytest.raises(ValueError, match="grid is only used with regions='grid'"):
        toy(grid=ManualGridFinder(row_edges=[0, 10], col_edges=[0, 10]))


def test_an_image_with_no_objects_embeds_to_an_empty_table(toy_embedder, plate):
    plate.objmap[:] = 0
    frame = toy_embedder.measure(plate)
    assert list(frame.columns) == [str(OBJECT.LABEL), "embedding"]
    assert len(frame) == 0


def test_a_wrong_vector_length_is_refused(toy_embedder, detected_plate, monkeypatch):
    def _short(self, image, boxes):
        return np.zeros((len(boxes), TOY_DIM - 1), dtype=np.float32)

    monkeypatch.setattr(type(toy_embedder), "_embed_boxes", _short)
    with pytest.raises(OperationFailedError, match="dim"):
        toy_embedder.measure(detected_plate)


def test_the_grid_round_trips_through_json(toy_grid_embedder):
    """``grid`` is an ``OperationField``: it serializes class-tagged and rebuilds."""
    import json

    params = json.loads(toy_grid_embedder.to_json())["params"]
    assert params["grid"]["class"] == "ManualGridFinder"
    clone = type(toy_grid_embedder).model_validate(params)
    assert clone.regions == "grid"
    assert list(clone.grid.row_edges) == list(TOY_GRID_ROW_EDGES)
    assert clone.payload_key() == toy_grid_embedder.payload_key()


# ---------------------------------------------------------------------------
# #281 named regions: ``name`` (user decisions 2026-11-19)
# ---------------------------------------------------------------------------


def test_name_sits_right_after_grid():
    from tests._tiny_run import toy_embedder_class

    fields = list(toy_embedder_class().model_fields)
    assert fields[fields.index("grid") + 1] == "name"
    assert toy_embedder_class()().name is None


def test_name_is_not_tunable():
    from phenotypic.sdk_.typing_ import TuneSpec

    (spec,) = [
        meta for meta in MeasureEmbedding.model_fields["name"].metadata
        if isinstance(meta, TuneSpec)
    ]
    assert spec.tunable is False


def test_an_object_embedder_takes_a_name_too(detected_plate):
    from tests._tiny_run import toy_embedder_class

    embedder = toy_embedder_class()(name="pool2")
    assert embedder.requires_objects is True
    assert len(embedder.measure(detected_plate)) == detected_plate.num_objects


@pytest.mark.parametrize(
    "name",
    ["", "Wells", "-wells", "_wells", "wells-a", "wells/a", "we lls", "wells\n",
     "a" * 33, "é"],
)
def test_a_name_that_is_not_a_lowercase_slug_is_refused(name):
    from tests._tiny_run import toy_embedder_class, toy_grid_embedder_of

    with pytest.raises(ValueError, match="name must be a lowercase slug"):
        toy_grid_embedder_of(name=name)
    with pytest.raises(ValueError, match="name must be a lowercase slug"):
        toy_embedder_class()(name=name)


@pytest.mark.parametrize("name", ["wells", "agar", "a", "0", "rig1_wells", "a" * 32])
def test_a_lowercase_slug_name_is_accepted(name):
    from tests._tiny_run import toy_grid_embedder_of

    assert toy_grid_embedder_of(name=name).name == name


@pytest.mark.parametrize(
    ("regions", "name", "key"),
    [
        ("objects", None, "toy-v1/rgb/objects"),
        ("objects", "pool2", "toy-v1/rgb/objects-pool2"),
        ("grid", None, "toy-v1/rgb/grid"),
        ("grid", "wells", "toy-v1/rgb/grid-wells"),
        ("grid", "agar", "toy-v1/rgb/grid-agar"),
    ],
)
def test_the_payload_key_names_the_region_and_the_name(regions, name, key):
    from tests._tiny_run import toy_embedder_class, toy_grid_embedder_of

    embedder = (
        toy_embedder_class()(name=name) if regions == "objects"
        else toy_grid_embedder_of(name=name)
    )
    assert embedder.payload_key() == key


def test_a_named_grid_needs_no_objects(plate):
    from tests._tiny_run import toy_grid_embedder_of

    embedder = toy_grid_embedder_of(name="wells")
    assert embedder.requires_objects is False
    assert len(embedder.measure(plate)) == 4          # no detection ran


def test_each_named_grid_embeds_its_own_cells(plate):
    """The toy vector is the box's coordinate sum and its mean pixel, so each
    named grid's rows come from that grid's cells and pixels."""
    from tests._tiny_run import (
        TOY_AGAR_COL_EDGES,
        TOY_AGAR_ROW_EDGES,
        named_grid_embedders,
        toy_box_layer_mean,
        toy_grid_embedder_of,
    )

    grids = named_grid_embedders(toy_grid_embedder_of)
    wells, agar = grids["wells"].measure(plate), grids["agar"].measure(plate)
    assert len(wells) == 4 and len(agar) == 1
    box = np.array(
        [[TOY_AGAR_ROW_EDGES[0], TOY_AGAR_COL_EDGES[0], TOY_AGAR_ROW_EDGES[1], TOY_AGAR_COL_EDGES[1]]]
    )
    (vector,) = agar["embedding"]
    assert vector[0] == np.float16(box.sum())
    assert vector[-1] == np.float16(toy_box_layer_mean(np.asarray(plate.rgb[:]), box)[0])
    assert vector[0] not in {cell[0] for cell in wells["embedding"]}


def test_the_name_round_trips_through_json():
    import json

    from tests._tiny_run import toy_grid_embedder_of

    embedder = toy_grid_embedder_of(name="wells")
    params = json.loads(embedder.to_json())["params"]
    assert params["name"] == "wells"
    clone = type(embedder).model_validate(params)
    assert clone.payload_key() == "toy-v1/rgb/grid-wells"


def test_an_unnamed_rgb_embedder_keeps_its_params_digest():
    """``name=None`` and ``layer="rgb"`` are left out of
    ``producer.params_sha256``, so an embedder that predates both fields keeps
    its digest, and so its deliverable's descriptor."""
    import hashlib
    import json

    from tests._tiny_run import toy_embedder_class, toy_grid_embedder_of

    for embedder in (toy_grid_embedder_of(), toy_embedder_class()()):
        dumped = embedder.model_dump(mode="json")
        assert (dumped.pop("name"), dumped.pop("layer")) == (None, "rgb")
        before = hashlib.sha256(
            json.dumps(dumped, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        assert embedder._params_sha256() == before


def test_name_and_layer_each_change_the_params_digest():
    """``producer.params_sha256`` is the canonical dump, so a name and a layer
    each make a distinct descriptor."""
    from tests._tiny_run import toy_grid_embedder_of

    digests = {
        toy_grid_embedder_of()._params_sha256(),
        toy_grid_embedder_of(name="wells")._params_sha256(),
        toy_grid_embedder_of(name="agar")._params_sha256(),
        toy_grid_embedder_of(layer="gray")._params_sha256(),
        toy_grid_embedder_of(name="wells", layer="gray")._params_sha256(),
    }
    assert len(digests) == 5


# ---------------------------------------------------------------------------
# #281 the layer: rgb or gray, its own key segment (user decisions 2026-11-19)
# ---------------------------------------------------------------------------


def test_layer_follows_name_defaults_to_rgb_and_is_not_tunable():
    from phenotypic.sdk_.typing_ import TuneSpec
    from tests._tiny_run import toy_embedder_class

    fields = list(toy_embedder_class().model_fields)
    assert fields[fields.index("name") + 1] == "layer"
    assert toy_embedder_class()().layer == "rgb"
    (spec,) = [
        meta for meta in MeasureEmbedding.model_fields["layer"].metadata
        if isinstance(meta, TuneSpec)
    ]
    assert spec.tunable is False


@pytest.mark.parametrize("layer", ["hsv", "RGB", "detect_mat", ""])
def test_a_layer_other_than_rgb_or_gray_is_refused(layer):
    from tests._tiny_run import toy_grid_embedder_of

    with pytest.raises(ValueError, match="layer"):
        toy_grid_embedder_of(layer=layer)


@pytest.mark.parametrize(
    ("regions", "layer", "name", "key"),
    [
        ("objects", "gray", None, "toy-v1/gray/objects"),
        ("objects", "gray", "pool2", "toy-v1/gray/objects-pool2"),
        ("grid", "gray", None, "toy-v1/gray/grid"),
        ("grid", "gray", "wells", "toy-v1/gray/grid-wells"),
        ("grid", "rgb", "wells", "toy-v1/rgb/grid-wells"),
    ],
)
def test_the_payload_key_names_the_layer(regions, layer, name, key):
    from tests._tiny_run import toy_embedder_class, toy_grid_embedder_of

    embedder = (
        toy_embedder_class()(name=name, layer=layer) if regions == "objects"
        else toy_grid_embedder_of(name=name, layer=layer)
    )
    assert embedder.payload_key() == key


def test_the_gray_layer_is_three_identical_uint8_channels(plate):
    from tests._tiny_run import toy_grid_embedder_of

    pixels = toy_grid_embedder_of(layer="gray")._layer_pixels(plate)
    assert pixels.dtype == np.uint8 and pixels.shape == (*plate.shape[:2], 3)
    assert (pixels[..., 0] == pixels[..., 1]).all() and (pixels[..., 1] == pixels[..., 2]).all()
    expected = (np.clip(np.asarray(plate.gray[:]), 0, 1) * 255).astype(np.uint8)
    assert (pixels[..., 0] == expected).all()
    rgb = toy_grid_embedder_of()._layer_pixels(plate)
    assert rgb.dtype == np.asarray(plate.rgb[:]).dtype and (rgb == plate.rgb[:]).all()


def _equal_channel_plate(plate):
    """The synth plate with its three channels made equal: gray is then rgb."""
    from phenotypic import Image

    value = np.asarray(plate.rgb[:]).mean(axis=2).astype(np.uint8)
    return Image(np.stack([value] * 3, axis=-1))


def test_a_gray_grid_differs_from_rgb_on_a_coloured_plate_and_not_on_a_gray_one(plate):
    from tests._tiny_run import toy_grid_embedder_of

    rgb, gray = toy_grid_embedder_of(), toy_grid_embedder_of(layer="gray")
    coloured = np.stack(gray.measure(plate)["embedding"]) - np.stack(rgb.measure(plate)["embedding"])
    assert np.abs(coloured).max() > 1e-3
    equal = _equal_channel_plate(plate)
    np.testing.assert_allclose(
        np.stack(gray.measure(equal)["embedding"]).astype(np.float32),
        np.stack(rgb.measure(equal)["embedding"]).astype(np.float32),
        atol=1e-3,
    )


def test_the_entry_records_the_layer_and_binds_the_stored_series(plate):
    from tests._tiny_run import toy_grid_embedder_of

    rgb = toy_grid_embedder_of().payload(plate)
    gray = toy_grid_embedder_of(layer="gray").payload(plate)
    assert rgb.descriptor["input"] == {"series": "rgb", "level": 0, "layer": "rgb"}
    assert gray.descriptor["input"] == {"series": "rgb", "level": 0, "layer": "gray"}
    # gray is computed from the stored rgb, so staleness tracks those pixels.
    assert gray.descriptor["binds"] == rgb.descriptor["binds"]
    assert gray.descriptor["producer"]["params_sha256"] != rgb.descriptor["producer"]["params_sha256"]
    assert (rgb.key, gray.key) == ("toy-v1/rgb/grid", "toy-v1/gray/grid")
