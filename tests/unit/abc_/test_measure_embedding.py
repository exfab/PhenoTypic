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
    assert toy_embedder.payload_key() == "toy-v1/crops"
    assert toy_grid_embedder.payload_key() == "toy-v1/grid"


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
