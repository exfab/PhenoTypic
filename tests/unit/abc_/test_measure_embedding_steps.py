"""``MeasureEmbedding`` as three steps: prepare, encode, payload_from (#282 P1 T1).

The inline :meth:`~phenotypic.abc_.MeasureEmbedding.payload` is
``prepare_regions`` -> ``encode`` (with the inline encoder) -> ``payload_from``,
so the GPU Stage 4 of #282 can run the same CPU preparation and the same
payload assembly around a different encoder. These tests pin that composition
on the model-free toy embedder; the DINO measurer's steps are pinned in
``tests/unit/measure/test_measure_dino_embedding_steps.py``.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from phenotypic.abc_ import BackboneInputs, PreparedRegions
from phenotypic.abc_._measure_embedding import GridCellOutsideImageError
from phenotypic.schema import GRID, OBJECT
from phenotypic.sdk_.exceptions_ import OperationFailedError
from tests._tiny_run import TOY_DIM


def toy_grid_with_edges(toy_grid_embedder, row_edges, col_edges):
    """The toy grid embedder over a grid of the given pixel edges."""
    from phenotypic.grid import ManualGridFinder

    return toy_grid_embedder.model_copy(
        update={"grid": ManualGridFinder(row_edges=list(row_edges), col_edges=list(col_edges))}
    )


def _empty(plate):
    """*plate* with no objects: a copy whose objmap is cleared."""
    empty = plate.copy()
    empty.objmap[:] = 0
    return empty


def test_payload_is_prepare_encode_payload_from(toy_embedder, detected_plate):
    from phenotypic.sdk_._embedding_tables import embedding_arrow_table

    prepared = toy_embedder.prepare_regions(detected_plate)
    vectors = toy_embedder.encode(prepared, toy_embedder.inline_encoder(detected_plate))
    stepwise = toy_embedder.payload_from(prepared, vectors, toy_embedder.entry_descriptor())
    whole = toy_embedder.payload(detected_plate)
    assert stepwise.key == whole.key and stepwise.descriptor == whole.descriptor
    assert embedding_arrow_table(stepwise).equals(embedding_arrow_table(whole))


def test_a_grid_payload_is_the_same_three_steps(toy_grid_embedder, plate):
    from phenotypic.sdk_._embedding_tables import embedding_arrow_table

    prepared = toy_grid_embedder.prepare_regions(plate)
    assert prepared.key_column == str(GRID.ROW_MAJOR_IDX)
    vectors = toy_grid_embedder.encode(prepared, toy_grid_embedder.inline_encoder(plate))
    stepwise = toy_grid_embedder.payload_from(
        prepared, vectors, toy_grid_embedder.entry_descriptor()
    )
    whole = toy_grid_embedder.payload(plate)
    assert stepwise.descriptor == whole.descriptor
    assert embedding_arrow_table(stepwise).equals(embedding_arrow_table(whole))


def test_the_abc_prepares_boxes_in_table_order(toy_embedder, detected_plate):
    prepared = toy_embedder.prepare_regions(detected_plate)
    assert isinstance(prepared, PreparedRegions)
    assert isinstance(prepared.inputs, BackboneInputs)
    assert prepared.inputs.kind == "boxes" and prepared.inputs.pooling == "none"
    assert prepared.key == toy_embedder.payload_key()
    assert prepared.key_column == str(OBJECT.LABEL)
    assert list(prepared.keys.columns) == [str(OBJECT.LABEL)]
    boxes = prepared.inputs.arrays[0]
    assert boxes.dtype == np.int64 and boxes.shape == (detected_plate.num_objects, 4)
    assert prepared.pool_plan is None


def test_prepared_regions_pickle(toy_grid_embedder, plate):
    prepared = toy_grid_embedder.prepare_regions(plate)
    again = pickle.loads(pickle.dumps(prepared))
    assert again.key == prepared.key and again.keys.equals(prepared.keys)
    assert again.binds == prepared.binds and again.input == prepared.input
    np.testing.assert_array_equal(again.inputs.arrays[0], prepared.inputs.arrays[0])


def test_zero_regions_call_neither_the_encoder_nor_dim(toy_embedder, plate, monkeypatch):
    prepared = toy_embedder.prepare_regions(_empty(plate))

    class Refuse:
        def run(self, inputs):
            raise AssertionError("encoder called for zero regions")

    descriptor = {**toy_embedder.entry_descriptor()}
    monkeypatch.setattr(type(toy_embedder), "_dim", lambda self: pytest.fail("_dim called"))
    vectors = toy_embedder.encode(prepared, Refuse())
    assert vectors.shape == (0, 0) and vectors.dtype == np.float32
    payload = toy_embedder.payload_from(prepared, vectors, descriptor)
    assert len(payload.frame) == 0 and payload.dim == TOY_DIM


def test_payload_from_refuses_non_finite_float16(toy_embedder, detected_plate):
    prepared = toy_embedder.prepare_regions(detected_plate)
    huge = np.full((len(prepared.keys), TOY_DIM), 1e6, dtype=np.float32)
    with pytest.raises(ValueError, match="not finite in float16"):
        toy_embedder.payload_from(prepared, huge, toy_embedder.entry_descriptor())


def test_payload_from_refuses_the_wrong_shape(toy_embedder, detected_plate):
    prepared = toy_embedder.prepare_regions(detected_plate)
    short = np.zeros((len(prepared.keys), TOY_DIM - 1), dtype=np.float32)
    with pytest.raises(ValueError, match="dim"):
        toy_embedder.payload_from(prepared, short, toy_embedder.entry_descriptor())


def test_grid_outside_the_image_is_raised_by_prepare(toy_grid_embedder, plate):
    m = toy_grid_with_edges(
        toy_grid_embedder, row_edges=[0, plate.shape[0] + 10], col_edges=[0, 10]
    )
    with pytest.raises(GridCellOutsideImageError):
        m.prepare_regions(plate)


def test_payload_still_wraps_an_image_failure(toy_grid_embedder, plate):
    """``payload`` fails an image as ``measure`` does: an ``OperationFailedError``
    whose ``err_type`` is the cause, so the CLI journals it unchanged."""
    m = toy_grid_with_edges(
        toy_grid_embedder, row_edges=[0, plate.shape[0] + 10], col_edges=[0, 10]
    )
    with pytest.raises(OperationFailedError) as excinfo:
        m.payload(plate)
    assert excinfo.value.err_type is GridCellOutsideImageError
