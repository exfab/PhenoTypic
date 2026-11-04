"""``measure()`` joins measurer tables on ``Object_Label`` and never guesses.

Every measurer returns one row per colony keyed by ``Object_Label``, and some
emit the same columns: ``MeasureBounds``, a ``GridFinder`` and the image's own
info block all carry the ``Bbox_*`` geometry, and a ``GridFinder`` and a
``GridImage``'s info block both carry the ``Grid_*`` assignment. A shared
column is only safe to keep once if both tables give every colony the same
value in it.

Before this file, ``_merge_on_object_labels`` compared the incoming table's
column with *itself*, so the check always passed, every shared column became an
inner-join key, and a colony whose shared values differed was silently dropped.
On the 8 x 12 synthetic plate, ``ImagePipeline(meas=[MeasureSize()], nrows=4,
ncols=6).measure(plate)`` returned 1 row of 96 and ``nrows=2, ncols=12``
returned none. The maintainer's decision: a disagreement raises
``MeasurementConflictError``, naming the columns, how many colonies differ, and
the fix. Nothing is dropped and nothing is chosen.

The shape tests import the error lazily and match on ``ValueError`` so that,
run against the old merge, they fail on the missing raise rather than at
collection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phenotypic import GridImage, ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.measure import MeasureBounds, MeasureShape, MeasureSize
from phenotypic.schema import GRID, OBJECT

#: The four grid-assignment columns both the injected finder and the image's
#: own ``grid.info()`` emit.
GRID_ASSIGNMENT_COLUMNS = (
    "Grid_RowNum",
    "Grid_ColNum",
    "Grid_RowMajorIdx",
    "Grid_ColMajorIdx",
)

#: The synthetic plate holds one colony per well of its 8 x 12 grid.
N_COLONIES = 96


def _conflict_error_type() -> type:
    from phenotypic.sdk_.exceptions_ import MeasurementConflictError

    return MeasurementConflictError


@pytest.fixture
def plate() -> GridImage:
    """A fresh 8 x 12 synthetic plate (default CenteredAutoGridFinder)."""
    return load_synth_yeast_plate()


# ---------------------------------------------------------------------------
# The reported shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("nrows", "ncols", "n_differing"),
    [
        # 1 colony survived the old inner join, so 95 differ.
        pytest.param(4, 6, 95, id="4x6"),
        # No colony survived it, so all 96 differ.
        pytest.param(2, 12, 96, id="2x12"),
    ],
)
def test_a_preset_grid_unlike_the_images_grid_raises(
    plate: GridImage, nrows: int, ncols: int, n_differing: int
) -> None:
    pipeline = ImagePipeline(
        meas={"MeasureSize": MeasureSize()}, nrows=nrows, ncols=ncols
    )

    with pytest.raises(ValueError) as caught:
        pipeline.measure(plate)

    error = caught.value
    assert isinstance(error, _conflict_error_type()), repr(error)
    assert set(error.columns) == set(GRID_ASSIGNMENT_COLUMNS)
    assert error.n_objects == n_differing
    assert error.n_compared == N_COLONIES

    message = str(error)
    for column in GRID_ASSIGNMENT_COLUMNS:
        assert column in message
    assert f"{n_differing} of {N_COLONIES} colonies" in message
    # The fix, in terms of the default finder and both shapes.
    assert "CenteredAutoGridFinder" in message
    assert f"nrows={nrows}, ncols={ncols}" in message
    assert "lays out 8 rows x 12 columns" in message
    assert "leave nrows and ncols off the pipeline" in message


def test_a_preset_grid_like_the_images_grid_measures_every_colony(
    plate: GridImage,
) -> None:
    with_preset = ImagePipeline(
        meas={"MeasureSize": MeasureSize()}, nrows=8, ncols=12
    ).measure(plate.copy(), include_metadata=False)
    without_preset = ImagePipeline(
        meas={"MeasureSize": MeasureSize()}
    ).measure(plate.copy(), include_metadata=False)

    assert len(with_preset) == N_COLONIES
    # The injected finder's columns duplicate the image's info block exactly,
    # so they collapse into it and the two tables are the same table.
    pd.testing.assert_frame_equal(with_preset, without_preset, check_exact=True)


def test_shared_bbox_columns_from_measure_bounds_collapse_into_one(
    plate: GridImage,
) -> None:
    df = ImagePipeline(
        meas={"MeasureBounds": MeasureBounds(), "MeasureShape": MeasureShape()},
        nrows=8,
        ncols=12,
    ).measure(plate, include_metadata=False)

    assert len(df) == N_COLONIES
    assert not [c for c in df.columns if str(c).endswith("_merged")]
    assert df.columns.is_unique


def test_an_explicit_grid_finder_unlike_the_images_grid_raises(
    plate: GridImage,
) -> None:
    from phenotypic.grid import CenteredAutoGridFinder

    pipeline = ImagePipeline(
        meas={
            "plate_grid": CenteredAutoGridFinder(nrows=4, ncols=6),
            "MeasureSize": MeasureSize(),
        }
    )

    with pytest.raises(_conflict_error_type()) as caught:
        pipeline.measure(plate)

    message = str(caught.value)
    assert "'plate_grid'" in message
    assert "CenteredAutoGridFinder" in message
    assert "nrows=8, ncols=12" in message
    assert caught.value.n_objects == 95


def test_the_old_merge_dropped_colonies_on_this_exact_input(
    plate: GridImage,
) -> None:
    """The control: the tables really disagree, and an inner join on them drops.

    Rebuilds what the old merge did (every shared column a join key) from the
    same two tables measure() merges, so the raise above is about a real
    disagreement and not about a false positive in the new comparison.
    """
    from phenotypic.grid import CenteredAutoGridFinder

    finder_table = CenteredAutoGridFinder(nrows=4, ncols=6).measure(plate)
    info_table = plate.grid.info(include_metadata=False)
    shared = [c for c in finder_table.columns if c in info_table.columns]

    joined = finder_table.merge(info_table, on=shared)

    assert len(joined) == N_COLONIES - 95


# ---------------------------------------------------------------------------
# The comparison itself, on hand-built tables
# ---------------------------------------------------------------------------


def _table(labels, **columns) -> pd.DataFrame:
    return pd.DataFrame({OBJECT.LABEL: labels, **columns})


def test_tables_in_different_row_orders_are_aligned_before_comparing() -> None:
    left = _table([1, 2, 3], Bbox_CenterRR=[10.0, 20.0, 30.0], Size_Area=[5, 6, 7])
    right = _table([3, 1, 2], Bbox_CenterRR=[30.0, 10.0, 20.0], Shape_Solidity=[0.9, 0.7, 0.8])

    merged = ImagePipeline._merge_on_object_labels([left, right])

    assert list(merged[OBJECT.LABEL]) == [1, 2, 3]
    assert list(merged["Bbox_CenterRR"]) == [10.0, 20.0, 30.0]
    assert list(merged["Shape_Solidity"]) == [0.7, 0.8, 0.9]
    assert list(merged.columns) == [OBJECT.LABEL, "Bbox_CenterRR", "Size_Area", "Shape_Solidity"]


def test_nan_in_a_shared_column_on_both_sides_is_agreement() -> None:
    left = _table([1, 2], Bbox_IntensityWeightedCenterRR=[np.nan, 4.0], Size_Area=[5, 6])
    right = _table([1, 2], Bbox_IntensityWeightedCenterRR=[np.nan, 4.0], Shape_Solidity=[0.7, 0.8])

    merged = ImagePipeline._merge_on_object_labels([left, right])

    assert len(merged) == 2
    # The old merge treated NaN != NaN as a disagreement and kept a second
    # copy of the column under a ``_merged`` suffix.
    assert list(merged.columns) == [
        OBJECT.LABEL,
        "Bbox_IntensityWeightedCenterRR",
        "Size_Area",
        "Shape_Solidity",
    ]
    assert np.isnan(merged["Bbox_IntensityWeightedCenterRR"].iloc[0])


def test_nan_on_one_side_only_is_a_disagreement() -> None:
    left = _table([1, 2], Bbox_IntensityWeightedCenterRR=[np.nan, 4.0])
    right = _table([1, 2], Bbox_IntensityWeightedCenterRR=[3.0, 4.0])

    with pytest.raises(_conflict_error_type()) as caught:
        ImagePipeline._merge_on_object_labels([left, right])

    assert caught.value.columns == ("Bbox_IntensityWeightedCenterRR",)
    assert caught.value.n_objects == 1


def test_categorical_grid_columns_with_different_categories_are_compared() -> None:
    """Different grid shapes give categoricals with different categories.

    pandas refuses ``==`` between such categoricals, so a naive comparison
    raises ``TypeError`` instead of reporting the conflict.
    """
    left = _table([1, 2], Grid_RowNum=pd.Categorical([0, 1], categories=range(4)))
    right = _table([1, 2], Grid_RowNum=pd.Categorical([0, 3], categories=range(8)))

    with pytest.raises(_conflict_error_type()) as caught:
        ImagePipeline._merge_on_object_labels([left, right])

    assert caught.value.n_objects == 1
    assert GRID.owns_header("Grid_RowNum")


def test_floats_are_compared_exactly() -> None:
    """Shared columns are computed by the same code from the same object map.

    Measured on the synthetic and 72 h plates, every shared ``Bbox_*`` value
    is bit-identical across ``MeasureBounds``, ``CenteredAutoGridFinder`` and
    ``grid.info()``, so no tolerance is applied: a difference of one ulp is a
    real disagreement.
    """
    value = 12.5
    left = _table([1], Bbox_CenterRR=[value])
    right = _table([1], Bbox_CenterRR=[np.nextafter(value, np.inf)])

    with pytest.raises(_conflict_error_type()):
        ImagePipeline._merge_on_object_labels([left, right])


def test_a_conflict_names_both_sources_and_a_generic_fix() -> None:
    left = _table([1, 2, 3], Size_Area=[5, 6, 7])
    right = _table([1, 2, 3], Size_Area=[5, 60, 70])

    with pytest.raises(_conflict_error_type()) as caught:
        ImagePipeline._merge_on_object_labels(
            [left, right], sources=["MeasureSize", "MeasureSize_coarse"]
        )

    message = str(caught.value)
    assert "'MeasureSize'" in message and "'MeasureSize_coarse'" in message
    assert "2 of 3 colonies" in message
    assert "Size_Area" in message
    assert caught.value.sources == ("MeasureSize", "MeasureSize_coarse")


def test_labels_present_on_one_side_only_keep_the_inner_join() -> None:
    """Unchanged behaviour: a colony one measurer did not report is dropped.

    Only the colonies both tables carry are compared, so a missing row is not
    a disagreement.
    """
    left = _table([1, 2, 3], Bbox_CenterRR=[1.0, 2.0, 3.0])
    right = _table([1, 3], Bbox_CenterRR=[1.0, 3.0], Size_Area=[5, 7])

    merged = ImagePipeline._merge_on_object_labels([left, right])

    assert list(merged[OBJECT.LABEL]) == [1, 3]
