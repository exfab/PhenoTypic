"""Merging per-measurer frames, and refusing same-scale texture measurers.

``ImagePipelineCore._merge_on_object_labels`` joins the frames every measurer
returns (plus the image-info frame) on ``Object_Label``. Measurers routinely
emit the same column — image-info, ``MeasureBounds`` and
``MeasureGridLinRegStats`` all emit the ten ``Bbox_*`` columns — so the merge
must keep one copy of a shared column when the sources agree and refuse when
they disagree. It used to compare each incoming column with *itself*, which
turned every NaN-free shared column into an inner-join key (a disagreement then
silently dropped rows) and suffixed every NaN-bearing one with ``_merged``.

``ImagePipeline`` also refuses two ``MeasureTexture`` measurers at the same
``scale``: their ``Texture_{scale:02d}px-*`` columns are spelled identically,
so they would either be merged as one or collide (plan D8′).

A conflict names both producers (measurement keys, and ``'image info'``), and
the ``nrows``/``ncols`` preset no longer injects a grid finder into a
``GridImage``'s measurements: the image's own grid is authoritative, and a
second finder could only duplicate or contradict it (phase-2 review I-1, I-2).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from phenotypic import ImagePipeline
from phenotypic._core._pipeline_parts._image_pipeline_core import ImagePipelineCore
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.measure import (
    MeasureBounds,
    MeasureGridLinRegStats,
    MeasureShape,
    MeasureTexture,
)
from phenotypic.schema import OBJECT

LABEL = str(OBJECT.LABEL)


def _merge(*frames: pd.DataFrame, producers: list[str] | None = None) -> pd.DataFrame:
    return ImagePipelineCore._merge_on_object_labels(list(frames), producers=producers)


# --------------------------------------------------------------------------- #
# _merge_on_object_labels                                                     #
# --------------------------------------------------------------------------- #


def test_shared_identical_column_is_kept_once():
    left = pd.DataFrame({LABEL: [1, 2, 3], "A": [0.1, 0.2, 0.3],
                         "Bbox_MinRR": [4.0, 5.0, 6.0]})
    right = pd.DataFrame({LABEL: [1, 2, 3], "Bbox_MinRR": [4.0, 5.0, 6.0],
                          "B": [7, 8, 9]})

    merged = _merge(left, right)

    assert list(merged.columns) == [LABEL, "A", "Bbox_MinRR", "B"]
    assert merged["Bbox_MinRR"].tolist() == [4.0, 5.0, 6.0]
    assert merged["B"].tolist() == [7, 8, 9]


def test_shared_column_with_nan_in_the_same_rows_is_kept_once():
    """NaN in the same rows is agreement, not a reason to duplicate the column."""
    left = pd.DataFrame({LABEL: [1, 2, 3], "Grid_RowNum": [0.0, np.nan, 2.0]})
    right = pd.DataFrame({LABEL: [1, 2, 3], "Grid_RowNum": [0.0, np.nan, 2.0],
                          "B": [7, 8, 9]})

    merged = _merge(left, right)

    assert list(merged.columns) == [LABEL, "Grid_RowNum", "B"]
    assert not [c for c in merged.columns if str(c).endswith("_merged")]
    assert len(merged) == 3
    np.testing.assert_array_equal(merged["Grid_RowNum"].to_numpy(),
                                  [0.0, np.nan, 2.0])


def test_conflicting_shared_column_raises_naming_it():
    """Two measurers disagreeing on a shared column must not lose rows silently."""
    left = pd.DataFrame({LABEL: [1, 2, 3], "Bbox_MinRR": [4.0, 5.0, 6.0],
                         "Bbox_MaxRR": [9.0, 9.0, 9.0]})
    right = pd.DataFrame({LABEL: [1, 2, 3], "Bbox_MinRR": [4.0, 5.5, 6.0],
                          "Bbox_MaxRR": [9.0, 9.0, 9.0]})

    with pytest.raises(ValueError, match="Bbox_MinRR") as excinfo:
        _merge(left, right)
    assert "Bbox_MaxRR" not in str(excinfo.value)
    assert "conflicting" in str(excinfo.value)


def test_nan_against_a_value_is_a_conflict():
    left = pd.DataFrame({LABEL: [1, 2], "Grid_RowNum": [0.0, np.nan]})
    right = pd.DataFrame({LABEL: [1, 2], "Grid_RowNum": [0.0, 1.0]})

    with pytest.raises(ValueError, match="Grid_RowNum"):
        _merge(left, right)


def test_categorical_and_integer_copies_of_a_column_compare_by_value():
    """Image-info emits Grid_RowNum as a categorical; MeasureGridLinRegStats as int.

    The earlier frame's copy (and dtype) is kept. Categoricals whose category
    sets differ are compared by value too (pandas refuses ``==`` on them).
    """
    ints = pd.DataFrame({LABEL: [1, 2, 3], "Grid_RowNum": [0, 1, 1]})
    cats = pd.DataFrame({
        LABEL: [1, 2, 3],
        "Grid_RowNum": pd.Categorical([0, 1, 1], categories=[0, 1, 2],
                                      ordered=True),
    })
    other_cats = pd.DataFrame({
        LABEL: [1, 2, 3],
        "Grid_RowNum": pd.Categorical([0, 1, 1], categories=[1, 0]),
    })

    merged = _merge(ints, cats)
    assert list(merged.columns) == [LABEL, "Grid_RowNum"]
    assert merged["Grid_RowNum"].dtype == np.int64

    merged = _merge(cats, other_cats, ints)
    assert list(merged.columns) == [LABEL, "Grid_RowNum"]
    assert merged["Grid_RowNum"].dtype == cats["Grid_RowNum"].dtype

    conflicting = cats.assign(
            Grid_RowNum=pd.Categorical([0, 2, 1], categories=[0, 1, 2])
    )
    with pytest.raises(ValueError, match="Grid_RowNum"):
        _merge(ints, conflicting)


def test_disjoint_columns_merge_on_label_only_preserving_rows():
    left = pd.DataFrame({LABEL: [3, 1, 2], "A": [0.3, 0.1, 0.2]})
    right = pd.DataFrame({LABEL: [1, 2, 3], "B": [10, 20, 30]})

    merged = _merge(left, right)

    assert list(merged.columns) == [LABEL, "A", "B"]
    # Row order follows the left frame; values align by label, not position.
    assert merged[LABEL].tolist() == [3, 1, 2]
    assert merged["B"].tolist() == [30, 10, 20]
    assert merged[LABEL].dtype == left[LABEL].dtype


def test_shared_column_is_compared_by_label_not_by_position():
    """Identical values in a different row order are agreement."""
    left = pd.DataFrame({LABEL: [1, 2, 3], "Bbox_MinRR": [4.0, 5.0, 6.0]})
    right = pd.DataFrame({LABEL: [3, 1, 2], "Bbox_MinRR": [6.0, 4.0, 5.0],
                          "B": [30, 10, 20]})

    merged = _merge(left, right)

    assert list(merged.columns) == [LABEL, "Bbox_MinRR", "B"]
    assert merged["B"].tolist() == [10, 20, 30]


def test_label_indexed_frames_are_merged_on_the_label():
    left = pd.DataFrame({"A": [0.1, 0.2]},
                        index=pd.Index([1, 2], name=LABEL))
    right = pd.DataFrame({LABEL: [1, 2], "A": [0.1, 0.2], "B": [7, 8]})

    merged = _merge(left, right)

    assert list(merged.columns) == [LABEL, "A", "B"]
    assert merged[LABEL].tolist() == [1, 2]


def test_end_to_end_pipeline_has_no_merged_columns_and_keeps_every_object():
    """Image-info, MeasureBounds and MeasureGridLinRegStats all emit Bbox_*.

    On HEAD before this change the frame had 552 rows (== num_objects) and no
    ``_merged`` column for this pipeline; the fix must keep both.
    """
    image = load_synth_yeast_plate()
    pipe = ImagePipeline(
            ops=[OtsuDetector()],
            meas=[MeasureBounds(), MeasureShape(), MeasureGridLinRegStats()],
    )
    applied = pipe.apply(image)
    df = pipe.measure(applied)

    assert not [c for c in df.columns if str(c).endswith("_merged")]
    assert len(df) == applied.num_objects
    assert df[LABEL].is_unique
    assert df.columns.is_unique


# --------------------------------------------------------------------------- #
# Same-scale MeasureTexture pre-check                                         #
# --------------------------------------------------------------------------- #


def test_two_texture_measurers_at_the_same_scale_are_refused():
    with pytest.raises(ValueError, match=r"scale=5") as excinfo:
        ImagePipeline(meas=[MeasureTexture(scale=5), MeasureTexture(scale=5)])
    message = str(excinfo.value)
    assert "'MeasureTexture'" in message
    assert "'MeasureTexture_1'" in message
    assert "distinct scale" in message


def test_same_scale_different_quant_lvl_is_still_refused():
    with pytest.raises(ValueError, match=r"scale=5"):
        ImagePipeline(meas=[MeasureTexture(scale=5, quant_lvl=8),
                            MeasureTexture(scale=5, quant_lvl=32)])


def test_texture_measurers_at_distinct_scales_construct():
    pipe = ImagePipeline(meas=[MeasureTexture(scale=3), MeasureTexture(scale=5)])
    assert [m.scale for m in pipe.get_meas().values()] == [3, 5]


def test_dict_keyed_same_scale_texture_measurers_are_refused():
    with pytest.raises(ValueError, match=r"scale=4") as excinfo:
        ImagePipeline(meas={"fine": MeasureTexture(scale=4),
                            "Shape": MeasureShape(),
                            "coarse": MeasureTexture(scale=4, enhance=True)})
    assert "'fine'" in str(excinfo.value)
    assert "'coarse'" in str(excinfo.value)


def test_texture_subclass_counts_as_a_texture_measurer():
    class TunedTexture(MeasureTexture):
        pass

    with pytest.raises(ValueError, match=r"scale=5"):
        ImagePipeline(meas=[MeasureTexture(scale=5), TunedTexture(scale=5)])


def test_set_meas_refuses_same_scale_texture_measurers():
    pipe = ImagePipeline(meas=[MeasureTexture(scale=3)])
    with pytest.raises(ValueError, match=r"scale=3"):
        pipe.set_meas([MeasureTexture(scale=3), MeasureTexture(scale=3)])
    # The refused assignment leaves the pipeline unchanged.
    assert list(pipe.get_meas()) == ["MeasureTexture"]


def test_from_json_refuses_same_scale_texture_measurers():
    config = json.loads(
            ImagePipeline(
                    meas=[MeasureTexture(scale=3), MeasureTexture(scale=5)]
            ).to_json()
    )
    config["meas"]["MeasureTexture"]["params"]["scale"] = 5

    with pytest.raises(ValueError, match=r"scale=5"):
        ImagePipeline.from_json(json.dumps(config))


# --------------------------------------------------------------------------- #
# Conflict messages name the producers (phase-2 review I-2)                   #
# --------------------------------------------------------------------------- #


def test_conflict_message_names_both_producers_count_and_an_example_label():
    left = pd.DataFrame({LABEL: [1, 2, 3, 4], "Bbox_MinRR": [4.0, 5.0, 6.0, 7.0]})
    middle = pd.DataFrame({LABEL: [1, 2, 3, 4], "Shape_Area": [1.0, 2.0, 3.0, 4.0]})
    right = pd.DataFrame({LABEL: [1, 2, 3, 4], "Bbox_MinRR": [4.0, 5.5, 6.0, 7.5]})

    with pytest.raises(ValueError) as excinfo:
        _merge(left, middle, right, producers=["bounds", "shape", "image info"])
    message = str(excinfo.value)
    # The left copy belongs to the frame that first emitted the column, not to
    # whichever frame happens to precede the conflicting one.
    assert "'bounds'" in message
    assert "'image info'" in message
    assert "'shape'" not in message
    assert "Bbox_MinRR" in message
    assert "2 of 4 objects" in message
    assert f"{LABEL}=2" in message
    # Bbox_* is not a grid column, so no grid hint.
    assert "GridFinder" not in message


def test_grid_column_conflict_adds_a_grid_finder_hint():
    left = pd.DataFrame({LABEL: [1, 2], "Grid_RowNum": [0, 1]})
    right = pd.DataFrame({LABEL: [1, 2], "Grid_RowNum": [0, 2]})

    with pytest.raises(ValueError) as excinfo:
        _merge(left, right, producers=["my_grid", "image info"])
    message = str(excinfo.value)
    assert "GridFinder" in message
    assert "nrows/ncols" in message


def test_producers_must_match_the_frames_one_to_one():
    frame = pd.DataFrame({LABEL: [1], "A": [0.0]})
    with pytest.raises(ValueError, match="producer"):
        _merge(frame, frame, producers=["a"])


# --------------------------------------------------------------------------- #
# nrows/ncols preset vs a GridImage's own grid (phase-2 review I-1)           #
# --------------------------------------------------------------------------- #

GRID_HEADERS = [
    "Grid_RowNum", "Grid_ColNum", "Grid_RowMajorIdx", "Grid_ColMajorIdx",
]


@pytest.fixture(scope="module")
def detected_plate():
    """Synth plate (8x12 GridImage, CenteredAutoGridFinder) after Otsu detection."""
    image = load_synth_yeast_plate()
    OtsuDetector().apply(image, inplace=True)
    return image


def _measure(image, **preset) -> pd.DataFrame:
    pipe = ImagePipeline(meas=[MeasureBounds(), MeasureShape()], **preset)
    return pipe.measure(image.copy())


def test_gridimage_measure_ignores_a_preset_that_differs_from_its_grid(
        detected_plate,
):
    """The CLI's --nrows/--ncols reshape the image but not the pipeline preset.

    The preset used to inject a 16x24 finder whose Grid_* disagreed with the
    image's own 8x12 grid: 2 of 552 rows survived the old inner join, and the
    value-checked merge raised on every image. The image's grid is authoritative.
    """
    df = _measure(detected_plate, nrows=16, ncols=24)
    baseline = _measure(detected_plate)

    assert len(df) == detected_plate.num_objects
    own_grid = detected_plate.grid.info().set_index(LABEL)[GRID_HEADERS]
    got = df.set_index(LABEL)[GRID_HEADERS]
    pd.testing.assert_frame_equal(got, own_grid.loc[got.index])
    pd.testing.assert_frame_equal(df, baseline)


def test_gridimage_matching_preset_is_identical_to_no_preset(detected_plate):
    pd.testing.assert_frame_equal(
            _measure(detected_plate, nrows=8, ncols=12),
            _measure(detected_plate),
    )


def test_preset_still_injects_a_grid_finder_for_a_plain_image(detected_plate):
    """A plain Image has no grid of its own, so the preset stays in the run order.

    Only the run order is asserted: ``GridFinder.measure`` refuses a
    non-``GridImage``, so measuring a plain Image with a preset fails exactly as
    it did before this change.
    """
    from phenotypic import Image
    from phenotypic.grid import CenteredAutoGridFinder

    plain = Image(detected_plate.rgb[:])
    pipe = ImagePipeline(meas=[MeasureShape()], nrows=8, ncols=12)

    run_order = pipe._build_measurement_run_order(plain)
    assert list(run_order) == ["CenteredAutoGridFinder", "MeasureShape"]
    injected = run_order["CenteredAutoGridFinder"]
    assert isinstance(injected, CenteredAutoGridFinder)
    assert (injected.nrows, injected.ncols) == (8, 12)

    assert list(pipe._build_measurement_run_order(detected_plate)) == [
        "MeasureShape"
    ]


def test_explicit_grid_finder_disagreeing_with_the_image_grid_raises_clearly(
        detected_plate,
):
    """An explicit GridFinder in meas is a second grid definition: refuse it loudly."""
    from phenotypic.grid import AutoGridFinder

    finder = AutoGridFinder(nrows=8, ncols=12)
    pipe = ImagePipeline(meas={"my_grid": finder, "shape": MeasureShape()})

    # Independent witness: which objects the two grid definitions place differently.
    theirs = finder.measure(detected_plate.copy()).set_index(LABEL)[GRID_HEADERS]
    ours = detected_plate.grid.info().set_index(LABEL)[GRID_HEADERS]
    ours = ours.loc[theirs.index]
    per_column = {h: theirs[h].astype("int64").to_numpy()
                  != ours[h].astype("int64").to_numpy() for h in GRID_HEADERS}
    differs = np.logical_or.reduce(list(per_column.values()))
    n_differ = int(differs.sum())
    assert n_differ > 0, "fixture no longer exercises a grid conflict"

    with pytest.raises(ValueError) as excinfo:
        pipe.measure(detected_plate.copy())
    message = str(excinfo.value)
    assert "'my_grid'" in message
    assert "'image info'" in message
    assert "'shape'" not in message
    for header, mask in per_column.items():
        assert (header in message) == bool(mask.any())
    assert f"{n_differ} of {len(theirs)} objects" in message
    example = int(message.split(f"{LABEL}=")[1].split(")")[0])
    assert bool(differs[theirs.index.get_loc(example)])
    assert "GridFinder" in message
