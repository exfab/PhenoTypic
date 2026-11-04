"""The shared expected-positions helper.

Two QC checks derive "how many colonies should this group have" from the frame
they are handed. This module pins the two guards that stop that derivation from
answering when the frame cannot actually say — because both failure modes are
**silent** rather than loud: without them, expected would equal detected and
every group would pass.
"""

from __future__ import annotations

import pandas as pd
import pytest

from phenotypic.analysis.qc._expected_positions import (
    DEFAULT_POSITION_BY,
    expected_positions,
    numeric_order_key,
)
from phenotypic.schema import GRID, IMAGE, METADATA_MATCH, OBJECT

IMAGE_NAME = str(IMAGE.IMAGE_NAME)
ROW_NUM = str(GRID.ROW_NUM)
COL_NUM = str(GRID.COL_NUM)
LABEL = str(OBJECT.LABEL)
FLAG = str(METADATA_MATCH.METADATA_ONLY)


def _joined_frame() -> pd.DataFrame:
    """A CLI-shaped left join: four positions, one of them a phantom."""
    return pd.DataFrame({
        IMAGE_NAME: ["p1.png", "p1.png", "p1.png", "p2.png"],
        ROW_NUM: [1, 1, 2, 1],
        COL_NUM: [1, 2, 1, 1],
        LABEL: [1.0, 2.0, None, 3.0],
        FLAG: [False, False, True, False],
    })


def test_the_default_positions_are_spelled_through_the_schema():
    """Image name plus grid cell — asked of the schema, not hard-coded."""
    assert DEFAULT_POSITION_BY == (IMAGE_NAME, ROW_NUM, COL_NUM)


def test_one_row_per_distinct_position():
    positions = expected_positions(_joined_frame())
    assert positions is not None
    assert len(positions) == 4
    assert list(positions.index) == [0, 1, 2, 3]


def test_a_phantom_counts_toward_expected():
    """It carries its position; only its detection is missing."""
    positions = expected_positions(_joined_frame())
    assert positions is not None
    phantom = positions.loc[positions[FLAG]]
    assert len(phantom) == 1
    assert phantom[LABEL].isna().all()


def test_two_objects_in_one_well_count_once_toward_expected():
    """A doublet is one expected position and two detections.

    Collapsing it on the expected side is the whole point: the divergence
    between the two counts is what a count check exists to show, so expected
    must not inherit the detected side's double-count.
    """
    frame = _joined_frame()
    doublet = frame.iloc[[0]].copy()
    doublet[LABEL] = 99.0
    frame = pd.concat([frame, doublet], ignore_index=True)

    positions = expected_positions(frame)
    assert positions is not None
    assert len(frame) == 5
    assert len(positions) == 4


def test_guard_1_returns_none_without_the_metadata_only_column():
    """A frame that never went through the metadata join cannot say.

    Returned rather than raised: a notebook ``image.measure()`` frame
    legitimately has no phantoms, and only the calling check knows whether that
    is fatal for it.
    """
    frame = _joined_frame().drop(columns=[FLAG])
    assert expected_positions(frame) is None


def test_guard_2_refuses_a_row_with_a_blank_position():
    """Per-image metadata's phantoms carry an image name and no grid cell."""
    frame = _joined_frame()
    frame.loc[2, COL_NUM] = None
    with pytest.raises(ValueError, match="blank value"):
        expected_positions(frame)


def test_guard_2_refuses_a_whitespace_only_position_value():
    """A CSV round-trip turns an empty field into ``""``, not into NaN."""
    frame = _joined_frame()
    frame[ROW_NUM] = frame[ROW_NUM].astype(str)
    frame.loc[3, ROW_NUM] = "  "
    with pytest.raises(ValueError, match="blank value"):
        expected_positions(frame)


def test_guard_2_calls_a_blank_phantom_metadata_at_the_wrong_grain():
    frame = _joined_frame()
    frame.loc[2, COL_NUM] = None  # row 2 is the phantom
    with pytest.raises(ValueError) as excinfo:
        expected_positions(frame)
    message = str(excinfo.value)
    assert "1 row(s) with no detected object" in message
    assert "wrong grain" in message
    assert "off-grid" not in message


def test_guard_2_calls_a_blank_detection_an_off_grid_object():
    """A detected object with no grid cell is not a metadata problem."""
    frame = _joined_frame()
    frame.loc[0, ROW_NUM] = None  # row 0 is a detection
    with pytest.raises(ValueError) as excinfo:
        expected_positions(frame)
    message = str(excinfo.value)
    assert "1 detected object(s)" in message
    assert "off-grid" in message
    assert "wrong grain" not in message


def test_guard_2_names_both_causes_when_both_occur():
    frame = _joined_frame()
    frame.loc[0, ROW_NUM] = None
    frame.loc[2, COL_NUM] = None
    with pytest.raises(ValueError) as excinfo:
        expected_positions(frame)
    message = str(excinfo.value)
    assert "1 row(s) with no detected object" in message
    assert "1 detected object(s)" in message


def test_guard_2_without_a_label_column_cannot_tell_the_causes_apart():
    frame = _joined_frame().drop(columns=[LABEL])
    frame.loc[0, ROW_NUM] = None
    with pytest.raises(ValueError, match="1 row\\(s\\) carry a blank value") as excinfo:
        expected_positions(frame)
    message = str(excinfo.value)
    assert "off-grid" not in message
    assert "wrong grain" not in message


def test_guard_2_refuses_rather_than_dropping():
    """The refusal is the point: dropping the row shrinks the denominator."""
    frame = _joined_frame()
    frame.loc[0, ROW_NUM] = None
    with pytest.raises(ValueError):
        expected_positions(frame)


def test_a_missing_position_column_raises_key_error():
    frame = _joined_frame().drop(columns=[COL_NUM])
    with pytest.raises(KeyError, match=COL_NUM):
        expected_positions(frame)


def test_an_empty_position_by_is_refused():
    with pytest.raises(ValueError, match="at least one column"):
        expected_positions(_joined_frame(), [])


def test_an_explicit_position_by_overrides_the_default():
    """Keyed on the image alone, the four rows collapse to two images."""
    positions = expected_positions(_joined_frame(), [IMAGE_NAME])
    assert positions is not None
    assert len(positions) == 2


def test_the_result_is_a_copy():
    frame = _joined_frame()
    positions = expected_positions(frame)
    assert positions is not None
    positions.loc[0, ROW_NUM] = 999
    assert frame.loc[0, ROW_NUM] == 1


# --- numeric_order_key --------------------------------------------------------


def test_a_text_index_is_ordered_numerically_not_lexicographically():
    """The defect this helper exists for: ``"10"`` must not sort before ``"2"``."""
    values = pd.Series(["2", "6", "10"])

    ordered = numeric_order_key(values)

    assert ordered.tolist() == [2, 6, 10]
    assert sorted(values.tolist()) == ["10", "2", "6"]
    assert sorted(ordered.tolist()) == [2, 6, 10]


def test_a_dictionary_encoded_index_is_ordered_numerically():
    """A low-cardinality parquet column reads back as a Categorical.

    A categorical sorts by its *category* order, which for strings round-
    tripped through parquet is the same lexicographic order as the raw text —
    so decoding it first is what keeps the coercion effective.
    """
    values = pd.Series(pd.Categorical(["2", "6", "10"]))

    assert numeric_order_key(values).tolist() == [2, 6, 10]


def test_a_datetime_index_is_left_alone():
    """``to_numeric`` would publish 1.7e18 where the user wrote a timestamp.

    A datetime column already sorts chronologically, so coercing it fixes
    nothing and destroys the frame labels of every table built on it. Only
    text is converted.
    """
    values = pd.Series(pd.to_datetime(["2026-01-02", "2026-01-10", "2026-01-03"]))

    ordered = numeric_order_key(values)

    assert ordered is values
    assert pd.api.types.is_datetime64_any_dtype(ordered)


def test_a_timedelta_index_is_left_alone():
    values = pd.Series(pd.to_timedelta([2, 10, 3], unit="h"))

    assert numeric_order_key(values) is values


def test_a_genuinely_categorical_axis_is_left_alone():
    """``"pre"``/``"day1"`` must keep ordering the way it always has.

    And a downstream numeric guard must still see the user's own values
    rather than a column of nulls this helper made.
    """
    values = pd.Series(["pre", "day1", "day10"])

    assert numeric_order_key(values) is values


def test_an_already_numeric_index_is_returned_unchanged():
    values = pd.Series([2, 6, 10])

    assert numeric_order_key(values) is values


def test_the_module_doctests_run():
    """The examples in the helper's docstring are executable."""
    import doctest

    from phenotypic.analysis.qc import _expected_positions

    results = doctest.testmod(_expected_positions, verbose=False)
    assert results.failed == 0
    assert results.attempted > 0
