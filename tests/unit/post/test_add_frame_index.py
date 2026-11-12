"""Tests for the ``AddFrameIndex`` post op.

``AddFrameIndex`` ranks each plate's *distinct* ``order_by`` values and
stamps every row of the corresponding image with that rank, **1-based**
(matching the schema's own documented ``CULTURE.FRAME_INDEX`` semantics — the
first image of a plate is frame 1). Unlike every other post op it is
cross-image: it is only correct applied once, to the whole aggregated
measurement table (see the op's own docstring for the CLI-path argument).

A pre-existing, non-empty ``Metadata_FrameIndex`` (e.g. supplied by the
user's ``--metadata`` CSV) is never touched or partially filled — see
``TestUserSuppliedFrameIndex``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from phenotypic.post import AddFrameIndex
from phenotypic.schema import CULTURE

FRAME_INDEX = str(CULTURE.FRAME_INDEX)

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "frame_index"
    / "synthetic_frame_index.parquet"
)
#: The reference's own 0-based frame index, recorded beside the inputs.
_REFERENCE = "reference_frame_index"


def test_two_plates_ranked_independently_and_out_of_order():
    """Each plate's images rank by ``order_by``, not by row order in the frame."""
    df = pd.DataFrame(
        {
            "Metadata_PlateID": ["P2", "P1", "P1", "P2", "P1"],
            "Metadata_ImageName": ["p2_t1", "p1_t1", "p1_t0", "p2_t0", "p1_t2"],
            "Object_Label": [1, 1, 1, 1, 1],
        }
    )
    op = AddFrameIndex(plate_by=["PlateID"])

    result = op.apply(df)

    expected = {
        "p1_t0": 1,
        "p1_t1": 2,
        "p1_t2": 3,
        "p2_t0": 1,
        "p2_t1": 2,
    }
    for image_name, index in expected.items():
        rows = result[result["Metadata_ImageName"] == image_name]
        assert (rows[FRAME_INDEX] == index).all()


def test_duplicate_rows_per_image_share_one_index():
    """Multiple objects on the same image all get that image's rank."""
    df = pd.DataFrame(
        {
            "Metadata_PlateID": ["P1", "P1", "P1", "P1"],
            "Metadata_ImageName": ["p1_t0", "p1_t0", "p1_t1", "p1_t1"],
            "Object_Label": [1, 2, 1, 2],
        }
    )
    op = AddFrameIndex(plate_by=["PlateID"])

    result = op.apply(df)

    assert result.loc[result["Metadata_ImageName"] == "p1_t0", FRAME_INDEX].tolist() == [
        1,
        1,
    ]
    assert result.loc[result["Metadata_ImageName"] == "p1_t1", FRAME_INDEX].tolist() == [
        2,
        2,
    ]


def test_null_plate_value_gets_blank_index_not_a_false_group():
    """A row with no plate value is excluded from ranking, not pooled under NA."""
    df = pd.DataFrame(
        {
            "Metadata_PlateID": ["P1", None, "P1"],
            "Metadata_ImageName": ["p1_t0", "orphan", "p1_t1"],
            "Object_Label": [1, 1, 1],
        }
    )
    op = AddFrameIndex(plate_by=["PlateID"])

    result = op.apply(df)

    assert pd.isna(result.loc[1, FRAME_INDEX])
    assert result.loc[0, FRAME_INDEX] == 1
    assert result.loc[2, FRAME_INDEX] == 2


def test_null_order_by_value_gets_blank_index():
    """A row with no order_by value is excluded from ranking too."""
    df = pd.DataFrame(
        {
            "Metadata_PlateID": ["P1", "P1", "P1"],
            "Metadata_ImageName": ["p1_t0", None, "p1_t1"],
            "Object_Label": [1, 1, 1],
        }
    )
    op = AddFrameIndex(plate_by=["PlateID"])

    result = op.apply(df)

    assert pd.isna(result.loc[1, FRAME_INDEX])
    assert result.loc[0, FRAME_INDEX] == 1
    assert result.loc[2, FRAME_INDEX] == 2


def test_frame_index_is_nullable_integer_dtype():
    df = pd.DataFrame(
        {
            "Metadata_PlateID": ["P1", "P1"],
            "Metadata_ImageName": ["p1_t0", "p1_t1"],
            "Object_Label": [1, 1],
        }
    )
    op = AddFrameIndex(plate_by=["PlateID"])

    result = op.apply(df)

    assert result[FRAME_INDEX].dtype == pd.Int64Dtype()


def test_applying_twice_is_idempotent():
    """The second apply sees its own output already populated and leaves it be."""
    df = pd.DataFrame(
        {
            "Metadata_PlateID": ["P1", "P1", "P2", "P2"],
            "Metadata_ImageName": ["p1_t1", "p1_t0", "p2_t0", "p2_t1"],
            "Object_Label": [1, 1, 1, 1],
        }
    )
    op = AddFrameIndex(plate_by=["PlateID"])

    once = op.apply(df)
    twice = op.apply(once)

    pd.testing.assert_frame_equal(once, twice)


def test_multi_column_plate_by():
    """plate_by can be more than one column identifying the plate."""
    df = pd.DataFrame(
        {
            "Metadata_Dataset": ["A", "A", "A", "B"],
            "Metadata_PlateID": ["P1", "P1", "P1", "P1"],
            "Metadata_ImageName": ["t1", "t0", "t2", "t0"],
            "Object_Label": [1, 1, 1, 1],
        }
    )
    op = AddFrameIndex(plate_by=["Dataset", "PlateID"])

    result = op.apply(df)

    a_rows = result[result["Metadata_Dataset"] == "A"].set_index("Metadata_ImageName")
    assert a_rows.loc["t0", FRAME_INDEX] == 1
    assert a_rows.loc["t1", FRAME_INDEX] == 2
    assert a_rows.loc["t2", FRAME_INDEX] == 3
    # A different dataset/plate combination ranks independently, starting at
    # 1 even though it shares a PlateID with group A.
    assert result.loc[result["Metadata_Dataset"] == "B", FRAME_INDEX].iloc[0] == 1


def test_missing_column_raises_key_error():
    df = pd.DataFrame(
        {
            "Metadata_ImageName": ["p1_t0"],
            "Object_Label": [1],
        }
    )
    op = AddFrameIndex(plate_by=["PlateID"])

    with pytest.raises(KeyError):
        op.apply(df)


def test_empty_plate_by_raises_value_error_when_deriving():
    df = pd.DataFrame(
        {
            "Metadata_PlateID": ["P1"],
            "Metadata_ImageName": ["p1_t0"],
            "Object_Label": [1],
        }
    )
    op = AddFrameIndex(plate_by=[])

    with pytest.raises(ValueError, match="plate_by"):
        op.apply(df)


def test_schema_member_resolves():
    from phenotypic.sdk_ import metadata_member_for_header

    assert metadata_member_for_header(FRAME_INDEX) is CULTURE.FRAME_INDEX


class TestUserSuppliedFrameIndex:
    """A populated ``Metadata_FrameIndex`` is authoritative and untouched."""

    def test_absent_column_is_derived(self):
        df = pd.DataFrame(
            {
                "Metadata_PlateID": ["P1", "P1"],
                "Metadata_ImageName": ["p1_t0", "p1_t1"],
                "Object_Label": [1, 1],
            }
        )
        assert FRAME_INDEX not in df.columns
        op = AddFrameIndex(plate_by=["PlateID"])

        result = op.apply(df)

        assert result[FRAME_INDEX].tolist() == [1, 2]

    def test_entirely_null_column_is_derived(self):
        df = pd.DataFrame(
            {
                "Metadata_PlateID": ["P1", "P1"],
                "Metadata_ImageName": ["p1_t0", "p1_t1"],
                FRAME_INDEX: pd.array([pd.NA, pd.NA], dtype="Int64"),
                "Object_Label": [1, 1],
            }
        )
        op = AddFrameIndex(plate_by=["PlateID"])

        result = op.apply(df)

        assert result[FRAME_INDEX].tolist() == [1, 2]

    def test_populated_column_is_left_untouched_and_not_recomputed(self):
        """A user-supplied index (even one that disagrees with a derived one)
        survives byte-for-byte, and derivation never runs."""
        df = pd.DataFrame(
            {
                "Metadata_PlateID": ["P1", "P1"],
                # Out-of-capture-order names: a derived index would give
                # [2, 1], not [1, 2] -- proof that derivation did not run.
                "Metadata_ImageName": ["p1_t1", "p1_t0"],
                FRAME_INDEX: pd.array([1, 2], dtype="Int64"),
                "Object_Label": [1, 1],
            }
        )
        op = AddFrameIndex(plate_by=["PlateID"])

        result = op.apply(df)

        pd.testing.assert_series_equal(result[FRAME_INDEX], df[FRAME_INDEX])

    def test_partially_populated_column_is_left_untouched(self):
        """Any non-null value means the whole column is user-owned -- no
        filling in just the null rows."""
        df = pd.DataFrame(
            {
                "Metadata_PlateID": ["P1", "P1", "P1"],
                "Metadata_ImageName": ["p1_t0", "p1_t1", "p1_t2"],
                FRAME_INDEX: pd.array([1, pd.NA, pd.NA], dtype="Int64"),
                "Object_Label": [1, 1, 1],
            }
        )
        op = AddFrameIndex(plate_by=["PlateID"])

        result = op.apply(df)

        pd.testing.assert_series_equal(result[FRAME_INDEX], df[FRAME_INDEX])

    def test_populated_column_skips_plate_by_validation_entirely(self):
        """When nothing needs deriving, an invalid plate_by never surfaces."""
        df = pd.DataFrame(
            {
                # No Metadata_PlateID at all -- would KeyError if derivation ran.
                "Metadata_ImageName": ["p1_t0", "p1_t1"],
                FRAME_INDEX: pd.array([1, 2], dtype="Int64"),
                "Object_Label": [1, 1],
            }
        )
        op = AddFrameIndex(plate_by=["PlateID"])

        result = op.apply(df)

        assert result[FRAME_INDEX].tolist() == [1, 2]


@pytest.fixture(scope="module")
def reference_rows() -> pd.DataFrame:
    """Synthetic plates with the reference's own frame index (fixture README)."""
    return pd.read_parquet(_FIXTURE)


def test_the_fixture_exercises_what_it_claims(reference_rows: pd.DataFrame):
    """Shuffled rows, a phantom per image, and one plate that skips a capture."""
    per_image = reference_rows.drop_duplicates(
        ["Metadata_PlateID", "Metadata_ImageName"]
    )
    assert reference_rows["Object_Label"].isna().sum() == len(per_image)
    capture_order = reference_rows.sort_values(
        ["Metadata_PlateID", "Metadata_ImageDatetime"]
    ).index
    assert not capture_order.equals(reference_rows.index)
    # A skipped capture shows as a gap of about 8 h between consecutive images.
    times = pd.to_datetime(per_image["Metadata_ImageDatetime"])
    gaps = (
        per_image.assign(_t=times)
        .sort_values(["Metadata_PlateID", "_t"])
        .groupby("Metadata_PlateID")["_t"]
        .diff()
        .dropna()
    )
    assert (gaps > pd.Timedelta(hours=6)).sum() == 1


def test_parity_with_the_reference_on_synthetic_plates(reference_rows: pd.DataFrame):
    """Our 1-based index is exactly one more than the reference's, per plate-image.

    The reference ranks a plate's images by parsed capture time; this op ranks
    them by image name. On names that end in an ISO timestamp the two orders
    are the same, which is the shape the op's default ``order_by`` assumes.
    """
    assert FRAME_INDEX not in reference_rows.columns, (
        "fixture must not already carry Metadata_FrameIndex, or derivation "
        "would be skipped and this test would prove nothing"
    )

    op = AddFrameIndex(plate_by=["PlateID"], order_by="ImageName")
    result = op.apply(reference_rows)

    ours = result[FRAME_INDEX].astype("Int64")
    theirs_zero_based = result[_REFERENCE].astype("Int64")
    mismatches = result.loc[ours != theirs_zero_based + 1]
    assert mismatches.empty, (
        f"{len(mismatches)} rows disagree with the reference + 1:\n"
        f"{mismatches[['Metadata_PlateID', 'Metadata_ImageName']].head(10)}"
    )


def test_the_synthetic_window_series_gets_thirty_frames_per_plate():
    """The fitting-window example's own input: every plate numbers 1 to 30.

    Its image names carry zero-padded hours, so name order is time order and
    the frame index is the time rank on every plate.
    """
    from phenotypic.data import make_synthetic_window_series

    table = AddFrameIndex(plate_by=["PlateID"]).apply(make_synthetic_window_series())
    per_plate = table.groupby("Metadata_PlateID")[FRAME_INDEX]
    assert per_plate.min().eq(1).all()
    assert per_plate.max().eq(30).all()
    rank = table.groupby("Metadata_PlateID")["Metadata_Time"].rank(method="dense")
    assert (table[FRAME_INDEX] == rank.astype("Int64")).all()


def test_doctest_example_in_add_frame_index_module():
    import doctest

    import phenotypic.post._add_frame_index as module

    results = doctest.testmod(module, verbose=False)
    assert results.failed == 0
