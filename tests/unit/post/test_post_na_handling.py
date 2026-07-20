"""NA handling for the metadata post-measurement transforms.

A left-joined ``--metadata`` frame carries "phantom rows": expected wells whose
colony was never detected, so every measurement column is null. These four ops
run over that frame. Two invariants are pinned here for each op:

1. **NA propagates as NA.** Never the literal string ``"nan"`` (which would
   look like a real value and silently group unrelated rows), never a crash.
2. **No-op when nothing is NA.** An NA-free frame must come out byte-identical
   to what the pre-fix code produced, so existing runs' outputs do not move.
"""

import numpy as np
import pandas as pd
import pytest

from phenotypic.post import AppendString, ExpandMetadata, MergeMetadata, PrependString


def _assert_no_literal_nan_string(series: pd.Series) -> None:
    """Fail if any cell is a string that stringified an NA (e.g. ``'nan_A1'``)."""
    offenders = [
        value
        for value in series
        if isinstance(value, str) and "nan" in value.lower()
    ]
    assert not offenders, f"NA was stringified into: {offenders!r}"


class TestMergeMetadataNA:
    """MergeMetadata must not forge a group key out of a missing value."""

    def test_na_source_yields_na_key(self):
        """A row with an NA in any source column merges to NaN, not 'nan_A1'."""
        df = pd.DataFrame(
            {
                "Metadata_A": ["WT", None, "mut", np.nan],
                "Metadata_B": ["A1", "A2", None, np.nan],
                "Object_Label": [1, 2, 3, 4],
            }
        )
        op = MergeMetadata(columns=["A", "B"], label="Key", delimiter="_")

        merged = op.apply(df)["Metadata_Key"]

        _assert_no_literal_nan_string(merged)
        assert merged.iloc[0] == "WT_A1"
        assert merged.isna().tolist() == [False, True, True, True]

    def test_all_na_column_yields_all_na(self):
        """A fully-null source column (a phantom measurement) nulls every key."""
        df = pd.DataFrame(
            {
                "Metadata_A": ["WT", "mut"],
                "Metadata_B": [np.nan, np.nan],
                "Object_Label": [1, 2],
            }
        )
        op = MergeMetadata(columns=["A", "B"], label="Key")

        merged = op.apply(df)["Metadata_Key"]

        assert merged.isna().all()

    def test_three_columns_any_na_masks(self):
        """The mask ORs across *all* sources, not just the first two."""
        df = pd.DataFrame(
            {
                "Metadata_A": ["WT", "WT"],
                "Metadata_B": ["A1", "A1"],
                "Metadata_C": ["x", None],
                "Object_Label": [1, 2],
            }
        )
        op = MergeMetadata(columns=["A", "B", "C"], label="Key")

        merged = op.apply(df)["Metadata_Key"]

        assert merged.iloc[0] == "WT_A1_x"
        assert pd.isna(merged.iloc[1])

    def test_no_op_without_na(self):
        """An NA-free frame merges exactly as before the NA masking was added."""
        df = pd.DataFrame(
            {
                "Metadata_A": ["WT", "mut", "ko"],
                "Metadata_B": ["A1", "B2", "C3"],
                "Object_Label": [1, 2, 3],
            }
        )
        op = MergeMetadata(columns=["A", "B"], label="Key", delimiter="-")

        result = op.apply(df)

        expected = df.copy()
        expected.insert(2, "Metadata_Key", ["WT-A1", "mut-B2", "ko-C3"])
        pd.testing.assert_frame_equal(result, expected)


class TestAppendStringNA:
    """AppendString must not turn a missing value into 'nanC'."""

    def test_na_stays_na(self):
        df = pd.DataFrame(
            {
                "Metadata_T": ["30", None, np.nan],
                "Object_Label": [1, 2, 3],
            }
        )
        op = AppendString(column="T", value="C")

        col = op.apply(df)["Metadata_T"]

        _assert_no_literal_nan_string(col)
        assert col.iloc[0] == "30C"
        assert col.isna().tolist() == [False, True, True]

    def test_numeric_na_column_stays_na(self):
        """A float measurement column (the phantom-row shape) is masked, too."""
        df = pd.DataFrame(
            {
                "Metadata_T": [30.0, np.nan],
                "Object_Label": [1, 2],
            }
        )
        op = AppendString(column="T", value="C")

        col = op.apply(df)["Metadata_T"]

        _assert_no_literal_nan_string(col)
        assert col.iloc[0] == "30.0C"
        assert pd.isna(col.iloc[1])

    def test_no_op_without_na(self):
        df = pd.DataFrame(
            {
                "Metadata_T": ["30", "37"],
                "Object_Label": [1, 2],
            }
        )
        op = AppendString(column="T", value="C")

        result = op.apply(df)

        expected = pd.DataFrame(
            {
                "Metadata_T": ["30C", "37C"],
                "Object_Label": [1, 2],
            }
        )
        pd.testing.assert_frame_equal(result, expected)


class TestPrependStringNA:
    """PrependString must not turn a missing value into 'WT-nan'."""

    def test_na_stays_na(self):
        df = pd.DataFrame(
            {
                "Metadata_S": ["001", None, np.nan],
                "Object_Label": [1, 2, 3],
            }
        )
        op = PrependString(column="S", value="WT-")

        col = op.apply(df)["Metadata_S"]

        _assert_no_literal_nan_string(col)
        assert col.iloc[0] == "WT-001"
        assert col.isna().tolist() == [False, True, True]

    def test_no_op_without_na(self):
        df = pd.DataFrame(
            {
                "Metadata_S": ["001", "002"],
                "Object_Label": [1, 2],
            }
        )
        op = PrependString(column="S", value="WT-")

        result = op.apply(df)

        expected = pd.DataFrame(
            {
                "Metadata_S": ["WT-001", "WT-002"],
                "Object_Label": [1, 2],
            }
        )
        pd.testing.assert_frame_equal(result, expected)


class TestExpandMetadataNA:
    """ExpandMetadata hard-crashed on NA in both branches; it must not."""

    @pytest.mark.parametrize("regex", [False, True])
    def test_na_row_expands_to_na(self, regex):
        """NA rows are excluded from the split and reindexed back as NaN.

        Without the fix: ``regex=True`` splits ``str(nan)`` -> ``['nan']`` -> 1
        part -> arity ValueError; ``regex=False`` gets NaN (not a list) from
        ``.str.split`` -> ``TypeError: object of type 'float' has no len()``.
        """
        df = pd.DataFrame(
            {
                "Metadata_Name": ["WT_30C_24h", None, "mut_37C_48h"],
                "Object_Label": [1, 2, 3],
            }
        )
        op = ExpandMetadata(
            column="Name",
            labels=["Strain", "Cond", "Time"],
            delimiter="_",
            regex=regex,
        )

        result = op.apply(df)

        for label in ["MetadataGenetic_Strain", "Metadata_Cond", "MetadataCulture_Time"]:
            _assert_no_literal_nan_string(result[label])
            assert pd.isna(result.loc[1, label]), f"{label} row 1 should be NaN"
        assert list(result["MetadataGenetic_Strain"]) [::2] == ["WT", "mut"]
        assert result.loc[0, "MetadataCulture_Time"] == "24h"
        assert result.loc[2, "Metadata_Cond"] == "37C"

    @pytest.mark.parametrize("regex", [False, True])
    def test_all_na_column_expands_to_all_na(self, regex):
        """A fully-null source column yields all-NaN outputs, not an arity error."""
        df = pd.DataFrame(
            {
                "Metadata_Name": [None, np.nan],
                "Object_Label": [1, 2],
            }
        )
        op = ExpandMetadata(
            column="Name", labels=["Strain", "Cond"], delimiter="_", regex=regex
        )

        result = op.apply(df)

        assert result["MetadataGenetic_Strain"].isna().all()
        assert result["Metadata_Cond"].isna().all()

    def test_na_rows_do_not_mask_a_real_arity_error(self):
        """A present-but-malformed value still raises; NA exclusion is not a blanket skip."""
        df = pd.DataFrame(
            {
                "Metadata_Name": ["WT_30C", None, "mut_37C_48h"],
                "Object_Label": [1, 2, 3],
            }
        )
        op = ExpandMetadata(column="Name", labels=["Strain", "Cond", "Time"])

        with pytest.raises(ValueError, match="produced 2 parts for value 'WT_30C'"):
            op.apply(df)

    def test_arity_error_reports_the_right_row_on_a_string_index(self):
        """The error message uses positional lookup, not a label passed to .iloc.

        With the old ``df[col].iloc[first_bad_idx]``, a non-integer index made
        this raise the wrong exception type entirely.
        """
        df = pd.DataFrame(
            {
                "Metadata_Name": ["WT_30C_24h", "BROKEN"],
                "Object_Label": [1, 2],
            },
            index=["row_a", "row_b"],
        )
        op = ExpandMetadata(column="Name", labels=["Strain", "Cond", "Time"])

        with pytest.raises(ValueError, match="produced 1 parts for value 'BROKEN'"):
            op.apply(df)

    def test_arity_error_reports_the_right_row_after_na_exclusion(self):
        """Positions shift once NA rows drop out; the reported value must still be right."""
        df = pd.DataFrame(
            {
                "Metadata_Name": [None, None, "BROKEN"],
                "Object_Label": [1, 2, 3],
            }
        )
        op = ExpandMetadata(column="Name", labels=["Strain", "Cond"])

        with pytest.raises(ValueError, match="produced 1 parts for value 'BROKEN'"):
            op.apply(df)

    @pytest.mark.parametrize("regex", [False, True])
    def test_no_op_without_na(self, regex):
        """An NA-free frame expands exactly as before the NA exclusion was added."""
        df = pd.DataFrame(
            {
                "Metadata_Name": ["WT_30C_24h", "mut_37C_48h"],
                "Object_Label": [1, 2],
            }
        )
        op = ExpandMetadata(
            column="Name",
            labels=["Strain", "Cond", "Time"],
            delimiter="_",
            regex=regex,
        )

        result = op.apply(df)

        expected = pd.DataFrame(
            {
                "Metadata_Name": ["WT_30C_24h", "mut_37C_48h"],
                "MetadataGenetic_Strain": ["WT", "mut"],
                "Metadata_Cond": ["30C", "37C"],
                "MetadataCulture_Time": ["24h", "48h"],
                "Object_Label": [1, 2],
            }
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_no_op_preserves_non_default_index(self):
        """Reindexing NA rows back must not disturb a non-RangeIndex frame."""
        df = pd.DataFrame(
            {
                "Metadata_Name": ["WT_30C", "mut_37C"],
                "Object_Label": [1, 2],
            },
            index=["row_a", "row_b"],
        )
        op = ExpandMetadata(column="Name", labels=["Strain", "Cond"])

        result = op.apply(df)

        assert list(result.index) == ["row_a", "row_b"]
        assert result.loc["row_b", "MetadataGenetic_Strain"] == "mut"
