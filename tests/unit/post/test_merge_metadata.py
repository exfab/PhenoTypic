"""Tests for MergeMetadata post-measurement transform."""

import pytest
import pandas as pd

from phenotypic.post import MergeMetadata


class TestMergeMetadataInit:
    """Test construction and parameter validation."""

    def test_basic_construction(self):
        """MergeMetadata can be created with columns, label, delimiter."""
        mm = MergeMetadata(columns=["Strain", "Condition"], label="SampleID", delimiter="_")
        assert mm.columns == ["Metadata_Strain", "Metadata_Condition"]
        assert mm.label == "Metadata_SampleID"
        assert mm.delimiter == "_"

    def test_auto_prepends_metadata_prefix(self):
        """Column names and label get Metadata_ prefix if missing."""
        mm = MergeMetadata(columns=["A", "B"], label="AB")
        assert mm.columns == ["Metadata_A", "Metadata_B"]
        assert mm.label == "Metadata_AB"

    def test_no_double_prefix(self):
        """Already-prefixed names are not double-prefixed."""
        mm = MergeMetadata(
            columns=["Metadata_Strain", "Condition"],
            label="Metadata_SampleID",
        )
        assert mm.columns == ["Metadata_Strain", "Metadata_Condition"]
        assert mm.label == "Metadata_SampleID"

    def test_empty_columns_accepted_as_unset(self):
        """An empty columns list is the valid 'unset' state, not an error.

        Only a genuinely-invalid single-column merge raises (see
        ``test_single_column_raises``); the empty list is the field
        default and must validate so ``model_dump`` / ``model_validate``
        round-trip and assignment work.
        """
        mm = MergeMetadata(columns=[], label="SampleID")
        assert mm.columns == []

    def test_single_column_raises(self):
        """Single column raises ValueError -- need at least 2 to merge."""
        with pytest.raises(ValueError, match="columns"):
            MergeMetadata(columns=["A"], label="SampleID")


class TestMergeMetadataOperate:
    """Test the merge operation."""

    def test_basic_merge(self):
        """Merge two metadata columns into one."""
        mm = MergeMetadata(columns=["Strain", "Condition"], label="SampleID", delimiter="_")
        df = pd.DataFrame({
            "Metadata_Strain": ["WT", "mut"],
            "Metadata_Condition": ["30C", "37C"],
            "Object_Label": [1, 2],
            "Shape_Area": [100, 200],
        })
        result = mm.apply(df)
        assert "Metadata_SampleID" in result.columns
        assert list(result["Metadata_SampleID"]) == ["WT_30C", "mut_37C"]

    def test_original_columns_kept(self):
        """Source columns are always preserved."""
        mm = MergeMetadata(columns=["A", "B"], label="AB", delimiter="-")
        df = pd.DataFrame({
            "Metadata_A": ["x", "y"],
            "Metadata_B": ["1", "2"],
            "Object_Label": [1, 2],
        })
        result = mm.apply(df)
        assert "Metadata_A" in result.columns
        assert "Metadata_B" in result.columns

    def test_missing_column_raises(self):
        """Raises KeyError when a source column doesn't exist."""
        mm = MergeMetadata(columns=["A", "NonExistent"], label="AB")
        df = pd.DataFrame({"Metadata_A": ["x"], "Object_Label": [1]})
        with pytest.raises(KeyError):
            mm.apply(df)

    def test_custom_delimiter(self):
        """Custom delimiter is used in joined string."""
        mm = MergeMetadata(columns=["A", "B"], label="AB", delimiter="::")
        df = pd.DataFrame({
            "Metadata_A": ["x", "y"],
            "Metadata_B": ["1", "2"],
            "Object_Label": [1, 2],
        })
        result = mm.apply(df)
        assert list(result["Metadata_AB"]) == ["x::1", "y::2"]

    def test_three_columns(self):
        """Merge three columns into one."""
        mm = MergeMetadata(columns=["A", "B", "C"], label="ABC", delimiter="_")
        df = pd.DataFrame({
            "Metadata_A": ["x"],
            "Metadata_B": ["y"],
            "Metadata_C": ["z"],
            "Object_Label": [1],
        })
        result = mm.apply(df)
        assert list(result["Metadata_ABC"]) == ["x_y_z"]

    def test_new_column_inserted_after_last_source(self):
        """New column appears after the last source column."""
        mm = MergeMetadata(columns=["A", "B"], label="AB", delimiter="_")
        df = pd.DataFrame({
            "Metadata_A": ["x"],
            "Object_Label": [1],
            "Metadata_B": ["y"],
            "Shape_Area": [100],
        })
        result = mm.apply(df)
        cols = list(result.columns)
        last_src_idx = max(cols.index("Metadata_A"), cols.index("Metadata_B"))
        assert cols[last_src_idx + 1] == "Metadata_AB"
