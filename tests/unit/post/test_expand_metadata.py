"""Tests for ExpandMetadata post-measurement transform."""

import pytest
import pandas as pd

from phenotypic.post import ExpandMetadata


class TestExpandMetadataInit:
    """Test construction and parameter validation."""

    def test_basic_construction(self):
        """ExpandMetadata can be created with column, labels, delimiter."""
        em = ExpandMetadata(column="ImageName", labels=["Strain", "Cond"], delimiter="_")
        assert em.column == "MetadataImage_ImageName"
        assert em.labels == ["MetadataGenetic_Strain", "Metadata_Cond"]
        assert em.delimiter == "_"

    def test_auto_prepends_metadata_prefix(self):
        """Column and labels get Metadata_ prefix if missing."""
        em = ExpandMetadata(column="FileName", labels=["A", "B"])
        assert em.column == "Metadata_FileName"
        assert em.labels == ["Metadata_A", "Metadata_B"]

    def test_no_double_prefix(self):
        """Already-prefixed names are not double-prefixed."""
        em = ExpandMetadata(
            column="MetadataImage_ImageName",
            labels=["MetadataGenetic_Strain", "Condition"],
        )
        assert em.column == "MetadataImage_ImageName"
        assert em.labels == ["MetadataGenetic_Strain", "Metadata_Condition"]

    def test_empty_labels_accepted_as_unset(self):
        """An empty labels list is the valid 'unset' state, not an error.

        The empty list is the field default and must validate so
        ``model_dump`` / ``model_validate`` round-trip and assignment
        work; a genuinely-misconfigured (empty-labels) ExpandMetadata
        fails later, at ``apply()`` time, via the split-count check.
        """
        em = ExpandMetadata(column="ImageName", labels=[])
        assert em.labels == []


class TestExpandMetadataOperate:
    """Test the split operation."""

    def test_basic_split(self):
        """Split a metadata column into multiple new columns."""
        em = ExpandMetadata(column="ImageName", labels=["Strain", "Cond", "Time"], delimiter="_")
        df = pd.DataFrame({
            "MetadataImage_ImageName": ["WT_30C_24h", "mut_37C_48h"],
            "Object_Label": [1, 2],
            "Shape_Area": [100, 200],
        })
        result = em.apply(df)

        assert "MetadataGenetic_Strain" in result.columns
        assert "Metadata_Cond" in result.columns
        assert "MetadataCulture_Time" in result.columns
        assert list(result["MetadataGenetic_Strain"]) == ["WT", "mut"]
        assert list(result["Metadata_Cond"]) == ["30C", "37C"]
        assert list(result["MetadataCulture_Time"]) == ["24h", "48h"]

    def test_original_column_kept(self):
        """The original column is always preserved."""
        em = ExpandMetadata(column="ImageName", labels=["A", "B"], delimiter="_")
        df = pd.DataFrame({"MetadataImage_ImageName": ["x_y"], "Object_Label": [1]})
        result = em.apply(df)
        assert "MetadataImage_ImageName" in result.columns

    def test_mismatched_split_count_raises(self):
        """Raises ValueError when split produces wrong number of parts."""
        em = ExpandMetadata(column="ImageName", labels=["A", "B", "C"], delimiter="_")
        df = pd.DataFrame({
            "MetadataImage_ImageName": ["WT_30C_24h", "mut_37C"],  # second row has 2 parts, not 3
            "Object_Label": [1, 2],
        })
        with pytest.raises(ValueError, match="split"):
            em.apply(df)

    def test_missing_column_raises(self):
        """Raises KeyError when the source column doesn't exist."""
        em = ExpandMetadata(column="NonExistent", labels=["A", "B"], delimiter="_")
        df = pd.DataFrame({"Object_Label": [1], "Shape_Area": [100]})
        with pytest.raises(KeyError):
            em.apply(df)

    def test_regex_delimiter(self):
        """Regex delimiter splits on multiple characters."""
        em = ExpandMetadata(
            column="ImageName",
            labels=["Strain", "Cond", "Time"],
            delimiter=r"[_\-]",
            regex=True,
        )
        df = pd.DataFrame({
            "MetadataImage_ImageName": ["WT_30C-24h", "mut-37C_48h"],
            "Object_Label": [1, 2],
        })
        result = em.apply(df)
        assert list(result["MetadataGenetic_Strain"]) == ["WT", "mut"]
        assert list(result["Metadata_Cond"]) == ["30C", "37C"]
        assert list(result["MetadataCulture_Time"]) == ["24h", "48h"]

    def test_non_regex_delimiter(self):
        """Default string delimiter with special regex chars is treated literally."""
        em = ExpandMetadata(column="ImageName", labels=["A", "B"], delimiter=".")
        df = pd.DataFrame({
            "MetadataImage_ImageName": ["hello.world", "foo.bar"],
            "Object_Label": [1, 2],
        })
        result = em.apply(df)
        assert list(result["Metadata_A"]) == ["hello", "foo"]
        assert list(result["Metadata_B"]) == ["world", "bar"]

    def test_new_columns_inserted_after_source(self):
        """New columns appear adjacent to the source column."""
        em = ExpandMetadata(column="ImageName", labels=["A", "B"], delimiter="_")
        df = pd.DataFrame({
            "MetadataImage_ImageName": ["x_y"],
            "Object_Label": [1],
            "Shape_Area": [100],
        })
        result = em.apply(df)
        cols = list(result.columns)
        src_idx = cols.index("MetadataImage_ImageName")
        assert cols[src_idx + 1] == "Metadata_A"
        assert cols[src_idx + 2] == "Metadata_B"
