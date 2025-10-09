"""Tests for pandas Series and DataFrame persistence functionality in HDF class."""

import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from phenotypic.util.hdf_ import HDF


@pytest.fixture
def temp_hdf5_file():
    """Create a temporary HDF5 file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def hdf5_file_swmr(temp_hdf5_file):
    """Create an HDF5 file with SWMR mode enabled."""
    with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
        f.swmr_mode = True
        yield f


@pytest.fixture
def hdf5_file_no_swmr(temp_hdf5_file):
    """Create an HDF5 file without SWMR mode."""
    with h5py.File(temp_hdf5_file, "w") as f:
        yield f


class TestSWMRAssertions:
    """Test SWMR mode assertions."""

    def test_assert_swmr_on_success(self, hdf5_file_swmr):
        """Test that assert_swmr_on passes when SWMR is enabled."""
        group = hdf5_file_swmr.create_group("test")
        HDF.assert_swmr_on(group)  # Should not raise

    def test_assert_swmr_on_failure(self, hdf5_file_no_swmr):
        """Test that assert_swmr_on raises when SWMR is disabled."""
        group = hdf5_file_no_swmr.create_group("test")
        with pytest.raises(RuntimeError, match="SWMR mode is required"):
            HDF.assert_swmr_on(group)


class TestSeriesIO:
    """Test Series persistence functionality."""

    @pytest.mark.parametrize(
            "dtype,expected_kind",
            [
                (np.float32, "numeric_float64"),
                (np.float64, "numeric_float64"),
                (bool, "numeric_float64"),
                (str, "string_utf8_fixed"),
                (object, "string_utf8_fixed"),
            ],
    )
    def test_series_round_trip_dtypes(self, temp_hdf5_file, dtype, expected_kind):
        """Test round-trip for different data types."""
        # Create test data
        if dtype in [str, object]:
            data = ["apple", "banana", "cherry", None, "date"]
        elif dtype in [bool]:
            data = [True, False, True, False, False]
        else:
            data = [1.0, 2.0, 3.0, None, 5.0]

        series = pd.Series(data, dtype=dtype, name="test_series")

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            group = f.create_group("series")

            # Step 1: Create all objects BEFORE enabling SWMR
            if dtype in [str, object]:
                HDF.save_series_new(
                        group, series, string_fixed_length=10, require_swmr=False
                )
            else:
                HDF.save_series_new(group, series, require_swmr=False)

            # Step 2: Start SWMR mode
            f.swmr_mode = True

            # Step 3: Load under SWMR
            loaded = HDF.load_series(group, require_swmr=True)

            # Check values kind attribute
            values_kind_attr = group.attrs["values_kind"]
            if isinstance(values_kind_attr, bytes):
                values_kind_attr = values_kind_attr.decode("utf-8")
            assert values_kind_attr == expected_kind

            # For numeric/boolean data, check round-trip
            if expected_kind == "numeric_float64":
                expected_values = pd.Series(data, dtype=np.float64, name="test_series")
                np.testing.assert_array_equal(loaded.values, expected_values.values)
                assert loaded.name == expected_values.name
            else:
                # String data preserves None values
                np.testing.assert_array_equal(loaded.values, series.values)
                assert loaded.name == series.name

    def test_series_with_index(self, temp_hdf5_file):
        """Test Series with custom Index."""
        series = pd.Series([1, 2, 3, 4], index=["a", "b", "c", "d"], name="indexed")

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            group = f.create_group("series")

            # Step 1: Create all objects BEFORE enabling SWMR
            HDF.save_series_new(group, series, require_swmr=False)

            # Step 2: Start SWMR mode
            f.swmr_mode = True

            # Step 3: Load under SWMR
            loaded = HDF.load_series(group, require_swmr=True)

            pd.testing.assert_series_equal(loaded, series.astype(np.float64))
            assert loaded.name == "indexed"

    def test_series_with_multiindex(self, temp_hdf5_file):
        """Test Series with MultiIndex."""
        index = pd.MultiIndex.from_tuples(
                [("A", 1), ("A", 2), ("B", 1), ("B", 2), ("C", None)],
                names=["level1", "level2"],
        )
        series = pd.Series([10, 20, 30, 40, 50], index=index, name="multi_series")

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            group = f.create_group("series")

            # Step 1: Create all objects BEFORE enabling SWMR
            HDF.save_series_new(group, series, require_swmr=False)

            # Step 2: Start SWMR mode
            f.swmr_mode = True

            # Step 3: Load under SWMR
            loaded = HDF.load_series(group, require_swmr=True)

            # Check attributes
            assert group.attrs["index_is_multiindex"] == 1
            assert group.attrs["index_levels"] == 2
            index_names_attr = group.attrs["index_names"]
            if isinstance(index_names_attr, bytes):
                index_names_attr = index_names_attr.decode("utf-8")
            assert json.loads(index_names_attr) == ["level1", "level2"]

            # Check that values are correct
            np.testing.assert_array_equal(
                    loaded.values, series.astype(np.float64).values
            )
            assert loaded.name == series.name
            assert isinstance(loaded.index, pd.MultiIndex)
            assert loaded.index.names == series.index.names

    def test_series_append_functionality(self, temp_hdf5_file):
        """Test Series append functionality."""
        series1 = pd.Series([1, 2, 3], name="test_append")
        series2 = pd.Series([4, 5, 6], name="test_append")

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            group = f.create_group("series")

            # Create initial series
            HDF.save_series_new(group, series1, require_swmr=False)

            # Enable SWMR mode
            f.swmr_mode = True

            # Append more data
            HDF.save_series_append(group, series2, require_swmr=True)

            # Load and verify
            loaded = HDF.load_series(group, require_swmr=True)

            # Check values and length - index will be string type due to HDF5 storage
            assert len(loaded) == 6
            np.testing.assert_array_equal(loaded.values, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            assert loaded.name == "test_append"

    def test_empty_series_validation(self, temp_hdf5_file):
        """Test validation for empty series."""
        empty_series = pd.Series([], name="empty")

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            group = f.create_group("series")

            with pytest.raises(ValueError, match="Cannot save empty series"):
                HDF.save_series_new(group, empty_series, require_swmr=False)


class TestDataFrameIO:
    """Test DataFrame persistence functionality."""

    def test_frame_round_trip_basic(self, temp_hdf5_file):
        """Test basic DataFrame round-trip."""
        df = pd.DataFrame(
                {
                    "int_col"  : [1, 2, 3, None],
                    "float_col": [1.1, 2.2, 3.3, 4.4],
                    "str_col"  : ["a", "b", None, "d"],
                    "bool_col" : [True, False, True, None],
                }
        )

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            group = f.create_group("frame")

            # Save DataFrame
            HDF.save_frame_new(group, df, require_swmr=False)

            # Enable SWMR mode
            f.swmr_mode = True

            # Load under SWMR
            loaded = HDF.load_frame(group, require_swmr=True)

            # Check structure
            assert list(loaded.columns) == list(df.columns)
            assert len(loaded) == len(df)

    def test_frame_with_custom_index(self, temp_hdf5_file):
        """Test DataFrame with custom index."""
        df = pd.DataFrame(
                {"A": [1, 2, 3], "B": [4.0, 5.0, 6.0]}, index=["x", "y", "z"]
        )

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            group = f.create_group("frame")

            # Save DataFrame
            HDF.save_frame_new(group, df, require_swmr=False)

            # Enable SWMR mode
            f.swmr_mode = True

            # Load under SWMR
            loaded = HDF.load_frame(group, require_swmr=True)

            # Check that data is preserved
            assert list(loaded.columns) == list(df.columns)
            assert len(loaded) == len(df)
            np.testing.assert_array_equal(loaded["A"].values, [1.0, 2.0, 3.0])
            np.testing.assert_array_equal(loaded["B"].values, [4.0, 5.0, 6.0])

    def test_frame_append_functionality(self, temp_hdf5_file):
        """Test DataFrame append functionality."""
        df1 = pd.DataFrame({"A": [1, 2], "B": [3.0, 4.0]})
        df2 = pd.DataFrame({"A": [5, 6], "B": [7.0, 8.0]})

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            group = f.create_group("frame")

            # Create initial DataFrame
            HDF.save_frame_new(group, df1, require_swmr=False)

            # Enable SWMR mode
            f.swmr_mode = True

            # Append more data
            HDF.save_frame_append(group, df2, require_swmr=True)

            # Load and verify
            loaded = HDF.load_frame(group, require_swmr=True)

            assert len(loaded) == 4
            assert list(loaded.columns) == ["A", "B"]
            np.testing.assert_array_equal(loaded["A"].values, [1.0, 2.0, 5.0, 6.0])
            np.testing.assert_array_equal(loaded["B"].values, [3.0, 4.0, 7.0, 8.0])

    def test_empty_dataframe_validation(self, temp_hdf5_file):
        """Test validation for empty DataFrame."""
        empty_df = pd.DataFrame()

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            group = f.create_group("frame")

            with pytest.raises(ValueError, match="Cannot save empty DataFrame"):
                HDF.save_frame_new(group, empty_df, require_swmr=False)

    def test_column_order_mismatch(self, temp_hdf5_file):
        """Test validation for column order mismatch."""
        df1 = pd.DataFrame({"A": [1, 2], "B": [3.0, 4.0]})
        df2 = pd.DataFrame({"B": [7.0, 8.0], "A": [5, 6]})  # Different column order

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            group = f.create_group("frame")

            # Create initial DataFrame
            HDF.save_frame_new(group, df1, require_swmr=False)

            # Enable SWMR mode
            f.swmr_mode = True

            # Try to append with different column order
            with pytest.raises(ValueError, match="Column order mismatch"):
                HDF.save_frame_append(group, df2, require_swmr=True)


class TestStringFixedLength:
    """Test fixed-length string functionality."""

    def test_fixed_length_strings_series(self, temp_hdf5_file):
        """Test Series with fixed-length strings."""
        series = pd.Series(["short", "a very long string", "medium"], name="strings")

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            group = f.create_group("series")

            # Save with fixed length of 10 characters
            HDF.save_series_new(group, series, string_fixed_length=10, require_swmr=False)

            # Enable SWMR mode
            f.swmr_mode = True

            # Load and verify
            loaded = HDF.load_series(group, require_swmr=True)

            # Check that values are truncated/padded appropriately
            assert loaded.iloc[0] == "short"  # No padding shown after trimming
            assert loaded.iloc[1] == "a very lon"  # Truncated to 10 chars  
            assert loaded.iloc[2] == "medium"  # No padding shown after trimming

    def test_fixed_length_strings_dataframe(self, temp_hdf5_file):
        """Test DataFrame with fixed-length strings."""
        df = pd.DataFrame(
                {
                    "str_col": ["short", "very long string here", "medium"],
                    "num_col": [1, 2, 3],
                }
        )

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            group = f.create_group("frame")

            # Save with fixed length of 8 characters for strings
            HDF.save_frame_new(group, df, string_fixed_length=8, require_swmr=False)

            # Enable SWMR mode
            f.swmr_mode = True

            # Load and verify
            loaded = HDF.load_frame(group, require_swmr=True)

            # Check string column is truncated appropriately
            assert loaded["str_col"].iloc[0] == "short"
            assert loaded["str_col"].iloc[1] == "very lon"  # Truncated to 8 chars
            assert loaded["str_col"].iloc[2] == "medium"

            # Check numeric column is preserved
            np.testing.assert_array_equal(loaded["num_col"].values, [1.0, 2.0, 3.0])


class TestPreallocation:
    """Test preallocation functionality."""

    def test_series_preallocation(self, temp_hdf5_file):
        """Test Series preallocation."""
        series = pd.Series([1, 2, 3], name="test")

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            group = f.create_group("series")

            # Preallocate layout
            HDF.preallocate_series_layout(group, series, preallocate=100)

            # Check that length is 0 initially
            assert group.attrs["len"] == 0

            # Enable SWMR mode
            f.swmr_mode = True

            # Now save data using existing layout
            HDF.save_series_new(group, series, require_swmr=True)

            # Verify data - index will be string type due to HDF5 storage
            loaded = HDF.load_series(group, require_swmr=True)
            np.testing.assert_array_equal(loaded.values, series.astype(np.float64).values)
            assert loaded.name == series.name
            assert len(loaded) == len(series)

    def test_frame_preallocation(self, temp_hdf5_file):
        """Test DataFrame preallocation."""
        df = pd.DataFrame({"A": [1, 2], "B": [3.0, 4.0]})

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            group = f.create_group("frame")

            # Preallocate layout
            HDF.preallocate_frame_layout(group, df, preallocate=100, require_swmr=False)

            # Check that length is 0 initially
            assert group.attrs["len"] == 0

            # Enable SWMR mode
            f.swmr_mode = True

            # Now save data using existing layout
            HDF.save_frame_new(group, df, require_swmr=True)

            # Verify data
            loaded = HDF.load_frame(group, require_swmr=True)
            assert len(loaded) == 2
            assert list(loaded.columns) == ["A", "B"]


if __name__ == "__main__":
    pytest.main([__file__])
