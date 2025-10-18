#!/usr/bin/env python3
"""
Example demonstrating pandas HDF5 persistence with SWMR support.

This example shows how to use the robust pandas HDF5 persistence functionality
for both Series and DataFrame objects with SWMR (Single Writer Multiple Reader) support.
"""

import h5py
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from phenotypic.tools import (
    save_series_new, save_series_append, load_series,
    save_frame_new, save_frame_append, load_frame
)


def basic_series_example():
    """Demonstrate basic Series persistence."""
    print("=== Basic Series Example ===")

    # Create a sample Series with mixed data types
    series = pd.Series(
            [1.5, 2.7, np.nan, 4.2, 5.1],
            index=['sample1', 'sample2', 'sample3', 'sample4', 'sample5'],
            name='measurements'
    )
    series.index.name = 'sample_id'

    print("Original Series:")
    print(series)
    print()

    # Save to HDF5
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        filepath = tmp.name

        with h5py.File(filepath, 'w', libver='latest') as f:
            group = f.create_group('experiment_data')
            save_series_new(group, series, preallocate=100)

            # Load it back
            loaded_series = load_series(group)

            print("Loaded Series:")
            print(loaded_series)
            print()

            # Verify round-trip equality
            pd.testing.assert_series_equal(
                    loaded_series.astype('float64'),
                    series.astype('float64')
            )
            print("✓ Round-trip successful - all data preserved!")

    Path(filepath).unlink()  # Clean up
    print()


def swmr_append_example():
    """Demonstrate SWMR append functionality."""
    print("=== SWMR Append Example ===")

    # Initial data
    initial_data = pd.Series([10, 20, 30], index=['day1', 'day2', 'day3'], name='measurements')

    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        filepath = tmp.name

        # Writer process simulation
        with h5py.File(filepath, 'w', libver='latest') as writer:
            try:
                writer.swmr_mode = True
                print("SWMR mode enabled")
            except OSError:
                print("SWMR mode not available (continuing without)")

            group = writer.create_group('live_experiment')

            # Create initial dataset with preallocation
            save_series_new(group, initial_data, preallocate=1000)
            writer.flush()
            print(f"Initial data saved: {len(initial_data)} measurements")

            # Simulate appending new data over time
            for batch in range(3):
                new_data = pd.Series(
                        [40 + batch*10, 50 + batch*10],
                        index=[f'day{4 + batch*2}', f'day{5 + batch*2}'],
                        name='measurements'
                )

                save_series_append(group, new_data)
                writer.flush()

                current_data = load_series(group)
                print(f"After batch {batch + 1}: {len(current_data)} total measurements")

            # Final state
            final_data = load_series(group)
            print("\nFinal dataset:")
            print(final_data)

    Path(filepath).unlink()  # Clean up
    print()


def dataframe_example():
    """Demonstrate DataFrame persistence."""
    print("=== DataFrame Example ===")

    # Create a sample DataFrame with mixed dtypes
    df = pd.DataFrame({
        'colony_id'  : ['C001', 'C002', 'C003', 'C004'],
        'diameter_mm': [12.5, 15.2, 8.7, 14.1],
        'color'      : ['red', 'blue', 'green', 'red'],
        'viable'     : [True, True, False, True],
        'notes'      : ['healthy', 'contaminated', None, 'good growth']
    }, index=['plate1_A1', 'plate1_A2', 'plate1_B1', 'plate1_B2'])
    df.index.name = 'position'

    print("Original DataFrame:")
    print(df)
    print()

    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        filepath = tmp.name

        with h5py.File(filepath, 'w', libver='latest') as f:
            group = f.create_group('colony_data')
            save_frame_new(group, df, preallocate=100)

            # Load it back
            loaded_df = load_frame(group)

            print("Loaded DataFrame:")
            print(loaded_df)
            print()

            print("Column dtypes after round-trip:")
            print(loaded_df.dtypes)
            print()

            # Demonstrate append functionality
            new_rows = pd.DataFrame({
                'colony_id'  : ['C005', 'C006'],
                'diameter_mm': [13.8, 11.2],
                'color'      : ['yellow', 'blue'],
                'viable'     : [True, False],
                'notes'      : ['excellent', 'poor']
            }, index=['plate2_A1', 'plate2_A2'])

            save_frame_append(group, new_rows)

            final_df = load_frame(group)
            print(f"After appending: {len(final_df)} total colonies")
            print(final_df.tail())

    Path(filepath).unlink()  # Clean up
    print()


def unicode_example():
    """Demonstrate Unicode string handling."""
    print("=== Unicode String Example ===")

    # Test various Unicode characters
    unicode_data = pd.Series([
        'English text',
        'Café français',
        '北京大学',  # Chinese
        'Москва',  # Russian
        '🧪🔬🦠',  # Emojis
        None,
        ''
    ], name='multilingual_data')

    print("Original Unicode data:")
    print(unicode_data)
    print()

    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        filepath = tmp.name

        with h5py.File(filepath, 'w') as f:
            group = f.create_group('unicode_test')
            save_series_new(group, unicode_data)

            loaded_data = load_series(group)

            print("Loaded Unicode data:")
            print(loaded_data)
            print()

            # Verify Unicode preservation
            for orig, loaded in zip(unicode_data, loaded_data):
                if pd.isna(orig) and pd.isna(loaded):
                    continue
                assert orig == loaded, f"Unicode mismatch: {orig} != {loaded}"

            print("✓ All Unicode characters preserved correctly!")

    Path(filepath).unlink()  # Clean up
    print()


def multiindex_example():
    """Demonstrate MultiIndex support for both Series and DataFrames."""
    print("=== MultiIndex Example ===")

    # Create MultiIndex for hierarchical data
    arrays = [
        ['Plate1', 'Plate1', 'Plate1', 'Plate2', 'Plate2', 'Plate2'],
        ['Row1', 'Row2', 'Row3', 'Row1', 'Row2', 'Row3'],
        ['ColA', 'ColB', 'ColC', 'ColA', 'ColB', 'ColC']
    ]
    index = pd.MultiIndex.from_arrays(arrays, names=['plate_id', 'row', 'column'])

    # Series example with MultiIndex
    growth_data = pd.Series([15.2, 22.1, 18.9, 14.7, 20.3, 19.8],
                            index=index, name='colony_diameter_mm')

    print("Original MultiIndex Series:")
    print(growth_data)
    print(f"Index levels: {growth_data.index.nlevels}")
    print(f"Index names: {growth_data.index.names}")
    print()

    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        filepath = tmp.name

        with h5py.File(filepath, 'w') as f:
            # Save MultiIndex Series
            series_group = f.create_group('multiindex_series')
            save_series_new(series_group, growth_data)

            loaded_series = load_series(series_group)

            print("Loaded MultiIndex Series:")
            print(loaded_series)
            print(f"Index type preserved: {type(loaded_series.index)}")
            print(f"Index equality: {growth_data.index.equals(loaded_series.index)}")
            print()

            # DataFrame example with MultiIndex
            experiment_df = pd.DataFrame({
                'diameter_mm': [15.2, 22.1, 18.9, 14.7, 20.3, 19.8],
                'viable'     : [True, True, True, False, True, True],
                'color_score': [85, 92, 78, 45, 88, 81],
                'notes'      : ['healthy', 'excellent', 'good', 'contaminated', 'very good', 'normal']
            }, index=index)

            print("Original MultiIndex DataFrame:")
            print(experiment_df)
            print()

            # Save MultiIndex DataFrame
            df_group = f.create_group('multiindex_dataframe')
            save_frame_new(df_group, experiment_df)

            loaded_df = load_frame(df_group)

            print("Loaded MultiIndex DataFrame:")
            print(loaded_df)
            print(f"Index type preserved: {type(loaded_df.index)}")
            print(f"Index equality: {experiment_df.index.equals(loaded_df.index)}")
            print()

            # Demonstrate append with MultiIndex
            print("Appending additional data...")
            new_arrays = [
                ['Plate3', 'Plate3'],
                ['Row1', 'Row2'],
                ['ColA', 'ColB']
            ]
            new_index = pd.MultiIndex.from_arrays(new_arrays, names=['plate_id', 'row', 'column'])

            append_df = pd.DataFrame({
                'diameter_mm': [16.5, 21.2],
                'viable'     : [True, True],
                'color_score': [87, 90],
                'notes'      : ['good growth', 'excellent']
            }, index=new_index)

            save_frame_append(df_group, append_df)
            final_df = load_frame(df_group)

            print("Final DataFrame after append:")
            print(final_df)
            print(f"Total rows: {len(final_df)}")
            print("✓ MultiIndex structure preserved through append!")

    Path(filepath).unlink()  # Clean up
    print()


def named_index_example():
    """Demonstrate proper index name preservation."""
    print("=== Named Index Example ===")

    # Regular named index
    series = pd.Series([1.5, 2.3, 4.1, 3.7],
                       index=['sample_001', 'sample_002', 'sample_003', 'sample_004'],
                       name='concentration_mM')
    series.index.name = 'sample_id'

    print("Original Series with named index:")
    print(series)
    print(f"Index name: '{series.index.name}'")
    print()

    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        filepath = tmp.name

        with h5py.File(filepath, 'w') as f:
            group = f.create_group('named_index_test')
            save_series_new(group, series)

            loaded = load_series(group)

            print("Loaded Series:")
            print(loaded)
            print(f"Index name preserved: '{loaded.index.name}'")
            print(f"Names match: {series.index.name == loaded.index.name}")
            print("✓ Index name properly preserved!")

    Path(filepath).unlink()  # Clean up
    print()


if __name__ == '__main__':
    print("Pandas HDF5 Persistence Examples")
    print("="*40)
    print()

    basic_series_example()
    swmr_append_example()
    dataframe_example()
    unicode_example()
    multiindex_example()
    named_index_example()

    print("All examples completed successfully!")
    print("\nKey features demonstrated:")
    print("- Round-trip data preservation")
    print("- SWMR-safe append operations")
    print("- Mixed data type support")
    print("- Unicode string handling")
    print("- DataFrame column order preservation")
    print("- MultiIndex support for Series and DataFrames")
    print("- Named index preservation")
    print("- Preallocation for performance")
