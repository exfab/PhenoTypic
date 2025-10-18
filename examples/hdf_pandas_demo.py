"""
Demonstration of integrated pandas2hdf functionality in PhenoTypic HDF class.

This example shows how to use the newly integrated pandas Series and DataFrame
persistence capabilities with SWMR support for microbiology data workflows.
"""

import tempfile
import h5py
import pandas as pd
import numpy as np
from pathlib import Path

from phenotypic.tools.hdf_ import HDF


def demo_series_persistence():
    """Demonstrate Series persistence for bacterial colony measurements."""
    print("=== Series Persistence Demo ===")

    # Create sample bacterial colony measurement data
    colony_data = pd.Series(
            [12.5, 15.2, 9.8, 22.1, 18.7, None, 14.3, 11.9],
            index=[f"colony_{i:03d}" for i in range(1, 9)],
            name="diameter_mm"
    )

    print("Original colony diameter data:")
    print(colony_data)
    print()

    # Create temporary HDF5 file
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        temp_file = f.name

    try:
        # Step 1: Create HDF5 file and save data
        with h5py.File(temp_file, "w", libver="latest") as f:
            group = f.create_group("measurements")

            # Save with SWMR-compatible fixed-length strings for indices
            HDF.save_series_new(
                    group,
                    colony_data,
                    string_fixed_length=20,  # Sufficient for colony names
                    require_swmr=False
            )

            print("Data saved to HDF5")

            # Step 2: Enable SWMR mode for real-time access
            f.swmr_mode = True
            print("SWMR mode enabled")

            # Step 3: Load data under SWMR
            loaded_data = HDF.load_series(group, require_swmr=True)
            print("Loaded colony diameter data:")
            print(loaded_data)
            print()

            # Step 4: Append new measurements (simulating real-time data collection)
            new_measurements = pd.Series(
                    [16.4, 13.8, 20.2],
                    index=[f"colony_{i:03d}" for i in range(9, 12)],
                    name="diameter_mm"
            )

            HDF.save_series_append(group, new_measurements, require_swmr=True)
            print("Appended new measurements")

            # Step 5: Load complete dataset
            complete_data = HDF.load_series(group, require_swmr=True)
            print("Complete dataset after append:")
            print(complete_data)
            print()

    finally:
        # Cleanup
        Path(temp_file).unlink()


def demo_dataframe_persistence():
    """Demonstrate DataFrame persistence for bacterial growth experiments."""
    print("=== DataFrame Persistence Demo ===")

    # Create sample bacterial growth experiment data
    growth_data = pd.DataFrame({
        "strain"               : ["E.coli_K12", "E.coli_DH5α", "B.subtilis_168", "S.aureus_MRSA"],
        "initial_od"           : [0.05, 0.03, 0.08, 0.06],
        "final_od"             : [1.85, 1.92, 1.44, 1.67],
        "growth_rate"          : [0.72, 0.81, 0.58, 0.63],
        "antibiotic_resistance": [False, False, True, True]
    }, index=[f"exp_{i:03d}" for i in range(1, 5)])

    print("Original bacterial growth data:")
    print(growth_data)
    print()

    # Create temporary HDF5 file
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        temp_file = f.name

    try:
        # Step 1: Create HDF5 file and save data
        with h5py.File(temp_file, "w", libver="latest") as f:
            group = f.create_group("growth_experiments")

            # Save with SWMR-compatible settings
            HDF.save_frame_new(
                    group,
                    growth_data,
                    string_fixed_length=30,  # Sufficient for strain names
                    require_swmr=False
            )

            print("Growth data saved to HDF5")

            # Step 2: Enable SWMR mode
            f.swmr_mode = True
            print("SWMR mode enabled")

            # Step 3: Load data under SWMR
            loaded_data = HDF.load_frame(group, require_swmr=True)
            print("Loaded growth experiment data:")
            print(loaded_data)
            print()

            # Step 4: Append new experiment results
            new_experiments = pd.DataFrame({
                "strain"               : ["P.aeruginosa_PAO1", "L.monocytogenes"],
                "initial_od"           : [0.04, 0.07],
                "final_od"             : [1.78, 1.55],
                "growth_rate"          : [0.69, 0.51],
                "antibiotic_resistance": [True, False]
            }, index=[f"exp_{i:03d}" for i in range(5, 7)])

            HDF.save_frame_append(group, new_experiments, require_swmr=True)
            print("Appended new experiments")

            # Step 5: Load complete dataset
            complete_data = HDF.load_frame(group, require_swmr=True)
            print("Complete dataset after append:")
            print(complete_data)
            print()
            print(f"Dataset shape: {complete_data.shape}")
            print(f"Column types preserved: {complete_data.dtypes}")

    finally:
        # Cleanup
        Path(temp_file).unlink()


def demo_preallocation_workflow():
    """Demonstrate preallocation for high-throughput screening."""
    print("=== Preallocation Workflow Demo ===")

    # Create sample high-throughput screening data
    screening_data = pd.DataFrame({
        "compound_id"       : [f"CMP_{i:06d}" for i in range(1, 5)],
        "ic50_um"           : [12.5, 8.3, 45.2, 2.1],
        "viability_percent" : [15.2, 8.7, 62.1, 5.4],
        "cytotoxicity_score": [8.5, 9.2, 4.1, 9.8]
    })

    print("Sample screening data:")
    print(screening_data)
    print()

    # Create temporary HDF5 file
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        temp_file = f.name

    try:
        with h5py.File(temp_file, "w", libver="latest") as f:
            group = f.create_group("screening_results")

            # Step 1: Preallocate for 10,000 compounds (typical screening size)
            HDF.preallocate_frame_layout(
                    group,
                    screening_data,
                    preallocate=10000,  # Space for 10K compounds
                    string_fixed_length=15,  # Sufficient for compound IDs
                    require_swmr=False
            )

            print("Preallocated space for 10,000 compounds")
            print(f"Initial logical length: {group.attrs['len']}")

            # Step 2: Enable SWMR for real-time screening
            f.swmr_mode = True
            print("SWMR mode enabled for real-time data collection")

            # Step 3: Write initial batch of screening results
            HDF.save_frame_new(group, screening_data, require_swmr=True)
            print(f"Saved initial batch, logical length: {group.attrs['len']}")

            # Step 4: Simulate continuous screening by appending batches
            for batch in range(2, 4):
                batch_data = pd.DataFrame({
                    "compound_id"       : [f"CMP_{i:06d}" for i in range(batch*4 - 3, batch*4 + 1)],
                    "ic50_um"           : np.random.uniform(1, 50, 4),
                    "viability_percent" : np.random.uniform(5, 80, 4),
                    "cytotoxicity_score": np.random.uniform(1, 10, 4)
                })

                HDF.save_frame_append(group, batch_data, require_swmr=True)
                print(f"Appended batch {batch}, logical length: {group.attrs['len']}")

            # Step 5: Load final results
            final_results = HDF.load_frame(group, require_swmr=True)
            print("\nFinal screening results:")
            print(final_results)
            print(f"\nTotal compounds screened: {len(final_results)}")

    finally:
        # Cleanup
        Path(temp_file).unlink()


if __name__ == "__main__":
    print("PhenoTypic HDF Pandas Integration Demo")
    print("="*50)
    print()

    # Run demonstrations
    demo_series_persistence()
    print("\n" + "="*50 + "\n")

    demo_dataframe_persistence()
    print("\n" + "="*50 + "\n")

    demo_preallocation_workflow()

    print("\n" + "="*50)
    print("Demo completed successfully!")
    print("The pandas2hdf functionality is now integrated into PhenoTypic's HDF class.")
