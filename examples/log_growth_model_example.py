#!/usr/bin/env python3
"""
Example demonstrating the use of LogGrowthModel with phenotypic data.

This example shows how to:
1. Load measurement data
2. Fit a logistic growth model
3. Visualize the results
4. Access fitted parameters
"""

import pandas as pd
from phenotypic.analysis._log_growth_model import LogGrowthModel
import matplotlib.pyplot as plt


def main():
    """Main example function."""
    print("LogGrowthModel Example")
    print("="*50)

    # Load a subset of the measurement data
    print("Loading measurement data...")
    data_path = "src/phenotypic/data/meas/meas.csv"
    # Load only first 50k nrows for faster processing in this example
    measurements = pd.read_csv(data_path, nrows=50000)

    print(f"Loaded {len(measurements)} measurements")
    print(f"Columns: {list(measurements.columns[:10])}...")

    # Filter for a specific condition and strain
    subset = measurements[
        (measurements['Metadata_Condition'] == '30C') &
        (measurements['Metadata_Strain'] == 'CBS11445')
        ].copy()

    print(f"Filtered to {len(subset)} measurements for 30C, CBS11445")

    if subset.empty:
        print("No data found for the specified condition/strain. Using synthetic data instead.")
        # Create synthetic data for demonstration
        import numpy as np

        np.random.seed(42)

        time_points = np.arange(0, 10, 1)
        r_true, K_true, N0_true = 0.3, 2000, 100

        t_data = []
        size_data = []

        for t in time_points:
            size = K_true/(1 + (K_true - N0_true)/N0_true*np.exp(-r_true*t))
            size_noisy = size + np.random.normal(0, size*0.1)
            t_data.extend([t]*5)  # 5 replicates per time point
            size_data.extend([size_noisy]*5)

        subset = pd.DataFrame({
            'Metadata_Time'     : t_data,
            'Shape_Area'        : size_data,
            'Metadata_Condition': ['30C']*len(t_data),
            'Metadata_Strain'   : ['CBS11445']*len(t_data),
            'Metadata_Replicate': list(range(len(t_data)))
        })

    # Create and fit the model
    print("\nFitting logistic growth model...")
    model = LogGrowthModel(
            on='Shape_Area',  # Column containing the size measurements
            groupby=['Metadata_Condition', 'Metadata_Strain'],  # Group by these columns
            time_label='Metadata_Time',  # Time column
            verbose=False
    )

    # Analyze the data
    results = model.analyze(subset)
    print(f"Model fitting completed. Results shape: {results.shape}")

    # Display results
    print("\nFitted Parameters:")
    print(results[['Metadata_Condition', 'Metadata_Strain',
                   'LogGrowthModel_r', 'LogGrowthModel_K', 'LogGrowthModel_N0',
                   'LogGrowthModel_d(N)/dt', 'LogGrowthModel_MAE']].head())

    # Create visualization
    print("\nCreating visualization...")
    fig, ax = plt.subplots(figsize=(10, 6))
    model.show(criteria={'Metadata_Condition': '30C'}, ax=ax)
    ax.set_title('Logistic Growth Model Fit')
    ax.set_xlabel('Time')
    ax.set_ylabel('Colony Area')
    plt.tight_layout()

    # Show some statistics
    print("\nModel Statistics:")
    print(f"Mean R-squared: {results['LogGrowthModel_r'].mean():.4f}")
    print(f"Mean MAE: {results['LogGrowthModel_MAE'].mean():.2f}")
    print(f"Mean RMSE: {results['LogGrowthModel_RMSE'].mean():.2f}")

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
