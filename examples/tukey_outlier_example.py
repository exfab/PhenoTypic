"""
Example: Using TukeyOutlierDetector for Quality Control in Microbial Growth Assays

This example demonstrates how to use TukeyOutlierDetector to identify anomalous
colony measurements in arrayed microbial growth experiments on agar plates.
"""

import numpy as np
import pandas as pd
from phenotypic.analysis import TukeyOutlierDetector

# Simulate colony area measurements from multiple 96-well agar plates
# Real data would come from image analysis of bacterial/fungal growth
np.random.seed(42)

plates = []
for plate_id in ['Plate_A', 'Plate_B', 'Plate_C']:
    # Simulate typical colony growth with some measurement noise
    normal_colonies = np.random.normal(loc=250, scale=40, size=90)
    
    # Add some realistic outliers:
    # - Contamination (abnormally large colonies)
    # - Poor inoculation (very small or missing colonies)
    # - Edge effects (not corrected yet)
    outliers = np.array([50, 80, 450, 480, 500, 520])  # 6 problematic colonies
    
    areas = np.concatenate([normal_colonies, outliers])
    np.random.shuffle(areas)
    
    plate_data = pd.DataFrame({
        'Plate': plate_id,
        'Well': [f"{chr(65+r)}{c+1:02d}" for r in range(8) for c in range(12)],
        'Colony_Area': areas
    })
    plates.append(plate_data)

# Combine all plate data
data = pd.concat(plates, ignore_index=True)

print("="*70)
print("Tukey Outlier Detection for Microbial Colony Quality Control")
print("="*70)
print(f"\nDataset: {len(data)} colonies across {data['Plate'].nunique()} plates")
print(f"Mean colony area: {data['Colony_Area'].mean():.1f} pixels")
print(f"Std deviation: {data['Colony_Area'].std():.1f} pixels")

# Initialize the detector with standard parameters
# k=1.5 is standard for identifying outliers
# k=3.0 would identify only extreme outliers
detector = TukeyOutlierDetector(
    on='Colony_Area',
    groupby=['Plate'],  # Detect outliers within each plate independently
    measurement_col='Colony_Area',
    k=1.5  # Standard Tukey fence multiplier
)

# Detect outliers
results = detector.analyze(data)

print("\n" + "="*70)
print("Detection Results")
print("="*70)

# Summary by plate
for plate in results['Plate'].unique():
    plate_data = results[results['Plate'] == plate]
    outliers = plate_data[plate_data['is_outlier']]
    
    print(f"\n{plate}:")
    print(f"  Total colonies: {len(plate_data)}")
    print(f"  Outliers detected: {len(outliers)} ({100*len(outliers)/len(plate_data):.1f}%)")
    print(f"  Fence range: [{plate_data['lower_fence'].iloc[0]:.1f}, "
          f"{plate_data['upper_fence'].iloc[0]:.1f}]")
    
    if len(outliers) > 0:
        low_outliers = outliers[outliers['outlier_type'] == 'low']
        high_outliers = outliers[outliers['outlier_type'] == 'high']
        
        if len(low_outliers) > 0:
            print(f"  Low outliers: {len(low_outliers)} "
                  f"(values: {low_outliers['Colony_Area'].min():.1f} - "
                  f"{low_outliers['Colony_Area'].max():.1f})")
        if len(high_outliers) > 0:
            print(f"  High outliers: {len(high_outliers)} "
                  f"(values: {high_outliers['Colony_Area'].min():.1f} - "
                  f"{high_outliers['Colony_Area'].max():.1f})")

# Demonstrate filtering strategies
print("\n" + "="*70)
print("Downstream Analysis Options")
print("="*70)

# Option 1: Remove outliers
clean_data = results[~results['is_outlier']]
print(f"\nOption 1 - Remove outliers:")
print(f"  Original: {len(results)} colonies, mean = {results['Colony_Area'].mean():.1f}")
print(f"  Cleaned:  {len(clean_data)} colonies, mean = {clean_data['Colony_Area'].mean():.1f}")

# Option 2: Flag for review
flagged_data = results.copy()
flagged_data['QC_Status'] = flagged_data['is_outlier'].map({
    True: 'Review Required',
    False: 'Pass'
})
needs_review = flagged_data[flagged_data['QC_Status'] == 'Review Required']
print(f"\nOption 2 - Flag for manual review:")
print(f"  Colonies flagged: {len(needs_review)}")
print(f"  Ready for analysis: {len(flagged_data) - len(needs_review)}")

# Option 3: Compare different stringencies
print(f"\nOption 3 - Compare detection stringencies:")
for k_val in [1.5, 2.0, 3.0]:
    detector_k = TukeyOutlierDetector(
        on='Colony_Area',
        groupby=['Plate'],
        measurement_col='Colony_Area',
        k=k_val
    )
    results_k = detector_k.analyze(data)
    n_out = results_k['is_outlier'].sum()
    print(f"  k={k_val}: {n_out} outliers ({100*n_out/len(data):.1f}% of data)")

print("\n" + "="*70)
print("✓ Example complete!")
print("="*70)

