# Measurement Metrics and Their Biological Meaning

PhenoTypic's measurement operations extract quantitative features from
detected colonies. This page explains what each metric means biologically
and when to use it.

## Size Metrics (MeasureSize)

| Metric | Unit | Biological meaning |
|--------|------|-------------------|
| Area | pixels | Colony size; proxy for biomass or growth |
| IntegratedIntensity | sum of pixel values | Total pigmentation; proxy for metabolic output |

Area is the most commonly used growth metric. For calibrated images,
convert pixels to physical units (mm²) using the known pixel pitch.

## Shape Metrics (MeasureShape)

| Metric | Range | Biological meaning |
|--------|-------|-------------------|
| Circularity | 0–1 | How round the colony is (1 = perfect circle). Irregular shapes suggest stress, mutation, or sectoring. |
| Solidity | 0–1 | Ratio of area to convex hull area. Low solidity indicates lobed or branching morphology. |
| Eccentricity | 0–1 | Elongation (0 = circular, ~1 = very elongated). Elevated in filamentous or swarming colonies. |
| Perimeter | pixels | Colony boundary length. Increases faster than area for irregular shapes. |
| MajorAxisLength / MinorAxisLength | pixels | Fitted ellipse dimensions. The ratio indicates elongation. |
| Compactness | ≥1 | Perimeter² / (4π × Area). Equals 1 for a perfect circle; increases with irregularity. |

Shape metrics are particularly useful for distinguishing wild-type from
mutant morphology, or for detecting contamination.

## Intensity Metrics (MeasureIntensity)

| Metric | Biological meaning |
|--------|-------------------|
| MeanIntensity | Average colony brightness; correlates with pigment concentration |
| MedianIntensity | Robust center of the intensity distribution; less affected by bright/dark spots |
| StandardDeviationIntensity | Internal color variation; high values suggest heterogeneous pigmentation or sectoring |
| MinimumIntensity / MaximumIntensity | Intensity extremes within the colony |
| CoefficientVarianceIntensity | Normalized variation (StdDev / Mean); comparable across colonies of different brightness |

## Color Metrics (MeasureColor)

Color measurements use the CIELAB, HSV, and CIE XYZ color spaces.
Key columns include mean, median, and standard deviation for each
channel. These are essential for:

- Quantifying pigmentation differences between strains
- Detecting sectoring (regions of different color within a colony)
- Normalizing color across imaging sessions (after `ColorCorrector`)

## Texture Metrics (MeasureTexture)

Haralick texture features computed from gray-level co-occurrence
matrices at specified scales. These capture spatial patterns within
colonies that size and shape metrics miss:

- Smooth vs. rough colony surfaces
- Concentric ring patterns
- Internal structure differences between strains

## Choosing Metrics

| Biological question | Metrics |
|--------------------|---------|
| Growth rate comparison | Area (time series) |
| Strain fitness ranking | Area + MeanIntensity |
| Morphology screening | Circularity + Solidity + Eccentricity |
| Pigmentation assay | MeanIntensity + CIELAB color channels |
| Colony heterogeneity | StandardDeviationIntensity + texture features |
