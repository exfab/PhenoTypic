# Measurement Metrics and Their Biological Meaning

PhenoTypic's measurement operations extract quantitative features from
detected colonies. This page explains what each metric means biologically
and when to use it.

> For *how to apply* each metric — which to report directly vs. use as a
> classifier feature — see
> [Measurement Classification: Phenotypes vs. Features](measurement_classification_system.md).

## Size Metrics (MeasureSize)

```{versionchanged} 0.20.0
Size magnitudes moved from `MeasureShape` to `MeasureSize`, and the radius
columns were rebuilt. `Size_MedianRadius`, `Size_MeanRadius` and
`Size_MaxRadius` are **not** the retired `Shape_*` columns of the same name;
see the rename table on the {ref}`SIZE reference <measurement-info-size>`.
```

| Metric | Unit | Biological meaning |
|--------|------|-------------------|
| Area | pixels | Colony size; proxy for biomass or growth |
| IntegratedIntensity | sum of pixel values | Total pigmentation; proxy for metabolic output |
| Perimeter | pixels | Colony boundary length. Increases faster than area for irregular shapes. |
| ConvexArea | pixels | Area of the colony's convex hull — its "filled-in" footprint if indentations were removed; a low Solidity (Area/ConvexArea) signals invasive or spreading growth. |
| BboxArea | pixels | Area of the smallest bounding rectangle around the colony; a low Extent (Area/BboxArea) indicates a spread-out or irregular colony, or possible interference with a neighboring well. |
| MajorAxisLength / MinorAxisLength | pixels | Fitted ellipse dimensions. The ratio indicates elongation. |
| InscribedRadius | pixels | Radius of the largest circle that fits entirely inside the colony. Reflects the colony's narrowest dimension, not its overall extent — an elongated colony reports half its width regardless of length (a 100×20 px colony reports 10), and a runner or spur leaves it unchanged. |
| MedianRadius | pixels | Median distance from the colony's center to its boundary, sampled uniformly by angle around every piece of the colony; for an ideal disk it equals the disk radius. |
| MeanRadius | pixels | Mean distance from the colony's center to its boundary, sampled by angle; a runner or spur pulls it upward in proportion to its angular width (use RobustMeanRadius for the compact body). |
| RobustMeanRadius | pixels | Trimmed mean (20% from each end by default) of the same angularly-sampled boundary distances, discounting a runner or spur to estimate the radius of the colony's compact body. |
| MaxRadius | pixels | Largest angularly-sampled boundary distance from the colony's center — its farthest reach; a value far above RobustMeanRadius signals a protrusion, spur, or runner. |

Area is the most commonly used growth metric. For calibrated images,
convert pixels to physical units (mm²) using the known pixel pitch.

## Shape Metrics (MeasureShape)

| Metric | Range | Biological meaning |
|--------|-------|-------------------|
| Circularity | 0–1 | How round the colony is (1 = perfect circle). Irregular shapes suggest stress, mutation, or sectoring. |
| Solidity | 0–1 | Ratio of area to convex hull area. Low solidity indicates lobed or branching morphology. |
| Eccentricity | 0–1 | Elongation (0 = circular, ~1 = very elongated). Elevated in filamentous or swarming colonies. |
| Compactness | ≥1 | Perimeter² / (4π × Area). Equals 1 for a perfect circle; increases with irregularity. |
| MeanBoundaryDist | pixels | Mean distance from each colony pixel to the nearest edge, computed on the object in isolation — a measure of interior thickness, not a radius. High values relative to InscribedRadius indicate a compact, convex colony; low values indicate a thin or filamentous one. |
| MedianBoundaryDist | pixels | Median distance from each colony pixel to the nearest edge — the same interior-thickness measure as MeanBoundaryDist, but more robust to boundary raggedness. |

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

Color measurements report compact, outlier-resistant colorimetric
summaries rather than a per-channel statistical suite. The default output
covers two perceptually meaningful spaces, CIE L\*a\*b\* and HSV.

**Two robust center colors (CIE L\*a\*b\*).** L\*a\*b\* is designed so
Euclidean distance approximates perceived color difference (ΔE76).

- `L*GeoMedian`, `a*GeoMedian`, `b*GeoMedian` — the **ΔE76 (Euclidean)
  geometric median** of the colony's pixels. The multivariate median
  (breakdown point 0.5), so specular highlights, agar bleed-through, and
  debris cannot drag the center color.
- `L*Medoid`, `a*Medoid`, `b*Medoid` — the **ΔE2000 medoid**: the single
  real colony pixel minimizing total ΔE2000 to all other pixels. Because
  ΔE2000 is not a true metric, a medoid (which needs no metric axioms) is
  used instead of a geometric median, and the result is always an actual
  colony color.

**Within-colony color consistency (ΔE2000, from the medoid).** A
perceptually-corrected uniformity profile:

- `DeltaE2000MedianFromMedoid` — robust "perceptual MAD."
- `DeltaE2000MeanFromMedoid` — the color-science uniformity standard.
- `DeltaE2000P95FromMedoid` — near-worst-case deviation; flags sectoring
  and contamination without being dominated by a single stray pixel.
- `LabTotalVariance` — the trace of the L\*a\*b\* covariance (var L\* +
  var a\* + var b\*), a single Euclidean spread scalar.

**Robust HSV.** Because hue is circular and HSV is not perceptually
uniform, each pixel is embedded into Cartesian cone coordinates before
the robust center is computed:

- `HueRobustMean`, `SaturationRobustMean`, `ValueRobustMean` — the
  cone-embedded geometric-median center converted back to H, S, V
  (circular-correct; unreliable hue at low saturation collapses toward
  the achromatic axis automatically).
- `HSVConeVariance` — the trace of the cone-Cartesian covariance.

**Plot swatch.** `MedoidColorHex` is the sRGB hex string of the ΔE2000
medoid color, intended **for visualization only** — it is never used as a
numeric measurement or analysis input.

These metrics are essential for:

- Quantifying pigmentation differences between strains
- Detecting sectoring (regions of different color within a colony)
- Normalizing color across imaging sessions (after `ColorCorrector`)

CIE XYZ and xy chromaticity remain available as opt-in legacy per-channel
suites (`MeasureColor(include_XYZ=True)`, `MeasureColor(include_xy=True)`)
but are hidden from the default colorimetric output.

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
