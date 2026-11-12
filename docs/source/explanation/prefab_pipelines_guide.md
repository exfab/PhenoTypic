# Prefab Pipelines: Which One for Which Organism

Prefab pipelines are pre-configured `ImagePipeline` subclasses that
bundle operations, refiners, and measurements into a single ready-to-use
workflow. Each is tuned for a specific imaging scenario.

## Overview

| Prefab                     | Organism                          | Plate type         | Key strategy                            |
|----------------------------|-----------------------------------|--------------------|-----------------------------------------|
| `HeavyOtsuPipeline`        | Yeast, bacteria                   | 96/384-well grid   | Multi-stage Otsu with refinement        |
| `HeavyWatershedPipeline`   | Yeast, bacteria                   | Dense grids        | Watershed for touching colonies         |
| `RoundPeaksPipeline`       | Yeast                             | Grid plates        | Peak detection, lightweight             |
| `HeavyRoundPeaksPipeline`  | Yeast                             | Grid plates        | Extended peak detection with refinement |
| `FilamentousFungiPipeline` | *Neurospora*, *Aspergillus*, etc. | Grid plates        | BM3D + FilamentousFungiDetector         |
| `GridSectionPipeline`      | Any                               | Pre-tiled sections | Per-section processing                  |
| `BacterSamPipeline`        | Bacteria                          | 96-well grid       | SAM2 on a smoothed, flattened image     |

## HeavyOtsuPipeline

**Best for:** General-purpose colony detection on clean, well-lit plates
with uniform backgrounds.

**Pipeline steps:**
BlurGauss → EnhanceLocalContrast → MedianFilter → FocusEdgeSobel → OtsuDetector →
MaskOpening → RemoveBorderObjects → RemoveSmallObjects → MaskFill →
RemoveOversizedGridObjects → ReduceSectionsByLine → GridAligner

**Measurements:** MeasureSize, MeasureShape, MeasureColor, MeasureTexture,
MeasureIntensity

**When to choose:** This is the default starting point. Try it first;
switch to a specialized prefab only if results are unsatisfactory.

## HeavyWatershedPipeline

**Best for:** Plates where colonies grow large enough to touch or
overlap. Watershed segmentation separates merged colonies that Otsu
would detect as a single object.

**When to choose:** Colony overlap is visible in the overlay — adjacent
colonies sharing a boundary without clear separation.

## RoundPeaksPipeline

**Best for:** Grid plates with well-separated round colonies. Faster
than the Heavy pipelines because it uses fewer refinement steps.

**When to choose:** Speed matters and colonies are round, well-spaced,
and clearly visible.

## FilamentousFungiPipeline

**Best for:** Filamentous organisms with branching hyphae. Uses BM3D
denoising, homomorphic illumination correction, and the specialized
`FilamentousFungiDetector` with Dijkstra reconnection.

**When to choose:** The organism grows as a mycelial network rather
than a compact round colony. Standard detectors miss the outer hyphae.

## BacterSamPipeline

**Best for:** Bacterial colonies pinned in a grid and imaged over a growth
time course, where a global threshold merges colony edges with agar texture.

**Input assumptions:** Images must already be colour-corrected and denoised.
The pipeline does neither. Run a pipeline with `ColorCorrector` or
`CalibrateColorRpcc` followed by `ColorDenoise` first, export it with
`python -m phenotypic --mode process --layer rgb`, and give the resulting
OME-Zarr tree to this pipeline as `--input`.

**Pipeline steps:**
CropImage → BlurGauss → SubtractGaussian → Sam2 → RemoveLowCircularity →
RemoveBorderObjects → SmallObjectRemover → KeepSectionLargest

**Measurements:** MeasureNeighborDist, MeasureShape, MeasureSize,
MeasureIntensity, MeasureColor, and MeasureTexture at 10, 20, 30 and 40 px

**When to choose:** Classical detectors miss small early colonies or split
large late ones, and a GPU is available. The CLI splits the run into CPU
preprocessing, GPU detection and CPU measurement on its own. The crop
defaults fit one imager, so check them against your plates first.

The GUI pipeline builder cannot load this prefab yet, because it does not
offer GPU detectors. Use it from Python or the CLI.

## Customizing Prefabs

Prefabs expose their key parameters through `__init__`:

```python
from phenotypic.prefab import HeavyOtsuPipeline

pipeline = HeavyOtsuPipeline(
    gaussian_sigma=3,       # Stronger blur
    small_object_min_size=100,  # Remove larger fragments
    benchmark=True,         # Track execution times
)
```

For more extensive customization, build a pipeline from scratch using
`ImagePipeline(ops=[...], meas=[...])`.
