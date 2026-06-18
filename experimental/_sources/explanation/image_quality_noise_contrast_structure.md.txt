# Image Quality: Noise, Contrast, and Structure

The quality of a plate image directly determines how well detectors and
enhancers can perform. PhenoTypic's `plot.diagnostics()` method provides
quantitative metrics for the three dimensions of image quality.

## Noise

Noise is random pixel-level variation that is not part of the true image
signal. It introduces false texture in the agar background and blurs
colony boundaries.

**Sources:** Sensor read noise, shot noise (photon counting), thermal
noise (long exposures), compression artifacts.

**Metrics:**

- **SNR (Signal-to-Noise Ratio)** — ratio of mean signal to noise
  standard deviation. Higher is better. Values below 10 indicate
  significant noise.
- **Correlation length** — the spatial extent of correlated noise.
  Short correlation (a few pixels) indicates random noise; long
  correlation suggests structured artifacts (e.g., banding, uneven
  illumination).

**Mitigation:** `GaussianBlur`, `MedianFilter`, `DenoiseBlockMatch` (BM3D),
`VisuShrinkEnhancer` (wavelet).

## Contrast

Contrast measures how well colonies separate from background in
intensity.

**Sources of low contrast:** Under-exposure, faint pigmentation (early
time points), agar that matches colony color, condensation on plate lid.

**Metrics:**

- **RMS contrast** — root-mean-square intensity deviation. Low values
  mean the histogram is narrow (everything looks similar).
- **Michelson contrast** — (Imax − Imin) / (Imax + Imin). Values near
  1.0 indicate strong separation; values near 0 mean colonies and
  background have similar brightness.
- **Dynamic range** — fraction of the theoretical bit depth in use. A
  16-bit image that only uses values 0–1000 has very low dynamic range.

**Mitigation:** `EnhanceLocalContrast`, `ContrastStretching`, `FlattenIllumination`.

## Structure

Structure metrics assess the spatial organization of intensity patterns —
how sharp the colony edges are and how consistent the spatial layout is.

**Metrics:**

- **Gradient mean** — average edge strength across the image. Higher
  values mean sharper colony boundaries, which makes threshold-based
  detection easier.
- **Coherence** — consistency of gradient orientation. High coherence on
  a grid plate suggests well-organized, regularly spaced colonies.

**Mitigation:** `SharpenEdgeGauss`, `FocusEdgeSobel` (enhance edges before
detection).

## Quality-Guided Pipeline Design

Assess image quality *before* choosing enhancers:

1. **High noise, adequate contrast** → denoise first, then detect
2. **Low noise, low contrast** → contrast enhancement, then detect
3. **Uneven illumination** → `FlattenIllumination` before anything else
4. **Soft edges** → `SharpenEdgeGauss` or edge-enhancing filter
5. **All metrics good** → skip enhancement, detect directly
