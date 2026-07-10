# What Enhancement Actually Does

Enhancers modify the `detect_mat` accessor — the grayscale representation
that detectors consume. They never touch `rgb` or `gray`, preserving the
original image data for visualization and color measurements.

## Why Enhance?

Raw plate images often have properties that confuse detectors:

- **Noise** — random pixel variation that creates false positives
- **Low contrast** — faint colonies barely distinguishable from agar
- **Uneven illumination** — brightness gradients across the plate
- **Texture** — agar surface patterns that mimic colony edges

Enhancement addresses these problems by transforming `detect_mat` into
a version where colonies stand out more clearly.

## Categories of Enhancement

### Smoothing (Noise Reduction)

Reduce random variation while preserving colony boundaries.

- **GaussianBlur** — isotropic smoothing; fast but blurs edges
- **MedianFilter** — removes salt-and-pepper noise; preserves edges better
- **LocalEdgeDenoise** — smooths within regions; preserves edges explicitly
- **DenoiseBlockMatch (BM3D)** — state-of-the-art block-matching denoising

### Contrast Enhancement

Increase the separation between colony and background intensities.

- **EnhanceLocalContrast** — local adaptive histogram equalization; handles spatially
  varying contrast
- **ContrastStretching** — linear remapping to fill the dynamic range
- **SharpenEdgeGauss** — sharpens edges by subtracting a blurred version

### Illumination Correction

Remove large-scale brightness gradients.

- **FlattenIllumination** — frequency-domain separation of illumination
  and reflectance
- **SubtractGaussian** — subtracts a heavily blurred background estimate
- **SubtractRollingBall** — morphological background estimation

### Structural Enhancement

Enhance specific morphological features.

- **FocusEdgeFrangi** — enhances tubular structures (hyphae, branches)
- **FocusEdgeSobel** — highlights edges
- **FocusEdgePhase** — illumination-invariant edge detection
- **FocusEdgeMonogenicPhase** — the same illumination invariance without the orientation
  sweep, via the monogenic signal. Cheaper and isotropic; use it when colony edges have no
  preferred direction. Its `output="orientation"` / `"feature_type"` modes are diagnostic
  angle maps, not detection inputs.
- **FocusEdgeColorPhase** — runs the monogenic chain on three colour channels and fuses
  them, so a channel with amplitude but no phase agreement can *veto* a spurious luminance
  edge. **It reads `rgb`, not `detect_mat`**, which makes it a pipeline *source*: any
  enhancer placed before it has no effect. Reach for it on **filamentous** plates. On
  round-colony plates, `FocusEdgeMonogenicPhase` on luminance measurably localizes
  boundaries better than any of its fusion modes, so colour buys nothing there. Rejects
  achromatic images outright.

## Stacking Enhancers

Enhancers compose linearly — each reads `detect_mat`, modifies it, and
writes it back. Order matters:

```{note}
**One exception.** `FocusEdgeColorPhase` reads `image.rgb`, not `detect_mat`, because
colour phase congruency is defined on colour. It is a *source*: it discards whatever
`detect_mat` held and writes a fresh map. Anything upstream of it in the chain is wasted
work. Put it first, or not at all.
```

1. **Denoise first** — reduce noise before amplifying contrast
2. **Correct illumination** — normalize brightness before thresholding
3. **Enhance contrast** — maximize colony/background separation last

A typical preprocessing chain:
`GaussianBlur → FlattenIllumination → EnhanceLocalContrast`.

## The Enhancement ↔ Detection Interface

```
detect_mat  ──[Enhancer 1]──→  detect_mat  ──[Enhancer 2]──→  detect_mat
                                                                    │
                                                              [Detector]
                                                                    │
                                                              objmask, objmap
```

This clean interface means you can swap any enhancer without affecting
the detector, and vice versa. The pipeline model makes this
experimentation easy.
