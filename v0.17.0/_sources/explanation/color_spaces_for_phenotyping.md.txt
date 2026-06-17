# Color Spaces for Phenotyping

Colony color is a biologically meaningful signal — pigmentation changes
can indicate metabolic state, gene expression, or environmental stress.
PhenoTypic provides access to multiple color spaces through the
`image.color` accessor, each suited to different analytical questions.

## Available Color Spaces

### RGB

The raw sensor output. Three channels (red, green, blue) represent the
image as captured. RGB values depend on the camera sensor and illumination,
making them poorly suited for comparing color across different imaging
setups.

Access: `image.rgb[:]`

### CIELAB (L\*a\*b\*)

A perceptually uniform color space. The Euclidean distance between two
CIELAB colors correlates with the perceived color difference. This makes
CIELAB the preferred space for measuring colony color differences.

- **L\*** — lightness (0 = black, 100 = white)
- **a\*** — green–red axis (negative = green, positive = red)
- **b\*** — blue–yellow axis (negative = blue, positive = yellow)

Access: `image.color.Lab[:]`

### HSV

Hue-Saturation-Value separates chromatic content (hue) from intensity
(value). Useful for segmenting colonies by pigmentation independent of
brightness, but not perceptually uniform.

Access: `image.color.hsv[:]`

### CIE XYZ and xy Chromaticity

The CIE 1931 XYZ color space is the foundation for all colorimetric
measurements. The xy chromaticity diagram is useful for visualizing the
gamut of colony colors and checking illuminant consistency.

Access: `image.color.XYZ[:]`, `image.color.xy[:]`

## Which Space to Use

| Question | Color space | Why |
|----------|------------|-----|
| Are these colonies the same color? | CIELAB | Perceptually uniform distances |
| Segment by pigmentation regardless of brightness | HSV (hue channel) | Hue is brightness-independent |
| Compare across imaging setups | CIELAB after `ColorCorrector` | Corrected to standard illuminant |
| Visualize color distribution | xy chromaticity | 2D projection, intuitive gamut view |

## Gamma and Illuminant

PhenoTypic assumes sRGB gamma encoding and D65 illuminant by default. These
are set at image creation and affect all color space conversions. If your
images use a different encoding, specify `gamma` and `illuminant` when
constructing the `Image` object.

## References

[1] CIE, "Colorimetry — Part 4: CIE 1976 L\*a\*b\* colour space,"
CIE 015:2004, 2004.
