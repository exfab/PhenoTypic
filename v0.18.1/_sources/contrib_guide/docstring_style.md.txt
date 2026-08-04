# ImageOperation Docstring Style Guide

This guide defines the docstring template for all `ImageOperation` subclasses in
PhenoTypic. The goal is a **layered, progressive disclosure** structure: readers
can stop early and still get value, while those who need depth can continue.

Detailed algorithm discussion belongs in [Explanation pages](../explanation/),
not in docstrings. Docstrings answer "what does this do and when should I use
it?" -- Explanation pages answer "how and why does it work?"

## Target Length

| Operation complexity | Target lines | Example |
|---------------------|-------------|---------|
| Simple (1-3 params) | 25--40 | `BlurGauss`, `MaskDilation` |
| Moderate (4-8 params) | 40--60 | `OtsuDetector`, `EnhanceLocalContrast` |
| Complex (8+ params) | 60--90 | `FilamentousFungiDetector`, `RoundPeaksDetector` |

These are guidelines, not hard limits. Prefer clarity over brevity, but respect
the reader's time -- every line should earn its place.

## Template

```python
class ExampleDetector(ObjectDetector):
    """One-line summary in imperative mood, stating what the operation does.

    Extended summary: 2-3 sentences describing the mechanism and expected
    output. Frame in terms of what the user will observe, not the internal
    algorithm. Link to an Explanation page for algorithm details.

    For algorithm details, see :doc:`/explanation/detection_strategies`.

    Best For:
        - Scenario where this operation excels.
        - Another ideal use case.
        - Characteristic of images where this works well.

    Consider Also:
        - :class:`AlternativeOp` when [scenario where it is better suited].
        - :class:`AnotherOp` for [different scenario].

    Args:
        param1: Description. Typical range: X--Y. Default: Z.
        param2: Description. Higher values produce [effect].

    Returns:
        Image: Description of what changed on the returned image.

    Raises:
        ValueError: When [condition].

    Examples:
        >>> from phenotypic import load_synth_yeast_plate
        >>> plate = load_synth_yeast_plate()
        >>> result = ExampleDetector().apply(plate)
        >>> result.num_objects > 0
        True

    References:
        [1] A. Author, "Paper title," *Journal*, vol. X, no. Y,
        pp. Z--Z, Month Year.
    """
```

## Section-by-Section Rules

### One-Line Summary

- Imperative mood: "Detect colonies using..." not "Detects colonies using..."
- State the mechanism, not just the category: "Detect colonies by finding an
  optimal intensity threshold" not "Detect colonies"
- Do not repeat the class name

```python
# Good
"""Detect colonies by minimizing intra-class variance of the intensity histogram."""

# Bad
"""OtsuDetector detects things."""
"""A detector that uses Otsu's method to detect colonies in images."""
```

### Extended Summary

2-3 sentences maximum. Describe:

1. What the operation does to the image (observable effect)
2. What kind of result the user should expect
3. Link to the Explanation page for the underlying algorithm

```python
"""Detect colonies by minimizing intra-class variance of the intensity histogram.

Computes a single global threshold that separates colony pixels from background.
Works best when the image histogram has two distinct peaks. Returns an image with
objmask set to the thresholded binary mask and objmap set to labeled connected
components.

For a comparison of thresholding strategies, see
:doc:`/explanation/detection_strategies`.
"""
```

Do **not** explain the algorithm here. "Minimizing intra-class variance" is
sufficient -- the reader who wants to know how that works follows the link.

### Best For

Positive framing only. Describe the scenarios, image characteristics, or
experimental setups where this operation excels.

**Rules:**

- 3-5 bullet points
- Use concrete, observable characteristics: "plates with uniform illumination"
  not "well-conditioned inputs"
- Use microbiology context: "dense yeast colonies" not "many small objects"
- Do not qualify with "only" or "exclusively" -- these are ideal scenarios, not
  the only valid ones

```python
Best For:
    - Plates with uniform illumination and clean backgrounds.
    - Yeast colonies that appear as bright spots on dark agar.
    - Images where the intensity histogram shows two distinct peaks.
    - Quick prototyping before tuning a more specialized detector.
```

### Consider Also

Suggest alternatives without discouraging use. This section helps the reader
discover operations they may not know about and navigate to the right tool.

**Rules:**

- 2-4 bullet points
- Always name a specific class with a Sphinx cross-reference
- Describe *when* the alternative is better suited, not *why* the current
  operation fails
- Use "when" or "for" phrasing, never "don't use" or "avoid"

```python
Consider Also:
    - :class:`HysteresisDetector` when colonies have soft edges that a single
      threshold misses.
    - :class:`RoundPeaksDetector` for dense grid plates with known colony
      spacing.
    - :class:`WatershedDetector` when touching colonies need to be separated.
```

**What not to write:**

```python
# Bad -- falsely prescriptive
Consider Also:
    - Don't use this on noisy images; use HysteresisDetector instead.
    - This will fail on filamentous fungi.

# Bad -- no cross-reference, no context
Consider Also:
    - HysteresisDetector
    - WatershedDetector
```

### Args

Google-style [1]. Each parameter gets a one-line description followed by
practical guidance.

**Rules:**

- Include the default value if not obvious from the signature
- Include the typical range or meaningful values for numeric parameters
- Group related parameters with a blank line and a comment if there are more
  than 8 parameters
- Inline tuning guidance replaces a separate "Parameter Effects" section

```python
Args:
    sigma: Standard deviation of the Gaussian kernel in pixels. Controls
        blur strength. Typical range: 0.5--5.0. Default: 1.0.
    ignore_zeros: Exclude zero-valued pixels from threshold computation.
        Enable for images with black borders or padding. Default: False.

    # Reconnection parameters (advanced)
    tile_size: Side length of square tiles for tiled processing.
        Larger tiles use more memory. Default: 1200.
```

For parameters that accept string method names, list the accepted values:

```python
Args:
    method: Threshold selection strategy. Accepted values: ``'otsu'``,
        ``'triangle'``, ``'li'``, ``'yen'``, ``'isodata'``, ``'mean'``,
        ``'minimum'``. Default: ``'otsu'``.
```

### Returns

Brief description of what changed on the returned `Image`. Specify which
components were modified.

```python
Returns:
    Image: Input image with ``objmask`` set to the thresholded binary mask
    and ``objmap`` set to labeled connected components.
```

For enhancers:

```python
Returns:
    Image: Input image with ``detect_mat`` smoothed by the Gaussian kernel.
    ``rgb`` and ``gray`` are unchanged.
```

### Raises

Only document exceptions the user can trigger through parameter choices or input
characteristics. Do not document internal assertion errors.

```python
Raises:
    ValueError: If ``sigma`` is not positive.
    TypeError: If ``detector`` is not an ObjectDetector or ImagePipeline.
```

### Examples

Include a runnable **doctest** ``Examples`` section. Every example must execute
against the bundled ``load_synth_yeast_plate()`` fixture so the docstrings run
under ``pytest --doctest-modules``. Keep output deterministic (assert a boolean
or a small invariant rather than printing arrays), and use microbiology context
(colony visibility, edge sharpness, mask quality). Place the section after
``Raises`` and before ``References``.

```python
Examples:
    >>> from phenotypic import load_synth_yeast_plate
    >>> from phenotypic.detect import OtsuDetector
    >>> plate = load_synth_yeast_plate()
    >>> detected = OtsuDetector().apply(plate)
    >>> detected.num_objects > 0
    True
```

For visual, interactive demonstrations that text output cannot convey, **also**
point readers to a tutorial/how-to via `See Also` — but the runnable doctest
stays in the docstring. Do not move examples into separate notebooks or example
files; examples live in the docstring.

### References

IEEE citation style [2]. Include when the docstring names a specific published
method. Place at the end of the docstring, after Examples.

```python
References:
    [1] N. Otsu, "A threshold selection method from gray-level histograms,"
    *IEEE Trans. Syst., Man, Cybern.*, vol. 9, no. 1, pp. 62--66,
    Jan. 1979.
```

**When to include references:**

- The operation implements a named algorithm (Otsu, Frangi, BM3D, EnhanceLocalContrast, etc.)
- The operation references a specific paper's formulation or parameters
- The operation uses a domain-specific technique from published literature

**When to omit:**

- Generic operations (Gaussian blur, median filter, morphological open/close) --
  these are standard and do not require citation
- Compositions of other operations (prefab pipelines)

### See Also

Link to relevant user guide pages and Explanation pages. This section
complements the in-docstring example and the algorithm discussion on Explanation
pages by directing readers to interactive visual demonstrations and conceptual
deep-dives.

**Rules:**

- Include at least one link: either a user guide page (tutorial or how-to) or an
  Explanation page
- User guide links point to pages where the operation is demonstrated with
  interactive Plotly output
- Explanation links point to pages covering the underlying algorithm or theory
- Order: user guide links first (practical), then explanation links (conceptual)

```python
See Also:
    :doc:`/tutorials/notebooks/detecting_colonies` for a visual walkthrough
    of this detector on real plate images.
    :doc:`/explanation/detection_strategies` for a comparison of thresholding
    methods and their failure modes.
```

## Adapting the Template by ABC Type

The template above is the general form. Each ABC type has slight variations in
emphasis.

### ImageEnhancer

- **Extended summary** should describe the visual effect on `detect_mat`
- **Best For** focuses on image quality problems the enhancer solves
- **Returns** always notes that `rgb` and `gray` are unchanged
- **Consider Also** points to other enhancers, not detectors

### ObjectDetector / ThresholdDetector

- **Extended summary** should describe what the resulting mask looks like
- **Best For** focuses on plate/image characteristics
- **Consider Also** points to other detectors with different strategies

### ObjectRefiner

- **Extended summary** should describe what mask artifacts it fixes
- **Best For** focuses on detection artifacts (fragmentation, noise, oversized
  objects)
- **Consider Also** points to refiners that address related but different
  artifacts

### ImageCorrector

- **Extended summary** should describe the geometric or radiometric
  transformation
- **Best For** focuses on acquisition artifacts (vignetting, rotation, cropping)
- **Returns** notes that all image components are transformed

### MeasureFeatures

- **Extended summary** should describe what columns appear in the output
  DataFrame
- **Best For** focuses on the biological questions the measurements answer
- **Returns** describes the DataFrame structure, not an Image

### PrefabPipeline

- **Extended summary** should list the pipeline steps as a numbered sequence
- **Best For** focuses on the organism type and imaging conditions
- **Consider Also** points to other prefab pipelines for different organisms
- Include a **Steps** section listing each operation in order

## Common Mistakes

**Writing algorithm explanations in the docstring.** Move these to an Explanation
page and link to it. The docstring says "minimizes intra-class variance" -- the
Explanation page explains what intra-class variance is and why minimizing it
works.

**Listing every possible use case.** "Best For" should have 3-5 representative
bullets, not an exhaustive list. The reader infers applicability from the
described characteristics.

**Using negative framing in "Consider Also."** Never write "don't use this
when..." or "this fails on..." -- frame as a positive recommendation of an
alternative. Image processing is empirical; an operation may work in cases you
would not predict.

**Duplicating parameter documentation.** Args appear in the class docstring, not
in `__init__`. Sphinx's `autodoc` will render the class docstring as the primary
documentation.

**Omitting practical tuning guidance from Args.** A bare "sigma: Standard
deviation" is insufficient. Include the typical range and the effect of adjusting
the parameter.

**Omitting the runnable doctest example.** Every operation docstring needs an
``Examples`` section that executes against ``load_synth_yeast_plate()`` under
``pytest --doctest-modules``. Link to user guide tutorials for the *visual*
walkthrough via `See Also`, but keep the runnable example in the docstring.

## Worked Example: OtsuDetector

Below is a complete docstring following this template, for reference.

```python
class OtsuDetector(ThresholdDetector):
    """Detect colonies by minimizing intra-class variance of the intensity histogram.

    Computes a single global threshold that separates colony pixels from
    background. Works best when the image histogram has two distinct peaks
    (bimodal distribution). Returns an image with objmask set to the thresholded
    binary mask and objmap set to labeled connected components.

    For a comparison of thresholding strategies, see
    :doc:`/explanation/detection_strategies`.

    Best For:
        - Plates with uniform illumination and clean backgrounds.
        - Yeast colonies that appear as distinct spots against agar.
        - Images where the intensity histogram shows two clear peaks.
        - Quick initial detection before refining with specialized operations.

    Consider Also:
        - :class:`HysteresisDetector` when colonies have soft or noisy edges
          that a single threshold misses.
        - :class:`RoundPeaksDetector` for dense grid plates with known colony
          spacing.
        - :class:`TriangleDetector` when one peak is much larger than the
          other (e.g., few colonies on a large plate).

    Args:
        ignore_zeros: Exclude zero-valued pixels from the histogram before
            computing the threshold. Enable when images have black borders
            or padding regions. Default: False.
        ignore_borders: Exclude a 1-pixel border from the threshold
            computation. Default: False.
        connectivity: Pixel connectivity for labeling connected components
            after thresholding. ``1`` for 4-connectivity, ``2`` for
            8-connectivity. Default: 2.

    Returns:
        Image: Input image with ``objmask`` set to the thresholded binary
        mask and ``objmap`` set to labeled connected components.

    Raises:
        ValueError: If the threshold computation fails on a uniform image.

    Examples:
        >>> from phenotypic import load_synth_yeast_plate
        >>> from phenotypic.detect import OtsuDetector
        >>> plate = load_synth_yeast_plate()
        >>> detected = OtsuDetector().apply(plate)
        >>> detected.num_objects > 0
        True

    References:
        [1] N. Otsu, "A threshold selection method from gray-level
        histograms," *IEEE Trans. Syst., Man, Cybern.*, vol. 9, no. 1,
        pp. 62--66, Jan. 1979.

    See Also:
        :doc:`/tutorials/notebooks/detecting_colonies` for a visual
        walkthrough of threshold-based detection on real plate images.
        :doc:`/explanation/detection_strategies` for a comparison of
        thresholding methods and their failure modes.
    """
```

## References

[1] Google, "Google Python style guide: docstrings," 2024. [Online]. Available:
https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings

[2] IEEE, "IEEE reference guide," 2024. [Online]. Available:
https://ieee-dataport.org/sites/default/files/analysis/27/IEEE%20Citation%20Guidelines.pdf
