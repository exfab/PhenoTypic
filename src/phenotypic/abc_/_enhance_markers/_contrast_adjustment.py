"""Marker ABC for contrast-adjusting image enhancers."""

from __future__ import annotations

from abc import ABC

from .._image_enhancer import ImageEnhancer


class ContrastAdjustment(ImageEnhancer, ABC):
    """Marker ABC for intensity/contrast remapping enhancers on ``detect_mat``.

    Subclasses are conventional :class:`ImageEnhancer` operations that
    remap ``detect_mat`` intensities to make faint colonies more visible --
    histogram equalisation, percentile stretching, inversion, or unsharp
    detail boosting. They adjust *how* existing structure is displayed
    rather than detecting new structure; the output still looks like the
    plate, with contrast rebalanced. The base class adds no new abstract
    methods -- it categorizes the family for documentation, GUI listing,
    and shared tooling.

    All :class:`ContrastAdjustment` subclasses inherit the integrity check
    from :meth:`ImageEnhancer.apply`, which protects ``image.rgb`` and
    ``image.gray`` from mutation. Contrast adjustment is therefore confined
    to ``image.detect_mat``.

    **Quick Decision Guide:**

    - **ContrastAdjustment (this class):** Remaps intensities/contrast.
      Examples: adaptive histogram equalisation, contrast stretching,
      inversion, unsharp masking.
    - **FocusEdge:** Replaces the plate with an edge response, rather
      than rebalancing its contrast.
    - **ImageEnhancer (parent):** Any other ``detect_mat`` preprocessing.

    See Also:
        :class:`phenotypic.abc_.ImageEnhancer` for the broader enhancer
        contract.
    """
