"""Marker ABC for background-subtracting image enhancers."""

from __future__ import annotations

from abc import ABC

from .._image_enhancer import ImageEnhancer


class BackgroundSubtraction(ImageEnhancer, ABC):
    """Marker ABC for enhancers that remove slow-varying background.

    Subclasses are conventional :class:`ImageEnhancer` operations that
    estimate a low-frequency background -- uneven illumination, scanner
    vignetting, agar-thickness gradients -- and subtract it from
    ``detect_mat`` so colonies sit on a flat baseline. The background model
    may be a large Gaussian, a rolling ball, a wide morphological opening,
    or a homomorphic illumination estimate. The output still looks like the
    plate, with the gradient flattened. The base class adds no new abstract
    methods -- it categorizes the family for documentation, GUI listing,
    and shared tooling.

    All :class:`BackgroundSubtraction` subclasses inherit the integrity
    check from :meth:`ImageEnhancer.apply`, which protects ``image.rgb``
    and ``image.gray`` from mutation. Background subtraction is therefore
    confined to ``image.detect_mat``.

    **Quick Decision Guide:**

    - **BackgroundSubtraction (this class):** Removes large-scale
      background/illumination. Examples: subtract-Gaussian, rolling ball,
      wide morphological opening, illumination flattening.
    - **MorphologicalFiltering:** Targets *small* bright/dark structures
      with a compact structuring element, not the broad baseline.
    - **ImageEnhancer (parent):** Any other ``detect_mat`` preprocessing.

    See Also:
        :class:`phenotypic.abc_.ImageEnhancer` for the broader enhancer
        contract; :class:`phenotypic.abc_.MorphologicalFiltering` for the
        small-feature morphology sibling.
    """
