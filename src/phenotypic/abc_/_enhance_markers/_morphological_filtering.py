"""Marker ABC for morphological image enhancers."""

from __future__ import annotations

from abc import ABC

from .._image_enhancer import ImageEnhancer


class MorphologicalFiltering(ImageEnhancer, ABC):
    """Marker ABC for structuring-element enhancers on ``detect_mat``.

    Subclasses are conventional :class:`ImageEnhancer` operations that
    reshape ``detect_mat`` with a compact morphological structuring
    element -- opening, white top-hat, or their differences -- to isolate
    or suppress *small* bright structures (dust, speckle, satellite
    colonies, glare). They typically mix in
    :class:`phenotypic.tools_.mixin.FootprintMixin` for ``_make_footprint``.
    The base class adds no new abstract methods -- it categorizes the
    family for documentation, GUI listing, and shared tooling.

    All :class:`MorphologicalFiltering` subclasses inherit the integrity
    check from :meth:`ImageEnhancer.apply`, which protects ``image.rgb``
    and ``image.gray`` from mutation. Morphological filtering is therefore
    confined to ``image.detect_mat``.

    **Quick Decision Guide:**

    - **MorphologicalFiltering (this class):** Compact structuring-element
      ops on small features. Examples: grayscale opening, white top-hat,
      top-hat suppression.
    - **BackgroundSubtraction:** Removes the broad low-frequency baseline,
      not small features.
    - **ImageEnhancer (parent):** Any other ``detect_mat`` preprocessing.

    See Also:
        :class:`phenotypic.abc_.ImageEnhancer` for the broader enhancer
        contract; :class:`phenotypic.tools_.mixin.FootprintMixin` for
        structuring-element construction.
    """
