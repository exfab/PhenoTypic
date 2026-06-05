"""Marker ABC for blob-isolating image enhancers."""

from __future__ import annotations

from abc import ABC

from .._image_enhancer import ImageEnhancer


class FocusBlob(ImageEnhancer, ABC):
    """Marker ABC for enhancers whose ``detect_mat`` output is a blob response.

    Subclasses are conventional :class:`ImageEnhancer` operations that
    replace ``detect_mat`` with a scale-space blob-detector response (e.g.
    a scale-normalised Laplacian-of-Gaussian maximised across radii).
    Unlike :class:`FocusEdge`, which highlights boundaries, a blob
    response highlights *filled* circular structures -- whole colonies or
    inocula -- making size-invariant colony emphasis the goal. The base
    class adds no new abstract methods -- it categorizes the family for
    documentation, GUI listing, and shared tooling.

    All :class:`FocusBlob` subclasses inherit the integrity check from
    :meth:`ImageEnhancer.apply`, which protects ``image.rgb`` and
    ``image.gray`` from mutation. Blob isolation is therefore confined to
    ``image.detect_mat``.

    **Quick Decision Guide:**

    - **FocusBlob (this class):** Output is a blob/scale-space response
      that fills colony interiors. Example: scale-normalised LoG.
    - **FocusEdge:** Output is an edge/ridge response that marks
      boundaries, not interiors.
    - **ImageEnhancer (parent):** Any other ``detect_mat`` preprocessing
      that keeps the plate appearance.

    See Also:
        :class:`phenotypic.abc_.ImageEnhancer` for the broader enhancer
        contract; :class:`phenotypic.abc_.FocusEdge` for the
        boundary-response sibling.
    """
