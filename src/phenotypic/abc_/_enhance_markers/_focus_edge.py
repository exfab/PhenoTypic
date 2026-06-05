"""Marker ABC for edge- and ridge-isolating image enhancers."""

from __future__ import annotations

from abc import ABC

from .._image_enhancer import ImageEnhancer


class FocusEdge(ImageEnhancer, ABC):
    """Marker ABC for enhancers whose ``detect_mat`` output *is* the edges.

    Subclasses are conventional :class:`ImageEnhancer` operations that
    replace ``detect_mat`` with a feature-response map -- gradient,
    Laplacian, phase-congruency, or Hessian ridge/vesselness strength --
    rather than a cleaned-up version of the plate. After an
    :class:`FocusEdge` step the matrix no longer looks like the agar
    plate; bright pixels mark colony boundaries, septa, hyphal walls, and
    other high-curvature structure. The base class adds no new abstract
    methods -- it categorizes the family for documentation, GUI listing,
    and shared tooling.

    All :class:`FocusEdge` subclasses inherit the integrity check from
    :meth:`ImageEnhancer.apply`, which protects ``image.rgb`` and
    ``image.gray`` from mutation. Edge isolation is therefore confined to
    ``image.detect_mat``.

    **Quick Decision Guide:**

    - **FocusEdge (this class):** Output is an edge/ridge response map.
      Examples: Sobel, Laplacian, Frangi/Sato/Meijering ridge filters,
      Hessian, phase congruency.
    - **FocusBlob:** Output is a blob/scale-space response (filled
      regions, not boundaries).
    - **ImageEnhancer (parent):** Any other ``detect_mat`` preprocessing
      that keeps the plate appearance (denoise, smooth, contrast,
      background subtraction, morphology).

    See Also:
        :class:`phenotypic.abc_.ImageEnhancer` for the broader enhancer
        contract; :class:`phenotypic.abc_.FocusBlob` for the
        region-response sibling.
    """
