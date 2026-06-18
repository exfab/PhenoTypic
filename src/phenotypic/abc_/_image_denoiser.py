"""Marker ABC for noise-driven image enhancers."""

from __future__ import annotations

from abc import ABC

from ._image_enhancer import ImageEnhancer


class ImageDenoiser(ImageEnhancer, ABC):
    """Marker ABC for noise-driven enhancers that operate on ``detect_mat``.

    Subclasses are conventional :class:`ImageEnhancer` operations whose
    primary parameter is a noise estimate (``sigma``, ``sigma_psd``,
    ``sigma_color``, ``h``, ...). The base class adds no new abstract
    methods -- it categorizes the family for documentation, GUI listing,
    and shared tooling.

    All :class:`ImageDenoiser` subclasses inherit the integrity check from
    :meth:`ImageEnhancer.apply`, which protects ``image.rgb`` and
    ``image.gray`` from mutation. Denoising is therefore confined to
    ``image.detect_mat``.

    Concrete subclasses typically also mix in
    :class:`phenotypic.sdk_.mixin._GATSupportMixin` to expose the optional
    Generalized Anscombe Transform variance-stabilization pairing for
    Poisson-Gaussian noise.

    **Quick Decision Guide:**

    - **ImageDenoiser (this class):** Noise-driven preprocessing on
      ``detect_mat``. Examples: BM3D, BayesShrink, VisuShrink, NLM,
      bilateral.
    - **ImageEnhancer (parent):** Any other ``detect_mat`` preprocessing
      (contrast, blur, ridge filters, morphology).

    See Also:
        :class:`phenotypic.abc_.ImageEnhancer` for the broader enhancer
        contract; :class:`phenotypic.sdk_.mixin._GATSupportMixin` for
        optional variance stabilization.
    """
