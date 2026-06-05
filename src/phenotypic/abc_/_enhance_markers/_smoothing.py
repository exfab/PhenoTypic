"""Marker ABC for smoothing image enhancers."""

from __future__ import annotations

from abc import ABC

from .._image_enhancer import ImageEnhancer


class Smoothing(ImageEnhancer, ABC):
    """Marker ABC for kernel/diffusion smoothing enhancers on ``detect_mat``.

    Subclasses are conventional :class:`ImageEnhancer` operations that
    attenuate high-frequency detail with a fixed spatial kernel or a
    diffusion process -- Gaussian convolution, median/rank filtering, or
    anisotropic coherence-enhancing diffusion. They are driven by a
    geometric scale (footprint size, ``sigma``, diffusion steps) rather
    than a statistical noise estimate, which distinguishes them from
    :class:`ImageDenoiser`. The output still looks like the plate, just
    softer. The base class adds no new abstract methods -- it categorizes
    the family for documentation, GUI listing, and shared tooling.

    All :class:`Smoothing` subclasses inherit the integrity check from
    :meth:`ImageEnhancer.apply`, which protects ``image.rgb`` and
    ``image.gray`` from mutation. Smoothing is therefore confined to
    ``image.detect_mat``.

    **Quick Decision Guide:**

    - **Smoothing (this class):** Scale-driven blur/diffusion. Examples:
      Gaussian blur, median filter, structure-tensor diffusion.
    - **ImageDenoiser:** Noise-estimate-driven restoration (BM3D,
      BayesShrink, NLM, bilateral).
    - **ImageEnhancer (parent):** Any other ``detect_mat`` preprocessing.

    See Also:
        :class:`phenotypic.abc_.ImageEnhancer` for the broader enhancer
        contract; :class:`phenotypic.abc_.ImageDenoiser` for the
        noise-model-driven sibling.
    """
