"""Pure overlay backend for the ``/tune/`` Curate view (B-i).

The Curate Dash surface (built in B-ii) lets a user audit a tuning trial's
segmentation on a chosen plate: it renders the candidate's objmap over the
detect_mat, diffs two candidates' objects, and caches the rendered arrays on a
background thread. This module is the **pure**, Dash-free backend those
callbacks call:

* :func:`render_candidate_overlay` — ``build_pipeline(base, params)`` →
  ``pipeline.apply(plate)`` → an RGB ``label2rgb`` overlay **array** for a
  Plotly ``go.Image`` (NOT base64, NOT PNG bytes).

**Lazy-optuna lock.** Importing this module — like
:mod:`phenotypic.gui.tune` itself — must never drag ``optuna`` into
``sys.modules``. ``build_pipeline`` lives in the optuna-free
:mod:`phenotypic.tune._evaluation._builder`; the overlay core is the builder's
:func:`~phenotypic.gui.builder._image_renderer.to_overlay_rgb_array`. Neither
imports optuna, so the lock holds.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from phenotypic.gui.builder._image_renderer import to_overlay_rgb_array
from phenotypic.tune._evaluation._builder import build_pipeline

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from phenotypic import Image, ImagePipeline


def render_candidate_overlay(
    base_pipeline: "ImagePipeline",
    params: dict[str, Any],
    plate_image: "Image",
    *,
    max_dim: int = 640,
) -> np.ndarray:
    """Render a tuning candidate's segmentation as an RGB overlay array.

    Overlays ``params`` onto ``base_pipeline`` via
    :func:`~phenotypic.tune._evaluation._builder.build_pipeline` (the same
    flat ``{"<pos>.<field>": value}`` combo grammar the strategies use),
    applies the resulting pipeline to a fresh copy of ``plate_image``, and
    composites the detected objmap over the post-op detect_mat with
    ``skimage.color.label2rgb`` — the exact same core the builder's preview
    uses, so a tuned candidate and a hand-built pipeline look identical.

    The returned array is RGB and display-ready for a Plotly ``go.Image``
    trace; it is **not** PNG-encoded or base64-wrapped.

    Args:
        base_pipeline: The base :class:`~phenotypic.ImagePipeline` embedded in
            the tuning spec. Not mutated — ``build_pipeline`` deep-copies it.
        params: A flat combo (``{"<pos>.<field>": value}``, e.g.
            ``{"0.sigma": 2.0}``) addressing ops by position index, exactly as
            ``build_pipeline`` expects.
        plate_image: The plate :class:`~phenotypic.Image` (or
            :class:`~phenotypic.GridImage`) to segment. Copied before
            ``apply`` so the caller's image is untouched.
        max_dim: Maximum length of the longer spatial side of the overlay, in
            pixels. Defaults to ``640``.

    Returns:
        An ``(H, W, 3)`` uint8 RGB overlay array.

    Raises:
        IndexError / ValueError / pydantic.ValidationError: Propagated from
            ``build_pipeline`` when ``params`` carries a bad key or an
            out-of-bounds value (see its docstring).

    Examples:
        >>> from phenotypic import ImagePipeline
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic.enhance import GaussianBlur
        >>> base = ImagePipeline(ops=[GaussianBlur(sigma=1.0), OtsuDetector()])
        >>> overlay = render_candidate_overlay(
        ...     base, {"0.sigma": 2.0}, load_synth_yeast_plate()
        ... )
        >>> overlay.ndim, overlay.shape[2]
        (3, 3)
    """
    pipeline = build_pipeline(base_pipeline, params)
    segmented = pipeline.apply(plate_image.copy())
    return to_overlay_rgb_array(segmented, max_dim=max_dim)


__all__ = ["render_candidate_overlay"]
