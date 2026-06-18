"""Shared helpers for uploading image arrays as napari layers."""

from __future__ import annotations

import numpy as np


def add_image_layer(viewer, array, *, name: str, visible: bool = True, gamma: float = 1.0):
    """Add (or replace) an image layer with non-stretched contrast limits.

    Mirrors the contrast convention of the detect-mode preview tool
    (``DetectMat.preview_modes``): integer layers use the full dtype range and
    single-channel normalized float intensities use a fixed ``(0.0, 1.0)``
    range. This renders each layer at its true brightness instead of letting
    napari auto-stretch the display to the data's own ``(min, max)`` — which
    exaggerates contrast and makes layers look washed out when several are
    shown together. Three-dimensional arrays are treated as RGB.

    Args:
        viewer: A napari ``Viewer`` (or any object exposing ``layers`` and
            ``add_image``).
        array: The image array — 2-D intensity or 3-D RGB.
        name: Layer name. An existing layer with this name is updated in place
            rather than duplicated.
        visible: Initial layer visibility. Defaults to True.
        gamma: Display gamma. Defaults to 1.0 (no gamma adjustment).

    Returns:
        The created or updated napari image layer.
    """
    imdata = np.asarray(array)
    is_rgb = imdata.ndim == 3

    contrast_limits: tuple[float, float]
    if np.issubdtype(imdata.dtype, np.integer):
        contrast_limits = (0, int(np.iinfo(imdata.dtype).max))
    else:
        contrast_limits = (0.0, 1.0)

    try:
        layer = viewer.layers[name]
        layer.data = imdata
        layer.contrast_limits = contrast_limits
        return layer
    except KeyError:
        return viewer.add_image(
            imdata,
            name=name,
            rgb=is_rgb,
            visible=visible,
            contrast_limits=contrast_limits,
            gamma=gamma,
        )
