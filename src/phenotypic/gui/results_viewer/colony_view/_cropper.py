"""Per-colony overlay cropping for the results viewer's colony-view tab.

The cropper itself is tab-agnostic, so the implementation now lives in
:mod:`phenotypic.gui._shared.tiles` and is shared with the QC review
tab. This module re-exports :func:`crop_overlay` (and the LRU
:func:`_load_overlay_rgb` cache) so the colony view's existing import
path keeps working.

See :func:`phenotypic.gui._shared.tiles.crop_overlay` for the full
contract: it slices a fixed-size, centroid-aligned square crop out of
the overlay PNG the CLI writes at
``<root>/deliverables/overlays/<dataset>/<stem>.png``, padding any portion
that spills past the image edge so the result is always exactly
``size`` x ``size`` PNG-encoded bytes.
"""

from __future__ import annotations

from phenotypic.gui._shared.tiles import (
    _load_overlay_rgb,
    crop_colony,
    crop_overlay,
)

__all__ = ["crop_overlay", "crop_colony", "_load_overlay_rgb"]
