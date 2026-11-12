"""Private compatibility aliases for renamed serialized classes."""

from __future__ import annotations

from typing import Final


_LEGACY_CLASS_ALIASES: Final[dict[str, tuple[str, str]]] = {
    # Pre-0.17 serialized pipelines used the original BM3D operation name.
    # Resolve it here instead of exporting a module alias that would appear in
    # the GUI operation registry as a duplicate palette entry.
    "BM3DDenoiser": ("phenotypic.enhance", "EnhanceBlockMatch"),
    # Preserve pipelines written before the correction operations adopted
    # verb-first public names.
    "ImageCropper": ("phenotypic.correction", "CropImage"),
    "ImagePadder": ("phenotypic.correction", "PadImage"),
    # Preserve pipelines serialized before GaussianBlur adopted the
    # noun-first public name used by the other Gaussian operations.
    "GaussianBlur": ("phenotypic.enhance", "BlurGauss"),
    # Preserve pipelines written before the enhance and refine operations
    # adopted verb-first public names.
    "ImageInverter": ("phenotypic.enhance", "InvertImage"),
    "NonLocalMeansDenoiser": ("phenotypic.enhance", "DenoiseNonLocalMeans"),
    "RankMedianEnhancer": ("phenotypic.enhance", "EnhanceRankMedian"),
    "BayesShrinkEnhancer": ("phenotypic.enhance", "DenoiseBayesShrink"),
    "VisuShrinkEnhancer": ("phenotypic.enhance", "DenoiseVisuShrink"),
    "GridOversizedObjectRemover": ("phenotypic.refine", "RemoveOversizedGridObjects"),
    "SmallObjectRemover": ("phenotypic.refine", "RemoveSmallObjects"),
    "NearestNeighborMerger": ("phenotypic.refine", "MergeNearestNeighbors"),
    "SmallToLargeMerger": ("phenotypic.refine", "MergeSmallToLarge"),
    "GridAlignmentRefiner": ("phenotypic.refine", "RefineGridAlignment"),
}


def canonical_class_name(class_name: str) -> str:
    """Return the current public name for a serialized class name."""
    alias = _LEGACY_CLASS_ALIASES.get(class_name)
    return alias[1] if alias is not None else class_name
