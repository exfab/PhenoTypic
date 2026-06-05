"""Purpose-group marker ABCs for :class:`ImageEnhancer` subclasses.

Each class is a pure marker (no methods or fields) that labels what an
enhancer's ``_operate`` produces, so an enhancer's purpose is visible at
the type level. They subclass :class:`phenotypic.abc_.ImageEnhancer` and
are re-exported from the top-level :mod:`phenotypic.abc_` namespace.

The pre-existing :class:`phenotypic.abc_.ImageDenoiser` is the seventh
enhancer marker; it lives in ``abc_/_image_denoiser.py`` for back-compat.
"""

from ._focus_edge import FocusEdge
from ._focus_blob import FocusBlob
from ._smoothing import Smoothing
from ._background_subtraction import BackgroundSubtraction
from ._morphological_filtering import MorphologicalFiltering
from ._contrast_adjustment import ContrastAdjustment

__all__ = [
    "FocusEdge",
    "FocusBlob",
    "Smoothing",
    "BackgroundSubtraction",
    "MorphologicalFiltering",
    "ContrastAdjustment",
]
