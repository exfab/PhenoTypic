"""Provides the base classes for set analysis operations.

This module imports and exposes the `SetAnalyzer` class from the internal
`_set_analyzer` module. The `SetAnalyzer` class is designed for performing
various analytical operations on sets. This is a utility module intended for
use in applications that require set analysis functionality. Only the
`SetAnalyzer` class is explicitly exposed as part of this module's public API.

"""

from ._set_analyzer import SetAnalyzer
from ._model_fitter import ModelFitter

__all__ = [
    "SetAnalyzer",
    "ModelFitter",
]