"""Preload custom operation modules before pipeline deserialization.

The implementation lives in :mod:`phenotypic.sdk_._preload`, because class
resolution in ``_core`` calls it and ``_core`` may not import ``_cli``. This
module keeps the name every CLI caller and test already imports.
"""

from __future__ import annotations

from phenotypic.sdk_._preload import (
    PRELOAD_MODULES_ENV as _PRELOAD_MODULES_ENV,
    preload_custom_operation_modules,
)

__all__ = ["preload_custom_operation_modules", "_PRELOAD_MODULES_ENV"]
