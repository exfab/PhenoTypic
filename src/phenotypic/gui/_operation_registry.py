"""Back-compat shim. Implementation in :mod:`phenotypic._services.registry`.

Re-exports the *same* objects — in particular the ``_REGISTRY`` singleton lives
in the promoted module's namespace, so both import paths share one instance.
"""

from __future__ import annotations

from phenotypic._services.registry import (  # noqa: F401
    ColumnRefSpec,
    OperationInfo,
    OperationRegistry,
    ParamInfo,
    get_registry,
)

__all__ = [
    "ColumnRefSpec",
    "OperationInfo",
    "OperationRegistry",
    "ParamInfo",
    "get_registry",
]
