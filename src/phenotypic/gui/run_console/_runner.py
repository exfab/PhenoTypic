"""Back-compat shim. Implementation in :mod:`phenotypic._services.runs`.

``LocalRunner`` lives beside ``RunRegistry`` because allocate -> start -> CAS is
one flow; both are re-exported here unchanged.
"""

from __future__ import annotations

from phenotypic._services.runs import (
    LocalRunHandle,
    LocalRunner,
)

__all__ = [
    "LocalRunHandle",
    "LocalRunner",
]
