"""Back-compat shim. Implementation in :mod:`phenotypic._services.runs`.

Re-exports the *same* objects, so ``RunRegistry`` is one class no matter which
path a caller imports it through.
"""

from __future__ import annotations

from phenotypic._services.runs import (  # noqa: F401
    RunMode,
    RunRecord,
    RunRegistry,
    RunStatus,
    run_status_is_nonterminal,
)

__all__ = [
    "RunMode",
    "RunStatus",
    "RunRecord",
    "RunRegistry",
    "run_status_is_nonterminal",
]

