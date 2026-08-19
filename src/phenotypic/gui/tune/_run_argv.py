"""Back-compat shim. Implementation in :mod:`phenotypic._services.argv`.

Re-exports the *same* functions, so the tune surface and the MCP server build
one spelling of the ``phenotypic.tune run`` command line.
"""

from __future__ import annotations

from phenotypic._services.argv import (
    tune_run_argv,
    tune_run_argv_from_tail,
    tune_run_tail,
)

__all__ = ["tune_run_argv", "tune_run_argv_from_tail", "tune_run_tail"]
