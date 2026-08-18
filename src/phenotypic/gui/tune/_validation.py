"""Back-compat shim. Implementation in :mod:`phenotypic._services.tune_spec`.

Re-exports the *same* objects, so an ``Issue`` raised by the MCP server and one
raised by the Setup view are the same class. ``_callbacks.py:87`` and
``tests/unit/gui/tune/test_validation.py`` import through here.
"""

from __future__ import annotations

from phenotypic._services.tune_spec import (  # noqa: F401
    Blocks,
    Issue,
    can_deploy,
    preflight_issues,
    spec_path_issue,
    validate_setup,
)

__all__ = [
    "Blocks",
    "Issue",
    "can_deploy",
    "preflight_issues",
    "spec_path_issue",
    "validate_setup",
]
