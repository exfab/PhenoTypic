"""Shell component IDs (namespaced ``SHELL_*``) (Phase 3).

Phase 0 placeholder — implementation lands in Phase 3.

Per plan: every shell-owned component carries a ``SHELL_`` prefix so the
chrome-vs-existing-id check in Phase 5 can assert the shell namespace does
not collide with the wrapped app's pre-wrap layout.
"""
from __future__ import annotations

# TODO(Phase 3): SHELL_TOPBAR, SHELL_SIDEBAR, SHELL_RSS, SHELL_RELEASE, etc.

__all__: list[str] = []
