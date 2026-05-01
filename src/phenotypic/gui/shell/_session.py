"""``ToolSession`` lifecycle primitive (Phase 1).

Phase 0 placeholder — implementation lands in Phase 1. See ``GUI_SPEC_V1.md``
section 4 (Shell chrome + lifecycle).

Per plan: lazy-init ``get`` / ``touch`` / ``idle_seconds`` / ``release`` with
threading lock and monotonic-time ``_last_access`` field. Idle release uses a
single daemon thread polling ``time.monotonic()`` (NOT ``threading.Timer``,
which races with touch resets).
"""
from __future__ import annotations

# TODO(Phase 1): ``ToolSession`` class + ``start_idle_release_thread`` daemon.

__all__: list[str] = []
