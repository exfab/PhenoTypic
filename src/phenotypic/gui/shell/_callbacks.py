"""Shell-side Dash callbacks (Phase 3).

Phase 0 placeholder — implementation lands in Phase 3. See ``GUI_SPEC_V1.md``
section 4.

Per plan, ``register_chrome_callbacks(app, sandbox, sessions)`` is invoked by
``wrap_in_chrome`` per app instance and wires:
    * RSS-readout ``dcc.Interval`` tick.
    * Sidebar refresh button + tree-expand callbacks.
    * Release-button click → ``ToolSession.release()`` for the active tab.
    * Hidden/symlink toggle stores.
"""
from __future__ import annotations

# TODO(Phase 3): register_chrome_callbacks(app, ...).

__all__: list[str] = []
