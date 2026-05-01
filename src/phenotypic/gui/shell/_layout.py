"""Shell chrome layout helpers (Phase 3).

Phase 0 placeholder — implementation lands in Phase 3. See ``GUI_SPEC_V1.md``
section 4.

Per plan:
    * :func:`build_top_bar` — title, root display, tab nav, RSS readout, help
      modal trigger.
    * :func:`build_sidebar` — sandboxed file tree with capability badges.
    * :func:`wrap_layout_in_chrome` (a.k.a. ``wrap_in_chrome``) — mutates
      ``app.layout`` AND registers chrome callbacks (RSS interval, sidebar
      refresh, release-button click) on the specific app instance.
"""
from __future__ import annotations

# TODO(Phase 3): build_top_bar, build_sidebar, wrap_layout_in_chrome.

__all__: list[str] = []
