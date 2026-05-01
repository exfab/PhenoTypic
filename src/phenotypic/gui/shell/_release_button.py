"""Reusable per-tool Release button + RSS readout (Phase 3).

Phase 0 placeholder — implementation lands in Phase 3. See ``GUI_SPEC_V1.md``
section 4.

UX honesty (per plan-reviewer feedback): label is "Release loaded data" with
a tooltip explaining that process RSS may stay elevated. We do NOT promise
RSS reduction — Python allocator behaviour means freed objects rarely shrink
RSS. The honest claim is "subsequent viewer access re-loads from disk."
"""
from __future__ import annotations

# TODO(Phase 3): build_release_button(tool_name) -> dash component;
# build_rss_readout() -> dash component using ``psutil``.

__all__: list[str] = []
