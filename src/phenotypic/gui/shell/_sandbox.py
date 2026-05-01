"""``SandboxRoot`` dataclass + safe-resolve helpers (Phase 1).

Phase 0 placeholder — implementation lands in Phase 1. See ``GUI_SPEC_V1.md``
section 3 (Sandbox & file browser).

TODO(cloud-deploy): when ``--mode=cloud`` ships, wire an auth gate via a Flask
``@before_request`` hook on every ``/sandbox/api/*`` and ``/runs/*`` route. The
sandbox is currently frozen-at-launch (single-user); cloud mode must make the
root selectable per session.
"""
from __future__ import annotations

# TODO(Phase 1): ``SandboxRoot`` dataclass with ``resolve``/``contains``/
# ``list_children`` methods. ``resolve`` MUST raise ``ValueError`` for paths
# outside ``self.root`` and for symlinks pointing outside.

__all__: list[str] = []
