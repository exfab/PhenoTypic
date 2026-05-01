"""Path → ``Capabilities`` classifier (Phase 1).

Phase 0 placeholder — implementation lands in Phase 1. See ``GUI_SPEC_V1.md``
section 3.

Per plan: stat-only; cheap; first-4KB peek for pipeline-json detection;
``functools.lru_cache``-keyed on ``(path, mtime)``.
"""
from __future__ import annotations

# TODO(Phase 1): ``Capabilities`` dataclass + ``classify(path) -> Capabilities``.

__all__: list[str] = []
