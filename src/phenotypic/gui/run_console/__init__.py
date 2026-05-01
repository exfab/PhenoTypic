"""Run console — Local + SLURM submit + Recent Runs panel.

The run console iframes the existing ``_cli/_dashboard/_generator.py`` output
as the canonical "run in progress" view, with one ``postMessage`` upgrade door
for cross-app interactions. Job manager (cancel/retry/persistent history) is
deferred. See ``GUI_SPEC_V1.md`` section 5.

Public API (filled in across phases):
    * :func:`create_app` — Run console Dash factory (Phase 6).
"""
from __future__ import annotations

# Phase 0 scaffolding: lazy re-exports land in Phase 6.
__all__: list[str] = []
