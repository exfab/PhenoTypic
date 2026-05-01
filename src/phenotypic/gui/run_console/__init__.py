"""Run console — Local + SLURM submit + Recent Runs panel.

The run console iframes the existing ``_cli/_dashboard/_generator.py`` output
as the canonical "run in progress" view, with one ``postMessage`` upgrade door
for cross-app interactions. Job manager (cancel/retry/persistent history) is
deferred. See ``GUI_SPEC_V1.md`` section 5.

Public API:
    * :func:`create_app` — Run console Dash factory. Phase 5 ships a
      placeholder layout under the right URL prefix; Phase 6 fills it
      in with the pipeline form, log tail, and Recent Runs panel.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from phenotypic.gui.run_console._app import create_app as _create_app  # noqa: F401

__all__ = ["create_app"]


def __getattr__(name: str) -> Any:
    if name == "create_app":
        from phenotypic.gui.run_console._app import create_app

        return create_app
    raise AttributeError(name)
