"""The ``/tune/`` GUI co-pilot — a **read-only** view over a tune output dir.

The tune sub-app never re-optimizes; it only *reads* a finished or in-flight
tuning run's markers, trial journal, and study. This package's import surface is
deliberately **optuna-free**: importing :mod:`phenotypic.gui.tune` (and its
:mod:`._run_root` / :mod:`._study_read` / :mod:`._app` helpers) must not import
``optuna``. A study is read only through the ``StudyStore`` protocol /
``JournalStudyStore``; the live :class:`~phenotypic.tune._study.OptunaStudyStore`
is opened lazily inside the Monitor poll callback, never at import / build time.

Exports:
    TuneRunRoot: A validated, described handle on a tune output directory.
    TuneRunRootError: Raised when a directory is not a recognizable tune output.
    create_app: The Dash app factory (empty-state when ``root is None``).

``create_app`` resolves lazily (PEP 562) because it is the only export reaching
``._app``, and therefore ``dash``. Importing any submodule of this package
executes this ``__init__``, so an eager import here dragged the Dash stack into
every consumer of ``._space`` / ``._run_argv`` and failed the gate in
``tests/unit/services/test_import_purity.py``. ``._run_root`` imports only the
stdlib and :mod:`phenotypic.sdk_`, so ``TuneRunRoot`` / ``TuneRunRootError``
stay eager — a smaller change is a smaller regression surface.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._run_root import TuneRunRoot, TuneRunRootError

if TYPE_CHECKING:  # type-checker only; never executed at runtime
    from ._app import create_app  # noqa: F401

__all__ = ["TuneRunRoot", "TuneRunRootError", "create_app"]


def __getattr__(name: str) -> Any:
    if name == "create_app":
        from ._app import create_app

        return create_app
    raise AttributeError(name)
