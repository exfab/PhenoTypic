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
"""
from __future__ import annotations

from ._app import create_app
from ._run_root import TuneRunRoot, TuneRunRootError

__all__ = ["TuneRunRoot", "TuneRunRootError", "create_app"]
