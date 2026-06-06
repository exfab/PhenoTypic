"""The ``/tune/`` GUI co-pilot — a **read-only** view over a tune output dir.

The tune sub-app never re-optimizes; it only *reads* a finished or in-flight
tuning run's markers, trial journal, and study. This package's import surface is
deliberately small and **optuna-free**: importing :mod:`phenotypic.gui.tune` (and
its :mod:`._run_root` / :mod:`._study_read` helpers) must not import ``optuna``.
A study is read only through the ``StudyStore`` protocol / ``JournalStudyStore``;
the heavier ``create_app`` Dash factory is added later and intentionally *not*
re-exported here yet (an early import would pull Dash into this cheap module).

Exports:
    TuneRunRoot: A validated, described handle on a tune output directory.
    TuneRunRootError: Raised when a directory is not a recognizable tune output.
"""
from __future__ import annotations

from ._run_root import TuneRunRoot, TuneRunRootError

__all__ = ["TuneRunRoot", "TuneRunRootError"]
