"""Staged startup feedback for the ``phenotypic-gui`` launcher.

The console-script entry point forces the entire (heavy) ``phenotypic`` import
chain to run *before* ``main()`` is ever called, so a live progress bar cannot
animate that ~1.7s import — it is already spent by the time launcher code runs.
What this reporter *can* do is:

    * report the measured core-import duration retroactively as the first
      completed step (the launcher passes the elapsed time computed from
      :data:`phenotypic._startup_perf.IMPORT_STARTED_AT`),
    * show a live :mod:`rich` progress bar over the phases ``main()`` *does*
      control — sandbox resolution and hub composition (whose six sub-apps
      tick the bar via :meth:`StartupReporter.detail`),
    * fall back to plain :mod:`logging` lines when stdout is not a TTY (SSH
      pipes, log files) or when ``--debug`` is set.

This module imports only the standard library at module scope (``rich`` is
imported lazily inside :meth:`StartupReporter.__enter__`) so it stays cheap to
import and never adds to the very startup cost it is reporting on.
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from typing import Any, Generator, Literal, Optional

__all__ = ["StartupReporter", "should_use_rich"]

logger = logging.getLogger("phenotypic.gui.startup")

#: Opt-out env var: set to any non-empty value to force plain logging output
#: even on an interactive terminal (handy for CI captures and screen scrapers).
NO_PROGRESS_ENV = "PHENOTYPIC_GUI_NO_PROGRESS"


def should_use_rich(*, debug: bool, stream: Any = None) -> bool:
    """Decide whether to render the rich progress bar or plain log lines.

    Rich output is used only for an interactive terminal session. It is
    suppressed when ``--debug`` is set (Dash auto-reload + verbose logging
    would fight the live display), when :data:`NO_PROGRESS_ENV` is set, or
    when the output stream is not a TTY (SSH pipe, log file, captured run).

    Args:
        debug: The launcher's ``--debug`` flag.
        stream: Stream to probe for TTY-ness. Defaults to ``sys.stderr``.

    Returns:
        ``True`` to use the rich progress bar, ``False`` for plain logging.
    """
    if debug or os.environ.get(NO_PROGRESS_ENV):
        return False
    stream = stream if stream is not None else sys.stderr
    try:
        return bool(stream.isatty())
    except Exception:  # pragma: no cover - defensive: exotic stream objects
        return False


class StartupReporter:
    """Render staged launcher progress as a rich bar or plain log lines.

    Use as a context manager wrapping the boot sequence; call
    :meth:`record_done` for an already-finished phase (the core import),
    :meth:`stage` (a context manager) around each timed phase, and
    :meth:`detail` to annotate the in-flight phase with a sub-step label.

    Args:
        total_steps: Number of bar segments — one per :meth:`record_done`
            plus one per :meth:`stage`.
        use_rich: When ``True``, render a live :mod:`rich` progress bar;
            otherwise emit plain :mod:`logging` lines.
        import_elapsed: Measured core-library import duration (seconds),
            surfaced verbatim by the launcher's first :meth:`record_done`.
    """

    def __init__(
        self,
        *,
        total_steps: int,
        use_rich: bool,
        import_elapsed: Optional[float] = None,
    ) -> None:
        self.total_steps = total_steps
        self.use_rich = use_rich
        self.import_elapsed = import_elapsed
        self._progress: Any = None
        self._task: Any = None
        self._base_desc: str = ""
        self._saved_log_level: Optional[int] = None

    # -- context management -------------------------------------------------

    def __enter__(self) -> "StartupReporter":
        if self.use_rich:
            try:
                from rich.progress import (
                    BarColumn,
                    MofNCompleteColumn,
                    Progress,
                    SpinnerColumn,
                    TextColumn,
                    TimeElapsedColumn,
                )

                self._progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                )
                self._progress.__enter__()
                self._task = self._progress.add_task(
                    "Starting PhenoTypic GUI…", total=self.total_steps
                )
                # Silence INFO logs (e.g. compose_hub's summary) while the
                # live bar owns the terminal — they would corrupt the render.
                root = logging.getLogger()
                self._saved_log_level = root.level
                root.setLevel(logging.WARNING)
            except Exception:  # pragma: no cover - rich missing/odd terminal
                self._progress = None
                self.use_rich = False
        return self

    def __exit__(self, *exc_info: Any) -> Literal[False]:
        if self._progress is not None:
            try:
                self._progress.__exit__(*exc_info)
            finally:
                self._progress = None
                if self._saved_log_level is not None:
                    logging.getLogger().setLevel(self._saved_log_level)
                    self._saved_log_level = None
        return False  # never suppress exceptions

    # -- step reporting -----------------------------------------------------

    def record_done(self, label: str, elapsed: Optional[float]) -> None:
        """Mark an already-completed phase (advances the bar by one)."""
        suffix = f" ({elapsed:.2f}s)" if elapsed is not None else ""
        if self._progress is not None:
            self._progress.update(
                self._task, advance=1, description=f"{label}{suffix}"
            )
        else:
            logger.info("✓ %s%s", label, suffix)

    @contextmanager
    def stage(self, label: str) -> Generator["StartupReporter", None, None]:
        """Time a phase, updating the bar/log on entry, exit, and failure."""
        import time

        self._base_desc = label
        if self._progress is not None:
            self._progress.update(self._task, description=f"{label}…")
        else:
            logger.info("→ %s…", label)
        start = time.perf_counter()
        try:
            yield self
        except BaseException:
            if self._progress is None:
                logger.error("✗ %s failed", label)
            raise
        else:
            elapsed = time.perf_counter() - start
            if self._progress is not None:
                self._progress.update(
                    self._task, advance=1, description=f"{label} ({elapsed:.2f}s)"
                )
            else:
                logger.info("✓ %s (%.2fs)", label, elapsed)

    def detail(self, text: str) -> None:
        """Annotate the in-flight :meth:`stage` with a sub-step label."""
        if self._progress is not None:
            self._progress.update(
                self._task, description=f"{self._base_desc} · {text}…"
            )
        else:
            logger.debug("  · %s", text)
