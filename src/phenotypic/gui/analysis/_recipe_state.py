"""Thin wrapper around ``<output>/pipeline.json`` for the analysis sub-app.

The canonical ``pipeline.json`` written by the CLI's
:func:`~phenotypic._cli._cli_output_manager._persist_pipeline_to_output_dir`
captures the entire reproducibility surface (operations, measurements,
post, filters, model). The analysis GUI reads from and writes back to
this file every time the user adds, removes, or re-parameterises a
section.

This module provides:

- :class:`RecipeState`, a dataclass wrapping the in-memory
  :class:`~phenotypic._core._image_pipeline.ImagePipeline` instance plus
  the on-disk ``pipeline.json`` path.
- :meth:`RecipeState.load`, the boot-time loader.
- :meth:`RecipeState.save`, atomic JSON write + mtime refresh.
- mtime-staleness detection mirroring the pattern in
  :mod:`phenotypic.gui.results_viewer._filtered_state`. When the on-disk
  mtime no longer matches what we observed at load time (typical cause:
  a CLI ``--recompile`` ran while the viewer session was open), we
  refuse to clobber the fresh seed and surface a "reload required"
  banner instead.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from phenotypic.gui._config import PIPELINE_JSON

if TYPE_CHECKING:
    from phenotypic._core._image_pipeline import ImagePipeline

logger = logging.getLogger(__name__)


@dataclass
class RecipeState:
    """In-memory + on-disk view of an output dir's ``pipeline.json``.

    Attributes:
        path: Path to the ``pipeline.json`` file under management.
        pipeline: The currently-loaded :class:`ImagePipeline` instance.
            Mutate this reference directly (e.g. ``state.pipeline.set_model(...)``)
            then call :meth:`save` to persist.
        seed_mtime_ns: Nanosecond mtime of :attr:`path` as last observed
            by this instance. ``None`` means the file did not exist when
            the instance was built. Refreshed by :meth:`load` and
            :meth:`save`.
    """

    path: Path
    pipeline: "ImagePipeline"
    seed_mtime_ns: Optional[int] = None
    #: JSON string from the most recent successful :meth:`save`. Callbacks
    #: read this for the ``ANALYSIS_PIPELINE_STORE`` payload instead of
    #: re-serializing the pipeline a second time.
    last_json: str = ""
    _lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False
    )

    @classmethod
    def load(cls, output_dir: Path) -> "RecipeState":
        """Load (or seed) the recipe state for *output_dir*.

        When ``<output>/pipeline.json`` is present it is parsed via
        :meth:`ImagePipeline.from_json`. When absent, an empty
        :class:`ImagePipeline` is used and the file is *not* created
        until the first :meth:`save` — the empty file would just be
        chrome with no information, and creating it eagerly would break
        the "freshly-curated output dir without analysis configured"
        affordance the GUI relies on.

        Args:
            output_dir: Path to a CLI output root (must contain
                ``master_measurements.parquet`` for the broader sub-app
                to function, but :class:`RecipeState` itself only cares
                about the pipeline JSON).

        Returns:
            A :class:`RecipeState` ready for in-place mutation +
            :meth:`save`.
        """
        from phenotypic._core._image_pipeline import ImagePipeline

        pipeline_path = output_dir / PIPELINE_JSON

        if pipeline_path.exists():
            pipeline = ImagePipeline.from_json(pipeline_path)
            mtime = pipeline_path.stat().st_mtime_ns
        else:
            pipeline = ImagePipeline(name=f"analysis-{output_dir.name}")
            mtime = None

        return cls(
            path=pipeline_path,
            pipeline=pipeline,
            seed_mtime_ns=mtime,
        )

    def is_stale(self) -> bool:
        """Return ``True`` when the on-disk file changed since load.

        The CLI seeds ``pipeline.json`` on every aggregate run; if the
        user re-runs the CLI while a viewer session is open, this method
        will start returning ``True`` until they reload via :meth:`load`.
        Callers should refuse to :meth:`save` until the staleness is
        cleared so we don't overwrite a fresh seed with a stale recipe.
        """
        if self.seed_mtime_ns is None:
            # No on-disk file yet — nothing to be stale against.
            return False
        try:
            current = self.path.stat().st_mtime_ns
        except FileNotFoundError:
            # File deleted out from under us; treat as stale.
            return True
        return current != self.seed_mtime_ns

    def save(self) -> bool:
        """Atomically write :attr:`pipeline` to :attr:`path`.

        Caches the serialized JSON on :attr:`last_json` on success so
        callers (e.g. the analysis-page store) can read it without
        re-serializing the pipeline a second time.

        Returns:
            ``True`` when the write succeeded, ``False`` when the file
            was stale (caller must reload before retrying) or the
            atomic rename failed. Failures other than staleness are
            logged at WARNING.
        """
        from phenotypic._cli._cli_output_manager import _atomic_write

        with self._lock:
            if self.is_stale():
                logger.warning(
                    "Refusing to overwrite %s — mtime changed since "
                    "load (likely a CLI --recompile re-run). Reload "
                    "before saving again.",
                    self.path,
                )
                return False

            payload = self.pipeline.to_json() or ""

            def _write(p: str) -> None:
                Path(p).write_text(payload, encoding="utf-8")

            try:
                _atomic_write(self.path, _write)
            except Exception:
                logger.warning(
                    "Atomic write failed for %s", self.path, exc_info=True
                )
                return False

            self.seed_mtime_ns = self.path.stat().st_mtime_ns
            self.last_json = payload
            return True

    def reload(self) -> None:
        """Re-read the on-disk pipeline, replacing :attr:`pipeline`.

        Used after :meth:`is_stale` returns ``True`` to pick up a fresh
        CLI seed before resuming edits.
        """
        with self._lock:
            from phenotypic._core._image_pipeline import ImagePipeline

            if self.path.exists():
                self.pipeline = ImagePipeline.from_json(self.path)
                self.seed_mtime_ns = self.path.stat().st_mtime_ns
            else:
                self.pipeline = ImagePipeline(
                    name=f"analysis-{self.path.parent.name}"
                )
                self.seed_mtime_ns = None
