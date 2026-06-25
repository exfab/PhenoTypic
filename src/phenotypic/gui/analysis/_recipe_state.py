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
  a CLI recompile-mode run happened while the viewer session was open), we
  refuse to clobber the fresh seed and surface a "reload required"
  banner instead.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from phenotypic._core._pipeline_parts._serializable_pipeline import (
    PipelineLoadWarning,
)
from phenotypic.sdk_ import (
    DIR_DELIVERABLES,
    pipeline_json_path,
    resolve_pipeline_config_path,
)

if TYPE_CHECKING:
    from phenotypic._core._image_pipeline import ImagePipeline
    from phenotypic.sdk_ import BundleLayout

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
    source_path: Optional[Path] = None
    #: JSON string from the most recent successful :meth:`save`. Callbacks
    #: read this for the ``ANALYSIS_PIPELINE_STORE`` payload instead of
    #: re-serializing the pipeline a second time.
    last_json: str = ""
    #: Filter / model entries that the on-disk JSON referenced but whose
    #: class could not be resolved in the live ``phenotypic`` namespace
    #: (typical cause: an analyzer was renamed or removed since the
    #: pipeline was saved). The analysis page renders a banner listing
    #: these so the user can manually re-add a replacement; the file on
    #: disk is left untouched until the next user-driven save.
    load_warnings: List[PipelineLoadWarning] = field(default_factory=list)
    #: Resolved bundle topology this recipe was built from, when constructed
    #: via :meth:`from_layout`. ``None`` for the legacy ``output_dir``-rooted
    #: :meth:`load` path. When set, :meth:`reload` re-resolves through it so a
    #: standalone bundle (``root`` IS the deliverables folder) never
    #: double-joins ``deliverables/`` on a staleness refresh.
    layout: Optional["BundleLayout"] = None
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
        pipeline_path = pipeline_json_path(output_dir)
        read_path = resolve_pipeline_config_path(output_dir)
        return cls._load_from_paths(
            read_path, pipeline_path, name_hint=output_dir.name
        )

    @classmethod
    def from_layout(cls, layout: "BundleLayout") -> "RecipeState":
        """Load (or seed) the recipe state from a resolved :class:`BundleLayout`.

        The :class:`BundleLayout`-aware sibling of :meth:`load`, mirroring
        :meth:`phenotypic.sdk_._qc_recipe._recipe.QcRecipe.from_layout`. Anchors
        the pipeline config on ``layout.deliverables_base`` directly (with the
        same legacy plain-``pipeline.json`` fallback ``resolve_pipeline_config_path``
        provides), so a standalone deliverables bundle — whose
        ``layout.output_root is None`` and whose viewer ``root`` is already the
        deliverables folder — resolves ``pipeline.json`` *inside the bundle*
        rather than via ``pipeline_json_path(output_root)``, which would
        double-join ``deliverables/``.

        The resolved ``layout`` is retained on :attr:`layout` so
        :meth:`reload` re-resolves through it instead of the ``output_dir``
        heuristic.

        Args:
            layout: Resolved bundle topology.

        Returns:
            A :class:`RecipeState` ready for in-place mutation + :meth:`save`.
        """
        from phenotypic.sdk_._io_constants import _LEGACY_PIPELINE_JSON

        pipeline_path = layout.pipeline_config_path
        legacy = layout.deliverables_base / _LEGACY_PIPELINE_JSON
        if pipeline_path.exists():
            read_path = pipeline_path
        elif legacy.exists():
            read_path = legacy
        else:
            read_path = pipeline_path
        name_root = (
            layout.output_root
            if layout.output_root is not None
            else layout.deliverables_base
        )
        return cls._load_from_paths(
            read_path, pipeline_path, name_hint=name_root.name, layout=layout
        )

    @classmethod
    def _load_from_paths(
        cls,
        read_path: Path,
        pipeline_path: Path,
        *,
        name_hint: str,
        layout: "BundleLayout | None" = None,
    ) -> "RecipeState":
        """Build a recipe from explicit read + canonical-write paths.

        Shared core of :meth:`load` and :meth:`from_layout`. ``read_path`` is the
        existing config to parse (canonical typed or legacy ``.json``);
        ``pipeline_path`` is the canonical typed path future writes target.
        ``name_hint`` seeds the empty-pipeline name when no config exists.
        """
        from phenotypic._core._image_pipeline import ImagePipeline

        load_warnings: List[PipelineLoadWarning] = []

        if read_path.exists():
            pipeline = ImagePipeline.from_json(
                read_path,
                skip_unknown_analyzers=True,
                load_warnings=load_warnings,
            )
            try:
                mtime = read_path.stat().st_mtime_ns
            except OSError:
                mtime = None
            if load_warnings:
                logger.warning(
                    "%s referenced %d unknown analyzer class(es); the "
                    "analysis page will render a banner. Skipped: %s",
                    read_path,
                    len(load_warnings),
                    ", ".join(w.class_name for w in load_warnings),
                )
        else:
            pipeline = ImagePipeline(name=f"analysis-{name_hint}")
            mtime = None

        return cls(
            path=pipeline_path,
            pipeline=pipeline,
            seed_mtime_ns=mtime,
            source_path=read_path if read_path != pipeline_path else None,
            load_warnings=load_warnings,
            layout=layout,
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
        tracked_path = self._tracked_path()
        try:
            current = tracked_path.stat().st_mtime_ns
        except FileNotFoundError:
            # File deleted out from under us; treat as stale.
            return True
        return current != self.seed_mtime_ns

    def _tracked_path(self) -> Path:
        """Return the path whose mtime should be compared against the seed."""
        if self.path.exists() or self.source_path is None:
            return self.path
        return self.source_path

    def _output_root(self) -> Path:
        """Return the output root that owns :attr:`path`."""
        if self.path.parent.name == DIR_DELIVERABLES:
            return self.path.parent.parent
        return self.path.parent

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
                    "load (likely a CLI recompile-mode run). Reload "
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
            self.source_path = None
            self.last_json = payload
            return True

    def reload(self) -> None:
        """Re-read the on-disk pipeline, replacing :attr:`pipeline`.

        Used after :meth:`is_stale` returns ``True`` to pick up a fresh
        CLI seed before resuming edits. When this state was built via
        :meth:`from_layout`, re-resolve through the retained layout so a
        standalone bundle never double-joins ``deliverables/`` here.
        """
        with self._lock:
            if self.layout is not None:
                fresh = type(self).from_layout(self.layout)
            else:
                fresh = type(self).load(self._output_root())
            self.pipeline = fresh.pipeline
            self.seed_mtime_ns = fresh.seed_mtime_ns
            self.source_path = fresh.source_path
            self.load_warnings = fresh.load_warnings
