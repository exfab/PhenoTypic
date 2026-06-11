"""``TuneRunRoot`` — validate + describe a tune output directory (read-only).

The single seam the ``/tune/`` GUI uses to turn an output path into a typed,
validated handle: where the trial journal is, which study (URL + name) backs it,
whether the run was multi-objective, and where the calibration images / tuned
winner live. :meth:`TuneRunRoot.discover` reads three markers in **precedence
order** so a live run (only ``run.json`` written) and a finished run (full
``deliverables/``) both resolve, and a non-tune directory is rejected:

1. ``.pht-tune-cache/run.json`` — the run-START marker (Chunk 0). It carries the
   resolved (non-null) ``storage_url``, ``study_name``, ``images_dir``, and the
   ``is_multi_objective`` flag, so it is the cheapest and most authoritative
   source; it wins outright when present.
2. ``deliverables/tuning_spec.json`` — the resolved recipe. The storage URL comes
   from ``strategy.storage_url`` (only an ``OptunaConfig`` strategy carries one)
   and the objective ``directions`` from the scorer.
3. The legacy output root — recognized solely by a ``trials.parquet``.

**Optuna-free.** This module reads only the markers (JSON) and the spec model; it
never imports ``optuna`` or any engine/run-loop code (the GUI never re-optimizes).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from phenotypic.tools_ import (
    best_pipeline_path,
    resolve_tuning_spec_path,
    trials_parquet_path,
    tune_cache_run_marker_path,
)

#: The study name every tune run uses (mirrors ``_tune_cli._run._STUDY_NAME``;
#: kept in lockstep by ``test_study_name_cutover.py``). Used as the fallback when
#: discovering from a ``tuning_spec.json`` (which, unlike the ``run.json`` marker,
#: does not record a study name).
_DEFAULT_STUDY_NAME: str = "tune_cost_v1"

#: The placeholder multi-objective ``directions`` synthesized from the
#: ``run.json`` ``is_multi_objective`` flag. The marker records only the boolean
#: (not the per-axis names), so a multi-objective run is represented as a 2-axis
#: minimize vector — enough for ``is_multi_objective(root)`` (len > 1) and the
#: GUI's "this is a Pareto run" branch. Every tuning objective is a cost
#: (lower-is-better — cost convention), so the synthesized axes are both
#: ``"minimize"``.
_MULTI_OBJECTIVE_PLACEHOLDER_DIRECTIONS: list[str] = ["minimize", "minimize"]


class TuneRunRootError(ValueError):
    """A directory is not a recognizable tune output.

    Raised by :meth:`TuneRunRoot.discover` when none of the three discovery
    markers resolves: no ``run.json``, no study URL from a ``tuning_spec.json``,
    and no ``trials.parquet``.
    """


@dataclass(frozen=True)
class TuneRunRoot:
    """A validated handle on a tune output directory.

    Args:
        path: The tune output directory (the run's ``--output`` root).
        trials_path: The trial journal ``trials.parquet``, or ``None`` when it
            has not been written yet (a live run discovered via ``run.json``).
        storage_url: The resolved Optuna storage URL, or ``None`` when the run is
            parquet-journal-only.
        study_name: The study name (``"tune_cost_v1"``).
        directions: The per-objective Optuna ``directions`` (length ≥ 2) for a
            multi-objective run, or ``None`` for a single-objective study.
        images_dir: The calibration image directory, or ``None`` when unknown
            (e.g. discovered from a ``tuning_spec.json`` or a legacy root).
        best_pipeline_path: Where the tuned winner ``best_pipeline.json`` lives
            (a pure path expression; the file may not exist yet).
    """

    path: Path
    trials_path: Path | None
    storage_url: str | None
    study_name: str
    directions: list[str] | None
    images_dir: Path | None
    best_pipeline_path: Path

    @classmethod
    def discover(cls, path: Path) -> "TuneRunRoot":
        """Validate ``path`` as a tune output and describe it.

        Reads the three discovery markers in precedence order (``run.json`` →
        ``tuning_spec.json`` → legacy ``trials.parquet`` root). Locates the trial
        journal via :func:`trials_parquet_path` and the tuned winner via
        :func:`best_pipeline_path` regardless of which marker matched.

        Args:
            path: The candidate tune output directory.

        Returns:
            The validated :class:`TuneRunRoot`.

        Raises:
            TuneRunRootError: When ``path`` carries none of {``run.json``, a study
                URL from a ``tuning_spec.json``, a ``trials.parquet``}.
        """
        path = Path(path)
        trials = trials_parquet_path(path)
        trials_path = trials if trials.exists() else None

        marker = cls._read_run_marker(path)
        if marker is not None:
            storage_url, study_name, directions, images_dir = marker
            return cls(
                path=path,
                trials_path=trials_path,
                storage_url=storage_url,
                study_name=study_name,
                directions=directions,
                images_dir=images_dir,
                best_pipeline_path=best_pipeline_path(path),
            )

        spec = cls._read_spec_markers(path)
        if spec is not None:
            storage_url, directions = spec
            # A spec only validates the directory when it yields a real study URL
            # or a trial journal exists — a spec with neither (no URL, no parquet)
            # is not yet a readable tune output, so fall through to the error.
            if storage_url is not None or trials_path is not None:
                return cls(
                    path=path,
                    trials_path=trials_path,
                    storage_url=storage_url,
                    study_name=_DEFAULT_STUDY_NAME,
                    directions=directions,
                    images_dir=None,
                    best_pipeline_path=best_pipeline_path(path),
                )

        if trials_path is not None:
            # Legacy root: recognized solely by a trial journal (no marker, no
            # study). Parquet-only, single-objective view.
            return cls(
                path=path,
                trials_path=trials_path,
                storage_url=None,
                study_name=_DEFAULT_STUDY_NAME,
                directions=None,
                images_dir=None,
                best_pipeline_path=best_pipeline_path(path),
            )

        raise TuneRunRootError(
            f"{path} is not a tune output: no .pht-tune-cache/run.json, no study "
            "URL in deliverables/tuning_spec.json, and no trials.parquet."
        )

    @staticmethod
    def _read_run_marker(
        path: Path,
    ) -> tuple[str | None, str, list[str] | None, Path | None] | None:
        """Parse ``.pht-tune-cache/run.json`` → (url, study, directions, images).

        The marker is the run-START sidecar (Chunk 0). It carries the resolved
        ``storage_url``, ``study_name``, ``images_dir``, and the
        ``is_multi_objective`` flag — directions are synthesized from that flag
        (the marker does not record per-axis names; see
        :data:`_MULTI_OBJECTIVE_PLACEHOLDER_DIRECTIONS`). A missing flag leaves
        ``directions`` ``None``.

        Args:
            path: The tune output directory.

        Returns:
            ``(storage_url, study_name, directions, images_dir)`` when the marker
            exists and parses, else ``None``.
        """
        marker_path = tune_cache_run_marker_path(path)
        if not marker_path.exists():
            return None
        marker = json.loads(marker_path.read_text())
        storage_url = marker.get("storage_url")
        study_name = marker.get("study_name") or _DEFAULT_STUDY_NAME
        is_multi = marker.get("is_multi_objective")
        directions = (
            list(_MULTI_OBJECTIVE_PLACEHOLDER_DIRECTIONS) if is_multi else None
        )
        images_raw = marker.get("images_dir")
        images_dir = Path(images_raw) if images_raw else None
        return storage_url, study_name, directions, images_dir

    @staticmethod
    def _read_spec_markers(path: Path) -> tuple[str | None, list[str] | None] | None:
        """Parse ``deliverables/tuning_spec.json`` → (storage_url, directions).

        The fallback when no ``run.json`` exists. Loads the resolved
        :class:`~phenotypic.tune._spec.TuningSpec`, reads the storage URL via
        ``getattr(spec.strategy, "storage_url", None)`` (only ``OptunaConfig``
        carries the field), and the objective ``directions`` via the private
        :func:`phenotypic.tune._multi_objective.objective_directions` (``None``
        for a single-objective scorer). Returns ``None`` when no spec file exists.

        Args:
            path: The tune output directory.

        Returns:
            ``(storage_url, directions)`` when the spec exists, else ``None``.
        """
        spec_path = resolve_tuning_spec_path(path)
        if not spec_path.exists():
            return None
        # Lazy, function-local imports keep module import optuna-free and cheap:
        # the spec model + the private multi-objective helper are only pulled in
        # when a spec actually needs decoding.
        from phenotypic.tune._multi_objective import objective_directions
        from phenotypic.tune._spec import TuningSpec

        spec = TuningSpec.model_validate_json(spec_path.read_text())
        storage_url = getattr(spec.strategy, "storage_url", None)
        directions = objective_directions(spec.scorer)
        return storage_url, directions
