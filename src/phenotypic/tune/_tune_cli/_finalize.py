"""Re-runnable finalize for a study whose search ran somewhere else.

``run_tuning`` with ``--slurm`` is fire-and-forget: it pre-creates the shared
study, submits the worker array, and **returns**. Every ``deliverables/`` write
lives below that early return, so a SLURM-launched study finishes with a full
Optuna store and an output directory that has none of the artifacts a reader
expects. Most damagingly ``best_params.json`` is never written, and
``prepare_best_from_run`` hard-requires it — so the plain export path raises
``FileNotFoundError`` on **every** distributed study.

:func:`finalize_distributed_study` is the entry point that closes that gap. It
opens the finished study, re-loads the calibration plates the marker recorded,
and runs the same four finalize steps ``run_tuning`` runs locally, **in the same
order**, because the order is load-bearing:

1. :func:`~._run._finalize_outputs` — ``trials.parquet``,
   ``param_importance.json``, ``best_pipeline.json``;
2. :func:`~._run._finalize_pareto_outputs` — the Pareto front, per-axis winners,
   and it **overwrites** ``best_pipeline.json`` with the knee;
3. :func:`~._run._finalize_best_params` — ``best_params.json``, **last, and
   deliberately**: it is the de-facto completion marker, so writing it earlier
   would leave an interrupted finalize looking exportable when it is not;
4. :func:`~._run._finalize_generalization` — ``generalization.json``.

The four steps are imported, not moved. They still have their call sites in
``_run.run_tuning`` and ``tests/unit/tune/test_run_tuning_pareto.py``, and one
definition run by both paths is the only way the local and distributed outputs
stay identical.

Two hazards the ordering alone does not close, both **reported rather than
hidden**:

* A kill *inside* step 2 leaves ``best_pipeline.json`` holding step 1's scalar
  best, which a later export would mislabel ``pareto_knee``. So a
  ``finalize_in_progress`` sentinel is written before step 1 and cleared after
  step 4, and a finalize that finds one refuses with ``finalize_incomplete``.
* Finalize is not safe against a *still-running* study: two concurrent calls
  each compute a different ``_headline_winner`` as trials land and overwrite
  each other's deliverables. So the study must be terminal, and a live one is
  refused with ``study_not_finished``.

``force=True`` overrides both refusals, for the operator who knows the fleet is
dead and wants the artifacts anyway.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from phenotypic.sdk_ import _io_constants as io

from .._multi_objective import objective_directions
from .._spec import TuningSpec
from .._study._protocol import StudyStore
from ._run import (
    _finalize_best_params,
    _finalize_generalization,
    _finalize_outputs,
    _finalize_pareto_outputs,
    _headline_winner,
    _load_images,
    _open_store,
    _pipeline_for_trial,
    _resolve_calibration_images,
    _selection_label,
)

__all__ = ["FinalizeResult", "finalize_distributed_study"]


@dataclass(frozen=True)
class FinalizeResult:
    """What one finalize produced, including what it deliberately did not.

    Attributes:
        output_dir: The finalized run directory.
        n_trials: How many trials the study held when it was read.
        winner_trial_number: The headline winner's trial number, or ``None``
            when the study recorded no successful trial.
        selection: ``"pareto_knee"`` for a multi-objective run, else
            ``"single_best"`` — the label stamped into ``best_params.json``.
        best_params_written: Whether ``best_params.json`` landed. ``False``
            means the study has no winner; ``_finalize_best_params`` no-ops
            silently in that case, and the failure would otherwise resurface
            much later as a misleading ``FileNotFoundError`` from
            ``prepare_best_from_run``.
        pareto_published: Whether ``deliverables/pareto/`` was written (a
            single-objective run's empty front makes step 2 a no-op).
        generalization_written: Whether ``generalization.json`` landed. ``False``
            with a non-null winner means the calibration plates could not be
            re-loaded, so the held-out ``gap`` stays unknown for this study.
        warnings: Human-readable notes for everything above that is ``False``
            for a reason worth surfacing.
    """

    output_dir: Path
    n_trials: int
    winner_trial_number: Optional[int]
    selection: str
    best_params_written: bool
    pareto_published: bool
    generalization_written: bool
    warnings: tuple[str, ...] = ()


def finalize_distributed_study(
    output_dir: Path, *, force: bool = False
) -> FinalizeResult:
    """Write the ``deliverables/`` a ``--slurm`` run's early return skipped.

    Safe to run again: every step overwrites its own output atomically, so a
    second call on an already-finalized directory simply reproduces it.

    Args:
        output_dir: The tune run directory (the ``-o`` of the original run).
        force: Finalize even when the study cannot be shown to be terminal, and
            even when a previous finalize left its sentinel behind. For the
            operator who knows the fleet is gone; the concurrency hazard the
            gate exists for is real, so this is not a default.

    Returns:
        A :class:`FinalizeResult` describing what was written.

    Raises:
        FileNotFoundError: When ``output_dir`` carries no ``run.json`` marker or
            no resolved ``tuning_spec.json`` — without both there is no study to
            open and no spec to rebuild pipelines from.
        RuntimeError: ``finalize_incomplete`` when a previous finalize was
            interrupted, or ``study_not_finished`` when the study is still
            accepting trials. Both are cleared by ``force=True``.
    """
    output_dir = Path(output_dir)
    marker = _read_run_marker(output_dir)
    spec = _read_resolved_spec(output_dir)

    sentinel = io.tune_finalize_marker_path(output_dir)
    if sentinel.exists() and not force:
        raise RuntimeError(
            f"finalize_incomplete: {sentinel} is left over from a finalize that "
            "did not run to completion, so best_pipeline.json may hold the "
            "scalar best under a name an export reads as the Pareto knee. "
            "Re-run with force=True to finalize from scratch."
        )

    store = _open_finished_store(spec, output_dir, marker)
    trials = list(store.trials)
    _require_terminal_study(marker, n_seen=len(trials), force=force)

    warnings: list[str] = []
    headline = _headline_winner(store)
    selection = _selection_label(store)

    # The sentinel is written BEFORE step 1 and removed only after step 4
    # returns normally. There is deliberately no try/finally: an interrupted
    # finalize must LEAVE it behind — that is the whole signal.
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("")
    # 1 — trials.parquet + param_importance.json + best_pipeline.json.
    _finalize_outputs(
        store,
        io.trials_parquet_path(output_dir),
        output_dir,
        _pipeline_for_trial(spec, headline),
    )
    # 2 — the Pareto front, and the best_pipeline.json OVERWRITE with the knee.
    _finalize_pareto_outputs(store, spec, output_dir)
    # 3 — best_params.json, last: it is the de-facto completion marker.
    _finalize_best_params(headline, output_dir, selection=selection)
    # 4 — the report-only held-out verdict.
    generalization_written = _finalize_generalization_from_disk(
        headline, spec, output_dir, marker, warnings
    )
    sentinel.unlink(missing_ok=True)

    best_params_written = io.best_params_path(output_dir).is_file()
    if headline is None:
        warnings.append(
            "the study recorded no successful trial, so best_params.json was "
            "not written; prepare_best_from_run will still raise "
            "FileNotFoundError for this directory."
        )
    return FinalizeResult(
        output_dir=output_dir,
        n_trials=len(trials),
        winner_trial_number=None if headline is None else headline.number,
        selection=selection,
        best_params_written=best_params_written,
        pareto_published=io.pareto_dir(output_dir).is_dir(),
        generalization_written=generalization_written,
        warnings=tuple(warnings),
    )


def _read_run_marker(output_dir: Path) -> dict[str, Any]:
    """Load ``.pht-tune-cache/run.json`` — the only record of how the run ran."""
    marker_path = io.tune_cache_run_marker_path(output_dir)
    if not marker_path.is_file():
        raise FileNotFoundError(
            f"{output_dir} carries no {marker_path.name} marker, so the study "
            "URL, the study name and the calibration image directory are all "
            "unknown. Only a run started by this version of the tune CLI can "
            "be finalized."
        )
    payload = json.loads(marker_path.read_text())
    if not isinstance(payload, dict):
        raise FileNotFoundError(f"{marker_path} is not a run marker object")
    return payload


def _read_resolved_spec(output_dir: Path) -> TuningSpec:
    """Load the RESOLVED spec the workers ran (not the user's input file)."""
    spec_path = io.resolve_tuning_spec_path(output_dir)
    if not spec_path.is_file():
        raise FileNotFoundError(
            f"{output_dir} carries no resolved tuning spec at {spec_path}; "
            "without it the winning parameters cannot be rebuilt into a "
            "pipeline."
        )
    return TuningSpec.model_validate_json(spec_path.read_text())


def _open_finished_store(
    spec: TuningSpec, output_dir: Path, marker: dict[str, Any]
) -> StudyStore:
    """Reopen the study the fleet wrote into.

    The marker's ``storage_url`` is the URL the workers actually opened — the
    one place the env-var and spec fallbacks were already collapsed — so it is
    preferred over re-deriving one here, where ``$PHENOTYPIC_TUNE_STORAGE_URL``
    may hold something different from what the run used.
    """
    return _open_store(
        spec.strategy,
        output_dir,
        storage_url=marker.get("storage_url"),
        resume_path=io.trials_parquet_path(output_dir),
        directions=objective_directions(spec.scorer),
    )


def _require_terminal_study(
    marker: dict[str, Any], *, n_seen: int, force: bool
) -> None:
    """Refuse to finalize a study that may still be gaining trials.

    Two concurrent finalizes on a live study each pick a different headline
    winner as trials land, and each overwrites the other's ``best_pipeline.json``
    / ``best_params.json`` — so the directory ends up describing a trial that
    was never the winner. The budget recorded in the marker is the available
    terminal signal: once the fleet has drained it, no worker asks for more.

    Args:
        marker: The parsed run marker.
        n_seen: Trials present in the store right now.
        force: Skip the check.

    Raises:
        RuntimeError: When the study is not demonstrably finished.
    """
    if force:
        return
    budget = marker.get("n_trials")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        raise RuntimeError(
            "study_not_finished: the run marker records no trial budget, so "
            "this study cannot be shown to have finished and may still be "
            "running. Re-run with force=True once the fleet is done."
        )
    if n_seen < budget:
        raise RuntimeError(
            f"study_not_finished: the study holds {n_seen} of {budget} trials, "
            "so the fleet is still running. Finalizing now would race a live "
            "worker and publish a winner that later trials supersede. Re-run "
            "with force=True to finalize a fleet you know is dead."
        )


def _finalize_generalization_from_disk(
    headline: Any,
    spec: TuningSpec,
    output_dir: Path,
    marker: dict[str, Any],
    warnings: list[str],
) -> bool:
    """Step 4, with the calibration plates re-loaded from ``images_dir``.

    The local path already holds the loaded ``GridImage`` objects when it
    reaches this step; a finalize started in a fresh process holds none of
    them, and the report needs both the held-out plates and the full loaded set
    (for the dataset-identity comparison that flags a changed dataset). Dropping
    the step instead would leave ``generalization.json`` unwritten and the
    held-out ``gap`` permanently null for every distributed study — which is
    exactly the signal that catches an arm that won by overfitting.

    The scan reproduces the run's fixed ``nrows``/``ncols`` (marker v2). Without
    them a ``Grid_RowNum``/``Grid_ColNum``-grouped scorer would score the
    held-out pass on a different grid than the search used.

    Returns:
        Whether ``generalization.json`` was written.
    """
    if headline is None:
        return False  # no winner → no verdict; already reported by the caller
    images_raw = marker.get("images_dir")
    if not images_raw:
        warnings.append(
            "the run marker records no images_dir, so the held-out "
            "generalization pass was skipped and gap stays unknown."
        )
        return False
    images_dir = Path(images_raw)
    if not images_dir.is_dir():
        warnings.append(
            f"the calibration image directory {images_dir} no longer exists, so "
            "the held-out generalization pass was skipped and gap stays unknown."
        )
        return False
    images = _load_images(
        images_dir, nrows=marker.get("nrows"), ncols=marker.get("ncols")
    )
    if not images:
        warnings.append(
            f"no readable images under {images_dir}, so the held-out "
            "generalization pass was skipped and gap stays unknown."
        )
        return False
    split, images_by_name, _cal = _resolve_calibration_images(
        spec, images, output_dir
    )
    _finalize_generalization(
        headline, spec, output_dir, split, images, images_by_name
    )
    return io.generalization_path(output_dir).is_file()
