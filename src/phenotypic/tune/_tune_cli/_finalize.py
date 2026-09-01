"""Lifecycle-owned publication for a finished distributed Tune study."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from phenotypic._cli._cli_slurm_lifecycle import (
    SlurmGenerationInactiveError,
    _deactivate_generation_locked,
    cancel_generation,
    generation_publication_guard,
    lifecycle_lock_path,
    load_slurm_lifecycle,
    mark_generation_failed,
)
from phenotypic.sdk_ import _io_constants as io
from phenotypic.sdk_._file_locking import exclusive_path_lock

from .._multi_objective import (
    is_multi_objective,
    objective_directions,
    objective_names,
)
from .._spec import TuningSpec
from .._study._protocol import StudyStore
from ._run import (
    _STUDY_NAME,
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


@dataclass(frozen=True)
class FinalizeResult:
    """Durable publication result for one finished distributed study."""

    output_dir: Path
    n_trials: int
    winner_trial_number: int
    selection: str
    best_params_written: bool
    pareto_published: bool
    generalization_written: bool
    warnings: tuple[str, ...] = ()


def _read_run_marker(output_dir: Path) -> dict[str, Any]:
    """Read the recorded backend, budget, and calibration source."""
    marker_path = io.tune_cache_run_marker_path(output_dir)
    if not marker_path.is_file():
        raise FileNotFoundError(
            f"{output_dir} carries no {marker_path.name} run marker"
        )
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{marker_path} is not a run marker object")
    return payload


def _read_resolved_spec(output_dir: Path) -> TuningSpec:
    """Read the exact resolved tuning spec used by the worker fleet."""
    spec_path = io.resolve_tuning_spec_path(output_dir)
    if not spec_path.is_file():
        raise FileNotFoundError(
            f"{output_dir} carries no resolved tuning spec at {spec_path}"
        )
    return TuningSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))


def _validated_marker_study_name(marker: dict[str, Any]) -> str:
    """Require the current supported study identity before storage access."""
    study_name = marker.get("study_name")
    if study_name != _STUDY_NAME:
        raise RuntimeError(
            "the run marker study_name does not match the supported study identity: "
            f"expected {_STUDY_NAME!r}, found {study_name!r}"
        )
    return study_name


def _open_finished_store(
    spec: TuningSpec, output_dir: Path, marker: dict[str, Any]
) -> StudyStore:
    """Open the marker-bound backing store without create semantics."""
    study_name = _validated_marker_study_name(marker)
    objective_axes = (
        tuple(objective_names(spec.scorer))
        if is_multi_objective(spec.scorer)
        else None
    )
    return _open_store(
        spec.strategy,
        output_dir,
        storage_url=marker.get("storage_url"),
        resume_path=io.trials_parquet_path(output_dir),
        directions=objective_directions(spec.scorer),
        objective_axes=objective_axes,
        study_name=study_name,
        create=False,
    )


def _expected_terminal_budget(marker: dict[str, Any]) -> int:
    """Return the recorded positive terminal-trial budget or fail closed."""
    budget = marker.get("n_trials")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        raise RuntimeError(
            "the run marker has no valid positive expected terminal trial budget"
        )
    return budget


def _best_params_matches(
    output_dir: Path, *, winner: Any, selection: str
) -> bool:
    """Return whether the final completion artifact names this winner."""
    path = io.best_params_path(output_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(payload, dict)
        and payload.get("trial_number") == winner.number
        and payload.get("params") == winner.params
        and payload.get("selection") == selection
    )


def _publish_distributed_study(
    output_dir: Path, *, expected_generation: Optional[str] = None
) -> FinalizeResult:
    """Publish a finished distributed study while the caller owns exclusion."""
    output = Path(output_dir)
    marker = _read_run_marker(output)
    if (
        expected_generation is not None
        and marker.get("generation") != expected_generation
    ):
        raise RuntimeError(
            "the run marker generation does not match the active finalizer: "
            f"expected {expected_generation!r}, found {marker.get('generation')!r}"
        )
    _validated_marker_study_name(marker)
    spec = _read_resolved_spec(output)
    store = _open_finished_store(spec, output, marker)
    trials = list(store.trials)
    terminal_trials = list(store.terminal_trials())
    expected = _expected_terminal_budget(marker)
    completed = store.completed_count()
    if completed < expected:
        raise RuntimeError(
            "study_not_finished: the study holds "
            f"{completed} of {expected} completed-or-pruned terminal trials"
        )

    scorer = getattr(spec, "scorer", None)
    multi_objective_run = (
        is_multi_objective(scorer) if scorer is not None else None
    )
    objective_axes = (
        tuple(objective_names(scorer))
        if multi_objective_run
        else None
    )
    winner = _headline_winner(
        store,
        multi_objective=multi_objective_run,
        objective_axes=objective_axes,
    )
    if winner is None:
        raise RuntimeError(
            "the terminal study has no valid winner; refusing publication"
        )
    selection = _selection_label(
        store,
        multi_objective=multi_objective_run,
        objective_axes=objective_axes,
    )
    warnings: list[str] = []
    in_flight = len(trials) - len(terminal_trials)
    if in_flight:
        warnings.append(
            f"{in_flight} in-flight trial(s) were excluded from publication"
        )

    _finalize_outputs(
        store,
        io.trials_parquet_path(output),
        output,
        _pipeline_for_trial(spec, winner),
    )
    _finalize_pareto_outputs(store, spec, output, objective_axes)
    generalization_written = _finalize_generalization_from_disk(
        winner, spec, output, marker, warnings
    )
    # This is the final write: readers use best_params.json as the durable
    # signal that every earlier output for the chosen winner has landed.
    _finalize_best_params(winner, output, selection=selection)
    best_params_written = _best_params_matches(
        output, winner=winner, selection=selection
    )
    if not best_params_written:
        raise RuntimeError(
            "best_params_written is false after distributed publication"
        )

    return FinalizeResult(
        output_dir=output,
        n_trials=len(trials),
        winner_trial_number=winner.number,
        selection=selection,
        best_params_written=True,
        pareto_published=io.pareto_dir(output).is_dir(),
        generalization_written=generalization_written,
        warnings=tuple(warnings),
    )


def _finalize_generalization_from_disk(
    winner: Any,
    spec: TuningSpec,
    output_dir: Path,
    marker: dict[str, Any],
    warnings: list[str],
) -> bool:
    """Reload calibration images and publish the held-out verdict."""
    images_raw = marker.get("images_dir")
    if not images_raw:
        raise RuntimeError(
            "the run marker records no calibration images_dir; "
            "refusing publication"
        )
    images_dir = Path(images_raw)
    if not images_dir.is_dir():
        raise RuntimeError(
            f"the calibration image directory {images_dir} is unavailable; "
            "refusing publication"
        )
    images = _load_images(
        images_dir,
        nrows=marker.get("nrows"),
        ncols=marker.get("ncols"),
    )
    if not images:
        raise RuntimeError(
            f"no readable calibration images under {images_dir}; "
            "refusing publication"
        )
    split, images_by_name, _calibration = _resolve_calibration_images(
        spec, images, output_dir
    )
    _finalize_generalization(
        winner, spec, output_dir, split, images, images_by_name
    )
    return io.generalization_path(output_dir).is_file()


def finalize_owned_generation(
    output_dir: Path, generation: str
) -> FinalizeResult:
    """Publish and close exactly one active Slurm lifecycle generation."""
    output = Path(output_dir)
    try:
        with generation_publication_guard(output, generation):
            result = _publish_distributed_study(
                output, expected_generation=generation
            )
            if not _deactivate_generation_locked(output, generation):
                raise SlurmGenerationInactiveError(
                    f"SLURM generation {generation!r} became inactive"
                )
            return result
    except SlurmGenerationInactiveError:
        raise
    except Exception as exc:
        mark_generation_failed(output, generation, str(exc))
        raise


def _marker_generation(marker: dict[str, Any]) -> Optional[str]:
    """Return a valid marker generation, preserving generation-less legacy runs."""
    if "generation" not in marker:
        return None
    generation = marker["generation"]
    if not isinstance(generation, str) or not generation:
        raise RuntimeError("the run marker carries an invalid lifecycle generation")
    return generation


def _require_generation_authority(
    output_dir: Path, marker_generation: str
) -> dict[str, Any]:
    """Return lifecycle state matching a new-style run marker or fail closed."""
    state = load_slurm_lifecycle(output_dir)
    if state is None:
        raise RuntimeError(
            "lifecycle authority is missing or corrupt for run marker generation "
            f"{marker_generation!r}"
        )
    state_generation = str(state.get("generation"))
    if state_generation != marker_generation:
        raise RuntimeError(
            "lifecycle authority generation does not match the run marker: "
            f"expected {marker_generation!r}, found {state_generation!r}"
        )
    return state


def _recorded_generation(
    output_dir: Path, *, marker_generation: Optional[str] = None
) -> Optional[str]:
    """Read the generation to cancel without holding the lock during cancellation."""
    if marker_generation is not None:
        _require_generation_authority(output_dir, marker_generation)
    with exclusive_path_lock(lifecycle_lock_path(output_dir), timeout=300.0):
        if marker_generation is not None:
            _require_generation_authority(output_dir, marker_generation)
            return marker_generation
        state = load_slurm_lifecycle(output_dir)
        if state is None:
            return None
        return str(state["generation"])


def finalize_distributed_study(
    output_dir: Path, *, force: bool = False
) -> FinalizeResult:
    """Manually publish a distributed run with lifecycle-wide exclusion."""
    output = Path(output_dir)
    marker = _read_run_marker(output)
    _validated_marker_study_name(marker)
    marker_generation = _marker_generation(marker)
    if marker_generation is not None:
        _require_generation_authority(output, marker_generation)
    cancelled_generation: Optional[str] = None
    if force:
        cancelled_generation = _recorded_generation(
            output, marker_generation=marker_generation
        )
        if cancelled_generation is not None:
            cancellation = cancel_generation(output, cancelled_generation)
            if not cancellation.quiescent:
                raise RuntimeError(
                    "forced finalization requires CancellationResult.quiescent "
                    "to be true"
                )
            if cancellation.unresolved_tokens:
                tokens = ", ".join(cancellation.unresolved_tokens)
                raise RuntimeError(
                    "forced finalization has unresolved scheduler tokens: "
                    f"{tokens}"
                )

    with exclusive_path_lock(lifecycle_lock_path(output), timeout=300.0):
        state = (
            _require_generation_authority(output, marker_generation)
            if marker_generation is not None
            else load_slurm_lifecycle(output)
        )
        if state is not None and state.get("active") is True:
            raise RuntimeError(
                "manual finalization refuses active Slurm generation "
                f"{state['generation']!r}"
            )
        if (
            force
            and cancelled_generation is not None
            and (
                state is None
                or str(state.get("generation")) != cancelled_generation
            )
        ):
            successor = None if state is None else state.get("generation")
            raise RuntimeError(
                "a new Slurm generation became owner after cancellation: "
                f"{successor!r}"
            )
        if marker_generation is None:
            return _publish_distributed_study(output)
        return _publish_distributed_study(
            output, expected_generation=marker_generation
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m phenotypic.tune._tune_cli._finalize"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--generation", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the exact-generation terminal finalizer."""
    import sys

    raw = list(sys.argv[1:]) if argv is None else list(argv)
    args = _build_parser().parse_args(raw)
    finalize_owned_generation(Path(args.output), args.generation)


__all__ = [
    "FinalizeResult",
    "finalize_distributed_study",
    "finalize_owned_generation",
]


if __name__ == "__main__":
    main()
