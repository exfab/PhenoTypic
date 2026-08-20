"""``python -m phenotypic.tune._tune_cli._worker`` — one distributed tune worker.

A single SLURM array task runs this module: it opens the **shared** Optuna study
by ``--study-name`` + ``--storage-url`` (the ``journal://`` file backend a
``--slurm`` run defaults to, or an explicit Postgres / SQLite-WAL URL), binds a
:class:`~phenotypic.tune.TuningEngine` to that resumable
:class:`~phenotypic.tune._study._optuna_store.OptunaStudyStore`, and runs the
ask→evaluate→tell loop until the shared budget exhausts (optuna-integration §7).
Concurrency comes from running **one worker per array task** against the one
study — the storage backend serializes the trial writes, and the engine skips the
deterministic replay because the Optuna store resumes in place.

Unlike the forward CLI's per-image array body, a tune worker carries **no
image-chunk sentinels**: there is nothing to checkpoint/finalize per task — the
study *is* the shared state, and the final ``trials.parquet`` export happens once
in the CLI's finalize step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .._spec import TuningSpec
from ._run import _load_images


def build_worker_store(
    *, storage_url: str, study_name: str, directions: Optional[list[str]] = None
):
    """Open the shared, resumable Optuna study this worker contributes trials to.

    Args:
        storage_url: The Optuna storage URL (``journal:///…`` for the fleet
            default, or an explicit ``postgresql+psycopg://…`` /
            ``sqlite:///…``). Every worker in the fleet passes the same URL so
            they share one study; the scheme is dispatched inside
            :class:`OptunaStudyStore`.
        study_name: The shared study name (same across the fleet).
        directions: Per-objective directions inferred from the scorer (multi-
            objective) or ``None`` (single-objective). Must match what the
            submitter pre-created the study with and what each worker's engine
            strategy infers — else the ``load_if_exists`` open conflicts on the
            study's objective shape.

    Returns:
        An :class:`~phenotypic.tune._study._optuna_store.OptunaStudyStore` bound to
        the shared study (``is_resumable_in_place() → True``).
    """
    from .._study._optuna_store import OptunaStudyStore

    return OptunaStudyStore(
        storage_url=storage_url, study_name=study_name, directions=directions
    )


def run_worker(
    *,
    spec_path: Path,
    images_dir: Path,
    split_path: Path,
    storage_url: str,
    study_name: str,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
) -> None:
    """Run one worker's ask→evaluate→tell loop against the shared study.

    Loads the spec + calibration images, opens the shared Optuna store, and drives
    a :class:`~phenotypic.tune.TuningEngine` over it. The engine's budget is read
    from the shared study (completed + pruned trials), so the fleet collectively
    stops at the spec's ``n_trials`` rather than each worker running a full budget.

    Args:
        spec_path: Path to the ``tuning_spec.json`` (the strategy must be an
            Optuna strategy so the engine binds to the shared study).
        images_dir: The calibration image directory.
        split_path: Path to the persisted split; held-out images are filtered
            before optimization.
        storage_url: The shared Optuna storage URL.
        study_name: The shared study name.
    """
    from .._engine import TuningEngine
    from .._multi_objective import objective_directions
    from .._evaluation import Split

    spec = TuningSpec.model_validate_json(Path(spec_path).read_text())
    images = _load_images(Path(images_dir), nrows=nrows, ncols=ncols)
    if not images:
        raise SystemExit(f"no images found under {str(images_dir)!r}")
    split = Split(**json.loads(Path(split_path).read_text()))
    held_out = set(split.held_out)
    images = [image for image in images if image.name not in held_out]
    if not images:
        raise SystemExit("no calibration images remain after applying held-out split")

    # Match the study's objective shape (single vs multi) so binding to the shared,
    # submitter-pre-created study never conflicts on directions.
    directions = objective_directions(spec.scorer)
    store = build_worker_store(
        storage_url=storage_url, study_name=study_name, directions=directions
    )
    TuningEngine(spec, store=store).optimize(images)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m phenotypic.tune._tune_cli._worker",
        description="Run one distributed tune worker bound to a shared study.",
    )
    parser.add_argument("--spec", required=True, help="path to tuning_spec.json")
    parser.add_argument(
        "--images", required=True, help="calibration image directory"
    )
    parser.add_argument(
        "--split", required=True, help="persisted held-out split.json"
    )
    parser.add_argument(
        "--study-name", required=True, help="the shared Optuna study name"
    )
    parser.add_argument(
        "--storage-url", required=True, help="the shared Optuna storage URL"
    )
    parser.add_argument(
        "--nrows", type=int, default=None,
        help="fixed grid row count for GridImage.imread (grid-cell scoring)",
    )
    parser.add_argument(
        "--ncols", type=int, default=None,
        help="fixed grid column count for GridImage.imread (grid-cell scoring)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point for one tune worker. See ``--help``."""
    import sys

    raw = list(sys.argv[1:]) if argv is None else list(argv)
    args = _build_parser().parse_args(raw)
    run_worker(
        spec_path=Path(args.spec),
        images_dir=Path(args.images),
        split_path=Path(args.split),
        storage_url=args.storage_url,
        study_name=args.study_name,
        nrows=args.nrows,
        ncols=args.ncols,
    )


if __name__ == "__main__":
    main()
