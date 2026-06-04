"""Run-a-tuning-spec orchestration + the ``deliverables/`` writes.

The ``run`` subcommand's body (``__main__`` parses the flags and forwards them
here). Beyond the Phase-1 local engine run, this resolves the ``--strategy`` flag
into the right :class:`~phenotypic.tune.StrategyConfig` (grid/random → the Phase-1
configs; an Optuna sampler → :class:`~phenotypic.tune.OptunaConfig`), selects the
matching study backend (a resumable :class:`OptunaStudyStore` for an Optuna
strategy, else the :class:`JournalStudyStore`), optionally screens (the two-round
freeze, ``--screen``), and — on the Optuna path — **also exports
``trials.parquet``** at finalize so ``deliverables/`` stay backend-agnostic. With
``--slurm`` the run submits a distributed worker fleet via
:class:`~phenotypic._execution.SlurmExecutor` instead of running in-process.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from phenotypic import GridImage
from phenotypic._execution._slurm import SlurmExecutor
from phenotypic.tools_ import _io_constants as io

from .._engine import TuningEngine
from .._screening import compute_param_importance
from .._screening_freeze import ScreeningConfig, ScreeningController
from .._spec import TuningSpec
from .._strategies._config import (
    PHENOTYPIC_TUNE_STORAGE_URL_ENV,
    GridConfig,
    OptunaConfig,
    RandomConfig,
    StrategyConfig,
)
from .._study._protocol import StudyStore
from .._study_store import JournalStudyStore, Trial

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".h5"}

#: Strategy names that map onto an Optuna sampler (an Optuna study backend); the
#: remaining names (``grid`` / ``random``) use the homegrown journal backend.
_OPTUNA_SAMPLERS = frozenset({"tpe", "cmaes", "gp", "nsga2"})


def _load_images(input_dir: Path) -> list:
    """Load every image file under ``input_dir`` as a ``GridImage``.

    Mirrors the forward CLI's directory scan; tuning targets arrayed plates, so
    images load as ``GridImage`` via ``imread``. Unreadable / non-grid files are
    skipped (warned) rather than aborting the whole run.

    Args:
        input_dir: The directory to scan (non-recursive).

    Returns:
        The loaded ``GridImage`` instances, in sorted filename order.
    """
    paths = sorted(
        p for p in Path(input_dir).iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )
    images: list = []
    failures: list[tuple[str, str]] = []
    for path in paths:
        try:
            images.append(GridImage.imread(path))
        except Exception as exc:  # skip unreadable / non-grid files, don't abort
            failures.append((path.name, str(exc)))
    if failures:
        logging.getLogger(__name__).warning(
            "skipped %d unreadable image(s): %s",
            len(failures), ", ".join(name for name, _ in failures),
        )
    return images


def resolve_strategy(
    name: str, *, n_trials: Optional[int], storage_url: Optional[str]
) -> StrategyConfig:
    """Map a ``--strategy`` name onto a :class:`StrategyConfig`.

    ``grid`` / ``random`` build the Phase-1 configs (random needs ``n_trials``);
    an Optuna sampler (``tpe`` / ``cmaes`` / ``gp`` / ``nsga2``) builds an
    :class:`OptunaConfig`, first calling ``_require_optuna`` so a missing ``tune``
    extra raises an **actionable** ``ImportError`` (pointing at
    ``uv sync --extras tune``) rather than a bare ``KeyError`` deep in ``build``.

    Args:
        name: The strategy name (a closed set; validated by the CLI choices).
        n_trials: The trial budget (required for ``random`` and the Optuna
            samplers; defaults to ``50`` for an Optuna sampler when omitted).
        storage_url: The Optuna storage URL (ignored by grid/random).

    Returns:
        The resolved :class:`StrategyConfig`.

    Raises:
        ImportError: When an Optuna sampler is requested without the ``tune``
            extra.
        ValueError: For an unknown strategy name (defence in depth — the CLI
            ``choices`` should reject it first).
    """
    if name == "grid":
        return GridConfig()
    if name == "random":
        return RandomConfig(n_trials=n_trials if n_trials is not None else 50)
    if name in _OPTUNA_SAMPLERS:
        # Fail fast + actionable when the extra is missing, before constructing a
        # config the engine could not build.
        from .._strategies import _optuna_support

        _optuna_support._require_optuna()
        return OptunaConfig(
            sampler=name,  # type: ignore[arg-type]  # name ∈ SamplerKind here
            n_trials=n_trials if n_trials is not None else 50,
            storage_url=storage_url,
        )
    raise ValueError(f"unknown strategy {name!r}")


def _is_optuna_strategy(strategy: StrategyConfig) -> bool:
    """Whether ``strategy`` drives an Optuna study backend."""
    return isinstance(strategy, OptunaConfig)


def _open_store(
    strategy: StrategyConfig,
    output_dir: Path,
    *,
    storage_url: Optional[str],
    resume_path: Path,
) -> StudyStore:
    """Select + open the study backend matching ``strategy``.

    An Optuna strategy gets a resumable :class:`OptunaStudyStore` on the shared
    ``study.db`` (or the explicit ``storage_url``); any other strategy gets the
    homegrown :class:`JournalStudyStore` (resumed from ``trials.parquet`` when one
    already exists).

    Args:
        strategy: The resolved strategy config.
        output_dir: The run directory (for ``study.db`` placement).
        storage_url: An explicit Optuna storage URL; ``None`` → ``study.db``.
        resume_path: The ``trials.parquet`` path the journal resumes from.

    Returns:
        The opened store.
    """
    if _is_optuna_strategy(strategy):
        from .._study._optuna_store import OptunaStudyStore

        url = storage_url or f"sqlite:///{io.study_db_path(output_dir)}"
        return OptunaStudyStore(storage_url=url, study_name="tune")
    if resume_path.exists():
        return JournalStudyStore.from_parquet(resume_path)
    return JournalStudyStore()


def run_tuning(
    spec: TuningSpec,
    images: list,
    output_dir: Path,
    *,
    strategy: Optional[str] = None,
    n_trials: Optional[int] = None,
    screen: bool = False,
    storage_url: Optional[str] = None,
    slurm: bool = False,
    spec_path: Optional[Path] = None,
    images_dir: Optional[Path] = None,
) -> Optional[Trial]:
    """Run ``spec`` over ``images`` and write the ``deliverables/`` artifacts.

    Writes ``trials.parquet`` (root) and, under ``deliverables/``,
    ``tuning_spec.json`` / ``best_pipeline.json`` / ``param_importance.json``.
    ``--strategy`` overrides the spec's strategy (selecting the Optuna study
    backend for an Optuna sampler); ``--screen`` runs the two-round freeze;
    ``--storage-url`` / ``$PHENOTYPIC_TUNE_STORAGE_URL`` names a shared Optuna
    store; ``--slurm`` submits a distributed worker fleet.

    Args:
        spec: The tuning recipe.
        images: The calibration images.
        output_dir: The run directory.
        strategy: Optional ``--strategy`` override (grid/random/tpe/cmaes/gp/nsga2).
        n_trials: Optional trial-budget override forwarded to the strategy.
        screen: Whether to run the two-round screening freeze.
        storage_url: Optional Optuna storage URL (falls back to the env var).
        slurm: Whether to submit a distributed worker fleet instead of running
            locally.
        spec_path: Path to the on-disk ``tuning_spec.json`` (required for
            ``--slurm`` so each worker can load it).
        images_dir: The calibration image directory (required for ``--slurm``).

    Returns:
        The best :class:`Trial`, or ``None`` (e.g. a fire-and-forget SLURM
        submission, or no successful trial).
    """
    output_dir = Path(output_dir)
    io.deliverables_dir(output_dir).mkdir(parents=True, exist_ok=True)

    resolved_spec = spec
    if strategy is not None:
        resolved_spec = spec.model_copy(
            update={
                "strategy": resolve_strategy(
                    strategy, n_trials=n_trials, storage_url=storage_url
                )
            }
        )

    # Always echo the resolved spec so the deliverable is re-runnable.
    io.tuning_spec_path(output_dir).write_text(
        resolved_spec.model_dump_json(indent=2)
    )

    if slurm:
        return _submit_slurm_fleet(
            resolved_spec,
            output_dir,
            storage_url=storage_url,
            spec_path=spec_path,
            images_dir=images_dir,
        )

    trials_path = io.trials_parquet_path(output_dir)
    store = _open_store(
        resolved_spec.strategy,
        output_dir,
        storage_url=storage_url,
        resume_path=trials_path,
    )

    if screen:
        best = _run_screened(resolved_spec, images, store)
        winner_pipeline = _best_pipeline(resolved_spec, store)
    else:
        engine = TuningEngine(resolved_spec, store=store)
        best = engine.optimize(images)
        winner_pipeline = engine.best_pipeline()

    _finalize_outputs(store, trials_path, output_dir, winner_pipeline)
    return best


def _run_screened(
    spec: TuningSpec, images: list, store: StudyStore
) -> Optional[Trial]:
    """Run the two-round screening freeze, journaling into ``store``.

    The controller drives its own explore/focused stores; we mirror the combined
    trials into ``store`` so the standard finalize writes one ``trials.parquet``
    and the importance report covers both rounds.
    """
    controller = ScreeningController(spec, config=ScreeningConfig())
    result = controller.run(images)
    combined = controller.explore_store.trials + (
        controller.focused_store.trials if controller.focused_store else []
    )
    for offset, trial in enumerate(combined):
        store.append(trial.model_copy(update={"number": offset}))
    return result.winner


def _best_pipeline(spec: TuningSpec, store: StudyStore):
    """Build the winning pipeline from the store's best trial (or ``None``)."""
    from .._evaluation import build_pipeline

    best = store.best()
    if best is None:
        return None
    return build_pipeline(spec.pipeline, best.params)


def _submit_slurm_fleet(
    spec: TuningSpec,
    output_dir: Path,
    *,
    storage_url: Optional[str],
    spec_path: Optional[Path],
    images_dir: Optional[Path],
) -> Optional[Trial]:
    """Submit a distributed worker fleet via :class:`SlurmExecutor`.

    The shared study URL is the explicit ``storage_url``, the
    ``$PHENOTYPIC_TUNE_STORAGE_URL`` env fallback, or the run's ``study.db``.
    Fire-and-forget: the fleet writes into the shared study; the final
    ``trials.parquet`` export happens on a later ``--recompile`` finalize.
    """
    if spec_path is None or images_dir is None:
        raise ValueError(
            "--slurm requires the on-disk spec path and image directory "
            "(each worker reloads them)"
        )
    url = (
        storage_url
        or os.environ.get(PHENOTYPIC_TUNE_STORAGE_URL_ENV)
        or f"sqlite:///{io.study_db_path(output_dir)}"
    )
    n_trials = getattr(spec.strategy, "n_trials", None)
    n_workers = min(8, n_trials) if isinstance(n_trials, int) and n_trials > 0 else 4
    executor = SlurmExecutor(
        output_dir=output_dir,
        spec_path=Path(spec_path),
        images_dir=Path(images_dir),
        study_name="tune",
        n_workers=n_workers,
        slurm_args={"slurm_partition": "batch"},
        storage_url=url,
    )
    executor.run(lambda w: w, list(range(n_workers)))
    return None


def _finalize_outputs(
    store: StudyStore,
    trials_path: Path,
    output_dir: Path,
    winner_pipeline: Any,
) -> None:
    """Export the journal + write the importance report + best pipeline.

    Always writes ``trials.parquet`` (exported from whatever backend ran, so an
    Optuna run's ``deliverables/`` are backend-agnostic) and
    ``param_importance.json``; writes ``best_pipeline.json`` when a winner exists.
    """
    _export_trials_parquet(store, trials_path)
    io.param_importance_path(output_dir).write_text(
        json.dumps(compute_param_importance(store), indent=2)
    )
    if winner_pipeline is not None:
        io.best_pipeline_path(output_dir).write_text(winner_pipeline.to_json() or "")


def _export_trials_parquet(store: StudyStore, trials_path: Path) -> None:
    """Write ``trials.parquet`` from any store (journal-native or via a mirror).

    A :class:`JournalStudyStore` writes itself; any other backend (e.g.
    :class:`OptunaStudyStore`) is mirrored into a fresh journal first so the
    parquet schema is identical regardless of the backend that produced it.
    """
    if isinstance(store, JournalStudyStore):
        store.to_parquet(trials_path)
        return
    mirror = JournalStudyStore(list(store.trials))
    mirror.to_parquet(trials_path)
