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
from .._evaluation import (
    Split,
    _dataset_identity,
    infer_group_key,
    resolve_split,
    run_held_out,
)
from .._multi_objective import (
    objective_directions,
    reject_grid_random_multi_objective,
)
from .._screening import compute_param_importance
from .._screening_freeze import ScreeningConfig, ScreeningController
from .._spec import TuningSpec
from .._strategies._config import (
    OPTUNA_SAMPLERS,
    PHENOTYPIC_TUNE_STORAGE_URL_ENV,
    GridConfig,
    OptunaConfig,
    RandomConfig,
    StrategyConfig,
)
from .._study._protocol import StudyStore
from .._study_store import JournalStudyStore, Trial

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".h5"}


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
    if name in OPTUNA_SAMPLERS:
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
    directions: Optional[list[str]] = None,
) -> StudyStore:
    """Select + open the study backend matching ``strategy``.

    An Optuna strategy gets a resumable :class:`OptunaStudyStore` on the shared
    ``study.db`` (or the explicit ``storage_url``); any other strategy gets the
    homegrown :class:`JournalStudyStore` (resumed from ``trials.parquet`` when one
    already exists). A multi-objective run passes ``directions`` so the Optuna
    store opens a multi-objective study (its ``append`` records the per-objective
    vector and ``pareto_front`` reads the study's native ``best_trials``).

    Args:
        strategy: The resolved strategy config.
        output_dir: The run directory (for ``study.db`` placement).
        storage_url: An explicit Optuna storage URL; ``None`` → ``study.db``.
        resume_path: The ``trials.parquet`` path the journal resumes from.
        directions: Per-objective ``["maximize"] * n`` for a multi-objective run
            (Optuna store only); ``None`` → single-objective.

    Returns:
        The opened store.
    """
    if _is_optuna_strategy(strategy):
        from .._study._optuna_store import OptunaStudyStore

        url = storage_url or f"sqlite:///{io.study_db_path(output_dir)}"
        return OptunaStudyStore(
            storage_url=url, study_name="tune", directions=directions
        )
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
    held_out_fraction: Optional[float] = None,
    cv_group: Optional[str] = None,
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
        held_out_fraction: Optional ``--held-out-fraction`` override of the spec's
            :attr:`HeldOutConfig.held_out_fraction` (robust-eval). ``None`` keeps
            the spec value. CLI flag > spec value > inference.
        cv_group: Optional ``--cv-group`` override of the held-out grouping column
            (:attr:`HeldOutConfig.group_key`). ``None`` keeps the spec value (then
            the scorer's inferred ``groupby[0]``). The gap margins stay spec-only.

    Returns:
        The best :class:`Trial`, or ``None`` (e.g. a fire-and-forget SLURM
        submission, or no successful trial).
    """
    output_dir = Path(output_dir)

    resolved_spec = spec
    if strategy is not None:
        resolved_spec = spec.model_copy(
            update={
                "strategy": resolve_strategy(
                    strategy, n_trials=n_trials, storage_url=storage_url
                )
            }
        )
    # A ``--strategy grid``/``random`` override bypasses TuningSpec's
    # construction-time guard (``model_copy`` skips validators), so re-assert it
    # at run validation — before any output is written — so a multi-objective
    # scorer without an Optuna strategy aborts cleanly with an actionable error.
    reject_grid_random_multi_objective(
        resolved_spec.scorer, resolved_spec.strategy
    )

    # Held-out CLI overrides (robust-eval): --held-out-fraction / --cv-group take
    # precedence over the spec's HeldOutConfig (CLI flag > spec value > inference)
    # and are folded in BEFORE the resolved spec is persisted, so tuning_spec.json
    # records the policy the run actually used. The gap margins stay spec-only.
    resolved_spec = _apply_held_out_overrides(
        resolved_spec, held_out_fraction=held_out_fraction, cv_group=cv_group
    )

    io.deliverables_dir(output_dir).mkdir(parents=True, exist_ok=True)

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
    # Multi-objective is inferred from the scorer (plan §0b): the directions feed
    # both the Optuna store (Pareto front) and, via the engine, the NSGA-II study.
    directions = objective_directions(resolved_spec.scorer)
    store = _open_store(
        resolved_spec.strategy,
        output_dir,
        storage_url=storage_url,
        resume_path=trials_path,
        directions=directions,
    )

    # Robust-eval held-out split (4.5p2): resolve (read-if-exists-else-derive)
    # the calibration / held-out partition, then run the search on the
    # calibration plates ONLY — the held-out plates are reserved for the
    # report-only generalization pass and never touch the optimizer. The split
    # lives in the run layer; the engine stays a pure optimizer (RESOLVED design).
    split, images_by_name, cal_images = _resolve_calibration_images(
        resolved_spec, images, output_dir
    )

    if screen:
        best = _run_screened(resolved_spec, cal_images, store)
        winner_pipeline = _best_pipeline(resolved_spec, store)
    else:
        engine = TuningEngine(resolved_spec, store=store)
        best = engine.optimize(cal_images)
        winner_pipeline = engine.best_pipeline()

    _finalize_outputs(store, trials_path, output_dir, winner_pipeline)
    # Multi-objective runs additionally publish deliverables/pareto/ (front +
    # per-objective best pipelines) and overwrite best_pipeline.json with the
    # knee; a single-objective run's empty front makes this a no-op (no pareto/
    # dir — the back-compat lock, plan §0b).
    _finalize_pareto_outputs(store, resolved_spec, output_dir)
    # Report-only held-out generalization verdict → deliverables/generalization.json.
    _finalize_generalization(
        store, resolved_spec, output_dir, split, images, images_by_name
    )
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


def _apply_held_out_overrides(
    spec: TuningSpec,
    *,
    held_out_fraction: Optional[float],
    cv_group: Optional[str],
) -> TuningSpec:
    """Fold the ``--held-out-fraction`` / ``--cv-group`` flags into the spec.

    Returns ``spec`` untouched when neither flag is given; otherwise a copy whose
    :class:`~phenotypic.tune.HeldOutConfig` carries the overrides (the only fields
    the flags touch — the gap margins stay spec-only). CLI flag > spec value >
    inference precedence: a flag wins over the spec block, and an unset flag
    leaves the spec value (which itself defers to inference downstream).

    Args:
        spec: The resolved tuning spec.
        held_out_fraction: The ``--held-out-fraction`` override, or ``None``.
        cv_group: The ``--cv-group`` grouping-column override, or ``None``.

    Returns:
        The spec (possibly a ``model_copy`` with an overridden ``held_out`` block).
    """
    updates: dict[str, Any] = {}
    if held_out_fraction is not None:
        updates["held_out_fraction"] = held_out_fraction
    if cv_group is not None:
        updates["group_key"] = cv_group
    if not updates:
        return spec
    return spec.model_copy(
        update={"held_out": spec.held_out.model_copy(update=updates)}
    )


def _resolve_calibration_images(
    spec: TuningSpec, images: list, output_dir: Path
) -> tuple[Split, dict[str, Any], list]:
    """Resolve the held-out split and the calibration-only search set.

    Reads-if-exists-else-derives the persisted split (so resume reuses the
    original partition regardless of the new master seed), then partitions the
    loaded plates by **name-membership** (RESOLVED design): a held-out plate is
    one whose ``name`` is in ``split.held_out``; calibration is everything else,
    so a NEW plate present in neither list falls into calibration (never
    silently held out). The master seed is the strategy's ``seed`` (grid/random),
    and the group key is the explicit ``held_out.group_key`` or, when unset, the
    count scorer's inferred ``groupby[0]`` (CLI flag > spec value > inference is
    enforced upstream when the spec's ``held_out`` is overridden).

    Args:
        spec: The resolved tuning spec (``strategy.seed`` + ``held_out`` policy +
            ``scorer`` for group-key inference).
        images: The loaded plates.
        output_dir: The run directory (where ``splits/split.json`` lives).

    Returns:
        ``(split, images_by_name, calibration_images)`` — the resolved split, the
        ``{name: image}`` index of the loaded plates, and the calibration subset
        the search runs on.
    """
    held_out = spec.held_out
    master_seed = int(getattr(spec.strategy, "seed", 0) or 0)
    group_key = held_out.group_key or infer_group_key(spec.scorer)
    split = resolve_split(
        output_dir,
        images,
        master_seed=master_seed,
        group_key=group_key,
        held_out_fraction=held_out.held_out_fraction,
        min_heldout_plates=held_out.min_heldout_plates,
    )
    images_by_name = {im.name: im for im in images}
    held_out_names = set(split.held_out)
    # Calibration = every loaded plate NOT in the held-out list (a new plate,
    # absent from both lists, falls here rather than being silently reserved).
    cal_images = [im for im in images if im.name not in held_out_names]
    return split, images_by_name, cal_images


def _finalize_generalization(
    store: StudyStore,
    spec: TuningSpec,
    output_dir: Path,
    split: Split,
    images: list,
    images_by_name: dict[str, Any],
) -> None:
    """Run the report-only held-out pass on the winner → ``generalization.json``.

    Re-evaluates ``store.best()`` on the held-out plates (the 3-tier verdict by
    ``split.kind``) and writes ``deliverables/generalization.json``. A run with no
    successful trial (no winner) skips the report. The dataset-changed flag is
    resolved here by comparing the current :func:`_dataset_identity` of the loaded
    plates against the persisted ``split.dataset_identity``.

    Args:
        store: The finished study store (its ``best()`` is the winner).
        spec: The resolved tuning spec (its ``evaluator`` runs the held-out pass).
        output_dir: The run directory.
        split: The resolved held-out split.
        images: The loaded plates (for the current dataset identity).
        images_by_name: ``{name: image}`` of the loaded plates.
    """
    winner = store.best()
    if winner is None:
        return  # no successful trial → no generalization verdict
    report = run_held_out(
        spec,
        winner,
        split,
        images_by_name,
        current_identity=_dataset_identity(images),
    )
    io.generalization_path(output_dir).write_text(
        json.dumps(report.to_dict(), indent=2)
    )


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


def _finalize_pareto_outputs(
    store: StudyStore, spec: TuningSpec, output_dir: Path
) -> None:
    """Publish ``deliverables/pareto/`` when the run was multi-objective.

    A multi-objective run (a scorer whose ``finalize`` returns a dict, so trials
    carry ``Trial.objectives``) has a non-empty :meth:`StudyStore.pareto_front`;
    a single-objective run's front is empty and this is a **no-op** — no
    ``pareto/`` directory is created (the back-compat lock, plan §0b). When the
    front is non-empty it writes, under :func:`pareto_dir`:

    * ``pareto_front.parquet`` — the front's trials (same schema as
      ``trials.parquet``, ``objectives_json`` populated);
    * ``best_<objective>.json`` — the front pipeline maximizing each objective
      axis, plus that axis's :func:`compute_param_importance`;
    * and it overwrites the top-level ``best_pipeline.json`` with the **knee**
      (the max-curvature compromise pick).

    Args:
        store: The finished study store (any backend).
        spec: The resolved tuning spec (its ``pipeline`` is the build base).
        output_dir: The run directory.
    """
    from .._evaluation import build_pipeline

    front = store.pareto_front()
    if not front:
        return  # single-objective run — no pareto/ dir (back-compat lock)

    pareto_dir = io.pareto_dir(output_dir)
    pareto_dir.mkdir(parents=True, exist_ok=True)

    # The front parquet (mirror the front into a journal for a uniform schema).
    JournalStudyStore(list(front)).to_parquet(io.pareto_front_parquet_path(output_dir))

    # One best pipeline + importance per objective axis (stable name order).
    # Source the axis order from the scorer (authoritative — every axis the
    # study optimized) rather than an arbitrary front member, so each axis gets
    # a best_<axis>.json even when the front's first trial floored one to 0.0;
    # fall back to the first trial's keys for a scorer exposing no names.
    from .._multi_objective import objective_names as _scorer_objective_axes

    objective_axes = _scorer_objective_axes(spec.scorer) or list(
        front[0].objectives or {}
    )
    for name in objective_axes:
        winner = max(
            (t for t in front if t.objectives and name in t.objectives),
            key=lambda t: t.objectives[name],  # type: ignore[index]
            default=None,
        )
        if winner is not None:
            pipeline = build_pipeline(spec.pipeline, winner.params)
            io.pareto_best_pipeline_path(output_dir, name).write_text(
                pipeline.to_json() or ""
            )
        io.pareto_importance_path(output_dir, name).write_text(
            json.dumps(compute_param_importance(store, objective=name), indent=2)
        )

    # The knee is the run's headline winner — overwrite best_pipeline.json.
    knee = store.knee_point(front)
    if knee is not None:
        knee_pipeline = build_pipeline(spec.pipeline, knee.params)
        io.best_pipeline_path(output_dir).write_text(knee_pipeline.to_json() or "")
