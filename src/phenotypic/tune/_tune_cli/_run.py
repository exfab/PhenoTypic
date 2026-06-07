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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Optional
from urllib.parse import urlsplit

from phenotypic import GridImage
from phenotypic._execution._slurm import SlurmExecutor
from phenotypic.tools_ import _io_constants as io
from phenotypic.tools_ import atomic_write_text

from .._engine import TuningEngine
from .._evaluation import (
    Split,
    _dataset_identity,
    infer_group_key,
    resolve_split,
    run_held_out,
)
from .._multi_objective import (
    is_multi_objective,
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

#: Default trial budget when ``--n-trials`` is omitted for a ``random`` or Optuna
#: strategy (grid is exhaustive and ignores it).
_DEFAULT_N_TRIALS: Final[int] = 50

#: Schema version of the ``.pht-tune-cache/run.json`` marker (bump on any key
#: change so a reader can branch on the contract).
_RUN_MARKER_VERSION: Final[int] = 1

#: The study name every tune run uses (the Optuna ``study_name`` + the marker's
#: ``study_name`` field). A single constant keeps the store, the SLURM fleet, and
#: the marker in lockstep.
_STUDY_NAME: Final[str] = "tune"


def _default_study_db_url(output_dir: Path) -> str:
    """SQLite URL for the run's ``study.db``, resuming a legacy-root copy.

    Resolves the study-DB location via :func:`resolve_study_db_path` (the hidden
    tune cache, falling back to a legacy output-root ``study.db`` so an
    in-flight legacy run resumes in place). A fresh run gets the new cache
    location (the resolver's no-file default).
    """
    return f"sqlite:///{io.resolve_study_db_path(output_dir)}"


def _resolve_storage_url(storage_url: Optional[str], output_dir: Path) -> str:
    """Resolve the Optuna storage URL with the canonical 3-way fallback.

    ``storage_url`` (the explicit ``--storage-url`` / param) wins; otherwise the
    ``$PHENOTYPIC_TUNE_STORAGE_URL`` env var (a shared distributed Postgres);
    otherwise the run's local ``study.db`` (resolved via
    :func:`_default_study_db_url`). Single-sourced so the SLURM fleet submission
    and the ``run.json`` marker agree on the URL a worker will actually open —
    a null URL in the marker would silently force the GUI Monitor into
    parquet-only mode for the env-driven distributed case.

    This is also the single chokepoint that **rejects a password-bearing URL**:
    an inline ``postgresql://user:secret@host/db`` password would be persisted
    verbatim into the ``run.json`` marker and the generated SLURM worker script
    (both world-readable on a shared cluster filesystem). Since this resolver
    feeds BOTH writers, guarding here covers every downstream consumer. The
    local SQLite fallback carries no password and always passes.

    Args:
        storage_url: The explicit storage URL, or ``None``.
        output_dir: The run output directory (for the ``study.db`` fallback).

    Returns:
        The resolved, non-null storage URL.

    Raises:
        ValueError: When the resolved URL embeds an inline password (keep the
            secret out of the URL: use ``~/.pgpass``, ``$PGPASSWORD``, or a
            ``PGSERVICE`` entry instead).
    """
    url = (
        storage_url
        or os.environ.get(PHENOTYPIC_TUNE_STORAGE_URL_ENV)
        or _default_study_db_url(output_dir)
    )
    _reject_password_in_url(url)
    return url


def _reject_password_in_url(url: str) -> None:
    """Raise if ``url`` embeds an inline password (the single chokepoint, B2).

    An inline password in the storage URL would be written verbatim into the
    GUI-discovery ``run.json`` marker and the generated SLURM worker script,
    both readable by anyone with filesystem access on a shared cluster. We
    refuse it here, the one place every resolved URL flows through, so neither
    sink ever sees the secret. A password-less Postgres URL (libpq resolves the
    secret from ``~/.pgpass`` / ``$PGPASSWORD`` / a ``PGSERVICE`` entry per
    worker) and the local SQLite fallback both pass.

    Args:
        url: The resolved storage URL.

    Raises:
        ValueError: When :attr:`urllib.parse.SplitResult.password` is non-null.
    """
    if urlsplit(url).password is not None:
        raise ValueError(
            "the Optuna storage URL embeds an inline password, which would be "
            "written in plaintext to the run.json marker and the SLURM worker "
            "script. Remove the password from the URL and let libpq resolve it: "
            "use a ~/.pgpass file, the $PGPASSWORD environment variable, or a "
            "PGSERVICE entry (e.g. postgresql+psycopg://user@host:5432/db)."
        )


def _strategy_marker_name(strategy: StrategyConfig) -> str:
    """The marker's ``strategy`` label for ``strategy``.

    An Optuna strategy reports its ``sampler`` (``"tpe"`` / ``"cmaes"`` /
    ``"gp"`` / ``"nsga2"``) — the user-facing name; grid/random report their
    ``kind`` discriminator. Falls back to ``kind`` for any other config.
    """
    sampler = getattr(strategy, "sampler", None)
    if sampler is not None:
        return str(sampler)
    return str(getattr(strategy, "kind", ""))


def _write_run_marker(
    output_dir: Path,
    spec: TuningSpec,
    *,
    storage_url: Optional[str],
    images_dir: Optional[Path],
    slurm: bool,
) -> None:
    """Write the ``.pht-tune-cache/run.json`` marker at run START.

    Emitted right after the ``deliverables/`` mkdir and BEFORE the engine/SLURM
    branch, so a live run is marked before any deliverable lands and the GUI
    shell classifier can recognise the output. The ``storage_url`` is resolved
    through the SAME 3-way fallback the SLURM fleet uses
    (:func:`_resolve_storage_url`), so the recorded URL is the one a worker will
    actually open — never null — keeping the GUI Monitor out of parquet-only
    mode for the env-var-driven distributed-Postgres case.

    Args:
        output_dir: The run output directory.
        spec: The RESOLVED tuning spec (its ``strategy`` + ``scorer`` populate
            the marker).
        storage_url: The explicit storage URL, or ``None`` (resolved here).
        images_dir: The calibration image directory (the ``-i`` arg), or ``None``.
        slurm: Whether this run submits a distributed worker fleet.
    """
    marker = {
        "version": _RUN_MARKER_VERSION,
        "study_name": _STUDY_NAME,
        "storage_url": _resolve_storage_url(storage_url, output_dir),
        "images_dir": str(images_dir) if images_dir is not None else None,
        "strategy": _strategy_marker_name(spec.strategy),
        "n_trials": getattr(spec.strategy, "n_trials", None),
        "is_multi_objective": is_multi_objective(spec.scorer),
        "slurm": slurm,
        "start_time": datetime.now(timezone.utc).isoformat(),
    }
    marker_path = io.tune_cache_run_marker_path(output_dir)
    # The marker is a GUI-discovery SIDECAR, not a deliverable: a read-only /
    # over-quota output FS (the HPCC reality) raising OSError here must NOT abort
    # the run before it starts. Catch-and-warn and proceed — mirrors the lost-race
    # tolerance in ``migrate_legacy_machine_state``.
    try:
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(marker_path, json.dumps(marker, indent=2))
    except OSError:
        logging.getLogger(__name__).warning(
            "could not write the tune run.json marker at %s; the run proceeds "
            "but the GUI may not auto-discover it",
            marker_path,
            exc_info=True,
        )


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
        return RandomConfig(
            n_trials=n_trials if n_trials is not None else _DEFAULT_N_TRIALS
        )
    if name in OPTUNA_SAMPLERS:
        # Fail fast + actionable when the extra is missing, before constructing a
        # config the engine could not build.
        from .._strategies import _optuna_support

        _optuna_support._require_optuna()
        return OptunaConfig(
            sampler=name,  # type: ignore[arg-type]  # name ∈ SamplerKind here
            n_trials=n_trials if n_trials is not None else _DEFAULT_N_TRIALS,
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
        storage_url: An explicit Optuna storage URL; ``None`` resolves via the
            3-way fallback (env var > local ``study.db``) in
            :func:`_resolve_storage_url`.
        resume_path: The ``trials.parquet`` path the journal resumes from.
        directions: Per-objective ``["maximize"] * n`` for a multi-objective run
            (Optuna store only); ``None`` → single-objective.

    Returns:
        The opened store.
    """
    if _is_optuna_strategy(strategy):
        from .._study._optuna_store import OptunaStudyStore

        # Resolve via the SAME 3-way fallback the run.json marker + SLURM fleet
        # use (explicit > $PHENOTYPIC_TUNE_STORAGE_URL > local study.db), so the
        # engine opens exactly the URL the marker records — no marker-vs-engine
        # divergence for an env-var-driven local run.
        url = _resolve_storage_url(storage_url, output_dir)
        return OptunaStudyStore(
            storage_url=url, study_name=_STUDY_NAME, directions=directions
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
    n_workers: Optional[int] = None,
    slurm_partition: Optional[str] = None,
    slurm_mem: Optional[str] = None,
    slurm_time: Optional[str] = None,
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
        n_workers: Optional ``--n-workers`` override of the SLURM fleet size
            (``--slurm`` only); ``None`` keeps the ``min(8, n_trials)`` default.
        slurm_partition: Optional ``--slurm-partition`` for the worker fleet
            (``--slurm`` only); ``None`` omits the ``#SBATCH --partition``
            directive (the cluster default partition).
        slurm_mem: Optional ``--slurm-mem`` per worker (``--slurm`` only), e.g.
            ``"8G"``; ``None`` omits the directive.
        slurm_time: Optional ``--slurm-time`` wall-clock limit per worker
            (``--slurm`` only), e.g. ``"04:00:00"``; ``None`` omits the directive.

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
    atomic_write_text(
        io.tuning_spec_path(output_dir), resolved_spec.model_dump_json(indent=2)
    )

    # Mark the run as a tune output at START — before the engine/SLURM branch, so
    # a live run (and a fire-and-forget SLURM submission) is GUI-discoverable
    # before any deliverable lands. The resolved (non-null) storage URL keeps the
    # GUI Monitor off parquet-only mode for an env-driven distributed-Postgres run.
    _write_run_marker(
        output_dir,
        resolved_spec,
        storage_url=storage_url,
        images_dir=images_dir,
        slurm=slurm,
    )

    if slurm:
        return _submit_slurm_fleet(
            resolved_spec,
            output_dir,
            storage_url=storage_url,
            spec_path=spec_path,
            images_dir=images_dir,
            n_workers=n_workers,
            slurm_partition=slurm_partition,
            slurm_mem=slurm_mem,
            slurm_time=slurm_time,
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
    n_workers: Optional[int] = None,
    slurm_partition: Optional[str] = None,
    slurm_mem: Optional[str] = None,
    slurm_time: Optional[str] = None,
) -> Optional[Trial]:
    """Submit a distributed worker fleet via :class:`SlurmExecutor`.

    The shared study URL is the explicit ``storage_url``, the
    ``$PHENOTYPIC_TUNE_STORAGE_URL`` env fallback, or the run's ``study.db``.
    Fire-and-forget: the fleet writes into the shared study; the final
    ``trials.parquet`` export happens on a later ``--recompile`` finalize.

    Args:
        spec: The resolved tuning spec.
        output_dir: The run directory.
        storage_url: The explicit Optuna storage URL, or ``None`` (resolved via
            the 3-way fallback).
        spec_path: The on-disk spec path (required; each worker reloads it).
        images_dir: The calibration image directory (required; each worker scans
            it).
        n_workers: The fleet size; ``None`` falls back to ``min(8, n_trials)``
            (or ``4`` when the strategy carries no positive ``n_trials``).
        slurm_partition: The SLURM partition; ``None`` omits the
            ``#SBATCH --partition`` directive (the cluster default).
        slurm_mem: The per-worker ``--mem`` (e.g. ``"8G"``); ``None`` omits it.
        slurm_time: The per-worker ``--time`` limit (e.g. ``"04:00:00"``);
            ``None`` omits it.
    """
    if spec_path is None or images_dir is None:
        raise ValueError(
            "--slurm requires the on-disk spec path and image directory "
            "(each worker reloads them)"
        )
    url = _resolve_storage_url(storage_url, output_dir)
    n_trials = getattr(spec.strategy, "n_trials", None)
    default_workers = (
        min(8, n_trials) if isinstance(n_trials, int) and n_trials > 0 else 4
    )
    resolved_workers = n_workers if n_workers is not None else default_workers
    # Pre-create the shared study (and its RDB schema) in THIS process BEFORE the
    # fleet starts. A cold Postgres DB has no Optuna schema, so N workers opening
    # it simultaneously race to CREATE TYPE studydirection / the trial tables and
    # all but one crash with a duplicate-key UniqueViolation. Materializing the
    # study once here (Optuna's documented distributed pattern) means every worker
    # finds an existing study and only reads/appends trials. The directions match
    # what each worker's engine infers from the scorer, so the load_if_exists open
    # never conflicts.
    from .._study._optuna_store import OptunaStudyStore

    directions = objective_directions(spec.scorer)
    OptunaStudyStore(storage_url=url, study_name=_STUDY_NAME, directions=directions)
    # Each worker launches with the submitting process's own venv interpreter
    # (the absolute sys.executable, shared across the cluster filesystem) — bare
    # ``python`` on a fresh compute node would not resolve phenotypic/optuna. This
    # reuses the forward CLI's SLURM interpreter resolution for parity.
    from phenotypic._cli._cli_utils import get_python_command

    python_command, _ = get_python_command(for_slurm=True)
    # Build the #SBATCH passthrough from the explicit flags only — an unset flag
    # is OMITTED so format_sbatch_directives emits no directive for it (the
    # cluster default applies). Notably ``slurm_partition`` is no longer hardcoded
    # to "batch": a cluster without a "batch" partition would reject every job.
    slurm_args: dict[str, Any] = {}
    if slurm_partition is not None:
        slurm_args["slurm_partition"] = slurm_partition
    if slurm_mem is not None:
        slurm_args["slurm_mem"] = slurm_mem
    if slurm_time is not None:
        slurm_args["slurm_time"] = slurm_time
    # Workers reload the RESOLVED spec persisted to deliverables/tuning_spec.json
    # (written above), NOT the raw input ``spec_path``: the --strategy / --n-trials
    # / --held-out overrides live only in the resolved spec, so handing the workers
    # the input file would silently run the original (e.g. grid) strategy on every
    # node and the distributed Optuna study would never form.
    executor = SlurmExecutor(
        output_dir=output_dir,
        spec_path=io.tuning_spec_path(output_dir),
        images_dir=Path(images_dir),
        study_name=_STUDY_NAME,
        n_workers=resolved_workers,
        slurm_args=slurm_args,
        storage_url=url,
        python_command=python_command,
    )
    executor.run(lambda w: w, list(range(resolved_workers)))
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
    atomic_write_text(
        io.generalization_path(output_dir), json.dumps(report.to_dict(), indent=2)
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
    atomic_write_text(
        io.param_importance_path(output_dir),
        json.dumps(compute_param_importance(store), indent=2),
    )
    if winner_pipeline is not None:
        atomic_write_text(
            io.best_pipeline_path(output_dir), winner_pipeline.to_json() or ""
        )


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
            atomic_write_text(
                io.pareto_best_pipeline_path(output_dir, name),
                pipeline.to_json() or "",
            )
        atomic_write_text(
            io.pareto_importance_path(output_dir, name),
            json.dumps(compute_param_importance(store, objective=name), indent=2),
        )

    # The knee is the run's headline winner: overwrite best_pipeline.json.
    knee = store.knee_point(front)
    if knee is not None:
        knee_pipeline = build_pipeline(spec.pipeline, knee.params)
        atomic_write_text(
            io.best_pipeline_path(output_dir), knee_pipeline.to_json() or ""
        )
