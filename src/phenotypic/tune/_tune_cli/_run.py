"""Run-a-tuning-spec orchestration + the ``deliverables/`` writes.

The ``run`` subcommand's body (``__main__`` parses the flags and forwards them
here). Beyond the Phase-1 local engine run, this resolves the ``--strategy`` flag
into the right :class:`~phenotypic.tune.strategy.StrategyConfig` (grid/random → the Phase-1
configs; an Optuna sampler → :class:`~phenotypic.tune.strategy.OptunaConfig`), selects the
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
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Optional
from urllib.parse import urlsplit

from phenotypic import GridImage
from phenotypic._execution._slurm import SlurmExecutor
from phenotypic.sdk_ import _io_constants as io
from phenotypic.sdk_ import atomic_write_text

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
    objective_names,
    reject_grid_random_multi_objective,
    validate_objective_axes,
)
from .._screening import compute_param_importance
from .._screening_freeze import ScreeningConfig, ScreeningController
from .._spec import TuningSpec
from ..strategy._config import (
    OPTUNA_SAMPLERS,
    PHENOTYPIC_TUNE_STORAGE_URL_ENV,
    GridConfig,
    OptunaConfig,
    RandomConfig,
    StrategyConfig,
)
from .._study._protocol import StudyStore
from .._study._storage import is_sqlite_url as _is_sqlite_url, journal_url_for_path
from .._study_store import JournalStudyStore, Trial

#: Ordinary image files ``GridImage.imread`` decodes. ``.h5`` used to be
#: listed here, but ``imread`` never accepted it (``IO.ACCEPTED_FILE_EXTENSIONS``
#: has no HDF entry), so every ``.h5`` was silently counted as "unreadable"
#: and skipped. Processed images now arrive as ``*.ome.zarr`` **directories**
#: and are loaded through :func:`load_image_from_store`, below.
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

#: Default trial budget when ``--n-trials`` is omitted for a ``random`` or Optuna
#: strategy (grid is exhaustive and ignores it).
_DEFAULT_N_TRIALS: Final[int] = 50

#: Schema version of the ``.pht-tune-cache/run.json`` marker (bump on any key
#: change so a reader can branch on the contract).
_RUN_MARKER_VERSION: Final[int] = 3

#: The study name every tune run uses (the Optuna ``study_name`` + the marker's
#: ``study_name`` field). A single constant keeps the store, the SLURM fleet, and
#: the marker in lockstep. **Bumped from ``"tune"`` for the minimize-cost cutover
#: (spec §7 Phase 2, OQ7):** new code only ever opens this study, so a pre-cutover
#: ``"tune"`` (maximize) study is never reopened — the silent-maximize hazard is
#: impossible by construction, not contingent on a runtime guard.
_STUDY_NAME: Final[str] = "tune_cost_v1"

_SLURM_SUGAR_KEYS: Final[frozenset[str]] = frozenset(
    {"partition", "mem", "time", "constraint"}
)


def _default_study_db_url(output_dir: Path) -> str:
    """SQLite URL for the run's ``study.db``, resuming a legacy-root copy.

    Resolves the study-DB location via :func:`resolve_study_db_path` (the hidden
    tune cache, falling back to a legacy output-root ``study.db`` so an
    in-flight legacy run resumes in place). A fresh run gets the new cache
    location (the resolver's no-file default).
    """
    return f"sqlite:///{io.resolve_study_db_path(output_dir)}"


def _default_journal_url(output_dir: Path) -> str:
    """Return the absolute run-local journal URL for a Slurm fleet."""
    return journal_url_for_path(
        io.tune_cache_journal_path(Path(output_dir).absolute())
    )


@dataclass(frozen=True)
class _ResolvedRunConfig:
    """The side-effect-free runtime config resolved before any run writes."""

    spec: TuningSpec
    storage_url: Optional[str]
    is_optuna: bool


def _resolve_storage_url(
    storage_url: Optional[str],
    output_dir: Path,
    *,
    spec_storage_url: Optional[str] = None,
    slurm: bool = False,
) -> str:
    """Resolve the Optuna storage URL with the canonical 4-way fallback.

    ``storage_url`` (the explicit ``--storage-url`` / param) wins; otherwise the
    spec's ``OptunaConfig.storage_url`` wins; otherwise the
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
        or spec_storage_url
        or os.environ.get(PHENOTYPIC_TUNE_STORAGE_URL_ENV)
        or (_default_journal_url(output_dir) if slurm else _default_study_db_url(output_dir))
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


def _effective_terminal_budget(spec: TuningSpec) -> Optional[int]:
    """Return the first terminal-trial cap the engine can reach."""
    budget = getattr(spec, "budget", None)
    caps = (
        getattr(spec.strategy, "n_trials", None),
        getattr(budget, "n_trials", None),
    )
    defined = [int(cap) for cap in caps if cap is not None]
    return min(defined) if defined else None


def _write_run_marker(
    output_dir: Path,
    spec: TuningSpec,
    *,
    storage_url: Optional[str],
    images_dir: Optional[Path],
    slurm: bool,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    generation: Optional[str] = None,
    required: bool = False,
) -> None:
    """Write the ``.pht-tune-cache/run.json`` marker at run START.

    Local runs emit after creating ``deliverables/`` and before optimization.
    Slurm runs emit only after lifecycle ownership and shared-study preparation,
    inside the generation publication guard. The caller passes the already
    resolved Optuna URL for Optuna runs and ``None`` for grid/random runs, so a
    live run remains GUI-discoverable and its marker names the actual backend.

    Args:
        output_dir: The run output directory.
        spec: The RESOLVED tuning spec (its ``strategy`` + ``scorer`` populate
            the marker).
        storage_url: The resolved Optuna storage URL, or ``None`` for
            non-Optuna strategies.
        images_dir: The calibration image directory (the ``-i`` arg), or ``None``.
        slurm: Whether this run submits a distributed worker fleet.
        nrows: Fixed calibration grid row count, or ``None``.
        ncols: Fixed calibration grid column count, or ``None``.
        generation: Exact Slurm lifecycle generation owning this marker.
        required: Re-raise write failures for a load-bearing Slurm marker.
    """
    marker = {
        "version": _RUN_MARKER_VERSION,
        "study_name": _STUDY_NAME,
        "storage_url": storage_url,
        "images_dir": (
            str(Path(images_dir).resolve()) if images_dir is not None else None
        ),
        "nrows": nrows,
        "ncols": ncols,
        "strategy": _strategy_marker_name(spec.strategy),
        "n_trials": _effective_terminal_budget(spec),
        "is_multi_objective": is_multi_objective(spec.scorer),
        "slurm": slurm,
        "start_time": datetime.now(timezone.utc).isoformat(),
    }
    if generation is not None:
        marker["generation"] = generation
    marker_path = io.tune_cache_run_marker_path(output_dir)
    # The marker is a GUI-discovery SIDECAR, not a deliverable: a read-only /
    # over-quota output FS (the HPCC reality) raising OSError here must NOT abort
    # the run before it starts. Catch-and-warn and proceed — mirrors the lost-race
    # tolerance in ``migrate_legacy_machine_state``.
    try:
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(marker_path, json.dumps(marker, indent=2))
    except OSError:
        if required:
            raise
        logging.getLogger(__name__).warning(
            "could not write the tune run.json marker at %s; the run proceeds "
            "but the GUI may not auto-discover it",
            marker_path,
            exc_info=True,
        )


def _load_images(
    input_dir: Path,
    *,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
) -> list:
    """Load every image under ``input_dir`` as a ``GridImage``.

    Mirrors the forward CLI's directory scan; tuning targets arrayed plates, so
    images load as ``GridImage``. Ordinary image files go through ``imread``;
    ``*.ome.zarr`` **store directories** written by a previous forward run go
    through :func:`load_image_from_store`, so a tuning run can be pointed
    straight at ``results/<dataset>/zarr/``. Unreadable / non-grid entries are
    skipped (warned) rather than aborting the whole run.

    Args:
        input_dir: The directory to scan (non-recursive).
        nrows: Optional fixed grid row count passed to ``GridImage.imread`` so
            every calibration plate carries a known uniform ``nrows × ncols``
            grid (required for grid-cell-aware scoring, e.g. an
            ``ExpectedVsDetectedCount`` grouped by ``Grid_RowNum``/``Grid_ColNum``).
            When ``None`` the imread default grid is used.
        ncols: Optional fixed grid column count (see ``nrows``). Both must be
            given together to take effect.

    Returns:
        The loaded ``GridImage`` instances, in sorted filename order.
    """
    from phenotypic.sdk_ import STORE_SUFFIX, load_image_from_store

    def _is_store(path: Path) -> bool:
        # A store is a DIRECTORY, and its `.part`/`.trash` siblings are
        # dotted -- the same two tests the CLI's own store scan makes.
        return (
            path.is_dir()
            and path.name.endswith(STORE_SUFFIX)
            and not path.name.startswith(".")
        )

    paths = sorted(
        p for p in Path(input_dir).iterdir()
        if (p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES)
        or _is_store(p)
    )
    grid_kwargs: dict[str, int] = {}
    if nrows is not None and ncols is not None:
        grid_kwargs = {"nrows": nrows, "ncols": ncols}
    images: list = []
    failures: list[tuple[str, str]] = []
    for path in paths:
        try:
            if _is_store(path):
                # A store already records its own grid state; the
                # nrows/ncols imread overrides do not apply to it.
                images.append(
                    load_image_from_store(path, fallback="GridImage")
                )
            else:
                images.append(GridImage.imread(path, **grid_kwargs))
        except Exception as exc:  # skip unreadable / non-grid entries, don't abort
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
    ``uv sync --extra tune``) rather than a bare ``KeyError`` deep in ``build``.

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
        if n_trials is not None:
            raise ValueError("--n-trials requires a random or Optuna strategy, not grid")
        return GridConfig()
    if name == "random":
        return RandomConfig(
            n_trials=n_trials if n_trials is not None else _DEFAULT_N_TRIALS
        )
    if name in OPTUNA_SAMPLERS:
        # Fail fast + actionable when the extra is missing, before constructing a
        # config the engine could not build.
        from ..strategy import _optuna_support

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


def _strategy_with_n_trials(strategy: StrategyConfig, n_trials: int) -> StrategyConfig:
    """Return ``strategy`` with a CLI ``--n-trials`` override applied.

    Args:
        strategy: The already-resolved strategy config.
        n_trials: The CLI trial-budget override.

    Returns:
        A copied strategy with ``n_trials`` updated.

    Raises:
        ValueError: If the strategy has no trial-budget field.
    """
    if isinstance(strategy, GridConfig):
        raise ValueError("--n-trials requires a random or Optuna strategy, not grid")
    if isinstance(strategy, (RandomConfig, OptunaConfig)):
        return strategy.model_copy(update={"n_trials": n_trials})
    if hasattr(strategy, "n_trials"):
        return strategy.model_copy(update={"n_trials": n_trials})
    raise ValueError(
        f"--n-trials is not supported for {type(strategy).__name__}"
    )


def _resolve_run_config(
    spec: TuningSpec,
    output_dir: Path,
    *,
    strategy: Optional[str],
    n_trials: Optional[int],
    storage_url: Optional[str],
    held_out_fraction: Optional[float],
    cv_group: Optional[str],
    slurm: bool = False,
) -> _ResolvedRunConfig:
    """Resolve CLI overrides and storage policy before any run side effects."""
    # Reject explicit secrets before an Optuna strategy override imports optuna.
    if storage_url is not None:
        _reject_password_in_url(storage_url)
    spec_storage_url = getattr(spec.strategy, "storage_url", None)
    if storage_url is None and spec_storage_url is not None:
        _reject_password_in_url(spec_storage_url)

    resolved_spec = spec
    if strategy is not None:
        resolved_strategy = resolve_strategy(
            strategy,
            n_trials=n_trials,
            storage_url=storage_url or spec_storage_url,
        )
        resolved_spec = spec.model_copy(update={"strategy": resolved_strategy})
    elif n_trials is not None:
        resolved_spec = spec.model_copy(
            update={"strategy": _strategy_with_n_trials(spec.strategy, n_trials)}
        )

    reject_grid_random_multi_objective(
        resolved_spec.scorer, resolved_spec.strategy
    )
    resolved_spec = _apply_held_out_overrides(
        resolved_spec, held_out_fraction=held_out_fraction, cv_group=cv_group
    )

    is_optuna = _is_optuna_strategy(resolved_spec.strategy)
    effective_storage_url: Optional[str] = None
    if is_optuna:
        spec_storage_url = getattr(resolved_spec.strategy, "storage_url", None)
        effective_storage_url = _resolve_storage_url(
            storage_url,
            output_dir,
            spec_storage_url=spec_storage_url,
            slurm=slurm,
        )
        resolved_spec = resolved_spec.model_copy(
            update={
                "strategy": resolved_spec.strategy.model_copy(
                    update={"storage_url": effective_storage_url}
                )
            }
        )

    return _ResolvedRunConfig(
        spec=resolved_spec,
        storage_url=effective_storage_url,
        is_optuna=is_optuna,
    )


def _assert_scorer_available(spec: TuningSpec) -> None:
    """Fail before optimization when the scorer cannot safely run."""
    if spec.scorer.availability():
        return
    cls_name = type(spec.scorer).__name__
    extra = (
        " Run ReferenceFreeScorer.meta_validate() successfully before "
        "unattended reference-free tuning."
        if cls_name == "ReferenceFreeScorer"
        else ""
    )
    raise ValueError(f"{cls_name} is unavailable for tuning.{extra}")


def _open_store(
    strategy: StrategyConfig,
    output_dir: Path,
    *,
    storage_url: Optional[str],
    resume_path: Path,
    directions: Optional[list[str]] = None,
    objective_axes: Optional[Sequence[str]] = None,
    study_name: str = _STUDY_NAME,
    create: bool = True,
) -> StudyStore:
    """Select + open the study backend matching ``strategy``.

    An Optuna strategy gets a resumable :class:`OptunaStudyStore` on the shared
    ``study.db`` (or the explicit ``storage_url``); any other strategy gets the
    homegrown :class:`JournalStudyStore` (resumed from ``trials.parquet`` when one
    already exists). A multi-objective run passes ``directions`` so the Optuna
    store opens a multi-objective study. Its ``append`` records the per-objective
    vector, and ``pareto_front`` applies the shared COMPLETE-only selector over
    terminal trials.

    Args:
        strategy: The resolved strategy config.
        output_dir: The run directory (for ``study.db`` placement).
        storage_url: An explicit Optuna storage URL; ``None`` resolves via the
            3-way fallback (env var > local ``study.db``) in
            :func:`_resolve_storage_url`.
        resume_path: The ``trials.parquet`` path the journal resumes from.
        directions: Per-objective ``["minimize"] * n`` for a multi-objective run
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
        url = _resolve_storage_url(
            storage_url,
            output_dir,
            spec_storage_url=getattr(strategy, "storage_url", None),
        )
        store_kwargs: dict[str, Any] = {
            "storage_url": url,
            "study_name": study_name,
            "directions": directions,
        }
        if objective_axes is not None:
            store_kwargs["objective_axes"] = tuple(objective_axes)
        if not create:
            store_kwargs["create"] = False
        return OptunaStudyStore(**store_kwargs)
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
    slurm_constraint: Optional[str] = None,
    slurm_args: Optional[dict[str, Any]] = None,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
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

    resolved = _resolve_run_config(
        spec,
        output_dir,
        strategy=strategy,
        n_trials=n_trials,
        storage_url=storage_url,
        held_out_fraction=held_out_fraction,
        cv_group=cv_group,
        slurm=slurm,
    )
    resolved_spec = resolved.spec
    effective_storage_url = resolved.storage_url
    _assert_scorer_available(resolved_spec)

    if slurm:
        _validate_slurm_request(
            resolved,
            spec_path=spec_path,
            images_dir=images_dir,
            n_workers=n_workers,
            screen=screen,
        )

    if slurm:
        from phenotypic._cli._cli_slurm_lifecycle import (
            generation_publication_guard,
            initialize_slurm_lifecycle,
            mark_generation_failed,
            new_slurm_generation,
        )

        assert effective_storage_url is not None
        generation = new_slurm_generation()
        # The ownership claim must be the first shared mutation. A conflicting
        # rerun therefore cannot rewrite the incumbent split/spec/marker or open
        # the incumbent study through a potentially mutating constructor.
        initialize_slurm_lifecycle(
            output_dir,
            generation=generation,
            mode="tune",
        )
        try:
            with generation_publication_guard(output_dir, generation):
                split, images_by_name, cal_images = _resolve_calibration_images(
                    resolved_spec, images, output_dir
                )
                io.deliverables_dir(output_dir).mkdir(parents=True, exist_ok=True)
                atomic_write_text(
                    io.tuning_spec_path(output_dir),
                    resolved_spec.model_dump_json(indent=2),
                )
                _precreate_shared_optuna_study(
                    resolved_spec, effective_storage_url
                )
                _write_run_marker(
                    output_dir,
                    resolved_spec,
                    storage_url=effective_storage_url,
                    images_dir=images_dir,
                    slurm=True,
                    nrows=nrows,
                    ncols=ncols,
                    generation=generation,
                    required=True,
                )
        except Exception as exc:
            mark_generation_failed(output_dir, generation, str(exc))
            raise
        try:
            return _submit_slurm_fleet(
                resolved_spec,
                output_dir,
                storage_url=effective_storage_url,
                spec_path=spec_path,
                images_dir=images_dir,
                split_path=io.tune_cache_split_assignment_path(output_dir),
                n_workers=n_workers,
                slurm_args=merge_slurm_args(
                    slurm_args,
                    partition=slurm_partition,
                    mem=slurm_mem,
                    time=slurm_time,
                    constraint=slurm_constraint,
                ),
                nrows=nrows,
                ncols=ncols,
                generation=generation,
                shared_state_prepared=True,
            )
        except Exception as exc:
            mark_generation_failed(output_dir, generation, str(exc))
            raise

    split, images_by_name, cal_images = _resolve_calibration_images(
        resolved_spec, images, output_dir
    )

    io.deliverables_dir(output_dir).mkdir(parents=True, exist_ok=True)

    # Always echo the resolved spec so the deliverable is re-runnable.
    atomic_write_text(
        io.tuning_spec_path(output_dir), resolved_spec.model_dump_json(indent=2)
    )

    # Mark the run as a tune output at START so a live local run is
    # GUI-discoverable before any deliverable lands.
    _write_run_marker(
        output_dir,
        resolved_spec,
        storage_url=effective_storage_url,
        images_dir=images_dir,
        slurm=False,
        nrows=nrows,
        ncols=ncols,
    )

    trials_path = io.trials_parquet_path(output_dir)
    # Multi-objective is inferred from the scorer (plan §0b): the directions feed
    # both the Optuna store (Pareto front) and, via the engine, the NSGA-II study.
    directions = objective_directions(resolved_spec.scorer)
    store_axes = (
        tuple(objective_names(resolved_spec.scorer))
        if is_multi_objective(resolved_spec.scorer)
        else None
    )
    store = _open_store(
        resolved_spec.strategy,
        output_dir,
        storage_url=effective_storage_url,
        resume_path=trials_path,
        directions=directions,
        objective_axes=store_axes,
    )

    if screen:
        best = _run_screened(resolved_spec, cal_images, store)
    else:
        engine = TuningEngine(resolved_spec, store=store)
        best = engine.optimize(cal_images)
    multi_objective_run = is_multi_objective(resolved_spec.scorer)
    objective_axes = (
        tuple(objective_names(resolved_spec.scorer))
        if multi_objective_run
        else None
    )
    headline = _headline_winner(
        store,
        multi_objective=multi_objective_run,
        objective_axes=objective_axes,
    )
    if multi_objective_run and headline is None:
        raise RuntimeError(
            "the terminal study has no valid winner; refusing publication"
        )
    winner_pipeline = _pipeline_for_trial(resolved_spec, headline)

    _finalize_outputs(store, trials_path, output_dir, winner_pipeline)
    # Multi-objective runs additionally publish deliverables/pareto/ (front +
    # per-objective best pipelines) and overwrite best_pipeline.json with the
    # knee; a single-objective run's empty front makes this a no-op (no pareto/
    # dir — the back-compat lock, plan §0b).
    _finalize_pareto_outputs(store, resolved_spec, output_dir, objective_axes)
    _finalize_best_params(
        headline,
        output_dir,
        selection=_selection_label(
            store,
            multi_objective=multi_objective_run,
            objective_axes=objective_axes,
        ),
    )
    # Report-only held-out generalization verdict → deliverables/generalization.json.
    _finalize_generalization(
        headline, resolved_spec, output_dir, split, images, images_by_name
    )
    return headline if multi_objective_run else best


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


def _validate_slurm_request(
    resolved: _ResolvedRunConfig,
    *,
    spec_path: Optional[Path],
    images_dir: Optional[Path],
    n_workers: Optional[int],
    screen: bool = False,
) -> None:
    """Reject unsupported SLURM combinations before any run artifact is written."""
    if screen:
        raise ValueError("--screen is not supported with --slurm")
    if not resolved.is_optuna:
        raise ValueError("--slurm requires an Optuna strategy")
    if _is_sqlite_url(resolved.storage_url):
        raise ValueError("--slurm cannot use a SQLite storage URL")
    if spec_path is None or images_dir is None:
        raise ValueError(
            "--slurm requires the on-disk spec path and image directory "
            "(each worker reloads them)"
        )
    if n_workers is not None and n_workers <= 0:
        raise ValueError("--n-workers must be a positive integer")


def _canonical_slurm_argument(key: str, value: Any) -> tuple[str, Any]:
    """Collapse aliases onto the same key used by SBATCH directive rendering."""
    if key == "mem_gb":
        return "slurm_mem", f"{value}G"
    if key.startswith("slurm_"):
        return key, value
    return f"slurm_{key}", value


def merge_slurm_args(
    explicit: Optional[dict[str, Any]],
    *,
    partition: Optional[str],
    mem: Optional[str],
    time: Optional[str],
    constraint: Optional[str],
) -> dict[str, Any]:
    """Merge legacy Slurm flags with explicit pairs; explicit values win."""
    merged: dict[str, Any] = {}
    for name, value in (
        ("partition", partition),
        ("mem", mem),
        ("time", time),
        ("constraint", constraint),
    ):
        if value is not None:
            merged[f"slurm_{name}"] = value
    for key, value in (explicit or {}).items():
        canonical_key, canonical_value = _canonical_slurm_argument(str(key), value)
        merged[canonical_key] = canonical_value
    return merged


def _store_has_multi_objective_history(store: StudyStore) -> bool:
    """Infer multi-objective identity from sidecars for direct private callers."""
    return any(
        getattr(trial, "objectives", None) is not None for trial in store.trials
    )


def _headline_winner(
    store: StudyStore,
    *,
    multi_objective: Optional[bool] = None,
    objective_axes: Sequence[str] | None = None,
) -> Optional[Trial]:
    """Return a headline winner without scalar or partial-vector fallback."""
    if objective_axes is not None:
        objective_axes = validate_objective_axes(objective_axes)
    resolved_multi_objective = (
        True
        if objective_axes is not None
        else (
            _store_has_multi_objective_history(store)
            if multi_objective is None
            else multi_objective
        )
    )
    if resolved_multi_objective:
        if objective_axes is None:
            front = store.pareto_front()
            return store.knee_point(front) if front else None
        front = store.pareto_front(objective_axes=objective_axes)
        return (
            store.knee_point(front, objective_axes=objective_axes)
            if front
            else None
        )
    return store.best()


def _selection_label(
    store: StudyStore,
    *,
    multi_objective: Optional[bool] = None,
    objective_axes: Sequence[str] | None = None,
) -> str:
    """Return the selection label from identity, not front cardinality."""
    if objective_axes is not None:
        objective_axes = validate_objective_axes(objective_axes)
    resolved_multi_objective = (
        True
        if objective_axes is not None
        else (
            _store_has_multi_objective_history(store)
            if multi_objective is None
            else multi_objective
        )
    )
    return "pareto_knee" if resolved_multi_objective else "single_best"


def _pipeline_for_trial(spec: TuningSpec, trial: Optional[Trial]):
    """Build a candidate pipeline for ``trial`` or ``None`` when no winner exists."""
    from .._evaluation import build_pipeline

    if trial is None:
        return None
    return build_pipeline(spec.pipeline, trial.params)


def _finalize_best_params(
    winner: Optional[Trial],
    output_dir: Path,
    *,
    selection: str,
) -> None:
    """Write the Monitor-facing params sidecar for the selected headline trial."""
    if winner is None:
        return
    payload = {
        "trial_number": winner.number,
        "score": winner.score,
        "objectives": winner.objectives or {},
        "params": winner.params,
        "selection": selection,
    }
    atomic_write_text(io.best_params_path(output_dir), json.dumps(payload, indent=2))


def _precreate_shared_optuna_study(spec: TuningSpec, storage_url: str) -> None:
    """Create the shared Optuna study once after lifecycle ownership is held.

    A cold Postgres database cannot safely be initialized concurrently by every
    worker. The submitter materializes its schema and study under the generation
    publication guard, so workers only open and append to an existing study.
    """
    from .._study._optuna_store import OptunaStudyStore

    OptunaStudyStore(
        storage_url=storage_url,
        study_name=_STUDY_NAME,
        directions=objective_directions(spec.scorer),
        objective_axes=(
            tuple(objective_names(spec.scorer))
            if is_multi_objective(spec.scorer)
            else None
        ),
    )


def _submit_slurm_fleet(
    spec: TuningSpec,
    output_dir: Path,
    *,
    storage_url: str,
    spec_path: Optional[Path],
    images_dir: Optional[Path],
    split_path: Path,
    n_workers: Optional[int] = None,
    slurm_args: Optional[dict[str, Any]] = None,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    generation: Optional[str] = None,
    shared_state_prepared: bool = False,
) -> Optional[Trial]:
    """Submit a distributed worker fleet via :class:`SlurmExecutor`.

    The shared study URL is already resolved by run preflight. Fire-and-forget:
    the fleet writes into the shared study; the final ``trials.parquet`` export
    happens on a later ``--recompile`` finalize.

    Args:
        spec: The resolved tuning spec.
        output_dir: The run directory.
        storage_url: The resolved Optuna storage URL.
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
    from phenotypic._cli._cli_slurm_lifecycle import (
        generation_publication_guard,
        initialize_slurm_lifecycle,
        mark_generation_failed,
        new_slurm_generation,
    )

    try:
        if spec_path is None or images_dir is None:
            raise ValueError(
                "--slurm requires the on-disk spec path and image directory "
                "(each worker reloads them)"
            )
        url = storage_url
        n_trials = getattr(spec.strategy, "n_trials", None)
        default_workers = (
            min(8, n_trials) if isinstance(n_trials, int) and n_trials > 0 else 4
        )
        resolved_workers = n_workers if n_workers is not None else default_workers
        if resolved_workers <= 0:
            raise ValueError("--n-workers must be a positive integer")
        # Each worker launches with the submitting process's own venv interpreter
        # (the absolute sys.executable, shared across the cluster filesystem) —
        # bare ``python`` on a fresh compute node would not resolve
        # phenotypic/optuna. This reuses the forward CLI's SLURM interpreter
        # resolution for parity.
        from phenotypic._cli._cli_utils import get_python_command

        python_command, _ = get_python_command(for_slurm=True)
        # Workers reload the RESOLVED spec persisted to
        # deliverables/tuning_spec.json (written above), NOT the raw input
        # ``spec_path``: CLI overrides live only in the resolved spec.
        if generation is None:
            generation = new_slurm_generation()
            initialize_slurm_lifecycle(
                output_dir,
                generation=generation,
                mode="tune",
            )
        if not shared_state_prepared:
            with generation_publication_guard(output_dir, generation):
                _precreate_shared_optuna_study(spec, url)
                _write_run_marker(
                    output_dir,
                    spec,
                    storage_url=url,
                    images_dir=Path(images_dir),
                    slurm=True,
                    nrows=nrows,
                    ncols=ncols,
                    generation=generation,
                    required=True,
                )
        executor = SlurmExecutor(
            output_dir=output_dir,
            spec_path=io.tuning_spec_path(output_dir),
            images_dir=Path(images_dir),
            split_path=Path(split_path),
            study_name=_STUDY_NAME,
            n_workers=resolved_workers,
            slurm_args=dict(slurm_args or {}),
            storage_url=url,
            python_command=python_command,
            nrows=nrows,
            ncols=ncols,
            lifecycle_generation=generation,
        )
        executor.run(lambda w: w, list(range(resolved_workers)))
    except Exception as exc:
        if generation is not None:
            mark_generation_failed(output_dir, generation, str(exc))
        raise
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
    winner: Optional[Trial],
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
        winner: The headline winner selected from the finished study.
        spec: The resolved tuning spec (its ``evaluator`` runs the held-out pass).
        output_dir: The run directory.
        split: The resolved held-out split.
        images: The loaded plates (for the current dataset identity).
        images_by_name: ``{name: image}`` of the loaded plates.
    """
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
    mirror = JournalStudyStore(store.terminal_trials())
    mirror.to_parquet(trials_path)


def _finalize_pareto_outputs(
    store: StudyStore,
    spec: TuningSpec,
    output_dir: Path,
    objective_axes: Sequence[str] | None = None,
) -> None:
    """Publish ``deliverables/pareto/`` when the run was multi-objective.

    A multi-objective run (a scorer whose ``finalize`` returns a dict, so trials
    carry ``Trial.objectives``) can have a non-empty
    :meth:`StudyStore.pareto_front`. An empty front is a **no-op** here; callers
    use scorer identity to distinguish a scalar run from an invalid
    multi-objective run and refuse the latter before invoking publishers. When
    the front is non-empty this writes, under :func:`pareto_dir`:

    * ``pareto_front.parquet`` — the front's trials (same schema as
      ``trials.parquet``, ``objectives_json`` populated);
    * ``best_<objective>.json`` — the front pipeline minimizing each objective
      axis's cost (lowest-cost trial per axis), plus that axis's
      :func:`compute_param_importance`;
    * and it overwrites the top-level ``best_pipeline.json`` with the **knee**
      (the max-curvature compromise pick).

    Args:
        store: The finished study store (any backend).
        spec: The resolved tuning spec (its ``pipeline`` is the build base).
        output_dir: The run directory.
        objective_axes: The authoritative scorer-required objective names.
            Direct legacy callers may omit them only when the spec has no
            multi-objective scorer contract.
    """
    from .._evaluation import build_pipeline

    if objective_axes is None and is_multi_objective(spec.scorer):
        resolved_axes: tuple[str, ...] | None = tuple(
            objective_names(spec.scorer)
        )
    else:
        resolved_axes = (
            validate_objective_axes(objective_axes)
            if objective_axes is not None
            else None
        )
    front = (
        store.pareto_front()
        if resolved_axes is None
        else store.pareto_front(objective_axes=resolved_axes)
    )
    if not front:
        return  # No eligible front; the caller owns scalar-vs-multi identity.

    pareto_dir = io.pareto_dir(output_dir)
    pareto_dir.mkdir(parents=True, exist_ok=True)

    # The front parquet (mirror the front into a journal for a uniform schema).
    JournalStudyStore(list(front)).to_parquet(io.pareto_front_parquet_path(output_dir))

    # One best pipeline + importance per objective axis (stable name order).
    # Source the axis order from the scorer (authoritative — every axis the
    # study optimized); only legacy scalar-spec helpers infer sidecar names.
    axes = resolved_axes or tuple(front[0].objectives or {})
    for name in axes:
        winner = min(
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
            json.dumps(
                compute_param_importance(
                    store,
                    objective=name,
                    objective_axes=resolved_axes,
                ),
                indent=2,
            ),
        )

    # The knee is the run's headline winner: overwrite best_pipeline.json.
    knee = (
        store.knee_point(front)
        if resolved_axes is None
        else store.knee_point(front, objective_axes=resolved_axes)
    )
    if knee is not None:
        knee_pipeline = build_pipeline(spec.pipeline, knee.params)
        atomic_write_text(
            io.best_pipeline_path(output_dir), knee_pipeline.to_json() or ""
        )
