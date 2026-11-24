"""``--embed-stage``: where a run's embeddings are computed (#282 spec §4.1).

A run embeds either **inline**, in the per-image core (Stage 3, or the
single-pass core), or in a **GPU Stage 4** after the store is final. The
user asks with ``--embed-stage {auto,gpu,inline}`` (default ``auto``); the
CLI resolves that once, in the read-only half, with
:func:`resolve_embed_stage`, and records the answer as
``processing_state.json`` config ``"embed_stage"`` and job metadata
``EMBED_STAGE``. The value is configuration, never part of the work-id
digest; a state without it reads as ``"inline"``.

The resolution table (spec §4.1, U2, U9, U10, O5):

========================================================  ======================
Run                                                       ``auto`` resolves to
========================================================  ======================
full, SLURM, a ``GpuDetector`` or ``--gpu-slurm`` given   ``gpu``
full, SLURM, neither                                      ``inline``
full, local                                               ``inline``
``--mode measure``, SLURM with ``--gpu-slurm``            ``gpu``
``--mode measure``, otherwise                             ``inline``
``--mode process``                                        ``inline`` (R1 refuses ``gpu``)
an in-scope embedder pinned to ``device="cpu"``           ``inline``, with a note (R2 refuses ``gpu``)
an in-scope embedder with no Stage-4 encoder              ``inline``, with a note (R2 refuses ``gpu``)
no in-scope embedder                                      ``inline`` (``gpu`` too, with a note)
========================================================  ======================

The three refusals are each one :class:`click.UsageError` line (exit 2), in
the read-only half, outside ``--skip-validation`` and above ``--restart``,
``--overwrite``, ``mint_run_identity`` and ``--dry-run``:

- **R1** (:func:`refuse_embed_stage_for_process`): ``--embed-stage gpu`` with
  ``--mode process``, which stays inline on the CPU (H5).
- **R2** (:func:`refuse_unstageable_embedders`): ``--embed-stage gpu`` with
  an embedder pinned to ``device="cpu"`` (U9) or an embedder with no
  Stage-4 encoder (O3). Under ``auto`` either resolves the run inline.
- **R3** (:func:`refuse_embed_stage_switch`): a continuation whose resolved
  stage differs from the recorded one (U3).

**An interim gate** (#282 P3-P4 review C1): :func:`stage4_supported` says
which runs have a Stage-4 executor today: full runs, local or on SLURM (P5 T2
lifted SLURM). In ``--mode measure``, local or SLURM, ``auto`` resolves
``inline`` with a one-line note, and an explicit ``--embed-stage gpu`` is one
usage-error line, until P6 T1 widens that one predicate for it.

**One revision per run** (U5, #282 P5 T2): :func:`resolve_embed_revisions`
asks the hub once, in the submitting process at a Stage-4 run's first
invocation, for the commit of every unpinned ``MeasureDinoEmbedding``;
:func:`recorded_embed_revisions` reads back what an earlier invocation
recorded, and a continuation reuses it rather than resolving again.

:func:`prepare_stage4_environment` is the startup half of a ``gpu`` run: the
cuBLAS workspace variable, and for a local run the CUDA check, both before
anything could initialise CUDA.

Heavy imports (the pipeline, the embedders, torch) stay inside functions.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import click

if TYPE_CHECKING:
    from collections.abc import Mapping

    from phenotypic._core._image_pipeline import ImagePipeline
    from phenotypic.abc_ import MeasureEmbedding

    from ._cli_preflight import RunMode
    from ._cli_types import ExecutionConfig

#: A resolved stage: where the run's embeddings are computed.
EmbedStage = Literal["gpu", "inline"]
#: What ``--embed-stage`` asks for; ``auto`` is resolved by :func:`resolve_embed_stage`.
EmbedStageRequest = Literal["auto", "gpu", "inline"]

#: The ``processing_state.json`` config key holding the resolved stage.
EMBED_STAGE_CONFIG_KEY = "embed_stage"
#: What a recorded state without :data:`EMBED_STAGE_CONFIG_KEY` reads as (pre-#282).
DEFAULT_EMBED_STAGE: EmbedStage = "inline"


def pipeline_embedders(pipeline: "ImagePipeline") -> list[tuple[str, "MeasureEmbedding"]]:
    """The root ``meas`` embedders, as ``(meas key, embedder)``, in ``meas`` order.

    Only the root ``meas`` slot is in scope: a pipeline nested in ``ops``
    contributes only its own ``ops`` (``MODE_SLOTS``), so an embedder inside
    it never runs.
    """
    from phenotypic.abc_ import MeasureEmbedding

    return [
        (key, step)
        for key, step in pipeline.get_meas().items()
        if isinstance(step, MeasureEmbedding)
    ]


def _cpu_pinned_key(pipeline: "ImagePipeline") -> str | None:
    """The first in-scope embedder's ``meas`` key pinned to ``device="cpu"``, or ``None``."""
    for key, embedder in pipeline_embedders(pipeline):
        if getattr(embedder, "device", None) == "cpu":
            return key
    return None


def _has_stage4_encoder(embedder: "MeasureEmbedding") -> bool:
    """Whether Stage 4 can batch *embedder*: today only ``MeasureDinoEmbedding`` (O3)."""
    from phenotypic.measure import MeasureDinoEmbedding

    return isinstance(embedder, MeasureDinoEmbedding)


def _unstageable_embedder(pipeline: "ImagePipeline") -> tuple[str, str] | None:
    """The first in-scope embedder with no Stage-4 encoder, as ``(key, class)``, or ``None``."""
    for key, embedder in pipeline_embedders(pipeline):
        if not _has_stage4_encoder(embedder):
            return key, type(embedder).__name__
    return None


def resolve_embed_stage(
    requested: EmbedStageRequest,
    *,
    mode: "RunMode",
    is_slurm: bool,
    has_gpu_detector: bool,
    gpu_slurm_given: bool,
    has_embedders: bool,
    cpu_pinned: bool,
    unstageable: bool = False,
) -> EmbedStage:
    """Resolve ``--embed-stage`` to where the run embeds (spec §4.1).

    Args:
        requested: The ``--embed-stage`` value.
        mode: The run mode. ``process`` always resolves ``inline`` (H5;
            R1 refuses an explicit ``gpu`` before this is asked).
        is_slurm: Whether the run is submitted to SLURM.
        has_gpu_detector: Whether the pipeline holds a ``GpuDetector``.
            It counts only for ``full``; in ``measure`` only ``--gpu-slurm``
            is a GPU profile (U10).
        gpu_slurm_given: Whether ``--gpu-slurm`` was given.
        has_embedders: Whether the pipeline has an in-scope embedder. With
            none, nothing runs in Stage 4, so the run is ``inline`` (O5).
        cpu_pinned: Whether an in-scope embedder is pinned to
            ``device="cpu"``. Under ``auto`` the pin wins (U9); an explicit
            ``gpu`` resolves ``gpu`` and R2 refuses it.
        unstageable: Whether an in-scope embedder has no Stage-4 encoder
            (anything but ``MeasureDinoEmbedding``). Treated as a CPU pin:
            ``auto`` resolves ``inline``, and R2 refuses an explicit ``gpu``
            (O3, amended 2026-11-20).

    Returns:
        ``"gpu"`` or ``"inline"``.
    """
    if mode == "process" or not has_embedders or requested == "inline":
        return "inline"
    if requested == "gpu":
        return "gpu"
    if cpu_pinned or unstageable or not is_slurm:
        return "inline"
    gpu_profile = gpu_slurm_given or (mode == "full" and has_gpu_detector)
    return "gpu" if gpu_profile else "inline"


def refuse_embed_stage_for_process(requested: EmbedStageRequest, mode: "RunMode") -> None:
    """R1: refuse ``--embed-stage gpu`` with ``--mode process`` (H5).

    Raises:
        click.UsageError: ``requested`` is ``gpu`` and ``mode`` is ``process``.
    """
    if requested == "gpu" and mode == "process":
        raise click.UsageError(
            "--embed-stage gpu does not apply to --mode process: process mode "
            "embeds inline on the CPU"
        )


def refuse_unstageable_embedders(
    pipeline: "ImagePipeline", requested: EmbedStageRequest, resolved: EmbedStage
) -> None:
    """R2: refuse ``--embed-stage gpu`` over an embedder Stage 4 cannot run.

    - An in-scope embedder pinned to ``device="cpu"`` (U9).
    - A ``MeasureEmbedding`` that is not a ``MeasureDinoEmbedding``: only
      DINO has a Stage-4 encoder (O3).

    Only an explicit ``gpu`` is refused. Under ``auto`` either resolves the
    run inline instead (:func:`resolve_embed_stage`), so a SLURM detector
    pipeline carrying such an embedder keeps running as it did.

    Args:
        pipeline: The run's pipeline.
        requested: The ``--embed-stage`` value.
        resolved: What it resolved to; nothing is refused for ``inline``
            (a pipeline with no embedder resolves so, O5).

    Raises:
        click.UsageError: Naming the first such ``meas`` key and its class.
    """
    if requested != "gpu" or resolved != "gpu":
        return
    for key, embedder in pipeline_embedders(pipeline):
        cls = type(embedder).__name__
        if getattr(embedder, "device", None) == "cpu":
            raise click.UsageError(
                f"--embed-stage gpu was given, but meas '{key}' ({cls}) is "
                "pinned to device='cpu'; set device='auto' or 'cuda', or drop "
                "--embed-stage gpu to embed inline"
            )
        if not _has_stage4_encoder(embedder):
            raise click.UsageError(
                f"--embed-stage gpu was given, but meas '{key}' ({cls}) has "
                "no GPU Stage-4 encoder; pass --embed-stage inline"
            )


def refuse_embed_stage_switch(
    recorded: str | None,
    resolved: EmbedStage,
    *,
    is_slurm: bool,
    stage4_available: bool,
) -> None:
    """R3: refuse a continuation that would embed elsewhere than it started (U3).

    The remedy names only a command that can run now: ``--embed-stage
    <recorded>`` when this invocation could honour it, otherwise
    ``--restart``, which starts the run over and re-resolves its stage
    (review C1: a run recorded ``gpu`` where Stage 4 is gated cannot be
    continued as ``gpu``).

    Args:
        recorded: The state's recorded ``embed_stage``; ``None`` (a state
            written before #282) reads as ``"inline"``.
        resolved: What this invocation resolved to.
        is_slurm: Whether this invocation is a SLURM run; a ``gpu`` remedy
            then also names ``--gpu-slurm``, which ``auto`` resolves from.
        stage4_available: :func:`stage4_supported` for this invocation.

    Raises:
        click.UsageError: The two differ.
    """
    recorded_stage = DEFAULT_EMBED_STAGE if recorded is None else recorded
    if recorded_stage == resolved:
        return
    if recorded_stage == "gpu" and not stage4_available:
        remedy = (
            "Stage 4 cannot run this invocation yet, so start the run over "
            "with --restart"
        )
    else:
        remedy = f"Re-run with --embed-stage {recorded_stage}"
        if recorded_stage == "gpu" and is_slurm:
            remedy += " (on SLURM, also --gpu-slurm)"
    raise click.UsageError(
        f"this run was started with --embed-stage {recorded_stage}; continuing "
        f"it as {resolved} would mix CPU and GPU embeddings. {remedy}"
    )


def adopt_recorded_embed_stage(
    recorded: str | None,
    requested: EmbedStageRequest,
    resolved: EmbedStage,
    *,
    is_slurm: bool,
    stage4_available: bool,
    gpu_profile: bool = True,
) -> tuple[EmbedStage, str | None]:
    """The stage a continuation embeds in: the recorded one, unless a flag conflicts (U12).

    Under ``--embed-stage auto`` a continuation **adopts** the run's
    recorded ``embed_stage``, so a run never switches mid-way (U3's purpose)
    and its plain re-run command keeps working across an upgrade that changed
    what ``auto`` resolves to. That is what keeps an in-flight pre-P5 SLURM
    run, recorded ``inline`` or recording nothing (read as ``inline``),
    embedding inline (review I1, user decision U12, 2026-11-23). Only an
    explicit ``--embed-stage`` that conflicts is refused, by
    :func:`refuse_embed_stage_switch` (R3), as is an ``auto`` that would have
    to adopt ``gpu`` where no Stage 4 runs this invocation.

    Args:
        recorded: The state's recorded ``embed_stage``; ``None`` reads as
            ``"inline"``.
        requested: This invocation's ``--embed-stage``.
        resolved: What *requested* resolved to.
        is_slurm: Passed to the refusal's remedy.
        stage4_available: :func:`stage4_supported` for this invocation.
        gpu_profile: Whether this invocation has a GPU profile
            (:func:`has_slurm_gpu_profile`); consulted only for a SLURM
            ``auto`` that would adopt ``gpu``.

    Returns:
        The stage to run, and the one-line note to print when it differs
        from *resolved* (``None`` when it does not).

    Raises:
        click.UsageError: R3, for an explicit conflicting flag, or an
            ``auto`` that would adopt ``gpu`` where Stage 4 cannot run; and
            a SLURM ``auto`` that would adopt ``gpu`` with no GPU profile
            (no ``GpuDetector`` and no ``--gpu-slurm``), which would put
            Stage 4 on the CPU profile without a word.
    """
    recorded_stage: EmbedStage = (
        DEFAULT_EMBED_STAGE if recorded is None else cast(EmbedStage, recorded)
    )
    if recorded_stage == resolved:
        return resolved, None
    if requested == "auto" and (recorded_stage == "inline" or stage4_available):
        if recorded_stage == "gpu" and is_slurm and not gpu_profile:
            raise click.UsageError(
                "this run was started with --embed-stage gpu, and this "
                "continuation has no GPU profile, so its Stage 4 would be "
                "submitted on the --slurm CPU profile; re-run with the run's "
                "--gpu-slurm options"
            )
        return recorded_stage, (
            f"this run was started with --embed-stage {recorded_stage}, so "
            f"--embed-stage auto continues it as {recorded_stage}"
        )
    refuse_embed_stage_switch(
        recorded, resolved, is_slurm=is_slurm, stage4_available=stage4_available
    )
    return resolved, None  # pragma: no cover - the refusal always raises here


def has_slurm_gpu_profile(config: "ExecutionConfig") -> bool:
    """U2's GPU profile: ``--gpu-slurm`` given, or a ``GpuDetector`` in the pipeline.

    A pipeline that cannot be loaded has none; the validation reports it.
    """
    if config.gpu_slurm_args:
        return True
    pipeline = load_pipeline_or_none(config.pipeline_json)
    return pipeline is not None and _has_gpu_detector(pipeline)


def stage4_supported(config: "ExecutionConfig") -> bool:
    """Whether a Stage-4 executor runs this configuration's embeddings today.

    The interim gate of #282 P3-P4 review C1: a **forward run** (``--mode
    full``) has one, locally (the local strategies, staged or single-pass)
    and on SLURM (the staged controller's GPU rounds, P5 T2, which routes a
    SLURM ``gpu`` run to ``StagedSlurmStrategy`` with or without a
    ``GpuDetector``). ``--mode measure``, local or on SLURM, would defer its
    embeddings to a Stage 4 that nothing on its path runs, so the run could
    never finish. ``--mode process`` never reaches this: it resolves
    ``inline`` and R1 refuses ``gpu``.

    P6 T1 lifts measure mode here, in the commit that builds its executor.
    """
    from ._cli_preflight import run_mode_of

    return run_mode_of(config) == "full"


def stage4_unavailable_reason(config: "ExecutionConfig") -> str:
    """Why :func:`stage4_supported` is false for *config*: the note's and refusal's words."""
    del config  # only measure mode is gated now (P5 T2 lifted SLURM)
    return "Stage 4 in --mode measure is not yet available"


def resolve_embed_stage_for_config(
    config: "ExecutionConfig", pipeline: "ImagePipeline | None"
) -> tuple[EmbedStage, list[str]]:
    """Resolve ``config.embed_stage_request`` for this run, with its note lines.

    The facts :func:`resolve_embed_stage` needs come from *pipeline*, the
    run's loaded pipeline. ``None`` (a pipeline that failed to load) resolves
    ``inline``, and the pipeline validation reports the failure in its own
    words.

    A run :func:`stage4_supported` refuses is the interim gate's (review
    C1): ``auto`` that would have resolved ``gpu`` resolves ``inline`` with
    a note, and an explicit ``gpu`` is refused.

    Returns:
        The resolved stage, and the one-line notes to print: U9's when a CPU
        pin, and O3's when an embedder with no Stage-4 encoder, is what made
        ``auto`` resolve inline; O5's when ``--embed-stage gpu`` meets a
        pipeline with no embedder; the gate's when only the gate did.

    Raises:
        click.UsageError: ``--embed-stage gpu`` on a run with no Stage-4
            executor yet (:func:`stage4_supported`).
    """
    from ._cli_preflight import run_mode_of

    requested = config.embed_stage_request
    if pipeline is None:
        return "inline", []
    mode = run_mode_of(config)
    embedders = pipeline_embedders(pipeline)
    pinned = _cpu_pinned_key(pipeline)
    unstageable = _unstageable_embedder(pipeline)
    is_slurm = config.is_slurm_mode()
    has_gpu_detector = _has_gpu_detector(pipeline)

    def _resolve(cpu_pinned: bool, unstageable: bool) -> EmbedStage:
        return resolve_embed_stage(
            requested,
            mode=mode,
            is_slurm=is_slurm,
            has_gpu_detector=has_gpu_detector,
            gpu_slurm_given=bool(config.gpu_slurm_args),
            has_embedders=bool(embedders),
            cpu_pinned=cpu_pinned,
            unstageable=unstageable,
        )

    resolved = _resolve(pinned is not None, unstageable is not None)
    notes: list[str] = []
    # A note only when the embedder decided the outcome: auto would otherwise
    # have been gpu. A local auto run is inline anyway and says nothing.
    if requested == "auto" and resolved == "inline" and _resolve(False, False) == "gpu":
        if pinned is not None:
            notes.append(
                f"meas '{pinned}' is pinned to device='cpu', so --embed-stage "
                "auto resolves to inline"
            )
        if unstageable is not None:
            key, cls = unstageable
            notes.append(
                f"meas '{key}' ({cls}) has no GPU Stage-4 encoder, so "
                "--embed-stage auto resolves to inline"
            )
    if requested == "gpu" and not embedders and mode != "process":
        notes.append(
            "--embed-stage gpu: the pipeline has no embedder, so nothing runs "
            "in Stage 4"
        )
    if resolved == "gpu" and not stage4_supported(config):
        why = stage4_unavailable_reason(config)
        if requested == "gpu":
            raise click.UsageError(
                f"--embed-stage gpu: {why}; pass --embed-stage inline to embed "
                "inline in Stage 3"
            )
        notes.append(f"{why}, so --embed-stage auto resolves to inline")
        resolved = "inline"
    return resolved, notes


def _has_gpu_detector(pipeline: "ImagePipeline") -> bool:
    """Whether *pipeline* holds a ``GpuDetector`` anywhere in its tree."""
    from ._cli_validation import find_gpu_detectors

    try:
        return bool(find_gpu_detectors(pipeline))
    except ValueError:
        # An unstageable placement: the staged-GPU probe above the resolver
        # has already refused it; a pipeline built by hand is not a GPU profile.
        return False


def load_pipeline_or_none(pipeline_json: Path | None) -> "ImagePipeline | None":
    """The run's pipeline, or ``None`` when it fails to load (the validation reports it)."""
    if pipeline_json is None:
        return None
    from phenotypic._core._image_pipeline import ImagePipeline

    try:
        return ImagePipeline.from_json(pipeline_json)
    except Exception:  # noqa: BLE001 - reported by the pipeline validation
        return None


def defers_embeddings(config: "ExecutionConfig") -> bool:
    """Whether the run's per-image core defers its embeddings to Stage 4."""
    return config.embed_stage == "gpu"


def apply_stage4_counts(
    counts: "dict[str, dict[str, int]]",
    stage4: "Mapping[str, tuple[int, int]]",
) -> None:
    """Move a deferring run's Stage-3 counts to what Stage 4 left, in place (#282 P4 T3).

    Under deferral a Stage-3 success is progress, not completion: Stage 4
    publishes the record. *counts* are a local strategy's per-dataset
    ``{"completed", "failed"}`` after its Stage-3 loop, which count an image
    that finished Stages 1-3 (now or in an earlier invocation) as completed.
    Each image Stage 4 failed moves from completed to failed; one it
    published stays completed. ``run_local_stage4`` raises rather than leave
    an image it was handed pending, so nothing else moves.

    Args:
        counts: ``{dataset: {"completed": n, "failed": n}}``, updated.
        stage4: ``run_local_stage4``'s ``{dataset: (published, failed)}``.
    """
    for dataset, (_published, failed) in stage4.items():
        entry = counts.get(dataset)
        if entry is None or not failed:
            continue
        entry["completed"] -= failed
        entry["failed"] += failed


def prepare_stage4_environment(config: "ExecutionConfig") -> None:
    """Prepare the process for a ``gpu`` run, before anything can initialise CUDA.

    - ``CUBLAS_WORKSPACE_CONFIG``: set to ``:4096:8`` when unset. cuBLAS
      reads it once, at the process's first handle, and a local Stage 4 runs
      in this process, which Stage 2 may put on the GPU first (plan review
      I4). A user's own ``:16:8`` is kept. When CUDA is already initialised
      and the variable is not one of the two deterministic values, setting it
      now would change nothing, so the run stops (P1/P2 review M12). torch is
      never imported to ask.
    - **A local run only**: the CUDA check, :func:`stage4_device`. A machine
      with no usable CUDA stops here, before Stage 1 writes anything (plan
      review I16). This imports torch, and only for an explicit local
      ``--embed-stage gpu`` (a local ``auto`` never resolves ``gpu``); a
      ``--dry-run`` with that flag pays the import too. A SLURM shard checks
      its own device at shard start.

    A no-op for a run that embeds inline.

    Raises:
        click.ClickException: One line, exit 1: an environment error, not a
            refused option (the exit-code table's ``ERROR``).
    """
    if not defers_embeddings(config):
        return
    _set_cublas_workspace(local=not config.is_slurm_mode())
    if config.is_slurm_mode():
        return
    from ._cli_stage4_worker import Stage4DeviceError, stage4_device

    try:
        stage4_device()
    except Stage4DeviceError as exc:
        raise click.ClickException(
            f"--embed-stage gpu runs Stage 4 on this machine: {exc}. Run on a "
            "GPU node, or pass --embed-stage inline to embed on the CPU"
        ) from exc


def _set_cublas_workspace(*, local: bool) -> None:
    """``setdefault`` the cuBLAS workspace variable, refusing a value that comes too late.

    Args:
        local: Whether Stage 4 will run in this process. A preset value that
            is not deterministic is then refused now rather than by
            ``check_cublas_workspace`` after Stages 1-3 have run.

    Raises:
        click.ClickException: CUDA is already initialised in this process and
            the variable is not deterministic; or, locally, the variable is
            preset to a value that is not.
    """
    from phenotypic.detect.nn._helper._batch_encoder import (
        CUBLAS_WORKSPACE_DEFAULT,
        CUBLAS_WORKSPACE_VARIABLE,
        DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS,
    )

    value = os.environ.get(CUBLAS_WORKSPACE_VARIABLE)
    if value in DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS:
        return
    accepted = " or ".join(DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS)
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        found = "unset" if value is None else repr(value)
        raise click.ClickException(
            f"{CUBLAS_WORKSPACE_VARIABLE} is {found}, and this process has "
            "already initialised CUDA, so cuBLAS would not see a new value; "
            f"export {CUBLAS_WORKSPACE_VARIABLE}={CUBLAS_WORKSPACE_DEFAULT} "
            f"({accepted}) before running phenotypic"
        )
    if value is not None and local:
        raise click.ClickException(
            f"{CUBLAS_WORKSPACE_VARIABLE} is {value!r}; deterministic cuBLAS "
            f"needs {accepted}. Export {CUBLAS_WORKSPACE_VARIABLE}="
            f"{CUBLAS_WORKSPACE_DEFAULT}, or unset it"
        )
    os.environ.setdefault(CUBLAS_WORKSPACE_VARIABLE, CUBLAS_WORKSPACE_DEFAULT)


#: The ``processing_state.json`` config key holding a Stage-4 run's revisions (U5).
EMBED_REVISIONS_CONFIG_KEY = "embed_revisions"


def _hub_commit(hf_id: str) -> str:
    """The Hugging Face hub's current commit of *hf_id*'s default branch.

    The one network seam of U5: tests replace it, and nothing else in
    PhenoTypic asks the hub for a commit.

    Raises:
        Exception: Whatever ``huggingface_hub`` raises (offline, no
            network, a gated repo without a token, ...).
    """
    from huggingface_hub import HfApi

    sha = HfApi().model_info(hf_id).sha
    if not sha:
        raise RuntimeError(f"the hub reported no commit for {hf_id}")
    return str(sha)


def _hub_offline() -> bool:
    """Whether ``HF_HUB_OFFLINE`` asks for no hub access (``1``/``true``/``yes``/``on``)."""
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_embed_revisions(
    pipeline: "ImagePipeline", *, known: "Mapping[str, str | None] | None" = None
) -> dict[str, str | None]:
    """``{hf_id: commit}`` for every unpinned in-scope ``MeasureDinoEmbedding`` (U5).

    Called once, in the submitting process, at a Stage-4 run's first
    invocation (O9: Stage-4 runs only), so every shard and round, and every
    continuation (:func:`recorded_embed_revisions`), loads the same weights.
    An embedder that pins ``revision`` needs nothing and is not listed. Each
    model is asked once however many embedders share it.

    With ``HF_HUB_OFFLINE`` set, or on any hub error, a model's value is
    ``None`` and one warning line names it: each process then resolves the
    revision on its own, the per-process behaviour ``PF-EMBED-REVISION-UNPINNED``
    already warns of.

    Args:
        pipeline: The run's pipeline.
        known: Commits a previous invocation of this run recorded; a model
            with one here is kept as it is and never asked again.

    Returns:
        The map, in ``meas`` order; empty when every embedder is pinned.
    """
    from phenotypic.detect.nn._helper._dino_support import hf_dino_id
    from phenotypic.measure import MeasureDinoEmbedding

    revisions: dict[str, str | None] = {}
    for _key, embedder in pipeline_embedders(pipeline):
        if not isinstance(embedder, MeasureDinoEmbedding) or embedder.revision is not None:
            continue
        hf_id = hf_dino_id(embedder.version, embedder.backbone_key().size)
        if hf_id in revisions:
            continue
        if known is not None and known.get(hf_id) is not None:
            revisions[hf_id] = known[hf_id]
            continue
        if _hub_offline():
            reason = "HF_HUB_OFFLINE is set"
            commit = None
        else:
            try:
                commit = _hub_commit(hf_id)
                reason = ""
            except Exception as exc:  # noqa: BLE001 - any hub failure falls back, U5
                commit = None
                reason = f"{type(exc).__name__}: {exc}".splitlines()[0]
        if commit is None:
            click.echo(
                f"Warning: could not resolve the revision of {hf_id} ({reason}); "
                "each Stage-4 process resolves on its own, so shards may load "
                "different weights. Pin revision= in the pipeline to avoid this.",
                err=True,
            )
        revisions[hf_id] = commit
    return revisions


def recorded_embed_revisions(output_dir: Path) -> dict[str, str | None] | None:
    """The revisions an earlier invocation of this run recorded, or ``None`` (U5).

    Read from ``processing_state.json`` config :data:`EMBED_REVISIONS_CONFIG_KEY`,
    then from ``job_metadata.json`` ``JobMetadataKey.EMBED_REVISIONS``. A
    continuation, local or on SLURM, reuses them and never re-resolves;
    ``None`` (a first invocation, a run that embedded inline, a pre-#282
    run) resolves (:func:`resolve_embed_revisions`).
    """
    from phenotypic.sdk_ import JobMetadataKey, progress_dir

    from ._cli_state_management import load_processing_state
    from ._cli_utils import load_job_metadata

    state = load_processing_state(output_dir)
    recorded = state.config.get(EMBED_REVISIONS_CONFIG_KEY) if state is not None else None
    if recorded is None:
        metadata = load_job_metadata(progress_dir(output_dir))
        if isinstance(metadata, dict):
            recorded = metadata.get(JobMetadataKey.EMBED_REVISIONS)
    if not isinstance(recorded, dict):
        return None
    return {str(hf_id): (None if commit is None else str(commit)) for hf_id, commit in recorded.items()}


def pin_embed_revisions(config: "ExecutionConfig", output_dir: Path) -> None:
    """Set ``config.embed_revisions`` for this invocation: reused, resolved, or ``None`` (U5).

    A run that embeds inline gets ``None``. A ``gpu`` continuation reuses
    what its first invocation recorded; a first invocation (or ``--restart``,
    which cleared the record) resolves, and so does a continuation for each
    model recorded ``None`` (resolved offline, so never pinned; review M10). Called in the mutating half, below
    the ``--dry-run`` exit, before the processing state is written, so the
    state and the job metadata record the same map.
    """
    if not defers_embeddings(config):
        config.embed_revisions = None
        return
    recorded = recorded_embed_revisions(output_dir) if config.resume else None
    if recorded is not None and all(commit is not None for commit in recorded.values()):
        config.embed_revisions = recorded
        return
    pipeline = load_pipeline_or_none(config.pipeline_json)
    # A recorded None was never a pin, only U5's offline fallback: a
    # continuation resolves that model again, and keeps every commit it did
    # record (review M10). Resolved here, at submission, so the shards of
    # this invocation still share one commit.
    config.embed_revisions = (
        {} if pipeline is None else resolve_embed_revisions(pipeline, known=recorded)
    )
