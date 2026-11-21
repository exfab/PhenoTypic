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
no in-scope embedder                                      ``inline`` (``gpu`` too, with a note)
========================================================  ======================

The three refusals are each one :class:`click.UsageError` line (exit 2), in
the read-only half, outside ``--skip-validation`` and above ``--restart``,
``--overwrite``, ``mint_run_identity`` and ``--dry-run``:

- **R1** (:func:`refuse_embed_stage_for_process`): ``--embed-stage gpu`` with
  ``--mode process``, which stays inline on the CPU (H5).
- **R2** (:func:`refuse_unstageable_embedders`): ``--embed-stage gpu`` with
  an embedder pinned to ``device="cpu"`` (U9), or a run resolved to ``gpu``
  holding an embedder with no Stage-4 encoder (O3).
- **R3** (:func:`refuse_embed_stage_switch`): a continuation whose resolved
  stage differs from the recorded one (U3).

:func:`prepare_stage4_environment` is the startup half of a ``gpu`` run: the
cuBLAS workspace variable, and for a local run the CUDA check, both before
anything could initialise CUDA.

Heavy imports (the pipeline, the embedders, torch) stay inside functions.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import click

if TYPE_CHECKING:
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


def resolve_embed_stage(
    requested: EmbedStageRequest,
    *,
    mode: "RunMode",
    is_slurm: bool,
    has_gpu_detector: bool,
    gpu_slurm_given: bool,
    has_embedders: bool,
    cpu_pinned: bool,
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

    Returns:
        ``"gpu"`` or ``"inline"``.
    """
    if mode == "process" or not has_embedders or requested == "inline":
        return "inline"
    if requested == "gpu":
        return "gpu"
    if cpu_pinned or not is_slurm:
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
    """R2: refuse an embedder that cannot run in the stage the run resolved to.

    - ``--embed-stage gpu`` with an in-scope embedder pinned to
      ``device="cpu"`` (U9). Under ``auto`` the pin resolves the run inline
      instead, so it is accepted.
    - A run resolved to ``gpu`` holding a ``MeasureEmbedding`` that is not a
      ``MeasureDinoEmbedding``: only DINO has a Stage-4 encoder (O3).

    Raises:
        click.UsageError: Naming the first such ``meas`` key and its class.
    """
    from phenotypic.measure import MeasureDinoEmbedding

    for key, embedder in pipeline_embedders(pipeline):
        cls = type(embedder).__name__
        if requested == "gpu" and getattr(embedder, "device", None) == "cpu":
            raise click.UsageError(
                f"--embed-stage gpu was given, but meas '{key}' ({cls}) is "
                "pinned to device='cpu'; set device='auto' or 'cuda', or drop "
                "--embed-stage gpu to embed inline"
            )
        if resolved == "gpu" and not isinstance(embedder, MeasureDinoEmbedding):
            raise click.UsageError(
                f"--embed-stage resolves to gpu, but meas '{key}' ({cls}) has "
                "no GPU Stage-4 encoder; pass --embed-stage inline"
            )


def refuse_embed_stage_switch(
    recorded: str | None, resolved: EmbedStage, *, is_slurm: bool
) -> None:
    """R3: refuse a continuation that would embed elsewhere than it started (U3).

    Args:
        recorded: The state's recorded ``embed_stage``; ``None`` (a state
            written before #282) reads as ``"inline"``.
        resolved: What this invocation resolved to.
        is_slurm: Whether this invocation is a SLURM run; the remedy then
            also names ``--gpu-slurm``, which ``auto`` resolves from.

    Raises:
        click.UsageError: The two differ.
    """
    recorded_stage = DEFAULT_EMBED_STAGE if recorded is None else recorded
    if recorded_stage == resolved:
        return
    remedy = f"Re-run with --embed-stage {recorded_stage}"
    if is_slurm:
        remedy += " (on SLURM, also --gpu-slurm)"
    raise click.UsageError(
        f"this run was started with --embed-stage {recorded_stage}; continuing "
        f"it as {resolved} would mix CPU and GPU embeddings. {remedy}, or "
        "re-embed the finished run with --mode measure --embed-stage gpu"
    )


def resolve_embed_stage_for_config(
    config: "ExecutionConfig", pipeline: "ImagePipeline | None"
) -> tuple[EmbedStage, list[str]]:
    """Resolve ``config.embed_stage_request`` for this run, with its note lines.

    The facts :func:`resolve_embed_stage` needs come from *pipeline*, the
    run's loaded pipeline. ``None`` (a pipeline that failed to load) resolves
    ``inline``, and the pipeline validation reports the failure in its own
    words.

    Returns:
        The resolved stage, and the one-line notes to print: U9's when a CPU
        pin is what made ``auto`` resolve inline, O5's when ``--embed-stage
        gpu`` meets a pipeline with no embedder.
    """
    from ._cli_preflight import run_mode_of

    requested = config.embed_stage_request
    if pipeline is None:
        return "inline", []
    mode = run_mode_of(config)
    embedders = pipeline_embedders(pipeline)
    pinned = _cpu_pinned_key(pipeline)
    is_slurm = config.is_slurm_mode()
    has_gpu_detector = _has_gpu_detector(pipeline)

    def _resolve(cpu_pinned: bool) -> EmbedStage:
        return resolve_embed_stage(
            requested,
            mode=mode,
            is_slurm=is_slurm,
            has_gpu_detector=has_gpu_detector,
            gpu_slurm_given=bool(config.gpu_slurm_args),
            has_embedders=bool(embedders),
            cpu_pinned=cpu_pinned,
        )

    resolved = _resolve(pinned is not None)
    notes: list[str] = []
    if pinned is not None and requested == "auto" and _resolve(False) == "gpu":
        notes.append(
            f"meas '{pinned}' is pinned to device='cpu', so --embed-stage auto "
            "resolves to inline"
        )
    if requested == "gpu" and not embedders and mode != "process":
        notes.append(
            "--embed-stage gpu: the pipeline has no embedder, so nothing runs "
            "in Stage 4"
        )
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
