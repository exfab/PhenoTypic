"""
Single image processor for the PhenoTypic CLI.

This module provides a standalone CLI for processing individual images,
designed to be called by SLURM batch scripts for autonomous execution.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
import json
import os
import sys
import logging
import click
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Mapping, cast
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

from phenotypic import Image, GridImage, ImagePipeline
from phenotypic._core._provenance import (
    initialize_cli_provenance,
    pipeline_source_identity,
    provenance_success_sink,
    resume_provenance_application,
    set_provenance_status,
    write_provenance_checkpoint,
)
from ._cli_output_manager import OutputManager
from ._cli_process_only import (
    process_only_output_path,
    process_single_apply_only_core,
    resolve_process_format,
)
from ._cli_completion import image_data_artifact, publish_image_success
from ._cli_update_state import (
    PROCESSING_GENERATION_ENV_VAR,
    SLURM_GENERATION_ENV_VAR,
    append_event,
    append_completion_event,
)
from ._cli_failure_tracker import (
    PerImageScientificError,
    _normalized_input_relative_path,
    append_failure,
    append_terminal_failure,
    compute_work_id,
    file_sha256,
    processing_configuration_digest_from_values,
)
from ._cli_slurm_lifecycle import (
    SlurmGenerationInactiveError,
    assert_generation_active,
    generation_publication_guard,
    slurm_generation_inactive_cause,
)
from ._cli_utils import normalize_extension
from phenotypic.sdk_ import (
    CommitGuard,
    EnvVar,
    MEASUREMENT_TABLE_RELATIVE_PATH,
    load_image_from_store,
    progress_dir,
    source_image_stem,
    store_stem,
    zarr_store_path,
)
from phenotypic.sdk_.typing_ import (
    CliMode,
    ImageTypeName,
    ProcessFormat,
    ProcessOnlyLayer,
)

logger = logging.getLogger(__name__)

ActiveCheck = Callable[[], None]


def _check_active(active_check: ActiveCheck | None) -> None:
    if active_check is not None:
        active_check()


def _ordinary_slurm_active_check(output_dir: Path) -> ActiveCheck | None:
    """Resolve the ordinary worker lifecycle fence, or None when local."""
    generation = os.environ.get(SLURM_GENERATION_ENV_VAR)
    if not os.environ.get(EnvVar.SLURM_JOB_ID) or not generation:
        return None

    def _check() -> None:
        assert_generation_active(output_dir, generation)

    return _check


def _ordinary_slurm_commit_guard(output_dir: Path) -> CommitGuard | None:
    """Resolve the narrow ordinary-worker publication guard, or None locally."""
    generation = os.environ.get(SLURM_GENERATION_ENV_VAR)
    if not os.environ.get(EnvVar.SLURM_JOB_ID) or not generation:
        return None

    def _guard() -> AbstractContextManager[None]:
        return generation_publication_guard(output_dir, generation)

    return _guard


def _authoritative_lifecycle_epoch() -> str:
    """Return the epoch fencing authoritative per-image publication."""
    if os.environ.get(EnvVar.SLURM_JOB_ID):
        return os.environ.get(SLURM_GENERATION_ENV_VAR, "slurm-unfenced")
    return os.environ.get(PROCESSING_GENERATION_ENV_VAR, "local-unfenced")


def _worker_work_identity(
    *,
    pipeline: Path,
    image: Path,
    input_root: Path | None,
    dataset_name: str,
    image_type: str,
    nrows: int | None,
    ncols: int | None,
    bit_depth: int | None,
    detect_mode: str,
    layer: str | None,
    ext: str,
    process_format: str,
    include_dataset_column: bool,
    overlay_alpha: float,
    save_overlays: bool,
    drop_originals: bool = False,
    mode: str,
) -> tuple[str, str]:
    """Calculate the same work identity used by top-level selection."""
    relative_path = _normalized_input_relative_path(
        input_root, image
    ).as_posix()
    return (
        compute_work_id(
            dataset=dataset_name,
            relative_image_path=relative_path,
            input_sha256=file_sha256(image),
            pipeline_fingerprint=file_sha256(pipeline),
            processing_config_digest=processing_configuration_digest_from_values(
                image_type=image_type,
                nrows=nrows,
                ncols=ncols,
                bit_depth=bit_depth,
                detect_mode=detect_mode,
                process_only_layer=layer if mode == "process" else None,
                ext=normalize_extension(ext, ".tiff"),
                # Mirrors the `process_only_layer` line directly above, so
                # the two stay consistent: outside process mode the digest
                # drops both, and a stray format can never reach it.
                process_format=(
                    process_format if mode == "process" else "tiff"
                ),
                include_dataset_column=include_dataset_column,
                overlay_alpha=overlay_alpha,
                save_overlays=save_overlays,
                drop_originals=drop_originals,
            ),
            mode=mode,
        ),
        relative_path,
    )


def process_single_image_core(
    pipeline_path: Path,
    image_path: Path,
    output_dir: Path,
    dataset_name: str,
    image_type: ImageTypeName,
    read_kwargs: Dict[str, Any],
    output_manager: OutputManager,
    cli_nrows: Optional[int] = None,
    cli_ncols: Optional[int] = None,
    drop_originals: bool = False,
    pipeline_identity: Mapping[str, str] | None = None,
    active_check: ActiveCheck | None = None,
    commit_guard: CommitGuard | None = None,
) -> bool:
    """
    Core processing logic for a single image.

    Args:
        pipeline_path: Path to pipeline JSON file
        image_path: Path to input image
        output_dir: Base output directory
        dataset_name: Dataset name for this image
        image_type: "Image" or "GridImage"
        read_kwargs: Kwargs for imread (bit_depth, detect_mode). Should NOT
            include ``nrows``/``ncols`` — those are resolved here from the CLI
            override (``cli_nrows``/``cli_ncols``) and the pipeline preset.
        output_manager: OutputManager instance
        cli_nrows: Explicit CLI ``--nrows`` override, or ``None`` if not passed.
        cli_ncols: Explicit CLI ``--ncols`` override, or ``None`` if not passed.
        drop_originals: Skip decoded-source retention when true.
        pipeline_identity: Original user pipeline source identity when
            ``pipeline_path`` is an immutable worker snapshot.
        active_check: Optional lifecycle ownership assertion for SLURM workers.
        commit_guard: Optional narrow lifecycle guard around each canonical
            filesystem commit point.

    Returns:
        True if successful. This function always returns True on success;
        failures are communicated by raising exceptions rather than returning False.

    Raises:
        Exception: Any exception from pipeline loading, image reading, or processing
            will propagate to the caller. The caller is responsible for catching
            exceptions and handling failures appropriately.
    """
    image_stem = source_image_stem(image_path)
    store = zarr_store_path(output_dir, dataset_name, image_stem)
    image: Image | GridImage | None = None
    checkpoint_ready = False

    def _write_checkpoint(
        updated: Image, *, journal_only: bool = False
    ) -> None:
        _check_active(active_check)
        write_provenance_checkpoint(
            store,
            updated,
            journal_only=journal_only,
            commit_guard=commit_guard,
        )

    def _mark_failed_checkpoint() -> None:
        if image is None or not checkpoint_ready:
            return
        try:
            _check_active(active_check)
            set_provenance_status(image, "failed")
            write_provenance_checkpoint(
                store, image, commit_guard=commit_guard
            )
        except SlurmGenerationInactiveError:
            return
        except Exception:
            logger.exception(
                "Failed to mark provenance checkpoint failed: %s", store
            )

    try:
        # Load, decode, and run the configured scientific computation. Output
        # publication intentionally occurs after this boundary.
        pipeline = ImagePipeline.from_json(pipeline_path)
        image_cls = GridImage if image_type == "GridImage" else Image

        if image_type == "GridImage":
            from ._cli_utils import resolve_grid_shape

            nrows, ncols = resolve_grid_shape(
                cli_nrows=cli_nrows,
                cli_ncols=cli_ncols,
                pipeline_nrows=pipeline.nrows,
                pipeline_ncols=pipeline.ncols,
            )
            read_kwargs = dict(read_kwargs)
            read_kwargs["nrows"] = nrows
            read_kwargs["ncols"] = ncols

        detect_mode = read_kwargs.pop("detect_mode", "gray")
        image = image_cls.imread(image_path, **read_kwargs)
        resolved_pipeline_identity = (
            pipeline_source_identity(pipeline_path)
            if pipeline_identity is None
            else pipeline_identity
        )
        resumed = False
        checkpoint_root = store / "zarr.json"
        if checkpoint_root.is_file():
            checkpoint_payload = json.loads(
                checkpoint_root.read_text(encoding="utf-8")
            )
            checkpoint_journal = (
                checkpoint_payload.get("attributes", {})
                .get("phenotypic", {})
                .get("provenance")
            )
            if isinstance(checkpoint_journal, dict):
                resumed = resume_provenance_application(
                    image,
                    checkpoint_journal,
                    kind="full",
                    input_filename=image_path.name,
                    pipeline_identity=resolved_pipeline_identity,
                )
        if not resumed:
            initialize_cli_provenance(
                image,
                pipeline_path,
                kind="full",
                input_filename=image_path.name,
                pipeline_identity=resolved_pipeline_identity,
            )
        if drop_originals:
            _write_checkpoint(image, journal_only=True)
        else:
            _check_active(active_check)
            image._retain_original()
            saved_store = output_manager.save_image_store(
                image,
                dataset_name,
                image_stem,
                commit_guard=commit_guard,
            )
            if saved_store is None:
                raise RuntimeError(
                    f"Initial image checkpoint failed for {dataset_name}/{image_stem}"
                )
        checkpoint_ready = True
        if detect_mode != "gray":
            image.set_detect_mode(detect_mode)
        with provenance_success_sink(_write_checkpoint):
            measurements = pipeline.apply_and_measure(
                image, inplace=True, apply_post=False
            )
        if output_manager.save_overlays:
            _check_active(active_check)
            output_manager.save_overlay(
                image,
                dataset_name,
                image_stem,
                commit_guard=commit_guard,
            )
        from phenotypic.plotting._pipeline import PlotCoordinator

        _check_active(active_check)
        PlotCoordinator(
            pipeline, output_dir, commit_guard=commit_guard
        ).emit_image(
            image,
            dataset=dataset_name,
            image_stem=image_stem,
        )
        _check_active(active_check)
        set_provenance_status(image, "complete")
        saved_store = output_manager.save_image_store(
            image,
            dataset_name,
            image_stem,
            commit_guard=commit_guard,
            measurements=measurements,
        )
        if saved_store is None:
            raise RuntimeError(
                f"Final image store publication failed for {dataset_name}/{image_stem}"
            )
    except SlurmGenerationInactiveError:
        raise
    except MemoryError:
        _mark_failed_checkpoint()
        raise
    except Exception as exc:
        inactive = slurm_generation_inactive_cause(exc)
        if inactive is not None:
            raise inactive
        _mark_failed_checkpoint()
        raise PerImageScientificError("full", exc) from exc

    return True


def process_single_store_measure_core(
    pipeline_path: Path,
    store_path: Path,
    output_dir: Path,
    dataset_name: str,
    image_type: ImageTypeName,
    output_manager: OutputManager,
    commit_guard: CommitGuard | None = None,
) -> bool:
    """Rerun pipeline.measure() on an already-processed OME-Zarr store.

    Loads ``store_path`` as :class:`Image` or :class:`GridImage` — dispatched
    from the store's own ``phenotypic.image_class``, with ``image_type`` as the
    fallback for a block that carries none — runs
    :meth:`ImagePipeline.measure` only (no apply / no detection), and rewrites
    the embedded measurement table. Does not regenerate overlays or pixel arrays.

    Args:
        pipeline_path: Path to pipeline JSON file.
        store_path: Path to an existing ``*.ome.zarr`` store produced by a
            prior forward run.
        output_dir: Base output directory used for plots and marker refresh.
        dataset_name: Dataset name for measurement output.
        image_type: ``"Image"`` or ``"GridImage"`` — the fallback used when the
            store's ``phenotypic`` block carries no ``image_class``.
        output_manager: :class:`OutputManager` for writing the parquet.

    Returns:
        ``True`` on success. Exceptions propagate to the caller, which is
        responsible for logging/handling them — mirroring
        :func:`process_single_image_core`.

    Raises:
        Exception: Any exception from pipeline loading, store loading, or
            measurement will propagate.
    """
    # Load pipeline
    pipeline = ImagePipeline.from_json(pipeline_path)

    # Dispatch on the store's recorded image_class so a GridImage rehydrates
    # with its grid state intact; ``image_type`` is only the fallback.
    image = load_image_from_store(store_path, fallback=image_type)

    # Measurement only — no apply / detection. apply_post=False matches
    # the forward path so store re-measure parquets are also post-free.
    measurements = pipeline.measure(image, apply_post=False)

    # ``store_stem``, never ``Path.stem``: ``.ome.zarr`` is a double suffix,
    # so ``.stem`` would name the parquet ``<stem>.ome.parquet`` and key the
    # plot binding on ``<stem>.ome`` — plausible-looking wrong names that
    # nothing raises on.
    stem = store_stem(store_path)

    # Publish the authoritative table inside the existing store. Descriptor
    # changes use a root-last store transaction; compatible tables use one
    # validated same-directory atomic file replacement.
    output_manager.replace_image_store_measurements(
        store_path,
        measurements,
        dataset_name,
        commit_guard=commit_guard,
    )
    from phenotypic.plotting._pipeline import PlotCoordinator

    PlotCoordinator(
        pipeline, output_dir, commit_guard=commit_guard
    ).emit_image(
        image,
        dataset=dataset_name,
        image_stem=stem,
    )

    # Marker refresh is the final successful per-image publication. If any
    # earlier table or plot work raises, the old marker remains stale against
    # the new table/root and therefore cannot authorize a partial update.
    from phenotypic.sdk_ import image_completion_marker_path

    marker_path = image_completion_marker_path(output_dir, dataset_name, stem)
    if marker_path.is_file():
        from ._cli_recompile_tables import _republish_table_marker

        _republish_table_marker(
            output_dir, marker_path, commit_guard=commit_guard
        )

    return True


@click.command()
@click.option(
    "--pipeline",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to pipeline config file",
)
@click.option(
    "--image",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to input image",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Base output directory",
)
@click.option(
    "--dataset-name",
    required=True,
    help="Dataset name (subdirectory name or '_root')",
)
@click.option(
    "--image-type",
    type=click.Choice(["Image", "GridImage"]),
    default="GridImage",
    help="Image class to use",
)
@click.option(
    "--nrows",
    type=int,
    default=None,
    help="Number of grid rows (for GridImage). Overrides any pipeline-level "
    "preset; falls back to the pipeline preset or 8 when omitted.",
)
@click.option(
    "--ncols",
    type=int,
    default=None,
    help="Number of grid columns (for GridImage). Overrides any pipeline-level "
    "preset; falls back to the pipeline preset or 12 when omitted.",
)
@click.option(
    "--bit-depth", type=int, default=None, help="Bit depth (8 or 16)"
)
@click.option(
    "--detect-mode",
    type=click.Choice(["gray", "red", "green", "blue"]),
    default="gray",
    help="Color channel for detection matrix",
)
@click.option(
    "--ext",
    default="tiff",
    help="File extension for rgb, gray, detect_mat layers",
)
@click.option(
    "--overlay-alpha",
    type=float,
    default=0.3,
    help="Alpha transparency for label overlay (0.0-1.0)",
)
@click.option(
    "--dataset-column/--no-dataset-column",
    "include_dataset_column",
    default=True,
    help="Include the Metadata_Dataset column in measurements "
    "(included by default; --no-dataset-column excludes it)",
)
@click.option(
    "--event-log",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to event log file (for status updates)",
)
@click.option(
    "--mode",
    type=click.Choice(["full", "measure", "process"]),
    default="full",
    show_default=True,
    help="Per-image worker mode.",
)
@click.option(
    "--save-overlays/--no-save-overlays",
    default=True,
    help="Save a PNG overlay per image (default: on). Ignored in measure mode.",
)
@click.option(
    "--layer",
    "layer",
    type=click.Choice(["rgb", "gray", "detect_mat", "objmap"]),
    default=None,
    help="Layer exported by --mode process.",
)
@click.option(
    "--process-format",
    "process_format",
    type=click.Choice(["tiff", "zarr"]),
    default=None,
    help=(
        "Output format for --mode process. Default: zarr for rgb/gray (a "
        "single-series OME-Zarr store), tiff for detect_mat (float TIFF) and "
        "objmap (a 16-bit raw-label PNG)."
    ),
)
@click.option(
    "--input-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Root of the input tree, used to compute the mirrored output path "
    "in process mode.",
)
@click.option(
    "--durable-writes/--no-durable-writes",
    "durable_writes",
    default=None,
    help=(
        "fsync each image store before promoting it. Unset auto-detects SLURM "
        "in THIS process, which is why the submitter only needs to emit the "
        "flag when it was given explicitly."
    ),
)
@click.option(
    "--drop-originals",
    is_flag=True,
    help="Do not retain decoded source pixels in full-forward image stores.",
)
@click.option("--provenance-pipeline-source-path", default=None, hidden=True)
@click.option("--provenance-pipeline-sha256", default=None, hidden=True)
@click.option("--expected-work-id", default=None, hidden=True)
@click.option("--expected-input-sha256", default=None, hidden=True)
@click.option("--expected-pipeline-sha256", default=None, hidden=True)
@click.option("--attempt-id", default=None, hidden=True)
def main(
    pipeline: Path,
    image: Path,
    output_dir: Path,
    dataset_name: str,
    image_type: str,
    nrows: Optional[int],
    ncols: Optional[int],
    bit_depth: Optional[int],
    detect_mode: str,
    ext: str,
    overlay_alpha: float,
    include_dataset_column: bool,
    event_log: Optional[Path],
    mode: str,
    save_overlays: bool,
    layer: Optional[str],
    process_format: Optional[str],
    input_root: Optional[Path],
    durable_writes: Optional[bool],
    drop_originals: bool,
    provenance_pipeline_source_path: Optional[str],
    provenance_pipeline_sha256: Optional[str],
    expected_work_id: Optional[str],
    expected_input_sha256: Optional[str],
    expected_pipeline_sha256: Optional[str],
    attempt_id: Optional[str],
):
    """
    Process a single image with PhenoTypic pipeline.

    This is designed to be called by SLURM batch scripts for autonomous
    execution. It processes one image and logs completion to event log.
    """
    attempt_id = attempt_id or uuid4().hex
    active_check = _ordinary_slurm_active_check(output_dir)
    commit_guard = _ordinary_slurm_commit_guard(output_dir)
    try:
        cli_mode = cast(CliMode, mode)
        if drop_originals and cli_mode != "full":
            raise click.UsageError(
                f"--drop-originals is not accepted with --mode {cli_mode}"
            )
        measure_only = cli_mode == "measure"
        process_only_layer: Optional[ProcessOnlyLayer] = None
        resolved_process_format: ProcessFormat = "tiff"
        if cli_mode == "process":
            if layer is None:
                raise click.UsageError("--mode process requires --layer")
            process_only_layer = cast(ProcessOnlyLayer, layer)
            resolved_process_format = resolve_process_format(
                process_only_layer,
                cast("ProcessFormat | None", process_format),
            )
        else:
            if layer is not None:
                raise click.UsageError(
                    "--layer can only be used with --mode process"
                )
            if process_format is not None:
                raise click.UsageError(
                    "--process-format can only be used with --mode process"
                )

        provenance_identity_values = (
            provenance_pipeline_source_path,
            provenance_pipeline_sha256,
        )
        if any(value is not None for value in provenance_identity_values):
            if not all(
                value is not None for value in provenance_identity_values
            ):
                raise click.UsageError(
                    "Incomplete provenance pipeline identity"
                )
            pipeline_identity = {
                "source_path": cast(str, provenance_pipeline_source_path),
                "sha256": cast(str, provenance_pipeline_sha256),
            }
        else:
            pipeline_identity = None

        expected_identity = (
            expected_work_id,
            expected_input_sha256,
            expected_pipeline_sha256,
        )
        if any(value is not None for value in expected_identity):
            if not all(value for value in expected_identity):
                raise RuntimeError("Incomplete immutable SLURM task identity")
            if file_sha256(image) != expected_input_sha256:
                raise RuntimeError(
                    "Input changed after SLURM worklist creation"
                )
            if file_sha256(pipeline) != expected_pipeline_sha256:
                raise RuntimeError(
                    "Pipeline changed after SLURM worklist creation"
                )
            actual_work_id, _ = _worker_work_identity(
                pipeline=pipeline,
                image=image,
                input_root=input_root,
                dataset_name=dataset_name,
                image_type=image_type,
                nrows=nrows,
                ncols=ncols,
                bit_depth=bit_depth,
                detect_mode=detect_mode,
                layer=layer,
                ext=ext,
                process_format=resolved_process_format,
                include_dataset_column=include_dataset_column,
                overlay_alpha=overlay_alpha,
                save_overlays=save_overlays,
                drop_originals=drop_originals,
                mode=mode,
            )
            if actual_work_id != expected_work_id:
                raise RuntimeError(
                    "SLURM task work identity does not match worklist "
                    f"(expected {expected_work_id}, computed {actual_work_id}); "
                    "the worker's processing configuration differs from the "
                    "one selection digested"
                )

        # Process-only (apply-only) mode: run pipeline.apply() and export one
        # layer, mirroring the input tree. No measurement / aggregation output.
        if process_only_layer is not None:
            if input_root is None:
                raise click.UsageError("--mode process requires --input-root")
            process_only_read_kwargs: Dict[str, Any] = {}
            if bit_depth is not None:
                process_only_read_kwargs["bit_depth"] = bit_depth
            if detect_mode != "gray":
                process_only_read_kwargs["detect_mode"] = detect_mode
            if event_log is not None:
                append_event(
                    event_log=event_log,
                    dataset=dataset_name,
                    image=image.name,
                    status="started",
                    slurm_job_id=os.environ.get(EnvVar.SLURM_JOB_ID, ""),
                    slurm_array_task_id=os.environ.get(
                        EnvVar.SLURM_ARRAY_TASK_ID, ""
                    ),
                    attempt_id=attempt_id,
                    work_id=expected_work_id or "",
                )
            click.echo(
                f"Processing (apply-only, {process_only_layer}) {image.name}..."
            )
            process_single_apply_only_core(
                pipeline_path=pipeline,
                image_path=image,
                input_root=input_root,
                output_dir=output_dir,
                image_type=image_type,  # type: ignore[arg-type]
                layer=process_only_layer,  # type: ignore[arg-type]
                read_kwargs=process_only_read_kwargs,
                cli_nrows=nrows,
                cli_ncols=ncols,
                commit_guard=commit_guard,
                process_format=resolved_process_format,
            )
            work_id, relative_path = _worker_work_identity(
                pipeline=pipeline,
                image=image,
                input_root=input_root,
                dataset_name=dataset_name,
                image_type=image_type,
                nrows=nrows,
                ncols=ncols,
                bit_depth=bit_depth,
                detect_mode=detect_mode,
                layer=layer,
                ext=ext,
                process_format=resolved_process_format,
                include_dataset_column=include_dataset_column,
                overlay_alpha=overlay_alpha,
                save_overlays=save_overlays,
                drop_originals=drop_originals,
                mode=mode,
            )
            publish_image_success(
                output_dir,
                work_id=work_id,
                dataset=dataset_name,
                relative_image_path=relative_path,
                image_stem=source_image_stem(image),
                mode="process",
                attempt_id=attempt_id,
                lifecycle_epoch=_authoritative_lifecycle_epoch(),
                artifacts={
                    "process_output": process_only_output_path(
                        output_dir,
                        image,
                        input_root,
                        process_only_layer,
                        fmt=resolved_process_format,
                    )
                },
                commit_guard=commit_guard,
            )
            if event_log is not None:
                append_completion_event(
                    event_log=event_log,
                    dataset=dataset_name,
                    image=image.name,
                    status="completed",
                    error_msg="",
                    commit_guard=commit_guard,
                )
            click.echo(f"✓ Successfully processed {image.name}")
            sys.exit(0)

        # Validate extension (used for overlay / legacy call sites)
        try:
            ext_normalized = normalize_extension(ext, ".tiff")
        except click.BadParameter as e:
            logger.error(f"Invalid extension parameter: {e}")
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

        # Log "started" event (with SLURM env vars when available)
        if event_log is not None:
            append_event(
                event_log=event_log,
                dataset=dataset_name,
                image=image.name,
                status="started",
                slurm_job_id=os.environ.get(EnvVar.SLURM_JOB_ID, ""),
                slurm_array_task_id=os.environ.get(
                    EnvVar.SLURM_ARRAY_TASK_ID, ""
                ),
                attempt_id=attempt_id,
                work_id=expected_work_id or "",
            )

        if measure_only:
            # Measure-only mode: --image points to an existing OME-Zarr store.
            # The store's own ``phenotypic.image_class`` decides the class;
            # ``--image-type`` is only the fallback for a block that carries
            # none. The dispatch lives in ``load_image_from_store``, which the
            # measure core calls, so there is nothing to pre-resolve here --
            # the previous hand-rolled h5py probe is gone with the HDF.
            # Measure mode never writes overlays regardless of the flag.
            output_manager = OutputManager.from_config(
                base_dir=output_dir,
                ext=ext_normalized,
                include_dataset_column=include_dataset_column,
                overlay_alpha=overlay_alpha,
                save_overlays=False,
                durable_writes=durable_writes,
            )

            click.echo(f"Measuring {image.name} (store rerun)...")
            process_single_store_measure_core(
                pipeline_path=pipeline,
                store_path=image,
                output_dir=output_dir,
                dataset_name=dataset_name,
                image_type=cast(ImageTypeName, image_type),
                output_manager=output_manager,
                commit_guard=commit_guard,
            )
        else:
            # Forward run: prepare read kwargs and dispatch to the detection path.
            read_kwargs: Dict[str, Any] = {}
            if bit_depth is not None:
                read_kwargs["bit_depth"] = bit_depth
            if detect_mode != "gray":
                read_kwargs["detect_mode"] = detect_mode

            output_manager = OutputManager.from_config(
                base_dir=output_dir,
                ext=ext_normalized,
                include_dataset_column=include_dataset_column,
                overlay_alpha=overlay_alpha,
                save_overlays=save_overlays,
                durable_writes=durable_writes,
            )

            click.echo(f"Processing {image.name}...")
            process_single_image_core(
                pipeline_path=pipeline,
                image_path=image,
                output_dir=output_dir,
                dataset_name=dataset_name,
                image_type=image_type,  # type: ignore[arg-type]
                read_kwargs=read_kwargs,
                output_manager=output_manager,
                cli_nrows=nrows,
                cli_ncols=ncols,
                drop_originals=drop_originals,
                pipeline_identity=pipeline_identity,
                active_check=active_check,
                commit_guard=commit_guard,
            )
            work_id, relative_path = _worker_work_identity(
                pipeline=pipeline,
                image=image,
                input_root=input_root,
                dataset_name=dataset_name,
                image_type=image_type,
                nrows=nrows,
                ncols=ncols,
                bit_depth=bit_depth,
                detect_mode=detect_mode,
                layer=None,
                ext=ext,
                process_format=resolved_process_format,
                include_dataset_column=include_dataset_column,
                overlay_alpha=overlay_alpha,
                save_overlays=save_overlays,
                drop_originals=drop_originals,
                mode=mode,
            )
            # The same resolver the local strategy and the staged SLURM
            # worker use. It is what keeps this site from certifying a `.h5`
            # that `process_single_image_core` no longer writes -- and
            # `publish_image_success` resolves every artifact strict=True, so
            # naming a dead file is a FileNotFoundError, not a stale marker.
            data_key, data_path = image_data_artifact(
                output_dir,
                output_manager,
                dataset_name,
                source_image_stem(image),
            )
            artifacts = {
                "measurements": data_path / MEASUREMENT_TABLE_RELATIVE_PATH,
                data_key: data_path,
            }
            if save_overlays:
                artifacts["overlay"] = output_manager.get_output_path(
                    dataset_name, "overlays", source_image_stem(image)
                )
            publish_image_success(
                output_dir,
                work_id=work_id,
                dataset=dataset_name,
                relative_image_path=relative_path,
                image_stem=source_image_stem(image),
                mode="full",
                attempt_id=attempt_id,
                lifecycle_epoch=_authoritative_lifecycle_epoch(),
                artifacts=artifacts,
                commit_guard=commit_guard,
            )

        # Log completion if event log provided
        if event_log is not None:
            append_completion_event(
                event_log=event_log,
                dataset=dataset_name,
                image=image.name,
                status="completed",
                error_msg="",
                commit_guard=commit_guard,
            )

        click.echo(f"✓ Successfully processed {image.name}")
        sys.exit(0)

    except Exception as e:
        if (inactive := slurm_generation_inactive_cause(e)) is not None:
            raise inactive
        error_msg = f"{type(e).__name__}: {str(e)}"
        tb = traceback.format_exc()

        if isinstance(e, PerImageScientificError):
            try:
                lifecycle_epoch = _authoritative_lifecycle_epoch()
                work_id, relative_path = _worker_work_identity(
                    pipeline=pipeline,
                    image=image,
                    input_root=input_root,
                    dataset_name=dataset_name,
                    image_type=image_type,
                    nrows=nrows,
                    ncols=ncols,
                    bit_depth=bit_depth,
                    detect_mode=detect_mode,
                    layer=layer,
                    ext=ext,
                    process_format=resolved_process_format,
                    include_dataset_column=include_dataset_column,
                    overlay_alpha=overlay_alpha,
                    save_overlays=save_overlays,
                    drop_originals=drop_originals,
                    mode=mode,
                )
                committed = append_terminal_failure(
                    output_dir,
                    work_id=work_id,
                    dataset=dataset_name,
                    relative_image_path=relative_path,
                    failed_stage=e.stage,
                    exception=e.cause,
                    attempt_id=attempt_id,
                    lifecycle_epoch=lifecycle_epoch,
                    traceback=tb,
                    slurm_job_id=os.environ.get(EnvVar.SLURM_JOB_ID, ""),
                    commit_guard=commit_guard,
                )
                if not committed:
                    logger.warning(
                        "Scientific failure remains pending because its "
                        "terminal record was not committed"
                    )
            except (OSError, RuntimeError, ValueError) as publication_error:
                inactive = slurm_generation_inactive_cause(publication_error)
                if inactive is not None:
                    raise inactive
                logger.warning(
                    "Scientific failure remains pending because its work "
                    "identity could not be committed",
                    exc_info=True,
                )

        click.echo(f"✗ Failed to process {image.name}: {error_msg}", err=True)
        click.echo(f"Traceback:\n{tb}", err=True)

        # Log failure if event log provided
        if event_log is not None:
            try:
                append_completion_event(
                    event_log=event_log,
                    dataset=dataset_name,
                    image=image.name,
                    status="failed",
                    error_msg=error_msg,
                    commit_guard=commit_guard,
                )
            except Exception as publication_error:
                inactive = slurm_generation_inactive_cause(publication_error)
                if inactive is not None:
                    raise inactive
                logger.warning("Failed to write event log", exc_info=True)

        # Write structured failure record
        try:
            prog_dir = progress_dir(output_dir)
            slurm_job_id = os.environ.get(EnvVar.SLURM_JOB_ID, "")
            slurm_task_id = os.environ.get(EnvVar.SLURM_ARRAY_TASK_ID, "")
            full_slurm_id = (
                f"{slurm_job_id}_{slurm_task_id}"
                if slurm_job_id and slurm_task_id
                else slurm_job_id
            )
            append_failure(
                prog_dir,
                dataset=dataset_name,
                image=image.name,
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=tb,
                slurm_job_id=full_slurm_id,
                commit_guard=commit_guard,
            )
        except Exception as publication_error:
            inactive = slurm_generation_inactive_cause(publication_error)
            if inactive is not None:
                raise inactive
            logger.warning("Failed to write failure record", exc_info=True)

        sys.exit(1)


if __name__ == "__main__":
    main()
