"""
Single image processor for the PhenoTypic CLI.

This module provides a standalone CLI for processing individual images,
designed to be called by SLURM batch scripts for autonomous execution.
"""

from __future__ import annotations

import os
import sys
import logging
import click
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, cast

import h5py  # type: ignore[import-untyped]
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

from phenotypic import Image, GridImage, ImagePipeline
from ._cli_output_manager import OutputManager
from ._cli_process_only import process_single_apply_only_core
from ._cli_update_state import append_event, append_completion_event
from ._cli_failure_tracker import append_failure
from ._cli_utils import normalize_extension
from phenotypic.sdk_ import EnvVar, HdfAttr, progress_dir
from phenotypic.sdk_.typing_ import CliMode, ImageTypeName, ProcessOnlyLayer

logger = logging.getLogger(__name__)


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

    Returns:
        True if successful. This function always returns True on success;
        failures are communicated by raising exceptions rather than returning False.

    Raises:
        Exception: Any exception from pipeline loading, image reading, or processing
            will propagate to the caller. The caller is responsible for catching
            exceptions and handling failures appropriately.
    """
    # Load pipeline
    pipeline = ImagePipeline.from_json(pipeline_path)

    # Determine image class
    image_cls = GridImage if image_type == "GridImage" else Image

    if image_type == "GridImage":
        from ._cli_utils import resolve_grid_shape

        nrows, ncols = resolve_grid_shape(
            cli_nrows=cli_nrows,
            cli_ncols=cli_ncols,
            pipeline_nrows=pipeline.nrows,
            pipeline_ncols=pipeline.ncols,
        )
        read_kwargs = dict(read_kwargs)  # local copy; do not mutate caller's dict
        read_kwargs["nrows"] = nrows
        read_kwargs["ncols"] = ncols

    # Load image
    detect_mode = read_kwargs.pop("detect_mode", "gray")
    image = image_cls.imread(image_path, **read_kwargs)

    # Apply detect mode if not default
    if detect_mode != "gray":
        image.set_detect_mode(detect_mode)

    # Execute pipeline. apply_post=False keeps per-image parquets clean;
    # post ops are applied once in aggregate_measurements() against the
    # full master so master_measurements.{csv,parquet} stay post-free.
    measurements = pipeline.apply_and_measure(image, inplace=True, apply_post=False)

    # Get image stem for output filenames
    image_stem = image_path.stem

    # Save measurements + HDF5 (always) and overlay (opt-in), then dispatch
    # configured PlotImage bindings while measurer caches still refer to this
    # exact image instance.
    output_manager.save_measurements(measurements, dataset_name, image_stem)
    output_manager.save_image_hdf(image, dataset_name, image_stem)
    if output_manager.save_overlays:
        output_manager.save_overlay(image, dataset_name, image_stem)
    from phenotypic.plotting._pipeline import PlotCoordinator

    PlotCoordinator(pipeline, output_dir).emit_image(
        image,
        dataset=dataset_name,
        image_stem=image_stem,
    )

    return True


def process_single_hdf_measure_core(
    pipeline_path: Path,
    hdf_path: Path,
    output_dir: Path,
    dataset_name: str,
    image_type: ImageTypeName,
    output_manager: OutputManager,
) -> bool:
    """Rerun pipeline.measure() on an already-processed HDF file.

    Loads ``hdf_path`` as :class:`Image` or :class:`GridImage` (per
    ``image_type``), runs :meth:`ImagePipeline.measure` only (no apply / no
    detection), and rewrites the measurements parquet.  Does NOT regenerate
    overlays or touch the HDF file itself.

    Args:
        pipeline_path: Path to pipeline JSON file.
        hdf_path: Path to an existing ``.h5`` file produced by a prior
            forward run.
        output_dir: Base output directory (unused here, but kept symmetric
            with :func:`process_single_image_core`).
        dataset_name: Dataset name for measurement output.
        image_type: ``"Image"`` or ``"GridImage"`` — dictates which loader
            to call.
        output_manager: :class:`OutputManager` for writing the parquet.

    Returns:
        ``True`` on success. Exceptions propagate to the caller, which is
        responsible for logging/handling them — mirroring
        :func:`process_single_image_core`.

    Raises:
        Exception: Any exception from pipeline loading, HDF loading, or
            measurement will propagate.
    """
    # Load pipeline
    pipeline = ImagePipeline.from_json(pipeline_path)

    # Determine image class and load from HDF5
    image_cls = GridImage if image_type == "GridImage" else Image
    image = image_cls.load_hdf5(hdf_path)

    # Measurement only — no apply / detection. apply_post=False matches
    # the forward path so HDF re-measure parquets are also post-free.
    measurements = pipeline.measure(image, apply_post=False)

    # Save measurements parquet (overlay + HDF intentionally skipped)
    output_manager.save_measurements(measurements, dataset_name, hdf_path.stem)
    from phenotypic.plotting._pipeline import PlotCoordinator

    PlotCoordinator(pipeline, output_dir).emit_image(
        image,
        dataset=dataset_name,
        image_stem=hdf_path.stem,
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
    "--dataset-name", required=True, help="Dataset name (subdirectory name or '_root')"
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
@click.option("--bit-depth", type=int, default=None, help="Bit depth (8 or 16)")
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
    "--no-dataset-column",
    "include_dataset_column",
    is_flag=True,
    flag_value=False,
    default=True,
    help="Exclude Metadata_Dataset column from measurements CSV (included by default)",
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
    "--input-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Root of the input tree, used to compute the mirrored output path "
    "in process mode.",
)
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
    input_root: Optional[Path],
):
    """
    Process a single image with PhenoTypic pipeline.

    This is designed to be called by SLURM batch scripts for autonomous
    execution. It processes one image and logs completion to event log.
    """
    try:
        cli_mode = cast(CliMode, mode)
        measure_only = cli_mode == "measure"
        process_only_layer: Optional[ProcessOnlyLayer] = None
        if cli_mode == "process":
            if layer is None:
                raise click.UsageError("--mode process requires --layer")
            process_only_layer = cast(ProcessOnlyLayer, layer)
        elif layer is not None:
            raise click.UsageError("--layer can only be used with --mode process")

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
            )
            if event_log is not None:
                append_completion_event(
                    event_log=event_log,
                    dataset=dataset_name,
                    image=image.name,
                    status="completed",
                    error_msg="",
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
                slurm_array_task_id=os.environ.get(EnvVar.SLURM_ARRAY_TASK_ID, ""),
            )

        if measure_only:
            # Measure-only mode: --image points to an existing HDF5 file.
            # Determine the image class from the saved file's root attr.
            # Legacy v1 files lack the attr — fall back to the configured
            # --image-type so behaviour matches the local executor.
            resolved_image_type: ImageTypeName = (
                image_type  # type: ignore[assignment]
            )
            try:
                with h5py.File(image, "r") as hf:
                    saved_class = hf.attrs.get(HdfAttr.PHENOTYPIC_CLASS)
                    if isinstance(saved_class, bytes):
                        saved_class = saved_class.decode("utf-8", errors="replace")
                    if saved_class == "GridImage":
                        resolved_image_type = "GridImage"
                    elif saved_class == "Image":
                        resolved_image_type = "Image"
                    elif saved_class is None:
                        logger.warning(
                            "phenotypic_class attr absent in %s; falling back to "
                            "configured image_type=%s. If this file was saved as a "
                            "different class (e.g. legacy v1 format), measurements "
                            "may be incorrect.",
                            image,
                            resolved_image_type,
                        )
            except (OSError, KeyError) as hdf_err:
                logger.warning(
                    "Could not read phenotypic_class from %s (%s: %s); "
                    "falling back to configured image_type=%s",
                    image,
                    type(hdf_err).__name__,
                    hdf_err,
                    resolved_image_type,
                )

            # Measure mode never writes overlays regardless of the flag.
            output_manager = OutputManager.from_config(
                base_dir=output_dir,
                ext=ext_normalized,
                include_dataset_column=include_dataset_column,
                overlay_alpha=overlay_alpha,
                save_overlays=False,
            )

            click.echo(f"Measuring {image.name} (HDF rerun)...")
            process_single_hdf_measure_core(
                pipeline_path=pipeline,
                hdf_path=image,
                output_dir=output_dir,
                dataset_name=dataset_name,
                image_type=resolved_image_type,
                output_manager=output_manager,
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
            )

        # Log completion if event log provided
        if event_log is not None:
            append_completion_event(
                event_log=event_log,
                dataset=dataset_name,
                image=image.name,
                status="completed",
                error_msg="",
            )

        click.echo(f"✓ Successfully processed {image.name}")
        sys.exit(0)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        tb = traceback.format_exc()

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
                )
            except Exception:
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
            )
        except Exception:
            logger.warning("Failed to write failure record", exc_info=True)

        sys.exit(1)


if __name__ == "__main__":
    main()
