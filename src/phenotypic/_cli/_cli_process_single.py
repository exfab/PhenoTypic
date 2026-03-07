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
from typing import Optional, Literal, Dict, Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

from phenotypic import Image, GridImage, ImagePipeline
from ._cli_output_manager import OutputManager
from ._cli_update_state import append_event, append_completion_event
from ._cli_failure_tracker import append_failure
from ._cli_utils import normalize_extension

logger = logging.getLogger(__name__)


def process_single_image_core(
    pipeline_path: Path,
    image_path: Path,
    output_dir: Path,
    dataset_name: str,
    image_type: Literal["Image", "GridImage"],
    read_kwargs: Dict[str, Any],
    output_manager: OutputManager,
) -> bool:
    """
    Core processing logic for a single image.

    Args:
        pipeline_path: Path to pipeline JSON file
        image_path: Path to input image
        output_dir: Base output directory
        dataset_name: Dataset name for this image
        image_type: "Image" or "GridImage"
        read_kwargs: Kwargs for imread (nrows, ncols, bit_depth)
        output_manager: OutputManager instance

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

    # Load image
    detect_mode = read_kwargs.pop("detect_mode", "gray")
    image = image_cls.imread(image_path, **read_kwargs)

    # Apply detect mode if not default
    if detect_mode != "gray":
        image.set_detect_mode(detect_mode)

    # Execute pipeline
    measurements = pipeline.apply_and_measure(image, inplace=True)

    # Get image stem for output filenames
    image_stem = image_path.stem

    # Save measurements
    output_manager.save_measurements(measurements, dataset_name, image_stem)

    # Save overlay
    output_manager.save_overlay(image, dataset_name, image_stem)

    # Save optional image layers
    output_manager.save_image_layers(image, dataset_name, image_stem)

    return True


@click.command()
@click.option(
    "--pipeline",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to pipeline JSON file",
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
@click.option("--nrows", type=int, default=8, help="Number of grid rows (for GridImage)")
@click.option(
    "--ncols", type=int, default=12, help="Number of grid columns (for GridImage)"
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
def main(
    pipeline: Path,
    image: Path,
    output_dir: Path,
    dataset_name: str,
    image_type: str,
    nrows: int,
    ncols: int,
    bit_depth: Optional[int],
    detect_mode: str,
    ext: str,
    overlay_alpha: float,
    include_dataset_column: bool,
    event_log: Optional[Path],
):
    """
    Process a single image with PhenoTypic pipeline.

    This is designed to be called by SLURM batch scripts for autonomous
    execution. It processes one image and logs completion to event log.
    """
    try:
        # Prepare read kwargs
        read_kwargs = {}
        if image_type == "GridImage":
            read_kwargs["nrows"] = nrows
            read_kwargs["ncols"] = ncols
        if bit_depth is not None:
            read_kwargs["bit_depth"] = bit_depth
        if detect_mode != "gray":
            read_kwargs["detect_mode"] = detect_mode

        # Validate extension
        try:
            ext_normalized = normalize_extension(ext, ".tiff")
        except click.BadParameter as e:
            logger.error(f"Invalid extension parameter: {e}")
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

        # Create output manager
        output_manager = OutputManager.from_config(
            base_dir=output_dir,
            ext=ext_normalized,
            include_dataset_column=include_dataset_column,
            overlay_alpha=overlay_alpha,
        )

        # Log "started" event (with SLURM env vars when available)
        if event_log is not None:
            append_event(
                event_log=event_log,
                dataset=dataset_name,
                image=image.name,
                status="started",
                slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
                slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID", ""),
            )

        # Process image
        click.echo(f"Processing {image.name}...")
        success = process_single_image_core(
            pipeline_path=pipeline,
            image_path=image,
            output_dir=output_dir,
            dataset_name=dataset_name,
            image_type=image_type,
            read_kwargs=read_kwargs,
            output_manager=output_manager,
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
            progress_dir = output_dir / "progress"
            slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
            slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "")
            full_slurm_id = (
                f"{slurm_job_id}_{slurm_task_id}"
                if slurm_job_id and slurm_task_id
                else slurm_job_id
            )
            append_failure(
                progress_dir,
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
