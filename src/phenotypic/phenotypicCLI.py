"""
PhenoTypic CLI (v2.0)
=====================

A command-line interface for executing PhenoTypic ImagePipelines on images or
directories of images with support for local parallel processing and autonomous
SLURM cluster execution.

Features:
    - Automatic timestamped output directories
    - Recursive directory support (1 level deep)
    - Dry-run mode for previewing processing plans
    - Sample processing mode for testing pipelines
    - Resume capability with state tracking
    - Local parallel execution (joblib)
    - Autonomous SLURM execution with bash scripts
    - HTML failure reports with tracebacks
    - Progress monitoring tools_

Usage:
    python -m phenotypic PIPELINE_JSON INPUT_PATH [OPTIONS]

Examples:
    # Basic usage with auto-generated output directory
    uv run python -m phenotypic pipeline.json ./images

    # Specify output directory
    uv run python -m phenotypic pipeline.json ./images -o ./results

    # Dry-run to preview processing plan
    uv run python -m phenotypic pipeline.json ./images --dry-run

    # Sample 5 images per dataset for testing
    uv run python -m phenotypic pipeline.json ./images --sample 5

    # Resume interrupted processing
    uv run python -m phenotypic pipeline.json ./images -o ./results --resume

    # Restart processing from beginning (clears previous state)
    uv run python -m phenotypic pipeline.json ./images -o ./results --restart

    # SLURM execution (autonomous)
    uv run python -m phenotypic pipeline.json ./images \
        --slurm-args slurm_partition=compute \
        --slurm-args slurm_account=proj \
        --slurm-args mem_gb=16

    # SLURM with progress monitoring
    uv run python -m phenotypic pipeline.json ./images \
        --slurm-args slurm_partition=compute \
        --slurm-args slurm_account=proj \
        --wait

    # Save intermediate layers
    uv run python -m phenotypic pipeline.json ./images \
        --save-rgb --save-gray --save-objmask

    # GridImage with custom dimensions
    uv run python -m phenotypic pipeline.json ./plates \
        --image-type GridImage --nrows 16 --ncols 24

SLURM Execution (Autonomous HPC Cluster Processing):
    Use --slurm-args to submit jobs to an HPC cluster via SLURM. The CLI will:
    1. Generate SBATCH scripts for each dataset
    2. Create array jobs for parallel image processing
    3. Automatically handle dependencies and chunking
    4. Support optional job monitoring with --wait

    Common Academic HPC SLURM Parameters:
        slurm_partition    Partition/queue name (e.g., compute, gpu, highmem)
        slurm_account      Account for billing/fairshare (required on most clusters)
        slurm_qos          Quality of Service tier (e.g., normal, high)
        time               Wall time in minutes (auto-converts to HH:MM:SS)
        mem_gb             Memory per node in GB (convenience param, adds "G" suffix)
        slurm_cpus_per_task CPUs per task (useful for joblib parallelism)
        slurm_constraint   Node features/constraints (e.g., gpu_type, cpu_generation)
        slurm_mail_type    Email notifications (e.g., END, FAIL, ALL)
        slurm_mail_user    Email address for notifications

    Advanced SLURM Parameters:
        slurm_nodes        Number of nodes (default: 1)
        slurm_mem          Memory with custom units (e.g., "32G", "1024M")
        slurm_mem_per_cpu  Memory per CPU instead of per node
        slurm_gpus_per_node GPUs per node for GPU-accelerated operations

    Time Parameter Notes:
        - Use 'time' or 'slurm_time' with integer minutes
        - Automatically converts to HH:MM:SS format (e.g., time=120 → 02:00:00)
        - Valid range: 1-10080 minutes (1 minute to 7 days)

    Example: Submit with account, partition, memory, and time limits
        uv run python -m phenotypic pipeline.json ./images \\
            --slurm-args slurm_partition=compute \\
            --slurm-args slurm_account=lab_proj \\
            --slurm-args mem_gb=32 \\
            --slurm-args time=120 \\
            --slurm-args slurm_mail_type=END \\
            --slurm-args slurm_mail_user=user@university.edu \\
            --wait

    Example: Dry-run to preview SLURM submission plan
        uv run python -m phenotypic pipeline.json ./images \\
            --slurm-args slurm_partition=compute \\
            --slurm-args slurm_account=lab_proj \\
            --dry-run

Migration Notes (v1.x → v2.0):
    - OUTPUT_DIR is now optional (generates timestamped dir if not provided)
    - Use -o/--output-dir instead of positional OUTPUT_DIR argument
    - --slurm-params KEY=VALUE replaced with --slurm-args KEY=VALUE
    - --slurm-kwds renamed to --slurm-args (breaking change in v2.0)
    - Recursive directory processing now preserves subdirectory hierarchy
"""

import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import click

# Set up logger
logger = logging.getLogger(__name__)

from phenotypic import Image, GridImage, ImagePipeline
from phenotypic._core._image_parts.detection_modes import available_modes
from phenotypic._cli._cli_directory_scanner import (
    generate_timestamped_output_dir,
    organize_by_dataset,
    scan_directory_structure,
)
from phenotypic._cli._cli_execution_strategies import create_execution_strategy
from phenotypic._cli._cli_interactive import (
    execute_dry_run,
    get_sample_datasets,
)
from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic._cli._cli_report_generator import HTMLReportGenerator
from phenotypic._cli._cli_state_management import (
    create_initial_state,
    get_remaining_images_for_datasets,
    load_processing_state,
    save_processing_state,
    update_state_from_events,
    validate_resume_compatibility,
)
from phenotypic._cli._cli_types import ExecutionConfig
from phenotypic._cli._cli_utils import normalize_extension
from phenotypic._cli._cli_validation import (
    full_validation,
    validate_execution_config,
    validate_pipeline,
    validate_pipeline_on_test_image,
)
from phenotypic._cli._cli_constants import (
    MIN_SLURM_TIME_MINUTES,
    MAX_SLURM_TIME_MINUTES,
)


def setup_logging(debug: bool = False):
    """Configure logging for CLI."""
    level = logging.DEBUG if debug else logging.INFO
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)


def error_exit(message: str, details: Optional[str] = None, code: int = 1) -> None:
    """Exit with consistent error formatting.

    Args:
        message: Main error message
        details: Optional additional details
        code: Exit code (default: 1)
    """
    click.echo(f"Error: {message}", err=True)
    if details:
        click.echo(f"\n{details}", err=True)
    sys.exit(code)


def _parse_slurm_args(slurm_args: Sequence[str]) -> dict:
    """
    Parse space-separated KEY=VALUE pairs into dictionary.

    Args:
        slurm_args: Sequence of "KEY=VALUE" strings

    Returns:
        Dictionary of parsed parameters

    Raises:
        click.BadParameter: If parsing fails
    """
    import ast

    parsed = {}
    for param in slurm_args:
        if "=" not in param:
            raise click.BadParameter(
                    "--slurm-args must be KEY=VALUE pairs",
                    param_hint="--slurm-args",
            )

        key, value = param.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            raise click.BadParameter(
                    "SLURM parameter keys cannot be empty",
                    param_hint="--slurm-args",
            )

        # Try to parse value as Python literal
        try:
            parsed_value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # Keep as string if not a valid literal
            parsed_value = value

        parsed[key] = parsed_value

    return parsed


def _validate_resume_input_images(
    state,
    current_datasets
) -> tuple[bool, Optional[str]]:
    """
    Validate that input image set matches between resume runs.

    Checks:
    1. All datasets from previous run are present
    2. Image filenames match exactly (not just counts)

    Args:
        state: Saved processing state
        current_datasets: Currently scanned datasets

    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message is None
    """
    # Build mapping of current datasets
    current_datasets_map = {ds.name: ds for ds in current_datasets}

    # Check all previous datasets still exist
    for ds_name in state.datasets.keys():
        if ds_name not in current_datasets_map:
            return False, f"Dataset '{ds_name}' from previous run not found in input directory"

    # For each dataset, compare actual image names
    for ds_name, ds_state in state.datasets.items():
        # Get previous image names from state
        # Prefer initial_images (set at run start) for accurate validation,
        # fallback to completed|failed for backward compatibility with older state files
        if ds_state.initial_images:
            prev_images = ds_state.initial_images
        else:
            prev_images = ds_state.completed | ds_state.failed

        # Get current image names from scan
        current_dataset = current_datasets_map[ds_name]
        curr_images = {img.name for img in current_dataset.images}

        # Check if sets match exactly (only if we have a valid previous set)
        if prev_images and prev_images != curr_images:
            missing = prev_images - curr_images
            added = curr_images - prev_images

            error_parts = [f"Image set mismatch in dataset '{ds_name}':"]
            if missing:
                error_parts.append(f"  - Missing {len(missing)} images (e.g., {list(missing)[:3]})")
            if added:
                error_parts.append(f"  - Added {len(added)} new images (e.g., {list(added)[:3]})")

            return False, "\n".join(error_parts)

    return True, None


def _format_slurm_time(minutes: int) -> str:
    """
    Convert integer minutes to HH:MM:SS SLURM time format.

    Args:
        minutes: Time in minutes

    Returns:
        Formatted time string in HH:MM:SS format (or "X days" for multi-day times)

    Examples:
        >>> _format_slurm_time(90)
        '01:30:00'
        >>> _format_slurm_time(120)
        '02:00:00'
        >>> _format_slurm_time(1440)
        '1 day'
    """
    if minutes >= 1440:  # 24 hours or more
        days = minutes // 1440
        remaining_minutes = minutes % 1440
        if remaining_minutes == 0:
            return f"{days} day{'s' if days > 1 else ''}"
        else:
            hours = remaining_minutes // 60
            mins = remaining_minutes % 60
            return f"{days}d {hours:02d}:{mins:02d}:00"
    else:
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:02d}:{mins:02d}:00"


def _format_slurm_key(key: str) -> str:
    """
    Convert SLURM parameter key to user-friendly display name.

    Args:
        key: SLURM parameter key (e.g., 'slurm_partition', 'mem_gb')

    Returns:
        Human-readable display name (e.g., 'Partition', 'Memory')
    """
    # Map of known keys to friendly display names
    key_mapping = {
        "slurm_partition": "Partition",
        "slurm_account": "Account",
        "slurm_qos": "QoS",
        "mem_gb": "Memory",
        "slurm_mem": "Memory",
        "slurm_mem_per_cpu": "Memory/CPU",
        "time": "Time Limit",
        "slurm_time": "Time Limit",
        "slurm_cpus_per_task": "CPUs/Task",
        "slurm_nodes": "Nodes",
        "slurm_gpus_per_node": "GPUs/Node",
        "slurm_constraint": "Constraint",
        "slurm_mail_type": "Mail Type",
        "slurm_mail_user": "Mail User",
    }

    if key in key_mapping:
        return key_mapping[key]

    # For unknown keys, convert to title case and remove slurm_ prefix
    display = key.replace("slurm_", "").replace("_", " ").title()
    return display


def _format_slurm_value(key: str, value) -> str:
    """
    Format SLURM parameter value for display.

    Args:
        key: SLURM parameter key
        value: Parameter value

    Returns:
        Formatted string for display
    """
    # Handle time parameters
    if key in ("time", "slurm_time") and isinstance(value, int):
        return _format_slurm_time(value)

    # Handle memory in GB
    if key == "mem_gb":
        return f"{value} GB"

    # Default: convert to string
    return str(value)


def _display_execution_config(
    config: ExecutionConfig,
    datasets: list
) -> None:
    """Display execution configuration in structured rich Table format.

    Args:
        config: ExecutionConfig containing pipeline and execution settings
        datasets: List of Dataset objects being processed
    """
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()

    # Determine backend
    backend = "SLURM Cluster" if config.is_slurm_mode() else "Local (joblib)"

    # Create table
    table = Table(
        title="Execution Configuration",
        show_header=False,
        box=None,
        padding=(0, 2)
    )
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    # Add configuration rows
    table.add_row("Backend", f"[bold]{backend}[/bold]")
    table.add_row("Pipeline", str(config.pipeline_json))
    table.add_row("Input Path", str(config.input_path))
    table.add_row("Output Dir", str(config.output_dir))
    table.add_row("", "")  # Spacer

    # Image settings
    table.add_row("Image Type", config.image_type)
    if config.image_type == "GridImage":
        table.add_row("Grid Dimensions", f"{config.nrows} × {config.ncols}")
    if config.bit_depth:
        table.add_row("Bit Depth", str(config.bit_depth))
    if config.detect_mode != "gray":
        table.add_row("Detect Mode", config.detect_mode)
    table.add_row("", "")  # Spacer

    # Execution settings
    if config.is_slurm_mode():
        # Show all SLURM parameters for debugging
        table.add_row("[bold]SLURM Settings[/bold]", "")
        for key in sorted(config.slurm_args.keys()):
            value = config.slurm_args[key]
            display_name = _format_slurm_key(key)
            display_value = _format_slurm_value(key, value)
            table.add_row(f"  {display_name}", display_value)
    else:
        # Show joblib settings
        n_jobs_str = "All cores" if config.n_jobs == -1 else str(config.n_jobs)
        table.add_row("Parallel Jobs", n_jobs_str)
    table.add_row("", "")  # Spacer

    # Dataset info
    total_images = sum(len(d.images) for d in datasets)
    table.add_row("Datasets", str(len(datasets)))
    table.add_row("Total Images", str(total_images))

    # Optional layers
    layers = []
    if config.save_rgb:
        layers.append("RGB")
    if config.save_gray:
        layers.append("Gray")
    if config.save_detect_mat:
        layers.append("Detection Matrix")
    if config.save_objmask:
        layers.append("Object Mask")
    if config.save_objmap:
        layers.append("Object Map")
    if config.save_objmap_overlay:
        layers.append("Object Map Overlay")
    if config.save_detect_mat_overlay:
        layers.append("Detection Matrix Overlay")
    if config.save_objmask_overlay:
        layers.append("Object Mask Overlay")

    if layers:
        table.add_row("", "")  # Spacer
        table.add_row("Saving Layers", ", ".join(layers))

    # Processing flags
    if config.sample or config.resume or config.retry_failures:
        table.add_row("", "")  # Spacer
        if config.sample:
            table.add_row("Sample Mode", f"{config.sample} images per dataset")
        if config.resume:
            resume_str = "Yes"
            if config.retry_failures:
                resume_str += " (with failures)"
            table.add_row("Resume Mode", resume_str)

    # Display the table in a panel
    console.print()
    console.print(Panel(table, border_style="blue", expand=False))
    console.print()


@click.command()
@click.argument(
        "pipeline_json",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
        "input_path",
        type=click.Path(exists=True, dir_okay=True, file_okay=True, path_type=Path),
)
@click.option(
        "-o",
        "--output-dir",
        type=click.Path(path_type=Path),
        default=None,
        help="Output directory (auto-generated if not specified)",
)
@click.option(
        "--image-type",
        type=click.Choice(["Image", "GridImage"], case_sensitive=False),
        default="GridImage",
        show_default=True,
        help="Type of image object to instantiate",
)
@click.option(
        "--nrows",
        type=click.IntRange(min=1),
        default=8,
        show_default=True,
        help="Number of rows for GridImage (must be positive)",
)
@click.option(
        "--ncols",
        type=click.IntRange(min=1),
        default=12,
        show_default=True,
        help="Number of columns for GridImage (must be positive)",
)
@click.option(
        "--bit-depth",
        type=int,
        default=None,
        help="Bit depth of input images (8 or 16)",
)
@click.option(
        "--detect-mode",
        type=click.Choice(list(available_modes())),
        default="gray",
        show_default=True,
        help="Source channel for the detection matrix",
)
@click.option(
        "--n-jobs",
        type=int,
        default=-1,
        show_default=True,
        help="Number of parallel jobs for local execution (-1 = all cores)",
)
@click.option(
        "--slurm-args",
        multiple=True,
        help="SLURM parameters as KEY=VALUE pairs. Pass multiple parameters with "
             "repeated --slurm-args flags (e.g., --slurm-args slurm_partition=compute "
             "--slurm-args mem_gb=16 --slurm-args time=60). Use slurm_ prefix for "
             "standard SBATCH directives, or use convenience params like mem_gb and time.",
)
@click.option(
        "--force-local",
        is_flag=True,
        help="Force local execution even if SLURM available",
)
@click.option(
        "--wait",
        is_flag=True,
        help="Wait and monitor SLURM jobs (default: return immediately)",
)
@click.option(
        "--save-rgb",
        is_flag=True,
        help="Save RGB images to OUTPUT_DIR/rgb/",
)
@click.option(
        "--save-gray",
        is_flag=True,
        help="Save grayscale images to OUTPUT_DIR/gray/",
)
@click.option(
        "--save-enh-gray",
        is_flag=True,
        help="Save detection matrix to OUTPUT_DIR/detect_mat/",
)
@click.option(
        "--save-objmask",
        is_flag=True,
        help="Save object masks to OUTPUT_DIR/objmask/",
)
@click.option(
        "--save-objmap",
        is_flag=True,
        help="Save object maps to OUTPUT_DIR/objmap/",
)
@click.option(
        "--save-objmap-overlay",
        is_flag=True,
        help="Save object map overlay (colorized labels) to OUTPUT_DIR/objmap_overlay/",
)
@click.option(
        "--save-enh-gray-overlay",
        is_flag=True,
        help="Save detection matrix overlay to OUTPUT_DIR/detect_mat_overlay/",
)
@click.option(
        "--save-objmask-overlay",
        is_flag=True,
        help="Save object mask overlay to OUTPUT_DIR/objmask_overlay/",
)
@click.option(
        "--rgb-ext",
        default="tiff",
        show_default=True,
        help="File extension for RGB saves",
)
@click.option(
        "--gray-ext",
        default="tiff",
        show_default=True,
        help="File extension for grayscale saves",
)
@click.option(
        "--enh-gray-ext",
        default="tiff",
        show_default=True,
        help="File extension for detection matrix saves",
)
@click.option(
        "--objmask-ext",
        default="png",
        show_default=True,
        help="File extension for object mask saves",
)
@click.option(
        "--objmap-ext",
        default="png",
        show_default=True,
        help="File extension for object map saves",
)
@click.option(
        "--objmap-overlay-ext",
        default="png",
        show_default=True,
        help="File extension for object map overlay saves",
)
@click.option(
        "--overlay-mode",
        type=click.Choice(["image", "figure"]),
        default="image",
        show_default=True,
        help="Overlay saving mode: 'image' for full-resolution, 'figure' for matplotlib",
)
@click.option(
        "--overlay-alpha",
        type=float,
        default=0.3,
        show_default=True,
        help="Alpha transparency for label overlay (0.0-1.0)",
)
@click.option(
        "--no-dataset-column",
        "include_dataset_column",
        is_flag=True,
        flag_value=False,
        default=True,
        help="Exclude 'Metadata_Dataset' column from master_measurements.csv (included by default)",
)
@click.option(
        "--dry-run",
        is_flag=True,
        help="Preview processing plan without executing",
)
@click.option(
        "--sample",
        type=int,
        default=None,
        help="Process N random images per dataset for testing",
)
@click.option(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for --sample reproducibility",
)
@click.option(
        "--resume",
        is_flag=True,
        help="Resume interrupted processing from checkpoint",
)
@click.option(
        "--retry-failures",
        is_flag=True,
        help="Include failed images when resuming (requires --resume)",
)
@click.option(
        "--restart",
        is_flag=True,
        help="Restart processing from beginning, clearing previous state (requires --output-dir)",
)
@click.option(
        "--skip-validation",
        is_flag=True,
        help="Skip pipeline validation (for advanced users)",
)
def main(
        pipeline_json: Path,
        input_path: Path,
        output_dir: Optional[Path],
        image_type: str,
        nrows: int,
        ncols: int,
        bit_depth: Optional[int],
        detect_mode: str,
        n_jobs: int,
        slurm_args: Sequence[str],
        force_local: bool,
        wait: bool,
        save_rgb: bool,
        save_gray: bool,
        save_detect_mat: bool,
        save_objmask: bool,
        save_objmap: bool,
        save_objmap_overlay: bool,
        save_detect_mat_overlay: bool,
        save_objmask_overlay: bool,
        rgb_ext: str,
        gray_ext: str,
        detect_mat_ext: str,
        objmask_ext: str,
        objmap_ext: str,
        objmap_overlay_ext: str,
        overlay_mode: str,
        overlay_alpha: float,
        include_dataset_column: bool,
        dry_run: bool,
        sample: Optional[int],
        random_seed: Optional[int],
        resume: bool,
        retry_failures: bool,
        restart: bool,
        skip_validation: bool,
):
    """
    Execute a PhenoTypic pipeline on images.

    PIPELINE_JSON: Path to pipeline configuration file

    INPUT_PATH: Image file or directory to process
    """
    try:
        resume_state = None

        # Validate extension arguments
        try:
            rgb_ext = normalize_extension(rgb_ext, ".tiff")
            gray_ext = normalize_extension(gray_ext, ".tiff")
            detect_mat_ext = normalize_extension(detect_mat_ext, ".tiff")
            objmask_ext = normalize_extension(objmask_ext, ".png")
            objmap_ext = normalize_extension(objmap_ext, ".png")
            objmap_overlay_ext = normalize_extension(objmap_overlay_ext, ".png")
        except click.BadParameter as e:
            click.echo(str(e), err=True)
            sys.exit(1)

        # Parse SLURM args
        slurm_args_dict = {}
        if slurm_args:
            try:
                slurm_args_dict = _parse_slurm_args(slurm_args)
            except click.BadParameter as e:
                click.echo(str(e), err=True)
                sys.exit(1)

        # Validate SLURM time parameter if present
        if slurm_args_dict:
            # Check for deprecated parameters
            if "time_min" in slurm_args_dict:
                click.echo(
                        "Warning: 'time_min' is deprecated. Use 'time' instead.",
                        err=True
                )
                # Auto-migrate
                if "time" not in slurm_args_dict:
                    slurm_args_dict["time"] = slurm_args_dict.pop("time_min")
                else:
                    slurm_args_dict.pop("time_min")

            # Validate time parameter type and range
            for time_key in ("time", "slurm_time"):
                if time_key in slurm_args_dict:
                    time_val = slurm_args_dict[time_key]
                    if not isinstance(time_val, int):
                        click.echo(
                                f"Error: '{time_key}' must be an integer (minutes), "
                                f"got {type(time_val).__name__}",
                                err=True
                        )
                        sys.exit(1)

                    # Validate reasonable time range
                    if time_val < MIN_SLURM_TIME_MINUTES:
                        click.echo(
                                f"Error: '{time_key}' must be >= {MIN_SLURM_TIME_MINUTES} minute, got {time_val}",
                                err=True
                        )
                        sys.exit(1)
                    elif time_val > MAX_SLURM_TIME_MINUTES:
                        days = MAX_SLURM_TIME_MINUTES / 1440
                        click.echo(
                                f"Warning: '{time_key}' is {time_val} minutes "
                                f"({time_val / 60:.1f} hours). This exceeds typical cluster limits "
                                f"({MAX_SLURM_TIME_MINUTES} min / {days:.1f} days).",
                                err=True
                        )

        # Validate flags
        if retry_failures and not resume:
            click.echo(
                    "Error: --retry-failures requires --resume", err=True
            )
            sys.exit(1)

        if restart and resume:
            click.echo(
                    "Error: --restart and --resume are mutually exclusive", err=True
            )
            sys.exit(1)

        if restart and output_dir is None:
            click.echo(
                    "Error: --restart requires --output-dir to be specified", err=True
            )
            click.echo(
                    "\nRestart mode clears previous processing state and starts fresh. "
                    "You must specify the output directory to restart.",
                    err=True
            )
            click.echo(
                    "\nExample:\n"
                    "  python -m phenotypic pipeline.json ./images \\\n"
                    "    --output-dir ./results_2024-01-12_10-30-45 \\\n"
                    "    --restart",
                    err=True
            )
            sys.exit(1)

        # Create ExecutionConfig
        config = ExecutionConfig(
                pipeline_json=pipeline_json,
                input_path=input_path,
                output_dir=output_dir,
                image_type=image_type,
                nrows=nrows,
                ncols=ncols,
                bit_depth=bit_depth,
                detect_mode=detect_mode,
                n_jobs=n_jobs,
                slurm_args=slurm_args_dict,
                force_local=force_local,
                wait=wait,
                save_rgb=save_rgb,
                save_gray=save_gray,
                save_detect_mat=save_detect_mat,
                save_objmask=save_objmask,
                save_objmap=save_objmap,
                save_objmap_overlay=save_objmap_overlay,
                save_detect_mat_overlay=save_detect_mat_overlay,
                save_objmask_overlay=save_objmask_overlay,
                rgb_ext=rgb_ext,
                gray_ext=gray_ext,
                detect_mat_ext=detect_mat_ext,
                objmask_ext=objmask_ext,
                objmap_ext=objmap_ext,
                objmap_overlay_ext=objmap_overlay_ext,
                overlay_mode=overlay_mode,
                overlay_alpha=overlay_alpha,
                include_dataset_column=include_dataset_column,
                dry_run=dry_run,
                sample=sample,
                resume=resume,
                retry_failures=retry_failures,
                skip_validation=skip_validation,
        )

        # Handle resume mode BEFORE creating output directory
        if config.resume:
            # For resume, output_dir must be specified
            if output_dir is None:
                click.echo(
                        "Error: --resume requires --output-dir to be specified",
                        err=True
                )
                click.echo(
                        "\nResume mode continues processing from a previous run. "
                        "You must specify the same output directory that was used before.",
                        err=True
                )
                click.echo(
                        "\nExample:\n"
                        "  python -m phenotypic pipeline.json ./images \\\n"
                        "    --output-dir ./results_2024-01-12_10-30-45 \\\n"
                        "    --resume",
                        err=True
                )
                sys.exit(1)

            # Check if output directory exists
            if not output_dir.exists():
                click.echo(
                        f"Error: Output directory does not exist: {output_dir}",
                        err=True
                )
                click.echo(
                        "\nCannot resume from a directory that doesn't exist. "
                        "Check the path and try again.",
                        err=True
                )
                sys.exit(1)

            # Check for processing state file
            state_file = output_dir / "processing_state.json"
            if not state_file.exists():
                click.echo(
                        f"Error: No processing state found in {output_dir}",
                        err=True
                )
                click.echo(
                        f"\nLooking for: {state_file}",
                        err=True
                )
                click.echo(
                        "\nThis directory may not contain PhenoTypic processing results, "
                        "or it was created with an older version that doesn't support resume.",
                        err=True
                )
                # List what's actually in the directory
                if output_dir.is_dir():
                    contents = list(output_dir.iterdir())
                    if contents:
                        click.echo(f"\nDirectory contents ({len(contents)} items):")
                        for item in sorted(contents)[:10]:  # Show first 10
                            click.echo(f"  - {item.name}")
                        if len(contents) > 10:
                            click.echo(f"  ... and {len(contents) - 10} more")
                sys.exit(1)

            click.echo(f"✓ Resuming from {output_dir}")

        # Handle restart mode - clear previous state
        if restart:
            state_file = output_dir / "processing_state.json"
            if output_dir.exists():
                if state_file.exists():
                    state_file.unlink()
                    click.echo(f"✓ Cleared previous processing state from {output_dir}")
                else:
                    click.echo(f"Note: No previous state found in {output_dir} (starting fresh)")
            else:
                click.echo(f"Note: Output directory {output_dir} does not exist yet (starting fresh)")

        # Generate or validate output directory
        if output_dir is None:
            output_dir = generate_timestamped_output_dir()
            click.echo(f"Auto-generated output directory: {output_dir}")

        config.output_dir = output_dir

        # Scan directory structure
        click.echo(f"Scanning {input_path}...")
        try:
            image_paths_by_dataset = scan_directory_structure(input_path)
            datasets = organize_by_dataset(
                    image_paths_by_dataset, output_dir
            )
        except (FileNotFoundError, ValueError) as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

        total_images = sum(len(d.images) for d in datasets)
        click.echo(f"Found {total_images} images in {len(datasets)} dataset(s)")

        # Validate configuration
        if not config.skip_validation:
            from rich.console import Console

            console = Console()
            console.print()  # Add blank line before validation

            # Step 1: Validate execution config
            with console.status("[bold cyan]Validating execution configuration...", spinner="dots"):
                config_valid, config_error = validate_execution_config(config)

            if not config_valid:
                console.print("[bold red]✗ Execution config validation failed:", style="bold red")
                console.print(f"  - {config_error}", style="red")
                sys.exit(1)
            console.print("[green]✓ Execution configuration validated")

            # Step 2: Validate pipeline loading
            with console.status("[bold cyan]Loading pipeline JSON...", spinner="dots"):
                pipeline_valid, pipeline_error = validate_pipeline(
                    config.pipeline_json,
                    config.skip_validation
                )

            if not pipeline_valid:
                console.print("[bold red]✗ Pipeline loading failed:", style="bold red")
                console.print(f"  - {pipeline_error}", style="red")
                sys.exit(1)
            console.print("[green]✓ Pipeline loaded successfully")

            # Step 3: Test pipeline on sample image
            test_image_path = None
            if datasets:
                for dataset in datasets:
                    if dataset.images:
                        test_image_path = dataset.images[0]
                        break

            if test_image_path:
                # Determine image class
                from phenotypic import Image, GridImage
                image_cls = GridImage if config.image_type == "GridImage" else Image

                # Prepare read kwargs
                read_kwargs = {}
                if config.image_type == "GridImage":
                    read_kwargs["nrows"] = config.nrows
                    read_kwargs["ncols"] = config.ncols
                if config.bit_depth is not None:
                    read_kwargs["bit_depth"] = config.bit_depth

                console.print(f"[cyan]Testing pipeline on: {test_image_path.name}")

                with console.status("[bold cyan]Processing test image...", spinner="dots"):
                    test_valid, test_error = validate_pipeline_on_test_image(
                        config.pipeline_json,
                        test_image_path,
                        image_cls,
                        read_kwargs,
                        config.skip_validation
                    )

                if not test_valid:
                    console.print("[bold red]✗ Test image processing failed:", style="bold red")
                    console.print(f"  - {test_error}", style="red")
                    sys.exit(1)
                console.print("[green]✓ Test image processed successfully")
            else:
                console.print("[yellow]⚠ No test image available - skipping pipeline test")

            console.print()  # Add blank line after validation
        else:
            from rich.console import Console
            console = Console()

        # Display execution configuration
        try:
            _display_execution_config(config, datasets)
        except Exception as e:
            click.echo(f"Error displaying configuration: {e}", err=True)
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # Handle dry-run mode
        if config.dry_run:
            execute_dry_run(config, datasets, output_dir)
            sys.exit(0)

        # Handle sample mode
        if config.sample is not None:
            click.echo(
                    f"\nSample mode: processing {config.sample} "
                    f"images per dataset"
            )
            datasets = get_sample_datasets(
                    datasets, config.sample, output_dir, random_seed
            )
            total_images = sum(len(d.images) for d in datasets)
            click.echo(f"Processing {total_images} sample images\n")

        # Handle resume mode - get remaining images
        if config.resume:
            # State was already validated earlier, just load it
            resume_state = load_processing_state(output_dir)
            if resume_state is None:
                click.echo(
                        f"Error: No processing state found in {output_dir}",
                        err=True
                )
                sys.exit(1)

            # Validate compatibility
            is_compatible, error = validate_resume_compatibility(resume_state, config)
            if not is_compatible:
                click.echo(f"Error: Cannot resume - {error}", err=True)
                click.echo(
                        "\nThe pipeline or configuration has changed since the "
                        "previous run. Resume is only possible with the same "
                        "pipeline and compatible settings.",
                        err=True
                )
                sys.exit(1)

            # Validate input image set hasn't changed
            images_valid, image_error = _validate_resume_input_images(
                    resume_state, datasets
            )
            if not images_valid:
                click.echo(f"Error: Cannot resume - {image_error}", err=True)
                click.echo(
                        "\nThe input image set has changed since the previous run. "
                        "Resume is only possible with the same input images.",
                        err=True
                )
                sys.exit(1)

            # Get remaining images
            datasets = get_remaining_images_for_datasets(
                    resume_state, datasets, config.retry_failures
            )
            remaining_images = sum(len(d.images) for d in datasets)

            if remaining_images == 0:
                click.echo("✓ All images already processed!")
                sys.exit(0)

            click.echo(
                    f"Resuming processing ({remaining_images} "
                    f"images remaining)"
            )
            if config.retry_failures:
                click.echo("  - Including previously failed images")

        # Ensure output directory exists for processing
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create initial state (or update if resuming)
        if config.resume:
            state = update_state_from_events(resume_state, output_dir)
            state.execution_mode = "slurm" if config.is_slurm_mode() else "local"
            state.pipeline_path = config.pipeline_json
            state.input_path = config.input_path
            state.output_dir = output_dir
            state.config = {
                "image_type": config.image_type,
                "nrows": config.nrows,
                "ncols": config.ncols,
                "bit_depth": config.bit_depth,
                "detect_mode": config.detect_mode,
                "n_jobs": config.n_jobs,
                "slurm_args": config.slurm_args,
                "save_layers": {
                    "rgb": config.save_rgb,
                    "gray": config.save_gray,
                    "detect_mat": config.save_detect_mat,
                    "objmask": config.save_objmask,
                    "objmap": config.save_objmap,
                    "objmap_overlay": config.save_objmap_overlay,
                    "detect_mat_overlay": config.save_detect_mat_overlay,
                    "objmask_overlay": config.save_objmask_overlay,
                },
            }
        else:
            state = create_initial_state(config, datasets, output_dir)
        save_processing_state(state, output_dir)

        # Create output manager
        output_manager = OutputManager(
                base_dir=output_dir,
                save_layers={
                    "rgb"              : config.save_rgb,
                    "gray"             : config.save_gray,
                    "detect_mat"         : config.save_detect_mat,
                    "objmask"          : config.save_objmask,
                    "objmap"           : config.save_objmap,
                    "objmap_overlay"   : config.save_objmap_overlay,
                    "detect_mat_overlay" : config.save_detect_mat_overlay,
                    "objmask_overlay"  : config.save_objmask_overlay,
                },
                extensions={
                    "rgb"            : config.rgb_ext,
                    "gray"           : config.gray_ext,
                    "detect_mat"       : config.detect_mat_ext,
                    "objmask"        : config.objmask_ext,
                    "objmap"         : config.objmap_ext,
                    "objmap_overlay" : config.objmap_overlay_ext,
                },
                include_dataset_column=config.include_dataset_column,
                overlay_mode=config.overlay_mode,
                overlay_alpha=config.overlay_alpha,
        )
        output_manager.create_structure(datasets)

        # Create execution strategy
        strategy = create_execution_strategy(config, output_manager)

        # Execute processing
        execution_mode = "SLURM" if config.is_slurm_mode() else "local"
        click.echo(f"\nStarting {execution_mode} processing...")

        results = strategy.execute(datasets, output_dir)

        # Aggregate master CSV (if we have completed results)
        if results.total_completed > 0:
            click.echo("\nAggregating measurements...")
            master_path = output_manager.aggregate_master_csv(datasets)
            if master_path:
                click.echo(f"✓ Master measurements: {master_path}")
            else:
                click.echo(
                        "⚠ Warning: Could not aggregate master CSV (check logs for details)",
                        err=True
                )

        # Generate HTML report
        click.echo("Generating HTML report...")
        report_gen = HTMLReportGenerator()
        report_path = output_dir / "processing_report.html"
        report_gen.generate_report(results, report_path)
        click.echo(f"✓ Report: {report_path}")

        # Generate README documentation
        click.echo("Generating README documentation...")
        try:
            from phenotypic._cli._cli_readme_generator import READMEGenerator

            pipeline = ImagePipeline.from_json(config.pipeline_json)
            readme_gen = READMEGenerator(config, pipeline)
            readme_path = readme_gen.generate(output_dir, datasets)
            click.echo(f"✓ README: {readme_path}")
        except Exception as e:
            logger.warning(f"Failed to generate README: {e}")
            click.echo(f"⚠ Warning: Could not generate README ({e})", err=True)

        # Print summary
        click.echo("\n" + "=" * 60)
        click.echo("PROCESSING COMPLETE")
        click.echo("=" * 60)
        click.echo(f"Completed: {results.total_completed}/{results.total_images}")
        click.echo(f"Failed:    {results.total_failed}")
        click.echo(
                f"Success rate: {results.success_rate * 100:.1f}%"
        )
        click.echo(f"Duration: {_format_duration(results.duration)}")
        click.echo(f"\nResults saved to: {output_dir}")

        # Exit with appropriate code
        sys.exit(0 if results.total_failed == 0 else 1)

    except KeyboardInterrupt:
        click.echo("\n\nInterrupted by user", err=True)
        sys.exit(130)
    except Exception as e:
        click.echo(f"\nUnexpected error: {e}", err=True)
        import traceback

        traceback.print_exc()
        sys.exit(1)


def _format_duration(seconds: float) -> str:
    """Format duration as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} min"
    else:
        return f"{seconds / 3600:.1f} hr"


if __name__ == "__main__":
    main()
