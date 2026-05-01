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
    - Progress monitoring tools

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
        --slurm slurm_partition=compute \
        --slurm slurm_account=proj \
        --slurm mem_gb=16

    # SLURM with progress monitoring
    uv run python -m phenotypic pipeline.json ./images \
        --slurm slurm_partition=compute \
        --slurm slurm_account=proj \
        --wait

    # GridImage with custom dimensions
    uv run python -m phenotypic pipeline.json ./plates \
        --image-type GridImage --nrows 16 --ncols 24

    # Rerun measurements on a previous forward run without re-detecting
    # (reads HDFs from <previous-output-dir>/results/*/hdf/, rewrites
    # parquet measurements + master CSV, skips detection, does NOT
    # regenerate overlays, does NOT touch processing state):
    uv run python -m phenotypic pipeline.json --measure \
        -o <previous-output-dir>

    # Recompile a previous output directory: re-aggregate the master
    # measurements CSV, fill in any overlay PNGs missing under
    # results/<ds>/overlays/ by reloading their HDFs (threaded across
    # --n-jobs workers, alpha from --overlay-alpha), rerun analysis
    # plugins, rebuild the manifest, and regenerate the HTML dashboard.
    # Existing overlays are left untouched.  Pipeline JSON is NOT
    # required:
    uv run python -m phenotypic --recompile <previous-output-dir>

Outputs:
    Forward runs write a single HDF5 per input image under
    `<output>/results/<dataset>/hdf/<stem>.h5` (layers + metadata + grid
    state, reloadable via `Image.load_hdf5` / `GridImage.load_hdf5`).
    Overlay PNGs are always written under
    `<output>/results/<dataset>/overlays/<stem>.png` for forward runs;
    `--measure` reruns reuse existing overlays and do not regenerate them.
    `--recompile` fills in only-missing overlay PNGs from HDFs but
    leaves existing ones untouched.

SLURM Execution (Autonomous HPC Cluster Processing):
    Use --slurm to submit jobs to an HPC cluster via SLURM. The CLI will:
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
            --slurm slurm_partition=compute \\
            --slurm slurm_account=lab_proj \\
            --slurm mem_gb=32 \\
            --slurm time=120 \\
            --slurm slurm_mail_type=END \\
            --slurm slurm_mail_user=user@university.edu \\
            --wait

    Example: Dry-run to preview SLURM submission plan
        uv run python -m phenotypic pipeline.json ./images \\
            --slurm slurm_partition=compute \\
            --slurm slurm_account=lab_proj \\
            --dry-run
"""

import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import click

from phenotypic import ImagePipeline
from phenotypic._core._image_parts.detection_modes import available_modes
from phenotypic._cli._cli_directory_scanner import (
    generate_timestamped_output_dir,
    organize_by_dataset,
    scan_directory_structure,
    scan_hdf_outputs,
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
from phenotypic._cli._cli_utils import (
    normalize_extension,
    parse_slurm_args,
    resolve_local_worker_count,
)
from phenotypic._cli._cli_recompile_slurm_scripts import (
    TASK_FINALIZE,
    build_recompile_tasks,
    generate_recompile_slurm_scripts,
)
from phenotypic._cli._cli_slurm_config import get_slurm_array_limit
from phenotypic._cli._cli_slurm_submission import submit_slurm_script_chain
from phenotypic._cli._cli_validation import (
    validate_execution_config,
    validate_pipeline,
)
from phenotypic._cli._cli_constants import (
    MIN_SLURM_TIME_MINUTES,
    MAX_SLURM_TIME_MINUTES,
)

# Set up logger
logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False):
    """Configure logging for CLI."""
    level = logging.DEBUG if debug else logging.INFO
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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
    """Parse space-separated KEY=VALUE pairs into dictionary.

    Thin wrapper around :func:`phenotypic._cli._cli_utils.parse_slurm_args`
    kept for backward compatibility within this module.
    """
    return parse_slurm_args(slurm_args)


def _validate_resume_input_images(state, current_datasets) -> tuple[bool, Optional[str]]:
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
            return (
                False,
                f"Dataset '{ds_name}' from previous run not found in input directory",
            )

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
                error_parts.append(
                    f"  - Missing {len(missing)} images (e.g., {list(missing)[:3]})"
                )
            if added:
                error_parts.append(
                    f"  - Added {len(added)} new images (e.g., {list(added)[:3]})"
                )

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


def _display_execution_config(config: ExecutionConfig, datasets: list) -> None:
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
        title="Execution Configuration", show_header=False, box=None, padding=(0, 2)
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
        if config.nrows is None and config.ncols is None:
            grid_str = "auto (from pipeline preset, default 8 × 12)"
        else:
            nr = "auto" if config.nrows is None else str(config.nrows)
            nc = "auto" if config.ncols is None else str(config.ncols)
            grid_str = f"{nr} × {nc}"
        table.add_row("Grid Dimensions", grid_str)
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
    default=None,
    required=False,
)
@click.argument(
    "input_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=True, path_type=Path),
    default=None,
    required=False,
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
    default=None,
    help="Number of rows for GridImage. Overrides any pipeline-level "
    "preset; when omitted, the pipeline preset is used, falling back to 8.",
)
@click.option(
    "--ncols",
    type=click.IntRange(min=1),
    default=None,
    help="Number of columns for GridImage. Overrides any pipeline-level "
    "preset; when omitted, the pipeline preset is used, falling back to 12.",
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
    "--slurm",
    "slurm_args",
    multiple=True,
    help="SLURM parameters as KEY=VALUE pairs. Pass multiple parameters with "
    "repeated --slurm flags (e.g., --slurm slurm_partition=compute "
    "--slurm mem_gb=16 --slurm time=60). Use slurm_ prefix for "
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
    "--ext",
    default="tiff",
    show_default=True,
    help="(deprecated for HDF output; still used for overlay PNG) "
    "File extension for legacy per-layer outputs. Forward runs now "
    "write a single .h5 per image; only overlay PNG rendering still "
    "consults this value.",
)
@click.option(
    "--measure",
    "measure_only",
    is_flag=True,
    default=False,
    help="Rerun pipeline.measure() on HDFs under "
    "<output-dir>/results/*/hdf/. Skips detection. Does not regenerate "
    "overlays.",
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
    "no_dataset_column",
    is_flag=True,
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
    "--overwrite",
    is_flag=True,
    help="Delete existing output directory contents before processing",
)
@click.option(
    "--metadata",
    "metadata_csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="CSV file to inner-join onto master_measurements.csv on shared columns",
)
@click.option(
    "--checkpoint-interval",
    type=int,
    default=None,
    help="Insert checkpoint tasks every N images in SLURM arrays (default: auto-estimate)",
)
@click.option(
    "--skip-validation",
    is_flag=True,
    help="Skip pipeline validation (for advanced users)",
)
@click.option(
    "--recompile",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Recompile master measurements from existing output directory. "
         "Skips pipeline execution — re-aggregates parquets, regenerates "
         "any overlay PNGs missing under results/<ds>/overlays/ from "
         "their HDFs (threaded by --n-jobs, alpha from --overlay-alpha), "
         "runs analysis, rebuilds manifest, and regenerates dashboard.",
)
def phenotypic_cli(
    pipeline_json: Optional[Path],
    input_path: Optional[Path],
    output_dir: Optional[Path],
    image_type: str,
    nrows: Optional[int],
    ncols: Optional[int],
    bit_depth: Optional[int],
    detect_mode: str,
    n_jobs: int,
    slurm_args: Sequence[str],
    force_local: bool,
    wait: bool,
    ext: str,
    measure_only: bool,
    overlay_alpha: float,
    no_dataset_column: bool,
    dry_run: bool,
    sample: Optional[int],
    random_seed: Optional[int],
    resume: bool,
    retry_failures: bool,
    restart: bool,
    overwrite: bool,
    metadata_csv: Optional[Path],
    checkpoint_interval: Optional[int],
    skip_validation: bool,
    recompile: Optional[Path],
):
    """
    Execute a PhenoTypic pipeline on images.

    PIPELINE_JSON: Path to pipeline configuration file

    INPUT_PATH: Image file or directory to process
    """
    try:
        include_dataset_column = not no_dataset_column

        # Parse SLURM args before the recompile branch so --recompile can
        # explicitly choose between local and SLURM recompile dispatch.
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
                    "Warning: 'time_min' is deprecated. Use 'time' instead.", err=True
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
                            err=True,
                        )
                        sys.exit(1)

                    # Validate reasonable time range
                    if time_val < MIN_SLURM_TIME_MINUTES:
                        click.echo(
                            f"Error: '{time_key}' must be >= {MIN_SLURM_TIME_MINUTES} minute, got {time_val}",
                            err=True,
                        )
                        sys.exit(1)
                    elif time_val > MAX_SLURM_TIME_MINUTES:
                        days = MAX_SLURM_TIME_MINUTES / 1440
                        click.echo(
                            f"Warning: '{time_key}' is {time_val} minutes "
                            f"({time_val / 60:.1f} hours). This exceeds typical cluster limits "
                            f"({MAX_SLURM_TIME_MINUTES} min / {days:.1f} days).",
                            err=True,
                        )

        if recompile is not None:
            if slurm_args_dict and not force_local:
                _handle_recompile_slurm(
                    output_dir=recompile,
                    metadata_csv=metadata_csv,
                    include_dataset_column=include_dataset_column,
                    overlay_alpha=overlay_alpha,
                    checkpoint_interval=checkpoint_interval,
                    slurm_args=slurm_args_dict,
                    wait=wait,
                )
            else:
                _handle_recompile(
                    recompile,
                    metadata_csv,
                    include_dataset_column,
                    overlay_alpha,
                    n_jobs,
                )
            sys.exit(0)

        # ---- Early validation for --measure (measure_only) -------------
        # --measure is a one-shot re-measurement run over HDFs already
        # written by a previous forward run.  It is incompatible with any
        # flag that implies a fresh detection pass or state mutation, and
        # it uses <output-dir>/results/*/hdf/ as its image source, so the
        # positional INPUT_PATH becomes optional.
        if measure_only:
            # Reject incompatible flags first so the user gets a pointed
            # rejection ("--measure cannot be combined with --X") regardless
            # of whether --output-dir / PIPELINE_JSON are also wrong.
            if resume:
                raise click.UsageError(
                    "--measure cannot be combined with --resume; "
                    "--measure is a one-shot re-measurement run that does "
                    "not touch processing state."
                )
            if restart:
                raise click.UsageError(
                    "--measure cannot be combined with --restart; "
                    "--measure reuses existing HDFs and does not clear state."
                )
            if retry_failures:
                raise click.UsageError(
                    "--measure cannot be combined with --retry-failures; "
                    "--retry-failures only applies to resume runs, and "
                    "--measure does not touch state."
                )
            if overwrite:
                raise click.UsageError(
                    "--measure cannot be combined with --overwrite; "
                    "--measure reruns measurements on existing HDFs and "
                    "must not delete output directory contents."
                )
            if sample is not None:
                raise click.UsageError(
                    "--measure cannot be combined with --sample; "
                    "--measure operates on every HDF discovered under "
                    "<output-dir>/results/*/hdf/."
                )

            if pipeline_json is None:
                raise click.UsageError(
                    "--measure requires PIPELINE_JSON to be specified."
                )
            if output_dir is None:
                raise click.UsageError(
                    "--measure requires --output-dir to be specified "
                    "(it reads HDFs from <output-dir>/results/*/hdf/)."
                )
            if not Path(output_dir).exists():
                raise click.UsageError(
                    f"--measure output directory does not exist: {output_dir}. "
                    "--measure is a rerun over an existing forward-run output; "
                    "point it at a directory produced by a previous "
                    "`python -m phenotypic ...` invocation."
                )
            if input_path is not None:
                click.echo(
                    f"Warning: INPUT_PATH ({input_path}) is ignored in "
                    "--measure mode; images are discovered under "
                    f"{output_dir}/results/*/hdf/.",
                    err=True,
                )

        if not measure_only and (pipeline_json is None or input_path is None):
            raise click.UsageError(
                "PIPELINE_JSON and INPUT_PATH are required unless --recompile is set."
            )

        resume_state = None

        # Validate extension argument
        try:
            ext = normalize_extension(ext, ".tiff")
        except click.BadParameter as e:
            click.echo(str(e), err=True)
            sys.exit(1)

        # Validate flags
        if retry_failures and not resume:
            click.echo("Error: --retry-failures requires --resume", err=True)
            sys.exit(1)

        if restart and resume:
            click.echo("Error: --restart and --resume are mutually exclusive", err=True)
            sys.exit(1)

        if overwrite and resume:
            click.echo(
                "Error: --overwrite and --resume are mutually exclusive", err=True
            )
            sys.exit(1)

        if restart and output_dir is None:
            click.echo(
                "Error: --restart requires --output-dir to be specified", err=True
            )
            click.echo(
                "\nRestart mode clears previous processing state and starts fresh. "
                "You must specify the output directory to restart.",
                err=True,
            )
            click.echo(
                "\nExample:\n"
                "  python -m phenotypic pipeline.json ./images \\\n"
                "    --output-dir ./results_2024-01-12_10-30-45 \\\n"
                "    --restart",
                err=True,
            )
            sys.exit(1)

        # Validate metadata CSV early
        if metadata_csv is not None:
            import pandas as pd
            try:
                meta_df = pd.read_csv(metadata_csv)
                if len(meta_df) == 0:
                    click.echo(
                        f"Warning: metadata CSV '{metadata_csv}' has zero rows",
                        err=True,
                    )
            except Exception as e:
                error_exit(f"Cannot read metadata CSV: {e}")

        # Create ExecutionConfig.  By this point either measure_only's
        # early validation or the non-measure check above has guaranteed
        # pipeline_json is non-None; input_path may only be None in
        # --measure mode, where we substitute output_dir to satisfy the
        # dataclass's non-optional ``input_path`` (the measure path
        # never consults it for image discovery).
        assert pipeline_json is not None  # narrowed by earlier UsageError branches
        effective_input_path = input_path if input_path is not None else output_dir
        assert effective_input_path is not None  # narrowed by earlier --output-dir checks
        config = ExecutionConfig(
            pipeline_json=pipeline_json,
            input_path=effective_input_path,
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
            ext=ext,
            overlay_alpha=overlay_alpha,
            include_dataset_column=include_dataset_column,
            dry_run=dry_run,
            sample=sample,
            resume=resume,
            retry_failures=retry_failures,
            skip_validation=skip_validation,
            metadata_csv=metadata_csv,
            checkpoint_interval=checkpoint_interval,
            measure_only=measure_only,
        )

        # Handle resume mode BEFORE creating output directory
        if config.resume:
            # For resume, output_dir must be specified
            if output_dir is None:
                click.echo(
                    "Error: --resume requires --output-dir to be specified", err=True
                )
                click.echo(
                    "\nResume mode continues processing from a previous run. "
                    "You must specify the same output directory that was used before.",
                    err=True,
                )
                click.echo(
                    "\nExample:\n"
                    "  python -m phenotypic pipeline.json ./images \\\n"
                    "    --output-dir ./results_2024-01-12_10-30-45 \\\n"
                    "    --resume",
                    err=True,
                )
                sys.exit(1)

            # Check if output directory exists
            if not output_dir.exists():
                click.echo(
                    f"Error: Output directory does not exist: {output_dir}", err=True
                )
                click.echo(
                    "\nCannot resume from a directory that doesn't exist. "
                    "Check the path and try again.",
                    err=True,
                )
                sys.exit(1)

            # Check for processing state file
            state_file = output_dir / "processing_state.json"
            if not state_file.exists():
                click.echo(f"Error: No processing state found in {output_dir}", err=True)
                click.echo(f"\nLooking for: {state_file}", err=True)
                click.echo(
                    "\nThis directory may not contain PhenoTypic processing results, "
                    "or it was created with an older version that doesn't support resume.",
                    err=True,
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
                    click.echo(
                        f"Note: No previous state found in {output_dir} (starting fresh)"
                    )
            else:
                click.echo(
                    f"Note: Output directory {output_dir} does not exist yet (starting fresh)"
                )

        # Generate or validate output directory
        if output_dir is None:
            output_dir = generate_timestamped_output_dir()
            click.echo(f"Auto-generated output directory: {output_dir}")

        config.output_dir = output_dir

        # Check for existing output directory contents (skip for resume/restart/measure)
        if not config.resume and not restart and not measure_only:
            if output_dir.exists() and any(output_dir.iterdir()):
                if overwrite:
                    import shutil

                    click.echo(
                        f"Overwriting existing output directory: {output_dir}"
                    )
                    shutil.rmtree(output_dir)
                else:
                    click.echo(
                        f"Error: Output directory already contains files: {output_dir}",
                        err=True,
                    )
                    click.echo(
                        "\nUse --overwrite to delete existing contents and replace them, "
                        "or choose a different output directory.",
                        err=True,
                    )
                    sys.exit(1)

        # Scan directory structure (or discover HDFs if in --measure mode)
        if measure_only:
            click.echo(f"Discovering HDF outputs under {output_dir}/results/...")
            try:
                datasets = scan_hdf_outputs(output_dir)
            except ValueError as e:
                click.echo(f"Error: {e}", err=True)
                sys.exit(1)
        else:
            # Not measure_only → input_path already validated as non-None.
            assert input_path is not None
            click.echo(f"Scanning {input_path}...")
            try:
                image_paths_by_dataset = scan_directory_structure(input_path)
                datasets = organize_by_dataset(image_paths_by_dataset, output_dir)
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
            with console.status(
                "[bold cyan]Validating execution configuration...", spinner="dots"
            ):
                config_valid, config_error = validate_execution_config(config)

            if not config_valid:
                console.print(
                    "[bold red]✗ Execution config validation failed:", style="bold red"
                )
                console.print(f"  - {config_error}", style="red")
                sys.exit(1)
            console.print("[green]✓ Execution configuration validated")

            # Step 2: Validate pipeline loading
            with console.status("[bold cyan]Loading pipeline JSON...", spinner="dots"):
                pipeline_valid, pipeline_error = validate_pipeline(
                    config.pipeline_json, config.skip_validation
                )

            if not pipeline_valid:
                console.print("[bold red]✗ Pipeline loading failed:", style="bold red")
                console.print(f"  - {pipeline_error}", style="red")
                sys.exit(1)
            console.print("[green]✓ Pipeline loaded successfully")

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
            click.echo(f"\nSample mode: processing {config.sample} images per dataset")
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
                click.echo(f"Error: No processing state found in {output_dir}", err=True)
                sys.exit(1)

            # Validate compatibility
            is_compatible, error = validate_resume_compatibility(resume_state, config)
            if not is_compatible:
                click.echo(f"Error: Cannot resume - {error}", err=True)
                click.echo(
                    "\nThe pipeline or configuration has changed since the "
                    "previous run. Resume is only possible with the same "
                    "pipeline and compatible settings.",
                    err=True,
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
                    err=True,
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

            click.echo(f"Resuming processing ({remaining_images} images remaining)")
            if config.retry_failures:
                click.echo("  - Including previously failed images")

        # Ensure output directory exists for processing
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create initial state (or update if resuming) — skipped in
        # --measure mode, which never mutates processing state.
        if not measure_only:
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
                    "ext": config.ext,
                    "save_overlays": config.save_overlays,
                }
            else:
                state = create_initial_state(config, datasets, output_dir)
                # Recorded for diagnostic transparency; no longer drives
                # resume compatibility (overlays are always-on).
                state.config["save_overlays"] = config.save_overlays
            save_processing_state(state, output_dir)

        # Create output manager
        output_manager = OutputManager.from_config(
            base_dir=output_dir,
            ext=config.ext,
            include_dataset_column=config.include_dataset_column,
            overlay_alpha=config.overlay_alpha,
            save_overlays=config.save_overlays,
        )
        output_manager.create_structure(datasets)

        # Copy pipeline JSON to output directory for reproducibility
        # (skip in --measure mode — the forward run already copied it).
        if not measure_only:
            try:
                copied = _copy_pipeline_to_output(config.pipeline_json, output_dir)
                if copied:
                    click.echo(f"  Pipeline: {copied}")
            except OSError as e:
                logger.warning(f"Failed to copy pipeline JSON: {e}")
                click.echo(f"⚠ Warning: Could not copy pipeline JSON ({e})", err=True)

        # Create execution strategy
        strategy = create_execution_strategy(config, output_manager)

        # Execute processing
        execution_mode = "SLURM" if config.is_slurm_mode() else "local"
        click.echo(f"\nStarting {execution_mode} processing...")

        results = strategy.execute(datasets, output_dir)

        # Load pipeline once for the finalizer — reused by both the
        # per-feature split in aggregate_master_csv and the README generator.
        finalizer_pipeline: Optional[ImagePipeline] = None
        try:
            finalizer_pipeline = ImagePipeline.from_json(config.pipeline_json)
        except Exception as e:
            logger.warning(f"Failed to load pipeline for finalizer: {e}")

        # Aggregate master CSV (if we have completed results)
        if results.total_completed > 0:
            click.echo("\nAggregating measurements...")
            master_path = output_manager.aggregate_master_csv(
                datasets,
                metadata_csv=config.metadata_csv,
                pipeline=finalizer_pipeline,
            )
            if master_path:
                click.echo(f"✓ Master measurements: {master_path}")
            else:
                click.echo(
                    "⚠ Warning: Could not aggregate master CSV (check logs for details)",
                    err=True,
                )

            # Write analysis sidecar data for the dashboard
            try:
                from phenotypic._cli._dashboard._analysis_data import write_analysis_sidecar
                write_analysis_sidecar(output_dir, metadata_csv=config.metadata_csv)
            except Exception:
                logger.warning("Analysis sidecar write failed", exc_info=True)

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

            readme_pipeline = (
                finalizer_pipeline
                if finalizer_pipeline is not None
                else ImagePipeline.from_json(config.pipeline_json)
            )
            readme_gen = READMEGenerator(config, readme_pipeline)
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
        click.echo(f"Success rate: {results.success_rate * 100:.1f}%")
        click.echo(f"Duration: {_format_duration(results.duration)}")
        click.echo(f"\nResults saved to: {output_dir}")

        # Exit with appropriate code
        sys.exit(0 if results.total_failed == 0 else 1)

    except KeyboardInterrupt:
        click.echo("\n\nInterrupted by user", err=True)
        sys.exit(130)
    except click.UsageError:
        # Let Click format UsageError (standard "Usage: ..." + "Error: ..."
        # two-line output); do NOT swallow into the "Unexpected error"
        # branch below.
        raise
    except Exception as e:
        click.echo(f"\nUnexpected error: {e}", err=True)
        import traceback

        traceback.print_exc()
        sys.exit(1)


def _regenerate_missing_overlays(
    output_dir: Path,
    overlay_alpha: float,
    n_jobs: int,
) -> None:
    """Re-render overlay PNGs that are missing under ``results/<ds>/overlays/``.

    HDF-driven: walks ``results/<ds>/hdf/*.h5`` (the same discovery
    used by ``--measure``) and, for each HDF whose corresponding
    overlay PNG is absent, loads the HDF as the right ``Image`` /
    ``GridImage`` subclass and writes the overlay using the same
    :class:`OutputManager` writer the forward run uses.  Existing
    overlays are left untouched.  Per-image failures are logged and
    swallowed so one corrupt HDF doesn't abort the rest.

    Parallelized with a thread pool sized by ``n_jobs``.  Threading
    rather than multiprocessing because the heavy ops (h5py reads,
    numpy/skimage label2rgb compositing, PNG zlib encoding) all
    release the GIL, and per-image memory is large enough that
    fan-out to processes risks exhausting RAM.

    Args:
        output_dir: Existing output directory.
        overlay_alpha: Alpha for overlay compositing, mirrors the
            ``--overlay-alpha`` flag used by forward runs.
        n_jobs: Number of worker threads.  ``-1`` means allocated CPUs
            under SLURM, otherwise host CPUs.  ``1`` runs in-thread.
            Mirrors the ``--n-jobs`` flag used by forward runs.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import h5py
    from rich.console import Console
    from rich.progress import (
        BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn,
    )

    from phenotypic import GridImage, Image

    console = Console()
    try:
        datasets = scan_hdf_outputs(output_dir)
    except ValueError:
        console.print(
            "[yellow]No HDFs found under results/; skipping overlay regeneration"
        )
        return

    output_manager = OutputManager.from_config(
        base_dir=output_dir,
        ext=".png",
        include_dataset_column=False,
        overlay_alpha=overlay_alpha,
        save_overlays=True,
    )

    work: list[tuple[str, Path]] = []
    for dataset in datasets:
        for hdf_path in dataset.images:
            overlay_path = output_manager.get_output_path(
                dataset.name, "overlays", hdf_path.stem
            )
            if not overlay_path.exists():
                work.append((dataset.name, hdf_path))

    if not work:
        console.print("[green]All overlays present; nothing to regenerate")
        return

    for dataset_name in {ds for ds, _ in work}:
        (output_dir / "results" / dataset_name / "overlays").mkdir(
            parents=True, exist_ok=True
        )

    workers = resolve_local_worker_count(n_jobs, len(work))

    def _render_one(dataset_name: str, hdf_path: Path) -> None:
        with h5py.File(hdf_path, "r") as fh:
            cls_attr = fh.attrs.get("phenotypic_class", "Image")
        if isinstance(cls_attr, bytes):
            cls_attr = cls_attr.decode("utf-8", errors="replace")
        image_cls = GridImage if cls_attr == "GridImage" else Image
        image = image_cls.load_hdf5(hdf_path)
        output_manager.save_overlay(image, dataset_name, hdf_path.stem)

    console.print(
        f"[cyan]Regenerating {len(work)} missing overlay(s) "
        f"with {workers} thread(s)..."
    )
    failures = 0
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("overlays", total=len(work))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_render_one, ds_name, hdf_path): (ds_name, hdf_path)
                for ds_name, hdf_path in work
            }
            for future in as_completed(futures):
                ds_name, hdf_path = futures[future]
                try:
                    future.result()
                except Exception:
                    failures += 1
                    logger.warning(
                        "Failed to regenerate overlay for %s/%s",
                        ds_name,
                        hdf_path.stem,
                        exc_info=True,
                    )
                finally:
                    progress.advance(task_id)

    if failures:
        console.print(
            f"[yellow]Overlay regeneration finished with {failures} failure(s); "
            f"see logs for details"
        )
    else:
        console.print("[green]Overlay regeneration complete")


def _handle_recompile_slurm(
    *,
    output_dir: Path,
    metadata_csv: Optional[Path],
    include_dataset_column: bool,
    overlay_alpha: float,
    checkpoint_interval: Optional[int],
    slurm_args: dict[str, Any],
    wait: bool,
) -> None:
    """Submit recompile work as a SLURM array chain.

    Args:
        output_dir: Existing output directory containing ``results/``.
        metadata_csv: Optional metadata CSV for the finalizer task.
        include_dataset_column: Whether measurement aggregation should add
            ``Metadata_Dataset`` when source files lack it.
        overlay_alpha: Alpha used when re-rendering missing overlays.
        checkpoint_interval: Reused as the measurement shard size when
            positive; otherwise defaults to 500 files per shard.
        slurm_args: Parsed SLURM key/value arguments.
        wait: Whether to wait for the finalizer task status.
    """
    import json
    from datetime import datetime

    from rich.console import Console

    from phenotypic._cli._cli_utils import load_job_metadata
    from phenotypic._cli._dashboard import generate_dashboard

    output_dir = Path(output_dir)
    progress_dir = output_dir / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    console = Console()

    job_meta = load_job_metadata(progress_dir)
    dataset_names = _discover_recompile_dataset_names(output_dir, job_meta)
    if not dataset_names:
        error_exit("No datasets found in output directory", str(output_dir))

    shard_size = (
        checkpoint_interval
        if checkpoint_interval is not None and checkpoint_interval > 0
        else 500
    )
    console.print(
        f"[bold]Submitting SLURM recompile[/bold] for {output_dir} "
        f"({len(dataset_names)} dataset(s))"
    )

    tasks = build_recompile_tasks(
        output_dir=output_dir,
        dataset_names=dataset_names,
        include_dataset_column=include_dataset_column,
        overlay_alpha=overlay_alpha,
        shard_size=shard_size,
    )
    finalizer_task_index: Optional[int] = None
    if tasks and tasks[-1].get("task_type") == TASK_FINALIZE:
        finalizer_task_index = len(tasks) - 1
        tasks[-1]["metadata_csv"] = str(metadata_csv) if metadata_csv else None

    console.print("[cyan]Querying SLURM array limits...[/cyan]")
    array_limit = get_slurm_array_limit()
    console.print(f"[green]SLURM array limit: {array_limit}[/green]")

    scripts = generate_recompile_slurm_scripts(
        tasks=tasks,
        output_dir=output_dir,
        slurm_args=slurm_args,
        array_limit=array_limit,
    )
    if not tasks or not scripts or finalizer_task_index is None:
        console.print(
            "[yellow]No recompile SLURM tasks/scripts were generated; "
            "running local recompile instead.[/yellow]"
        )
        _handle_recompile(
            output_dir,
            metadata_csv,
            include_dataset_column,
            overlay_alpha,
            -1,
        )
        return

    submission = submit_slurm_script_chain(
        flat_chunk_scripts=scripts,
        output_dir=output_dir,
        slurm_args=slurm_args,
        console=console,
    )
    flat_scripts = submission.flat_scripts
    job_ids = submission.job_ids

    recompile_manifest_path = (
        output_dir / "progress" / "recompile" / "task_manifest.json"
    )
    job_metadata = {
        "start_time": datetime.now().isoformat(timespec="milliseconds"),
        "execution_mode": "slurm",
        "datasets": _build_recompile_job_metadata_datasets(
            output_dir,
            dataset_names,
            job_meta,
        ),
        "chunk_scripts": [str(script) for script in flat_scripts],
        "chunk_job_ids": {"0": str(job_ids[0])},
        "include_dataset_column": include_dataset_column,
        "metadata_csv": str(metadata_csv) if metadata_csv else None,
        "input_path": str(output_dir),
        "recompile": {
            "task_manifest": str(recompile_manifest_path),
            "finalizer_task_index": finalizer_task_index,
        },
    }
    metadata_path = progress_dir / "job_metadata.json"
    metadata_path.write_text(
        json.dumps(job_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    console.print(f"[green]Job metadata: {metadata_path}[/green]")

    generate_dashboard(output_dir, execution_mode="slurm")
    console.print(
        f"[green]Dashboard: {output_dir / 'dashboard.html'}[/green]"
    )

    if wait:
        console.print(
            "\n[cyan]Waiting for recompile finalizer "
            f"task {finalizer_task_index}...[/cyan]"
        )
        _wait_for_recompile_finalizer_status(output_dir, finalizer_task_index)
        console.print("[bold green]SLURM recompilation complete[/bold green]")
    else:
        click.echo("\nRecompile jobs submitted. Monitor progress with:")
        click.echo(f"  Open: {output_dir / 'dashboard.html'}")
        click.echo("  squeue -u $USER --array")
        click.echo(
            f"  tail -f {output_dir / 'logs' / 'slurm' / 'recompile'}/*.log"
        )


def _discover_recompile_dataset_names(
    output_dir: Path,
    job_meta: Optional[dict[str, Any]],
) -> list[str]:
    """Discover datasets using the local recompile precedence."""
    results_dir = output_dir / "results"
    dataset_names: list[str] = []
    if results_dir.is_dir():
        dataset_names = sorted(
            d.name
            for d in results_dir.iterdir()
            if d.is_dir() and (d / "measurements").is_dir()
        )

    if not dataset_names and job_meta:
        dataset_names = sorted((job_meta.get("datasets", {}) or {}).keys())

    return dataset_names


def _build_recompile_job_metadata_datasets(
    output_dir: Path,
    dataset_names: list[str],
    job_meta: Optional[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build the normal nested ``job_metadata["datasets"]`` shape."""
    previous_datasets = (job_meta or {}).get("datasets", {}) or {}
    datasets: dict[str, dict[str, Any]] = {}
    for dataset_name in dataset_names:
        images = _recompile_dataset_image_names(output_dir, dataset_name)
        if not images:
            previous = previous_datasets.get(dataset_name, {})
            if isinstance(previous, dict):
                previous_images = previous.get("images", [])
                if isinstance(previous_images, list):
                    images = [str(image) for image in previous_images]
        datasets[dataset_name] = {"total": len(images), "images": images}
    return datasets


def _recompile_dataset_image_names(output_dir: Path, dataset_name: str) -> list[str]:
    """Infer image names from per-image measurement Parquets or HDFs."""
    dataset_dir = output_dir / "results" / dataset_name
    meas_dir = dataset_dir / "measurements"
    if meas_dir.is_dir():
        parquet_stems = sorted(
            path.stem
            for path in meas_dir.glob("*.parquet")
            if not path.name.startswith("_")
        )
        if parquet_stems:
            return parquet_stems

    hdf_dir = dataset_dir / "hdf"
    if hdf_dir.is_dir():
        hdf_stems = sorted(path.stem for path in hdf_dir.glob("*.h5"))
        if hdf_stems:
            return hdf_stems

    return []


def _wait_for_recompile_finalizer_status(
    output_dir: Path,
    finalizer_task_index: int,
    poll_interval: float = 10.0,
    timeout: Optional[float] = None,
) -> None:
    """Wait until the recompile finalizer status is completed or failed."""
    import json
    import time

    status_path = (
        output_dir
        / "progress"
        / "recompile"
        / "status"
        / f"task_{finalizer_task_index}.json"
    )
    deadline = time.monotonic() + timeout if timeout is not None else None
    while True:
        if status_path.exists():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("Failed to read recompile status %s", status_path)
            else:
                state = status.get("status")
                if state == "completed":
                    return
                if state == "failed":
                    error = status.get("error") or "recompile finalizer failed"
                    raise RuntimeError(str(error))

        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for recompile finalizer status: {status_path}"
            )
        time.sleep(poll_interval)


def _handle_recompile(
    output_dir: Path,
    metadata_csv: Optional[Path],
    include_dataset_column: bool,
    overlay_alpha: float,
    n_jobs: int,
) -> None:
    """Recompile master measurements and dashboard from existing results.

    Auto-discovers datasets under ``output_dir/results``, re-aggregates
    measurement Parquet files into ``master_measurements.csv``,
    regenerates any missing overlay PNGs from their HDFs, runs
    analysis plugins, rebuilds the progress manifest, and regenerates
    the HTML dashboard.

    Args:
        output_dir: Existing output directory containing ``results/``.
        metadata_csv: Optional path to an external metadata CSV for
            left-joining onto measurements.
        include_dataset_column: Whether to insert ``Metadata_Dataset``
            into measurements that lack it.
        overlay_alpha: Alpha used when re-rendering missing overlay
            PNGs from HDFs.  Forwarded from the ``--overlay-alpha``
            CLI flag.
        n_jobs: Worker thread count for overlay regeneration.
            Forwarded from the ``--n-jobs`` CLI flag (``-1`` = all
            cores, ``1`` = single-threaded).
    """
    from rich.console import Console

    from phenotypic._cli._cli_chunk_writer import _run_analysis_plugins
    from phenotypic._cli._cli_output_manager import aggregate_measurements
    from phenotypic._cli._cli_utils import load_job_metadata
    from phenotypic._cli._dashboard import (
        build_manifest,
        generate_dashboard,
    )

    console = Console()
    results_dir = output_dir / "results"
    progress_dir = output_dir / "progress"
    job_meta = load_job_metadata(progress_dir)

    dataset_names: list[str] = []
    if results_dir.is_dir():
        dataset_names = sorted(
            d.name
            for d in results_dir.iterdir()
            if d.is_dir() and (d / "measurements").is_dir()
        )

    if not dataset_names and job_meta:
        dataset_names = sorted(job_meta.get("datasets", {}).keys())

    if not dataset_names:
        error_exit("No datasets found in output directory", str(output_dir))

    console.print(
        f"[bold]Recompiling[/bold] from {output_dir} "
        f"({len(dataset_names)} dataset(s))"
    )

    console.print("[cyan]Aggregating measurements...")
    master_path = aggregate_measurements(
        output_dir=output_dir,
        dataset_names=dataset_names,
        include_dataset_column=include_dataset_column,
        metadata_csv=metadata_csv,
    )
    if master_path:
        console.print(f"[green]Master measurements: {master_path}")
    else:
        console.print("[yellow]No measurements found for aggregation")

    console.print("[cyan]Checking for missing overlays...")
    _regenerate_missing_overlays(output_dir, overlay_alpha, n_jobs)

    console.print("[cyan]Running analysis plugins...")
    try:
        import polars as pl

        merged_df: Optional[pl.DataFrame] = None
        if master_path and Path(master_path).exists():
            try:
                merged_df = pl.read_csv(master_path)
            except Exception:
                logger.warning(
                    "Failed to read master CSV for analysis plugins",
                    exc_info=True,
                )
        _run_analysis_plugins(output_dir, progress_dir, merged_df)
        console.print("[green]Analysis plugins complete")
    except Exception:
        logger.warning("Analysis plugin dispatch failed", exc_info=True)
        console.print("[yellow]Analysis plugin dispatch failed (see logs)")

    console.print("[cyan]Rebuilding manifest...")
    progress_dir.mkdir(parents=True, exist_ok=True)

    datasets_totals: dict[str, int] = {}
    for name in dataset_names:
        meas_dir = results_dir / name / "measurements"
        if meas_dir.is_dir():
            datasets_totals[name] = len(
                [
                    p for p in meas_dir.glob("*.parquet")
                    if not p.name.startswith("_")
                ]
            )
        else:
            datasets_totals[name] = 0

    build_manifest(
        output_dir=output_dir,
        progress_dir=progress_dir,
        datasets=datasets_totals,
        execution_mode=job_meta.get("execution_mode", "local") if job_meta else "local",
        start_time=job_meta.get("start_time", "") if job_meta else "",
        slurm_job_ids=job_meta.get("chunk_job_ids") if job_meta else None,
        chunk_scripts=job_meta.get("chunk_scripts") if job_meta else None,
        input_path=job_meta.get("input_path") if job_meta else None,
    )
    console.print("[green]Manifest rebuilt")

    console.print("[cyan]Regenerating dashboard...")
    execution_mode = job_meta.get("execution_mode", "local") if job_meta else "local"
    generate_dashboard(output_dir, execution_mode=execution_mode)
    console.print(f"[green]Dashboard: {output_dir / 'dashboard.html'}")

    console.print(f"\n[bold green]Recompilation complete: {output_dir}")


def _copy_pipeline_to_output(
    pipeline_path: Path, output_dir: Path
) -> Optional[Path]:
    """Copy pipeline JSON to output directory if not already present.

    Args:
        pipeline_path: Path to the source pipeline JSON file.
        output_dir: Output directory to copy into.

    Returns:
        Path to the copy if created, None if skipped.
    """
    dest = output_dir / pipeline_path.name
    if dest.exists():
        return None
    if pipeline_path.resolve() == dest.resolve():
        return None
    shutil.copy2(pipeline_path, dest)
    return dest


def _format_duration(seconds: float) -> str:
    """Format duration as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} min"
    else:
        return f"{seconds / 3600:.1f} hr"


if __name__ == "__main__":
    phenotypic_cli()
