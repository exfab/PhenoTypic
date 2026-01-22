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
    validate_resume_compatibility,
)
from phenotypic._cli._cli_types import ExecutionConfig
from phenotypic._cli._cli_utils import normalize_extension
from phenotypic._cli._cli_validation import full_validation
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
        prev_images = ds_state.completed | ds_state.failed

        # Get current image names from scan
        current_dataset = current_datasets_map[ds_name]
        curr_images = {img.name for img in current_dataset.images}

        # Check if sets match exactly
        if prev_images != curr_images:
            missing = prev_images - curr_images
            added = curr_images - prev_images

            error_parts = [f"Image set mismatch in dataset '{ds_name}':"]
            if missing:
                error_parts.append(f"  - Missing {len(missing)} images (e.g., {list(missing)[:3]})")
            if added:
                error_parts.append(f"  - Added {len(added)} new images (e.g., {list(added)[:3]})")

            return False, "\n".join(error_parts)

    return True, None


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
        help="Save enhanced grayscale to OUTPUT_DIR/enh_gray/",
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
        "--save-objmap-rgb",
        is_flag=True,
        help="Save object map RGB to OUTPUT_DIR/objmap_rgb/",
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
        help="File extension for enhanced grayscale saves",
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
        "--objmap-rgb-ext",
        default="png",
        show_default=True,
        help="File extension for object map RGB saves",
)
@click.option(
        "--include-dataset-column",
        is_flag=True,
        help="Add 'Dataset' column to master_measurements.csv",
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
        n_jobs: int,
        slurm_args: Sequence[str],
        force_local: bool,
        wait: bool,
        save_rgb: bool,
        save_gray: bool,
        save_enh_gray: bool,
        save_objmask: bool,
        save_objmap: bool,
        save_objmap_rgb: bool,
        rgb_ext: str,
        gray_ext: str,
        enh_gray_ext: str,
        objmask_ext: str,
        objmap_ext: str,
        objmap_rgb_ext: str,
        include_dataset_column: bool,
        dry_run: bool,
        sample: Optional[int],
        resume: bool,
        retry_failures: bool,
        skip_validation: bool,
):
    """
    Execute a PhenoTypic pipeline on images.

    PIPELINE_JSON: Path to pipeline configuration file

    INPUT_PATH: Image file or directory to process
    """
    try:
        # Validate extension arguments
        try:
            rgb_ext = normalize_extension(rgb_ext, ".tiff")
            gray_ext = normalize_extension(gray_ext, ".tiff")
            enh_gray_ext = normalize_extension(enh_gray_ext, ".tiff")
            objmask_ext = normalize_extension(objmask_ext, ".png")
            objmap_ext = normalize_extension(objmap_ext, ".png")
            objmap_rgb_ext = normalize_extension(objmap_rgb_ext, ".png")
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

        # Create ExecutionConfig
        config = ExecutionConfig(
                pipeline_json=pipeline_json,
                input_path=input_path,
                output_dir=output_dir,
                image_type=image_type,
                nrows=nrows,
                ncols=ncols,
                bit_depth=bit_depth,
                n_jobs=n_jobs,
                slurm_args=slurm_args_dict,
                force_local=force_local,
                wait=wait,
                save_rgb=save_rgb,
                save_gray=save_gray,
                save_enh_gray=save_enh_gray,
                save_objmask=save_objmask,
                save_objmap=save_objmap,
                save_objmap_rgb=save_objmap_rgb,
                rgb_ext=rgb_ext,
                gray_ext=gray_ext,
                enh_gray_ext=enh_gray_ext,
                objmask_ext=objmask_ext,
                objmap_ext=objmap_ext,
                objmap_rgb_ext=objmap_rgb_ext,
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

        # Generate or validate output directory
        if output_dir is None:
            output_dir = generate_timestamped_output_dir()
            click.echo(f"Auto-generated output directory: {output_dir}")

        # Create output directory (only if not resuming or doesn't exist)
        output_dir.mkdir(parents=True, exist_ok=True)

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
            click.echo("Validating pipeline configuration...")
            is_valid, errors = full_validation(config, datasets)
            if not is_valid:
                click.echo("Validation failed:", err=True)
                for error in errors:
                    click.echo(f"  - {error}", err=True)
                sys.exit(1)
            click.echo("✓ Validation passed")

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
                    datasets, config.sample, output_dir
            )
            total_images = sum(len(d.images) for d in datasets)
            click.echo(f"Processing {total_images} sample images\n")

        # Handle resume mode - get remaining images
        if config.resume:
            # State was already validated earlier, just load it
            state = load_processing_state(output_dir)

            # Validate compatibility
            is_compatible, error = validate_resume_compatibility(state, config)
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
            images_valid, image_error = _validate_resume_input_images(state, datasets)
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
                    state, datasets, config.retry_failures
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

        # Create initial state (or update if resuming)
        state = create_initial_state(config, datasets, output_dir)
        save_processing_state(state, output_dir)

        # Create output manager
        output_manager = OutputManager(
                base_dir=output_dir,
                save_layers={
                    "rgb"       : config.save_rgb,
                    "gray"      : config.save_gray,
                    "enh_gray"  : config.save_enh_gray,
                    "objmask"   : config.save_objmask,
                    "objmap"    : config.save_objmap,
                    "objmap_rgb": config.save_objmap_rgb,
                },
                extensions={
                    "rgb"       : config.rgb_ext,
                    "gray"      : config.gray_ext,
                    "enh_gray"  : config.enh_gray_ext,
                    "objmask"   : config.objmask_ext,
                    "objmap"    : config.objmap_ext,
                    "objmap_rgb": config.objmap_rgb_ext,
                },
                include_dataset_column=config.include_dataset_column,
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
