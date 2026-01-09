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

    # SLURM execution (autonomous)
    uv run python -m phenotypic pipeline.json ./images \
        --slurm-kwds slurm_partition=compute slurm_account=proj mem_gb=16

    # SLURM with progress monitoring
    uv run python -m phenotypic pipeline.json ./images \
        --slurm-kwds slurm_partition=compute slurm_account=proj \
        --wait

    # Save intermediate layers
    uv run python -m phenotypic pipeline.json ./images \
        --save-rgb --save-gray --save-objmask

    # GridImage with custom dimensions
    uv run python -m phenotypic pipeline.json ./plates \
        --image-type GridImage --nrows 16 --ncols 24

Migration Notes (v1.x → v2.0):
    - OUTPUT_DIR is now optional (generates timestamped dir if not provided)
    - Use -o/--output-dir instead of positional OUTPUT_DIR argument
    - --slurm-params KEY=VALUE replaced with --slurm-kwds KEY=VALUE (space-separated)
    - Recursive directory processing now preserves subdirectory hierarchy
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import click

from phenotypic import Image, GridImage, ImagePipeline
from phenotypic._core._cli_directory_scanner import (
    generate_timestamped_output_dir,
    organize_by_dataset,
    scan_directory_structure,
)
from phenotypic._core._cli_execution_strategies import create_execution_strategy
from phenotypic._core._cli_interactive import (
    execute_dry_run,
    get_sample_datasets,
)
from phenotypic._core._cli_output_manager import OutputManager
from phenotypic._core._cli_report_generator import HTMLReportGenerator
from phenotypic._core._cli_state_management import (
    create_initial_state,
    get_remaining_images_for_datasets,
    load_processing_state,
    save_processing_state,
    validate_resume_compatibility,
)
from phenotypic._core._cli_types import ExecutionConfig
from phenotypic._core._cli_validation import full_validation


def _parse_slurm_kwds(slurm_kwds: Sequence[str]) -> dict:
    """
    Parse space-separated KEY=VALUE pairs into dictionary.

    Args:
        slurm_kwds: Sequence of "KEY=VALUE" strings

    Returns:
        Dictionary of parsed parameters

    Raises:
        click.BadParameter: If parsing fails
    """
    import ast

    parsed = {}
    for kwd in slurm_kwds:
        if "=" not in kwd:
            raise click.BadParameter(
                "--slurm-kwds must be KEY=VALUE pairs",
                param_hint="--slurm-kwds",
            )

        key, value = kwd.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            raise click.BadParameter(
                "SLURM parameter keys cannot be empty",
                param_hint="--slurm-kwds",
            )

        # Try to parse value as Python literal
        try:
            parsed_value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # Keep as string if not a valid literal
            parsed_value = value

        parsed[key] = parsed_value

    return parsed


def _normalize_extension(ext: str, default: str = ".tiff") -> str:
    """Normalize extension to include leading dot."""
    if not ext:
        ext = default
    ext = ext.lower()
    if not ext.startswith("."):
        ext = f".{ext}"

    allowed = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
    if ext not in allowed:
        raise click.BadParameter(
            f"Unsupported extension '{ext}'. "
            f"Allowed: {', '.join(sorted(allowed))}"
        )

    return ext


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
    type=int,
    default=8,
    show_default=True,
    help="Number of rows for GridImage",
)
@click.option(
    "--ncols",
    type=int,
    default=12,
    show_default=True,
    help="Number of columns for GridImage",
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
    "--slurm-kwds",
    multiple=True,
    help="SLURM parameters as space-separated KEY=VALUE pairs "
    "(e.g., --slurm-kwds slurm_partition=compute mem_gb=16)",
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
    slurm_kwds: Sequence[str],
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
            rgb_ext = _normalize_extension(rgb_ext, ".tiff")
            gray_ext = _normalize_extension(gray_ext, ".tiff")
            enh_gray_ext = _normalize_extension(enh_gray_ext, ".tiff")
            objmask_ext = _normalize_extension(objmask_ext, ".png")
            objmap_ext = _normalize_extension(objmap_ext, ".png")
            objmap_rgb_ext = _normalize_extension(objmap_rgb_ext, ".png")
        except click.BadParameter as e:
            click.echo(str(e), err=True)
            sys.exit(1)

        # Parse SLURM kwargs
        slurm_kwds_dict = {}
        if slurm_kwds:
            try:
                slurm_kwds_dict = _parse_slurm_kwds(slurm_kwds)
            except click.BadParameter as e:
                click.echo(str(e), err=True)
                sys.exit(1)

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
            slurm_kwds=slurm_kwds_dict,
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

        # Generate output directory if not provided
        if output_dir is None:
            output_dir = generate_timestamped_output_dir()
            click.echo(f"Auto-generated output directory: {output_dir}")

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

        # Handle resume mode
        if config.resume:
            state = load_processing_state(output_dir)
            if state is None:
                click.echo(
                    "Error: No processing state found for resume", err=True
                )
                click.echo(
                    "Cannot resume - no previous processing found in "
                    f"{output_dir}",
                    err=True,
                )
                sys.exit(1)

            is_compatible, error = validate_resume_compatibility(
                state, config
            )
            if not is_compatible:
                click.echo(
                    f"Error: Cannot resume - {error}", err=True
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
                "rgb": config.save_rgb,
                "gray": config.save_gray,
                "enh_gray": config.save_enh_gray,
                "objmask": config.save_objmask,
                "objmap": config.save_objmap,
                "objmap_rgb": config.save_objmap_rgb,
            },
            extensions={
                "rgb": config.rgb_ext,
                "gray": config.gray_ext,
                "enh_gray": config.enh_gray_ext,
                "objmask": config.objmask_ext,
                "objmap": config.objmap_ext,
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
            click.echo(f"✓ Master measurements: {master_path}")

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
            f"Success rate: {results.success_rate*100:.1f}%"
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
        return f"{seconds/60:.1f} min"
    else:
        return f"{seconds/3600:.1f} hr"


if __name__ == "__main__":
    main()
