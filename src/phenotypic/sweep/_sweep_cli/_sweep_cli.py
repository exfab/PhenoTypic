"""Sweep CLI: execute parameter sweeps on a flat image directory.

Invoked via ``python -m phenotypic.sweep MANIFEST_JSON INPUT_DIR [OPTIONS]``.

Processes every image through all pipeline configurations defined in the
sweep manifest, with pipelines parallelized per image (joblib) or
distributed across SLURM array tasks.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

import click

from phenotypic._cli._cli_constants import MIN_SLURM_TIME_MINUTES
from phenotypic._cli._cli_directory_scanner import generate_timestamped_output_dir
from phenotypic._cli._cli_utils import parse_slurm_args
from phenotypic._core._image_parts.detection_modes import available_modes
from phenotypic.tools_.constants_ import IO

logger = logging.getLogger(__name__)


def _scan_flat_image_dir(input_dir: Path) -> List[Path]:
    """Scan a flat directory for image files.

    Only accepts directories with no subdirectories containing images.

    Args:
        input_dir: Directory to scan.

    Returns:
        Sorted list of image paths.

    Raises:
        FileNotFoundError: If *input_dir* does not exist.
        ValueError: If directory contains image-bearing subdirectories,
            or no valid images are found.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")

    valid_exts = {
        ext.lower() for ext in IO.ACCEPTED_FILE_EXTENSIONS + IO.RAW_FILE_EXTENSIONS
    }

    # Check for subdirectories with images
    for subdir in input_dir.iterdir():
        if subdir.is_dir():
            sub_images = [
                p
                for p in subdir.iterdir()
                if p.is_file() and p.suffix.lower() in valid_exts
            ]
            if sub_images:
                raise ValueError(
                        f"Sweep CLI requires a flat image directory (no subdirectories). "
                        f"Found images in subdirectory '{subdir.name}'. "
                        f"Move all images into a single flat directory."
                )

    images = sorted(
            p for p in input_dir.iterdir() if
            p.is_file() and p.suffix.lower() in valid_exts
    )

    if not images:
        raise ValueError(f"No valid images found in {input_dir}")

    return images


def _flatten_pipelines(manifest_path: Path) -> Dict[str, str]:
    """Load manifest and return ``{pipeline_name: json_str}`` without
    instantiating ``ImagePipeline`` objects.

    Reads the raw JSON and extracts each pipeline's dict as a JSON string,
    avoiding the expensive ``ImagePipeline.from_json()`` round-trip that
    would instantiate all operation objects.

    Args:
        manifest_path: Path to sweep manifest JSON.

    Returns:
        Dictionary mapping pipeline names to their JSON string representation.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(path.read_text())
    pipeline_json_strs: Dict[str, str] = {}

    for cfg_data in manifest.get("configs", {}).values():
        for pipe_name, pipe_dict in cfg_data.get("pipelines", {}).items():
            pipeline_json_strs[pipe_name] = json.dumps(pipe_dict)

    if not pipeline_json_strs:
        raise click.ClickException("Manifest contains no pipelines")

    return pipeline_json_strs


def _validate_sweep(
        pipeline_json_strs: Dict[str, str],
) -> None:
    """Validate that all pipelines can be loaded from JSON.

    Args:
        pipeline_json_strs: Pipeline name -> JSON str mapping.

    Raises:
        click.ClickException: If validation fails.
    """
    from rich.console import Console

    console = Console()

    console.print("[cyan]Validating pipeline configurations...")

    try:
        from phenotypic import ImagePipeline

        for pipe_name, pipe_json in pipeline_json_strs.items():
            ImagePipeline.from_json(pipe_json)

        console.print(
            f"[green]Validation passed ({len(pipeline_json_strs)} pipelines loaded)"
        )

    except Exception as e:
        raise click.ClickException(f"Validation failed: {type(e).__name__}: {e}")


def _display_sweep_config(
        manifest_path: Path,
        input_dir: Path,
        output_dir: Path,
        num_images: int,
        num_pipelines: int,
        image_type: str,
        nrows: Optional[int],
        ncols: Optional[int],
        n_jobs: int,
        slurm_args: Dict[str, Any],
        is_slurm: bool,
) -> None:
    """Display sweep configuration in a rich table."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()

    table = Table(
            title="Sweep Configuration",
            show_header=False,
            box=None,
            padding=(0, 2),
    )
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    backend = "SLURM Cluster" if is_slurm else "Local (joblib)"
    table.add_row("Backend", f"[bold]{backend}[/bold]")
    table.add_row("Manifest", str(manifest_path))
    table.add_row("Input Dir", str(input_dir))
    table.add_row("Output Dir", str(output_dir))
    table.add_row("", "")
    table.add_row("Images", str(num_images))
    table.add_row("Pipelines", str(num_pipelines))
    table.add_row("Total Runs", str(num_images * num_pipelines))
    table.add_row("", "")
    table.add_row("Image Type", image_type)
    if image_type == "GridImage":
        if nrows is None and ncols is None:
            grid_str = "auto (per-pipeline preset, default 8 x 12)"
        else:
            nr = "auto" if nrows is None else str(nrows)
            nc = "auto" if ncols is None else str(ncols)
            grid_str = f"{nr} x {nc}"
        table.add_row("Grid", grid_str)

    if not is_slurm:
        n_str = "All cores" if n_jobs == -1 else str(n_jobs)
        table.add_row("Parallel Jobs/Image", n_str)

    console.print()
    console.print(Panel(table, border_style="blue", expand=False))
    console.print()


def _format_duration(seconds: float) -> str:
    """Format duration as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} min"
    else:
        return f"{seconds / 3600:.1f} hr"


@click.command()
@click.argument(
        "manifest_json",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
        "input_dir",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
        "-o",
        "--output-dir",
        type=click.Path(path_type=Path),
        default=None,
        help="Output directory (auto-timestamped if omitted).",
)
@click.option(
        "--image-type",
        type=click.Choice(["Image", "GridImage"], case_sensitive=False),
        default="GridImage",
        show_default=True,
        help="Image class to instantiate.",
)
@click.option(
        "--nrows",
        type=click.IntRange(min=1),
        default=None,
        help="Grid rows (for GridImage). Overrides any per-pipeline preset; "
             "falls back to each pipeline's preset or 8 when omitted.",
)
@click.option(
        "--ncols",
        type=click.IntRange(min=1),
        default=None,
        help="Grid columns (for GridImage). Overrides any per-pipeline preset; "
             "falls back to each pipeline's preset or 12 when omitted.",
)
@click.option(
        "--bit-depth",
        type=int,
        default=None,
        help="Bit depth (8 or 16).",
)
@click.option(
        "--detect-mode",
        type=click.Choice(list(available_modes())),
        default="gray",
        show_default=True,
        help="Detection channel.",
)
@click.option(
        "--n-jobs",
        type=int,
        default=-1,
        show_default=True,
        help="Parallel pipeline jobs per image (-1 = all cores).",
)
@click.option(
        "--slurm",
        "slurm_args",
        multiple=True,
        help="SLURM parameters as KEY=VALUE pairs.",
)
@click.option("--force-local", is_flag=True, help="Force local execution.")
@click.option("--wait", is_flag=True, help="Monitor SLURM jobs.")
@click.option("--dry-run", is_flag=True, help="Preview without executing.")
@click.option("--skip-validation", is_flag=True, help="Skip pipeline validation.")
@click.option("-v", "--verbose", is_flag=True, help="Log per-operation pipeline steps to stderr.")
@click.option("--save-intermediates", is_flag=True, help="Save intermediate image state after each pipeline operation as HDF5.")
def sweep_cli(
        manifest_json: Path,
        input_dir: Path,
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
        dry_run: bool,
        skip_validation: bool,
        verbose: bool,
        save_intermediates: bool,
):
    """Execute a parameter sweep on a flat image directory.

    MANIFEST_JSON: Path to sweep manifest JSON.

    INPUT_DIR: Flat directory of images (no subdirectories with images).
    """
    try:
        # Configure per-operation pipeline logging
        if verbose:
            from ._sweep_process_image import _configure_pipeline_debug_logging
            _configure_pipeline_debug_logging()

        # Parse SLURM args
        slurm_args_dict: Dict[str, Any] = {}
        if slurm_args:
            try:
                slurm_args_dict = parse_slurm_args(slurm_args)
            except click.BadParameter as e:
                click.echo(str(e), err=True)
                sys.exit(1)

        # Validate SLURM time
        for time_key in ("time", "slurm_time"):
            if time_key in slurm_args_dict:
                time_val = slurm_args_dict[time_key]
                if not isinstance(time_val, int):
                    click.echo(
                            f"Error: '{time_key}' must be integer minutes",
                            err=True,
                    )
                    sys.exit(1)
                if time_val < MIN_SLURM_TIME_MINUTES:
                    click.echo(
                            f"Error: '{time_key}' must be >= {MIN_SLURM_TIME_MINUTES}",
                            err=True,
                    )
                    sys.exit(1)

        is_slurm = bool(slurm_args_dict) and not force_local

        # Scan input directory (flat only)
        click.echo(f"Scanning {input_dir}...")
        try:
            image_paths = _scan_flat_image_dir(input_dir)
        except (FileNotFoundError, ValueError) as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

        click.echo(f"Found {len(image_paths)} images")

        # Load manifest and flatten pipelines
        click.echo(f"Loading manifest {manifest_json}...")
        try:
            pipeline_json_strs = _flatten_pipelines(manifest_json)
        except Exception as e:
            click.echo(f"Error loading manifest: {e}", err=True)
            sys.exit(1)

        pipeline_names = list(pipeline_json_strs.keys())
        click.echo(f"Loaded {len(pipeline_names)} pipeline configurations")

        # Build read kwargs.
        read_kwargs: Dict[str, Any] = {}
        if bit_depth is not None:
            read_kwargs["bit_depth"] = bit_depth
        if detect_mode != "gray":
            read_kwargs["detect_mode"] = detect_mode

        # Validation
        if not skip_validation:
            _validate_sweep(pipeline_json_strs)

        # Generate output directory
        if output_dir is None:
            output_dir = generate_timestamped_output_dir()
            click.echo(f"Auto-generated output directory: {output_dir}")

        # Clear previous run if output directory has existing results
        from ._sweep_output import clear_previous_run

        if clear_previous_run(output_dir):
            click.echo(f"Cleared previous results from: {output_dir}")

        # Display config
        _display_sweep_config(
                manifest_path=manifest_json,
                input_dir=input_dir,
                output_dir=output_dir,
                num_images=len(image_paths),
                num_pipelines=len(pipeline_names),
                image_type=image_type,
                nrows=nrows,
                ncols=ncols,
                n_jobs=n_jobs,
                slurm_args=slurm_args_dict,
                is_slurm=is_slurm,
        )

        # Dry run
        if dry_run:
            click.echo("Dry-run mode: no processing will be performed.")
            click.echo(
                    f"\nWould process {len(image_paths)} images x {len(pipeline_names)} pipelines"
            )
            click.echo(
                f"= {len(image_paths) * len(pipeline_names)} total pipeline runs")
            click.echo("\nPipeline names:")
            for name in pipeline_names:
                click.echo(f"  - {name}")
            sys.exit(0)

        # Create output manager and directory structure
        from ._sweep_output import SweepOutputManager

        output_manager = SweepOutputManager(base_dir=output_dir)
        output_manager.create_structure()

        # Copy manifest to output directory for reproducibility
        dest_manifest = output_dir / "sweep_manifest.json"
        shutil.copy2(manifest_json, dest_manifest)
        click.echo(f"Manifest copied to {dest_manifest}")

        # Write dashboard metadata and initial empty dashboard
        from datetime import datetime as dt

        from ._sweep_progress_dashboard import (
            generate_sweep_progress_dashboard,
            write_sweep_progress_metadata,
        )

        sweep_start_time = dt.now()
        total_tasks = len(image_paths) * len(pipeline_names)
        event_log = output_dir / "processing_events.log"
        dashboard_path = output_dir / "sweep_progress.html"

        write_sweep_progress_metadata(
            output_dir=output_dir,
            total_tasks=total_tasks,
            num_images=len(image_paths),
            num_pipelines=len(pipeline_names),
            start_time=sweep_start_time,
        )
        generate_sweep_progress_dashboard(
            event_log=event_log,
            output_path=dashboard_path,
            total_tasks=total_tasks,
            start_time=sweep_start_time,
        )
        click.echo(f"Progress dashboard: {dashboard_path}")

        # Create execution strategy and run
        from ._sweep_execution import (
            LocalSweepStrategy,
            SLURMSweepStrategy,
            SweepExecutionStrategy,
        )

        strategy: SweepExecutionStrategy
        image_type_lit: Literal[
            "Image", "GridImage"] = image_type  # type: ignore[assignment]
        if is_slurm:
            strategy = SLURMSweepStrategy(
                    pipeline_json_strs=pipeline_json_strs,
                    image_type=image_type_lit,
                    read_kwargs=read_kwargs,
                    output_manager=output_manager,
                    manifest_path=dest_manifest,
                    slurm_args=slurm_args_dict,
                    wait=wait,
                    verbose=verbose,
                    save_intermediates=save_intermediates,
                    cli_nrows=nrows,
                    cli_ncols=ncols,
            )
        else:
            strategy = LocalSweepStrategy(
                    pipeline_json_strs=pipeline_json_strs,
                    image_type=image_type_lit,
                    read_kwargs=read_kwargs,
                    output_manager=output_manager,
                    n_jobs=n_jobs,
                    event_log=event_log,
                    save_intermediates=save_intermediates,
                    cli_nrows=nrows,
                    cli_ncols=ncols,
            )

        results = strategy.execute(image_paths, output_dir)

        # Generate final dashboard with auto-refresh disabled
        generate_sweep_progress_dashboard(
            event_log=event_log,
            output_path=dashboard_path,
            total_tasks=total_tasks,
            start_time=sweep_start_time,
            is_complete=True,
        )

        # Summary
        duration = (results["end_time"] - results["start_time"]).total_seconds()

        click.echo("\n" + "=" * 60)
        click.echo("SWEEP COMPLETE")
        click.echo("=" * 60)
        click.echo(f"Images: {results['total_images']}")
        click.echo(f"Pipelines: {len(pipeline_names)}")
        click.echo(f"Duration: {_format_duration(duration)}")

        failures = results.get("failures", [])
        if failures:
            click.echo(f"Pipeline failures: {len(failures)}")
            for f in failures[:5]:
                click.echo(f"  - {f['image']} / {f['pipeline']}")
            if len(failures) > 5:
                click.echo(f"  ... and {len(failures) - 5} more")
            click.echo(
                f"Detailed failure logs: {output_manager.failures_dir}"
            )

        click.echo(f"\nResults saved to: {output_dir}")
        click.echo(f"Progress dashboard: {dashboard_path}")
        sys.exit(0 if not failures else 1)

    except KeyboardInterrupt:
        click.echo("\n\nInterrupted by user", err=True)
        sys.exit(130)
    except click.ClickException:
        raise
    except Exception as e:
        click.echo(f"\nUnexpected error: {e}", err=True)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    sweep_cli()
