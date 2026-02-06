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

from phenotypic._cli._cli_constants import MIN_SLURM_TIME_MINUTES, MAX_SLURM_TIME_MINUTES
from phenotypic._cli._cli_directory_scanner import generate_timestamped_output_dir
from phenotypic._cli._cli_utils import normalize_extension, parse_slurm_args
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

    valid_exts = {ext.lower() for ext in IO.ACCEPTED_FILE_EXTENSIONS + IO.RAW_FILE_EXTENSIONS}

    # Check for subdirectories with images
    for subdir in input_dir.iterdir():
        if subdir.is_dir():
            sub_images = [
                p for p in subdir.iterdir()
                if p.is_file() and p.suffix.lower() in valid_exts
            ]
            if sub_images:
                raise ValueError(
                    f"Sweep CLI requires a flat image directory (no subdirectories). "
                    f"Found images in subdirectory '{subdir.name}'. "
                    f"Move all images into a single flat directory."
                )

    images = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_exts
    )

    if not images:
        raise ValueError(f"No valid images found in {input_dir}")

    return images


def _flatten_pipelines(manifest_path: Path) -> Dict[str, str]:
    """Load manifest and return ``{pipeline_name: json_str}`` for all pipelines.

    Args:
        manifest_path: Path to sweep manifest JSON.

    Returns:
        Dictionary mapping pipeline names to their JSON string representation.
    """
    from phenotypic.sweep import load_sweep_manifest

    configs = load_sweep_manifest(manifest_path)

    pipeline_json_strs: Dict[str, str] = {}
    for _cfg_name, pipes in configs.items():
        for pipe_name, pipe in pipes.items():
            pipeline_json_strs[pipe_name] = pipe.to_json_str()

    if not pipeline_json_strs:
        raise click.ClickException("Manifest contains no pipelines")

    return pipeline_json_strs


def _validate_sweep(
    manifest_path: Path,
    pipeline_json_strs: Dict[str, str],
    image_paths: List[Path],
    image_type: str,
    read_kwargs: Dict[str, Any],
) -> None:
    """Validate the first pipeline on the first image.

    Args:
        manifest_path: Path to manifest (for error messages).
        pipeline_json_strs: Pipeline name -> JSON str mapping.
        image_paths: List of images (uses first).
        image_type: Image class name.
        read_kwargs: Image read kwargs.

    Raises:
        click.ClickException: If validation fails.
    """
    from rich.console import Console

    console = Console()

    first_pipe_name = next(iter(pipeline_json_strs))
    first_pipe_json = pipeline_json_strs[first_pipe_name]
    first_image = image_paths[0]

    console.print(
        f"[cyan]Validating pipeline '{first_pipe_name}' on {first_image.name}..."
    )

    try:
        import json as _json

        from phenotypic import Image, GridImage, ImagePipeline

        pipeline = ImagePipeline.from_json(_json.loads(first_pipe_json))
        image_cls = GridImage if image_type == "GridImage" else Image
        rk = dict(read_kwargs)
        detect_mode = rk.pop("detect_mode", "gray")
        image = image_cls.imread(first_image, **rk)
        if detect_mode != "gray":
            image.set_detect_mode(detect_mode)

        measurements = pipeline.apply_and_measure(image, inplace=True)

        if measurements is None or len(measurements) == 0:
            raise click.ClickException("Pipeline produced no measurements on test image")

        console.print("[green]Validation passed")

    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Validation failed: {type(e).__name__}: {e}")


def _display_sweep_config(
    manifest_path: Path,
    input_dir: Path,
    output_dir: Path,
    num_images: int,
    num_pipelines: int,
    image_type: str,
    nrows: int,
    ncols: int,
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
        table.add_row("Grid", f"{nrows} x {ncols}")

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
    "-o", "--output-dir",
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
    default=8,
    show_default=True,
    help="Grid rows (for GridImage).",
)
@click.option(
    "--ncols",
    type=click.IntRange(min=1),
    default=12,
    show_default=True,
    help="Grid columns (for GridImage).",
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
    "--slurm-args",
    multiple=True,
    help="SLURM parameters as KEY=VALUE pairs.",
)
@click.option("--force-local", is_flag=True, help="Force local execution.")
@click.option("--wait", is_flag=True, help="Monitor SLURM jobs.")
@click.option("--save-rgb", is_flag=True, help="Save RGB layer.")
@click.option("--save-gray", is_flag=True, help="Save grayscale layer.")
@click.option("--save-enh-gray", is_flag=True, help="Save detection matrix.")
@click.option("--save-objmask", is_flag=True, help="Save object mask.")
@click.option("--save-objmap", is_flag=True, help="Save object map.")
@click.option("--save-objmap-overlay", is_flag=True, help="Save object map overlay.")
@click.option("--save-enh-gray-overlay", is_flag=True, help="Save detection matrix overlay.")
@click.option("--save-objmask-overlay", is_flag=True, help="Save object mask overlay.")
@click.option(
    "--overlay-mode",
    type=click.Choice(["image", "figure"]),
    default="image",
    show_default=True,
    help="Overlay saving mode.",
)
@click.option(
    "--overlay-alpha",
    type=float,
    default=0.3,
    show_default=True,
    help="Alpha transparency (0.0-1.0).",
)
@click.option("--dry-run", is_flag=True, help="Preview without executing.")
@click.option("--skip-validation", is_flag=True, help="Skip pipeline validation.")
def main(
    manifest_json: Path,
    input_dir: Path,
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
    overlay_mode: str,
    overlay_alpha: float,
    dry_run: bool,
    skip_validation: bool,
):
    """Execute a parameter sweep on a flat image directory.

    MANIFEST_JSON: Path to sweep manifest JSON.

    INPUT_DIR: Flat directory of images (no subdirectories with images).
    """
    try:
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

        # Build read kwargs
        read_kwargs: Dict[str, Any] = {}
        if image_type == "GridImage":
            read_kwargs["nrows"] = nrows
            read_kwargs["ncols"] = ncols
        if bit_depth is not None:
            read_kwargs["bit_depth"] = bit_depth
        if detect_mode != "gray":
            read_kwargs["detect_mode"] = detect_mode

        # Validation
        if not skip_validation:
            _validate_sweep(
                manifest_json,
                pipeline_json_strs,
                image_paths,
                image_type,
                read_kwargs,
            )

        # Generate output directory
        if output_dir is None:
            output_dir = generate_timestamped_output_dir()
            click.echo(f"Auto-generated output directory: {output_dir}")

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
            click.echo(f"\nWould process {len(image_paths)} images x {len(pipeline_names)} pipelines")
            click.echo(f"= {len(image_paths) * len(pipeline_names)} total pipeline runs")
            click.echo(f"\nPipeline names:")
            for name in pipeline_names:
                click.echo(f"  - {name}")
            sys.exit(0)

        # Build save layers config
        save_layers = {
            "rgb": save_rgb,
            "gray": save_gray,
            "detect_mat": save_detect_mat,
            "objmask": save_objmask,
            "objmap": save_objmap,
            "objmap_overlay": save_objmap_overlay,
            "detect_mat_overlay": save_detect_mat_overlay,
            "objmask_overlay": save_objmask_overlay,
        }

        extensions = {
            "rgb": ".tiff",
            "gray": ".tiff",
            "detect_mat": ".tiff",
            "objmask": ".png",
            "objmap": ".png",
            "objmap_overlay": ".png",
        }

        # Create output manager and directory structure
        from ._sweep_output import SweepOutputManager

        output_manager = SweepOutputManager(
            base_dir=output_dir,
            save_layers=save_layers,
            extensions=extensions,
            overlay_mode=overlay_mode,
            overlay_alpha=overlay_alpha,
        )
        output_manager.create_structure(pipeline_names)

        # Copy manifest to output directory for reproducibility
        dest_manifest = output_dir / "sweep_manifest.json"
        shutil.copy2(manifest_json, dest_manifest)
        click.echo(f"Manifest copied to {dest_manifest}")

        # Create execution strategy and run
        from ._sweep_execution import LocalSweepStrategy, SLURMSweepStrategy, SweepExecutionStrategy

        strategy: SweepExecutionStrategy
        image_type_lit: Literal["Image", "GridImage"] = image_type  # type: ignore[assignment]
        if is_slurm:
            strategy = SLURMSweepStrategy(
                pipeline_json_strs=pipeline_json_strs,
                image_type=image_type_lit,
                read_kwargs=read_kwargs,
                output_manager=output_manager,
                manifest_path=manifest_json,
                slurm_args=slurm_args_dict,
                wait=wait,
                save_layers=save_layers,
                overlay_mode=overlay_mode,
                overlay_alpha=overlay_alpha,
            )
        else:
            strategy = LocalSweepStrategy(
                pipeline_json_strs=pipeline_json_strs,
                image_type=image_type_lit,
                read_kwargs=read_kwargs,
                output_manager=output_manager,
                n_jobs=n_jobs,
            )

        results = strategy.execute(image_paths, output_dir)

        # Aggregate master CSV
        if results.get("completed", 0) > 0:
            click.echo("\nAggregating measurements...")
            master_path = output_manager.aggregate_master_csv(pipeline_names)
            if master_path:
                click.echo(f"Master measurements: {master_path}")
            else:
                click.echo(
                    "Warning: Could not aggregate master CSV",
                    err=True,
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

        click.echo(f"\nResults saved to: {output_dir}")
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
    main()
