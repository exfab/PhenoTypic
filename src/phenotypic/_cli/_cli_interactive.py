"""
Interactive features for the PhenoTypic CLI.

This module implements dry-run mode, sample processing, and resume logic.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, List, Optional

import click

from phenotypic.schema import EXPERIMENT
from ._cli_types import Dataset, ExecutionConfig


def _display_datasets_detail(datasets: List[Dataset]) -> int:
    """Display detailed dataset and image information."""
    click.echo("Datasets Discovered:")
    total_images = 0
    for dataset in datasets:
        click.echo(f"\n  Dataset: {dataset.name}")
        click.echo(f"    Input directory:  {dataset.input_dir}")
        click.echo(f"    Output directory: {dataset.output_dir}")
        click.echo(f"    Images ({len(dataset.images)}):")

        # Show first 10 image filenames
        for i, image_path in enumerate(dataset.images[:10]):
            click.echo(f"      - {image_path.name}")

        if len(dataset.images) > 10:
            click.echo(f"      ... and {len(dataset.images) - 10} more")

        total_images += len(dataset.images)

    click.echo(f"\n  Total images across all datasets: {total_images}")
    return total_images


def _display_execution_backend(config: ExecutionConfig) -> str:
    """Display execution backend determination and configuration."""
    click.echo("\nExecution Backend Determination:")
    is_slurm = config.is_slurm_mode()
    click.echo(f"  - SLURM args provided: {'Yes' if config.slurm_args else 'No'}")
    click.echo(f"  - Force local flag: {'Yes' if config.force_local else 'No'}")
    click.echo(f"  → Selected backend: {'SLURM (autonomous)' if is_slurm else 'Local Parallel (joblib)'}")
    return "slurm" if is_slurm else "local"


def _submitted_slurm_profiles(config: ExecutionConfig) -> list[tuple[str, dict]]:
    """Every SBATCH profile the run submits, as the run preflight computes them.

    One definition (``_cli_preflight._slurm_profiles``): a staged GPU run adds
    the GPU stage's profile, and a GPU pipeline the non-staged strategy runs
    submits ``--slurm`` with a GPU request added (Phase E review E9). Falls
    back to the ``--slurm`` profile alone when the pipeline cannot be read;
    the validation above reports that.
    """
    from ._cli_preflight import PreflightContext, _slurm_profiles, run_mode_of

    try:
        from phenotypic import ImagePipeline

        pipeline = ImagePipeline.from_json(config.pipeline_json)
        return _slurm_profiles(
            PreflightContext(config=config, pipeline=pipeline, datasets=(), mode=run_mode_of(config))
        )
    except Exception:  # noqa: BLE001 -- a preview must not fail the dry run
        return [("CPU profile (--slurm)", dict(config.slurm_args))]


def _display_slurm_config(slurm_args: dict, label: str = "") -> None:
    """Display SLURM configuration parameters as table."""
    if not slurm_args:
        return

    click.echo(f"\n  SLURM Configuration Parameters{f' ({label})' if label else ''}:")
    for key, value in slurm_args.items():
        click.echo(f"    {key:.<30} {value}")

    # The exact text a submission writes, from the one formatter scripts use,
    # not a re-implementation of it (spec 2026-09-24-cli-preflight F26).
    from pathlib import Path

    from phenotypic.sdk_.slurm import format_sbatch_directives

    click.echo("\n  Converted SBATCH Directives:")
    directives = format_sbatch_directives(
        job_name="phenotypic",
        slurm_args=slurm_args,
        output_log=Path("<log>"),
        error_log=Path("<log>"),
    )
    for line in directives.splitlines():
        if line.strip():
            click.echo(f"    {line}")


def _display_local_config(config: ExecutionConfig) -> None:
    """Display local parallel execution configuration."""
    click.echo("\n  Local Parallel Configuration:")
    if config.n_jobs == -1:
        click.echo("    n_jobs:.............. -1 (use all available CPU cores)")
    else:
        click.echo(f"    n_jobs:.............. {config.n_jobs}")


def _load_preview_pipeline(config: ExecutionConfig) -> tuple[Any, Optional[str]]:
    """``(pipeline, None)``, or ``(None, reason)`` when the pipeline cannot be read.

    ``config.pipeline_json`` is a path. The main CLI path loaded it before the
    dry-run exit, but ``--skip-validation`` skips that, so a preview must not
    fail on an unreadable file; it says so instead.
    """
    from phenotypic import ImagePipeline

    try:
        return ImagePipeline.from_json(config.pipeline_json), None
    except Exception as exc:  # noqa: BLE001 -- a preview must not fail the dry run
        lines = str(exc).strip().splitlines()
        reason = f"{type(exc).__name__}: {lines[0]}" if lines else type(exc).__name__
        return None, reason


_FROM_PIPELINE = "the pipeline"
_FROM_DEFAULT = "the built-in default"


def _grid_axis_source(cli_value: Optional[int], pipeline_value: Optional[int], flag: str) -> str:
    """Where one grid axis comes from, by ``resolve_grid_shape``'s precedence.

    The flag, then the pipeline's preset, then the built-in default.
    """
    if cli_value is not None:
        return flag
    if pipeline_value is not None:
        return _FROM_PIPELINE
    return _FROM_DEFAULT


def _display_grid_shape(config: ExecutionConfig, pipeline: Any) -> None:
    """Print the grid shape the run will load each ``GridImage`` with, and its source.

    ``resolve_grid_shape`` returns no source, so the source is worked out here
    with the same precedence, one axis at a time (``--nrows`` alone leaves the
    columns to the pipeline or the default).
    """
    if config.measure_only:
        click.echo(
            "  Grid shape: as stored in each image store "
            "(--mode measure re-measures existing stores)"
        )
        return
    if pipeline is None:
        click.echo("  Grid shape: unknown; the pipeline could not be read (see Pipeline Steps)")
        return

    from ._cli_utils import resolve_grid_shape

    nrows, ncols = resolve_grid_shape(
        cli_nrows=config.nrows,
        cli_ncols=config.ncols,
        pipeline_nrows=pipeline.nrows,
        pipeline_ncols=pipeline.ncols,
    )
    rows_from = _grid_axis_source(config.nrows, pipeline.nrows, "--nrows")
    cols_from = _grid_axis_source(config.ncols, pipeline.ncols, "--ncols")
    if rows_from == cols_from:
        origin = f"from {rows_from}"
    else:
        origin = f"rows from {rows_from}, columns from {cols_from}"
    click.echo(f"  Grid shape: {nrows} × {ncols} (rows × columns), {origin}")
    if _FROM_DEFAULT in (rows_from, cols_from):
        click.echo(
            "    If your plates differ, pass --nrows/--ncols or set "
            "ImagePipeline(nrows=, ncols=)."
        )


def _display_pipeline_steps(config: ExecutionConfig, pipeline: Any, load_error: Optional[str]) -> None:
    """Print ``ImagePipeline.summary()``: the steps the run executes, in order."""
    click.echo("\nPipeline Steps:")
    if pipeline is None:
        click.echo(f"  The pipeline could not be read: {load_error}")
        return
    table = pipeline.summary()
    if config.measure_only:
        click.echo("  --mode measure re-measures existing stores, so only the measurers run.")
        table = table[table["slot"] == "meas"]
    if table.empty:
        click.echo("  (no steps)")
        return
    for line in table.to_string(index=False).splitlines():
        click.echo(f"  {line.rstrip()}")


def _display_output_layout(config: ExecutionConfig, datasets: List[Dataset], output_dir: Path) -> None:
    """Print where the run's outputs land, from the ``phenotypic.sdk_`` path helpers."""
    from phenotypic.sdk_ import (
        dataset_overlays_dir,
        master_measurements_parquet_path,
        measurements_csv_path,
        measurements_parquet_path,
        phenotypic_cache_dir,
        pipeline_json_path,
        source_image_stem,
        zarr_store_path,
    )

    dataset, stem = "<dataset>", "<stem>"
    entries: List[tuple[Path, bool, str]] = [
        (
            measurements_csv_path(output_dir),
            False,
            "the table to analyse: measurements, joined with --metadata when "
            f"given (also {measurements_parquet_path(output_dir).name})",
        ),
        (
            master_measurements_parquet_path(output_dir),
            False,
            "every image's measurements, without metadata",
        ),
        (pipeline_json_path(output_dir), False, "the pipeline this run used"),
    ]
    if config.save_overlays and not config.measure_only:
        entries.append(
            (dataset_overlays_dir(output_dir, dataset), True, "one detection overlay PNG per image")
        )
    entries += [
        (
            zarr_store_path(output_dir, dataset, stem),
            True,
            "one OME-Zarr store per image: its layers and measurements",
        ),
        (phenotypic_cache_dir(output_dir), True, "machine state: progress, logs, SLURM scripts"),
    ]

    def _shown(path: Path, is_dir: bool) -> str:
        return f"{path.relative_to(output_dir)}{os.sep if is_dir else ''}"

    width = max(len(_shown(path, is_dir)) for path, is_dir, _ in entries)
    click.echo("\nOutput Layout:")
    click.echo(f"  {output_dir}{os.sep}")
    for path, is_dir, what in entries:
        click.echo(f"    {_shown(path, is_dir):<{width}}  {what}")

    example = next(((ds, ds.images[0]) for ds in datasets if ds.images), None)
    if example is not None:
        ds, image = example
        store = zarr_store_path(output_dir, ds.name, source_image_stem(image))
        click.echo(f"  For example, {image.name}: {_shown(store, True)}")

    if config.include_dataset_column:
        click.echo(
            f"  Tables include the {str(EXPERIMENT.DATASET)!r} column"
            " for multi-dataset analysis"
        )


def _display_size_estimate(total_images: int, *, overlays: bool) -> None:
    """Print a rough output size from fixed per-image sizes, not measured from the inputs."""
    click.echo("\nEstimated Output Size (rough; fixed per-image sizes, not measured from your images):")
    store_mb = total_images * (8 + 2 + 2 + 0.5 + 0.05)  # rgb, gray, detect_mat, objmap, table
    overlay_mb = total_images * 0.5 if overlays else 0.0
    click.echo(f"  - Image stores (rgb, gray, detect_mat, objmap, measurements): ~{store_mb:.1f} MB")
    if overlays:
        click.echo(f"  - Overlay PNGs: ~{overlay_mb:.1f} MB")
    click.echo(f"\n  Estimated total: ~{store_mb + overlay_mb:.1f} MB")


def execute_dry_run(
    config: ExecutionConfig, datasets: List[Dataset], output_dir: Path
) -> None:
    """
    Execute dry-run mode (verbose preview without processing).

    Displays what the run would do, without doing any of it:
    - The grid shape each ``GridImage`` is loaded with, and where it came
      from (``--nrows``/``--ncols``, the pipeline, or the built-in default)
    - The pipeline's steps, from :meth:`ImagePipeline.summary`
    - All datasets and images found
    - Validation results
    - Execution backend selection and configuration
    - The output layout, from the ``phenotypic.sdk_`` path helpers
    - Size estimates

    Args:
        config: Execution configuration
        datasets: List of datasets to process
        output_dir: Output directory path
    """
    pipeline, load_error = _load_preview_pipeline(config)

    click.echo("\n" + "=" * 80)
    click.echo("DRY-RUN MODE: Verbose Preview (No Jobs Will Be Executed)")
    click.echo("=" * 80)

    # Input/output info
    click.echo("\nConfiguration:")
    click.echo(f"  Pipeline config: {config.pipeline_json}")
    click.echo(f"  Input path:    {config.input_path}")
    click.echo(f"  Output dir:    {output_dir}")

    # Image type and grid configuration
    click.echo("\nImage Configuration:")
    click.echo(f"  Image type: {config.image_type}")
    if config.image_type == "GridImage":
        _display_grid_shape(config, pipeline)
    if config.bit_depth:
        click.echo(f"  Bit depth: {config.bit_depth}-bit")

    _display_pipeline_steps(config, pipeline, load_error)

    # Datasets discovery
    click.echo()
    total_images = _display_datasets_detail(datasets)

    # Execution backend
    click.echo()
    backend = _display_execution_backend(config)

    # Backend-specific configuration
    if backend == "slurm":
        for label, profile in _submitted_slurm_profiles(config):
            _display_slurm_config(profile, label)
    else:
        _display_local_config(config)

    # Validation
    # The main CLI path validated before reaching the dry-run exit and would
    # have exited on a failure, so this reports that outcome rather than
    # validating a second time.
    click.echo("\nPipeline Validation:")
    if not config.skip_validation:
        click.echo("  ✓ Configuration validation passed (see above)")
    else:
        click.echo("  ⊘ Validation skipped (--skip-validation flag)")

    _display_output_layout(config, datasets, output_dir)

    forward = not config.measure_only
    overlays = forward and config.save_overlays

    # Processing summary
    click.echo("\nProcessing Summary:")
    click.echo(f"  Total images to process: {total_images}")
    click.echo(f"  Total datasets: {len(datasets)}")
    stores = "written" if forward else "re-measured"
    click.echo(f"  Image stores {stores}: {total_images} (one per image)")
    if overlays:
        click.echo(f"  Overlay images: {total_images} (one per image)")

    # A measure run re-measures existing stores and writes no new pixels.
    if forward:
        _display_size_estimate(total_images, overlays=overlays)

    click.echo("\n" + "=" * 80)
    click.echo("To proceed with execution, run the same command without the --dry-run flag")
    click.echo("=" * 80 + "\n")


def get_sample_datasets(
    datasets: List[Dataset],
    n_samples: int,
    output_dir: Path,
    random_seed: Optional[int] = None
) -> List[Dataset]:
    """
    Create sample datasets with random subset of images.

    Args:
        datasets: Original datasets
        n_samples: Number of images to sample per dataset
        output_dir: Output directory
        random_seed: Optional seed for reproducible sampling

    Returns:
        List of Dataset objects with sampled images
    """
    if random_seed is not None:
        random.seed(random_seed)

    sample_datasets = []

    for dataset in datasets:
        if len(dataset.images) <= n_samples:
            # Dataset is small, use all images
            sample_datasets.append(dataset)
        else:
            # Sample random subset
            sample_images = random.sample(list(dataset.images), n_samples)
            sample_dataset = Dataset(
                name=dataset.name,
                images=sample_images,
                input_dir=dataset.input_dir,
                output_dir=output_dir,
            )
            sample_datasets.append(sample_dataset)

    return sample_datasets
