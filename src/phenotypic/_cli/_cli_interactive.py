"""
Interactive features for the PhenoTypic CLI.

This module implements dry-run mode, sample processing, and resume logic.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional

import click

from phenotypic.schema import EXPERIMENT_METADATA
from ._cli_types import Dataset, ExecutionConfig
from ._cli_validation import full_validation


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


def _display_slurm_config(slurm_args: dict) -> None:
    """Display SLURM configuration parameters as table."""
    if not slurm_args:
        return

    click.echo("\n  SLURM Configuration Parameters:")
    for key, value in slurm_args.items():
        click.echo(f"    {key:.<30} {value}")

    click.echo("\n  Converted SBATCH Directives:")
    for key, value in slurm_args.items():
        directive_name = key.replace("slurm_", "").replace("_", "-")

        # Handle special cases
        if key in ("time", "slurm_time"):
            if isinstance(value, int):
                hours = value // 60
                minutes = value % 60
                value_display = f"{hours:02d}:{minutes:02d}:00 (from {value} minutes)"
            else:
                value_display = value
            directive_name = "time"
        elif key == "mem_gb":
            value_display = f"{value}G"
            directive_name = "mem"
        elif key == "slurm_mem":
            value_display = value
            directive_name = "mem"
        else:
            value_display = value

        click.echo(f"    #SBATCH --{directive_name}={value_display}")


def _display_local_config(config: ExecutionConfig) -> None:
    """Display local parallel execution configuration."""
    click.echo("\n  Local Parallel Configuration:")
    if config.n_jobs == -1:
        click.echo("    n_jobs:.............. -1 (use all available CPU cores)")
    else:
        click.echo(f"    n_jobs:.............. {config.n_jobs}")


def _display_output_structure(config: ExecutionConfig, datasets: List[Dataset], output_dir: Path) -> None:
    """Display the output directory structure that will be created."""
    click.echo("\nOutput Directory Structure:")
    click.echo(f"  {output_dir}/")

    ext = config.ext.lstrip(".")

    # Dataset-first mode: dataset folders contain layer subdirectories
    dataset_names = [d.name for d in datasets[:2]]  # Show first 2 datasets
    for i, ds_name in enumerate(dataset_names):
        click.echo(f"    ├── {ds_name}/")
        click.echo("    │   ├── measurements/ (*.csv)")
        click.echo("    │   ├── overlays/ (*.png)")
        click.echo(f"    │   ├── rgb/ (*.{ext})")
        click.echo(f"    │   ├── gray/ (*.{ext})")
        click.echo(f"    │   ├── detect_mat/ (*.{ext})")
        click.echo("    │   ├── objmap/ (*.png)")

        if i == 0 and len(datasets) > 1:
            click.echo("    │   └── ...")

    if len(datasets) > 2:
        click.echo("    ├── ... (more datasets)")

    click.echo("    ├── logs/")
    click.echo("    │   └── slurm/ (if using SLURM execution)")
    click.echo("    ├── slurm_scripts/ (if using SLURM execution)")
    click.echo("    ├── processing_state.json")
    click.echo("    ├── processing_events.log")
    click.echo("    ├── processing_report.html")
    click.echo("    ├── master_measurements.csv")
    click.echo("    ├── measurements.csv  (editable copy for GUI; refreshed on every run)")
    click.echo("    ├── pipeline.json  (reproducibility spec — read by the analysis GUI)")
    click.echo("    ├── analysis.csv   (model fit; written when pipeline has a `model` configured)")
    click.echo("    └── ... (other results)")


def _display_save_configuration(config: ExecutionConfig) -> None:
    """Display save configuration."""
    ext = config.ext.lstrip(".")
    click.echo("\nSave Configuration:")
    click.echo("  Layers always saved:")
    click.echo(f"    - RGB (*.{ext})")
    click.echo(f"    - Grayscale (*.{ext})")
    click.echo(f"    - Detection matrix (*.{ext})")
    click.echo("    - Object map (*.png)")
    click.echo("    - Overlays (*.png)")
    click.echo("    - Measurements (*.csv)")

    if config.include_dataset_column:
        click.echo(
            f"  Master CSV will include {str(EXPERIMENT_METADATA.DATASET)!r} column"
            " for multi-dataset analysis"
        )


def execute_dry_run(
    config: ExecutionConfig, datasets: List[Dataset], output_dir: Path
) -> None:
    """
    Execute dry-run mode (verbose preview without processing).

    Displays comprehensive information about what would be processed including:
    - All datasets and images found
    - Pipeline operations and parameters
    - Validation results
    - Execution backend selection and configuration
    - Output directory structure
    - Size estimates

    Args:
        config: Execution configuration
        datasets: List of datasets to process
        output_dir: Output directory path
    """
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
        click.echo(f"  Grid dimensions: {config.nrows} rows × {config.ncols} columns")
    if config.bit_depth:
        click.echo(f"  Bit depth: {config.bit_depth}-bit")

    # Datasets discovery
    click.echo()
    total_images = _display_datasets_detail(datasets)

    # Execution backend
    click.echo()
    backend = _display_execution_backend(config)

    # Backend-specific configuration
    if backend == "slurm":
        _display_slurm_config(config.slurm_args)
    else:
        _display_local_config(config)

    # Validation
    click.echo("\nPipeline Validation:")
    if not config.skip_validation:
        is_valid, errors = full_validation(config)
        if is_valid:
            click.echo("  ✓ Configuration validation passed")
        else:
            click.echo("  ✗ Configuration validation FAILED:")
            for error in errors:
                click.echo(f"    - {error}")
    else:
        click.echo("  ⊘ Validation skipped (--skip-validation flag)")

    # Output configuration
    click.echo()
    _display_output_structure(config, datasets, output_dir)

    # Save options
    click.echo()
    _display_save_configuration(config)

    # Processing summary
    click.echo("\nProcessing Summary:")
    click.echo(f"  Total images to process: {total_images}")
    click.echo(f"  Total datasets: {len(datasets)}")
    click.echo(f"  Expected CSV files: {total_images} (one per image in measurements/)")
    click.echo(f"  Expected overlay images: {total_images} (one per image in overlays/)")

    # Output size estimate (rough)
    click.echo("\nEstimated Output Size:")
    meas_mb = total_images * 0.05
    overlay_mb = total_images * 0.5
    rgb_mb = total_images * 8
    gray_mb = total_images * 2
    detect_mb = total_images * 2
    objmap_mb = total_images * 0.5
    est_size_mb = meas_mb + overlay_mb + rgb_mb + gray_mb + detect_mb + objmap_mb

    click.echo(f"  - Measurements CSVs: ~{meas_mb:.1f} MB")
    click.echo(f"  - Overlay PNGs: ~{overlay_mb:.1f} MB")
    click.echo(f"  - RGB images: ~{rgb_mb:.1f} MB")
    click.echo(f"  - Grayscale images: ~{gray_mb:.1f} MB")
    click.echo(f"  - Detection matrix: ~{detect_mb:.1f} MB")
    click.echo(f"  - Object maps: ~{objmap_mb:.1f} MB")
    click.echo(f"\n  Estimated total: ~{est_size_mb:.1f} MB")

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
