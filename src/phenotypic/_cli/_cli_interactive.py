"""
Interactive features for the PhenoTypic CLI.

This module implements dry-run mode, sample processing, and resume logic.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List

import click

from ._cli_types import Dataset, ExecutionConfig


def execute_dry_run(
    config: ExecutionConfig, datasets: List[Dataset], output_dir: Path
) -> None:
    """
    Execute dry-run mode (preview without processing).

    Displays a summary of what would be processed without actually
    executing the pipeline.

    Args:
        config: Execution configuration
        datasets: List of datasets to process
        output_dir: Output directory path
    """
    click.echo("\n" + "=" * 60)
    click.echo("DRY RUN SUMMARY")
    click.echo("=" * 60 + "\n")

    # Input/output info
    click.echo(f"Pipeline: {config.pipeline_json}")
    click.echo(f"Input:    {config.input_path}")
    click.echo(f"Output:   {output_dir}\n")

    # Dataset info
    click.echo("Datasets Found:")
    total_images = 0
    for dataset in datasets:
        count = len(dataset.images)
        total_images += count
        display_name = (
            "Root directory images" if dataset.name == "_root" else dataset.name
        )
        click.echo(f"  - {display_name}: {count} images")
    click.echo(f"\n  Total: {total_images} images\n")

    # Image configuration
    click.echo("Image Configuration:")
    click.echo(f"  - Type: {config.image_type}")
    if config.image_type == "GridImage":
        click.echo(f"  - Grid dimensions: {config.nrows}x{config.ncols}")
    if config.bit_depth:
        click.echo(f"  - Bit depth: {config.bit_depth}")
    click.echo()

    # Execution strategy
    if config.is_slurm_mode():
        mode = "SLURM (autonomous)"
        click.echo(f"Execution Strategy: {mode}")
        click.echo(f"  - Jobs to submit: {total_images} (1 per image)")
        click.echo(f"  - SLURM parameters: {len(config.slurm_kwds)} configured")
        if config.wait:
            click.echo("  - Wait mode: enabled (will monitor progress)")
    else:
        mode = f"Local parallel (n_jobs={config.n_jobs})"
        click.echo(f"Execution Strategy: {mode}")
        if config.n_jobs == -1:
            click.echo("  - Using all available CPU cores")

    # Output size estimate (rough)
    click.echo("\nOutput Size Estimate:")
    est_size_mb = total_images * 2  # ~2MB per image (measurements + overlay)

    click.echo(f"  - Measurements CSVs: ~{total_images * 0.05:.1f} MB")
    click.echo(f"  - Overlay PNGs: ~{total_images * 0.5:.1f} MB")

    # Add estimates for optional layers
    if config.save_rgb or config.save_gray or config.save_enh_gray:
        layer_size_mb = 0
        if config.save_rgb:
            layer_size_mb += total_images * 8  # ~8MB per RGB image
        if config.save_gray:
            layer_size_mb += total_images * 2  # ~2MB per grayscale
        if config.save_enh_gray:
            layer_size_mb += total_images * 2

        est_size_mb += layer_size_mb
        click.echo(f"  - Optional layers: ~{layer_size_mb:.1f} MB")

    if config.save_objmask or config.save_objmap or config.save_objmap_rgb:
        mask_size_mb = 0
        if config.save_objmask:
            mask_size_mb += total_images * 0.5
        if config.save_objmap:
            mask_size_mb += total_images * 0.5
        if config.save_objmap_rgb:
            mask_size_mb += total_images * 2

        est_size_mb += mask_size_mb
        click.echo(f"  - Mask/objmap layers: ~{mask_size_mb:.1f} MB")

    click.echo(f"\n  Estimated total: ~{est_size_mb:.1f} MB")

    # Dataset column info
    if config.include_dataset_column:
        click.echo(
            "\nNote: Master CSV will include 'Dataset' column for multi-dataset analysis"
        )

    click.echo("\n" + "=" * 60)
    click.echo("To proceed, run without --dry-run flag")
    click.echo("=" * 60 + "\n")


def get_sample_datasets(
    datasets: List[Dataset], n_samples: int, output_dir: Path
) -> List[Dataset]:
    """
    Create sample datasets with random subset of images.

    Args:
        datasets: Original datasets
        n_samples: Number of images to sample per dataset
        output_dir: Output directory

    Returns:
        List of Dataset objects with sampled images
    """
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
