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


def load_preview_pipeline(config: ExecutionConfig) -> tuple[Any, Optional[str]]:
    """``(pipeline, None)``, or ``(None, message)`` when the pipeline cannot be read.

    ``config.pipeline_json`` is a path. The message is ``read_pipeline_file``'s,
    the wording CLI validation reports. ``--skip-validation`` skips that
    validation, so a preview must not fail on an unreadable file; it says so
    instead.
    """
    from ._cli_validation import read_pipeline_file

    pipeline, message, _ = read_pipeline_file(config.pipeline_json)
    if pipeline is None:
        lines = (message or "").strip().splitlines()
        return None, lines[0] if lines else "Failed to load pipeline"
    return pipeline, None


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


def describe_grid_shape(config: ExecutionConfig, pipeline: Any) -> tuple[str, bool]:
    """The grid shape a ``GridImage`` run will use, and where it came from.

    The one description both the configuration panel and the dry-run preview
    print, so the two cannot disagree. ``resolve_grid_shape`` returns no
    source, so the source is worked out here with the same precedence, one
    axis at a time (``--nrows`` alone leaves the columns to the pipeline or
    the default).

    Args:
        config: The run's configuration.
        pipeline: The loaded pipeline, or ``None`` when it could not be read.

    Returns:
        ``(text, from_default)``: the description, and whether either axis
        fell back to the built-in default.
    """
    if config.measure_only:
        return "as stored in each image store (--mode measure re-measures existing stores)", False
    if pipeline is None:
        return "unknown; the pipeline could not be read", False

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
    text = f"{nrows} × {ncols} (rows × columns), {origin}"
    return text, _FROM_DEFAULT in (rows_from, cols_from)


def _display_grid_shape(config: ExecutionConfig, pipeline: Any) -> None:
    """Print the grid shape the run will load each ``GridImage`` with, and its source."""
    text, from_default = describe_grid_shape(config, pipeline)
    click.echo(f"  Grid shape: {text}")
    if from_default:
        click.echo(
            "    If your plates differ, pass --nrows/--ncols or set "
            "ImagePipeline(nrows=, ncols=)."
        )


#: Width at which the ``changed`` column wraps, so one long cell (a prefab
#: detector holding a pipeline) does not widen every row of the table.
_CHANGED_WIDTH = 60


def _step_table_lines(table: Any) -> List[str]:
    """``summary()`` as left-aligned text, with ``changed`` wrapped at ``_CHANGED_WIDTH``."""
    import textwrap

    fixed = [column for column in table.columns if column != "changed"]
    rows = [{column: str(value) for column, value in row.items()} for _, row in table.iterrows()]
    widths = {
        column: max([len(column)] + [len(row[column]) for row in rows]) for column in fixed
    }
    indent = sum(widths.values()) + 2 * len(fixed)

    def _line(cells: dict, changed: str) -> str:
        left = "  ".join(cells[column].ljust(widths[column]) for column in fixed)
        return f"{left}  {changed}".rstrip()

    lines = [_line({column: column for column in fixed}, "changed")]
    for row in rows:
        wrapped = textwrap.wrap(row["changed"], width=_CHANGED_WIDTH, break_on_hyphens=False) or [""]
        lines.append(_line(row, wrapped[0]))
        lines.extend(" " * indent + more for more in wrapped[1:])
    return lines


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
    for line in _step_table_lines(table):
        click.echo(f"  {line}")


def _output_layout_entries(
    config: ExecutionConfig, output_dir: Path, pipeline: Any
) -> List[tuple[Path, bool, str]]:
    """``(path, is_dir, what)`` for each output the run writes, from the ``sdk_`` helpers.

    Only what the run actually writes: the analysis, plot and QC outputs
    appear when the pipeline configures them, the metadata snapshot when
    ``--metadata`` is given to a forward run, and overlays on a forward run.
    """
    from phenotypic.sdk_ import (
        analysis_manifest_path,
        dashboard_html_path,
        dataset_overlays_dir,
        deliverables_dir,
        master_measurements_parquet_path,
        measurements_by_category_dir,
        measurements_by_feature_dir,
        measurements_csv_path,
        measurements_parquet_path,
        measurements_tsv_path,
        metadata_snapshot_path,
        named_analysis_tsv_path,
        phenotypic_cache_dir,
        pipeline_json_path,
        plots_dir,
        qc_dir,
        readme_md_path,
        rembi_manifest_path,
        zarr_store_path,
    )

    forward = not config.measure_only
    entries: List[tuple[Path, bool, str]] = [
        (
            measurements_tsv_path(output_dir),
            False,
            "the table to analyse: measurements, joined with --metadata when "
            f"given (also {measurements_parquet_path(output_dir).name}, and "
            f"{measurements_csv_path(output_dir).name} for the results viewer)",
        ),
        (
            master_measurements_parquet_path(output_dir),
            False,
            "every image's measurements, without your --metadata columns",
        ),
        (measurements_by_feature_dir(output_dir), True, "the table split by measurer"),
        (measurements_by_category_dir(output_dir), True, "the table split by measurement category"),
    ]
    if pipeline is not None:
        model = pipeline.get_model()
        if model is not None:
            deliverables = deliverables_dir(output_dir)
            entries += [
                (
                    named_analysis_tsv_path(deliverables, type(model).__name__),
                    False,
                    "the model fit (also .parquet)",
                ),
                (analysis_manifest_path(deliverables), False, "the analysis outputs, listed"),
            ]
        if pipeline.get_plots():
            entries.append((plots_dir(output_dir), True, "the pipeline's configured plots"))
        if pipeline.get_qc() and not config.no_qc:
            entries.append((qc_dir(output_dir), True, "quality-control tables (qc.duckdb)"))
    if forward and config.metadata_csv is not None:
        entries.append(
            (
                metadata_snapshot_path(output_dir, Path(config.metadata_csv).suffix),
                False,
                "a byte-for-byte copy of your --metadata",
            )
        )
    entries += [
        (pipeline_json_path(output_dir), False, "the pipeline this run used"),
        (dashboard_html_path(output_dir), False, "the progress dashboard"),
        (readme_md_path(output_dir), False, "what each output and column holds"),
        (rembi_manifest_path(output_dir), False, "the REMBI run manifest"),
    ]
    if forward and config.save_overlays:
        entries.append(
            (dataset_overlays_dir(output_dir, "<dataset>"), True, "one detection overlay PNG per image")
        )
    entries += [
        (
            zarr_store_path(output_dir, "<dataset>", "<stem>"),
            True,
            "one OME-Zarr store per image: its layers and measurements",
        ),
        (phenotypic_cache_dir(output_dir), True, "machine state: progress, logs, SLURM scripts"),
    ]
    return entries


def _display_output_layout(
    config: ExecutionConfig, datasets: List[Dataset], output_dir: Path, pipeline: Any
) -> None:
    """Print where the run's outputs land, from the ``phenotypic.sdk_`` path helpers."""
    from phenotypic.sdk_ import source_image_stem, zarr_store_path

    entries = _output_layout_entries(config, output_dir, pipeline)

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
    pipeline, load_error = load_preview_pipeline(config)

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

    _display_output_layout(config, datasets, output_dir, pipeline)

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
