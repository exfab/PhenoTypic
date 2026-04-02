"""
Output file organization and management for the PhenoTypic CLI.

This module handles all output file creation, directory structure management,
and saving of image layers, measurements, and overlays with comprehensive
error logging to prevent silent data loss.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

import pandas as pd
import polars as pl

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from ._cli_types import Dataset
from ._cli_duckdb_agg import duckdb_aggregate

logger = logging.getLogger(__name__)


def _atomic_write(target: Path, write_func: Callable[[str], None]) -> None:
    """Write to *target* atomically via a temp file and ``os.replace``.

    Args:
        target: Final destination path.
        write_func: Callable that writes content to a given file path string.

    Raises:
        Any exception from *write_func* after cleaning up the temp file.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[str] = None
    try:
        fd = tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.stem}_",
            suffix=".tmp",
            delete=False,
        )
        tmp_path = fd.name
        fd.close()
        write_func(tmp_path)
        with open(tmp_path, "r+b") as f:
            os.fsync(f.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise


def join_metadata(df: "pl.DataFrame", metadata_csv: Path) -> "pl.DataFrame":
    """Join external metadata CSV onto a measurements DataFrame.

    Identifies columns common to both the measurements and metadata,
    casts them to ``String`` for a safe join, and performs an inner join
    with the metadata on the left.  Only rows present in both DataFrames
    survive.  Warns if the row count increases (duplicate metadata keys)
    or decreases (measurement rows with no matching metadata).

    Args:
        df: Measurements DataFrame (must have columns to join on).
        metadata_csv: Path to the metadata CSV file.

    Returns:
        DataFrame with metadata columns first, then measurement columns.
    """
    metadata_df = pl.read_csv(metadata_csv)
    common = list(set(df.columns) & set(metadata_df.columns))
    if not common:
        logger.warning(
            "Metadata CSV has no columns in common with measurements — skipping join"
        )
        return df

    logger.info("Joining metadata on columns: %s", common)
    df = df.with_columns(pl.col(col).cast(pl.String) for col in common)
    metadata_df = metadata_df.with_columns(
        pl.col(col).cast(pl.String) for col in common
    )
    n_rows_before = df.height
    n_cols_before = len(df.columns)
    df = metadata_df.join(df, on=common, how="inner")
    n_new_cols = len(df.columns) - n_cols_before
    if df.height > n_rows_before:
        logger.warning(
            "Metadata join increased row count from %d to %d — "
            "metadata CSV likely has duplicate keys on columns %s. "
            "Verify your metadata CSV has unique values on join columns.",
            n_rows_before,
            df.height,
            common,
        )
    n_dropped = n_rows_before - df.height
    if n_dropped > 0:
        logger.warning(
            "Metadata inner join dropped %d/%d measurement rows "
            "with no matching metadata on columns %s",
            n_dropped,
            n_rows_before,
            common,
        )
    logger.info(
        "Metadata join: added %d columns, %d/%d rows matched",
        n_new_cols,
        df.height,
        n_rows_before,
    )
    return df


def _scratch_dest_name(pq: Path) -> str:
    """Build a collision-safe filename for a parquet staged to $SCRATCH."""
    return f"{pq.parent.parent.name}_{pq.name}"


def _stage_to_scratch(parquet_files: List[Path]) -> Optional[Path]:
    """Copy parquet files to $SCRATCH for faster reading.

    Creates a staging directory using SLURM job/task IDs to avoid
    collisions when multiple aggregation tasks run on the same node.

    Args:
        parquet_files: Paths to copy.

    Returns:
        Path to staging directory, or ``None`` if $SCRATCH is unavailable.
    """
    scratch = os.environ.get("SCRATCH")
    if not scratch:
        return None

    scratch_path = Path(scratch)
    if not scratch_path.is_dir():
        return None

    job_id = os.environ.get("SLURM_JOB_ID", "")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "")
    if job_id and task_id:
        suffix = f"{job_id}_{task_id}"
    elif job_id:
        suffix = job_id
    else:
        suffix = str(os.getpid())

    staging_dir = scratch_path / f".phenotypic_stage_{suffix}"
    try:
        staging_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    try:
        for pq in parquet_files:
            shutil.copy2(pq, staging_dir / _scratch_dest_name(pq))
    except Exception:
        _cleanup_scratch(staging_dir)
        return None

    return staging_dir


def _remap_to_scratch(
    path_to_dataset: Dict[Path, str], scratch_dir: Path
) -> Dict[Path, str]:
    """Remap GPFS paths to their $SCRATCH copies.

    Args:
        path_to_dataset: Original path to dataset name mapping.
        scratch_dir: Staging directory on $SCRATCH.

    Returns:
        New mapping with paths pointing to scratch copies.
    """
    remapped: Dict[Path, str] = {}
    for original_path, dataset_name in path_to_dataset.items():
        remapped[scratch_dir / _scratch_dest_name(original_path)] = dataset_name
    return remapped


def _cleanup_scratch(staging_dir: Path) -> None:
    """Remove staging directory with error suppression."""
    try:
        shutil.rmtree(staging_dir)
    except Exception:
        pass


def aggregate_measurements(
    output_dir: Path,
    dataset_names: List[str],
    include_dataset_column: bool = True,
    metadata_csv: Optional[Path] = None,
) -> Optional[Path]:
    """Aggregate per-image Parquet files into a master CSV via DuckDB.

    Scans ``results/{name}/measurements/`` for each dataset, looking for
    Parquet (``.parquet``) files.  Prefers pre-aggregated
    ``_dataset_aggregated.parquet`` files when available, falling back to
    individual per-image files.

    Uses :func:`duckdb_aggregate` for efficient in-memory concatenation
    and writes both ``master_measurements.csv`` and
    ``master_measurements.parquet`` to *output_dir* using atomic writes.

    When ``$SCRATCH`` is available (node-local SSD), files are staged
    there first to avoid GPFS metadata overhead.

    Works without an :class:`OutputManager` instance so it can be called
    from the SLURM sentinel job.

    Args:
        output_dir: Base output directory (contains ``results/``).
        dataset_names: Names of datasets to scan.
        include_dataset_column: Whether to insert ``Metadata_Dataset``
            into each file that lacks it.
        metadata_csv: Optional path to an external CSV file. When
            provided, shared columns are used as join keys for an inner
            join with metadata on the left.  Only measurement rows that
            match the metadata are kept.

    Returns:
        Path to ``master_measurements.csv``, or ``None`` if no
        measurements were found.
    """
    results_dir = output_dir / "results"

    # -- File discovery ------------------------------------------------
    path_to_dataset: Dict[Path, str] = {}
    for dataset_name in dataset_names:
        meas_dir = results_dir / dataset_name / "measurements"
        if not meas_dir.is_dir():
            continue
        # Prefer pre-aggregated file
        agg_parquet = meas_dir / "_dataset_aggregated.parquet"
        if agg_parquet.exists():
            path_to_dataset[agg_parquet] = dataset_name
        else:
            for pq in sorted(meas_dir.glob("*.parquet")):
                if not pq.name.startswith("_"):
                    path_to_dataset[pq] = dataset_name

    # -- Stage to $SCRATCH ---------------------------------------------
    scratch_dir = _stage_to_scratch(list(path_to_dataset.keys()))
    if scratch_dir is not None:
        active_mapping = _remap_to_scratch(path_to_dataset, scratch_dir)
    else:
        active_mapping = path_to_dataset

    # -- DuckDB aggregation --------------------------------------------
    master_df = duckdb_aggregate(
        file_paths=list(active_mapping.keys()),
        path_to_dataset=active_mapping,
        include_dataset_column=include_dataset_column,
        keep_filename=True,
    )

    if scratch_dir is not None:
        _cleanup_scratch(scratch_dir)

    if master_df is None:
        logger.warning("No valid measurements found for aggregation")
        return None

    # Derive Metadata_ImageFile for the dashboard image viewer, then drop filename.
    if "Metadata_ImageFile" not in master_df.columns and "filename" in master_df.columns:
        master_df = master_df.with_columns(
            pl.col("filename").str.extract(r"([^/\\]+)\.[^.]+$", 1).alias("Metadata_ImageFile")
        )
    if "filename" in master_df.columns:
        master_df = master_df.drop("filename")

    # -- Join metadata -------------------------------------------------
    if metadata_csv is not None:
        try:
            master_df = join_metadata(master_df, metadata_csv)
        except Exception as e:
            logger.warning("Failed to join metadata CSV: %s: %s", type(e).__name__, e)

    # -- Write master CSV and Parquet ----------------------------------
    master_csv_path = output_dir / "master_measurements.csv"
    master_pq_path = output_dir / "master_measurements.parquet"

    try:
        _atomic_write(master_csv_path, master_df.write_csv)
    except Exception:
        logger.error("Failed to save master CSV")
        return None

    try:
        _atomic_write(
            master_pq_path,
            lambda p: master_df.write_parquet(
                p, compression="zstd", compression_level=3
            ),
        )
    except Exception:
        logger.warning("Failed to save master Parquet (CSV was saved)")

    logger.info(
        "Aggregated %d rows x %d cols into %s",
        master_df.height,
        master_df.width,
        master_csv_path.name,
    )
    return master_csv_path


class OutputManager:
    """
    Manages all output file creation and organization for CLI processing.

    Handles directory structure creation, output path resolution, and saving
    of measurements, overlays, and optional image layers (rgb, gray, masks, etc.).
    """

    def __init__(
        self,
        base_dir: Path,
        save_layers: Dict[str, bool],
        extensions: Dict[str, str],
        include_dataset_column: bool = True,
        overlay_alpha: float = 0.3,
    ):
        """
        Initialize OutputManager.

        Args:
            base_dir: Base output directory for all results
            save_layers: Which layers to save {"rgb": True, "gray": False, ...}
            extensions: File extensions for each layer {"rgb": ".tiff", ...}
            include_dataset_column: Whether to add Metadata_Dataset column to measurements (default: True)
            overlay_alpha: Alpha transparency for label overlay (0.0-1.0, default: 0.3)
        """
        self.base_dir = Path(base_dir)
        self.save_layers = save_layers
        self.extensions = extensions
        self.include_dataset_column = include_dataset_column
        self.overlay_alpha = overlay_alpha

        # Results directory for dataset outputs (images, measurements, overlays)
        self.results_dir = self.base_dir / "results"

        # Logs directory (always at root level)
        self.logs_dir = self.base_dir / "logs"

    @classmethod
    def from_config(
        cls,
        base_dir: Path,
        ext: str,
        include_dataset_column: bool = True,
        overlay_alpha: float = 0.3,
    ) -> "OutputManager":
        """Create an OutputManager with the standard fixed layer set.

        Args:
            base_dir: Base output directory.
            ext: Extension for rgb/gray/detect_mat (e.g. ".tiff").
            include_dataset_column: Add Metadata_Dataset to measurements.
            overlay_alpha: Alpha for overlay compositing.
        """
        return cls(
            base_dir=base_dir,
            save_layers={"rgb": True, "gray": True, "detect_mat": True, "objmap": True},
            extensions={"rgb": ext, "gray": ext, "detect_mat": ext, "objmap": ".png"},
            include_dataset_column=include_dataset_column,
            overlay_alpha=overlay_alpha,
        )

    def create_structure(self, datasets: List[Dataset]) -> None:
        """
        Create complete output directory structure.

        Always creates dataset-first structure with each dataset in its own folder.

        Args:
            datasets: List of datasets to create directories for
        """
        # Create base directory
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create results directory for dataset outputs
        self.results_dir.mkdir(exist_ok=True)

        # Create logs directory at root level
        self.logs_dir.mkdir(exist_ok=True)
        (self.logs_dir / "slurm").mkdir(exist_ok=True)

        # Create dataset folders with subdirectories under results/
        for dataset in datasets:
            dataset_dir = self.results_dir / dataset.name
            dataset_dir.mkdir(exist_ok=True)

            (dataset_dir / "measurements").mkdir(exist_ok=True)
            (dataset_dir / "overlays").mkdir(exist_ok=True)
            for layer_name, enabled in self.save_layers.items():
                if enabled:
                    (dataset_dir / layer_name).mkdir(exist_ok=True)

    def get_output_path(
        self,
        dataset_name: str,
        layer: str,
        image_stem: str
    ) -> Path:
        """
        Get the output path for a specific file.

        Args:
            dataset_name: Dataset name (e.g., "single_image", directory name, or subdirectory name)
            layer: Layer type ("measurements", "overlays", "rgb", etc.)
            image_stem: Image filename without extension

        Returns:
            Complete output path for the file
        """
        # Determine extension
        if layer == "measurements":
            ext = ".parquet"
        elif layer == "overlays":
            ext = ".png"
        else:
            if not self.save_layers.get(layer):
                raise ValueError(f"Layer '{layer}' is not enabled")
            ext = self.extensions.get(layer, ".png")

        # Always use: results/dataset/layer/file
        return self.results_dir / dataset_name / layer / f"{image_stem}{ext}"

    def save_measurements(
        self,
        measurements: pd.DataFrame,
        dataset_name: str,
        image_stem: str
    ) -> Path:
        """
        Save measurements as a Parquet file for a single image.

        Args:
            measurements: DataFrame with measurement data
            dataset_name: Dataset name
            image_stem: Image filename without extension

        Returns:
            Path where measurements were saved
        """
        # Add dataset column if requested
        if self.include_dataset_column and "Metadata_Dataset" not in measurements.columns:
            measurements = measurements.copy()
            measurements.insert(0, "Metadata_Dataset", dataset_name)

        output_path = self.get_output_path(dataset_name, "measurements", image_stem)
        parquet_df = pl.from_pandas(measurements)

        _atomic_write(
            output_path,
            lambda p: parquet_df.write_parquet(
                p, compression="zstd", compression_level=3
            ),
        )

        return output_path

    def save_overlay(
        self,
        image: Image,
        dataset_name: str,
        image_stem: str
    ) -> Path:
        """
        Save overlay visualization for a single image.

        Uses full-resolution save_overlay() from the image accessor.
        Prefers RGB overlay if available, falls back to grayscale.

        Args:
            image: Image object with processing results
            dataset_name: Dataset name
            image_stem: Image filename without extension

        Returns:
            Path where overlay was saved
        """
        output_path = self.get_output_path(dataset_name, "overlays", image_stem)

        if not image.rgb.isempty():
            image.rgb.save_overlay(
                filepath=output_path,
                overlay_alpha=self.overlay_alpha
            )
        else:
            image.gray.save_overlay(
                filepath=output_path,
                overlay_alpha=self.overlay_alpha
            )

        return output_path

    def _save_layer_safely(
        self,
        layer_name: str,
        dataset_name: str,
        image_stem: str,
        save_func: Callable[[Path], None],
    ) -> Optional[Path]:
        """Safely save an image layer with error logging.

        Args:
            layer_name: Name of layer (e.g., "rgb", "gray").
            dataset_name: Dataset name.
            image_stem: Image filename stem.
            save_func: Function to call for saving (takes path as argument).

        Returns:
            Path if successful, None if failed.
        """
        try:
            path = self.get_output_path(dataset_name, layer_name, image_stem)
            save_func(path)
            return path
        except Exception as e:
            logger.warning(
                "Failed to save %s for %s/%s: %s: %s",
                layer_name,
                dataset_name,
                image_stem,
                type(e).__name__,
                e,
            )
            return None

    def save_image_layers(
        self,
        image: Image,
        dataset_name: str,
        image_stem: str,
    ) -> Dict[str, Path]:
        """Save all requested image layers (rgb, gray, detect_mat, objmap).

        Args:
            image: Image object with processing results.
            dataset_name: Dataset name.
            image_stem: Image filename without extension.

        Returns:
            Dictionary mapping layer names to saved paths (only successful saves).
        """
        saved_paths: Dict[str, Path] = {}

        layer_accessors = {
            "rgb": image.rgb,
            "gray": image.gray,
            "detect_mat": image.detect_mat,
            "objmap": image.objmap,
        }

        for layer_name, accessor in layer_accessors.items():
            if not self.save_layers.get(layer_name) or accessor.isempty():
                continue
            path = self._save_layer_safely(
                layer_name,
                dataset_name,
                image_stem,
                lambda p, acc=accessor: acc.imsave(filepath=p),
            )
            if path:
                saved_paths[layer_name] = path

        return saved_paths

    def aggregate_master_csv(
        self,
        datasets: List[Dataset],
        metadata_csv: Optional[Path] = None,
    ) -> Optional[Path]:
        """Aggregate per-image measurement Parquet files into master CSV.

        Args:
            datasets: List of all datasets processed.
            metadata_csv: Optional path to external CSV for inner-join
                on shared columns.

        Returns:
            Path to master_measurements.csv, or None if no measurements found.
        """
        return aggregate_measurements(
            output_dir=self.base_dir,
            dataset_names=[ds.name for ds in datasets],
            include_dataset_column=self.include_dataset_column,
            metadata_csv=metadata_csv,
        )
