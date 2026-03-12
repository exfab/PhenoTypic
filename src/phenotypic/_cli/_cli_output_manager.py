"""
Output file organization and management for the PhenoTypic CLI.

This module handles all output file creation, directory structure management,
and saving of image layers, measurements, and overlays with comprehensive
error logging to prevent silent data loss.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import pandas as pd
import polars as pl

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from ._cli_types import Dataset

logger = logging.getLogger(__name__)


def aggregate_measurements(
    output_dir: Path,
    dataset_names: List[str],
    include_dataset_column: bool = True,
    metadata_csv: Optional[Path] = None,
) -> Optional[Path]:
    """Aggregate per-image measurement CSVs into a single master CSV.

    Scans ``results/{name}/measurements/*.csv`` for each dataset, optionally
    adds a ``Metadata_Dataset`` column, and writes a concatenated
    ``master_measurements.csv`` to *output_dir* using an atomic write.

    Works without an :class:`OutputManager` instance so it can be called
    from the SLURM sentinel job.

    Args:
        output_dir: Base output directory (contains ``results/``).
        dataset_names: Names of datasets to scan.
        include_dataset_column: Whether to insert ``Metadata_Dataset``
            into each CSV that lacks it.
        metadata_csv: Optional path to an external CSV file. When
            provided, shared columns are used as join keys for a left
            merge, adding any extra columns from the metadata CSV to
            the master DataFrame.

    Returns:
        Path to ``master_measurements.csv``, or ``None`` if no
        measurements were found.
    """
    results_dir = output_dir / "results"
    all_measurements: List[pl.DataFrame] = []
    n_skipped = 0

    for dataset_name in dataset_names:
        dataset_meas_dir = results_dir / dataset_name / "measurements"
        if not dataset_meas_dir.is_dir():
            continue

        for csv_file in sorted(dataset_meas_dir.glob("*.csv")):
            try:
                df = pl.read_csv(csv_file)
                if include_dataset_column and "Metadata_Dataset" not in df.columns:
                    df = df.insert_column(
                        0, pl.lit(dataset_name).alias("Metadata_Dataset")
                    )
                all_measurements.append(df)
            except Exception as e:
                logger.warning(
                    "Failed to read %s: %s: %s",
                    csv_file,
                    type(e).__name__,
                    e,
                )
                n_skipped += 1

    if n_skipped:
        logger.warning(
            "Skipped %d CSV file(s) due to read errors", n_skipped
        )

    if not all_measurements:
        logger.warning("No valid measurements found for aggregation")
        return None

    try:
        master_df = pl.concat(all_measurements, how="diagonal_relaxed")
    except Exception as e:
        logger.error("Failed to concatenate measurements: %s", e)
        return None

    # Join external metadata if provided
    if metadata_csv is not None:
        try:
            metadata_df = pl.read_csv(metadata_csv)
            common = list(set(master_df.columns) & set(metadata_df.columns))
            if not common:
                logger.warning(
                    "Metadata CSV has no columns in common with measurements — skipping join"
                )
            else:
                logger.info("Joining metadata on columns: %s", common)
                # Cast join keys to string so mismatched dtypes don't
                # cause silent null results (e.g. int vs str plate IDs)
                master_df = master_df.with_columns(
                    pl.col(col).cast(pl.String) for col in common
                )
                metadata_df = metadata_df.with_columns(
                    pl.col(col).cast(pl.String) for col in common
                )
                n_rows_before = master_df.height
                n_cols_before = len(master_df.columns)
                master_df = master_df.join(metadata_df, on=common, how="left")
                n_new_cols = len(master_df.columns) - n_cols_before
                if master_df.height > n_rows_before:
                    logger.warning(
                        "Metadata join increased row count from %d to %d — "
                        "metadata CSV likely has duplicate keys on columns %s. "
                        "Verify your metadata CSV has unique values on join columns.",
                        n_rows_before,
                        master_df.height,
                        common,
                    )
                metadata_only_cols = [
                    c for c in metadata_df.columns if c not in set(common)
                ]
                if n_new_cols > 0 and metadata_only_cols:
                    n_matched = master_df.filter(
                        ~pl.all_horizontal(
                            pl.col(c).is_null() for c in metadata_only_cols
                        )
                    ).height
                else:
                    n_matched = master_df.height
                logger.info(
                    "Metadata join: added %d columns, %d/%d rows matched",
                    n_new_cols,
                    n_matched,
                    master_df.height,
                )
        except Exception as e:
            logger.warning("Failed to join metadata CSV: %s: %s", type(e).__name__, e)

    master_path = output_dir / "master_measurements.csv"

    # Atomic write: temp file + os.replace()
    tmp_path = None
    try:
        fd = tempfile.NamedTemporaryFile(
            dir=output_dir,
            prefix=".master_measurements_",
            suffix=".tmp",
            delete=False,
        )
        tmp_path = fd.name
        fd.close()
        master_df.write_csv(tmp_path)
        with open(tmp_path, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp_path, master_path)
    except BaseException:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        logger.error("Failed to save master CSV")
        return None

    n_files = len(all_measurements)
    logger.info(
        "Aggregated %d CSV files into %s (%d total rows)",
        n_files,
        master_path.name,
        master_df.height,
    )
    return master_path


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
            include_dataset_column: Whether to add Metadata_Dataset column to CSVs (default: True)
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
            include_dataset_column: Add Metadata_Dataset to CSVs.
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
            ext = ".csv"
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
        Save measurements CSV for a single image.

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
        measurements.to_csv(output_path, index=False)
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
        image: Image,
        dataset_name: str,
        image_stem: str,
        save_func: callable
    ) -> Optional[Path]:
        """
        Safely save an image layer with error logging.

        Args:
            layer_name: Name of layer (e.g., "rgb", "gray")
            image: Image object
            dataset_name: Dataset name
            image_stem: Image filename stem
            save_func: Function to call for saving (takes path as argument)

        Returns:
            Path if successful, None if failed
        """
        try:
            path = self.get_output_path(dataset_name, layer_name, image_stem)
            save_func(path)
            return path
        except Exception as e:
            logger.warning(
                f"Failed to save {layer_name} for {dataset_name}/{image_stem}: "
                f"{type(e).__name__}: {e}"
            )
            return None

    def save_image_layers(
        self,
        image: Image,
        dataset_name: str,
        image_stem: str
    ) -> Dict[str, Path]:
        """
        Save all requested image layers (rgb, gray, detect_mat, objmap).

        Args:
            image: Image object with processing results
            dataset_name: Dataset name
            image_stem: Image filename without extension

        Returns:
            Dictionary mapping layer names to saved paths (only successful saves)
        """
        saved_paths = {}

        # Save RGB if requested and not empty
        if self.save_layers.get("rgb") and not image.rgb.isempty():
            path = self._save_layer_safely(
                "rgb", image, dataset_name, image_stem,
                lambda p: image.rgb.imsave(p)
            )
            if path:
                saved_paths["rgb"] = path

        # Save grayscale if requested
        if self.save_layers.get("gray") and not image.gray.isempty():
            path = self._save_layer_safely(
                "gray", image, dataset_name, image_stem,
                lambda p: image.gray.imsave(filepath=p)
            )
            if path:
                saved_paths["gray"] = path

        # Save detection matrix if requested
        if self.save_layers.get("detect_mat") and not image.detect_mat.isempty():
            path = self._save_layer_safely(
                "detect_mat", image, dataset_name, image_stem,
                lambda p: image.detect_mat.imsave(filepath=p)
            )
            if path:
                saved_paths["detect_mat"] = path

        # Save object map if requested
        if self.save_layers.get("objmap") and not image.objmap.isempty():
            path = self._save_layer_safely(
                "objmap", image, dataset_name, image_stem,
                lambda p: image.objmap.imsave(filepath=p)
            )
            if path:
                saved_paths["objmap"] = path

        return saved_paths
    
    def aggregate_master_csv(
        self,
        datasets: List[Dataset],
        metadata_csv: Optional[Path] = None,
    ) -> Optional[Path]:
        """Aggregate all individual measurement CSVs into master CSV.

        Args:
            datasets: List of all datasets processed.
            metadata_csv: Optional path to external CSV for left-join
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
