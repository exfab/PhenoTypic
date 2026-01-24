"""
Output file organization and management for the PhenoTypic CLI.

This module handles all output file creation, directory structure management,
and saving of image layers, measurements, and overlays with comprehensive
error logging to prevent silent data loss.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING
import pandas as pd
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from phenotypic import Image

from ._cli_types import Dataset

logger = logging.getLogger(__name__)


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
        overlay_mode: str = "image",
        overlay_alpha: float = 0.3,
    ):
        """
        Initialize OutputManager.

        Args:
            base_dir: Base output directory for all results
            save_layers: Which layers to save {"rgb": True, "gray": False, ...}
            extensions: File extensions for each layer {"rgb": ".tiff", ...}
            include_dataset_column: Whether to add Metadata_Dataset column to CSVs (default: True)
            overlay_mode: "image" for full-resolution save_overlay(), "figure" for matplotlib (default: "image")
            overlay_alpha: Alpha transparency for label overlay (0.0-1.0, default: 0.3)
        """
        self.base_dir = Path(base_dir)
        self.save_layers = save_layers
        self.extensions = extensions
        self.include_dataset_column = include_dataset_column
        self.overlay_mode = overlay_mode
        self.overlay_alpha = overlay_alpha

        # Results directory for dataset outputs (images, measurements, overlays)
        self.results_dir = self.base_dir / "results"

        # Logs directory (always at root level)
        self.logs_dir = self.base_dir / "logs"
    
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

        Uses the configured overlay_mode:
        - "image": Full-resolution save using accessor's save_overlay() method
        - "figure": Matplotlib figure saving (original behavior)

        Prefers RGB overlay if available, falls back to grayscale.

        Args:
            image: Image object with processing results
            dataset_name: Dataset name
            image_stem: Image filename without extension

        Returns:
            Path where overlay was saved
        """
        output_path = self.get_output_path(dataset_name, "overlays", image_stem)

        if self.overlay_mode == "figure":
            # Use matplotlib (original behavior)
            fig, ax = image.show_overlay()
            fig.savefig(output_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
        else:
            # Use full-resolution save_overlay (new default)
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
        Save all requested image layers (rgb, gray, masks, etc.).

        Args:
            image: Image object with processing results
            dataset_name: Dataset name
            image_stem: Image filename without extension

        Returns:
            Dictionary mapping layer names to saved paths (only successful saves)
        """
        saved_paths = {}

        # Save RGB if requested
        if self.save_layers.get("rgb") and not image.rgb.isempty():
            path = self._save_layer_safely(
                "rgb", image, dataset_name, image_stem,
                lambda p: image.rgb.imsave(filepath=p)
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

        # Save enhanced grayscale if requested
        if self.save_layers.get("enh_gray") and not image.enh_gray.isempty():
            path = self._save_layer_safely(
                "enh_gray", image, dataset_name, image_stem,
                lambda p: image.enh_gray.imsave(filepath=p)
            )
            if path:
                saved_paths["enh_gray"] = path

        # Save object mask if requested
        if self.save_layers.get("objmask") and not image.objmask.isempty():
            path = self._save_layer_safely(
                "objmask", image, dataset_name, image_stem,
                lambda p: image.objmask.imsave(filepath=p)
            )
            if path:
                saved_paths["objmask"] = path

        # Save object map if requested
        if self.save_layers.get("objmap") and not image.objmap.isempty():
            path = self._save_layer_safely(
                "objmap", image, dataset_name, image_stem,
                lambda p: image.objmap.imsave(filepath=p)
            )
            if path:
                saved_paths["objmap"] = path

        # Save object map overlay (label2rgb colorized, renamed from objmap_rgb)
        # Support both old "objmap_rgb" and new "objmap_overlay" keys for backward compatibility
        if (self.save_layers.get("objmap_overlay") or self.save_layers.get("objmap_rgb")) and not image.objmap.isempty():
            layer_name = "objmap_overlay" if self.save_layers.get("objmap_overlay") else "objmap_rgb"
            path = self._save_layer_safely(
                layer_name, image, dataset_name, image_stem,
                lambda p: image.objmap.imsave(filepath=p, use_label2rgb=True)
            )
            if path:
                saved_paths[layer_name] = path

        # Save enhanced grayscale overlay if requested
        if self.save_layers.get("enh_gray_overlay") and not image.enh_gray.isempty():
            path = self._save_layer_safely(
                "enh_gray_overlay", image, dataset_name, image_stem,
                lambda p: image.enh_gray.save_overlay(filepath=p, overlay_alpha=self.overlay_alpha)
            )
            if path:
                saved_paths["enh_gray_overlay"] = path

        # Save object mask overlay if requested
        if self.save_layers.get("objmask_overlay") and not image.objmask.isempty():
            path = self._save_layer_safely(
                "objmask_overlay", image, dataset_name, image_stem,
                lambda p: image.objmask.save_overlay(filepath=p, overlay_alpha=self.overlay_alpha)
            )
            if path:
                saved_paths["objmask_overlay"] = path

        return saved_paths
    
    def aggregate_master_csv(
        self,
        datasets: List[Dataset]
    ) -> Optional[Path]:
        """
        Aggregate all individual measurement CSVs into master CSV.

        Args:
            datasets: List of all datasets processed

        Returns:
            Path to master_measurements.csv, or None if no measurements found
        """
        all_measurements = []
        skipped_files = []

        for dataset in datasets:
            # Always use: results/dataset_name/measurements/
            dataset_meas_dir = self.results_dir / dataset.name / "measurements"

            # Read all CSV files in this dataset's measurement directory
            csv_files = list(dataset_meas_dir.glob("*.csv"))

            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)

                    # Add dataset column if requested and not already present
                    if self.include_dataset_column and "Metadata_Dataset" not in df.columns:
                        df.insert(0, "Metadata_Dataset", dataset.name)

                    all_measurements.append(df)

                except Exception as e:
                    logger.warning(
                        f"Failed to read {csv_file.relative_to(self.base_dir)}: "
                        f"{type(e).__name__}: {e}"
                    )
                    skipped_files.append((csv_file, str(e)))

        # Report skipped files
        if skipped_files:
            logger.warning(
                f"Skipped {len(skipped_files)} CSV file(s) due to read errors"
            )
            for csv_file, error in skipped_files[:5]:  # Show first 5
                logger.debug(f"  - {csv_file.name}: {error}")
            if len(skipped_files) > 5:
                logger.debug(f"  ... and {len(skipped_files) - 5} more")

        if not all_measurements:
            logger.warning("No valid measurements found for aggregation")
            return None

        # Concatenate all measurements
        try:
            master_df = pd.concat(all_measurements, axis=0, ignore_index=True)
        except Exception as e:
            logger.error(f"Failed to concatenate measurements: {e}")
            return None

        # Save master CSV
        master_path = self.base_dir / "master_measurements.csv"
        try:
            master_df.to_csv(master_path, index=False)
            logger.info(
                f"Aggregated {len(all_measurements)} CSV files "
                f"into {master_path.name} ({len(master_df)} total rows)"
            )
            return master_path
        except Exception as e:
            logger.error(f"Failed to save master CSV: {e}")
            return None
