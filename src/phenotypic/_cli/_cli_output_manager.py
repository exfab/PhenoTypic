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
        include_dataset_column: bool = False
    ):
        """
        Initialize OutputManager.
        
        Args:
            base_dir: Base output directory for all results
            save_layers: Which layers to save {"rgb": True, "gray": False, ...}
            extensions: File extensions for each layer {"rgb": ".tiff", ...}
            include_dataset_column: Whether to add dataset column to CSVs
        """
        self.base_dir = Path(base_dir)
        self.save_layers = save_layers
        self.extensions = extensions
        self.include_dataset_column = include_dataset_column
        
        # Core directories (always created)
        self.measurements_dir = self.base_dir / "measurements"
        self.overlays_dir = self.base_dir / "overlays"
        self.logs_dir = self.base_dir / "logs"
        
        # Optional layer directories
        self.layer_dirs = {
            "rgb": self.base_dir / "rgb" if save_layers.get("rgb") else None,
            "gray": self.base_dir / "gray" if save_layers.get("gray") else None,
            "enh_gray": self.base_dir / "enh_gray" if save_layers.get("enh_gray") else None,
            "objmask": self.base_dir / "objmask" if save_layers.get("objmask") else None,
            "objmap": self.base_dir / "objmap" if save_layers.get("objmap") else None,
            "objmap_rgb": self.base_dir / "objmap_rgb" if save_layers.get("objmap_rgb") else None,
        }
    
    def create_structure(self, datasets: List[Dataset]) -> None:
        """
        Create complete output directory structure.
        
        Args:
            datasets: List of datasets to create directories for
        """
        # Create base directory
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create core directories
        self.measurements_dir.mkdir(exist_ok=True)
        self.overlays_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        (self.logs_dir / "datasets").mkdir(exist_ok=True)
        
        # Create optional layer directories
        for layer_dir in self.layer_dirs.values():
            if layer_dir is not None:
                layer_dir.mkdir(exist_ok=True)
        
        # Create dataset-specific subdirectories if needed
        if len(datasets) > 1 or (len(datasets) == 1 and datasets[0].name != "_root"):
            for dataset in datasets:
                if dataset.name != "_root":
                    (self.measurements_dir / dataset.name).mkdir(exist_ok=True)
                    (self.overlays_dir / dataset.name).mkdir(exist_ok=True)
                    
                    # Create dataset subdirectories for optional layers
                    for layer_dir in self.layer_dirs.values():
                        if layer_dir is not None:
                            (layer_dir / dataset.name).mkdir(exist_ok=True)
    
    def get_output_path(
        self,
        dataset_name: str,
        layer: str,
        image_stem: str
    ) -> Path:
        """
        Get the output path for a specific file.
        
        Args:
            dataset_name: Dataset name ("_root" for root images)
            layer: Layer type ("measurements", "overlays", "rgb", etc.)
            image_stem: Image filename without extension
            
        Returns:
            Complete output path for the file
        """
        # Get the appropriate directory
        if layer == "measurements":
            base = self.measurements_dir
            ext = ".csv"
        elif layer == "overlays":
            base = self.overlays_dir
            ext = ".png"
        else:
            base = self.layer_dirs.get(layer)
            if base is None:
                raise ValueError(f"Layer '{layer}' is not enabled")
            ext = self.extensions.get(layer, ".png")
        
        # Add dataset subdirectory if not root
        if dataset_name != "_root":
            base = base / dataset_name
        
        return base / f"{image_stem}{ext}"
    
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
        if self.include_dataset_column and "Dataset" not in measurements.columns:
            measurements = measurements.copy()
            measurements.insert(0, "Dataset", dataset_name)
        
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
        
        Args:
            image: Image object with processing results
            dataset_name: Dataset name
            image_stem: Image filename without extension
            
        Returns:
            Path where overlay was saved
        """
        output_path = self.get_output_path(dataset_name, "overlays", image_stem)
        
        # Generate overlay
        fig, ax = image.show_overlay()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        
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

        # Save object map RGB visualization if requested
        if self.save_layers.get("objmap_rgb") and not image.objmap.isempty():
            path = self._save_layer_safely(
                "objmap_rgb", image, dataset_name, image_stem,
                lambda p: image.objmap.imsave(filepath=p, use_label2rgb=True)
            )
            if path:
                saved_paths["objmap_rgb"] = path

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
            dataset_meas_dir = self.measurements_dir
            if dataset.name != "_root":
                dataset_meas_dir = dataset_meas_dir / dataset.name

            # Read all CSV files in this dataset's measurement directory
            csv_files = list(dataset_meas_dir.glob("*.csv"))

            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)

                    # Add dataset column if requested and not already present
                    if self.include_dataset_column and "Dataset" not in df.columns:
                        df.insert(0, "Dataset", dataset.name)

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
