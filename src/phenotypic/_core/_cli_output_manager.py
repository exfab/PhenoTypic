"""
Output file organization and management for the PhenoTypic CLI.

This module handles all output file creation, directory structure management,
and saving of image layers, measurements, and overlays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING
import pandas as pd
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from phenotypic import Image

from ._cli_types import Dataset


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
            Dictionary mapping layer names to saved paths
        """
        saved_paths = {}
        
        # Save RGB if requested
        if self.save_layers.get("rgb") and not image.rgb.isempty():
            path = self.get_output_path(dataset_name, "rgb", image_stem)
            try:
                image.rgb.imsave(filepath=path)
                saved_paths["rgb"] = path
            except Exception:
                pass  # Skip if fails
        
        # Save grayscale if requested
        if self.save_layers.get("gray") and not image.gray.isempty():
            path = self.get_output_path(dataset_name, "gray", image_stem)
            try:
                image.gray.imsave(filepath=path)
                saved_paths["gray"] = path
            except Exception:
                pass
        
        # Save enhanced grayscale if requested
        if self.save_layers.get("enh_gray") and not image.enh_gray.isempty():
            path = self.get_output_path(dataset_name, "enh_gray", image_stem)
            try:
                image.enh_gray.imsave(filepath=path)
                saved_paths["enh_gray"] = path
            except Exception:
                pass
        
        # Save object mask if requested
        if self.save_layers.get("objmask") and not image.objmask.isempty():
            path = self.get_output_path(dataset_name, "objmask", image_stem)
            try:
                image.objmask.imsave(filepath=path)
                saved_paths["objmask"] = path
            except Exception:
                pass
        
        # Save object map if requested
        if self.save_layers.get("objmap") and not image.objmap.isempty():
            path = self.get_output_path(dataset_name, "objmap", image_stem)
            try:
                image.objmap.imsave(filepath=path)
                saved_paths["objmap"] = path
            except Exception:
                pass
        
        # Save object map RGB visualization if requested
        if self.save_layers.get("objmap_rgb") and not image.objmap.isempty():
            path = self.get_output_path(dataset_name, "objmap_rgb", image_stem)
            try:
                image.objmap.imsave(filepath=path, use_label2rgb=True)
                saved_paths["objmap_rgb"] = path
            except Exception:
                pass
        
        return saved_paths
    
    def aggregate_master_csv(
        self,
        datasets: List[Dataset]
    ) -> Path:
        """
        Aggregate all individual measurement CSVs into master CSV.
        
        Args:
            datasets: List of all datasets processed
            
        Returns:
            Path to master_measurements.csv
        """
        all_measurements = []
        
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
                except Exception:
                    continue  # Skip files that can't be read
        
        if not all_measurements:
            # No valid measurements found
            return None
        
        # Concatenate all measurements
        master_df = pd.concat(all_measurements, axis=0, ignore_index=True)
        
        # Save master CSV
        master_path = self.base_dir / "master_measurements.csv"
        master_df.to_csv(master_path, index=False)
        
        return master_path
