"""
Single image processor for the PhenoTypic CLI.

This module provides a standalone CLI for processing individual images,
designed to be called by SLURM batch scripts for autonomous execution.
"""

from __future__ import annotations

import sys
import logging
import click
import traceback
from pathlib import Path
from typing import Optional, Literal, Dict, Any

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from phenotypic import Image, GridImage, ImagePipeline
from ._cli_output_manager import OutputManager
from ._cli_update_state import append_completion_event
from ._cli_utils import normalize_extension

logger = logging.getLogger(__name__)


def process_single_image_core(
    pipeline_path: Path,
    image_path: Path,
    output_dir: Path,
    dataset_name: str,
    image_type: Literal["Image", "GridImage"],
    read_kwargs: Dict[str, Any],
    output_manager: OutputManager
) -> bool:
    """
    Core processing logic for a single image.

    Args:
        pipeline_path: Path to pipeline JSON file
        image_path: Path to input image
        output_dir: Base output directory
        dataset_name: Dataset name for this image
        image_type: "Image" or "GridImage"
        read_kwargs: Kwargs for imread (nrows, ncols, bit_depth)
        output_manager: OutputManager instance

    Returns:
        True if successful. This function always returns True on success;
        failures are communicated by raising exceptions rather than returning False.

    Raises:
        Exception: Any exception from pipeline loading, image reading, or processing
            will propagate to the caller. The caller is responsible for catching
            exceptions and handling failures appropriately.
    """
    # Load pipeline
    pipeline = ImagePipeline.from_json(pipeline_path)
    
    # Determine image class
    image_cls = GridImage if image_type == "GridImage" else Image
    
    # Load image
    image = image_cls.imread(image_path, **read_kwargs)
    
    # Execute pipeline
    measurements = pipeline.apply_and_measure(image, inplace=True)
    
    # Get image stem for output filenames
    image_stem = image_path.stem
    
    # Save measurements
    output_manager.save_measurements(measurements, dataset_name, image_stem)
    
    # Save overlay
    output_manager.save_overlay(image, dataset_name, image_stem)
    
    # Save optional image layers
    output_manager.save_image_layers(image, dataset_name, image_stem)
    
    return True


@click.command()
@click.option("--pipeline", type=click.Path(exists=True, path_type=Path), required=True,
              help="Path to pipeline JSON file")
@click.option("--image", type=click.Path(exists=True, path_type=Path), required=True,
              help="Path to input image")
@click.option("--output-dir", type=click.Path(path_type=Path), required=True,
              help="Base output directory")
@click.option("--dataset-name", required=True,
              help="Dataset name (subdirectory name or '_root')")
@click.option("--image-type", type=click.Choice(["Image", "GridImage"]), default="GridImage",
              help="Image class to use")
@click.option("--nrows", type=int, default=8,
              help="Number of grid rows (for GridImage)")
@click.option("--ncols", type=int, default=12,
              help="Number of grid columns (for GridImage)")
@click.option("--bit-depth", type=int, default=None,
              help="Bit depth (8 or 16)")
@click.option("--save-rgb", is_flag=True,
              help="Save RGB layer")
@click.option("--save-gray", is_flag=True,
              help="Save grayscale layer")
@click.option("--save-enh-gray", is_flag=True,
              help="Save enhanced grayscale layer")
@click.option("--save-objmask", is_flag=True,
              help="Save object mask")
@click.option("--save-objmap", is_flag=True,
              help="Save object map")
@click.option("--save-objmap-overlay", is_flag=True,
              help="Save object map overlay (colorized labels)")
@click.option("--save-enh-gray-overlay", is_flag=True,
              help="Save enhanced grayscale overlay")
@click.option("--save-objmask-overlay", is_flag=True,
              help="Save object mask overlay")
@click.option("--rgb-ext", default="tiff",
              help="File extension for RGB saves")
@click.option("--gray-ext", default="tiff",
              help="File extension for grayscale saves")
@click.option("--enh-gray-ext", default="tiff",
              help="File extension for enhanced grayscale saves")
@click.option("--objmask-ext", default="png",
              help="File extension for object mask saves")
@click.option("--objmap-ext", default="png",
              help="File extension for object map saves")
@click.option("--objmap-overlay-ext", default="png",
              help="File extension for object map overlay saves")
@click.option("--overlay-mode", type=click.Choice(["image", "figure"]), default="image",
              help="Overlay saving mode: 'image' for full-resolution, 'figure' for matplotlib")
@click.option("--overlay-alpha", type=float, default=0.3,
              help="Alpha transparency for label overlay (0.0-1.0)")
@click.option("--no-dataset-column", "include_dataset_column",
              is_flag=True, flag_value=False, default=True,
              help="Exclude Metadata_Dataset column from measurements CSV (included by default)")
@click.option("--event-log", type=click.Path(path_type=Path), default=None,
              help="Path to event log file (for status updates)")
def main(
    pipeline: Path,
    image: Path,
    output_dir: Path,
    dataset_name: str,
    image_type: str,
    nrows: int,
    ncols: int,
    bit_depth: Optional[int],
    save_rgb: bool,
    save_gray: bool,
    save_enh_gray: bool,
    save_objmask: bool,
    save_objmap: bool,
    save_objmap_overlay: bool,
    save_enh_gray_overlay: bool,
    save_objmask_overlay: bool,
    rgb_ext: str,
    gray_ext: str,
    enh_gray_ext: str,
    objmask_ext: str,
    objmap_ext: str,
    objmap_overlay_ext: str,
    overlay_mode: str,
    overlay_alpha: float,
    include_dataset_column: bool,
    event_log: Optional[Path]
):
    """
    Process a single image with PhenoTypic pipeline.
    
    This is designed to be called by SLURM batch scripts for autonomous
    execution. It processes one image and logs completion to event log.
    """
    try:
        # Prepare read kwargs
        read_kwargs = {}
        if image_type == "GridImage":
            read_kwargs["nrows"] = nrows
            read_kwargs["ncols"] = ncols
        if bit_depth is not None:
            read_kwargs["bit_depth"] = bit_depth
        
        # Prepare save layers
        save_layers = {
            "rgb": save_rgb,
            "gray": save_gray,
            "enh_gray": save_enh_gray,
            "objmask": save_objmask,
            "objmap": save_objmap,
            "objmap_overlay": save_objmap_overlay,
            "enh_gray_overlay": save_enh_gray_overlay,
            "objmask_overlay": save_objmask_overlay,
        }

        # Prepare extensions - validate all before processing
        try:
            extensions = {
                "rgb": normalize_extension(rgb_ext, ".tiff"),
                "gray": normalize_extension(gray_ext, ".tiff"),
                "enh_gray": normalize_extension(enh_gray_ext, ".tiff"),
                "objmask": normalize_extension(objmask_ext, ".png"),
                "objmap": normalize_extension(objmap_ext, ".png"),
                "objmap_overlay": normalize_extension(objmap_overlay_ext, ".png"),
            }
        except click.BadParameter as e:
            logger.error(f"Invalid extension parameter: {e}")
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

        # Create output manager
        output_manager = OutputManager(
            base_dir=output_dir,
            save_layers=save_layers,
            extensions=extensions,
            include_dataset_column=include_dataset_column,
            overlay_mode=overlay_mode,
            overlay_alpha=overlay_alpha,
        )
        
        # Process image
        click.echo(f"Processing {image.name}...")
        success = process_single_image_core(
            pipeline_path=pipeline,
            image_path=image,
            output_dir=output_dir,
            dataset_name=dataset_name,
            image_type=image_type,
            read_kwargs=read_kwargs,
            output_manager=output_manager
        )
        
        # Log completion if event log provided
        if event_log is not None:
            append_completion_event(
                event_log=event_log,
                dataset=dataset_name,
                image=image.name,
                status="completed",
                error_msg=""
            )
        
        click.echo(f"✓ Successfully processed {image.name}")
        sys.exit(0)
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        tb = traceback.format_exc()
        
        click.echo(f"✗ Failed to process {image.name}: {error_msg}", err=True)
        click.echo(f"Traceback:\n{tb}", err=True)
        
        # Log failure if event log provided
        if event_log is not None:
            try:
                append_completion_event(
                    event_log=event_log,
                    dataset=dataset_name,
                    image=image.name,
                    status="failed",
                    error_msg=error_msg
                )
            except Exception:
                pass  # Don't fail if logging fails
        
        sys.exit(1)


if __name__ == "__main__":
    main()
