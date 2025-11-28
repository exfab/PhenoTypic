"""
PhenoTypic CLI
==============

A command-line interface for executing PhenoTypic ImagePipelines on directories of images.
This script allows for parallel processing of images, saving both measurements and
visual quality control overlays.

Usage:
    python phenotypic-cli.py PIPELINE_JSON INPUT_DIR OUTPUT_DIR [OPTIONS]

Example:
    python phenotypic-cli.py my_pipeline.json ./raw_images ./results --n-jobs 4
"""

import sys
import click
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import Parallel, delayed
from typing import Optional, Type, Dict, Any, List

import phenotypic
from phenotypic import Image, GridImage, ImagePipeline
from phenotypic.tools.constants_ import IO

# Set non-interactive backend for headless execution
matplotlib.use('Agg')


def process_single_image(
        image_path: Path,
        meas_dir: Path,
        overlay_dir: Path,
        pipeline: ImagePipeline,
        image_cls: Type[Image],
        read_kwargs: Dict[str, Any]
) -> Optional[pd.DataFrame]:
    """
    Processes a single image of a microbe colony on solid media agar by applying an
    image processing pipeline, generating measurements, and creating a graphical
    overlay output. This function is highly versatile, allowing the user to control
    how images are read, analyzed, and stored based on provided arguments.

    Args:
        image_path (Path):
            Path to the image file representing the microbe colony on agar.
            Adjusting this variable changes which colony image is analyzed.
        meas_dir (Path):
            Directory where the measurement results (CSV) will be saved.
            The choice of directory affects the organization of analysis
            results and resultant data pipeline workflows.
        overlay_dir (Path):
            Directory for saving visual overlays. This allows inspection of
            how the overlay corresponds to the processed regions in the image.
            Choose a directory accessible to tools used for review.
        pipeline (ImagePipeline):
            A sequence of image processing steps applied to the input image.
            The pipeline heavily influences the analysis' sensitivity and accuracy
            in extracting colony features like size, shape, or density.
        image_cls (Type[Image]):
            Class responsible for reading and processing the input image. Changing
            this affects how the image format is handled (e.g., handling raw images
            produced in specific microscopy settings).
        read_kwargs (Dict[str, Any]):
            Parameters passed when reading the image (e.g., color modes, compression).
            Modifying these parameters tailors how images are interpreted and may
            change the fidelity of image data used in downstream analyses.

    Returns:
        Optional[pd.DataFrame]:
            A DataFrame containing microbiological measurements for the processed
            image, such as colony area, perimeter, and optical density. If processing
            fails, returns None. Adjustments in inputs or pipeline steps directly
            affect the resulting metrics.

    Raises:
        This function handles all internal exceptions and reports processing failures
        with user-friendly messages, allowing review of errors without interrupting a
        batch process.
    """
    try:
        # Create specific output path for this image's results
        # We use the image stem for naming
        image_stem = image_path.stem

        # Load image
        # We need to handle rawpy_params if needed, but for CLI we'll stick to basics for now
        image = image_cls.imread(image_path, **read_kwargs)

        # Execute pipeline
        # We use inplace=True to save memory, though pipeline operations might copy internally
        meas = pipeline.apply_and_measure(image, inplace=True)

        # Save measurements for this individual image
        meas_path = meas_dir/f"{image_stem}.csv"
        meas.to_csv(meas_path, index=False)

        # Generate and save overlay
        # We suppress the plot display since we are in a CLI
        fig, ax = image.show_overlay()
        overlay_path = overlay_dir/f"{image_stem}.png"
        fig.savefig(overlay_path, bbox_inches='tight')
        plt.close(fig)

        return meas

    except Exception as e:
        click.echo(f"Error processing {image_path.name}: {str(e)}", err=True)
        return None


@click.command()
@click.argument('pipeline_json', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument('output_dir', type=click.Path(path_type=Path))
@click.option('--image-type', type=click.Choice(['Image', 'GridImage'], case_sensitive=False),
              default='GridImage', help='Type of image object to instantiate.')
@click.option('--nrows', type=int, default=8, show_default=True,
              help='Number of rows for GridImage.')
@click.option('--ncols', type=int, default=12, show_default=True,
              help='Number of columns for GridImage.')
@click.option('--bit-depth', type=int, default=None,
              help='Bit depth of input images (8 or 16).')
@click.option('--n-jobs', type=int, default=-1, show_default=True,
              help='Number of parallel jobs. -1 uses all available cores.')
def main(pipeline_json: Path, input_dir: Path, output_dir: Path,
         image_type: str, nrows: int, ncols: int, bit_depth: Optional[int], n_jobs: int):
    """
    Execute a PhenoTypic pipeline on a directory of images.
    
    PIPELINE_JSON: Path to the exported pipeline configuration file.
    INPUT_DIR: Directory containing the images to process.
    OUTPUT_DIR: Directory where results (CSVs and overlays) will be saved.
    """

    # Setup
    output_dir.mkdir(parents=True, exist_ok=True)

    meas_dir = output_dir/'measurements'
    meas_dir.mkdir(parents=True, exist_ok=True)

    overlay_dir = output_dir/'overlays'
    overlay_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Loading pipeline from {pipeline_json}...")
    try:
        pipeline = ImagePipeline.from_json(pipeline_json)
    except Exception as e:
        click.echo(f"Failed to load pipeline: {e}", err=True)
        sys.exit(1)

    # Determine Image Class and Arguments
    if image_type == 'GridImage':
        image_cls = GridImage
        read_kwargs = {'nrows': nrows, 'ncols': ncols}
    else:
        image_cls = Image
        read_kwargs = {}

    if bit_depth:
        read_kwargs['bit_depth'] = bit_depth

    # Find images
    extensions = IO.ACCEPTED_FILE_EXTENSIONS + IO.RAW_FILE_EXTENSIONS
    image_paths = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]

    if not image_paths:
        click.echo(f"No valid images found in {input_dir}", err=True)
        sys.exit(1)

    click.echo(f"Found {len(image_paths)} images. Starting processing with {n_jobs} jobs...")

    # Parallel Execution
    # We use joblib to parallelize the processing
    results = Parallel(n_jobs=n_jobs)(
            delayed(process_single_image)(
                    path, output_dir, pipeline, image_cls, read_kwargs
            ) for path in image_paths
    )

    # Aggregate Results
    valid_results = [res for res in results if res is not None]

    if valid_results:
        click.echo(f"Successfully processed {len(valid_results)}/{len(image_paths)} images.")
        master_df = pd.concat(valid_results, axis=0, ignore_index=True)
        master_path = output_dir/"master_measurements.csv"
        master_df.to_csv(master_path, index=False)
        click.echo(f"Master measurements saved to {master_path}")
    else:
        click.echo("No images were successfully processed.", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
