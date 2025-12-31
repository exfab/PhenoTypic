"""
PhenoTypic CLI
==============

A command-line interface for executing PhenoTypic ImagePipelines on images or
directories of images. This script allows for parallel processing of images (locally
or via SLURM with submitit), saving both measurements and visual quality control
overlays.

Usage:
    python -m phenotypic PIPELINE_JSON INPUT_PATH OUTPUT_DIR [OPTIONS]

MIGRATION NOTE (v2.0+):
    The --save-*-dir PATH flags have been replaced with --save-* boolean flags.
    Directories are now auto-generated as OUTPUT_DIR/<layer_name>/.

    Old: --save-rgb-dir ./custom/path/rgb
    New: --save-rgb  (saves to OUTPUT_DIR/rgb/)

Example:
    python -m phenotypic my_pipeline.json ./raw_images ./results --n-jobs 4
    python -m phenotypic my_pipeline.json ./example.jpg ./results --slurm --slurm-params slurm_partition=gpu --slurm-params mem_gb=32

Examples:
    # Process a single image locally
    uv run python -m phenotypic my_pipeline.json ./plate_A01.png ./results

    # Process a directory with local parallelism (all cores)
    uv run python -m phenotypic my_pipeline.json ./raw_images ./results --n-jobs -1

    # Process on SLURM with submitit, overriding partition and memory
    uv run python -m phenotypic my_pipeline.json ./raw_images ./results --slurm \
        --slurm-params slurm_partition=gpu --slurm-params mem_gb=32

    # Override grid shape for plate images
    uv run python -m phenotypic my_pipeline.json ./plates ./results --image-type GridImage \
        --nrows 16 --ncols 24

    # Save intermediate layers (RGB, grayscale, masks, objmaps)
    # Output automatically saved to OUTPUT_DIR/rgb/, OUTPUT_DIR/gray/, etc.
    uv run python -m phenotypic my_pipeline.json ./raw_images ./results \
        --save-rgb \
        --save-gray \
        --save-enh-gray \
        --save-objmask \
        --save-objmap \
        --save-objmap-rgb \
        --rgb-ext png --gray-ext tiff --objmask-ext png
"""

import ast
import sys
import click
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import Parallel, delayed
from typing import Optional, Type, Dict, Any, Iterable, List, Sequence

from phenotypic import Image, GridImage, ImagePipeline
from phenotypic.tools.constants_ import IO

# Set non-interactive backend for headless execution
matplotlib.use("Agg")


def process_single_image(
    image_path: Path,
    meas_dir: Path,
    overlay_dir: Path,
    pipeline: ImagePipeline,
    image_cls: Type[Image],
    read_kwargs: Dict[str, Any],
    save_rgb_dir: Optional[Path] = None,
    save_gray_dir: Optional[Path] = None,
    save_enh_gray_dir: Optional[Path] = None,
    save_objmask_dir: Optional[Path] = None,
    save_objmap_dir: Optional[Path] = None,
    save_objmap_label2rgb_dir: Optional[Path] = None,
    rgb_ext: str = ".tiff",
    gray_ext: str = ".tiff",
    enh_gray_ext: str = ".tiff",
    objmask_ext: str = ".png",
    objmap_ext: str = ".png",
    objmap_rgb_ext: str = ".png",
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
        meas_path = meas_dir / f"{image_stem}.csv"
        meas.to_csv(meas_path, index=False)

        # Generate and save overlay
        # We suppress the plot display since we are in a CLI
        fig, ax = image.show_overlay()
        overlay_path = overlay_dir / f"{image_stem}.png"
        fig.savefig(overlay_path, bbox_inches="tight")
        plt.close(fig)

        # Optional intermediate exports
        if save_rgb_dir:
            _maybe_imsave(image.rgb, save_rgb_dir / f"{image_stem}{rgb_ext}")
        if save_gray_dir:
            _maybe_imsave(image.gray, save_gray_dir / f"{image_stem}{gray_ext}")
        if save_enh_gray_dir:
            _maybe_imsave(
                image.enh_gray, save_enh_gray_dir / f"{image_stem}{enh_gray_ext}"
            )
        if save_objmask_dir:
            _maybe_imsave(image.objmask, save_objmask_dir / f"{image_stem}{objmask_ext}")
        if save_objmap_dir:
            _maybe_imsave(image.objmap, save_objmap_dir / f"{image_stem}{objmap_ext}")
        if save_objmap_label2rgb_dir:
            _maybe_imsave(
                image.objmap,
                save_objmap_label2rgb_dir / f"{image_stem}{objmap_rgb_ext}",
                use_label2rgb=True,
            )

        return meas

    except Exception as e:
        click.echo(f"Error processing {image_path.name}: {str(e)}", err=True)
        return None


def _collect_image_paths(input_path: Path, extensions: Iterable[str]) -> List[Path]:
    """Return all valid image paths from a directory or a single file."""
    valid_exts = {ext.lower() for ext in extensions}

    if input_path.is_dir():
        image_paths = [
            p
            for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in valid_exts
        ]
        if not image_paths:
            raise click.ClickException(f"No valid images found in {input_path}")
        return sorted(image_paths)

    if input_path.is_file():
        if input_path.suffix.lower() not in valid_exts:
            raise click.ClickException(
                f"{input_path} is not a supported image type. "
                f"Supported extensions: {', '.join(sorted(valid_exts))}"
            )
        return [input_path]

    raise click.ClickException(f"{input_path} is not a valid file or directory")


def _parse_slurm_params(slurm_params: Sequence[str]) -> Dict[str, Any]:
    """Parse KEY=VALUE pairs from --slurm-params into a dictionary."""
    parsed: Dict[str, Any] = {}
    for param in slurm_params:
        if "=" not in param:
            raise click.BadParameter(
                "--slurm-params must be provided as KEY=VALUE", param_hint="--slurm-params"
            )
        key, value = param.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            raise click.BadParameter(
                "SLURM parameter keys cannot be empty", param_hint="--slurm-params"
            )

        try:
            parsed_value = ast.literal_eval(value)
        except Exception:
            parsed_value = value

        parsed[key] = parsed_value

    return parsed


def _normalize_extension(ext: Optional[str], default_ext: str) -> str:
    """Normalize file extension to include leading dot and validate against allowed set."""
    if not ext:
        ext = default_ext
    ext = ext.lower()
    if not ext.startswith("."):
        ext = f".{ext}"
    allowed = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
    if ext not in allowed:
        raise click.BadParameter(
            f"Unsupported extension '{ext}'. Allowed: {', '.join(sorted(allowed))}."
        )
    return ext


def _maybe_imsave(accessor, filepath: Path, **kwargs) -> None:
    """Safely save an accessor array and emit a friendly message on failure."""
    try:
        if hasattr(accessor, "isempty") and accessor.isempty():
            click.echo(f"Skipping save for {filepath.name}: accessor is empty", err=True)
            return
        accessor.imsave(filepath=filepath, **kwargs)
    except Exception as e:
        click.echo(f"Failed to save {filepath}: {e}", err=True)


def _run_submitit_jobs(
    image_paths: List[Path],
    meas_dir: Path,
    overlay_dir: Path,
    pipeline: ImagePipeline,
    image_cls: Type[Image],
    read_kwargs: Dict[str, Any],
    slurm_params: Dict[str, Any],
    save_rgb_dir: Optional[Path] = None,
    save_gray_dir: Optional[Path] = None,
    save_enh_gray_dir: Optional[Path] = None,
    save_objmask_dir: Optional[Path] = None,
    save_objmap_dir: Optional[Path] = None,
    save_objmap_label2rgb_dir: Optional[Path] = None,
    rgb_ext: str = ".tiff",
    gray_ext: str = ".tiff",
    enh_gray_ext: str = ".tiff",
    objmask_ext: str = ".png",
    objmap_ext: str = ".png",
    objmap_rgb_ext: str = ".png",
) -> List[Optional[pd.DataFrame]]:
    """Submit image processing jobs to SLURM via submitit."""
    try:
        import submitit  # noqa: F401
    except ImportError as e:
        raise click.ClickException(
            "submitit backend requested but submitit is not installed. "
            "Install with: pip install phenotypic[cluster]"
        ) from e

    try:
        from phenotypic.util._pipeline_grid_search import _create_submitit_executor
    except Exception as e:
        raise click.ClickException(
            f"Failed to initialize submitit executor: {e}"
        ) from e

    executor = _create_submitit_executor(slurm_params=slurm_params)
    jobs = [
        executor.submit(
            process_single_image,
            path,
            meas_dir,
            overlay_dir,
            pipeline,
            image_cls,
            read_kwargs,
            save_rgb_dir,
            save_gray_dir,
            save_enh_gray_dir,
            save_objmask_dir,
            save_objmap_dir,
            save_objmap_label2rgb_dir,
            rgb_ext,
            gray_ext,
            enh_gray_ext,
            objmask_ext,
            objmap_ext,
            objmap_rgb_ext,
        )
        for path in image_paths
    ]

    click.echo(f"Submitted {len(jobs)} job(s) to SLURM. Waiting for completion...")

    results: List[Optional[pd.DataFrame]] = []
    for job in jobs:
        try:
            results.append(job.result())
        except Exception as e:
            job_id = getattr(job, "job_id", "unknown")
            click.echo(f"SLURM job {job_id} failed: {e}", err=True)
            results.append(None)

    return results


@click.command()
@click.argument(
    "pipeline_json", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument(
    "input_path", type=click.Path(exists=True, dir_okay=True, file_okay=True, path_type=Path)
)
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option(
    "--image-type",
    type=click.Choice(["Image", "GridImage"], case_sensitive=False),
    default="GridImage",
    help="Type of image object to instantiate.",
)
@click.option(
    "--nrows",
    type=int,
    default=8,
    show_default=True,
    help="Number of rows for GridImage.",
)
@click.option(
    "--ncols",
    type=int,
    default=12,
    show_default=True,
    help="Number of columns for GridImage.",
)
@click.option(
    "--bit-depth", type=int, default=None, help="Bit depth of input images (8 or 16)."
)
@click.option(
    "--n-jobs",
    type=int,
    default=-1,
    show_default=True,
    help="Number of parallel jobs. -1 uses all available cores.",
)
@click.option(
    "--slurm",
    is_flag=True,
    help="Use submitit to run jobs on a SLURM cluster instead of local joblib.",
)
@click.option(
    "--slurm-params",
    multiple=True,
    help="SLURM parameters as KEY=VALUE pairs (e.g., --slurm-params slurm_partition=gpu --slurm-params mem_gb=32).",
)
@click.option(
    "--save-rgb",
    is_flag=True,
    help="Save RGB images from Image.rgb to OUTPUT_DIR/rgb/. File extension controlled by --rgb-ext.",
)
@click.option(
    "--save-gray",
    is_flag=True,
    help="Save grayscale images from Image.gray to OUTPUT_DIR/gray/. File extension controlled by --gray-ext.",
)
@click.option(
    "--save-enh-gray",
    is_flag=True,
    help="Save enhanced grayscale images from Image.enh_gray to OUTPUT_DIR/enh_gray/. File extension controlled by --enh-gray-ext.",
)
@click.option(
    "--save-objmask",
    is_flag=True,
    help="Save binary object masks from Image.objmask to OUTPUT_DIR/objmask/. File extension controlled by --objmask-ext.",
)
@click.option(
    "--save-objmap",
    is_flag=True,
    help="Save label maps from Image.objmap to OUTPUT_DIR/objmap/. File extension controlled by --objmap-ext.",
)
@click.option(
    "--save-objmap-rgb",
    is_flag=True,
    help="Save label maps rendered with label2rgb to OUTPUT_DIR/objmap_rgb/. File extension controlled by --objmap-rgb-ext.",
)
@click.option(
    "--rgb-ext",
    default="tiff",
    show_default=True,
    help="File extension for Image.rgb saves (e.g., tiff, png, jpg). Default is TIFF.",
)
@click.option(
    "--gray-ext",
    default="tiff",
    show_default=True,
    help="File extension for Image.gray saves. Default is TIFF.",
)
@click.option(
    "--enh-gray-ext",
    default="tiff",
    show_default=True,
    help="File extension for Image.enh_gray saves. Default is TIFF.",
)
@click.option(
    "--objmask-ext",
    default="png",
    show_default=True,
    help="File extension for Image.objmask saves. Default is PNG.",
)
@click.option(
    "--objmap-ext",
    default="png",
    show_default=True,
    help="File extension for Image.objmap saves. Default is PNG.",
)
@click.option(
    "--objmap-rgb-ext",
    default="png",
    show_default=True,
    help="File extension for label2rgb Image.objmap saves. Default is PNG.",
)
def main(
    pipeline_json: Path,
    input_path: Path,
    output_dir: Path,
    image_type: str,
    nrows: int,
    ncols: int,
    bit_depth: Optional[int],
    n_jobs: int,
    slurm: bool,
    slurm_params: Sequence[str],
    save_rgb: bool,
    save_gray: bool,
    save_enh_gray: bool,
    save_objmask: bool,
    save_objmap: bool,
    save_objmap_rgb: bool,
    rgb_ext: str,
    gray_ext: str,
    enh_gray_ext: str,
    objmask_ext: str,
    objmap_ext: str,
    objmap_rgb_ext: str,
):
    """
    Execute a PhenoTypic pipeline on an image file or directory of images.

    PIPELINE_JSON: Path to the exported pipeline configuration file.
    INPUT_PATH: Single image file or directory containing images to process.
    OUTPUT_DIR: Directory where results (CSVs and overlays) will be saved.

    The CLI supports two execution modes:
        - Local (default): joblib-backed parallelism controlled by --n-jobs.
        - SLURM: enable with --slurm to submit jobs via submitit; customize with
          --slurm-params KEY=VALUE (e.g., slurm_partition=gpu mem_gb=32).

    Optional exports (saved to OUTPUT_DIR subdirectories; defaults: rgb/gray/enh_gray=tiff, masks/objmap=png):
        --save-rgb                    Save Image.rgb arrays to OUTPUT_DIR/rgb/.
        --save-gray                   Save Image.gray arrays to OUTPUT_DIR/gray/.
        --save-enh-gray               Save Image.enh_gray arrays to OUTPUT_DIR/enh_gray/.
        --save-objmask                Save Image.objmask to OUTPUT_DIR/objmask/.
        --save-objmap                 Save Image.objmap label maps to OUTPUT_DIR/objmap/.
        --save-objmap-rgb             Save Image.objmap rendered with label2rgb to OUTPUT_DIR/objmap_rgb/.
        --rgb-ext EXT                 File extension for RGB saves (default: tiff).
        --gray-ext EXT                File extension for grayscale saves (default: tiff).
        --enh-gray-ext EXT            File extension for enhanced gray saves (default: tiff).
        --objmask-ext EXT             File extension for objmask saves (default: png).
        --objmap-ext EXT              File extension for objmap saves (default: png).
        --objmap-rgb-ext EXT          File extension for label2rgb objmap saves (default: png).

    Examples:
        uv run python -m phenotypic my_pipeline.json ./raw_images ./results --n-jobs 8
        uv run python -m phenotypic my_pipeline.json ./plate.png ./results --image-type Image
        uv run python -m phenotypic my_pipeline.json ./raw_images ./results --save-rgb --save-gray --rgb-ext png
        uv run python -m phenotypic my_pipeline.json ./raw_images ./results --slurm \
            --slurm-params slurm_partition=gpu --slurm-params mem_gb=32

    SLURM + submitit quick reference (use with --slurm-params KEY=VALUE):
        Mandatory runtime params (always set):
            slurm_partition=<partition>          # SBATCH --partition
            slurm_account=<account>              # SBATCH --account
            slurm_time=HH:MM:SS                  # SBATCH --time
            slurm_mem=<size> OR slurm_mem_per_cpu=<size>  # SBATCH --mem / --mem-per-cpu (pick one)
            slurm_cpus_per_task=<N>              # SBATCH --cpus-per-task

        Strongly recommended defaults:
            slurm_job_name=<name>                # SBATCH --job-name
            slurm_output=logs/%j.out             # SBATCH --output
            slurm_error=logs/%j.err              # SBATCH --error
            slurm_requeue=True                   # SBATCH --requeue
            slurm_signal_delay_s=120             # SBATCH --signal=USR1@120

        Conditional but high-value:
            slurm_gpus_per_node=<N> or slurm_gres=\"gpu:<N>\"   # SBATCH --gpus-per-node / --gres
            nodes=<N> and slurm_ntasks=<N>                      # SBATCH --nodes / --ntasks
            # Arrays: use executor.map_array(...)

        Guardrails:
            - Always set partition, account, time, memory, cpus
            - Never set both slurm_mem and slurm_mem_per_cpu
            - Do not assume implicit GPUs; request explicitly if needed
            - Prefer conservative resource requests

        Canonical minimal baseline:
            #SBATCH --partition=<partition>
            #SBATCH --account=<account>
            #SBATCH --time=HH:MM:SS
            #SBATCH --mem=<size>
            #SBATCH --cpus-per-task=<N>
            #SBATCH --job-name=<name>
            #SBATCH --output=logs/%j.out
            #SBATCH --error=logs/%j.err
            #SBATCH --requeue
            #SBATCH --signal=USR1@120

            executor.update_parameters(
                slurm_partition="<partition>",
                slurm_account="<account>",
                slurm_time="HH:MM:SS",
                slurm_mem="<size>",
                slurm_cpus_per_task=<N>,
                slurm_job_name="<name>",
                slurm_output="logs/%j.out",
                slurm_error="logs/%j.err",
                slurm_requeue=True,
                slurm_signal_delay_s=120,
            )
    """

    # Setup
    output_dir.mkdir(parents=True, exist_ok=True)

    meas_dir = output_dir / "measurements"
    meas_dir.mkdir(parents=True, exist_ok=True)

    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    # Compute optional directory paths based on flags
    save_rgb_dir = output_dir / "rgb" if save_rgb else None
    save_gray_dir = output_dir / "gray" if save_gray else None
    save_enh_gray_dir = output_dir / "enh_gray" if save_enh_gray else None
    save_objmask_dir = output_dir / "objmask" if save_objmask else None
    save_objmap_dir = output_dir / "objmap" if save_objmap else None
    save_objmap_rgb_dir = output_dir / "objmap_rgb" if save_objmap_rgb else None

    for optional_dir in [
        save_rgb_dir,
        save_gray_dir,
        save_enh_gray_dir,
        save_objmask_dir,
        save_objmap_dir,
        save_objmap_rgb_dir,
    ]:
        if optional_dir:
            optional_dir.mkdir(parents=True, exist_ok=True)

    try:
        rgb_ext = _normalize_extension(rgb_ext, ".tiff")
        gray_ext = _normalize_extension(gray_ext, ".tiff")
        enh_gray_ext = _normalize_extension(enh_gray_ext, ".tiff")
        objmask_ext = _normalize_extension(objmask_ext, ".png")
        objmap_ext = _normalize_extension(objmap_ext, ".png")
        objmap_rgb_ext = _normalize_extension(objmap_rgb_ext, ".png")
    except click.BadParameter as e:
        click.echo(str(e), err=True)
        sys.exit(1)

    click.echo(f"Loading pipeline from {pipeline_json}...")
    try:
        pipeline = ImagePipeline.from_json(pipeline_json)
    except Exception as e:
        click.echo(f"Failed to load pipeline: {e}", err=True)
        sys.exit(1)

    # Determine Image Class and Arguments
    if image_type == "GridImage":
        image_cls = GridImage
        read_kwargs = {"nrows": nrows, "ncols": ncols}
    else:
        image_cls = Image
        read_kwargs = {}

    if bit_depth:
        read_kwargs["bit_depth"] = bit_depth

    if slurm_params and not slurm:
        raise click.ClickException("--slurm-params requires --slurm to be set.")

    extensions = IO.ACCEPTED_FILE_EXTENSIONS + IO.RAW_FILE_EXTENSIONS
    try:
        image_paths = _collect_image_paths(input_path, extensions)
    except click.ClickException as e:
        click.echo(str(e), err=True)
        sys.exit(1)

    if len(image_paths) == 1:
        click.echo(f"Processing single image: {image_paths[0].name}")
    else:
        backend_desc = "submitit (SLURM)" if slurm else f"joblib with n_jobs={n_jobs}"
        click.echo(
            f"Found {len(image_paths)} images under {input_path}. "
            f"Starting processing using {backend_desc}..."
        )

    if slurm:
        slurm_kwargs = _parse_slurm_params(slurm_params)
        results = _run_submitit_jobs(
            image_paths,
            meas_dir,
            overlay_dir,
            pipeline,
            image_cls,
            read_kwargs,
            slurm_kwargs,
            save_rgb_dir,
            save_gray_dir,
            save_enh_gray_dir,
            save_objmask_dir,
            save_objmap_dir,
            save_objmap_rgb_dir,
            rgb_ext,
            gray_ext,
            enh_gray_ext,
            objmask_ext,
            objmap_ext,
            objmap_rgb_ext,
        )
    elif len(image_paths) == 1:
        results = [
            process_single_image(
                image_paths[0],
                meas_dir,
                overlay_dir,
                pipeline,
                image_cls,
                read_kwargs,
                save_rgb_dir,
                save_gray_dir,
                save_enh_gray_dir,
                save_objmask_dir,
                save_objmap_dir,
                save_objmap_rgb_dir,
                rgb_ext,
                gray_ext,
                enh_gray_ext,
                objmask_ext,
                objmap_ext,
                objmap_rgb_ext,
            )
        ]
    else:
        # Parallel Execution with joblib
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_single_image)(
                path,
                meas_dir,
                overlay_dir,
                pipeline,
                image_cls,
                read_kwargs,
                save_rgb_dir,
                save_gray_dir,
                save_enh_gray_dir,
                save_objmask_dir,
                save_objmap_dir,
                save_objmap_rgb_dir,
                rgb_ext,
                gray_ext,
                enh_gray_ext,
                objmask_ext,
                objmap_ext,
                objmap_rgb_ext,
            )
            for path in image_paths
        )

    # Aggregate Results
    valid_results = [res for res in results if res is not None]

    if valid_results:
        click.echo(
            f"Successfully processed {len(valid_results)}/{len(image_paths)} images."
        )
        master_df = pd.concat(valid_results, axis=0, ignore_index=True)
        master_path = output_dir / "master_measurements.csv"
        master_df.to_csv(master_path, index=False)
        click.echo(f"Master measurements saved to {master_path}")
    else:
        click.echo("No images were successfully processed.", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
