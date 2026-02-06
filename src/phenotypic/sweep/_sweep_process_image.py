"""Core image processing for sweep CLI.

Provides the per-image processing function that runs all pipelines on a
single image, and a standalone Click sub-CLI for SLURM workers.
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING, Tuple

import click

if TYPE_CHECKING:
    from ._sweep_output import SweepOutputManager

logger = logging.getLogger(__name__)


def _run_single_pipeline(
    pipeline_json_str: str,
    pipeline_name: str,
    image_path: Path,
    image_type: Literal["Image", "GridImage"],
    read_kwargs: Dict[str, Any],
    output_manager: SweepOutputManager,
) -> Tuple[str, bool, str]:
    """Run a single pipeline on an image.

    Each joblib worker calls this function. The image is re-read from disk
    (OS page cache makes this fast) and the pipeline is deserialized from
    its JSON string to avoid complex pickling issues.

    Args:
        pipeline_json_str: Pipeline serialized as JSON string.
        pipeline_name: Human-readable pipeline name.
        image_path: Path to the input image.
        image_type: ``"Image"`` or ``"GridImage"``.
        read_kwargs: Kwargs for ``imread`` (nrows, ncols, etc.).
        output_manager: Sweep output manager instance.

    Returns:
        Tuple of ``(pipeline_name, success, error_message)``.
    """
    import matplotlib
    matplotlib.use("Agg")

    from phenotypic import Image, GridImage, ImagePipeline

    try:
        # Deserialize pipeline
        pipeline = ImagePipeline.from_json(json.loads(pipeline_json_str))

        # Load image from disk
        image_cls = GridImage if image_type == "GridImage" else Image
        rk = dict(read_kwargs)  # shallow copy
        detect_mode = rk.pop("detect_mode", "gray")
        image = image_cls.imread(image_path, **rk)

        if detect_mode != "gray":
            image.set_detect_mode(detect_mode)

        # Execute pipeline
        measurements = pipeline.apply_and_measure(image, inplace=True)

        # Save results
        image_stem = image_path.stem
        output_manager.save_measurements(measurements, pipeline_name, image_stem)
        output_manager.save_overlay(image, pipeline_name, image_stem)
        output_manager.save_image_layers(image, pipeline_name, image_stem)

        return (pipeline_name, True, "")

    except Exception as e:
        tb = traceback.format_exc()
        return (pipeline_name, False, tb)


def process_image_all_pipelines(
    image_path: Path,
    pipeline_json_strs: Dict[str, str],
    image_type: Literal["Image", "GridImage"],
    read_kwargs: Dict[str, Any],
    output_manager: SweepOutputManager,
    n_jobs: int = -1,
) -> List[Tuple[str, bool, str]]:
    """Process a single image through all pipelines in parallel.

    Each pipeline worker re-reads the image from disk. The OS page cache
    ensures that subsequent reads are served from memory.

    Args:
        image_path: Path to the input image.
        pipeline_json_strs: Mapping of pipeline_name to JSON string.
        image_type: ``"Image"`` or ``"GridImage"``.
        read_kwargs: Kwargs for ``imread``.
        output_manager: Sweep output manager instance.
        n_jobs: Number of parallel jobs (``-1`` = all cores).

    Returns:
        List of ``(pipeline_name, success, error_message)`` tuples.
    """
    from joblib import Parallel, delayed

    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_single_pipeline)(
            pipeline_json_str=json_str,
            pipeline_name=pipe_name,
            image_path=image_path,
            image_type=image_type,
            read_kwargs=read_kwargs,
            output_manager=output_manager,
        )
        for pipe_name, json_str in pipeline_json_strs.items()
    )

    return results


def process_image_all_pipelines_sequential(
    image_path: Path,
    pipeline_json_strs: Dict[str, str],
    image_type: Literal["Image", "GridImage"],
    read_kwargs: Dict[str, Any],
    output_manager: SweepOutputManager,
) -> List[Tuple[str, bool, str]]:
    """Process a single image through all pipelines sequentially.

    Used by SLURM tasks where each task processes one image and runs
    pipelines one at a time for predictable memory usage.

    Args:
        image_path: Path to the input image.
        pipeline_json_strs: Mapping of pipeline_name to JSON string.
        image_type: ``"Image"`` or ``"GridImage"``.
        read_kwargs: Kwargs for ``imread``.
        output_manager: Sweep output manager instance.

    Returns:
        List of ``(pipeline_name, success, error_message)`` tuples.
    """
    results = []
    for pipe_name, json_str in pipeline_json_strs.items():
        result = _run_single_pipeline(
            pipeline_json_str=json_str,
            pipeline_name=pipe_name,
            image_path=image_path,
            image_type=image_type,
            read_kwargs=read_kwargs,
            output_manager=output_manager,
        )
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Standalone Click CLI for SLURM workers
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--manifest", type=click.Path(exists=True, path_type=Path), required=True,
    help="Path to sweep manifest JSON.",
)
@click.option(
    "--image", type=click.Path(exists=True, path_type=Path), required=True,
    help="Path to input image.",
)
@click.option(
    "--output-dir", type=click.Path(path_type=Path), required=True,
    help="Base output directory.",
)
@click.option(
    "--image-type", type=click.Choice(["Image", "GridImage"]),
    default="GridImage", help="Image class to use.",
)
@click.option("--nrows", type=int, default=8, help="Grid rows (for GridImage).")
@click.option("--ncols", type=int, default=12, help="Grid columns (for GridImage).")
@click.option("--bit-depth", type=int, default=None, help="Bit depth (8 or 16).")
@click.option("--detect-mode", default="gray", help="Detection channel.")
@click.option("--save-rgb", is_flag=True)
@click.option("--save-gray", is_flag=True)
@click.option("--save-enh-gray", is_flag=True)
@click.option("--save-objmask", is_flag=True)
@click.option("--save-objmap", is_flag=True)
@click.option("--save-objmap-overlay", is_flag=True)
@click.option("--save-enh-gray-overlay", is_flag=True)
@click.option("--save-objmask-overlay", is_flag=True)
@click.option("--overlay-mode", type=click.Choice(["image", "figure"]), default="image")
@click.option("--overlay-alpha", type=float, default=0.3)
@click.option(
    "--event-log", type=click.Path(path_type=Path), default=None,
    help="Path to event log file.",
)
def main(
    manifest: Path,
    image: Path,
    output_dir: Path,
    image_type: str,
    nrows: int,
    ncols: int,
    bit_depth: Optional[int],
    detect_mode: str,
    save_rgb: bool,
    save_gray: bool,
    save_detect_mat: bool,
    save_objmask: bool,
    save_objmap: bool,
    save_objmap_overlay: bool,
    save_detect_mat_overlay: bool,
    save_objmask_overlay: bool,
    overlay_mode: str,
    overlay_alpha: float,
    event_log: Optional[Path],
):
    """Process a single image through all sweep pipelines (SLURM worker)."""
    import matplotlib
    matplotlib.use("Agg")

    from phenotypic.sweep import load_sweep_manifest
    from ._sweep_output import SweepOutputManager

    try:
        # Build read kwargs
        read_kwargs: Dict[str, Any] = {}
        if image_type == "GridImage":
            read_kwargs["nrows"] = nrows
            read_kwargs["ncols"] = ncols
        if bit_depth is not None:
            read_kwargs["bit_depth"] = bit_depth
        if detect_mode != "gray":
            read_kwargs["detect_mode"] = detect_mode

        # Load pipelines from manifest
        configs = load_sweep_manifest(manifest)
        pipeline_json_strs: Dict[str, str] = {}
        for _cfg_name, pipes in configs.items():
            for pipe_name, pipe in pipes.items():
                pipeline_json_strs[pipe_name] = pipe.to_json_str()

        # Create output manager
        save_layers = {
            "rgb": save_rgb,
            "gray": save_gray,
            "detect_mat": save_detect_mat,
            "objmask": save_objmask,
            "objmap": save_objmap,
            "objmap_overlay": save_objmap_overlay,
            "detect_mat_overlay": save_detect_mat_overlay,
            "objmask_overlay": save_objmask_overlay,
        }
        output_manager = SweepOutputManager(
            base_dir=output_dir,
            save_layers=save_layers,
            extensions={},  # use defaults
            overlay_mode=overlay_mode,
            overlay_alpha=overlay_alpha,
        )

        # Process all pipelines sequentially (SLURM tasks)
        click.echo(f"Processing {image.name} through {len(pipeline_json_strs)} pipelines...")
        image_type_literal: Literal["Image", "GridImage"] = image_type  # type: ignore[assignment]
        results = process_image_all_pipelines_sequential(
            image_path=image,
            pipeline_json_strs=pipeline_json_strs,
            image_type=image_type_literal,
            read_kwargs=read_kwargs,
            output_manager=output_manager,
        )

        # Log results
        succeeded = sum(1 for _, ok, _ in results if ok)
        failed = sum(1 for _, ok, _ in results if not ok)

        if event_log is not None:
            from phenotypic._cli._cli_update_state import append_completion_event

            error_msg = "" if failed == 0 else f"{failed}/{len(results)} pipelines failed"
            if failed == 0:
                append_completion_event(
                    event_log=event_log,
                    dataset="sweep",
                    image=image.name,
                    status="completed",
                    error_msg=error_msg,
                )
            else:
                append_completion_event(
                    event_log=event_log,
                    dataset="sweep",
                    image=image.name,
                    status="failed",
                    error_msg=error_msg,
                )

        click.echo(
            f"Finished {image.name}: {succeeded} succeeded, {failed} failed"
        )
        sys.exit(0 if failed == 0 else 1)

    except Exception as e:
        click.echo(f"Failed to process {image.name}: {e}", err=True)
        click.echo(traceback.format_exc(), err=True)

        if event_log is not None:
            try:
                from phenotypic._cli._cli_update_state import append_completion_event

                append_completion_event(
                    event_log=event_log,
                    dataset="sweep",
                    image=image.name,
                    status="failed",
                    error_msg=str(e),
                )
            except Exception:
                pass

        sys.exit(1)


if __name__ == "__main__":
    main()
