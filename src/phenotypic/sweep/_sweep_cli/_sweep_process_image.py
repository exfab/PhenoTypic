"""Core image processing for sweep CLI.

Provides the per-image processing function that runs all pipelines on a
single image, and a standalone Click sub-CLI for SLURM workers.
"""

from __future__ import annotations

import logging
import sys
import time
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
        pipeline = ImagePipeline.from_json(pipeline_json_str)

        # Load image from disk
        image_cls = GridImage if image_type == "GridImage" else Image
        rk = dict(read_kwargs)  # shallow copy
        detect_mode = rk.pop("detect_mode", "gray")
        image = image_cls.imread(image_path, **rk)

        if detect_mode != "gray":
            image.set_detect_mode(detect_mode)

        # Execute pipeline
        if pipeline._meas:
            measurements = pipeline.apply_and_measure(image, inplace=True)
        else:
            pipeline.apply(image, inplace=True)
            measurements = None

        # Save results — each save is independent and non-fatal
        image_stem = image_path.stem
        if measurements is not None:
            output_manager.save_measurements(measurements, pipeline_name, image_stem)
        output_manager.save_image_hdf5(image, pipeline_name, image_stem)

        logger.info(
            f"Completed {pipeline_name} on {image_path.name}"
        )
        return (pipeline_name, True, "")

    except Exception as e:
        tb = traceback.format_exc()
        # Always print to stderr so SLURM logs capture the error
        print(
            f"[PIPELINE FAIL] {pipeline_name} on {image_path.name}: "
            f"{type(e).__name__}: {e}",
            file=sys.stderr,
            flush=True,
        )
        try:
            output_manager.write_failure_log(
                image_path=image_path,
                pipeline_name=pipeline_name,
                traceback_str=tb,
                pipeline_json_str=pipeline_json_str,
            )
        except Exception as log_exc:
            print(
                f"[WARN] Could not write failure log for {pipeline_name}: "
                f"{type(log_exc).__name__}: {log_exc}",
                file=sys.stderr,
                flush=True,
            )
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
    total = len(pipeline_json_strs)
    for idx, (pipe_name, json_str) in enumerate(pipeline_json_strs.items(), 1):
        click.echo(f"[{idx}/{total}] {pipe_name}... ", nl=False)
        t0 = time.monotonic()
        result = _run_single_pipeline(
            pipeline_json_str=json_str,
            pipeline_name=pipe_name,
            image_path=image_path,
            image_type=image_type,
            read_kwargs=read_kwargs,
            output_manager=output_manager,
        )
        elapsed = time.monotonic() - t0
        status = "OK" if result[1] else "FAILED"
        click.echo(f"{status} ({elapsed:.1f}s)")
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
@click.option(
    "--event-log", type=click.Path(path_type=Path), default=None,
    help="Path to event log file.",
)
@click.option(
    "--pipeline-name", type=str, default=None,
    help="Run only this pipeline (for per-pipeline SLURM tasks).",
)
def sweep_worker_cli(
    manifest: Path,
    image: Path,
    output_dir: Path,
    image_type: str,
    nrows: int,
    ncols: int,
    bit_depth: Optional[int],
    detect_mode: str,
    event_log: Optional[Path],
    pipeline_name: Optional[str],
):
    """Process a single image through all sweep pipelines (SLURM worker)."""
    import matplotlib
    matplotlib.use("Agg")

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
        pipeline_json_strs: Dict[str, str] = {}
        if pipeline_name is not None:
            # SLURM per-pipeline mode: only deserialize the one we need
            from phenotypic.sweep._generate_sweep import (
                load_single_pipeline_from_manifest,
            )
            try:
                json_str = load_single_pipeline_from_manifest(
                    manifest, pipeline_name,
                )
            except KeyError as exc:
                click.echo(f"ERROR: {exc}", err=True)
                sys.exit(1)
            pipeline_json_strs = {pipeline_name: json_str}
        else:
            # All-pipelines mode: extract JSON strings without instantiation
            import json as _json
            raw = _json.loads(Path(manifest).read_text())
            for cfg_data in raw.get("configs", {}).values():
                for pipe_name, pipe_dict in cfg_data.get("pipelines", {}).items():
                    pipeline_json_strs[pipe_name] = _json.dumps(pipe_dict)

        # Create output manager
        output_manager = SweepOutputManager(base_dir=output_dir)

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
            from ._sweep_progress_dashboard import maybe_regenerate_dashboard

            # Use composite ID for per-pipeline SLURM tasks
            event_image_id = (
                f"{image.name}::{pipeline_name}"
                if pipeline_name is not None
                else image.name
            )
            error_msg = "" if failed == 0 else f"{failed}/{len(results)} pipelines failed"
            if failed == 0:
                append_completion_event(
                    event_log=event_log,
                    dataset="sweep",
                    image=event_image_id,
                    status="completed",
                    error_msg=error_msg,
                )
            else:
                append_completion_event(
                    event_log=event_log,
                    dataset="sweep",
                    image=event_image_id,
                    status="failed",
                    error_msg=error_msg,
                )

            maybe_regenerate_dashboard(output_dir, event_log)

        click.echo(
            f"Finished {image.name}: {succeeded} succeeded, {failed} failed"
        )
        if failed > 0:
            # Print first failure traceback to SLURM log for diagnostics
            failure_results = [(n, tb) for n, ok, tb in results if not ok]
            n_show = min(3, len(failure_results))
            click.echo(
                f"\n--- First {n_show} of {len(failure_results)} failure tracebacks ---",
                err=True,
            )
            for pipe_name, tb in failure_results[:n_show]:
                click.echo(f"\n[{pipe_name}]\n{tb}", err=True)
            click.echo(
                f"Detailed failure logs: {output_manager.failures_dir}"
            )
        sys.exit(0 if failed == 0 else 1)

    except Exception as e:
        click.echo(f"Failed to process {image.name}: {e}", err=True)
        click.echo(traceback.format_exc(), err=True)

        if event_log is not None:
            try:
                from phenotypic._cli._cli_update_state import append_completion_event
                from ._sweep_progress_dashboard import (
                    maybe_regenerate_dashboard,
                )

                event_image_id = (
                    f"{image.name}::{pipeline_name}"
                    if pipeline_name is not None
                    else image.name
                )
                append_completion_event(
                    event_log=event_log,
                    dataset="sweep",
                    image=event_image_id,
                    status="failed",
                    error_msg=str(e),
                )
                maybe_regenerate_dashboard(output_dir, event_log)
            except Exception:
                pass

        sys.exit(1)


if __name__ == "__main__":
    sweep_worker_cli()
