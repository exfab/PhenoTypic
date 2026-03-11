"""SLURM sentinel job for monitoring pipeline progress.

This Click command runs as a SLURM job, periodically rebuilding the progress
manifest and resubmitting itself if work remains.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path

import click

from ._cli_manifest_builder import build_manifest

logger = logging.getLogger(__name__)


@click.command()
@click.option("--output-dir", type=Path, required=True)
@click.option("--progress-dir", type=Path, required=True)
@click.option(
    "--interval", type=int, default=60, help="Seconds between manifest rebuilds"
)
@click.option(
    "--max-runtime",
    type=int,
    default=1800,
    help="Max runtime in seconds (default 30 min)",
)
@click.option(
    "--sentinel-script",
    type=Path,
    default=None,
    help="Path to this sentinel's script for resubmission",
)
@click.option("--slurm-partition", type=str, default="batch")
def sentinel_main(
    output_dir: Path,
    progress_dir: Path,
    interval: int,
    max_runtime: int,
    sentinel_script: Path | None,
    slurm_partition: str,
) -> None:
    """Monitor SLURM pipeline progress and resubmit sentinel if work remains.

    Args:
        output_dir: Base output directory containing results.
        progress_dir: Directory for progress files and metadata.
        interval: Seconds between manifest rebuilds.
        max_runtime: Maximum runtime in seconds before the sentinel
            exits and optionally resubmits itself.
        sentinel_script: Path to the sentinel SLURM script for
            resubmission. If ``None``, no resubmission occurs.
        slurm_partition: SLURM partition for resubmitted sentinel jobs.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [sentinel] %(levelname)s %(message)s",
    )

    metadata_path = progress_dir / "job_metadata.json"
    if not metadata_path.exists():
        logger.error("job_metadata.json not found at %s", metadata_path)
        raise SystemExit(1)

    with open(metadata_path) as fh:
        job_metadata = json.load(fh)

    start_time = job_metadata["start_time"]
    datasets_info = job_metadata["datasets"]
    chunk_scripts = job_metadata.get("chunk_scripts", [])
    chunk_job_ids = job_metadata.get("chunk_job_ids", {})
    image_task_mapping = job_metadata.get("image_task_mapping", {})
    input_path = job_metadata.get("input_path")

    # Build {dataset_name: total_images} mapping
    datasets_totals: dict[str, int] = {
        name: info["total"] for name, info in datasets_info.items()
    }

    logger.info(
        "Sentinel started — datasets=%s, interval=%ds, max_runtime=%ds",
        list(datasets_totals.keys()),
        interval,
        max_runtime,
    )

    sentinel_start = time.monotonic()
    is_complete = False

    while True:
        # Rebuild the progress manifest
        build_manifest(
            output_dir=output_dir,
            progress_dir=progress_dir,
            datasets=datasets_totals,
            execution_mode="slurm",
            start_time=start_time,
            slurm_job_ids=chunk_job_ids,
            chunk_scripts=chunk_scripts,
            input_path=input_path,
        )

        # Check completion status from the freshly-written manifest
        manifest_path = progress_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as fh:
                manifest = json.load(fh)
            is_complete = manifest.get("is_complete", False)

        if is_complete:
            logger.info("All tasks complete — aggregating measurements.")
            try:
                from ._cli_output_manager import aggregate_measurements

                _metadata_csv_str = job_metadata.get("metadata_csv")
                master_path = aggregate_measurements(
                    output_dir=output_dir,
                    dataset_names=list(datasets_totals.keys()),
                    include_dataset_column=job_metadata.get(
                        "include_dataset_column", True
                    ),
                    metadata_csv=Path(_metadata_csv_str) if _metadata_csv_str else None,
                )
                if master_path:
                    logger.info("Master CSV written: %s", master_path)
                else:
                    logger.warning("No measurements found for aggregation.")
            except Exception:
                logger.error("Failed to aggregate master CSV", exc_info=True)

            logger.info("Sentinel exiting.")
            return

        # Check elapsed time before sleeping
        elapsed = time.monotonic() - sentinel_start
        if elapsed + interval > max_runtime:
            logger.info(
                "Approaching max_runtime (%.0fs elapsed, limit %ds) — exiting loop.",
                elapsed,
                max_runtime,
            )
            break

        time.sleep(interval)

    # If work remains and a sentinel script is available, resubmit
    if not is_complete and sentinel_script is not None:
        logger.info("Resubmitting sentinel via %s", sentinel_script)
        result = subprocess.run(
            ["sbatch", "--parsable", str(sentinel_script)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            new_job_id = result.stdout.strip()
            logger.info("Sentinel resubmitted as SLURM job %s", new_job_id)
        else:
            logger.error(
                "Failed to resubmit sentinel: %s", result.stderr.strip()
            )
    elif not is_complete:
        logger.warning(
            "Work remains but no sentinel_script provided — cannot resubmit."
        )


if __name__ == "__main__":
    sentinel_main()
