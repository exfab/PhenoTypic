"""Output file organization for the sweep CLI.

Manages the output directory structure, per-image HDF5 saves, per-image
CSV saves, and failure logging.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from phenotypic.tools_ import DIR_RESULTS

import pandas as pd

if TYPE_CHECKING:
    from phenotypic._core._image import Image

logger = logging.getLogger(__name__)


def clear_previous_run(output_dir: Path) -> bool:
    """Delete existing sweep results from the output directory.

    Detects a previous run by checking if ``output_dir/results`` exists
    and is non-empty. If so, deletes all contents of ``output_dir``.

    Args:
        output_dir: Base sweep output directory.

    Returns:
        ``True`` if clearing occurred, ``False`` otherwise.
    """
    output_dir = Path(output_dir)
    results_dir = output_dir / DIR_RESULTS

    if not results_dir.is_dir() or not any(results_dir.iterdir()):
        return False

    for item in output_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    logger.info(f"Cleared previous sweep results from {output_dir}")
    return True


class SweepOutputManager:
    """Manages output file creation and organization for sweep processing.

    Organizes output in an **image-first** layout::

        results/<image_stem>/<pipeline_name>/<image_stem>.h5
        results/<image_stem>/<pipeline_name>/<image_stem>.csv

    Args:
        base_dir: Base output directory for all results.
    """

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / DIR_RESULTS
        self.logs_dir = self.base_dir / "logs"
        self.failures_dir = self.logs_dir / "failures"

    def create_structure(self) -> None:
        """Create base output directory structure.

        Per-image and per-pipeline directories are created on-demand
        during saving.
        """
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.failures_dir.mkdir(exist_ok=True)
        (self.logs_dir / "slurm").mkdir(exist_ok=True)

    def write_failure_log(
        self,
        image_path: Path,
        pipeline_name: str,
        traceback_str: str,
        pipeline_json_str: str,
    ) -> Optional[Path]:
        """Write a detailed failure log for a single pipeline run.

        Args:
            image_path: Path to the input image that failed.
            pipeline_name: Name of the pipeline that failed.
            traceback_str: Full traceback string from the exception.
            pipeline_json_str: Pipeline JSON config for reproducibility.

        Returns:
            Path to the written log file, or ``None`` if writing failed.
        """
        try:
            # Ensure directory exists (SLURM workers skip create_structure)
            self.failures_dir.mkdir(parents=True, exist_ok=True)

            safe_name = pipeline_name.replace("/", "_").replace("\\", "_")
            filename = f"{image_path.stem}__{safe_name}.log"
            log_path = self.failures_dir / filename

            timestamp = datetime.now(tz=timezone.utc).isoformat()
            content = (
                f"Timestamp: {timestamp}\n"
                f"Image:     {image_path}\n"
                f"Pipeline:  {pipeline_name}\n"
                f"\n{'=' * 60}\n"
                f"TRACEBACK\n"
                f"{'=' * 60}\n"
                f"{traceback_str}\n"
                f"{'=' * 60}\n"
                f"PIPELINE CONFIG (JSON)\n"
                f"{'=' * 60}\n"
                f"{pipeline_json_str}\n"
            )
            log_path.write_text(content)
            return log_path

        except Exception as exc:
            logger.warning(
                f"Could not write failure log for "
                f"{pipeline_name}/{image_path.stem}: {exc}"
            )
            return None

    def _pipeline_dir(
        self, image_stem: str, pipeline_name: str
    ) -> Path:
        """Return ``results/<image_stem>/<pipeline_name>/``, creating it."""
        d = self.results_dir / image_stem / pipeline_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_measurements(
        self,
        measurements: pd.DataFrame,
        pipeline_name: str,
        image_stem: str,
    ) -> Optional[Path]:
        """Save measurements CSV for a single image under a pipeline.

        Args:
            measurements: DataFrame with measurement data.
            pipeline_name: Pipeline name.
            image_stem: Image filename without extension.

        Returns:
            Path where measurements were saved, or ``None`` if saving failed.
        """
        try:
            if "Metadata_Pipeline" not in measurements.columns:
                measurements = measurements.copy()
                measurements.insert(0, "Metadata_Pipeline", pipeline_name)

            output_path = (
                self._pipeline_dir(image_stem, pipeline_name)
                / f"{image_stem}.csv"
            )
            measurements.to_csv(output_path, index=False)
            logger.info(
                f"Saved measurements for {pipeline_name}/{image_stem}"
            )
            return output_path
        except Exception as e:
            logger.warning(
                f"Failed to save measurements for "
                f"{pipeline_name}/{image_stem}: {type(e).__name__}: {e}"
            )
            return None

    def save_image_hdf5(
        self,
        image: "Image",
        pipeline_name: str,
        image_stem: str,
    ) -> Optional[Path]:
        """Save processed image as HDF5 using ``Image.save2hdf5()``.

        Args:
            image: Image object with processing results.
            pipeline_name: Pipeline name.
            image_stem: Image filename without extension.

        Returns:
            Path where HDF5 was saved, or ``None`` if saving failed.
        """
        try:
            output_path = (
                self._pipeline_dir(image_stem, pipeline_name)
                / f"{image_stem}.h5"
            )
            image.save2hdf5(output_path)
            logger.info(
                f"Saved HDF5 for {pipeline_name}/{image_stem}"
            )
            return output_path
        except Exception as e:
            logger.warning(
                f"Failed to save HDF5 for "
                f"{pipeline_name}/{image_stem}: {type(e).__name__}: {e}"
            )
            return None
