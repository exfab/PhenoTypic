"""Output file organization for the sweep CLI.

Manages the output directory structure, per-image CSV saves, overlay saves,
and aggregation of measurements across pipelines.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

import pandas as pd
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from phenotypic import Image

logger = logging.getLogger(__name__)


class SweepOutputManager:
    """Manages output file creation and organization for sweep processing.

    Unlike the main CLI's :class:`OutputManager` which keys output by dataset
    name, this class keys output by **pipeline name** since the sweep CLI
    applies multiple pipeline configurations to a flat set of images.

    Args:
        base_dir: Base output directory for all results.
        save_layers: Which layers to save ``{"rgb": True, ...}``.
        extensions: File extensions for each layer ``{"rgb": ".tiff", ...}``.
        overlay_mode: ``"image"`` for full-resolution, ``"figure"`` for matplotlib.
        overlay_alpha: Alpha transparency for label overlay (0.0-1.0).
    """

    def __init__(
        self,
        base_dir: Path,
        save_layers: Dict[str, bool],
        extensions: Dict[str, str],
        overlay_mode: str = "image",
        overlay_alpha: float = 0.3,
    ):
        self.base_dir = Path(base_dir)
        self.save_layers = save_layers
        self.extensions = extensions
        self.overlay_mode = overlay_mode
        self.overlay_alpha = overlay_alpha

        self.results_dir = self.base_dir / "results"
        self.logs_dir = self.base_dir / "logs"
        self.failures_dir = self.logs_dir / "failures"

    def create_structure(self, pipeline_names: List[str]) -> None:
        """Create complete output directory structure.

        Args:
            pipeline_names: Pipeline names to create directories for.
        """
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.failures_dir.mkdir(exist_ok=True)
        (self.logs_dir / "slurm").mkdir(exist_ok=True)

        for pipe_name in pipeline_names:
            pipe_dir = self.results_dir / pipe_name
            pipe_dir.mkdir(exist_ok=True)
            (pipe_dir / "measurements").mkdir(exist_ok=True)
            (pipe_dir / "overlays").mkdir(exist_ok=True)
            for layer_name, enabled in self.save_layers.items():
                if enabled:
                    (pipe_dir / layer_name).mkdir(exist_ok=True)

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

    def get_output_path(
        self,
        pipeline_name: str,
        layer: str,
        image_stem: str,
    ) -> Path:
        """Get the output path for a specific file.

        Args:
            pipeline_name: Pipeline name (subdirectory under results/).
            layer: Layer type (``"measurements"``, ``"overlays"``, etc.).
            image_stem: Image filename without extension.

        Returns:
            Complete output path for the file.
        """
        if layer == "measurements":
            ext = ".csv"
        elif layer == "overlays":
            ext = ".png"
        else:
            if not self.save_layers.get(layer):
                raise ValueError(f"Layer '{layer}' is not enabled")
            ext = self.extensions.get(layer, ".png")

        return self.results_dir / pipeline_name / layer / f"{image_stem}{ext}"

    def save_measurements(
        self,
        measurements: pd.DataFrame,
        pipeline_name: str,
        image_stem: str,
    ) -> Path:
        """Save measurements CSV for a single image under a pipeline.

        Args:
            measurements: DataFrame with measurement data.
            pipeline_name: Pipeline name.
            image_stem: Image filename without extension.

        Returns:
            Path where measurements were saved.
        """
        if "Metadata_Pipeline" not in measurements.columns:
            measurements = measurements.copy()
            measurements.insert(0, "Metadata_Pipeline", pipeline_name)

        output_path = self.get_output_path(pipeline_name, "measurements", image_stem)
        measurements.to_csv(output_path, index=False)
        return output_path

    def save_overlay(
        self,
        image: "Image",
        pipeline_name: str,
        image_stem: str,
    ) -> Path:
        """Save overlay visualization for a single image.

        Args:
            image: Image object with processing results.
            pipeline_name: Pipeline name.
            image_stem: Image filename without extension.

        Returns:
            Path where overlay was saved.
        """
        output_path = self.get_output_path(pipeline_name, "overlays", image_stem)

        if self.overlay_mode == "figure":
            fig, ax = image.plot.overlay()
            fig.savefig(output_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
        else:
            if not image.rgb.isempty():
                image.rgb.save_overlay(
                    filepath=output_path,
                    overlay_alpha=self.overlay_alpha,
                )
            else:
                image.gray.save_overlay(
                    filepath=output_path,
                    overlay_alpha=self.overlay_alpha,
                )

        return output_path

    def _save_layer_safely(
        self,
        layer_name: str,
        image: "Image",
        pipeline_name: str,
        image_stem: str,
        save_func: Callable[[Path], None],
    ) -> Optional[Path]:
        """Safely save an image layer with error logging."""
        try:
            path = self.get_output_path(pipeline_name, layer_name, image_stem)
            save_func(path)
            return path
        except Exception as e:
            logger.warning(
                f"Failed to save {layer_name} for {pipeline_name}/{image_stem}: "
                f"{type(e).__name__}: {e}"
            )
            return None

    def save_image_layers(
        self,
        image: "Image",
        pipeline_name: str,
        image_stem: str,
    ) -> Dict[str, Path]:
        """Save all requested image layers.

        Args:
            image: Image object with processing results.
            pipeline_name: Pipeline name.
            image_stem: Image filename without extension.

        Returns:
            Dictionary mapping layer names to saved paths (successful saves only).
        """
        saved_paths: Dict[str, Path] = {}

        if self.save_layers.get("rgb") and not image.rgb.isempty():
            path = self._save_layer_safely(
                "rgb", image, pipeline_name, image_stem,
                lambda p: image.rgb.imsave(filepath=p),
            )
            if path:
                saved_paths["rgb"] = path

        if self.save_layers.get("gray") and not image.gray.isempty():
            path = self._save_layer_safely(
                "gray", image, pipeline_name, image_stem,
                lambda p: image.gray.imsave(filepath=p),
            )
            if path:
                saved_paths["gray"] = path

        if self.save_layers.get("detect_mat") and not image.detect_mat.isempty():
            path = self._save_layer_safely(
                "detect_mat", image, pipeline_name, image_stem,
                lambda p: image.detect_mat.imsave(filepath=p),
            )
            if path:
                saved_paths["detect_mat"] = path

        if self.save_layers.get("objmask") and not image.objmask.isempty():
            path = self._save_layer_safely(
                "objmask", image, pipeline_name, image_stem,
                lambda p: image.objmask.imsave(filepath=p),
            )
            if path:
                saved_paths["objmask"] = path

        if self.save_layers.get("objmap") and not image.objmap.isempty():
            path = self._save_layer_safely(
                "objmap", image, pipeline_name, image_stem,
                lambda p: image.objmap.imsave(filepath=p),
            )
            if path:
                saved_paths["objmap"] = path

        if (
            self.save_layers.get("objmap_overlay")
            and not image.objmap.isempty()
        ):
            path = self._save_layer_safely(
                "objmap_overlay", image, pipeline_name, image_stem,
                lambda p: image.objmap.imsave(filepath=p, use_label2rgb=True),
            )
            if path:
                saved_paths["objmap_overlay"] = path

        if (
            self.save_layers.get("detect_mat_overlay")
            and not image.detect_mat.isempty()
        ):
            path = self._save_layer_safely(
                "detect_mat_overlay", image, pipeline_name, image_stem,
                lambda p: image.detect_mat.save_overlay(
                    filepath=p, overlay_alpha=self.overlay_alpha
                ),
            )
            if path:
                saved_paths["detect_mat_overlay"] = path

        if (
            self.save_layers.get("objmask_overlay")
            and not image.objmask.isempty()
        ):
            path = self._save_layer_safely(
                "objmask_overlay", image, pipeline_name, image_stem,
                lambda p: image.objmask.save_overlay(
                    filepath=p, overlay_alpha=self.overlay_alpha
                ),
            )
            if path:
                saved_paths["objmask_overlay"] = path

        return saved_paths

    def aggregate_pipeline_csv(self, pipeline_name: str) -> Optional[Path]:
        """Combine all per-image CSVs for one pipeline into a single CSV.

        Args:
            pipeline_name: Pipeline name.

        Returns:
            Path to aggregated CSV, or ``None`` if no measurements found.
        """
        meas_dir = self.results_dir / pipeline_name / "measurements"
        csv_files = sorted(meas_dir.glob("*.csv"))

        if not csv_files:
            return None

        dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if "Metadata_Pipeline" not in df.columns:
                    df.insert(0, "Metadata_Pipeline", pipeline_name)
                dfs.append(df)
            except Exception as e:
                logger.warning(
                    f"Failed to read {csv_file.name}: {type(e).__name__}: {e}"
                )

        if not dfs:
            return None

        combined = pd.concat(dfs, axis=0, ignore_index=True)
        out_path = self.results_dir / pipeline_name / "pipeline_measurements.csv"
        combined.to_csv(out_path, index=False)
        return out_path

    def aggregate_master_csv(self, pipeline_names: List[str]) -> Optional[Path]:
        """Combine all pipeline CSVs into a single master CSV.

        Adds a ``Metadata_Pipeline`` column to identify each pipeline's rows.

        Args:
            pipeline_names: List of all pipeline names to aggregate.

        Returns:
            Path to ``master_measurements.csv``, or ``None`` if empty.
        """
        all_dfs = []

        for pipe_name in pipeline_names:
            # First aggregate per-pipeline
            self.aggregate_pipeline_csv(pipe_name)

            # Then read the aggregated file (or fall back to individual CSVs)
            agg_path = self.results_dir / pipe_name / "pipeline_measurements.csv"
            if agg_path.exists():
                try:
                    df = pd.read_csv(agg_path)
                    if "Metadata_Pipeline" not in df.columns:
                        df.insert(0, "Metadata_Pipeline", pipe_name)
                    all_dfs.append(df)
                except Exception as e:
                    logger.warning(
                        f"Failed to read {agg_path.name}: {type(e).__name__}: {e}"
                    )

        if not all_dfs:
            logger.warning("No valid measurements found for master aggregation")
            return None

        master_df = pd.concat(all_dfs, axis=0, ignore_index=True)
        master_path = self.base_dir / "master_measurements.csv"
        master_df.to_csv(master_path, index=False)
        logger.info(
            f"Aggregated {len(all_dfs)} pipeline CSVs into {master_path.name} "
            f"({len(master_df)} total rows)"
        )
        return master_path
