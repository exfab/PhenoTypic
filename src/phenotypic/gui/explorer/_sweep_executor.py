"""Batch execution engine for pipeline sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Union
import logging
import time
import traceback

from joblib import Parallel, delayed
import numpy as np
from skimage import io as skio

from phenotypic import Image, ImagePipeline

from ._pipeline_graph import PipelineGraph
from ._sweep_results import SweepResult, SweepResults

logger = logging.getLogger(__name__)


@dataclass
class ExecutionTask:
    """Single execution task."""

    variant_id: str
    pipeline: ImagePipeline
    config: Dict[str, Any]
    image_path: Path


class SweepExecutor:
    """Execute pipeline variants with parallel processing.

    Runs all pipeline variants from a PipelineGraph across one or more images,
    saving outputs and computing metrics.

    Args:
        graph: PipelineGraph defining variants to execute.
        output_dir: Directory to save outputs.
        data2save: Set of output views to save. Options include:
            'overlay', 'objmask', 'objmap', 'detect_mat', 'rgb', 'gray'.
        njobs: Number of parallel jobs (-1 for all CPUs).
        ground_truth_dir: Optional directory containing labeled PNG masks
            for computing IoU metrics.

    Examples:
        Basic usage:

        >>> executor = SweepExecutor(
        ...     graph=graph,
        ...     output_dir='./results',
        ...     data2save={'overlay', 'objmask'},
        ... )
        >>> results = executor.run(images=['./plate001.tif'])

        With ground truth comparison:

        >>> executor = SweepExecutor(
        ...     graph=graph,
        ...     output_dir='./results',
        ...     ground_truth_dir='./ground_truth',
        ... )
        >>> results = executor.run(images=['./plate001.tif'])
        >>> df = results.to_dataframe()
        >>> print(df[['variant_id', 'iou', 'precision', 'recall']])
    """

    def __init__(
        self,
        graph: PipelineGraph,
        output_dir: Union[str, Path],
        data2save: Optional[Set[str]] = None,
        njobs: int = -1,
        ground_truth_dir: Optional[Union[str, Path]] = None,
    ):
        self.graph = graph
        self.output_dir = Path(output_dir)
        self.data2save = data2save or {"overlay", "objmask"}
        self.njobs = njobs
        self.ground_truth_dir = Path(ground_truth_dir) if ground_truth_dir else None

        # Create output structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "pipelines").mkdir(exist_ok=True)

    def run(
        self,
        images: Union[Iterable[Union[str, Path]], Path, str],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> SweepResults:
        """Execute all pipeline variants on all images.

        Args:
            images: Images to process. Can be:
                - Single path to image file
                - Single path to directory (processes all images)
                - Glob pattern string (e.g., './plates/*.tif')
                - Iterable of paths
            progress_callback: Optional callback called with
                (current_index, total_count, message) for progress updates.

        Returns:
            SweepResults containing all execution results.

        Examples:
            >>> # Single image
            >>> results = executor.run(images='./plate001.tif')

            >>> # Directory
            >>> results = executor.run(images='./plates/')

            >>> # Glob pattern
            >>> results = executor.run(images='./plates/*.tif')

            >>> # Multiple specific images
            >>> results = executor.run(images=['./p1.tif', './p2.tif'])
        """
        # Validate graph has paths to output
        if self.graph.path_count == 0:
            raise ValueError(
                "Graph has no paths to output. Check that nodes are connected "
                "and at least one path leads to an output node."
            )

        # Resolve image paths
        image_paths = self._resolve_image_paths(images)
        if not image_paths:
            raise ValueError(f"No images found: {images}")

        logger.info(f"Starting sweep: {len(image_paths)} images, {self.graph.variant_count} variants")

        # Build task list
        tasks = self._build_tasks(image_paths)
        total = len(tasks)

        logger.info(f"Total tasks: {total}")

        # Execute tasks
        if self.njobs == 1:
            # Sequential execution (useful for debugging)
            results = []
            for idx, task in enumerate(tasks):
                if progress_callback:
                    progress_callback(idx, total, f"Running {task.variant_id}")
                result = self._execute_task(task)
                results.append(result)
        else:
            # Parallel execution
            # Note: Progress callback only fires at start/end for parallel execution.
            # For per-task progress, use njobs=1.
            if progress_callback:
                progress_callback(0, total, f"Running {total} tasks in parallel...")
            results = Parallel(n_jobs=self.njobs)(
                delayed(self._execute_task)(task)
                for task in tasks
            )

        # Finalize progress
        if progress_callback:
            progress_callback(total, total, "Complete")

        # Build results object
        sweep_results = SweepResults(
            sweep_dir=self.output_dir,
            results=results,
            created=datetime.now(),
            graph_config=self.graph.to_dict(),
        )

        # Save manifest
        sweep_results.save_manifest()

        # Save pipeline JSON files
        self._save_pipelines()

        logger.info(f"Sweep complete: {len(sweep_results.successful)}/{total} successful")

        return sweep_results

    def _resolve_image_paths(
        self,
        images: Union[Iterable[Union[str, Path]], Path, str],
    ) -> List[Path]:
        """Resolve input to list of image paths."""
        if isinstance(images, (str, Path)):
            path = Path(images)
            if path.is_file():
                return [path]
            elif path.is_dir():
                # Get all image files in directory
                extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
                return sorted([
                    p for p in path.iterdir()
                    if p.suffix.lower() in extensions
                ])
            elif "*" in str(path):
                # Glob pattern
                return sorted(Path(".").glob(str(path)))
            else:
                raise FileNotFoundError(f"Path not found: {path}")
        else:
            # Iterable of paths
            return [Path(p) for p in images]

    def _build_tasks(self, image_paths: List[Path]) -> List[ExecutionTask]:
        """Build list of execution tasks."""
        tasks = []

        for image_path in image_paths:
            for variant_id, pipeline, config in self.graph.enumerate_pipelines():
                tasks.append(ExecutionTask(
                    variant_id=variant_id,
                    pipeline=pipeline,
                    config=config,
                    image_path=image_path,
                ))

        return tasks

    def _execute_task(self, task: ExecutionTask) -> SweepResult:
        """Execute a single task."""
        start_time = time.perf_counter()

        try:
            # Load image from file
            image = Image.imread(task.image_path)

            # Execute pipeline
            result_image = task.pipeline.apply(image)

            # Save outputs
            outputs = self._save_outputs(task.variant_id, task.image_path, result_image)

            # Compute metrics
            metrics = self._compute_metrics(result_image, task.image_path)

            return SweepResult(
                variant_id=task.variant_id,
                pipeline_config=self._flatten_config(task.config),
                image_name=task.image_path.name,
                success=True,
                outputs=outputs,
                metrics=metrics,
                execution_time=time.perf_counter() - start_time,
            )

        except Exception as e:
            logger.error(f"Task failed: {task.variant_id} - {e}")
            logger.debug(traceback.format_exc())

            return SweepResult(
                variant_id=task.variant_id,
                pipeline_config=self._flatten_config(task.config),
                image_name=task.image_path.name,
                success=False,
                error=str(e),
                execution_time=time.perf_counter() - start_time,
            )

    def _save_outputs(
        self,
        variant_id: str,
        image_path: Path,
        result_image: Image,
    ) -> Dict[str, Path]:
        """Save output images for a result."""
        outputs = {}

        # Create variant directory
        variant_dir = self.output_dir / "images" / variant_id / image_path.stem
        variant_dir.mkdir(parents=True, exist_ok=True)

        for view in self.data2save:
            output_path = variant_dir / f"{view}.png"

            try:
                if view == "overlay":
                    # Use the accessor's save_overlay method for consistency with CLI
                    if result_image.objmap is None:
                        continue
                    # Prefer RGB overlay, fallback to grayscale
                    if not result_image.rgb.isempty():
                        result_image.rgb.save_overlay(
                            filepath=output_path,
                            overlay_alpha=0.3,
                        )
                    else:
                        result_image.gray.save_overlay(
                            filepath=output_path,
                            overlay_alpha=0.3,
                        )
                    outputs[view] = output_path
                    continue  # Already saved, skip the common save below
                elif view == "objmask":
                    if result_image.objmask is not None:
                        img_data = (result_image.objmask[:] * 255).astype(np.uint8)
                    else:
                        continue
                elif view == "objmap":
                    if result_image.objmap is not None:
                        # Normalize labeled map for visualization
                        objmap = result_image.objmap[:]
                        if objmap.max() > 0:
                            img_data = ((objmap / objmap.max()) * 255).astype(np.uint8)
                        else:
                            img_data = objmap.astype(np.uint8)
                    else:
                        continue
                elif view == "detect_mat":
                    img_data = result_image.detect_mat[:]
                    if img_data.max() <= 1.0:
                        img_data = (img_data * 255).astype(np.uint8)
                elif view == "rgb":
                    img_data = result_image.rgb[:]
                elif view == "gray":
                    img_data = result_image.gray[:]
                    if img_data.max() <= 1.0:
                        img_data = (img_data * 255).astype(np.uint8)
                else:
                    logger.warning(f"Unknown view type: {view}")
                    continue

                skio.imsave(str(output_path), img_data, check_contrast=False)
                outputs[view] = output_path

            except Exception as e:
                logger.warning(f"Failed to save {view}: {e}")

        return outputs

    def _compute_metrics(
        self,
        result_image: Image,
        image_path: Path,
    ) -> Dict[str, Any]:
        """Compute metrics for a processed image."""
        metrics = {}

        # Object count (always computed if objects detected)
        if result_image.objmask is not None:
            try:
                metrics["object_count"] = result_image.objects.count
            except Exception:
                # Fallback: count unique labels
                if result_image.objmap is not None:
                    unique = np.unique(result_image.objmap[:])
                    metrics["object_count"] = len(unique) - (1 if 0 in unique else 0)

        # Ground truth comparison (if available)
        if self.ground_truth_dir:
            gt_metrics = self._compute_gt_metrics(result_image, image_path)
            metrics.update(gt_metrics)

        # Include any measurements from pipeline (if MeasureFeatures was used)
        # This would be accessed via result_image.measurements or similar
        # depending on how measurements are stored

        return metrics

    def _compute_gt_metrics(
        self,
        result_image: Image,
        image_path: Path,
    ) -> Dict[str, Any]:
        """Compute metrics against ground truth."""
        metrics = {}

        # Find ground truth file
        gt_path = self.ground_truth_dir / f"{image_path.stem}_gt.png"
        if not gt_path.exists():
            # Try without suffix
            gt_path = self.ground_truth_dir / f"{image_path.stem}.png"

        if not gt_path.exists():
            return metrics

        try:
            # Load ground truth (labeled PNG)
            gt_mask = skio.imread(str(gt_path))
            if gt_mask.ndim > 2:
                gt_mask = gt_mask[:, :, 0]

            # Convert to binary for IoU
            gt_binary = gt_mask > 0

            if result_image.objmask is None:
                return metrics

            pred_binary = result_image.objmask[:]

            # Ensure same shape
            if gt_binary.shape != pred_binary.shape:
                logger.warning(
                    f"Shape mismatch: GT {gt_binary.shape} vs pred {pred_binary.shape}"
                )
                return metrics

            # Compute IoU
            intersection = np.logical_and(gt_binary, pred_binary).sum()
            union = np.logical_or(gt_binary, pred_binary).sum()
            iou = intersection / union if union > 0 else 0.0
            metrics["iou"] = float(iou)

            # Precision and recall
            tp = intersection
            fp = np.logical_and(pred_binary, ~gt_binary).sum()
            fn = np.logical_and(~pred_binary, gt_binary).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics["precision"] = float(precision)
            metrics["recall"] = float(recall)
            metrics["f1"] = float(f1)

            # Object count comparison
            gt_count = len(np.unique(gt_mask)) - (1 if 0 in gt_mask else 0)
            metrics["gt_object_count"] = int(gt_count)

        except Exception as e:
            logger.warning(f"Failed to compute GT metrics: {e}")

        return metrics

    def _flatten_config(self, config: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Flatten nested config dict for storage."""
        flat = {}
        for node_id, params in config.items():
            for param, value in params.items():
                # Use node_id.param as key
                key = f"{node_id[:8]}.{param}"
                # Handle non-serializable values
                if hasattr(value, "__class__") and not isinstance(value, (int, float, str, bool, list, dict)):
                    value = f"{value.__class__.__name__}(...)"
                flat[key] = value
        return flat

    def _save_pipelines(self) -> None:
        """Save pipeline JSON files for each variant."""
        pipelines_dir = self.output_dir / "pipelines"
        pipelines_dir.mkdir(exist_ok=True)

        for variant_id, pipeline, config in self.graph.enumerate_pipelines():
            try:
                pipeline.to_json(pipelines_dir / f"{variant_id}.json")
            except Exception as e:
                logger.warning(f"Failed to save pipeline {variant_id}: {e}")
