"""
Execution strategies for the PhenoTypic CLI.

This module implements the Strategy pattern for different execution modes:
- LocalParallelStrategy: joblib-based local parallelization
- AutonomousSLURMStrategy: SLURM cluster execution with array jobs via direct sbatch
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING

import click
import h5py  # type: ignore[import-untyped]
from joblib import Parallel, delayed

if TYPE_CHECKING:
    pass

from ._cli_types import (
    Dataset,
    DatasetResults,
    DatasetState,
    ExecutionConfig,
    ExecutionResults,
    ImageFailure,
)
from ._cli_output_manager import OutputManager
from ._cli_process_single import process_single_image_core
from phenotypic.sdk_.slurm import get_slurm_array_limit
from phenotypic.sdk_._file_locking import exclusive_path_lock
from ._cli_slurm_array_scripts import (
    generate_all_array_job_scripts,
    generate_terminal_finalizer_script,
)
from ._cli_slurm_submission import submit_slurm_script_chain
from ._cli_slurm_lifecycle import (
    initialize_slurm_lifecycle,
    mirror_job_to_metadata,
    new_slurm_generation,
)
from ._cli_update_state import append_event, append_completion_event, aggregate_state_from_events
from ._cli_failure_tracker import append_failure, read_failures
from ._dashboard import generate_dashboard, regenerate_dashboard_artifacts

from ._cli_constants import MAX_TRACEBACK_LINES
from phenotypic.sdk_ import (
    JobMetadataKey,
    HdfAttr,
    event_log_path,
    atomic_write_json,
    job_metadata_path,
    progress_dir,
)
from phenotypic.sdk_._io_constants import GUI_RECORD_GENERATION_ENV_VAR
from phenotypic.sdk_.typing_ import ImageTypeName

logger = logging.getLogger(__name__)


def _write_slurm_image_task_mapping(
    metadata_path: Path,
    image_task_mapping: Dict[str, List[str]],
) -> None:
    """Merge the initial array mapping without losing concurrent job records.

    Args:
        metadata_path: Canonical scheduler metadata path.
        image_task_mapping: Array-task keys mapped to dataset and image names.
    """
    metadata_lock = metadata_path.with_name(f".{metadata_path.name}.lock")
    with exclusive_path_lock(metadata_lock, timeout=60.0):
        job_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        job_metadata[JobMetadataKey.IMAGE_TASK_MAPPING] = image_task_mapping
        atomic_write_json(metadata_path, job_metadata)


def _truncate_error_message(error_msg: str, max_lines: int = MAX_TRACEBACK_LINES) -> str:
    """
    Truncate error messages to prevent event log bloat.

    For long error messages (e.g., full stack traces), keep first and last
    few lines and truncate middle.

    Args:
        error_msg: Error message to truncate
        max_lines: Maximum number of lines to keep (default: MAX_TRACEBACK_LINES)

    Returns:
        Truncated error message
    """
    lines = error_msg.split('\n')
    if len(lines) <= max_lines:
        return error_msg

    # Keep first 5 and last 5 lines
    kept_lines = lines[:5] + ['... (truncated) ...'] + lines[-5:]
    return '\n'.join(kept_lines)


class ExecutionStrategy(ABC):
    """Base class for execution strategies."""

    def __init__(self, config: ExecutionConfig, output_manager: OutputManager):
        """
        Initialize execution strategy.

        Args:
            config: Execution configuration
            output_manager: Output file manager
        """
        self.config = config
        self.output_manager = output_manager

    @abstractmethod
    def execute(
        self, datasets: List[Dataset], output_dir: Path
    ) -> ExecutionResults:
        """
        Execute processing for all datasets.

        Args:
            datasets: List of datasets to process
            output_dir: Base output directory

        Returns:
            Execution results with per-dataset statistics
        """
        pass


class LocalParallelStrategy(ExecutionStrategy):
    """Joblib-based local parallel execution."""

    def execute(
        self, datasets: List[Dataset], output_dir: Path
    ) -> ExecutionResults:
        """
        Execute processing locally with joblib parallelization.

        Args:
            datasets: List of datasets to process
            output_dir: Base output directory

        Returns:
            Execution results with success/failure statistics
        """
        start_time = datetime.now()
        event_log = event_log_path(output_dir)
        measure_only = bool(self.config.measure_only)

        # Flatten all images across datasets
        all_tasks = []
        for dataset in datasets:
            for image_path in dataset.images:
                all_tasks.append((dataset, image_path))

        if measure_only:
            logger.info(
                "Measure-only rerun: %d HDFs across %d datasets",
                len(all_tasks),
                len(datasets),
            )

        # Show dataset breakdown with rich
        from rich.console import Console

        console = Console()
        header = (
            "Measuring HDFs (Rerun)" if measure_only else "Processing Images"
        )
        console.print(f"\n[bold cyan]{header}[/bold cyan]")
        console.rule(style="cyan")

        for dataset in datasets:
            console.print(
                f"  Dataset: [cyan]{dataset.name}[/cyan] "
                f"([white]{len(dataset.images)} images[/white])"
            )

        console.rule(style="cyan")
        console.print(
            f"  Total: [bold]{len(all_tasks)} images[/bold] across "
            f"[bold]{len(datasets)} datasets[/bold]\n"
        )

        from ._cli_validation import pipeline_requires_gpu

        # Measure mode skips detection entirely; no pipeline ops are applied,
        # so GPU allocation is never required.
        has_gpu_ops = (
            False
            if measure_only
            else pipeline_requires_gpu(self.config.pipeline_json)
        )

        if has_gpu_ops:
            try:
                from phenotypic.detect.nn._checkpoint_manager import resolve_device

                device = resolve_device("auto")
                console.print(f"[green]✓ GPU detected: {device}[/green]")
            except ImportError:
                raise RuntimeError(
                    "Pipeline contains GPU-accelerated operations but PyTorch "
                    "is not installed. Install with: pip install phenotypic[torch]"
                )

            effective_n_jobs = 1
            console.print(
                "[yellow]Pipeline contains GPU operations — "
                "forcing sequential execution (n_jobs=1)[/yellow]"
            )
        else:
            effective_n_jobs = self.config.n_jobs

        if self.config.process_only_layer:
            worker = self._process_single_local_apply_only
        elif measure_only:
            worker = self._process_single_local_measure
        else:
            worker = self._process_single_local
        results = Parallel(n_jobs=effective_n_jobs, verbose=11)(
            delayed(worker)(dataset, image_path, output_dir, event_log)
            for dataset, image_path in all_tasks
        )

        # Aggregate results by dataset
        dataset_results = self._aggregate_results(datasets, results)

        end_time = datetime.now()

        # Generate progress manifest and dashboard (local mode — runs once)
        try:
            datasets_inventory = (
                self.config.full_dataset_inventory
                or {
                    ds.name: [image.name for image in ds.images]
                    for ds in datasets
                }
            )
            datasets_totals = {
                name: len(images)
                for name, images in datasets_inventory.items()
            }
            start_iso = start_time.isoformat(timespec="milliseconds")
            if self.config.process_only_layer:
                # Manifest only — no dashboard HTML, no aggregation (D13).
                from ._dashboard._manifest_builder import build_manifest

                build_manifest(
                    output_dir=output_dir,
                    progress_dir=progress_dir(output_dir),
                    datasets=datasets_totals,
                    execution_mode="local",
                    start_time=start_iso,
                    input_path=self.config.input_path.stem,
                    gui_record_generation=os.environ.get(
                        GUI_RECORD_GENERATION_ENV_VAR
                    ),
                    dataset_inventory=datasets_inventory,
                    processing_generation=self.config.processing_generation,
                )
            else:
                local_job_meta: dict = {
                    JobMetadataKey.START_TIME: start_iso,
                    JobMetadataKey.INPUT_PATH: self.config.input_path.stem,
                    JobMetadataKey.EXECUTION_MODE: "local",
                    JobMetadataKey.GUI_RECORD_GENERATION: os.environ.get(
                        GUI_RECORD_GENERATION_ENV_VAR
                    ),
                    JobMetadataKey.PROCESSING_GENERATION: (
                        self.config.processing_generation
                    ),
                    JobMetadataKey.DATASETS: {
                        name: {"total": len(images), "images": images}
                        for name, images in datasets_inventory.items()
                    },
                }
                regenerate_dashboard_artifacts(
                    output_dir,
                    local_job_meta,
                    datasets_totals,
                )
        except Exception:
            logger.debug("Failed to generate progress dashboard", exc_info=True)

        return ExecutionResults(
            datasets=dataset_results,
            total_images=len(all_tasks),
            total_completed=sum(r.completed for r in dataset_results.values()),
            total_failed=sum(r.failed for r in dataset_results.values()),
            execution_mode="local",
            start_time=start_time,
            end_time=end_time,
        )

    def _process_single_local(
        self,
        dataset: Dataset,
        image_path: Path,
        output_dir: Path,
        event_log: Path,
    ) -> tuple[str, str, bool, str]:
        """
        Process single image locally.

        Returns:
            Tuple of (dataset_name, image_name, success, error_message)
        """
        # Log "started" event
        append_event(event_log, dataset.name, image_path.name, "started")

        try:
            # Prepare read kwargs.
            read_kwargs: Dict[str, Any] = {}
            if self.config.bit_depth:
                read_kwargs["bit_depth"] = self.config.bit_depth
            if self.config.detect_mode != "gray":
                read_kwargs["detect_mode"] = self.config.detect_mode

            # Process
            process_single_image_core(
                pipeline_path=self.config.pipeline_json,
                image_path=image_path,
                output_dir=output_dir,
                dataset_name=dataset.name,
                image_type=self.config.image_type,
                read_kwargs=read_kwargs,
                output_manager=self.output_manager,
                cli_nrows=self.config.nrows,
                cli_ncols=self.config.ncols,
            )

            # Log success
            append_completion_event(
                event_log, dataset.name, image_path.name, "completed"
            )
            return (dataset.name, image_path.name, True, "")

        except Exception as e:
            # Log failure
            import traceback

            error_msg = str(e)
            tb = traceback.format_exc()

            logger.error(
                "Processing failed for %s/%s:\n%s",
                dataset.name, image_path.name, tb,
            )

            # Truncate error message to prevent event log bloat
            truncated_msg = _truncate_error_message(error_msg)

            append_completion_event(
                event_log, dataset.name, image_path.name, "failed", truncated_msg
            )

            # Write structured failure record
            try:
                prog_dir = progress_dir(output_dir)
                append_failure(
                    prog_dir,
                    dataset=dataset.name,
                    image=image_path.name,
                    error_type=type(e).__name__,
                    error_message=error_msg,
                    traceback=tb,
                )
            except Exception:
                logger.warning("Failed to write failure record", exc_info=True)

            return (dataset.name, image_path.name, False, tb)

    def _process_single_local_apply_only(
        self,
        dataset: Dataset,
        image_path: Path,
        output_dir: Path,
        event_log: Path,
    ) -> tuple[str, str, bool, str]:
        """Apply-only (process-only) per-image worker.

        Mirrors :meth:`_process_single_local` — same event-log helpers and
        ``(name, success, error)`` return shape so the manifest aggregator
        treats process-only results identically — but dispatches to
        :func:`phenotypic._cli._cli_process_only.process_single_apply_only_core`
        with ``input_root`` captured from ``self.config.input_path`` for
        mirrored output paths.
        """
        from ._cli_process_only import process_single_apply_only_core

        append_event(event_log, dataset.name, image_path.name, "started")
        try:
            read_kwargs: Dict[str, Any] = {}
            if self.config.bit_depth:
                read_kwargs["bit_depth"] = self.config.bit_depth
            if self.config.detect_mode != "gray":
                read_kwargs["detect_mode"] = self.config.detect_mode

            process_single_apply_only_core(
                pipeline_path=self.config.pipeline_json,
                image_path=image_path,
                input_root=self.config.input_path,
                output_dir=output_dir,
                image_type=self.config.image_type,
                layer=self.config.process_only_layer,  # type: ignore[arg-type]
                read_kwargs=read_kwargs,
                cli_nrows=self.config.nrows,
                cli_ncols=self.config.ncols,
            )
            append_completion_event(
                event_log, dataset.name, image_path.name, "completed"
            )
            return (dataset.name, image_path.name, True, "")
        except Exception as e:
            import traceback

            error_msg = str(e)
            tb = traceback.format_exc()
            logger.error(
                "Apply-only failed for %s/%s:\n%s",
                dataset.name, image_path.name, tb,
            )
            append_completion_event(
                event_log,
                dataset.name,
                image_path.name,
                "failed",
                _truncate_error_message(error_msg),
            )
            try:
                prog_dir = progress_dir(output_dir)
                append_failure(
                    prog_dir,
                    dataset=dataset.name,
                    image=image_path.name,
                    error_type=type(e).__name__,
                    error_message=error_msg,
                    traceback=tb,
                )
            except Exception:
                logger.warning("Failed to write failure record", exc_info=True)
            return (dataset.name, image_path.name, False, error_msg)

    def _process_single_local_measure(
        self,
        dataset: Dataset,
        image_path: Path,
        output_dir: Path,
        event_log: Path,
    ) -> tuple[str, str, bool, str]:
        """
        Rerun ``pipeline.measure()`` on a single already-processed HDF file.

        Mirrors :meth:`_process_single_local` — same event-log helpers and
        same (name, success, error) return shape so the dashboard aggregator
        treats measure-mode results identically to forward-run results —
        but dispatches to
        :func:`phenotypic._cli._cli_process_single.process_single_hdf_measure_core`
        instead of the detection path.  No state file is touched; the
        top-level CLI is responsible for state in forward mode only.

        Args:
            dataset: Dataset metadata (used for event logging).
            image_path: Path to the ``.h5`` file to reload.
            output_dir: Base output directory (passed through; measure path
                does not use it directly).
            event_log: Path to the processing event log.

        Returns:
            Tuple of ``(dataset_name, hdf_filename, success, error_or_tb)``
            matching the forward-path contract.
        """
        # Lazy-import the measure worker to match the forward-run pattern and
        # avoid any new top-level import cycle.
        from ._cli_process_single import process_single_hdf_measure_core

        append_event(event_log, dataset.name, image_path.name, "started")

        try:
            # Detect the saved image class from the HDF root attr so
            # GridImage files rehydrate with their grid state intact.
            resolved_image_type: ImageTypeName = (
                self.config.image_type  # type: ignore[assignment]
            )
            try:
                with h5py.File(image_path, "r") as hf:
                    saved_class = hf.attrs.get(HdfAttr.PHENOTYPIC_CLASS)
                    if isinstance(saved_class, bytes):
                        saved_class = saved_class.decode(
                            "utf-8", errors="replace"
                        )
                    if saved_class == "GridImage":
                        resolved_image_type = "GridImage"
                    elif saved_class == "Image":
                        resolved_image_type = "Image"
            except (OSError, KeyError) as hdf_err:
                logger.warning(
                    "Could not read phenotypic_class from %s (%s: %s); "
                    "falling back to configured image_type=%s",
                    image_path,
                    type(hdf_err).__name__,
                    hdf_err,
                    resolved_image_type,
                )

            process_single_hdf_measure_core(
                pipeline_path=self.config.pipeline_json,
                hdf_path=image_path,
                output_dir=output_dir,
                dataset_name=dataset.name,
                image_type=resolved_image_type,
                output_manager=self.output_manager,
            )

            append_completion_event(
                event_log, dataset.name, image_path.name, "completed"
            )
            return (dataset.name, image_path.name, True, "")

        except Exception as e:
            import traceback

            error_msg = str(e)
            tb = traceback.format_exc()

            logger.error(
                "Measure rerun failed for %s/%s:\n%s",
                dataset.name, image_path.name, tb,
            )

            truncated_msg = _truncate_error_message(error_msg)

            append_completion_event(
                event_log,
                dataset.name,
                image_path.name,
                "failed",
                truncated_msg,
            )

            try:
                prog_dir = progress_dir(output_dir)
                append_failure(
                    prog_dir,
                    dataset=dataset.name,
                    image=image_path.name,
                    error_type=type(e).__name__,
                    error_message=error_msg,
                    traceback=tb,
                )
            except Exception:
                logger.warning(
                    "Failed to write failure record", exc_info=True
                )

            return (dataset.name, image_path.name, False, tb)

    def _aggregate_results(
        self, datasets: List[Dataset], results: List[tuple]
    ) -> dict[str, DatasetResults]:
        """
        Aggregate processing results by dataset.

        Args:
            datasets: List of datasets
            results: List of (dataset_name, image_name, success, error) tuples

        Returns:
            Dictionary mapping dataset names to DatasetResults
        """
        # Initialize result containers
        dataset_results: dict[str, dict[str, Any]] = {}
        for dataset in datasets:
            dataset_results[dataset.name] = {
                "total": len(dataset.images),
                "completed": 0,
                "failed": 0,
                "failures": [],
            }

        # Process results
        for dataset_name, image_name, success, error_or_tb in results:
            ds_result = dataset_results[dataset_name]

            if success:
                ds_result["completed"] += 1
            else:
                ds_result["failed"] += 1

                # Parse error details
                if error_or_tb:
                    lines = error_or_tb.strip().split("\n")
                    # Extract exception type and message from last line
                    last_line = lines[-1] if lines else "Unknown error"
                    if ":" in last_line:
                        error_type, error_message = last_line.split(":", 1)
                        error_type = error_type.strip()
                        error_message = error_message.strip()
                    else:
                        error_type = "Exception"
                        error_message = last_line
                else:
                    error_type = "Exception"
                    error_message = "Unknown error"
                    error_or_tb = ""

                failure = ImageFailure(
                    dataset=dataset_name,
                    image_filename=image_name,
                    error_type=error_type,
                    error_message=error_message,
                    traceback=error_or_tb,
                    timestamp=datetime.now(),
                )
                ds_result["failures"].append(failure)

        # Convert to DatasetResults objects
        return {
            name: DatasetResults(
                name=name,
                total=data["total"],
                completed=data["completed"],
                failed=data["failed"],
                failures=data["failures"],
            )
            for name, data in dataset_results.items()
        }


class AutonomousSLURMStrategy(ExecutionStrategy):
    """Session-independent SLURM execution with array jobs via direct sbatch."""

    def execute(
        self, datasets: List[Dataset], output_dir: Path
    ) -> ExecutionResults:
        """
        Execute processing on SLURM cluster using array jobs.

        Generates array job scripts with automatic chunking based on SLURM
        limits, submits via direct sbatch with sequential dataset dependencies,
        and optionally monitors progress.

        Args:
            datasets: List of datasets to process
            output_dir: Base output directory

        Returns:
            Execution results (may be incomplete if jobs still running)
        """
        start_time = datetime.now()
        measure_only = bool(self.config.measure_only)

        total_images = sum(len(d.images) for d in datasets)
        if measure_only:
            logger.info(
                "Measure-only rerun: %d HDFs across %d datasets",
                total_images,
                len(datasets),
            )

        # Show dataset breakdown with rich
        from rich.console import Console

        console = Console()
        header = (
            "SLURM Measure Rerun Submission"
            if measure_only
            else "SLURM Job Submission"
        )
        console.print(f"\n[bold cyan]{header}[/bold cyan]")
        console.rule(style="cyan")

        for dataset in datasets:
            console.print(
                f"  Dataset: [cyan]{dataset.name}[/cyan] "
                f"([white]{len(dataset.images)} images[/white])"
            )

        console.rule(style="cyan")
        console.print(
            f"  Total: [bold]{total_images} images[/bold] across "
            f"[bold]{len(datasets)} datasets[/bold]\n"
        )

        # Query SLURM array limits
        console.print("[cyan]Querying SLURM array limits...[/cyan]")
        array_limit = get_slurm_array_limit()
        console.print(f"[green]✓[/green] SLURM array limit: [bold]{array_limit}[/bold]\n")

        from ._cli_validation import pipeline_requires_gpu

        # Measure mode never runs detection, so GPU provisioning is skipped.
        if not measure_only and pipeline_requires_gpu(self.config.pipeline_json):
            slurm_args = dict(self.config.slurm_args)

            if "slurm_gpus_per_node" not in slurm_args:
                slurm_args["slurm_gpus_per_node"] = 1
                console.print(
                    "[yellow]Pipeline contains GPU operations — "
                    "auto-requesting --gpus-per-node=1[/yellow]"
                )

            partition = slurm_args.get("slurm_partition")
            if partition:
                try:
                    result = subprocess.run(
                        ["sinfo", "-p", partition, "--Format=gres", "--noheader"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    gres_info = result.stdout.strip()
                    if "gpu" not in gres_info.lower():
                        raise RuntimeError(
                            f"Pipeline contains GPU operations but partition "
                            f"'{partition}' has no GPUs (sinfo gres: "
                            f"{gres_info!r}). Use "
                            f"--slurm slurm_partition=<gpu-partition>."
                        )
                except FileNotFoundError:
                    pass  # sinfo not available (not on a SLURM login node)
                except subprocess.TimeoutExpired:
                    pass  # sinfo hung, proceed anyway

            self.config.slurm_args = slurm_args

        # Generate array job scripts for all datasets
        console.print("[bold cyan]Generating array job scripts...[/bold cyan]")
        all_scripts = generate_all_array_job_scripts(
            datasets, self.config, output_dir, array_limit
        )

        # Show per-dataset results
        for dataset in datasets:
            scripts = all_scripts.get(dataset.name, [])
            num_images = len(dataset.images)

            if len(scripts) == 0:
                continue
            elif len(scripts) == 1:
                # Single script
                console.print(
                    f"  [green]✓[/green] {dataset.name}: 1 script "
                    f"([white]{num_images} images[/white])"
                )
            else:
                # Multiple scripts (chunked)
                console.print(
                    f"  [green]✓[/green] {dataset.name}: {len(scripts)} scripts "
                    f"([white]{num_images} images[/white])"
                )

        total_jobs = sum(len(scripts) for scripts in all_scripts.values())
        console.print(
            f"[green]✓ Generated {total_jobs} array job scripts[/green]\n"
        )

        # Flatten all chunk scripts in dataset order for dispatcher chain
        flat_scripts: List[Path] = []
        for dataset in datasets:
            flat_scripts.extend(all_scripts.get(dataset.name, []))

        if not flat_scripts:
            raise RuntimeError(
                "No array job scripts were generated. "
                "Check that datasets contain images."
            )

        # Publish the generation fence and metadata skeleton before the first
        # scheduler call. The lifecycle recorder fills both job registries
        # while the submit/cancel lock is still held.
        generation = new_slurm_generation()
        initialize_slurm_lifecycle(
            output_dir, generation=generation, mode="ordinary"
        )
        prog_dir = progress_dir(output_dir)
        prog_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = job_metadata_path(output_dir)
        job_metadata: Dict[str, Any] = {
            JobMetadataKey.START_TIME: start_time.isoformat(
                timespec="milliseconds"
            ),
            JobMetadataKey.EXECUTION_MODE: "slurm",
            JobMetadataKey.DATASETS: {
                name: {
                    "total": len(images),
                    "images": images,
                }
                for name, images in (
                    self.config.full_dataset_inventory
                    or {
                        ds.name: [img.name for img in ds.images]
                        for ds in datasets
                    }
                ).items()
            },
            JobMetadataKey.CHUNK_SCRIPTS: [str(s) for s in flat_scripts],
            JobMetadataKey.CHUNK_JOB_IDS: {},
            JobMetadataKey.SLURM_JOB_IDS: {},
            JobMetadataKey.IMAGE_TASK_MAPPING: {},
            JobMetadataKey.INCLUDE_DATASET_COLUMN: (
                self.config.include_dataset_column
            ),
            JobMetadataKey.METADATA_CSV: (
                str(self.config.metadata_csv)
                if self.config.metadata_csv
                else None
            ),
            JobMetadataKey.INPUT_PATH: self.config.input_path.stem,
            JobMetadataKey.GUI_RECORD_GENERATION: os.environ.get(
                "PHENOTYPIC_GUI_RECORD_GENERATION"
            ),
            JobMetadataKey.PROCESSING_GENERATION: (
                self.config.processing_generation
            ),
            "slurm_metadata_version": 2,
            "slurm_generation": generation,
        }
        atomic_write_json(metadata_path, job_metadata)

        finalizer_script = generate_terminal_finalizer_script(
            self.config,
            output_dir,
        )
        submission = submit_slurm_script_chain(
            flat_chunk_scripts=flat_scripts,
            output_dir=output_dir,
            slurm_args=self.config.slurm_args,
            console=console,
            finalizer_script=finalizer_script,
        )
        job_ids = submission.job_ids
        flat_scripts = submission.flat_scripts
        mirror_job_to_metadata(
            output_dir,
            generation=generation,
            token="chunk-0",
            role="chunk",
            job_id=str(job_ids[0]),
        )
        if len(job_ids) > 1:
            has_initial_dispatcher = bool(submission.dispatcher_scripts)
            mirror_job_to_metadata(
                output_dir,
                generation=generation,
                token=(
                    "dispatcher-1"
                    if has_initial_dispatcher
                    else "finalizer"
                ),
                role="dispatcher" if has_initial_dispatcher else "finalizer",
                job_id=str(job_ids[1]),
            )

        # ── Progress dashboard setup ──────────────────────────────────
        # Build image-task mapping: {job_id}_{array_idx} -> [dataset, image]
        # NOTE: Only chunk 0's job ID is known at submission time. Subsequent
        # chunk IDs are assigned by the drip-feed dispatcher after each chunk
        # completes, so OOM detection via sacct is limited to chunk 0 images
        # until the sentinel discovers later job IDs.
        image_task_mapping: Dict[str, List[str]] = {}
        array_offset = 0
        for dataset in datasets:
            for i, img_path in enumerate(dataset.images):
                task_key = f"{job_ids[0]}_{array_offset + i}"
                image_task_mapping[task_key] = [dataset.name, img_path.name]
            array_offset += len(dataset.images)

        # Preserve the lifecycle's role-bearing job records.
        _write_slurm_image_task_mapping(metadata_path, image_task_mapping)
        console.print(f"[green]✓[/green] Job metadata: [dim]{metadata_path}[/dim]")

        if self.config.process_only_layer:
            # Process-only (D13): no aggregation/dashboard. A dedicated
            # finalizer is dependency-ordered after the final image chunk.
            console.print(
                "[green]✓[/green] Manifest finalizer follows the last chunk "
                "(process-only: no aggregation/dashboard)\n"
            )
        else:
            console.print(
                "[green]✓[/green] Terminal finalizer follows the last image "
                "chunk; nonterminal checkpoints remain embedded"
            )

            # Generate dashboard HTML
            generate_dashboard(output_dir, execution_mode="slurm")
            console.print(
                f"[green]✓[/green] Dashboard: "
                f"[bold]{output_dir / 'dashboard.html'}[/bold]  "
                f"Analysis: [bold]{output_dir / 'analysis.html'}[/bold]\n"
            )

        # Wait if requested
        if self.config.wait:
            click.echo(
                "\nMonitoring progress (Ctrl+C to detach, jobs continue)..."
            )
            final_results = self._monitor_progress(output_dir, datasets)
        else:
            click.echo("\nJobs submitted. Monitor progress with:")
            click.echo(f"  Open: {output_dir / 'dashboard.html'}")
            click.echo(f"  Analysis: {output_dir / 'analysis.html'}")
            click.echo("  squeue -u $USER --array")
            click.echo(f"  tail -f {event_log_path(output_dir)}")
            final_results = None

        end_time = datetime.now()

        # Return results (may be incomplete if not waiting)
        if final_results:
            return final_results

        # Return partial results for non-wait mode
        total_images = sum(len(d.images) for d in datasets)
        return ExecutionResults(
            datasets={},  # Unknown until jobs complete
            total_images=total_images,
            total_completed=0,
            total_failed=0,
            execution_mode="slurm",
            start_time=start_time,
            end_time=end_time,
        )

    def _monitor_progress(
        self, output_dir: Path, datasets: List[Dataset]
    ) -> ExecutionResults:
        """
        Monitor SLURM job progress with live updates.

        Args:
            output_dir: Base output directory
            datasets: List of datasets being processed

        Returns:
            Final execution results after all jobs complete
        """
        event_log = event_log_path(output_dir)
        start_time = datetime.now()

        total_images = sum(len(d.images) for d in datasets)
        inventory = (
            self.config.full_dataset_inventory
            or {
                dataset.name: [image.name for image in dataset.images]
                for dataset in datasets
            }
        )
        last_completed = 0

        try:
            while True:
                # Aggregate latest events
                datasets_state = aggregate_state_from_events(
                    event_log,
                    inventory=inventory,
                    generation=self.config.processing_generation,
                )

                # Count completed and failed
                total_completed = sum(
                    len(ds.completed) for ds in datasets_state.values()
                )
                total_failed = sum(
                    len(ds.failed) for ds in datasets_state.values()
                )
                remaining = total_images - total_completed - total_failed

                # Show progress update
                if total_completed != last_completed:
                    click.echo(
                        f"Progress: {total_completed}/{total_images} completed, "
                        f"{total_failed} failed, {remaining} remaining"
                    )
                    last_completed = total_completed

                # Check if all done
                if remaining == 0:
                    click.echo("\n✓ All jobs complete!")
                    break

                # Wait before next check
                time.sleep(10)

        except KeyboardInterrupt:
            click.echo(
                "\n\nMonitoring stopped. Jobs continue running on SLURM."
            )

        # Build final results
        end_time = datetime.now()
        datasets_state = aggregate_state_from_events(
            event_log,
            inventory=inventory,
            generation=self.config.processing_generation,
        )

        # Enrich with structured failure data from failures.jsonl
        prog_dir = progress_dir(output_dir)
        failure_records = read_failures(prog_dir)
        failure_lookup: dict[tuple[str, str], dict] = {}
        for rec in failure_records:
            key = (rec.get("dataset", ""), rec.get("image", ""))
            failure_lookup[key] = rec  # Last record wins (retries)

        # Convert to DatasetResults
        dataset_results = {}
        for dataset in datasets:
            ds_state = datasets_state.get(dataset.name, DatasetState())

            failures = []
            for img_name in ds_state.failed:
                error_msg = ds_state.errors.get(img_name, "Unknown error")
                rec = failure_lookup.get((dataset.name, img_name), {})
                failures.append(
                    ImageFailure(
                        dataset=dataset.name,
                        image_filename=img_name,
                        error_type=rec.get("error_type", "Exception"),
                        error_message=rec.get("error_message", error_msg),
                        traceback=rec.get("traceback", ""),
                        timestamp=datetime.now(),
                    )
                )

            dataset_results[dataset.name] = DatasetResults(
                name=dataset.name,
                total=len(dataset.images),
                completed=len(ds_state.completed),
                failed=len(ds_state.failed),
                failures=failures,
            )

        return ExecutionResults(
            datasets=dataset_results,
            total_images=total_images,
            total_completed=sum(
                r.completed for r in dataset_results.values()
            ),
            total_failed=sum(r.failed for r in dataset_results.values()),
            execution_mode="slurm",
            start_time=start_time,
            end_time=end_time,
        )


def uses_staged_gpu_strategy(config: ExecutionConfig) -> bool:
    """Return whether *config* routes through the staged GPU engine."""
    if config.measure_only:
        return False

    from ._cli_validation import pipeline_requires_gpu

    if not pipeline_requires_gpu(config.pipeline_json):
        return False
    if config.is_slurm_mode():
        return config.process_only_layer is None
    return config.process_only_layer in (None, "objmap")


def create_execution_strategy(
    config: ExecutionConfig, output_manager: OutputManager
) -> ExecutionStrategy:
    """
    Factory function to create appropriate execution strategy.

    Args:
        config: Execution configuration
        output_manager: Output file manager

    Returns:
        ExecutionStrategy instance (Local or SLURM)
    """
    if config.is_slurm_mode():
        # SLURM: a forward GPU run becomes the staged 3-link afterany chain
        # (CPU preprocess -> resident-model GPU detect shards -> CPU measure).
        # measure-only and process-layer exports keep the per-image
        # AutonomousSLURMStrategy (the staged SLURM path is forward-only).
        if uses_staged_gpu_strategy(config):
            from ._cli_staged_slurm import StagedSlurmStrategy

            return StagedSlurmStrategy(config, output_manager)
        return AutonomousSLURMStrategy(config, output_manager)

    # Local: route forward GPU runs (and objmap export) through the staged
    # engine, which loads the resident model once instead of per image. The
    # measure-only path and other process-layer exports (rgb/gray/detect_mat,
    # which come from the pre-detector ops) keep the per-image
    # LocalParallelStrategy. Imports are local to avoid a circular import
    # (_cli_staged_strategy imports ExecutionStrategy from this module).
    if uses_staged_gpu_strategy(config):
        from ._cli_staged_strategy import StagedGpuStrategy

        return StagedGpuStrategy(config, output_manager)
    return LocalParallelStrategy(config, output_manager)
