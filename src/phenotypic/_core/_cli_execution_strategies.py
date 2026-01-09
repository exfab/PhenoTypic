"""
Execution strategies for the PhenoTypic CLI.

This module implements the Strategy pattern for different execution modes:
- LocalParallelStrategy: joblib-based local parallelization
- AutonomousSLURMStrategy: SLURM cluster execution with bash scripts + submitit
"""

from __future__ import annotations

import subprocess
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, TYPE_CHECKING

import click
from joblib import Parallel, delayed

if TYPE_CHECKING:
    from phenotypic import Image, GridImage

from ._cli_types import (
    Dataset,
    DatasetResults,
    ExecutionConfig,
    ExecutionResults,
    ImageFailure,
)
from ._cli_output_manager import OutputManager
from ._cli_process_single import process_single_image_core
from ._cli_slurm_scripts import generate_all_image_scripts
from ._cli_update_state import append_completion_event, aggregate_state_from_events


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
        event_log = output_dir / "processing_events.log"

        # Flatten all images across datasets
        all_tasks = []
        for dataset in datasets:
            for image_path in dataset.images:
                all_tasks.append((dataset, image_path))

        click.echo(f"Processing {len(all_tasks)} images with joblib (n_jobs={self.config.n_jobs})...")

        # Process in parallel
        results = Parallel(n_jobs=self.config.n_jobs)(
            delayed(self._process_single_local)(
                dataset, image_path, output_dir, event_log
            )
            for dataset, image_path in all_tasks
        )

        # Aggregate results by dataset
        dataset_results = self._aggregate_results(datasets, results)

        end_time = datetime.now()

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
        try:
            # Prepare read kwargs
            read_kwargs = {}
            if self.config.image_type == "GridImage":
                read_kwargs["nrows"] = self.config.nrows
                read_kwargs["ncols"] = self.config.ncols
            if self.config.bit_depth:
                read_kwargs["bit_depth"] = self.config.bit_depth

            # Process
            success = process_single_image_core(
                pipeline_path=self.config.pipeline_json,
                image_path=image_path,
                output_dir=output_dir,
                dataset_name=dataset.name,
                image_type=self.config.image_type,
                read_kwargs=read_kwargs,
                output_manager=self.output_manager,
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

            append_completion_event(
                event_log, dataset.name, image_path.name, "failed", error_msg
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
        dataset_results = {}
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
    """Session-independent SLURM execution with bash scripts."""

    def execute(
        self, datasets: List[Dataset], output_dir: Path
    ) -> ExecutionResults:
        """
        Execute processing on SLURM cluster.

        Generates bash scripts for all images, submits via submitit with
        dependency chains, and returns immediately (or waits if --wait flag set).

        Args:
            datasets: List of datasets to process
            output_dir: Base output directory

        Returns:
            Execution results (may be incomplete if jobs still running)
        """
        start_time = datetime.now()

        # Generate all bash scripts
        click.echo("Generating SLURM job scripts...")
        all_scripts = generate_all_image_scripts(
            datasets, self.config, output_dir, self.output_manager
        )

        total_scripts = sum(len(scripts) for scripts in all_scripts.values())
        click.echo(f"Generated {total_scripts} job scripts")

        # Submit jobs with submitit
        click.echo("Submitting jobs to SLURM...")
        job_ids = self._submit_jobs_with_dependencies(
            all_scripts, datasets, output_dir
        )

        click.echo(f"Submitted {len(job_ids)} jobs to SLURM")
        if job_ids:
            click.echo(
                f"Job IDs: {', '.join(str(j) for j in job_ids[:5])}..."
                f" (showing first 5)"
                if len(job_ids) > 5
                else f"Job IDs: {', '.join(str(j) for j in job_ids)}"
            )

        # Wait if requested
        if self.config.wait:
            click.echo(
                "\nMonitoring progress (Ctrl+C to detach, jobs continue)..."
            )
            final_results = self._monitor_progress(output_dir, datasets)
        else:
            click.echo("\nJobs submitted. Monitor progress with:")
            click.echo(
                f"  python -m phenotypic.tools.monitor_slurm_jobs {output_dir}"
            )
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

    def _submit_jobs_with_dependencies(
        self,
        all_scripts: dict[str, List[Path]],
        datasets: List[Dataset],
        output_dir: Path,
    ) -> List[str]:
        """
        Submit jobs with SLURM dependency chains.

        Args:
            all_scripts: Dictionary mapping dataset names to script paths
            datasets: List of datasets
            output_dir: Base output directory

        Returns:
            List of job IDs
        """
        try:
            import submitit
        except ImportError:
            raise RuntimeError(
                "submitit not installed. Install with: pip install submitit"
            )

        # Create executor
        executor_folder = output_dir / "submitit_logs"
        executor_folder.mkdir(parents=True, exist_ok=True)

        executor = submitit.AutoExecutor(folder=executor_folder)
        executor.update_parameters(**self.config.slurm_kwds)

        # Submit jobs (one per bash script)
        job_ids = []

        for dataset in datasets:
            scripts = all_scripts.get(dataset.name, [])
            if not scripts:
                continue

            # Submit all scripts for this dataset
            for script_path in scripts:
                # Submit bash script execution
                job = executor.submit(self._run_bash_script, script_path)
                job_ids.append(job.job_id)

        return job_ids

    @staticmethod
    def _run_bash_script(script_path: Path) -> int:
        """
        Execute a bash script (called by submitit).

        Args:
            script_path: Path to bash script

        Returns:
            Exit code
        """
        result = subprocess.run(
            ["bash", str(script_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            click.echo(f"Script {script_path.name} failed:", err=True)
            click.echo(result.stderr, err=True)
        return result.returncode

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
        event_log = output_dir / "processing_events.log"
        start_time = datetime.now()

        total_images = sum(len(d.images) for d in datasets)
        last_completed = 0

        try:
            while True:
                # Aggregate latest events
                datasets_state = aggregate_state_from_events(event_log)

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
        datasets_state = aggregate_state_from_events(event_log)

        # Convert to DatasetResults
        dataset_results = {}
        for dataset in datasets:
            ds_state = datasets_state.get(
                dataset.name, {"completed": set(), "failed": set(), "errors": {}}
            )

            failures = []
            for img_name in ds_state.get("failed", set()):
                error_msg = ds_state.get("errors", {}).get(img_name, "Unknown error")
                failures.append(
                    ImageFailure(
                        dataset=dataset.name,
                        image_filename=img_name,
                        error_type="Exception",
                        error_message=error_msg,
                        traceback="",
                        timestamp=datetime.now(),
                    )
                )

            dataset_results[dataset.name] = DatasetResults(
                name=dataset.name,
                total=len(dataset.images),
                completed=len(ds_state.get("completed", set())),
                failed=len(ds_state.get("failed", set())),
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
        return AutonomousSLURMStrategy(config, output_manager)
    else:
        return LocalParallelStrategy(config, output_manager)
