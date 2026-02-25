"""
Execution strategies for the PhenoTypic CLI.

This module implements the Strategy pattern for different execution modes:
- LocalParallelStrategy: joblib-based local parallelization
- AutonomousSLURMStrategy: SLURM cluster execution with array jobs via direct sbatch
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import click
from joblib import Parallel, delayed

if TYPE_CHECKING:
    pass

from ._cli_types import (
    Dataset,
    DatasetResults,
    ExecutionConfig,
    ExecutionResults,
    ImageFailure,
)
from ._cli_output_manager import OutputManager
from ._cli_process_single import process_single_image_core
from phenotypic.tools_.slurm import (
    get_slurm_array_limit,
    generate_dispatcher_chain,
    submit_drip_feed_start,
)
from ._cli_slurm_array_scripts import generate_all_array_job_scripts
from ._cli_update_state import append_completion_event, aggregate_state_from_events
from ._cli_constants import MAX_TRACEBACK_LINES


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
        event_log = output_dir / "processing_events.log"

        # Flatten all images across datasets
        all_tasks = []
        for dataset in datasets:
            for image_path in dataset.images:
                all_tasks.append((dataset, image_path))

        # Show dataset breakdown with rich
        from rich.console import Console

        console = Console()
        console.print("\n[bold cyan]Processing Images[/bold cyan]")
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

        # Process in parallel with verbose output
        results = Parallel(n_jobs=self.config.n_jobs, verbose=11)(
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
            if self.config.detect_mode != "gray":
                read_kwargs["detect_mode"] = self.config.detect_mode

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

            # Truncate error message to prevent event log bloat
            truncated_msg = _truncate_error_message(error_msg)

            append_completion_event(
                event_log, dataset.name, image_path.name, "failed", truncated_msg
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

        # Show dataset breakdown with rich
        from rich.console import Console

        console = Console()
        console.print("\n[bold cyan]SLURM Job Submission[/bold cyan]")
        console.rule(style="cyan")

        for dataset in datasets:
            console.print(
                f"  Dataset: [cyan]{dataset.name}[/cyan] "
                f"([white]{len(dataset.images)} images[/white])"
            )

        console.rule(style="cyan")
        total_images = sum(len(d.images) for d in datasets)
        console.print(
            f"  Total: [bold]{total_images} images[/bold] across "
            f"[bold]{len(datasets)} datasets[/bold]\n"
        )

        # Query SLURM array limits
        console.print("[cyan]Querying SLURM array limits...[/cyan]")
        array_limit = get_slurm_array_limit()
        console.print(f"[green]✓[/green] SLURM array limit: [bold]{array_limit}[/bold]\n")

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

        # Generate dispatcher chain (drip-feed submission)
        log_dir = output_dir / "logs" / "slurm"
        dispatcher_scripts = generate_dispatcher_chain(
            chunk_scripts=flat_scripts,
            output_dir=output_dir,
            slurm_args=self.config.slurm_args,
            log_dir=log_dir,
        )

        if not flat_scripts:
            raise RuntimeError(
                "No array job scripts were generated. "
                "Check that datasets contain images."
            )

        # Submit first chunk + first dispatcher only
        console.print("[bold cyan]Submitting jobs to SLURM...[/bold cyan]")

        job_ids, warning = submit_drip_feed_start(
            chunk_scripts=flat_scripts,
            dispatcher_scripts=dispatcher_scripts,
        )

        console.print(f"  Chunk 0: [green]Job {job_ids[0]}[/green]")
        if len(job_ids) > 1:
            console.print(
                f"  Dispatcher 1: [green]Job {job_ids[1]}[/green] "
                f"(depends on {job_ids[0]})"
            )
            console.print(
                f"  Remaining {len(flat_scripts) - 1} chunk(s) will be "
                f"auto-submitted as each completes"
            )
        if warning:
            console.print(f"  [yellow]Warning: {warning}[/yellow]")

        console.print(
            f"[green]Submitted {len(job_ids)} initial job(s) "
            f"(drip-feed dispatcher)[/green]\n"
        )

        # Wait if requested
        if self.config.wait:
            click.echo(
                "\nMonitoring progress (Ctrl+C to detach, jobs continue)..."
            )
            final_results = self._monitor_progress(output_dir, datasets)
        else:
            click.echo("\nJobs submitted. Monitor progress with:")
            click.echo("  squeue -u $USER --array")
            click.echo(f"  tail -f {output_dir}/processing_events.log")
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
