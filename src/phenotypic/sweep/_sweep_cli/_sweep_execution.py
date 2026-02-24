"""Execution strategies for the sweep CLI.

Implements the Strategy pattern for local (joblib) and SLURM execution
of parameter sweep processing.
"""

from __future__ import annotations

import math
import re
import subprocess
import time
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import click

from ._sweep_output import SweepOutputManager
from ._sweep_process_image import (
    process_image_all_pipelines,
)

logger = logging.getLogger(__name__)


class SweepExecutionStrategy(ABC):
    """Base class for sweep execution strategies."""

    def __init__(
        self,
        pipeline_json_strs: Dict[str, str],
        image_type: Literal["Image", "GridImage"],
        read_kwargs: Dict[str, Any],
        output_manager: SweepOutputManager,
    ):
        self.pipeline_json_strs = pipeline_json_strs
        self.image_type = image_type
        self.read_kwargs = read_kwargs
        self.output_manager = output_manager

    @abstractmethod
    def execute(
        self,
        image_paths: List[Path],
        output_dir: Path,
    ) -> Dict[str, Any]:
        """Execute sweep processing.

        Args:
            image_paths: List of image file paths.
            output_dir: Base output directory.

        Returns:
            Results dict with ``total_images``, ``completed``, ``failed``,
            ``failures``, ``start_time``, ``end_time``.
        """


class LocalSweepStrategy(SweepExecutionStrategy):
    """Local execution: sequential over images, parallel over pipelines."""

    def __init__(
        self,
        pipeline_json_strs: Dict[str, str],
        image_type: Literal["Image", "GridImage"],
        read_kwargs: Dict[str, Any],
        output_manager: SweepOutputManager,
        n_jobs: int = -1,
        event_log: Optional[Path] = None,
        save_intermediates: bool = False,
    ):
        super().__init__(pipeline_json_strs, image_type, read_kwargs, output_manager)
        self.n_jobs = n_jobs
        self.event_log = event_log
        self.save_intermediates = save_intermediates

    def execute(
        self,
        image_paths: List[Path],
        output_dir: Path,
    ) -> Dict[str, Any]:
        start_time = datetime.now()
        total_images = len(image_paths)
        completed = 0
        failed = 0
        failures: List[Dict[str, str]] = []

        from rich.console import Console

        console = Console()
        console.print(
            f"\n[bold cyan]Sweep Processing[/bold cyan] "
            f"({total_images} images x {len(self.pipeline_json_strs)} pipelines)"
        )
        console.rule(style="cyan")

        for idx, image_path in enumerate(image_paths, 1):
            console.print(
                f"  [{idx}/{total_images}] {image_path.name} "
                f"({len(self.pipeline_json_strs)} pipelines)...",
                end="",
            )

            results = process_image_all_pipelines(
                image_path=image_path,
                pipeline_json_strs=self.pipeline_json_strs,
                image_type=self.image_type,
                read_kwargs=self.read_kwargs,
                output_manager=self.output_manager,
                n_jobs=self.n_jobs,
                save_intermediates=self.save_intermediates,
            )

            img_ok = sum(1 for _, ok, _ in results if ok)
            img_fail = sum(1 for _, ok, _ in results if not ok)

            if img_fail == 0:
                console.print(f" [green]OK[/green] ({img_ok}/{len(results)})")
                completed += 1
            else:
                console.print(
                    f" [yellow]partial[/yellow] ({img_ok} ok, {img_fail} failed)"
                )
                completed += 1  # image was processed, some pipelines failed
                for pipe_name, ok, tb in results:
                    if not ok:
                        failures.append({
                            "image": image_path.name,
                            "pipeline": pipe_name,
                            "traceback": tb,
                        })

            # Log per-pipeline events and update dashboard
            if self.event_log is not None:
                from phenotypic._cli._cli_update_state import (
                    append_completion_event,
                )
                from ._sweep_progress_dashboard import (
                    maybe_regenerate_dashboard,
                )

                for pipe_name, ok, tb in results:
                    event_id = f"{image_path.name}::{pipe_name}"
                    status = "completed" if ok else "failed"
                    error_msg = "" if ok else tb[:200]
                    append_completion_event(
                        event_log=self.event_log,
                        dataset="sweep",
                        image=event_id,
                        status=status,
                        error_msg=error_msg,
                    )
                maybe_regenerate_dashboard(output_dir, self.event_log)

        console.rule(style="cyan")
        end_time = datetime.now()

        return {
            "total_images": total_images,
            "completed": completed,
            "failed": failed,
            "failures": failures,
            "start_time": start_time,
            "end_time": end_time,
        }


class SLURMSweepStrategy(SweepExecutionStrategy):
    """SLURM execution: array job with one task per (image, pipeline) pair."""

    def __init__(
        self,
        pipeline_json_strs: Dict[str, str],
        image_type: Literal["Image", "GridImage"],
        read_kwargs: Dict[str, Any],
        output_manager: SweepOutputManager,
        manifest_path: Path,
        slurm_args: Dict[str, Any],
        wait: bool = False,
        verbose: bool = False,
        save_intermediates: bool = False,
    ):
        super().__init__(pipeline_json_strs, image_type, read_kwargs, output_manager)
        self.manifest_path = manifest_path
        self.slurm_args = slurm_args
        self.wait = wait
        self.verbose = verbose
        self.save_intermediates = save_intermediates

    def execute(
        self,
        image_paths: List[Path],
        output_dir: Path,
    ) -> Dict[str, Any]:
        start_time = datetime.now()
        total_images = len(image_paths)

        from rich.console import Console

        console = Console()
        console.print("\n[bold cyan]SLURM Sweep Submission[/bold cyan]")
        console.rule(style="cyan")
        console.print(
            f"  Images: {total_images}\n"
            f"  Pipelines: {len(self.pipeline_json_strs)}\n"
            f"  Total pipeline runs: {total_images * len(self.pipeline_json_strs)}"
        )

        # Generate chunked array job scripts (respects MaxArraySize)
        from phenotypic._cli._cli_slurm_config import (
            get_slurm_array_limit,
            get_slurm_max_submit_jobs,
        )
        from ._sweep_slurm_scripts import generate_sweep_array_scripts_chunked

        num_pipelines = len(self.pipeline_json_strs)
        total_tasks = total_images * num_pipelines
        array_limit = get_slurm_array_limit()
        max_submit = get_slurm_max_submit_jobs() or 50
        pipeline_names = list(self.pipeline_json_strs.keys())

        # Auto-calculate batch size so chunks fit within max_submit
        max_schedulable = array_limit * max_submit
        batch_size = max(1, math.ceil(total_tasks / max_schedulable))
        effective_tasks = math.ceil(total_tasks / batch_size)
        num_chunks = math.ceil(effective_tasks / array_limit)

        console.print(
            f"  SLURM array limit: {array_limit}\n"
            f"  MaxSubmitJobs: {max_submit}\n"
            f"  Batch size: {batch_size} pair(s)/task\n"
            f"  Effective array tasks: {effective_tasks}\n"
            f"  Array scripts needed: {num_chunks}"
        )

        # Warn if chunks still exceed the submission limit
        if num_chunks > max_submit:
            console.print(
                f"  [yellow]Warning: {num_chunks} chunks needed but "
                f"MaxSubmitJobs={max_submit}. Some submissions may be "
                f"rejected by the scheduler.[/yellow]"
            )

        script_paths = generate_sweep_array_scripts_chunked(
            image_paths=image_paths,
            pipeline_names=pipeline_names,
            manifest_path=self.manifest_path,
            output_dir=output_dir,
            image_type=self.image_type,
            read_kwargs=self.read_kwargs,
            slurm_args=self.slurm_args,
            array_limit=array_limit,
            verbose=self.verbose,
            batch_size=batch_size,
            save_intermediates=self.save_intermediates,
        )

        # Submit chunk scripts with dependency chain
        job_ids: List[str] = []
        failed_submissions: List[str] = []
        prev_job_id: Optional[str] = None

        for idx, script_path in enumerate(script_paths):
            dep_str = f", depends on {prev_job_id}" if prev_job_id else ""
            console.print(
                f"  [{idx + 1}/{len(script_paths)}] {script_path.name}{dep_str}",
                end="",
            )

            try:
                job_id = self._submit(
                    script_path, dependency_job_id=prev_job_id
                )
                job_ids.append(job_id)
                prev_job_id = job_id
                console.print(f" -> [green]Job {job_id}[/green]")
            except RuntimeError as e:
                console.print(f" -> [red]Failed: {e}[/red]")
                failed_submissions.append(str(script_path))
                # Don't update prev_job_id — next chunk can still try

        if failed_submissions:
            console.print(
                f"\n  [yellow]Warning: {len(failed_submissions)} chunk(s) "
                f"failed to submit[/yellow]"
            )

        if not job_ids:
            raise RuntimeError(
                "No SLURM jobs were submitted successfully. "
                "Check cluster configuration and sbatch availability."
            )

        console.print(
            f"\n  [green]Submitted {len(job_ids)} job(s) "
            f"with dependency chain[/green]"
        )

        if self.wait:
            console.print("\nMonitoring progress (Ctrl+C to detach)...")
            self._monitor(output_dir, total_tasks)
        else:
            console.print("\nJobs submitted. Monitor with:")
            console.print("  squeue -u $USER --array")
            console.print(f"  tail -f {output_dir / 'processing_events.log'}")

        end_time = datetime.now()
        return {
            "total_images": total_images,
            "completed": 0,
            "failed": 0,
            "failures": [],
            "start_time": start_time,
            "end_time": end_time,
            "job_ids": job_ids,
        }

    def _submit(
        self,
        script_path: Path,
        dependency_job_id: Optional[str] = None,
    ) -> str:
        """Submit array job via sbatch with optional dependency.

        Args:
            script_path: Path to the SLURM batch script.
            dependency_job_id: When set, adds
                ``--dependency=afterany:<id>`` so this chunk only starts
                after the previous one finishes.

        Returns:
            SLURM job ID string.

        Raises:
            RuntimeError: If sbatch is unavailable or submission fails.
        """
        cmd = ["sbatch"]
        if dependency_job_id:
            cmd.extend(["--dependency", f"afterany:{dependency_job_id}"])
        cmd.append(str(script_path))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "sbatch not found. SLURM not available. Use --force-local."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"sbatch submission timed out for script: {script_path.name}"
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"sbatch failed for {script_path.name}:\n{e.stderr}")

        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if not match:
            raise RuntimeError(
                f"Could not parse job ID from sbatch output:\n{result.stdout}"
            )
        return match.group(1)

    def _monitor(self, output_dir: Path, total_tasks: int) -> None:
        """Monitor progress via event log."""
        from phenotypic._cli._cli_update_state import aggregate_state_from_events
        from ._sweep_progress_dashboard import (
            generate_sweep_progress_dashboard,
            load_sweep_progress_metadata,
            maybe_regenerate_dashboard,
        )

        event_log = output_dir / "processing_events.log"
        last_count = 0

        try:
            while True:
                if event_log.exists():
                    states = aggregate_state_from_events(event_log)
                    total_done = sum(
                        len(ds.completed) + len(ds.failed)
                        for ds in states.values()
                    )
                    if total_done != last_count:
                        click.echo(
                            f"Progress: {total_done}/{total_tasks} tasks processed"
                        )
                        last_count = total_done

                    # Regenerate dashboard on each poll
                    maybe_regenerate_dashboard(output_dir, event_log)

                    if total_done >= total_tasks:
                        click.echo("All tasks processed!")
                        # Final dashboard with auto-refresh disabled
                        meta = load_sweep_progress_metadata(output_dir)
                        if meta is not None:
                            from datetime import datetime as dt

                            generate_sweep_progress_dashboard(
                                event_log=event_log,
                                output_path=(
                                    output_dir / "sweep_progress.html"
                                ),
                                total_tasks=total_tasks,
                                start_time=dt.fromisoformat(
                                    meta["start_time"]
                                ),
                                is_complete=True,
                            )
                        break

                time.sleep(10)
        except KeyboardInterrupt:
            click.echo("\nMonitoring stopped. Jobs continue running.")
