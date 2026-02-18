"""Execution strategies for the sweep CLI.

Implements the Strategy pattern for local (joblib) and SLURM execution
of parameter sweep processing.
"""

from __future__ import annotations

import re
import subprocess
import time
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import click

from ._sweep_output import SweepOutputManager
from ._sweep_process_image import (
    process_image_all_pipelines,
    process_image_all_pipelines_sequential,
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
    ):
        super().__init__(pipeline_json_strs, image_type, read_kwargs, output_manager)
        self.n_jobs = n_jobs

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
        save_layers: Optional[Dict[str, bool]] = None,
        overlay_mode: str = "image",
        overlay_alpha: float = 0.3,
    ):
        super().__init__(pipeline_json_strs, image_type, read_kwargs, output_manager)
        self.manifest_path = manifest_path
        self.slurm_args = slurm_args
        self.wait = wait
        self.save_layers = save_layers or {}
        self.overlay_mode = overlay_mode
        self.overlay_alpha = overlay_alpha

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
        from phenotypic._cli._cli_slurm_config import get_slurm_array_limit
        from ._sweep_slurm_scripts import generate_sweep_array_scripts_chunked

        num_pipelines = len(self.pipeline_json_strs)
        total_tasks = total_images * num_pipelines
        array_limit = get_slurm_array_limit()
        pipeline_names = list(self.pipeline_json_strs.keys())

        console.print(
            f"  SLURM array limit: {array_limit}\n"
            f"  Array scripts needed: "
            f"{(total_tasks + array_limit - 1) // array_limit}"
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
            save_layers=self.save_layers,
            overlay_mode=self.overlay_mode,
            overlay_alpha=self.overlay_alpha,
        )

        # Submit all chunk scripts independently
        job_ids: List[str] = []
        for script_path in script_paths:
            console.print(f"\n  Script: {script_path}")
            job_id = self._submit(script_path)
            job_ids.append(job_id)
            console.print(f"  [green]Submitted job {job_id}[/green]")

        if self.wait:
            console.print("\nMonitoring progress (Ctrl+C to detach)...")
            self._monitor(output_dir, total_tasks)
        else:
            console.print("\nJobs submitted. Monitor with:")
            console.print(f"  squeue -u $USER --array")
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

    def _submit(self, script_path: Path) -> str:
        """Submit array job via sbatch."""
        try:
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "sbatch not found. SLURM not available. Use --force-local."
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"sbatch failed:\n{e.stderr}")

        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if not match:
            raise RuntimeError(
                f"Could not parse job ID from sbatch output:\n{result.stdout}"
            )
        return match.group(1)

    def _monitor(self, output_dir: Path, total_tasks: int) -> None:
        """Monitor progress via event log."""
        from phenotypic._cli._cli_update_state import aggregate_state_from_events

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

                    if total_done >= total_tasks:
                        click.echo("All tasks processed!")
                        break

                time.sleep(10)
        except KeyboardInterrupt:
            click.echo("\nMonitoring stopped. Jobs continue running.")
