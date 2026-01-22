"""
SLURM job progress monitor.

This script monitors SLURM job progress with live updates by reading
the event log and displaying real-time statistics.

Usage:
    python -m phenotypic.tools_.monitor_slurm_jobs OUTPUT_DIR
"""

import json
import sys
import time
from pathlib import Path

import click

from phenotypic._cli._cli_update_state import aggregate_state_from_events


@click.command()
@click.argument("output_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
        "--refresh-interval",
        default=30,
        show_default=True,
        help="Seconds between updates",
)
@click.option(
        "--no-clear",
        is_flag=True,
        help="Don't clear screen between updates (useful for logging)",
)
def main(output_dir: Path, refresh_interval: int, no_clear: bool):
    """
    Monitor SLURM job progress with live updates.

    OUTPUT_DIR: Directory containing processing_events.log

    This tool polls the event log periodically and displays processing
    progress. You can detach at any time with Ctrl+C - jobs will continue
    running on SLURM.

    Examples:
        # Monitor with default 30s refresh
        python -m phenotypic.tools_.monitor_slurm_jobs ./results

        # Monitor with faster refresh
        python -m phenotypic.tools_.monitor_slurm_jobs ./results --refresh-interval 10

        # Monitor without clearing screen (good for logging)
        python -m phenotypic.tools_.monitor_slurm_jobs ./results --no-clear
    """
    event_log = output_dir / "processing_events.log"
    state_file = output_dir / "processing_state.json"

    # Validate files exist
    if not state_file.exists():
        click.echo("Error: No processing_state.json found", err=True)
        click.echo(
                "This directory does not appear to contain active processing.",
                err=True,
        )
        sys.exit(1)

    # Load initial state to get total count
    try:
        config = json.loads(state_file.read_text())
        expected_datasets = config.get("datasets", {})
    except Exception as e:
        click.echo(f"Error reading state file: {e}", err=True)
        sys.exit(1)

    click.echo("PhenoTypic SLURM Job Monitor")
    click.echo("=" * 60)
    click.echo("Press Ctrl+C to exit (jobs will continue)\n")

    last_total_completed = 0

    try:
        while True:
            # Aggregate latest events
            if event_log.exists():
                datasets_state = aggregate_state_from_events(event_log)
            else:
                datasets_state = {}

            # Calculate totals
            total_completed = sum(
                    len(ds.completed) for ds in datasets_state.values()
            )
            total_failed = sum(
                    len(ds.failed) for ds in datasets_state.values()
            )
            total_processed = total_completed + total_failed

            # Calculate expected total from state file
            expected_total = sum(
                    ds.get("total", 0) for ds in expected_datasets.values()
            )
            if expected_total == 0:
                # Fallback: count from current state
                expected_total = total_processed

            remaining = max(0, expected_total - total_processed)

            # Clear screen if requested
            if not no_clear:
                click.clear()

            # Display header
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            click.echo(f"=== SLURM Processing Progress ===")
            click.echo(f"Last updated: {timestamp}\n")

            # Overall progress
            if expected_total > 0:
                progress_pct = (total_processed / expected_total) * 100
                progress_bar = _create_progress_bar(
                        total_processed, expected_total, width=50
                )
                click.echo(f"Overall Progress: {progress_bar}")
                click.echo(
                        f"  {total_processed}/{expected_total} "
                        f"({progress_pct:.1f}%)\n"
                )

            # Display per-dataset progress
            if datasets_state:
                click.echo("Dataset Status:")
                click.echo("-" * 60)

                for dataset_name, ds_state in sorted(
                        datasets_state.items()
                ):
                    completed = len(ds_state.completed)
                    failed = len(ds_state.failed)
                    total_ds = completed + failed

                    # Get expected total for this dataset
                    expected_ds = expected_datasets.get(dataset_name, {}).get(
                            "total", total_ds
                    )

                    click.echo(f"\n  {dataset_name}:")
                    click.echo(f"    Completed: {completed}")
                    click.echo(f"    Failed:    {failed}")

                    if expected_ds > 0:
                        remaining_ds = max(0, expected_ds - total_ds)
                        click.echo(f"    Remaining: {remaining_ds}")

                        progress_pct = (completed / expected_ds) * 100
                        progress_bar = _create_progress_bar(
                                completed, expected_ds, width=30
                        )
                        click.echo(
                                f"    Progress:  {progress_bar} "
                                f"{progress_pct:.1f}%"
                        )
            else:
                click.echo("Waiting for processing events...")

            # Summary
            click.echo("\n" + "-" * 60)
            click.echo(f"Total Completed: {total_completed}")
            click.echo(f"Total Failed:    {total_failed}")
            click.echo(f"Remaining:       {remaining}")

            # Check if all done
            if remaining == 0 and expected_total > 0:
                click.echo("\n✓ All jobs complete!")
                click.echo(
                        f"\nFinal Results:"
                )
                click.echo(
                        f"  Success rate: "
                        f"{(total_completed / expected_total) * 100:.1f}%"
                )
                click.echo(f"\nGenerate report with:")
                click.echo(
                        f"  python -m phenotypic.tools_.generate_report "
                        f"{output_dir}"
                )
                break

            # Show new completions since last update
            new_completions = total_completed - last_total_completed
            if new_completions > 0:
                click.echo(
                        f"\n({new_completions} new since last update)"
                )

            last_total_completed = total_completed

            # Wait before next update
            click.echo(f"\nRefreshing in {refresh_interval}s...")
            time.sleep(refresh_interval)

    except KeyboardInterrupt:
        click.echo(
                "\n\nMonitoring stopped. Jobs continue running on SLURM."
        )
        click.echo(
                f"\nResume monitoring with:"
        )
        click.echo(
                f"  python -m phenotypic.tools_.monitor_slurm_jobs {output_dir}"
        )
        return 0

    return 0


def _create_progress_bar(current: int, total: int, width: int = 30) -> str:
    """
    Create a text-based progress bar.

    Args:
        current: Current progress value
        total: Total value
        width: Width of progress bar in characters

    Returns:
        Progress bar string like "[=====>     ]"
    """
    if total == 0:
        filled = 0
    else:
        filled = int(width * current / total)

    bar = "=" * filled + ">" if filled < width else "=" * width
    empty = " " * (width - filled - (1 if filled < width else 0))

    return f"[{bar}{empty}]"


if __name__ == "__main__":
    sys.exit(main())
