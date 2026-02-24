"""
Standalone HTML report generator tool.

This script generates HTML processing reports from a PhenoTypic output directory.
Useful for generating reports on-demand after processing completes or during
long-running SLURM jobs.

Usage:
    python -m phenotypic.tools_.generate_report OUTPUT_DIR
"""

import sys
from datetime import datetime
from pathlib import Path

import click

from phenotypic._cli._cli_report_generator import HTMLReportGenerator
from phenotypic._cli._cli_types import (
    DatasetResults,
    ExecutionResults,
    ImageFailure,
)
from phenotypic._cli._cli_update_state import aggregate_state_from_events


@click.command()
@click.argument("output_dir", type=click.Path(exists=True, path_type=Path))
def generate_cli_report(output_dir: Path):
    """
    Generate HTML processing report from output directory.

    OUTPUT_DIR: Directory containing processing_events.log and processing_state.json

    This tool reads the event log and generates a visual HTML report showing
    processing results, success rates, and failure details with tracebacks.

    Examples:
        # Generate report after processing completes
        python -m phenotypic.tools_.generate_report ./results

        # Generate report during long-running SLURM job
        python -m phenotypic.tools_.generate_report ./phenotypic_results_20260108_143022
    """
    click.echo(f"Generating report for {output_dir}...")

    # Check for event log
    event_log = output_dir / "processing_events.log"
    if not event_log.exists():
        click.echo("Error: No processing_events.log found", err=True)
        click.echo(
                "This directory does not appear to contain PhenoTypic processing results.",
                err=True,
        )
        sys.exit(1)

    # Aggregate state from event log
    click.echo("Aggregating processing events...")
    datasets_state = aggregate_state_from_events(event_log)

    if not datasets_state:
        click.echo("Warning: No events found in log", err=True)
        click.echo("Processing may not have started yet.", err=True)
        sys.exit(1)

    # Convert to ExecutionResults format
    dataset_results = {}
    total_completed = 0
    total_failed = 0
    total_images = 0

    for dataset_name, ds_state in datasets_state.items():
        completed = len(ds_state.completed)
        failed = len(ds_state.failed)
        total = completed + failed

        total_completed += completed
        total_failed += failed
        total_images += total

        # Create ImageFailure objects from errors
        failures = []
        for img_name in ds_state.failed:
            error_msg = ds_state.errors.get(img_name, "Unknown error")

            # Parse error type from message if possible
            if ":" in error_msg:
                parts = error_msg.split(":", 1)
                error_type = parts[0].strip()
                error_message = parts[1].strip()
            else:
                error_type = "Exception"
                error_message = error_msg

            failures.append(
                    ImageFailure(
                            dataset=dataset_name,
                            image_filename=img_name,
                            error_type=error_type,
                            error_message=error_message,
                            traceback=error_msg,  # Full message as traceback
                            timestamp=datetime.now(),
                    )
            )

        dataset_results[dataset_name] = DatasetResults(
                name=dataset_name,
                total=total,
                completed=completed,
                failed=failed,
                failures=failures,
        )

    # Create ExecutionResults
    # Try to read timestamps from state file if it exists
    state_file = output_dir / "processing_state.json"
    if state_file.exists():
        import json

        state_dict = json.loads(state_file.read_text())
        start_time = datetime.fromisoformat(state_dict["timestamp"])
        end_time = datetime.now()
        execution_mode = state_dict.get("execution_mode", "local")
    else:
        start_time = datetime.now()
        end_time = datetime.now()
        execution_mode = "local"

    results = ExecutionResults(
            datasets=dataset_results,
            total_images=total_images,
            total_completed=total_completed,
            total_failed=total_failed,
            execution_mode=execution_mode,
            start_time=start_time,
            end_time=end_time,
    )

    # Generate report
    click.echo("Generating HTML report...")
    generator = HTMLReportGenerator()
    report_path = output_dir / "processing_report.html"
    generator.generate_report(results, report_path)

    click.echo("\n✓ Report generated successfully!")
    click.echo(f"  Location: {report_path}")
    click.echo(f"  Total images: {total_images}")
    click.echo(f"  Completed: {total_completed}")
    click.echo(f"  Failed: {total_failed}")
    click.echo(
            f"  Success rate: {results.success_rate * 100:.1f}%"
    )

    return 0


if __name__ == "__main__":
    sys.exit(generate_cli_report())
