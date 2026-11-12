"""``processing_report.html`` is written on demand only (0.21.0).

``python -m phenotypic`` no longer writes the report in any mode: not
in-process for a local run, and not from a SLURM finalizer. The one way to get
it is ``python -m phenotypic.sdk_.generate_report <output>``.
"""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from phenotypic._cli._cli_update_state import append_event
from phenotypic.phenotypicCLI import phenotypic_cli
from phenotypic.sdk_ import (
    event_log_path,
    processing_report_html_path,
    readme_md_path,
)
from phenotypic.sdk_.generate_report import generate_cli_report


def test_a_local_run_writes_no_report_and_the_command_writes_one(
    tmp_path: Path,
    synth_one_level_input: Path,
    simple_pipeline_json: Path,
) -> None:
    output = tmp_path / "out"

    run = CliRunner().invoke(
        phenotypic_cli,
        [
            "--pipeline",
            str(simple_pipeline_json),
            "--input",
            str(synth_one_level_input),
            "--output",
            str(output),
            "--force-local",
            "--njobs",
            "1",
            "--skip-validation",
        ],
    )

    assert run.exit_code == 0, run.output
    assert "PROCESSING COMPLETE" in run.output
    assert "Generating HTML report" not in run.output
    assert readme_md_path(output).is_file()
    assert not processing_report_html_path(output).exists()

    on_demand = CliRunner().invoke(generate_cli_report, [str(output)])

    assert on_demand.exit_code == 0, on_demand.output
    report = processing_report_html_path(output).read_text(encoding="utf-8")
    assert "PhenoTypic Processing Report" in report
    assert "1/1 successful" in report


def test_the_command_reports_a_failure_from_the_event_log(tmp_path: Path) -> None:
    output = tmp_path / "out"
    log = event_log_path(output)
    append_event(log, "plates", "a.tif", "completed")
    append_event(
        log, "plates", "b.tif", "failed", error_msg="NoObjectsError: no colonies"
    )

    result = CliRunner().invoke(generate_cli_report, [str(output)])

    assert result.exit_code == 0, result.output
    assert "Completed: 1" in result.output
    assert "Failed: 1" in result.output
    report = processing_report_html_path(output).read_text(encoding="utf-8")
    assert "<td><strong>b.tif</strong></td>" in report
    assert "<code>NoObjectsError</code>" in report
