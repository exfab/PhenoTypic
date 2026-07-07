"""Integration regression for the ``.phenotypic`` machine-state layout.

Two end-to-end safety nets exercised via :class:`click.testing.CliRunner`:

* A forward run writes its machine-state (progress dir + state file) under
  the hidden ``<output>/.phenotypic/`` cache — not at the output root — while
  the user-facing ``deliverables/`` and ``results/`` dirs are unchanged.
* A run whose state was relocated to the legacy output root (pre-migration
  layout) still resumes: ``--resume`` migrates the legacy state into
  ``.phenotypic/`` and completes.

Both pass ``--force-local`` so SLURM is never dispatched.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from click.testing import CliRunner

from phenotypic.phenotypicCLI import phenotypic_cli
from phenotypic.sdk_ import processing_state_path, progress_dir


def test_forward_run_writes_state_under_phenotypic(
    tmp_path: Path, synth_plate_dir: Path, simple_pipeline_json: Path
) -> None:
    out = tmp_path / "out"
    res = CliRunner().invoke(
        phenotypic_cli,
        [
            "--pipeline",
            str(simple_pipeline_json),
            "--input",
            str(synth_plate_dir),
            "--output",
            str(out),
            "--force-local",
            "--skip-validation",
            "--njobs",
            "1",
        ],
    )
    assert res.exit_code == 0, res.output
    assert progress_dir(out).is_dir()
    assert not (out / "progress").exists()  # not at the root anymore
    assert (out / "deliverables").exists()  # user-facing dirs unchanged
    assert (out / "results").exists()


def test_resume_of_legacy_layout_migrates_and_completes(
    tmp_path: Path, synth_plate_dir: Path, simple_pipeline_json: Path
) -> None:
    out = tmp_path / "out"
    CliRunner().invoke(
        phenotypic_cli,
        [
            "--pipeline",
            str(simple_pipeline_json),
            "--input",
            str(synth_plate_dir),
            "--output",
            str(out),
            "--force-local",
            "--skip-validation",
            "--njobs",
            "1",
        ],
    )
    # Simulate a pre-migration layout by moving state back to the root.
    cache = out / ".phenotypic"
    (cache / "progress").rename(out / "progress")
    (cache / "processing_state.json").rename(out / "processing_state.json")
    events = cache / "processing_events.log"
    if events.exists():
        events.rename(out / "processing_events.log")
    for generated_dir in ("logs", "slurm_scripts"):
        leftover = cache / generated_dir
        if leftover.exists():
            shutil.rmtree(leftover)
    cache.rmdir()
    res = CliRunner().invoke(
        phenotypic_cli,
        [
            "--pipeline",
            str(simple_pipeline_json),
            "--input",
            str(synth_plate_dir),
            "--output",
            str(out),
            "--force-local",
            "--skip-validation",
            "--njobs",
            "1",
            "--resume",
        ],
    )
    assert res.exit_code == 0, res.output
    assert processing_state_path(out).is_file()  # migrated into .phenotypic
    assert not (out / "processing_state.json").exists()


def test_restart_wipes_stale_machine_state_and_keeps_outputs(
    tmp_path: Path, synth_plate_dir: Path, simple_pipeline_json: Path
) -> None:
    """``--restart`` clears the prior run's machine-state — so a stale event does
    not survive into the new run's records — while preserving output artifacts
    (unlike ``--overwrite``, which wipes the whole dir)."""
    from phenotypic.sdk_ import deliverables_dir, event_log_path

    out = tmp_path / "out"
    args = [
        "--pipeline",
        str(simple_pipeline_json),
        "--input",
        str(synth_plate_dir),
        "--output",
        str(out),
        "--force-local",
        "--skip-validation",
        "--njobs",
        "1",
    ]
    first = CliRunner().invoke(phenotypic_cli, args)
    assert first.exit_code == 0, first.output
    assert deliverables_dir(out).exists()

    # Inject a phantom stale event that must NOT survive a restart.
    elog = event_log_path(out)
    elog.write_text(
        elog.read_text(encoding="utf-8") + '{"image": "ghost_stale_image"}\n',
        encoding="utf-8",
    )

    res = CliRunner().invoke(phenotypic_cli, args + ["--restart"])
    assert res.exit_code == 0, res.output
    # The stale event log was wiped and rebuilt fresh — no phantom carried over.
    assert "ghost_stale_image" not in event_log_path(out).read_text(encoding="utf-8")
    # Output artifacts are regenerated/preserved (restart keeps outputs).
    assert deliverables_dir(out).exists()
