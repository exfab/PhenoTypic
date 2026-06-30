"""End-to-end: a default forward CLI run always emits the REMBI manifest
``deliverables/rembi.yaml`` (Task A6).

Asserts the REAL wiring CLI -> aggregate_measurements -> finalize ->
write_rembi_manifest folds the post-applied measurements MIRROR into the
manifest. Mirrors the smallest real run in ``test_cli_metadata_deliverable.py``
(``--force-local --skip-validation --njobs 1`` over the ``synth_plate_dir``
fixture). A sibling assertion confirms ``--mode process`` (which writes no
deliverables) emits no manifest.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from click.testing import CliRunner

from phenotypic.phenotypicCLI import phenotypic_cli
from phenotypic.sdk_ import rembi_manifest_path


def test_manifest_emitted_on_default_run(
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
    manifest = rembi_manifest_path(out)
    assert manifest.exists(), f"expected {manifest} to exist after a default run"
    data = yaml.safe_load(manifest.read_text())
    assert "image_data" in data
    assert data["image_data"]["n_images"] >= 1


def test_manifest_absent_under_process_mode(
    tmp_path: Path, synth_one_level_input: Path, simple_pipeline_json: Path
) -> None:
    out = tmp_path / "out"
    res = CliRunner().invoke(
        phenotypic_cli,
        [
            "--pipeline",
            str(simple_pipeline_json),
            "--input",
            str(synth_one_level_input),
            "--output",
            str(out),
            "--mode",
            "process",
            "--layer",
            "detect_mat",
            "--force-local",
            "--njobs",
            "1",
        ],
    )

    assert res.exit_code == 0, res.output
    # process mode writes no deliverables/, hence no manifest.
    assert not rembi_manifest_path(out).exists()
