from pathlib import Path

import pytest
from click.testing import CliRunner

from phenotypic._cli._cli_failure_tracker import read_terminal_failures
from phenotypic._cli._cli_process_single import main


def test_process_only_option_parses(tmp_path, monkeypatch):
    called = {}

    def fake_core(**kwargs):
        called.update(kwargs)
        return True

    monkeypatch.setattr(
        "phenotypic._cli._cli_process_single.process_single_apply_only_core", fake_core
    )
    monkeypatch.setattr(
        "phenotypic._cli._cli_process_single.publish_image_success",
        lambda *args, **kwargs: None,
    )
    pipe = tmp_path / "p.json"
    pipe.write_text("{}", encoding="utf-8")
    img = tmp_path / "in" / "a.tif"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"x")
    res = CliRunner().invoke(
        main,
        [
            "--pipeline", str(pipe),
            "--image", str(img),
            "--output-dir", str(tmp_path / "out"),
            "--dataset-name", "in",
            "--mode", "process",
            "--layer", "detect_mat",
            "--input-root", str(tmp_path / "in"),
        ],
    )
    assert res.exit_code == 0, res.output
    assert called["layer"] == "detect_mat"
    assert str(called["input_root"]) == str(tmp_path / "in")


def test_process_only_requires_input_root(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "phenotypic._cli._cli_process_single.process_single_apply_only_core",
        lambda **kwargs: True,
    )
    pipe = tmp_path / "p.json"
    pipe.write_text("{}", encoding="utf-8")
    img = tmp_path / "in" / "a.tif"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"x")
    res = CliRunner().invoke(
        main,
        [
            "--pipeline", str(pipe),
            "--image", str(img),
            "--output-dir", str(tmp_path / "out"),
            "--dataset-name", "in",
            "--mode", "process",
            "--layer", "rgb",
        ],
    )
    assert res.exit_code != 0
    assert "input-root" in res.output.lower()


def _capture_identity_flags(tmp_path, monkeypatch, extra_args):
    """Invoke the worker and capture the flags it digests into its work id."""
    captured = {}

    def fake_identity(**kwargs):
        captured.update(kwargs)
        return ("work-id", "a.tif")

    monkeypatch.setattr(
        "phenotypic._cli._cli_process_single._worker_work_identity",
        fake_identity,
    )
    monkeypatch.setattr(
        "phenotypic._cli._cli_process_single.process_single_apply_only_core",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        "phenotypic._cli._cli_process_single.publish_image_success",
        lambda *args, **kwargs: None,
    )
    pipe = tmp_path / "p.json"
    pipe.write_text("{}", encoding="utf-8")
    img = tmp_path / "in" / "a.tif"
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"x")
    res = CliRunner().invoke(
        main,
        [
            "--pipeline", str(pipe),
            "--image", str(img),
            "--output-dir", str(tmp_path / "out"),
            "--dataset-name", "in",
            "--mode", "process",
            "--layer", "detect_mat",
            "--input-root", str(tmp_path / "in"),
            *extra_args,
        ],
    )
    assert res.exit_code == 0, res.output
    return captured


def test_dataset_column_included_when_flag_absent(tmp_path, monkeypatch):
    """Default must match the top-level CLI's ``not no_dataset_column``.

    When the two disagree the worker digests a different work id than
    selection did, and every SLURM array task dies with "work identity does
    not match worklist".
    """
    captured = _capture_identity_flags(tmp_path, monkeypatch, [])
    assert captured["include_dataset_column"] is True


def test_dataset_column_excluded_when_flag_present(tmp_path, monkeypatch):
    """``--no-dataset-column`` is the only thing that turns the column off."""
    captured = _capture_identity_flags(
        tmp_path, monkeypatch, ["--no-dataset-column"]
    )
    assert captured["include_dataset_column"] is False


def test_process_worker_terminalizes_provenance_initialization_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    simple_pipeline_json: Path,
    synth_one_level_input: Path,
) -> None:
    """The process wrapper must reach the worker's durable classification."""
    from phenotypic._cli import _cli_process_only

    original = RuntimeError("provenance initialization failed")

    def _fail_initialization(*args, **kwargs):
        del args, kwargs
        raise original

    monkeypatch.setattr(
        _cli_process_only, "initialize_cli_provenance", _fail_initialization
    )
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    image = next(synth_one_level_input.rglob("*.tif"))
    output_dir = tmp_path / "out"

    result = CliRunner().invoke(
        main,
        [
            "--pipeline",
            str(simple_pipeline_json),
            "--image",
            str(image),
            "--output-dir",
            str(output_dir),
            "--dataset-name",
            "ds",
            "--image-type",
            "Image",
            "--mode",
            "process",
            "--layer",
            "rgb",
            "--input-root",
            str(synth_one_level_input),
        ],
    )

    assert result.exit_code == 1
    records = read_terminal_failures(output_dir)
    assert len(records) == 1
    record = records[0]
    assert record.failed_stage == "process"
    assert record.exception_type == "RuntimeError"
    assert record.exception_message == str(original)
    assert record.slurm_job_id == "12345"
