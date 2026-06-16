"""Public CLI contract tests for consolidated mode selection."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from phenotypic.phenotypicCLI import phenotypic_cli


def test_help_accepts_short_h_and_shows_new_public_flags() -> None:
    result = CliRunner().invoke(phenotypic_cli, ["-h"])

    assert result.exit_code == 0, result.output
    assert "-m, --mode" in result.output
    assert "-o, --output" in result.output
    assert "--njobs" in result.output
    assert "--output-dir" not in result.output
    assert "--n-jobs" not in result.output
    assert "--process-only" not in result.output
    assert "--measure" not in result.output
    assert "--recompile" not in result.output


def test_full_mode_requires_output(
    simple_pipeline_json: Path, synth_one_level_input: Path
) -> None:
    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode",
            "full",
            "--pipeline",
            str(simple_pipeline_json),
            "--input",
            str(synth_one_level_input),
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    assert "--output" in result.output


def test_old_public_mode_flags_are_removed() -> None:
    for flag in ("--measure", "--recompile", "--process-only"):
        result = CliRunner().invoke(phenotypic_cli, [flag])
        assert result.exit_code != 0
        assert "No such option" in result.output


def test_old_public_alias_flags_are_removed() -> None:
    for args in (["--output-dir", "out"], ["--n-jobs", "1"]):
        result = CliRunner().invoke(phenotypic_cli, args)
        assert result.exit_code != 0
        assert "No such option" in result.output


def test_process_mode_requires_layer(
    tmp_path: Path, simple_pipeline_json: Path, synth_one_level_input: Path
) -> None:
    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode",
            "process",
            "--pipeline",
            str(simple_pipeline_json),
            "--input",
            str(synth_one_level_input),
            "--output",
            str(tmp_path / "out"),
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    assert "--layer" in result.output


def test_measure_rejects_input_and_dry_run(
    tmp_path: Path, simple_pipeline_json: Path, synth_one_level_input: Path
) -> None:
    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode",
            "measure",
            "--pipeline",
            str(simple_pipeline_json),
            "--input",
            str(synth_one_level_input),
            "--output",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0
    assert "--input" in result.output
    assert "measure" in result.output.lower()

    dry_result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode",
            "measure",
            "--pipeline",
            str(simple_pipeline_json),
            "--output",
            str(tmp_path / "out"),
            "--dry-run",
        ],
    )
    assert dry_result.exit_code != 0
    assert "--dry-run" in dry_result.output
    assert "measure" in dry_result.output.lower()


def test_recompile_rejects_pipeline_input_and_dry_run(
    tmp_path: Path, simple_pipeline_json: Path, synth_one_level_input: Path
) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    for flag, value in (
        ("--pipeline", str(simple_pipeline_json)),
        ("--input", str(synth_one_level_input)),
    ):
        result = CliRunner().invoke(
            phenotypic_cli,
            [
                "--mode",
                "recompile",
                "--output",
                str(output_dir),
                flag,
                value,
            ],
        )
        assert result.exit_code != 0
        assert flag in result.output
        assert "recompile" in result.output.lower()

    dry_result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode",
            "recompile",
            "--output",
            str(output_dir),
            "--dry-run",
        ],
    )
    assert dry_result.exit_code != 0
    assert "--dry-run" in dry_result.output
    assert "recompile" in dry_result.output.lower()
