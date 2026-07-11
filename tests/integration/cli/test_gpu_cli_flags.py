"""The staged-GPU CLI flags are exposed and parse (Spec 1 §10, Plan 3 Task 3)."""

from click.testing import CliRunner

from phenotypic.phenotypicCLI import phenotypic_cli


def test_gpu_flags_listed_in_help():
    runner = CliRunner()
    result = runner.invoke(phenotypic_cli, ["--help"])
    assert result.exit_code == 0
    for opt in (
        "--gpu-workers-per-gpu",
        "--gpu-shards",
        "--gpu-slurm",
    ):
        assert opt in result.output


def test_gpu_batch_size_flag_is_removed():
    """--gpu-batch-size was never wired into Stage 2; it must not be accepted.

    Batching is not widely supported by the segmentation models PhenoTypic
    targets, so the flag was removed rather than left as a silent no-op. Click
    must reject it outright instead of ignoring it.
    """
    runner = CliRunner()
    result = runner.invoke(phenotypic_cli, ["--help"])
    assert "--gpu-batch-size" not in result.output

    result = runner.invoke(
        phenotypic_cli, ["-o", "out", "--gpu-batch-size", "4"]
    )
    assert result.exit_code != 0
    assert "no such option" in result.output.lower()
