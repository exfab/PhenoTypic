"""The staged-GPU CLI flags are exposed and parse (Spec 1 §10, Plan 3 Task 3)."""

import click
import pytest
from click.testing import CliRunner

from phenotypic.phenotypicCLI import _parse_gpu_batch_size, phenotypic_cli


def test_gpu_flags_listed_in_help():
    runner = CliRunner()
    result = runner.invoke(phenotypic_cli, ["--help"])
    assert result.exit_code == 0
    for opt in (
        "--gpu-batch-size",
        "--gpu-workers-per-gpu",
        "--gpu-shards",
        "--gpu-slurm",
    ):
        assert opt in result.output


def test_gpu_batch_size_callback_accepts_int_and_auto():
    assert _parse_gpu_batch_size(None, None, "auto") == "auto"
    assert _parse_gpu_batch_size(None, None, "4") == 4


def test_gpu_batch_size_callback_rejects_garbage():
    with pytest.raises(click.BadParameter):
        _parse_gpu_batch_size(None, None, "banana")
