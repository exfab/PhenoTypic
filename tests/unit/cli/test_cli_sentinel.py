"""Focused tests for the deprecated sentinel's safe SLURM resubmission."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from phenotypic._cli._cli_sentinel import _submit_sentinel_script
from phenotypic.sdk_.slurm import SLURM_PYTHONPATH_ENV_VAR


def test_sentinel_resubmission_exports_namespaced_pythonpath(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A sentinel continuation should retain its exact Python source path."""
    script = tmp_path / "sentinel.sh"
    monkeypatch.setenv("PYTHONPATH", "/reviewed/sentinel/src")

    with patch(
        "phenotypic._cli._cli_sentinel.subprocess.run",
        return_value=subprocess.CompletedProcess([], 0, "801\n", ""),
    ) as run_mock:
        result = _submit_sentinel_script(script)

    assert result.stdout == "801\n"
    assert run_mock.call_args.args[0] == [
        "sbatch",
        "--parsable",
        "--export=ALL",
        str(script),
    ]
    environment = run_mock.call_args.kwargs["env"]
    assert environment[SLURM_PYTHONPATH_ENV_VAR] == "/reviewed/sentinel/src"
