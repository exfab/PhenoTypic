"""Safe CLI boundary tests for GUI-local completion publication."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest
from click.testing import CliRunner

from phenotypic._cli._cli_types import ExecutionResults
from phenotypic._cli._cli_update_state import PROCESSING_GENERATION_ENV_VAR
from phenotypic.phenotypicCLI import phenotypic_cli
from phenotypic.sdk_ import (
    atomic_write_json,
    manifest_json_path,
    run_completion_marker_path,
)
from phenotypic.sdk_._io_constants import GUI_RECORD_GENERATION_ENV_VAR


@pytest.mark.parametrize("process_only", [False, True])
@pytest.mark.parametrize(
    "existing_processing_generation",
    [None, "caller-owned-generation"],
)
def test_gui_slurm_shaped_run_never_publishes_local_completion(
    tmp_path: Path,
    synth_one_level_input: Path,
    simple_pipeline_json: Path,
    monkeypatch: pytest.MonkeyPatch,
    process_only: bool,
    existing_processing_generation: str | None,
) -> None:
    """Both CLI local-marker call sites stay unreachable in SLURM mode."""
    output = tmp_path / "output"
    generation = uuid4()
    monkeypatch.setenv(GUI_RECORD_GENERATION_ENV_VAR, str(generation))
    if existing_processing_generation is None:
        monkeypatch.delenv(PROCESSING_GENERATION_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(
            PROCESSING_GENERATION_ENV_VAR,
            existing_processing_generation,
        )
    scheduler_invocations: list[object] = []

    class InjectedSlurmShapedStrategy:
        def execute(self, _datasets, output_dir):
            now = datetime.now()
            path = manifest_json_path(output_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Make an erroneous local-publisher call capable of succeeding;
            # the test therefore proves the mode gate, not a helper failure.
            atomic_write_json(
                path,
                {
                    "execution_mode": "local",
                    "gui_record_generation": str(generation),
                    "is_complete": True,
                    "completed": 0,
                    "failed": 0,
                    "total_images": 0,
                },
            )
            return ExecutionResults(
                datasets={},
                total_images=0,
                total_completed=0,
                total_failed=0,
                execution_mode="slurm",
                start_time=now,
                end_time=now,
                remote_managed=False,
            )

    def _injected_strategy(config, _output_manager):
        assert config.is_slurm_mode()
        return InjectedSlurmShapedStrategy()

    def _forbid_scheduler_submission(*args, **kwargs):
        scheduler_invocations.append((args, kwargs))
        raise AssertionError("scheduler submission must not run")

    monkeypatch.setattr(
        "phenotypic.phenotypicCLI.create_execution_strategy",
        _injected_strategy,
    )
    monkeypatch.setattr(
        "phenotypic.phenotypicCLI.submit_slurm_script_chain",
        _forbid_scheduler_submission,
    )
    args = [
        "--pipeline",
        str(simple_pipeline_json),
        "--input",
        str(synth_one_level_input),
        "--output",
        str(output),
        "--slurm",
        "slurm_partition=test",
        "--skip-validation",
    ]
    if process_only:
        args.extend(["--mode", "process", "--layer", "detect_mat"])

    result = CliRunner().invoke(phenotypic_cli, args)

    assert result.exit_code == 0, result.output
    assert scheduler_invocations == []
    assert not run_completion_marker_path(output).exists()
    if existing_processing_generation is None:
        assert PROCESSING_GENERATION_ENV_VAR not in os.environ
    else:
        assert (
            os.environ[PROCESSING_GENERATION_ENV_VAR]
            == existing_processing_generation
        )
