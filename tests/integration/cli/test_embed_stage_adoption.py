"""U12: a continuation under ``--embed-stage auto`` adopts the run's recorded stage (#282 P5 review I1).

An in-flight pre-P5 SLURM run with a ``GpuDetector`` and a DINO embedder
recorded ``embed_stage: "inline"`` (P4's interim gate), or recorded nothing
(a #281 state, read as inline). After the upgrade, ``auto`` resolves ``gpu``
for that shape; the run's documented recovery, the plain re-run, must still
continue it inline, while an explicit conflicting ``--embed-stage gpu`` is
still refused (R3). The submitter is stubbed, so nothing is queued and no
backbone is loaded.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from phenotypic.sdk_ import RunExitCode, resolve_processing_state_path
from tests._tiny_run import run_cli, with_stage4_dino_objects


@pytest.fixture
def stub_submitter(monkeypatch):
    """``create_execution_strategy`` returning a strategy that submits nothing; records the config."""
    from phenotypic._cli._cli_types import ExecutionResults

    seen: list = []

    class _Submitted:
        def execute(self, datasets, output_dir):
            now = datetime.now()
            return ExecutionResults(datasets={}, total_images=2, total_completed=0, total_failed=0,
                                    execution_mode="slurm", start_time=now, end_time=now, submitted=True)

    def _strategy(config, output_manager):
        seen.append(config)
        return _Submitted()

    monkeypatch.setattr("phenotypic.phenotypicCLI.create_execution_strategy", _strategy)
    return seen


def _forget_embed_stage(out) -> None:
    """Rewrite the state as #281 wrote it: no ``embed_stage`` key at all."""
    path = resolve_processing_state_path(out)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["config"].pop("embed_stage")
    path.write_text(json.dumps(raw), encoding="utf-8")


@pytest.mark.parametrize("recorded", ["inline", None], ids=["recorded-inline", "recorded-nothing"])
def test_a_pre_p5_slurm_detector_dino_continuation_stays_inline(tmp_path, tiny_inputs, stub_submitter, recorded):
    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path, fake_gpu=True)
    first = run_cli(tiny_inputs, out, pipeline=pipe, slurm=True,
                    extra=["--skip-validation", "--embed-stage", "inline"])
    assert first.exit_code == RunExitCode.SUBMITTED, first.output
    if recorded is None:
        _forget_embed_stage(out)

    again = run_cli(tiny_inputs, out, pipeline=pipe, slurm=True, extra=["--skip-validation"])
    assert again.exit_code == RunExitCode.SUBMITTED, again.output
    assert stub_submitter[-1].embed_stage == "inline"
    assert "--embed-stage auto continues it as inline" in again.output


def test_an_explicit_conflicting_flag_is_still_refused(tmp_path, tiny_inputs, stub_submitter):
    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path, fake_gpu=True)
    first = run_cli(tiny_inputs, out, pipeline=pipe, slurm=True,
                    extra=["--skip-validation", "--embed-stage", "inline"])
    assert first.exit_code == RunExitCode.SUBMITTED, first.output

    refused = run_cli(tiny_inputs, out, pipeline=pipe, slurm=True,
                      extra=["--skip-validation", "--embed-stage", "gpu"])
    assert refused.exit_code == 2, refused.output
    assert "started with --embed-stage inline" in refused.output
    assert len(stub_submitter) == 1              # the refused continuation submitted nothing
