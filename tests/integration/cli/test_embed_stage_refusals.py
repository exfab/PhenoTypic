"""``--embed-stage`` refusals that need a real run on the stub backbone (#282 P4 T1). [torch]

Run in the foundation venv (``TPT``); in the shared env they skip.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from tests._tiny_run import run_cli, with_stage4_dino_objects  # noqa: E402


def test_r3_refuses_a_switched_continuation(tmp_path, tiny_inputs, stage4_cpu):
    out = tmp_path / "out"
    pipe = with_stage4_dino_objects(tmp_path)
    first = run_cli(tiny_inputs, out, pipeline=pipe)                         # local auto: inline, recorded
    assert first.exit_code == 0, first.output
    result = run_cli(tiny_inputs, out, pipeline=pipe, extra=["--embed-stage", "gpu"])
    assert result.exit_code == 2 and "started with --embed-stage inline" in result.output, result.output


def test_no_cuda_stops_before_stage1(tmp_path, tiny_inputs, param_stub_backbone, monkeypatch):
    from phenotypic._cli._cli_failure_tracker import read_terminal_failures

    monkeypatch.delenv("PHENOTYPIC_STAGE4_DEVICE", raising=False)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    out = tmp_path / "out"
    result = run_cli(tiny_inputs, out, pipeline=with_stage4_dino_objects(tmp_path), extra=["--embed-stage", "gpu"])
    assert result.exit_code == 1 and "CUDA" in result.output, result.output
    errors = [line for line in result.output.splitlines() if line.startswith("Error:")]
    assert len(errors) == 1, result.output
    assert not list(out.rglob("*.ome.zarr")) and read_terminal_failures(out) == []
