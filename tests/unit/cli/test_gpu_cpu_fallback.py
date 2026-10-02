"""A GPU pipeline on a machine (or partition) without a GPU runs on CPU.

The detectors' ``device="auto"`` falls back to CPU, so the CLI must not refuse
the run either: the local strategy reports the fallback instead of
``GPU detected: cpu``, and on SLURM an explicit ``slurm_gpus_per_node=0`` drops
the GPU request (SLURM rejects ``--gpus-per-node=0``) and skips the
GPU-partition refusal, as the run preflight already did.
"""

from __future__ import annotations

import pytest
from rich.console import Console

from phenotypic import ImagePipeline
from phenotypic._cli._cli_execution_strategies import (
    _report_gpu_pipeline_device,
    gpu_pipeline_slurm_args,
)
from phenotypic.sdk_.slurm import with_default_gpu_request
from phenotypic.sdk_.slurm import _config as slurm_config
from tests._fakes.fake_gpu_detector import FakeGpuDetector
from tests._fakes.fake_torch import install_fake_torch
from tests.unit.cli._preflight_support import make_context


def _recording_console() -> Console:
    return Console(record=True, width=200)


# --- local strategy device report ----------------------------------------------------


def test_no_accelerator_reports_a_cpu_run_rather_than_refusing(monkeypatch) -> None:
    install_fake_torch(monkeypatch)
    console = _recording_console()

    with pytest.warns(UserWarning, match="falling back to CPU"):
        device = _report_gpu_pipeline_device(console)

    text = console.export_text()
    assert device == "cpu"
    assert "will run on CPU" in text
    assert "GPU detected" not in text


def test_an_accelerator_is_reported_as_detected(monkeypatch) -> None:
    install_fake_torch(monkeypatch, cuda=True)
    console = _recording_console()

    assert _report_gpu_pipeline_device(console) == "cuda"
    assert "GPU detected: cuda" in console.export_text()


# --- with_default_gpu_request ---------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ({}, {"slurm_gpus_per_node": 1}),
        ({"slurm_gpus_per_node": 2}, {"slurm_gpus_per_node": 2}),
        ({"slurm_gpus_per_node": 0}, {}),
        ({"slurm_gpus_per_node": "0"}, {}),
    ],
)
def test_an_explicit_zero_removes_the_gpu_request(given, expected) -> None:
    assert with_default_gpu_request(given) == expected
    assert given == dict(given)  # the input is not mutated


# --- AutonomousSLURMStrategy's GPU profile --------------------------------------------


def _refusing_gres(calls: list[str]):
    def partition_gres_error(partition, run=None):
        calls.append(partition)
        return f"partition {partition!r} has no GPUs"

    return partition_gres_error


def test_a_gpu_request_on_a_gpu_less_partition_is_still_refused(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(slurm_config, "partition_gres_error", _refusing_gres(calls))

    with pytest.raises(RuntimeError, match="has no GPUs"):
        gpu_pipeline_slurm_args({"slurm_partition": "cpu"})
    assert calls == ["cpu"]


def test_an_explicit_zero_runs_on_a_cpu_partition(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(slurm_config, "partition_gres_error", _refusing_gres(calls))

    args = gpu_pipeline_slurm_args(
        {"slurm_partition": "cpu", "slurm_gpus_per_node": 0}
    )

    assert args == {"slurm_partition": "cpu"}
    assert calls == []


# --- run preflight agrees with submission ---------------------------------------------


def test_preflight_skips_the_partition_check_for_an_explicit_zero(monkeypatch) -> None:
    from phenotypic._cli import _cli_preflight

    monkeypatch.setattr(_cli_preflight, "_scheduler_available", lambda name: True)
    calls: list[str] = []
    monkeypatch.setattr(slurm_config, "partition_gres_error", _refusing_gres(calls))
    context = make_context(
        ImagePipeline(ops={"gpu": FakeGpuDetector()}),
        "process",
        slurm_args={"slurm_partition": "cpu", "slurm_gpus_per_node": 0},
        force_local=False,
    )

    assert _cli_preflight.check_gpu_partition(context) == []
    assert calls == []
