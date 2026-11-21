"""``--embed-stage``: its resolution, refusals R1-R3, and the Stage-4 environment (#282 P4 T1).

Spec ``docs/superpowers/specs/2026-11-12-gpu-embedding-staging/design.md``
§4.1, §4.7, U2, U3, U9, O3, O5. Torch-free: nothing here loads a model, and
a test that needs torch's answers puts a fake ``torch`` in ``sys.modules``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import click
import pytest

from phenotypic import ImagePipeline
from phenotypic._cli._cli_embed_stage import (
    prepare_stage4_environment,
    refuse_embed_stage_for_process,
    refuse_embed_stage_switch,
    refuse_unstageable_embedders,
    resolve_embed_stage,
    resolve_embed_stage_for_config,
    stage4_supported,
)
from phenotypic._cli._cli_failure_tracker import processing_configuration_digest
from phenotypic._cli._cli_stage4_worker import has_kernels_for, stage4_device
from phenotypic.measure import MeasureDinoEmbedding, MeasureSize

CUBLAS = "CUBLAS_WORKSPACE_CONFIG"


@pytest.mark.parametrize("requested, mode, slurm, detector, gpu_slurm, embedders, cpu_pinned, expected", [
    ("auto", "full", True, True, False, True, False, "gpu"),
    ("auto", "full", True, False, True, True, False, "gpu"),
    ("auto", "full", True, False, False, True, False, "inline"),
    ("auto", "full", False, True, True, True, False, "inline"),
    ("gpu", "full", False, False, False, True, False, "gpu"),
    ("auto", "measure", True, True, False, True, False, "inline"),   # only --gpu-slurm counts
    ("auto", "measure", True, False, True, True, False, "gpu"),
    ("auto", "process", True, True, True, True, False, "inline"),
    ("gpu", "full", True, True, True, False, False, "inline"),       # O5
    ("auto", "full", True, True, True, True, True, "inline"),        # U9
    ("inline", "full", True, True, True, True, False, "inline"),
])
def test_resolution_table(requested, mode, slurm, detector, gpu_slurm, embedders, cpu_pinned, expected):
    assert resolve_embed_stage(requested, mode=mode, is_slurm=slurm, has_gpu_detector=detector,
                               gpu_slurm_given=gpu_slurm, has_embedders=embedders,
                               cpu_pinned=cpu_pinned) == expected


def test_r1_refuses_gpu_in_process_mode():
    with pytest.raises(click.UsageError, match="does not apply to --mode process"):
        refuse_embed_stage_for_process("gpu", "process")
    refuse_embed_stage_for_process("auto", "process")
    refuse_embed_stage_for_process("gpu", "full")


def test_r2_refuses_a_cpu_pin_only_with_an_explicit_gpu():
    pipe = ImagePipeline(meas={"emb": MeasureDinoEmbedding(device="cpu")})
    with pytest.raises(click.UsageError, match=r"meas 'emb' \(MeasureDinoEmbedding\) is pinned to device='cpu'"):
        refuse_unstageable_embedders(pipe, "gpu", "gpu")
    refuse_unstageable_embedders(pipe, "auto", "inline")       # U9: auto resolved inline, accepted


def test_r2_refuses_a_non_dino_embedder_under_gpu(toy_embedder):
    """O3 (amended 2026-11-20): only an explicit ``--embed-stage gpu`` is refused."""
    pipe = ImagePipeline(meas={"emb": toy_embedder})
    cls = type(toy_embedder).__name__
    with pytest.raises(click.UsageError, match=(
        rf"^--embed-stage gpu was given, but meas 'emb' \({cls}\) has no GPU Stage-4 "
        r"encoder; pass --embed-stage inline$"
    )):
        refuse_unstageable_embedders(pipe, "gpu", "gpu")
    refuse_unstageable_embedders(pipe, "auto", "inline")       # auto resolved inline, accepted


@pytest.mark.parametrize("mode, gpu_detector, gpu_slurm", [
    ("full", True, False), ("full", False, True), ("measure", False, True),
], ids=["detector", "gpu-slurm", "measure-gpu-slurm"])
def test_auto_resolves_inline_for_a_non_dino_embedder(make_exec_config, tmp_path, toy_embedder,
                                                       mode, gpu_detector, gpu_slurm):
    """O3 as U9: a run that would be gpu under auto is inline, with one note naming the embedder."""
    from tests._fakes.fake_gpu_detector import FakeGpuDetector

    ops = [FakeGpuDetector()] if gpu_detector else []
    pipe = ImagePipeline(ops=ops, meas={"dino": MeasureDinoEmbedding(), "toy": toy_embedder})
    config = _slurm_config(make_exec_config, tmp_path, measure_only=mode == "measure",
                           gpu_slurm_args={"slurm_partition": "exfab"} if gpu_slurm else {})
    cls = type(toy_embedder).__name__
    assert resolve_embed_stage_for_config(config, pipe) == ("inline", [
        f"meas 'toy' ({cls}) has no GPU Stage-4 encoder, so --embed-stage auto resolves to inline"])
    refuse_unstageable_embedders(pipe, "auto", "inline")
    # The control: the DINO embedder alone would resolve gpu, so the toy is what
    # decided. Until P5/P6 lift the interim gate (review C1) it resolves inline
    # through the gate instead, with the gate's note, not the toy's.
    alone = ImagePipeline(ops=ops, meas={"dino": MeasureDinoEmbedding()})
    assert resolve_embed_stage_for_config(config, alone) == ("inline", [_gate_note(mode)])
    assert resolve_embed_stage("auto", mode=mode, is_slurm=True, has_gpu_detector=gpu_detector,
                               gpu_slurm_given=gpu_slurm, has_embedders=True, cpu_pinned=False,
                               unstageable=True) == "inline"


def test_explicit_gpu_with_a_non_dino_embedder_resolves_gpu_and_r2_refuses(make_exec_config, tmp_path, toy_embedder):
    pipe = ImagePipeline(meas={"toy": toy_embedder})
    config = make_exec_config(pipeline_json=tmp_path / "unused.json", input_path=tmp_path,
                              embed_stage_request="gpu")
    stage, notes = resolve_embed_stage_for_config(config, pipe)
    assert (stage, notes) == ("gpu", [])
    with pytest.raises(click.UsageError, match="has no GPU Stage-4 encoder"):
        refuse_unstageable_embedders(pipe, "gpu", stage)


def test_r3_names_the_recorded_flag_and_the_remedy():
    with pytest.raises(click.UsageError, match=r"started with --embed-stage inline.*Re-run with --embed-stage inline$"):
        refuse_embed_stage_switch("inline", "gpu", is_slurm=False, stage4_available=True)
    with pytest.raises(click.UsageError, match=r"Re-run with --embed-stage gpu \(on SLURM, also --gpu-slurm\)$"):
        refuse_embed_stage_switch("gpu", "inline", is_slurm=True, stage4_available=True)
    with pytest.raises(click.UsageError, match=r"Re-run with --embed-stage gpu$"):
        refuse_embed_stage_switch("gpu", "inline", is_slurm=False, stage4_available=True)
    refuse_embed_stage_switch(None, "inline", is_slurm=False, stage4_available=True)   # a pre-#282 state
    with pytest.raises(click.UsageError, match="started with --embed-stage inline"):
        refuse_embed_stage_switch(None, "gpu", is_slurm=False, stage4_available=True)


@pytest.mark.parametrize("recorded, resolved", [("inline", "gpu"), ("gpu", "inline"), (None, "gpu")])
@pytest.mark.parametrize("is_slurm", [False, True], ids=["local", "slurm"])
@pytest.mark.parametrize("available", [False, True], ids=["gated", "available"])
def test_r3_never_names_a_command_that_cannot_run(recorded, resolved, is_slurm, available):
    """Review C1: no remedy names ``--embed-stage gpu`` where Stage 4 cannot run, nor
    ``--mode measure --embed-stage gpu``, which the gate refuses until P6 T1."""
    with pytest.raises(click.UsageError) as raised:
        refuse_embed_stage_switch(recorded, resolved, is_slurm=is_slurm, stage4_available=available)
    remedy = raised.value.message.split("would mix CPU and GPU embeddings. ", 1)[1]
    assert "--mode measure" not in remedy
    assert ("--embed-stage gpu" in remedy) is (recorded == "gpu" and available)
    if recorded == "gpu" and not available:
        assert remedy == "Stage 4 cannot run this invocation yet, so start the run over with --restart"


def test_embed_stage_is_not_in_the_work_id(make_exec_config, tiny_pipeline_file, tmp_path):
    a = make_exec_config(pipeline_json=tiny_pipeline_file, input_path=tmp_path)
    b = dataclasses.replace(a, embed_stage="gpu", embed_stage_request="gpu")
    assert processing_configuration_digest(a) == processing_configuration_digest(b)


# ---------------------------------------------------------------------------
# Resolution from the run's pipeline and flags
# ---------------------------------------------------------------------------


def _slurm_config(make_exec_config, tmp_path: Path, **overrides: Any):
    return make_exec_config(
        pipeline_json=tmp_path / "unused.json", input_path=tmp_path,
        force_local=False, slurm_args={"slurm_partition": "intel"}, **overrides,
    )


def _gate_note(mode: str) -> str:
    where = "in --mode measure" if mode == "measure" else "on SLURM"
    return f"Stage 4 {where} is not yet available, so --embed-stage auto resolves to inline"


def test_a_slurm_gpu_detector_pipeline_resolves_inline_through_the_gate(make_exec_config, tmp_path):
    """Review C1: no SLURM path runs Stage 4 until P5 T2, so ``auto`` stays inline.

    P5 T2 flips this test: it lifts SLURM full runs in ``stage4_supported`` and
    the first assertion becomes ``("gpu", [])``.
    """
    from tests._fakes.fake_gpu_detector import FakeGpuDetector

    pipe = ImagePipeline(ops=[FakeGpuDetector()], meas={"emb": MeasureDinoEmbedding()})
    config = _slurm_config(make_exec_config, tmp_path)
    assert not stage4_supported(config)
    assert resolve_embed_stage_for_config(config, pipe) == ("inline", [_gate_note("full")])
    measure = _slurm_config(make_exec_config, tmp_path, measure_only=True)
    assert resolve_embed_stage_for_config(measure, pipe) == ("inline", [])   # U10: only --gpu-slurm
    with_profile = _slurm_config(make_exec_config, tmp_path, measure_only=True,
                                 gpu_slurm_args={"slurm_partition": "exfab"})
    assert resolve_embed_stage_for_config(with_profile, pipe) == ("inline", [_gate_note("measure")])


@pytest.mark.parametrize("where", ["slurm-full", "slurm-measure", "local-measure"])
def test_an_explicit_gpu_where_no_stage4_runs_is_one_usage_line(make_exec_config, tmp_path, where):
    """Review C1: ``--embed-stage gpu`` on SLURM (any mode) or in measure mode is refused.

    P5 T2 flips ``slurm-full`` (it resolves ``gpu``); P6 T1 flips the measure cases.
    """
    pipe = ImagePipeline(meas={"emb": MeasureDinoEmbedding()})
    measure = where.endswith("measure")
    if where.startswith("slurm"):
        config = _slurm_config(make_exec_config, tmp_path, measure_only=measure, embed_stage_request="gpu")
    else:
        config = make_exec_config(pipeline_json=tmp_path / "unused.json", input_path=tmp_path,
                                  measure_only=True, embed_stage_request="gpu")
    assert not stage4_supported(config)
    with pytest.raises(click.UsageError) as raised:
        resolve_embed_stage_for_config(config, pipe)
    message = raised.value.message
    assert "\n" not in message and message.endswith("pass --embed-stage inline to embed inline in Stage 3")
    assert message.startswith(
        "--embed-stage gpu: Stage 4 " + ("in --mode measure" if measure else "on SLURM") + " is not yet available")


def test_a_local_full_run_is_the_one_supported_stage4_path(make_exec_config, tmp_path):
    """The gate's positive control: local full runs keep ``--embed-stage gpu`` (P4)."""
    pipe = ImagePipeline(meas={"emb": MeasureDinoEmbedding()})
    config = make_exec_config(pipeline_json=tmp_path / "unused.json", input_path=tmp_path,
                              embed_stage_request="gpu")
    assert stage4_supported(config)
    assert resolve_embed_stage_for_config(config, pipe) == ("gpu", [])


def test_a_cpu_pin_that_decides_the_outcome_prints_the_u9_note(make_exec_config, tmp_path):
    pipe = ImagePipeline(meas={"size": MeasureSize(), "emb": MeasureDinoEmbedding(device="cpu")})
    config = _slurm_config(make_exec_config, tmp_path, gpu_slurm_args={"slurm_partition": "exfab"})
    # The pin alone forces inline, so its note (and not the C1 gate's) is printed.
    assert resolve_embed_stage_for_config(config, pipe) == (
        "inline", ["meas 'emb' is pinned to device='cpu', so --embed-stage auto resolves to inline"])
    # Locally auto is inline anyway, so the pin decides nothing and says nothing.
    local = make_exec_config(pipeline_json=tmp_path / "unused.json", input_path=tmp_path)
    assert resolve_embed_stage_for_config(local, pipe) == ("inline", [])


def test_gpu_without_an_embedder_resolves_inline_with_the_o5_note(make_exec_config, tmp_path):
    config = make_exec_config(pipeline_json=tmp_path / "unused.json", input_path=tmp_path,
                              embed_stage_request="gpu")
    assert resolve_embed_stage_for_config(config, ImagePipeline(meas={"size": MeasureSize()})) == (
        "inline", ["--embed-stage gpu: the pipeline has no embedder, so nothing runs in Stage 4"])


def test_an_unloadable_pipeline_resolves_inline(make_exec_config, tmp_path):
    config = make_exec_config(pipeline_json=tmp_path / "unused.json", input_path=tmp_path,
                              embed_stage_request="gpu")
    assert resolve_embed_stage_for_config(config, None) == ("inline", [])


# ---------------------------------------------------------------------------
# The Stage-4 environment: cuBLAS, and the local CUDA check
# ---------------------------------------------------------------------------


def _fake_torch(*, initialized: bool = False, available: bool = False) -> SimpleNamespace:
    return SimpleNamespace(cuda=SimpleNamespace(
        is_initialized=lambda: initialized, is_available=lambda: available))


def test_the_cublas_variable_is_set_before_any_strategy(monkeypatch, make_exec_config, tiny_pipeline_file, tmp_path):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setenv("PHENOTYPIC_STAGE4_DEVICE", "cpu")
    prepare_stage4_environment(dataclasses.replace(
        make_exec_config(pipeline_json=tiny_pipeline_file, input_path=tmp_path), embed_stage="gpu"))
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_an_inline_run_prepares_nothing(monkeypatch, make_exec_config, tiny_pipeline_file, tmp_path):
    monkeypatch.delenv(CUBLAS, raising=False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=False))
    prepare_stage4_environment(make_exec_config(pipeline_json=tiny_pipeline_file, input_path=tmp_path))
    assert CUBLAS not in os.environ


def test_a_cublas_setting_after_cuda_initialised_is_refused(monkeypatch, make_exec_config, tiny_pipeline_file, tmp_path):
    """P1/P2 review M12: cuBLAS read the variable at its first handle; a later value is inert."""
    monkeypatch.delenv(CUBLAS, raising=False)
    monkeypatch.setenv("PHENOTYPIC_STAGE4_DEVICE", "cpu")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(initialized=True))
    config = dataclasses.replace(
        make_exec_config(pipeline_json=tiny_pipeline_file, input_path=tmp_path), embed_stage="gpu")
    with pytest.raises(click.ClickException, match="CUBLAS_WORKSPACE_CONFIG is unset.*already initialised CUDA") as raised:
        prepare_stage4_environment(config)
    assert raised.value.exit_code == 1 and "\n" not in raised.value.message
    assert CUBLAS not in os.environ
    monkeypatch.setenv(CUBLAS, ":16:8")   # a preset deterministic value is kept
    prepare_stage4_environment(config)
    assert os.environ[CUBLAS] == ":16:8"


def test_a_local_nondeterministic_cublas_preset_is_refused(monkeypatch, make_exec_config, tiny_pipeline_file, tmp_path):
    """Refused at startup, not by ``check_cublas_workspace`` after Stages 1-3."""
    monkeypatch.setenv(CUBLAS, ":0:0")
    monkeypatch.setenv("PHENOTYPIC_STAGE4_DEVICE", "cpu")
    config = dataclasses.replace(
        make_exec_config(pipeline_json=tiny_pipeline_file, input_path=tmp_path), embed_stage="gpu")
    with pytest.raises(click.ClickException, match="':0:0'; deterministic cuBLAS needs :4096:8 or :16:8"):
        prepare_stage4_environment(config)


def test_a_slurm_submitter_sets_cublas_and_checks_no_device(monkeypatch, make_exec_config, tmp_path):
    """The shard checks its own device; the submitting host may have no GPU."""
    monkeypatch.delenv(CUBLAS, raising=False)
    monkeypatch.delenv("PHENOTYPIC_STAGE4_DEVICE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=False))
    prepare_stage4_environment(_slurm_config(make_exec_config, tmp_path, embed_stage="gpu"))
    assert os.environ[CUBLAS] == ":4096:8"


@pytest.mark.parametrize("torch_module", [None, "no-cuda"], ids=["no-torch", "no-cuda"])
def test_a_local_run_without_cuda_is_one_exit_1_line(monkeypatch, make_exec_config, tiny_pipeline_file, tmp_path, torch_module):
    monkeypatch.delenv(CUBLAS, raising=False)
    monkeypatch.delenv("PHENOTYPIC_STAGE4_DEVICE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", None if torch_module is None else _fake_torch(available=False))
    config = dataclasses.replace(
        make_exec_config(pipeline_json=tiny_pipeline_file, input_path=tmp_path), embed_stage="gpu")
    with pytest.raises(click.ClickException, match="CUDA") as raised:
        prepare_stage4_environment(config)
    assert raised.value.exit_code == 1 and "\n" not in raised.value.message
    assert "--embed-stage inline" in raised.value.message


def test_the_test_escape_runs_on_the_cpu_without_asking_torch(monkeypatch):
    monkeypatch.setenv("PHENOTYPIC_STAGE4_DEVICE", "cpu")
    monkeypatch.setitem(sys.modules, "torch", None)   # any import would raise
    assert stage4_device() == "cpu"


@pytest.mark.parametrize("capability, arches, usable", [
    ((8, 9), ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"], True),   # ada6000 on cu130
    ((8, 0), ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"], True),   # a100
    ((9, 0), ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"], True),   # h100
    ((12, 0), ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"], True),  # blackwell6000
    ((6, 0), ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"], False),  # p100
    ((8, 6), ["sm_89"], False),                 # a binary never runs on an older minor
    ((9, 0), ["sm_86"], False),                 # nor on another major
    ((8, 9), ["compute_80"], True),             # PTX compiles forward
    ((7, 5), ["compute_80"], False),
    ((9, 0), ["sm_90a"], True),
])
def test_has_kernels_for(capability, arches, usable):
    assert has_kernels_for(*capability, arches) is usable


# ---------------------------------------------------------------------------
# Recording: processing state and job metadata
# ---------------------------------------------------------------------------


def test_a_local_run_records_its_resolved_stage(tmp_path, tiny_inputs):
    from phenotypic.sdk_ import processing_state_path
    from tests._tiny_run import run_cli

    out = tmp_path / "out"
    result = run_cli(tiny_inputs, out)
    assert result.exit_code == 0, result.output
    config = json.loads(processing_state_path(out).read_text(encoding="utf-8"))["config"]
    assert config["embed_stage"] == "inline"


def test_the_staged_job_metadata_records_the_stage(tmp_path):
    from phenotypic._cli._cli_staged_slurm import _write_staged_job_metadata
    from phenotypic.sdk_ import JobMetadataKey

    config = SimpleNamespace(
        include_dataset_column=True, metadata_csv=None, input_path=tmp_path / "in",
        no_qc=False, study_yaml=None, pipeline_json=tmp_path / "pipeline.json",
        full_dataset_inventory={}, processing_generation="g", embed_stage="gpu",
    )
    scripts = {"stage1": [Path("s1.sh")], "stage2": Path("s2.sh"), "stage3": [Path("s3.sh")],
               "finalizer": Path("f.sh"), "controller": Path("c.sh")}
    path = _write_staged_job_metadata(datasets=[], output_dir=tmp_path, config=config, scripts=scripts,
                                      job_ids=[], start_time=__import__("datetime").datetime(2026, 11, 20),
                                      epoch="e")
    assert json.loads(path.read_text(encoding="utf-8"))[JobMetadataKey.EMBED_STAGE] == "gpu"


def test_the_autonomous_job_metadata_records_the_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, simple_pipeline_json: Path,
    make_exec_config: Callable[..., Any],
) -> None:
    from phenotypic._cli import _cli_execution_strategies as strategies
    from phenotypic._cli._cli_output_manager import OutputManager
    from phenotypic._cli._cli_types import Dataset
    from phenotypic.sdk_ import JobMetadataKey, job_metadata_path

    class _Stop(Exception):
        """End the run at submission."""

    seen: dict[str, Any] = {}

    def _at_submission(**kwargs: Any):
        seen["metadata"] = json.loads(job_metadata_path(out).read_text(encoding="utf-8"))
        raise _Stop()

    out = tmp_path / "out"
    out.mkdir()
    image = tmp_path / "plate.tif"
    image.write_bytes(b"")
    script = tmp_path / "chunk.sh"
    script.write_text("#!/bin/bash\n", encoding="utf-8")
    monkeypatch.setattr(strategies, "get_slurm_array_limit", lambda: 1000)
    monkeypatch.setattr(strategies, "generate_all_array_job_scripts",
                        lambda datasets, config, output_dir, limit: {"ds": [script]})
    monkeypatch.setattr(strategies, "generate_terminal_finalizer_script", lambda *a, **k: script)
    monkeypatch.setattr(strategies, "submit_slurm_script_chain", _at_submission)
    config = make_exec_config(pipeline_json=simple_pipeline_json, input_path=tmp_path, output_dir=out,
                              force_local=False, slurm_args={"slurm_partition": "short"},
                              embed_stage="gpu")
    strategy = strategies.AutonomousSLURMStrategy(config, OutputManager.from_config(out, ".tiff"))
    with pytest.raises(_Stop):
        strategy.execute([Dataset("ds", [image], tmp_path, out)], out)
    assert seen["metadata"][JobMetadataKey.EMBED_STAGE] == "gpu"
