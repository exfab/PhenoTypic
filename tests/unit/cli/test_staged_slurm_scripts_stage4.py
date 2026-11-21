"""The Stage-4 SLURM scripts, config v2, the no-detector plan, dispatch and U5 (#282 P5 T2).

Shared env: nothing here loads a backbone or imports torch. The fake
scheduler is the ``submissions`` fixture (``tests/unit/cli/conftest.py``).
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path

import pytest

from phenotypic._cli import _cli_embed_stage as embed_stage_module
from phenotypic._cli._cli_embed_stage import (
    pin_embed_revisions,
    recorded_embed_revisions,
    resolve_embed_revisions,
)
from phenotypic._cli._cli_execution_strategies import (
    AutonomousSLURMStrategy,
    create_execution_strategy,
    uses_staged_controller,
)
from phenotypic._cli._cli_staged_controller import _load_controller_config
from phenotypic._cli._cli_staged_slurm import (
    StagedSlurmStrategy,
    _write_staged_job_metadata,
    generate_staged_scripts,
)
from phenotypic._cli._cli_staged_orchestration import (
    load_orchestration_state,
    load_staged_manifest,
)
from phenotypic.sdk_ import RunExitCode
from tests._tiny_run import (
    STAGED_V2,
    dino_pipeline_path,
    output_manager_for,
    pipeline_with_dino,
    read_job_metadata,
    staged_script_kwargs,
    submit_staged_no_detector,
)

COMMIT = "c" * 40


def _end_submitted_run(out: Path) -> None:
    """The fake scheduler's jobs ended: close the epoch's fence, as a cancel would.

    ``submissions`` ledgers no job, so the fence would otherwise read as a
    submitter still preparing and refuse the next invocation.
    """
    from phenotypic._cli._cli_staged_orchestration import deactivate_orchestration

    assert deactivate_orchestration(out, "failed")


# ---------------------------------------------------------------------------
# Scripts and the v2 controller config
# ---------------------------------------------------------------------------


def test_stage4_script_is_one_unchunked_gpu_array(tmp_path):
    scripts = generate_staged_scripts(**staged_script_kwargs(
        tmp_path, phases=STAGED_V2, n_shards=3, array_limit=1,
        gpu_slurm_args={"slurm_partition": "exfab", "slurm_account": "exfab"}))
    text = scripts["stage4"].read_text(encoding="utf-8")
    assert "#SBATCH --array=0-2" in text and "--gpus-per-node=1" in text
    assert "#SBATCH --partition=exfab" in text and "#SBATCH --account=exfab" in text
    # The prelude exports the deterministic workspace on its own line, before Python.
    assert "\nexport CUBLAS_WORKSPACE_CONFIG=:4096:8\n" in text
    assert text.index("export CUBLAS_WORKSPACE_CONFIG") < text.index("_cli_staged_slurm_worker")
    assert "--stage 4" in text and "--index $SLURM_ARRAY_TASK_ID" in text
    assert f"--worklist {(tmp_path / 'out' / '.phenotypic' / 'slurm_scripts' / 'stage4_worklist.json').absolute()}" in text
    assert "--republish" not in text
    # Never chunked: array_limit=1 split the two-image Stage 1/3 into chunks, not Stage 4.
    assert len(scripts["stage3"]) == 2 and isinstance(scripts["stage4"], Path)


def test_the_stage4_profile_pins_no_gpu_type_and_carries_the_users_flags(tmp_path):
    """U6: shards may land on any GPU model; a constraint comes only from --gpu-slurm."""
    plain = generate_staged_scripts(**staged_script_kwargs(tmp_path / "a", n_shards=2))
    assert "--constraint" not in plain["stage4"].read_text(encoding="utf-8")
    pinned = generate_staged_scripts(**staged_script_kwargs(
        tmp_path / "b", n_shards=2,
        gpu_slurm_args={"slurm_partition": "short_gpu", "slurm_constraint": "gpu_latest",
                        "slurm_time": "01:55:00"}))
    text = pinned["stage4"].read_text(encoding="utf-8")
    assert "#SBATCH --partition=short_gpu" in text and "#SBATCH --constraint=gpu_latest" in text
    assert "#SBATCH --mem=8G" in text           # inherited from --slurm
    assert "#SBATCH --time=01:55:00" in text    # the GPU profile's delta wins
    assert "--constraint" not in pinned["stage3"][0].read_text(encoding="utf-8")


def test_the_v2_config_is_one_the_controller_loads(tmp_path):
    scripts = generate_staged_scripts(**staged_script_kwargs(tmp_path, n_shards=3))
    config = _load_controller_config(scripts["controller_config"])
    assert config["version"] == 2 and config["phases"] == STAGED_V2
    assert config["stage4_script"] == str(scripts["stage4"].absolute())
    assert config["stage4_worklist"].endswith("stage4_worklist.json")
    assert config["n_shards"] == 3 and config["stage3_role"] == "staged"
    assert config["pipeline_sha256"] is None and "republish" not in config
    assert config["detector_slot"] and config["stage1_scripts"] and config["stage2_script"]


def test_a_detector_run_that_embeds_inline_has_no_stage4(tmp_path):
    scripts = generate_staged_scripts(**staged_script_kwargs(
        tmp_path, phases=["stage1", "stage2", "stage3"]))
    config = _load_controller_config(scripts["controller_config"])
    assert config["phases"] == ["stage1", "stage2", "stage3"]
    assert "stage4" not in scripts and "stage4_script" not in config
    assert not (scripts["controller_config"].parent / "stage4.sh").exists()


def test_the_no_detector_plan_has_no_stage1_or_stage2(tmp_path):
    scripts = generate_staged_scripts(**staged_script_kwargs(
        tmp_path, phases=["stage3", "stage4"], stage3_role="single_pass", detector_slot=None,
        drop_originals=True, pipeline_identity={"source_path": "/src/p.json", "sha256": "e" * 64}))
    config = json.loads(scripts["controller_config"].read_text(encoding="utf-8"))
    assert config["version"] == 2 and config["phases"] == ["stage3", "stage4"]
    assert "stage1_scripts" not in config and "detector_slot" not in config and "republish" not in config
    assert "stage1" not in scripts and "stage2" not in scripts
    assert config["stage3_role"] == "single_pass"
    stage3 = scripts["stage3"][0].read_text(encoding="utf-8")
    # The single-pass role reads the input as Stage 1 does.
    assert "--stage3-role single_pass" in stage3 and "--drop-originals" in stage3
    assert "--provenance-pipeline-sha256" in stage3
    _load_controller_config(scripts["controller_config"])


def test_a_staged_stage3_names_no_role(tmp_path):
    scripts = generate_staged_scripts(**staged_script_kwargs(tmp_path, drop_originals=True))
    stage3 = scripts["stage3"][0].read_text(encoding="utf-8")
    assert "--stage3-role" not in stage3 and "--drop-originals" not in stage3


@pytest.mark.parametrize("phases, role, slot", [
    (["stage3", "stage4"], "staged", None),
    (STAGED_V2, "single_pass", "slot"),
    (STAGED_V2, "staged", None),
])
def test_a_role_or_slot_that_does_not_fit_the_phases_is_refused(tmp_path, phases, role, slot):
    with pytest.raises(ValueError):
        generate_staged_scripts(**staged_script_kwargs(
            tmp_path, phases=phases, stage3_role=role, detector_slot=slot))


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _slurm_config(make_exec_config, tmp_path, **overrides):
    return make_exec_config(
        pipeline_json=dino_pipeline_path(tmp_path), input_path=tmp_path, output_dir=tmp_path / "out",
        slurm_args={"slurm_partition": "intel"}, force_local=False, **overrides)


@pytest.mark.parametrize("embed_stage, strategy", [("inline", AutonomousSLURMStrategy), ("gpu", StagedSlurmStrategy)])
def test_slurm_dispatch_follows_the_resolved_embed_stage(make_exec_config, tmp_path, embed_stage, strategy):
    config = dataclasses.replace(_slurm_config(make_exec_config, tmp_path), embed_stage=embed_stage)
    assert uses_staged_controller(config) is (embed_stage == "gpu")
    assert isinstance(create_execution_strategy(config, output_manager_for(config)), strategy)


def test_a_local_gpu_run_does_not_take_the_controller(make_exec_config, tmp_path):
    config = make_exec_config(pipeline_json=dino_pipeline_path(tmp_path), input_path=tmp_path,
                              embed_stage="gpu")
    assert not uses_staged_controller(config)


# ---------------------------------------------------------------------------
# The no-detector submission, end to end through the fake scheduler
# ---------------------------------------------------------------------------


def test_a_no_detector_submission_writes_the_two_phase_plan(tmp_path, monkeypatch, submissions):
    monkeypatch.setattr(embed_stage_module, "_hub_commit", lambda hf_id: COMMIT)
    result = submit_staged_no_detector(tmp_path, n_shards=3)
    assert result.exit_code == RunExitCode.SUBMITTED, result.output
    assert submissions == ["staged"]
    out = tmp_path / "out"
    metadata = read_job_metadata(out)
    assert metadata["embed_stage"] == "gpu"
    config = _load_controller_config(Path(load_orchestration_state(out)["controller_config"]))
    assert config["phases"] == ["stage3", "stage4"] and config["n_shards"] == 3
    state = load_orchestration_state(out)
    assert state["phase"] == "stage3"
    # The finalizer derives a detector slot only for a run that has one.
    assert state["stage3_markers_required"] is False
    stage4 = Path(config["stage4_script"]).read_text(encoding="utf-8")
    assert "#SBATCH --partition=exfab" in stage4 and "#SBATCH --account=exfab" in stage4
    assert "#SBATCH --array=0-2" in stage4
    assert any(path.endswith("stage4.sh") for path in metadata["chunk_scripts"])


def test_revisions_are_resolved_once_and_recorded(tmp_path, monkeypatch, submissions):
    calls = []
    monkeypatch.setattr(embed_stage_module, "_hub_commit", lambda hf_id: calls.append(hf_id) or COMMIT)
    result = submit_staged_no_detector(tmp_path, n_shards=3)
    assert result.exit_code == RunExitCode.SUBMITTED, result.output
    assert calls == ["facebook/dinov2-base"]
    assert read_job_metadata(tmp_path / "out")["embed_revisions"] == {"facebook/dinov2-base": COMMIT}
    assert recorded_embed_revisions(tmp_path / "out") == {"facebook/dinov2-base": COMMIT}


def test_a_continuation_reuses_the_recorded_revision(tmp_path, monkeypatch, submissions):
    monkeypatch.setattr(embed_stage_module, "_hub_commit", lambda hf_id: COMMIT)
    first = submit_staged_no_detector(tmp_path, n_shards=3)
    assert first.exit_code == RunExitCode.SUBMITTED, first.output
    _end_submitted_run(tmp_path / "out")
    monkeypatch.setattr(embed_stage_module, "_hub_commit", lambda hf_id: pytest.fail("re-resolved"))
    again = submit_staged_no_detector(tmp_path, n_shards=3)                    # the same command again
    assert again.exit_code == RunExitCode.SUBMITTED, again.output
    assert "Continuing" in again.output
    assert read_job_metadata(tmp_path / "out")["embed_revisions"] == {"facebook/dinov2-base": COMMIT}
    assert recorded_embed_revisions(tmp_path / "out") == {"facebook/dinov2-base": COMMIT}


def test_a_restart_resolves_again(tmp_path, monkeypatch, submissions):
    monkeypatch.setattr(embed_stage_module, "_hub_commit", lambda hf_id: COMMIT)
    assert submit_staged_no_detector(tmp_path).exit_code == RunExitCode.SUBMITTED
    _end_submitted_run(tmp_path / "out")
    monkeypatch.setattr(embed_stage_module, "_hub_commit", lambda hf_id: "d" * 40)
    result = submit_staged_no_detector(tmp_path, extra=["--restart"])
    assert result.exit_code == RunExitCode.SUBMITTED, result.output
    assert read_job_metadata(tmp_path / "out")["embed_revisions"] == {"facebook/dinov2-base": "d" * 40}


def test_offline_revision_falls_back_with_one_warning(monkeypatch, capsys):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(embed_stage_module, "_hub_commit", lambda hf_id: pytest.fail("asked the hub offline"))
    assert resolve_embed_revisions(pipeline_with_dino()) == {"facebook/dinov2-base": None}
    err = capsys.readouterr().err
    assert err.count("resolves on its own") == 1 and "facebook/dinov2-base" in err


def test_a_hub_error_falls_back_with_one_warning(monkeypatch, capsys):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    def _down(hf_id):
        raise OSError("network is unreachable")

    monkeypatch.setattr(embed_stage_module, "_hub_commit", _down)
    assert resolve_embed_revisions(pipeline_with_dino()) == {"facebook/dinov2-base": None}
    err = capsys.readouterr().err
    assert err.count("resolves on its own") == 1 and "network is unreachable" in err


def test_a_pinned_embedder_and_a_shared_model_are_asked_once(monkeypatch):
    from phenotypic import ImagePipeline
    from phenotypic.measure import MeasureDinoEmbedding

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    calls = []
    monkeypatch.setattr(embed_stage_module, "_hub_commit", lambda hf_id: calls.append(hf_id) or COMMIT)
    pipe = ImagePipeline(meas={
        "a": MeasureDinoEmbedding(), "b": MeasureDinoEmbedding(name="second", pooling="crop_mean"),
        "pinned": MeasureDinoEmbedding(size="small", revision="a" * 40),
    })
    assert resolve_embed_revisions(pipe) == {"facebook/dinov2-base": COMMIT}
    assert calls == ["facebook/dinov2-base"]


def test_an_inline_run_pins_nothing(make_exec_config, tmp_path, monkeypatch):
    monkeypatch.setattr(embed_stage_module, "_hub_commit", lambda hf_id: pytest.fail("an inline run asked"))
    config = _slurm_config(make_exec_config, tmp_path)
    pin_embed_revisions(config, tmp_path / "out")
    assert config.embed_revisions is None


# ---------------------------------------------------------------------------
# The submitter reuses the identities a continuation's split built
# ---------------------------------------------------------------------------


def test_the_submitter_reuses_precomputed_entries(make_exec_config, tmp_path, monkeypatch, submissions):
    """Review I4: an entry the CLI's split built is not hashed again by the submitter."""
    from phenotypic._cli import _cli_staged_slurm
    from phenotypic._cli._cli_staged_orchestration import StagedManifestEntry
    from phenotypic._cli._cli_types import Dataset

    image = tmp_path / "in" / "ds" / "plate_1.tif"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"x")
    entry = StagedManifestEntry("ds", image.name, "plate_1", str(image), work_id="w-precomputed")
    monkeypatch.setattr(_cli_staged_slurm, "staged_manifest_entry",
                        lambda *a, **k: pytest.fail("hashed the input again"))
    config = dataclasses.replace(
        _slurm_config(make_exec_config, tmp_path), embed_stage="gpu",
        gpu_slurm_args={"slurm_partition": "exfab"},
        staged_manifest_entries={("ds", image): entry})
    out = tmp_path / "out"
    StagedSlurmStrategy(config, output_manager_for(config)).execute(
        [Dataset(name="ds", images=[image], input_dir=image.parent, output_dir=out)], out)
    manifest = load_staged_manifest(out / ".phenotypic" / "slurm_scripts" / "staged_manifest.json")
    assert [e.work_id for e in manifest] == ["w-precomputed"]


def test_the_job_metadata_records_the_revisions(make_exec_config, tmp_path):
    config = dataclasses.replace(_slurm_config(make_exec_config, tmp_path), embed_stage="gpu",
                                 embed_revisions={"facebook/dinov2-base": COMMIT})
    out = tmp_path / "out"
    _write_staged_job_metadata(
        datasets=[], output_dir=out, config=config,
        scripts={"stage3": [Path("s3.sh")], "stage4": Path("s4.sh"), "finalizer": Path("f.sh"),
                 "controller": Path("c.sh")},
        job_ids=[], start_time=datetime.now(), epoch="e")
    metadata = read_job_metadata(out)
    assert metadata["embed_revisions"] == {"facebook/dinov2-base": COMMIT}
    assert metadata["chunk_scripts"] == ["c.sh", "s3.sh", "s4.sh", "f.sh"]
