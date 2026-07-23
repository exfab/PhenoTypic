"""Shard partitioning + per-stage SLURM args resolution (Spec 1 §7, Plan 3)."""

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from phenotypic._cli._cli_staged_slurm import (
    StagedSlurmStrategy,
    _write_staged_job_metadata,
    generate_staged_scripts,
    partition_shards,
    resolve_stage_slurm_args,
)
from phenotypic._cli._cli_staged_orchestration import (
    StagedManifestEntry,
    load_orchestration_state,
)
from phenotypic._cli._cli_types import Dataset
from phenotypic.sdk_ import JOB_METADATA_JSON, progress_dir


def _manifest(n):
    return [
        StagedManifestEntry("ds", f"img{i}.tif", f"img{i}", f"/in/img{i}.tif")
        for i in range(n)
    ]


def test_partition_shards_even():
    items = list(range(10))
    shards = partition_shards(items, 3)
    assert [len(s) for s in shards] == [4, 3, 3]
    assert sorted(x for s in shards for x in s) == items  # no loss


def test_partition_shards_more_shards_than_items():
    shards = partition_shards([1, 2], 5)
    assert [len(s) for s in shards if s] == [1, 1]
    assert len([s for s in shards if s]) == 2


def test_partition_shards_single_shard():
    assert partition_shards([1, 2, 3], 1) == [[1, 2, 3]]


def test_gpu_stage_auto_requests_one_gpu():
    args = resolve_stage_slurm_args(gpu_slurm_args={"slurm_partition": "gpu"})
    assert args["slurm_gpus_per_node"] == 1  # auto-added when absent


def test_gpu_stage_respects_explicit_gpu_count():
    args = resolve_stage_slurm_args(
        gpu_slurm_args={"slurm_partition": "gpu", "slurm_gpus_per_node": 2}
    )
    assert args["slurm_gpus_per_node"] == 2


def test_gpu_stage_omits_gpu_directive_when_zero():
    # OQ4: explicit 0 -> CPU-only run of the GPU stage; the directive is omitted
    # so format_sbatch_directives never emits --gpus-per-node=0.
    args = resolve_stage_slurm_args(
        gpu_slurm_args={"slurm_partition": "short", "slurm_gpus_per_node": 0}
    )
    assert "slurm_gpus_per_node" not in args
    assert args["slurm_partition"] == "short"


def test_gpu_stage_inherits_shared_keys_and_overrides_partition():
    args = resolve_stage_slurm_args(
        gpu_slurm_args={
            "slurm_partition": "exfab",
            "slurm_account": "exfab_acct",
        },
        cpu_slurm_args={"slurm_partition": "short", "slurm_qos": "normal"},
    )
    assert args["slurm_partition"] == "exfab"  # gpu overrides cpu
    assert args["slurm_account"] == "exfab_acct"  # added for the gpu partition
    assert args["slurm_qos"] == "normal"  # inherited from --slurm
    assert args["slurm_gpus_per_node"] == 1


def test_generates_three_stage_scripts_with_correct_resources(tmp_path):
    scripts = generate_staged_scripts(
        pipeline_path=tmp_path / "p.json",
        datasets_manifest=_manifest(3),
        output_dir=tmp_path,
        image_type="Image",
        cpu_slurm_args={"slurm_partition": "batch", "slurm_cpus_per_task": 4},
        gpu_slurm_args={"slurm_partition": "gpu"},
        n_shards=2,
        epoch="epoch-1",
        overlay_alpha=0.65,
    )
    assert set(scripts) == {
        "stage1",
        "stage2",
        "stage3",
        "finalizer",
        "controller",
        "controller_config",
        "manifest",
    }
    # Image-arrayed stages return a list of chunk scripts (one chunk here);
    # the GPU stage is a single shard-arrayed script.
    s1 = scripts["stage1"][0].read_text(encoding="utf-8")
    s2 = scripts["stage2"].read_text(encoding="utf-8")
    s3 = scripts["stage3"][0].read_text(encoding="utf-8")
    finalizer = scripts["finalizer"].read_text(encoding="utf-8")

    # Stage 1 & 3 on the CPU partition; Stage 2 on the GPU partition + 1 GPU
    assert "--partition=batch" in s1 and "--partition=batch" in s3
    assert "--partition=gpu" in s2 and "--gpus-per-node=1" in s2
    # Stage 1/3 = array over images (0-2); Stage 2 = array over shards (0-1)
    assert "--array=0-2" in s1 and "--array=0-2" in s3
    assert "--array=0-1" in s2
    assert "--overlay-alpha 0.65" in s3
    assert "--overlay-alpha" not in s1 and "--overlay-alpha" not in s2
    # Stage 2 invokes the shard worker
    assert "_cli_staged_slurm_worker" in s2
    assert "_cli_checkpoint_handler" in finalizer
    assert "--checkpoint-type finalize" in finalizer
    assert "--epoch epoch-1" in finalizer
    assert "--partition=batch" in finalizer
    assert "_cli_staged_controller" in scripts["controller"].read_text()


def test_stage2_script_uses_controller_not_signal_requeue(tmp_path):
    scripts = generate_staged_scripts(
        pipeline_path=tmp_path / "p.json",
        datasets_manifest=_manifest(2),
        output_dir=tmp_path,
        image_type="Image",
        cpu_slurm_args={"slurm_partition": "batch"},
        gpu_slurm_args={"slurm_partition": "gpu"},
        n_shards=1,
        epoch="epoch-1",
    )
    s2 = scripts["stage2"].read_text(encoding="utf-8")
    assert "--signal=" not in s2
    assert "--requeue" not in s2
    assert "--stage3-markers-required" in s2
    assert "--epoch epoch-1" in s2


def test_image_stages_chunked_when_exceeding_array_limit(tmp_path):
    # 5 images / array_limit 2 -> ceil(5/2) = 3 chunks per image-arrayed stage;
    # the GPU stage is never chunked (it is an array over shards).
    scripts = generate_staged_scripts(
        pipeline_path=tmp_path / "p.json",
        datasets_manifest=_manifest(5),
        output_dir=tmp_path,
        image_type="Image",
        cpu_slurm_args={"slurm_partition": "batch"},
        gpu_slurm_args={"slurm_partition": "gpu"},
        n_shards=1,
        array_limit=2,
        epoch="epoch-1",
    )
    assert len(scripts["stage1"]) == 3
    assert len(scripts["stage3"]) == 3
    assert isinstance(scripts["stage2"], Path)

    arrays = [s.read_text(encoding="utf-8") for s in scripts["stage1"]]
    # Every chunk is a 0-based array within the limit (never an index >= limit).
    assert "--array=0-1" in arrays[0]
    assert "--array=0-1" in arrays[1]
    assert "--array=0-0" in arrays[2]  # trailing chunk: 1 image


def test_chunk_windows_map_to_absolute_manifest_indices(tmp_path):
    scripts = generate_staged_scripts(
        pipeline_path=tmp_path / "p.json",
        datasets_manifest=_manifest(5),
        output_dir=tmp_path,
        image_type="Image",
        cpu_slurm_args={"slurm_partition": "batch"},
        gpu_slurm_args={"slurm_partition": "gpu"},
        n_shards=1,
        array_limit=2,
        epoch="epoch-1",
    )
    c0, c1, c2 = (s.read_text(encoding="utf-8") for s in scripts["stage1"])
    # The worker resolves the ABSOLUTE manifest index via CURRENT_TASK_INDEX,
    # populated from the per-chunk TASK_INDICES window.
    assert "--index $CURRENT_TASK_INDEX" in c0
    assert "    0\n    1" in c0  # window [0, 2)
    assert "    2\n    3" in c1  # window [2, 4)
    assert "    4" in c2  # window [4, 5)


def test_single_chunk_keeps_plain_script_names(tmp_path):
    scripts = generate_staged_scripts(
        pipeline_path=tmp_path / "p.json",
        datasets_manifest=_manifest(3),
        output_dir=tmp_path,
        image_type="Image",
        cpu_slurm_args={"slurm_partition": "batch"},
        gpu_slurm_args={"slurm_partition": "gpu"},
        n_shards=1,
        array_limit=1000,
        epoch="epoch-1",
    )
    assert [p.name for p in scripts["stage1"]] == ["stage1.sh"]
    assert [p.name for p in scripts["stage3"]] == ["stage3.sh"]


def test_empty_manifest_generates_no_image_stage_arrays(tmp_path):
    scripts = generate_staged_scripts(
        pipeline_path=tmp_path / "p.json",
        datasets_manifest=[],
        output_dir=tmp_path,
        image_type="Image",
        cpu_slurm_args={"slurm_partition": "batch"},
        gpu_slurm_args={"slurm_partition": "gpu"},
        n_shards=1,
        array_limit=1000,
        epoch="epoch-1",
    )

    assert scripts["stage1"] == []
    assert scripts["stage3"] == []


def test_multi_chunk_scripts_get_indexed_names(tmp_path):
    scripts = generate_staged_scripts(
        pipeline_path=tmp_path / "p.json",
        datasets_manifest=_manifest(5),
        output_dir=tmp_path,
        image_type="Image",
        cpu_slurm_args={"slurm_partition": "batch"},
        gpu_slurm_args={"slurm_partition": "gpu"},
        n_shards=1,
        array_limit=2,
        epoch="epoch-1",
    )
    assert [p.name for p in scripts["stage1"]] == [
        "stage1_chunk0.sh",
        "stage1_chunk1.sh",
        "stage1_chunk2.sh",
    ]


@pytest.mark.parametrize(
    ("resume_phase", "finalizer_only", "expected_phase", "stage3_index"),
    [
        (None, False, "stage1", 0),
        ("stage2", False, "stage2", 0),
        ("stage3", True, "stage3", 1),
    ],
)
def test_strategy_reserves_two_max_submit_slots_for_controllers(
    monkeypatch,
    tmp_path,
    resume_phase,
    finalizer_only,
    expected_phase,
    stage3_index,
):
    """The array plus running and successor controllers fit MaxSubmitJobs."""
    captured = {}

    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_slurm.get_slurm_array_limit",
        lambda: 1000,
    )
    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_slurm.get_slurm_max_submit_jobs",
        lambda: 5,
    )

    def _fake_generate(**kwargs):
        captured["array_limit"] = kwargs["array_limit"]
        return {
            "stage1": [tmp_path / "s1.sh"],
            "stage2": tmp_path / "s2.sh",
            "stage3": [tmp_path / "s3.sh"],
            "finalizer": tmp_path / "finalizer.sh",
            "controller": tmp_path / "controller.sh",
            "controller_config": tmp_path / "controller.json",
        }

    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_slurm.generate_staged_scripts",
        _fake_generate,
    )
    submitted_roles = []

    def fake_submit(*args, **kwargs):
        submitted_roles.append(kwargs["role"])
        return "100"

    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_slurm.submit_with_intent",
        fake_submit,
    )

    config = type(
        "Config",
        (),
        {
            "pipeline_json": tmp_path / "pipeline.json",
            "image_type": "Image",
            "slurm_args": {},
            "gpu_slurm_args": {},
            "gpu_shards": 1,
            "ext": None,
            "overlay_alpha": 0.3,
            "include_dataset_column": False,
            "metadata_csv": None,
            "no_qc": False,
            "input_path": tmp_path / "inputs",
            "resume": False,
            "restart": False,
            "staged_resume_phase": resume_phase,
            "staged_finalizer_only": finalizer_only,
            "staged_stage3_markers": True,
            "wait": False,
            "full_dataset_inventory": {},
            "nrows": None,
            "ncols": None,
        },
    )()
    strategy = object.__new__(StagedSlurmStrategy)
    strategy.config = config
    strategy.execute([], tmp_path)

    assert captured["array_limit"] == 3
    assert submitted_roles == ["controller-initial"]
    state = load_orchestration_state(tmp_path)
    assert state is not None
    assert state["phase"] == expected_phase
    assert state["stage1_index"] == 0
    assert state["stage3_index"] == stage3_index
    assert state["active_job_id"] is None


def test_strategy_rejects_max_submit_limit_below_three(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_slurm.get_slurm_array_limit", lambda: 1000
    )
    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_slurm.get_slurm_max_submit_jobs", lambda: 2
    )
    config = SimpleNamespace(gpu_shards=1)
    strategy = object.__new__(StagedSlurmStrategy)
    strategy.config = config
    with pytest.raises(ValueError, match="at least 3"):
        strategy.execute([], tmp_path)


def test_staged_job_metadata_supports_canonical_finalizer(tmp_path):
    images = [tmp_path / f"image_{index}.tif" for index in range(3)]
    datasets = [
        Dataset(
            name="plate",
            images=images,
            input_dir=tmp_path,
            output_dir=tmp_path,
        )
    ]
    scripts = {
        "stage1": [Path("s1c0.sh"), Path("s1c1.sh")],
        "stage2": Path("s2.sh"),
        "stage3": [Path("s3c0.sh"), Path("s3c1.sh")],
        "finalizer": Path("finalizer.sh"),
        "controller": Path("controller.sh"),
    }
    config = SimpleNamespace(
        include_dataset_column=True,
        metadata_csv=tmp_path / "metadata.csv",
        input_path=tmp_path / "inputs",
        no_qc=True,
        pipeline_json=tmp_path / "pipeline.json",
        image_type="Image",
        nrows=None,
        ncols=None,
        full_dataset_inventory={"plate": [image.name for image in images]},
    )

    metadata_path = _write_staged_job_metadata(
        datasets=datasets,
        output_dir=tmp_path,
        config=config,
        scripts=scripts,
        job_ids=["s1a"],
        start_time=datetime(2026, 7, 16),
        epoch="epoch-1",
    )

    assert metadata_path == progress_dir(tmp_path) / JOB_METADATA_JSON
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["datasets"]["plate"]["total"] == 3
    assert metadata["chunk_job_ids"] == {"0": "s1a"}
    assert metadata["chunk_scripts"][-1] == "finalizer.sh"
    assert metadata["no_qc"] is True
    assert metadata["image_task_mapping"] == {}
