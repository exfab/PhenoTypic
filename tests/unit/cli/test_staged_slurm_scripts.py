"""Shard partitioning + per-stage SLURM args resolution (Spec 1 §7, Plan 3)."""

from phenotypic._cli._cli_staged_slurm import (
    STAGE_DEPENDENCY,
    generate_staged_scripts,
    partition_shards,
    resolve_stage_slurm_args,
)


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
        gpu_slurm_args={"slurm_partition": "exfab", "slurm_account": "exfab_acct"},
        cpu_slurm_args={"slurm_partition": "short", "slurm_qos": "normal"},
    )
    assert args["slurm_partition"] == "exfab"      # gpu overrides cpu
    assert args["slurm_account"] == "exfab_acct"   # added for the gpu partition
    assert args["slurm_qos"] == "normal"           # inherited from --slurm
    assert args["slurm_gpus_per_node"] == 1


def test_generates_three_stage_scripts_with_correct_resources(tmp_path):
    scripts = generate_staged_scripts(
        pipeline_path=tmp_path / "p.json",
        datasets_manifest=[("ds", "a"), ("ds", "b"), ("ds", "c")],
        output_dir=tmp_path,
        image_type="Image",
        cpu_slurm_args={"slurm_partition": "batch", "slurm_cpus_per_task": 4},
        gpu_slurm_args={"slurm_partition": "gpu"},
        n_shards=2,
    )
    assert set(scripts) == {"stage1", "stage2", "stage3"}
    s1 = scripts["stage1"].read_text(encoding="utf-8")
    s2 = scripts["stage2"].read_text(encoding="utf-8")
    s3 = scripts["stage3"].read_text(encoding="utf-8")

    # Stage 1 & 3 on the CPU partition; Stage 2 on the GPU partition + 1 GPU
    assert "--partition=batch" in s1 and "--partition=batch" in s3
    assert "--partition=gpu" in s2 and "--gpus-per-node=1" in s2
    # Stage 1/3 = array over images (0-2); Stage 2 = array over shards (0-1)
    assert "--array=0-2" in s1 and "--array=0-2" in s3
    assert "--array=0-1" in s2
    # Stage 2 invokes the shard worker
    assert "_cli_staged_slurm_worker" in s2


def test_chain_uses_afterany_dependencies():
    assert STAGE_DEPENDENCY == "afterany"


def test_stage2_script_carries_signal_directive(tmp_path):
    scripts = generate_staged_scripts(
        pipeline_path=tmp_path / "p.json",
        datasets_manifest=[("ds", "a"), ("ds", "b")],
        output_dir=tmp_path,
        image_type="Image",
        cpu_slurm_args={"slurm_partition": "batch"},
        gpu_slurm_args={"slurm_partition": "gpu"},
        n_shards=1,
        signal_grace=120,
    )
    assert "--signal=B:TERM@120" in scripts["stage2"].read_text(encoding="utf-8")
