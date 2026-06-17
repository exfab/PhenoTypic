"""Shard partitioning + per-stage SLURM args resolution (Spec 1 §7, Plan 3)."""

from phenotypic._cli._cli_staged_slurm import (
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
