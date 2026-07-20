"""Shard partitioning + per-stage SLURM args resolution (Spec 1 §7, Plan 3)."""

from pathlib import Path

from phenotypic._cli._cli_staged_slurm import (
    STAGE_DEPENDENCY,
    StagedSlurmStrategy,
    flatten_staged_scripts,
    generate_staged_scripts,
    partition_shards,
    resolve_stage_slurm_args,
    submit_staged_chain,
)


def _manifest(n):
    return [("ds", f"img{i}") for i in range(n)]


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
    # Image-arrayed stages return a list of chunk scripts (one chunk here);
    # the GPU stage is a single shard-arrayed script.
    s1 = scripts["stage1"][0].read_text(encoding="utf-8")
    s2 = scripts["stage2"].read_text(encoding="utf-8")
    s3 = scripts["stage3"][0].read_text(encoding="utf-8")

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
    s2 = scripts["stage2"].read_text(encoding="utf-8")
    assert "--signal=B:TERM@120" in s2
    # --requeue lets the worker requeue its own array task on the SIGTERM, so
    # Stage 3's afterany dependency waits for the continuation.
    assert "--requeue" in s2
    # only Stage 2 carries the walltime-survival directives
    assert "--requeue" not in scripts["stage1"][0].read_text(encoding="utf-8")


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
    )
    c0, c1, c2 = (s.read_text(encoding="utf-8") for s in scripts["stage1"])
    # The worker resolves the ABSOLUTE manifest index via CURRENT_TASK_INDEX,
    # populated from the per-chunk TASK_INDICES window.
    assert "--index $CURRENT_TASK_INDEX" in c0
    assert "    0\n    1" in c0  # window [0, 2)
    assert "    2\n    3" in c1  # window [2, 4)
    assert "    4" in c2         # window [4, 5)


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
    )
    assert [p.name for p in scripts["stage1"]] == ["stage1.sh"]
    assert [p.name for p in scripts["stage3"]] == ["stage3.sh"]


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
    )
    assert [p.name for p in scripts["stage1"]] == [
        "stage1_chunk0.sh", "stage1_chunk1.sh", "stage1_chunk2.sh",
    ]


def test_flatten_staged_scripts_orders_chunks_then_stages():
    scripts = {
        "stage1": [Path("s1c0.sh"), Path("s1c1.sh")],
        "stage2": Path("s2.sh"),
        "stage3": [Path("s3c0.sh"), Path("s3c1.sh")],
    }
    # Order encodes the stage dependencies for the drip-feed dispatcher:
    # every Stage-1 chunk, then Stage 2, then every Stage-3 chunk.
    assert [p.name for p in flatten_staged_scripts(scripts)] == [
        "s1c0.sh", "s1c1.sh", "s2.sh", "s3c0.sh", "s3c1.sh",
    ]


def test_submit_staged_chain_uses_drip_feed_dispatcher(monkeypatch):
    captured = {}

    class _FakeSubmission:
        job_ids = ["chunk0", "dispatch1"]

    def _fake_chain(*, flat_chunk_scripts, output_dir, slurm_args, console):
        captured["flat"] = [Path(p).name for p in flat_chunk_scripts]
        captured["slurm_args"] = slurm_args
        return _FakeSubmission()

    # Patch the shared drip-feed funnel at its definition site (submit_staged_chain
    # imports it locally, so the lookup resolves to this patched attribute).
    import phenotypic._cli._cli_slurm_submission as sub_mod

    monkeypatch.setattr(sub_mod, "submit_slurm_script_chain", _fake_chain)

    scripts = {
        "stage1": [Path("s1c0.sh"), Path("s1c1.sh")],
        "stage2": Path("s2.sh"),
        "stage3": [Path("s3c0.sh")],
    }
    job_ids = submit_staged_chain(
        scripts, output_dir=Path("/out"), slurm_args={"slurm_partition": "short"}
    )

    # Only the drip-feed's initial jobs come back (chunk 0 + dispatcher 1), NOT
    # one job per chunk — the rest are auto-submitted by each chunk's dispatcher.
    assert job_ids == ["chunk0", "dispatch1"]
    assert captured["flat"] == ["s1c0.sh", "s1c1.sh", "s2.sh", "s3c0.sh"]
    # The tiny dispatcher runs on the CPU (short) profile.
    assert captured["slurm_args"] == {"slurm_partition": "short"}


def test_strategy_reserves_max_submit_slot_for_dispatcher(
    monkeypatch, tmp_path
):
    """The active array plus dispatcher must fit MaxSubmitJobs."""
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
        }

    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_slurm.generate_staged_scripts",
        _fake_generate,
    )
    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_slurm.submit_staged_chain",
        lambda *args, **kwargs: ["chunk0", "dispatch1"],
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
        },
    )()
    strategy = object.__new__(StagedSlurmStrategy)
    strategy.config = config
    strategy.execute([], tmp_path)

    assert captured["array_limit"] == 4
