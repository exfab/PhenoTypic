"""The staged SLURM worker's ``--stage 4`` and single-pass "stage3" role (#282 P5 T2). [torch]

Run in the foundation venv (``TPT``); in the shared env every test skips.
Each test runs on :func:`~tests._tiny_run.stage4_slurm_tree`: two images that
finished Stage 3 of a no-detector run, under a live SLURM epoch, with the
round-1 worklist's shards 0 and 1 holding one image each and 2 and 3 empty.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

pytest.importorskip("torch")

from phenotypic._cli import _cli_stage4_worker  # noqa: E402
from phenotypic._cli._cli_completion import valid_image_success  # noqa: E402
from phenotypic._cli._cli_failure_tracker import read_terminal_failures  # noqa: E402
from phenotypic._cli._cli_staged_slurm_worker import (  # noqa: E402
    main,
    run_single_pass_step,
    run_stage4_shard,
)
from phenotypic.measure import MeasureDinoEmbedding  # noqa: E402
from phenotypic.sdk_ import read_image_record  # noqa: E402
from tests._tiny_run import (  # noqa: E402
    RECORDED_REVISION,
    first_entry,
    install_stub_backbone,
    invalidate_store,
    record_is_valid,
    run_items,
    signal_files,
    store_has_embeddings,
)


def _record(tree, index: int) -> dict:
    item = tree.items[index]
    return read_image_record(tree.out, item.dataset, item.stem)


def test_an_empty_shard_exits_zero(stage4_slurm_tree, stage4_cpu):
    assert run_stage4_shard(**stage4_slurm_tree.shard_kwargs(shard_index=3)) == 0
    assert stage4_cpu.load_calls == []                                    # nothing loaded


def test_a_shard_publishes_its_image_under_the_real_epoch(stage4_slurm_tree, stage4_cpu):
    tree = stage4_slurm_tree
    assert run_stage4_shard(**tree.shard_kwargs(shard_index=0)) == 0
    first, second = tree.items
    assert record_is_valid(tree.out, first) and store_has_embeddings(tree.out, first)
    assert not record_is_valid(tree.out, second)                          # shard 1's image
    assert _record(tree, 0)["lifecycle_epoch"] == tree.epoch              # never "local-unfenced"
    assert signal_files(tree.out, first) == []


def test_an_unset_cpu_count_does_not_raise(stage4_slurm_tree, stage4_cpu, monkeypatch):
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    assert run_stage4_shard(**stage4_slurm_tree.shard_kwargs(shard_index=0)) == 0


def test_the_loader_workers_come_from_the_allocation(stage4_slurm_tree, stage4_cpu, monkeypatch):
    seen = {}
    real = _cli_stage4_worker.run_stage4

    def _spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **{**kwargs, "num_workers": 0})

    monkeypatch.setattr(_cli_stage4_worker, "run_stage4", _spy)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "5")
    assert run_stage4_shard(**stage4_slurm_tree.shard_kwargs(shard_index=0)) == 0
    # One image: resolve_local_worker_count(-1, 1) is 1, so 0 workers, in-process.
    assert seen["num_workers"] == 0
    assert seen["epoch"] == stage4_slurm_tree.epoch
    assert seen["commit_guard"] is not None and seen["active_check"] is not None
    assert seen["revisions"] == {"facebook/dinov2-base": RECORDED_REVISION}


def test_every_shard_loads_the_recorded_revision(stage4_slurm_tree, stage4_cpu):
    for i in range(2):
        assert run_stage4_shard(**stage4_slurm_tree.shard_kwargs(shard_index=i)) == 0
    assert {c["revision"] for c in stage4_cpu.load_calls} == {RECORDED_REVISION}
    assert all(record_is_valid(stage4_slurm_tree.out, item) for item in stage4_slurm_tree.items)


def test_u5_records_a_revision_when_commit_hash_is_absent(stage4_slurm_tree, monkeypatch):
    # transformers >= 5.18: no config._commit_hash; the stub lays out a snapshot path instead
    install_stub_backbone(monkeypatch, parametric=True, commit_hash=None, snapshot_from_revision=True)
    monkeypatch.setenv("PHENOTYPIC_STAGE4_DEVICE", "cpu")
    assert run_stage4_shard(**stage4_slurm_tree.shard_kwargs(shard_index=0)) == 0
    assert first_entry(stage4_slurm_tree.out)["model"]["revision"] == RECORDED_REVISION


def test_the_params_digest_is_the_inline_one(stage4_slurm_tree, stage4_cpu):
    assert run_stage4_shard(**stage4_slurm_tree.shard_kwargs(shard_index=0)) == 0
    assert first_entry(stage4_slurm_tree.out)["producer"]["params_sha256"] == MeasureDinoEmbedding()._params_sha256()


def test_a_fenced_epoch_publishes_nothing(stage4_slurm_tree, stage4_cpu):
    from phenotypic._cli._cli_staged_orchestration import deactivate_orchestration
    from phenotypic._cli._cli_slurm_lifecycle import SlurmGenerationInactiveError

    tree = stage4_slurm_tree
    deactivate_orchestration(tree.out, "cancelled")
    with pytest.raises(SlurmGenerationInactiveError):
        run_stage4_shard(**tree.shard_kwargs(shard_index=0))
    assert not any(record_is_valid(tree.out, item) for item in tree.items)
    assert stage4_cpu.load_calls == []


@pytest.mark.parametrize("error", ["device", "oom", "host"])
def test_node_state_exits_1_and_journals_nothing(stage4_slurm_tree, stage4_cpu, monkeypatch, error, capsys):
    from phenotypic.detect.nn._helper._batch_encoder import Stage4OutOfMemory

    raised = {"device": _cli_stage4_worker.Stage4DeviceError("no CUDA"),
              "oom": Stage4OutOfMemory("m = 1"), "host": MemoryError()}[error]

    def _fail(*args, **kwargs):
        raise raised

    monkeypatch.setattr(_cli_stage4_worker, "run_stage4", _fail)
    assert run_stage4_shard(**stage4_slurm_tree.shard_kwargs(shard_index=0)) == 1
    assert read_terminal_failures(stage4_slurm_tree.out) == []
    assert "No image was recorded as failed" in capsys.readouterr().err


def test_an_unexpected_error_propagates(stage4_slurm_tree, stage4_cpu, monkeypatch):
    def _bug(*args, **kwargs):
        raise KeyError("a bug, not node state")

    monkeypatch.setattr(_cli_stage4_worker, "run_stage4", _bug)
    with pytest.raises(KeyError):
        run_stage4_shard(**stage4_slurm_tree.shard_kwargs(shard_index=0))


def test_main_dispatches_stage_4(stage4_slurm_tree, stage4_cpu):
    tree = stage4_slurm_tree
    argv = [
        "--stage", "4", "--pipeline", str(tree.pipeline), "--output-dir", str(tree.out),
        "--image-type", tree.image_type, "--manifest", str(tree.out / "unused-manifest.json"),
        "--index", "1", "--epoch", tree.epoch, "--worklist", str(tree.worklist),
    ]
    assert main(argv) == 0
    assert record_is_valid(tree.out, tree.items[1]) and not record_is_valid(tree.out, tree.items[0])


def test_main_refuses_stage_4_without_a_worklist(stage4_slurm_tree):
    tree = stage4_slurm_tree
    with pytest.raises(SystemExit) as raised:
        main(["--stage", "4", "--pipeline", str(tree.pipeline), "--output-dir", str(tree.out),
              "--manifest", "m.json", "--index", "0", "--epoch", tree.epoch])
    assert raised.value.code == 2


# ---------------------------------------------------------------------------
# The single-pass "stage3" role
# ---------------------------------------------------------------------------


def test_the_single_pass_role_defers_and_marks_stage3(stage4_slurm_tree, stage4_cpu):
    tree = stage4_slurm_tree
    item = tree.items[0]
    invalidate_store(tree.out, item)                       # Stages 1-3 must run again
    before = (tree.store(0) / "zarr.json").read_bytes()
    run_single_pass_step(**tree.single_pass_kwargs(index=0))
    root = json.loads((tree.store(0) / "zarr.json").read_text(encoding="utf-8"))
    assert root["attributes"]["phenotypic"]["work_id"] == item.work_id
    assert (tree.store(0) / "zarr.json").read_bytes() != before
    record = _record(tree, 0)
    assert set(record["stages"]) == {"stage3"} and not record.get("artifacts")
    assert not store_has_embeddings(tree.out, item)        # deferred to Stage 4
    assert stage4_cpu.load_calls == []                     # no backbone in Stage 3


def test_the_single_pass_role_skips_a_finished_image(stage4_slurm_tree, stage4_cpu, monkeypatch):
    from phenotypic._cli import _cli_staged_slurm_worker

    tree = stage4_slurm_tree
    root = (tree.store(0) / "zarr.json").read_bytes()
    monkeypatch.setattr(_cli_staged_slurm_worker, "process_single_image_core",
                        lambda *a, **k: pytest.fail("re-ran the core"))
    run_single_pass_step(**tree.single_pass_kwargs(index=0))      # the continuation
    assert (tree.store(0) / "zarr.json").read_bytes() == root
    assert run_stage4_shard(**tree.shard_kwargs(shard_index=0)) == 0
    published = (tree.store(0) / "zarr.json").read_bytes()
    run_single_pass_step(**tree.single_pass_kwargs(index=0))      # after its record
    assert (tree.store(0) / "zarr.json").read_bytes() == published


def test_the_single_pass_role_journals_a_scientific_failure_under_the_epoch(stage4_slurm_tree, stage4_cpu):
    tree = stage4_slurm_tree
    item = tree.items[0]
    invalidate_store(tree.out, item)
    entries = tree.manifest()
    broken = entries[0].__class__(**{**entries[0].__dict__, "input_path": str(tree.out / "missing.tif")})
    kwargs = {**tree.single_pass_kwargs(index=0), "manifest": [broken, *entries[1:]]}
    with pytest.raises(Exception):
        run_single_pass_step(**kwargs)
    failures = read_terminal_failures(tree.out)
    assert [(f.work_id, f.lifecycle_epoch) for f in failures] == [(item.work_id, tree.epoch)]


def test_main_dispatches_the_single_pass_role(stage4_slurm_tree, stage4_cpu, tmp_path):
    from phenotypic._cli._cli_staged_orchestration import write_staged_manifest

    tree = stage4_slurm_tree
    invalidate_store(tree.out, tree.items[1])
    manifest = tmp_path / "manifest.json"
    write_staged_manifest(manifest, tree.manifest())
    assert main(["--stage", "3", "--stage3-role", "single_pass", "--pipeline", str(tree.pipeline),
                 "--output-dir", str(tree.out), "--image-type", tree.image_type,
                 "--manifest", str(manifest), "--index", "1", "--epoch", tree.epoch]) == 0
    assert set(_record(tree, 1)["stages"]) == {"stage3"}
    assert not valid_image_success(tree.out, dataset=tree.items[1].dataset,
                                   image_stem=tree.items[1].stem, work_id=tree.items[1].work_id)


def test_the_tree_is_what_the_fixture_says(stage4_slurm_tree):
    """The fixture's own control: both images wait on Stage 4, under the live epoch."""
    from phenotypic._cli._cli_staged_orchestration import assert_active_epoch, load_stage4_worklist
    from phenotypic._cli._cli_staged_resume import stage4_status

    tree = stage4_slurm_tree
    assert_active_epoch(tree.out, tree.epoch)
    assert [stage4_status(tree.out, i.dataset, i.stem, i.work_id) for i in run_items(tree.out)] == ["pending"] * 2
    assert [len(load_stage4_worklist(tree.worklist, i).entries) for i in range(4)] == [1, 1, 0, 0]


# ---------------------------------------------------------------------------
# The CLI: the SLURM no-detector continuation, and U5 on a local continuation
# ---------------------------------------------------------------------------


def _slurm_continuation(tree, monkeypatch):
    """Run the same no-detector command again on SLURM with ``--gpu-slurm``; return the config."""
    from phenotypic._cli._cli_types import ExecutionResults
    from tests._tiny_run import run_cli

    seen = {}

    class _Submitted:
        def execute(self, datasets, output_dir):
            seen["datasets"] = datasets
            now = datetime.now()
            return ExecutionResults(datasets={}, total_images=0, total_completed=0, total_failed=0,
                                    execution_mode="slurm", start_time=now, end_time=now, submitted=True)

    def _strategy(config, output_manager):
        seen["config"] = config
        return _Submitted()

    monkeypatch.setattr("phenotypic.phenotypicCLI.create_execution_strategy", _strategy)
    result = run_cli(*tree.cli_args, **tree.cli_kwargs, slurm=True,
                     extra=["--embed-stage", "gpu", "--gpu-slurm", "partition=exfab", "--skip-validation"])
    assert "config" in seen, result.output
    return seen, result


def test_an_all_stage4_slurm_no_detector_continuation_starts_at_stage4(staged_tree_single_pass, monkeypatch):
    tree = staged_tree_single_pass
    seen, result = _slurm_continuation(tree, monkeypatch)
    config = seen["config"]
    assert config.staged_resume_phase == "stage4", result.output
    # The split's identities are handed on, so the submitter hashes nothing again.
    entries = config.staged_manifest_entries
    assert sorted(e.work_id for e in entries.values()) == sorted(i.work_id for i in tree.items)
    assert "2 image(s) need Stage 4 only" in result.output


def test_a_mixed_slurm_no_detector_continuation_starts_at_stage3(staged_tree_single_pass, monkeypatch):
    tree = staged_tree_single_pass
    invalidate_store(tree.out, tree.items[0])            # this one needs the core again
    seen, result = _slurm_continuation(tree, monkeypatch)
    assert seen["config"].staged_resume_phase is None, result.output
    assert "1 image(s) need Stage 4 only" in result.output
    assert sum(len(d.images) for d in seen["datasets"]) == 2   # both stay in the manifest


def test_a_local_continuation_reuses_the_recorded_revision(tmp_path, tiny_inputs, stage4_cpu, monkeypatch):
    from phenotypic._cli import _cli_embed_stage as embed_stage_module
    from tests._tiny_run import run_cli, stage4_stubbed_out, with_stage4_dino_objects

    monkeypatch.setattr(embed_stage_module, "_hub_commit", lambda hf_id: "c" * 40)
    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path, fake_gpu=True)
    with stage4_stubbed_out(monkeypatch):
        first = run_cli(tiny_inputs, out, pipeline=pipe, extra=["--embed-stage", "gpu"])
    assert "Traceback" not in first.output, first.output
    monkeypatch.setattr(embed_stage_module, "_hub_commit", lambda hf_id: pytest.fail("re-resolved"))
    result = run_cli(tiny_inputs, out, pipeline=pipe, extra=["--embed-stage", "gpu"])
    assert result.exit_code == 0, result.output
    assert {c["revision"] for c in stage4_cpu.load_calls} == {"c" * 40}
    state = json.loads((out / ".phenotypic" / "processing_state.json").read_text(encoding="utf-8"))
    assert state["config"]["embed_revisions"] == {"facebook/dinov2-base": "c" * 40}
