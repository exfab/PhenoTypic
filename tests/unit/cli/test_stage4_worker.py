"""The Stage-4 worker: backbone-major, per-key signals at the live root, commit then record (#282 P4 T3). [torch]

Every test runs ``run_stage4`` (or the CLI) over a tree whose images finished
Stages 1-3 under ``--embed-stage gpu`` and wait on Stage 4 (``staged_tree``
and its variants), on the CPU (``stage4_cpu``: the parametric stub backbone,
``PHENOTYPIC_STAGE4_DEVICE=cpu``, CUDA hidden). Per-image faults are planted
on disk (``plant_stage4_fault``), never by monkeypatch, so a spawned loader
worker sees them. Run in the foundation venv (``TPT``); in the shared env the
module skips.
"""

from __future__ import annotations

import pickle

import numpy as np
import pyarrow.parquet as pq
import pytest

pytest.importorskip("torch")

from phenotypic._cli._cli_slurm_lifecycle import SlurmGenerationInactiveError  # noqa: E402
from phenotypic._cli._cli_stage4_worker import (  # noqa: E402
    Stage4DeviceError,
    Stage4ShardError,
    sweep_orphan_stage4_signals,
)
from phenotypic.measure import MeasureDinoEmbedding  # noqa: E402
from tests._tiny_run import (  # noqa: E402
    STUB_HIDDEN_SIZE,
    _built_staged_tree,
    _stage4_pipeline_file,
    capture_dataloader_kwargs,
    corrupt_one_store,
    embedding_path,
    failure_stage,
    fence_after_first_signal,
    identity_of,
    inventory_and_work_ids,
    n_images,
    plant_stage4_fault,
    plant_stage4_signals,
    run_cli,
    run_entry,
    run_stage4_over,
    signal_files,
    spy_calls_by_image,
    stage4_dataset_for,
    stage4_loader_for,
    stage4_signal_dir,
    stage4_status_of,
    stores_of,
    terminal_failures,
    touch_root_before_commit,
    valid_record,
    with_stage4_dino_objects,
    write_valid_signals,
)

OBJECTS = "dinov2-vitb14/rgb/objects"
GPU = ["--embed-stage", "gpu"]
ORDER = ["replace_run_embeddings", "publish_image_success", "delete_stage4_signals"]


def _all_pending(tree) -> bool:
    return all(stage4_status_of(tree, item) == "pending" for item in tree.items)


def _nothing_committed(tree) -> bool:
    return all(not run_entry(store).get("embeddings") for store in stores_of(tree))


# ---------------------------------------------------------------------------
# The commit path: order, reuse, the root, the binds
# ---------------------------------------------------------------------------


def test_commit_then_record_then_signals_per_image(staged_tree, stage4_cpu, monkeypatch):
    calls = spy_calls_by_image(monkeypatch, staged_tree, ORDER)
    outcome = run_stage4_over(staged_tree)
    assert len(outcome.published) == n_images(staged_tree)
    assert set(calls) == set(outcome.published)
    assert all(sequence == ORDER for sequence in calls.values()), calls
    assert all(valid_record(staged_tree, item) for item in staged_tree.items)
    assert not any(stage4_signal_dir(staged_tree).rglob("*.parquet"))


def test_a_published_store_holds_its_embeddings_fresh(staged_tree, stage4_cpu):
    """An independent reader (zarr-python, pyarrow) sees the committed table and entry."""
    import zarr

    run_stage4_over(staged_tree)
    for store in stores_of(staged_tree):
        root = zarr.open_group(str(store), mode="r").attrs.asdict()
        runs = root["phenotypic"]["tables"]["runs"]
        (entry,) = [run["embeddings"][OBJECTS] for run in runs.values() if "embeddings" in run]
        assert entry["stale"] is False
        assert entry["compute"]["stage"] == "stage4"
        table = pq.read_table(embedding_path(store, OBJECTS))
        assert table.num_rows > 0
        assert table.column("embedding").type.list_size == STUB_HIDDEN_SIZE


def test_backbone_major_one_load_per_backbone(staged_tree_two_backbones, stage4_cpu):
    outcome = run_stage4_over(staged_tree_two_backbones)
    assert [(c["dino_size"], c["dtype"]) for c in stage4_cpu.load_calls] == [
        ("base", "float32"), ("small", "float32"),
    ]
    assert len(outcome.published) == n_images(staged_tree_two_backbones)
    for store in stores_of(staged_tree_two_backbones):
        assert set(run_entry(store)["embeddings"]) == {OBJECTS, "dinov2-vits14/rgb/objects"}


def test_objects_and_grid_share_one_load(staged_tree_objects_and_grid, stage4_cpu):
    outcome = run_stage4_over(staged_tree_objects_and_grid)
    assert len(stage4_cpu.load_calls) == 1
    assert len(outcome.published) == n_images(staged_tree_objects_and_grid)
    for store in stores_of(staged_tree_objects_and_grid):
        embeddings = run_entry(store)["embeddings"]
        assert set(embeddings) == {OBJECTS, "dinov2-vitb14/rgb/grid"}
        assert not any(entry["stale"] for entry in embeddings.values())


def test_a_second_round_embeds_only_the_keys_it_lacks(staged_tree_two_backbones, stage4_cpu):
    """Plan review C4: backbone 1's signals survive a walltime, so only backbone 2 loads."""
    assert write_valid_signals(staged_tree_two_backbones, keys=[OBJECTS]) == 2
    stage4_cpu.load_calls.clear()   # the inline payloads above loaded through the cache
    outcome = run_stage4_over(staged_tree_two_backbones)
    assert [c["dino_size"] for c in stage4_cpu.load_calls] == ["small"]
    assert len(outcome.published) == n_images(staged_tree_two_backbones)
    assert outcome.signals_written == n_images(staged_tree_two_backbones)


def test_all_valid_signals_commit_without_a_load(staged_tree, stage4_cpu, monkeypatch):
    """Every key valid: commit only, nothing embedded."""
    write_valid_signals(staged_tree)
    stage4_cpu.load_calls.clear()
    calls = spy_calls_by_image(monkeypatch, staged_tree, ["write_stage4_signal", *ORDER])
    outcome = run_stage4_over(staged_tree)
    assert stage4_cpu.load_calls == []
    assert len(outcome.published) == n_images(staged_tree)
    assert all(sequence == ORDER for sequence in calls.values()), calls


def test_a_signal_from_an_old_root_is_deleted_and_reembedded(staged_tree, stage4_cpu):
    write_valid_signals(staged_tree, root_sha256="0" * 64)    # before a reprocess
    stage4_cpu.load_calls.clear()
    outcome = run_stage4_over(staged_tree)
    assert len(stage4_cpu.load_calls) == 1 and len(outcome.published) == n_images(staged_tree)
    assert outcome.signals_written == n_images(staged_tree)


def test_a_root_change_at_commit_leaves_the_image_pending(staged_tree, stage4_cpu, monkeypatch):
    touched = touch_root_before_commit(monkeypatch, staged_tree, image=0)
    outcome = run_stage4_over(staged_tree)
    assert len(touched) == 1
    (pending,) = outcome.pending
    assert pending in {identity_of(staged_tree, i) for i in range(n_images(staged_tree))}
    assert len(outcome.published) == n_images(staged_tree) - 1
    assert terminal_failures(staged_tree) == []                     # never terminal
    item = next(i for i, e in zip(staged_tree.items, staged_tree.manifest()) if e.identity == pending)
    assert not signal_files(staged_tree.out, item)                  # its signals deleted


def test_signals_from_two_roots_leave_the_image_pending(staged_tree_two_backbones, stage4_cpu, monkeypatch):
    """The commit's own root check: one image's keys computed against two roots never commit.

    Image 0's root changes between the two backbones' loaders, so its base
    signal is bound to the old root and its small one to the new. Its store
    is left as it was and the image stays pending; image 1 is the control.
    """
    from phenotypic.measure import _measure_dino_embedding as module
    from tests._tiny_run import touch_root

    tree = staged_tree_two_backbones
    loaded = module.load_dino_backbone

    def _touch_before_the_second_load(*args, **kwargs):
        if args[1] == "small":
            touch_root(stores_of(tree)[0])
        return loaded(*args, **kwargs)

    monkeypatch.setattr(module, "load_dino_backbone", _touch_before_the_second_load)
    outcome = run_stage4_over(tree)
    assert outcome.pending == [identity_of(tree, 0)]
    assert outcome.published == [identity_of(tree, 1)]
    assert not run_entry(stores_of(tree)[0]).get("embeddings")
    assert terminal_failures(tree) == []


def test_stale_signals_are_deleted_before_the_first_load(staged_tree, stage4_cpu, monkeypatch):
    """A stale signal goes at classification, so a shard that then fails leaves none behind."""
    from phenotypic.measure import _measure_dino_embedding as module

    write_valid_signals(staged_tree, root_sha256="0" * 64)
    assert any(stage4_signal_dir(staged_tree).rglob("*.parquet"))

    def _no_load(*args, **kwargs):
        raise MemoryError("the host ran out loading the backbone")

    monkeypatch.setattr(module, "load_dino_backbone", _no_load)
    with pytest.raises(MemoryError):
        run_stage4_over(staged_tree)
    assert not any(stage4_signal_dir(staged_tree).rglob("*.parquet"))
    assert terminal_failures(staged_tree) == [] and _all_pending(staged_tree)


def test_a_local_run_left_pending_exits_1(tmp_path, tiny_inputs, stage4_cpu, monkeypatch):
    touch_root_before_commit(monkeypatch, None, image=0)            # inside the CLI run
    result = run_cli(tiny_inputs, tmp_path / "out", pipeline=with_stage4_dino_objects(tmp_path, fake_gpu=True),
                     extra=GPU)
    assert result.exit_code == 1, result.output
    assert "still need Stage 4" in result.output


def test_a_payload_bound_to_another_image_fails_the_image(staged_tree, stage4_cpu):
    write_valid_signals(staged_tree, binds_from_another_objmap=True)   # checked against the live store
    outcome = run_stage4_over(staged_tree)
    assert len(outcome.failed) == n_images(staged_tree)
    assert _nothing_committed(staged_tree)                          # never published-then-dropped
    assert not any(valid_record(staged_tree, item) for item in staged_tree.items)
    assert {failure_stage(staged_tree, item) for item in staged_tree.items} == {"stage4"}


def test_an_image_with_no_colonies_commits_an_empty_table(staged_tree_one_empty_plate, stage4_cpu, monkeypatch):
    """README Review Focus 1: no forward pass for it, an empty table, and its record."""
    from phenotypic.detect.nn._helper import _batch_encoder

    tree = staged_tree_one_empty_plate
    submitted: list = []
    original = _batch_encoder.BatchEncoder.submit

    def _spy(self, ticket, inputs):
        submitted.append(ticket[0])
        return original(self, ticket, inputs)

    monkeypatch.setattr(_batch_encoder.BatchEncoder, "submit", _spy)
    outcome = run_stage4_over(tree)
    empty = identity_of(tree, 0)
    assert empty in outcome.published and empty not in submitted
    assert identity_of(tree, 1) in submitted                       # the control did reach the encoder
    store = stores_of(tree)[0]
    assert pq.read_table(embedding_path(store, OBJECTS)).num_rows == 0
    assert run_entry(store)["embeddings"][OBJECTS]["compute"]["physical_batch"] is None


# ---------------------------------------------------------------------------
# Per-image failures: the loader, a corrupt store, a non-finite vector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_workers", [0, 2])
def test_a_loader_worker_error_fails_only_its_image(staged_tree_faulty, stage4_cpu, num_workers):
    bad = staged_tree_faulty.items[0]
    plant_stage4_fault(bad.stem)                                   # on disk: a spawned worker sees it
    outcome = run_stage4_over(staged_tree_faulty, num_workers=num_workers)
    assert set(outcome.failed) == {identity_of(staged_tree_faulty, 0)}
    assert outcome.failed[identity_of(staged_tree_faulty, 0)] == "ValueError"
    assert len(outcome.published) == n_images(staged_tree_faulty) - 1
    assert failure_stage(staged_tree_faulty, bad) == "stage4"


def test_a_corrupt_store_fails_only_its_image(staged_tree, stage4_cpu):
    bad = corrupt_one_store(staged_tree)                           # deletes its objmap chunks
    outcome = run_stage4_over(staged_tree, num_workers=2)
    assert set(outcome.failed) == {identity_of(staged_tree, 0)}
    assert failure_stage(staged_tree, bad) == "stage4"
    assert valid_record(staged_tree, staged_tree.items[1])


def test_a_non_finite_vector_fails_the_image(staged_tree, stage4_cpu, monkeypatch):
    monkeypatch.setattr(MeasureDinoEmbedding, "vectors_from",
                        lambda self, p, o: np.full((p.n_regions, STUB_HIDDEN_SIZE), 1e6, np.float32))
    outcome = run_stage4_over(staged_tree)
    assert len(outcome.failed) == n_images(staged_tree)
    assert _nothing_committed(staged_tree)
    assert {failure_stage(staged_tree, item) for item in staged_tree.items} == {"stage4"}


# ---------------------------------------------------------------------------
# The loader
# ---------------------------------------------------------------------------


def test_njobs_1_runs_the_loader_in_process(staged_tree, stage4_cpu, monkeypatch):
    seen = capture_dataloader_kwargs(monkeypatch)
    result = run_cli(*staged_tree.cli_args, **staged_tree.cli_kwargs, extra=GPU)   # run_cli passes --njobs 1
    assert result.exit_code == 0, result.output
    assert seen["num_workers"] == 0 and "multiprocessing_context" not in seen
    assert all(valid_record(staged_tree, item) for item in staged_tree.items)


def test_njobs_2_runs_one_spawned_loader_worker(staged_tree, stage4_cpu, monkeypatch):
    """``--njobs`` is honoured: two images, two jobs, so one worker beside this process."""
    seen = capture_dataloader_kwargs(monkeypatch)
    result = run_cli(*staged_tree.cli_args, **staged_tree.cli_kwargs, extra=[*GPU, "--njobs", "2"])
    assert result.exit_code == 0, result.output
    assert seen["num_workers"] == 1
    assert seen["multiprocessing_context"] == "spawn" and seen["persistent_workers"] is False
    assert seen["batch_size"] is None


def test_the_dataset_pickles_with_its_processor(staged_tree_dense, stage4_cpu):
    dataset = stage4_dataset_for(staged_tree_dense)
    assert dataset.processors                                       # a masked_dense backbone has one
    again = pickle.loads(pickle.dumps(dataset))                     # what spawn does
    assert again.processors.keys() == dataset.processors.keys()


def test_dense_tiles_through_a_spawned_worker(staged_tree_dense, stage4_cpu):
    """The pickled processor cuts the tiles in a worker process, and the images publish."""
    outcome = run_stage4_over(staged_tree_dense, num_workers=1)
    assert len(outcome.published) == n_images(staged_tree_dense)
    assert run_entry(stores_of(staged_tree_dense)[0])["embeddings"][OBJECTS]["pooling"] == "masked_dense"


def test_loader_items_arrive_as_ndarrays(staged_tree, stage4_cpu):
    item = next(iter(stage4_loader_for(staged_tree, num_workers=1)))
    assert isinstance(item.prepared[OBJECTS].inputs.arrays[0], np.ndarray)


def test_the_loader_hands_materialised_crops(staged_tree, stage4_cpu):
    """Step 3d: every crop input a worker returns is a plain ndarray, never deferred crops."""
    items = list(stage4_loader_for(staged_tree, num_workers=1))
    assert len(items) == n_images(staged_tree)
    for item in items:
        for prepared in item.prepared.values():
            assert all(type(array) is np.ndarray for array in prepared.inputs.arrays)


# ---------------------------------------------------------------------------
# Node and model state: never a per-image failure
# ---------------------------------------------------------------------------


def test_no_cuda_fails_the_stage_without_journalling_images(staged_tree, param_stub_backbone, monkeypatch):
    monkeypatch.delenv("PHENOTYPIC_STAGE4_DEVICE", raising=False)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(Stage4DeviceError, match="CUDA"):
        run_stage4_over(staged_tree)
    assert terminal_failures(staged_tree) == []
    assert _all_pending(staged_tree)


def test_dinov3_licence_is_refused_before_any_load(staged_tree_dinov3, stage4_cpu, monkeypatch):
    monkeypatch.delenv("PHENOTYPIC_ACCEPT_MODEL_LICENSE", raising=False)
    with pytest.raises(RuntimeError, match="licen"):
        run_stage4_over(staged_tree_dinov3)
    assert stage4_cpu.load_calls == []
    assert _all_pending(staged_tree_dinov3)


def test_a_patch_size_mismatch_fails_before_any_pass(staged_tree, stage4_cpu, monkeypatch):
    """Step 3a: the geometry check runs once per backbone after the load, and fails the shard."""
    from phenotypic.measure import _measure_dino_embedding as module

    loaded = module.load_dino_backbone

    def _sixteen(*args, **kwargs):
        model, processor = loaded(*args, **kwargs)
        model.config.patch_size = 16
        return model, processor

    monkeypatch.setattr(module, "load_dino_backbone", _sixteen)
    with pytest.raises(Stage4ShardError, match="patch size 16"):
        run_stage4_over(staged_tree)
    assert not any(stage4_signal_dir(staged_tree).rglob("*.parquet"))
    assert terminal_failures(staged_tree) == [] and _all_pending(staged_tree)


def test_a_failed_forward_pass_fails_the_shard_and_journals_nothing(tmp_path, tiny_inputs, stage4_cpu, monkeypatch):
    """Step 3e: a stub whose second pass raises poisons the encoder; every image stays pending.

    One pass a cell (``batch_size=1``) over a 2 x 2 grid, so the failed second
    pass holds a row of image 0, whose first cell already ran: no image is
    finished, and none is failed for a pass that is the node's.
    """
    from phenotypic.grid import ManualGridFinder
    from phenotypic.measure import _measure_dino_embedding as module
    from tests._tiny_run import TOY_GRID_COL_EDGES, TOY_GRID_ROW_EDGES

    grid = ManualGridFinder(row_edges=list(TOY_GRID_ROW_EDGES), col_edges=list(TOY_GRID_COL_EDGES))
    tree = _built_staged_tree(tmp_path, tiny_inputs, _stage4_pipeline_file(
        tmp_path, "stage4_m1", {"grid": MeasureDinoEmbedding(regions="grid", grid=grid, batch_size=1)}
    ))
    loaded = module.load_dino_backbone
    passes = {"n": 0}

    def _second_pass_fails(*args, **kwargs):
        model, processor = loaded(*args, **kwargs)
        forward = model.forward

        def _forward(pixel_values):
            passes["n"] += 1
            if passes["n"] == 2:
                raise ValueError("a driver fault in the second pass")
            return forward(pixel_values)

        model.forward = _forward
        return model, processor

    monkeypatch.setattr(module, "load_dino_backbone", _second_pass_fails)
    with pytest.raises(Stage4ShardError, match="driver fault"):
        run_stage4_over(tree)
    assert passes["n"] == 2                                          # it stopped at the poisoned pass
    assert terminal_failures(tree) == []
    assert [stage4_status_of(tree, item) for item in tree.items] == ["pending", "pending"]
    assert _nothing_committed(tree)


def test_a_cpu_stage4_on_a_gpu_node_needs_no_cublas_setting(staged_tree, stage4_cpu, monkeypatch):
    """Step 3b: ``configure_determinism`` gets the device, so CUDA being visible does not ask for cuBLAS."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    outcome = run_stage4_over(staged_tree)
    assert len(outcome.published) == n_images(staged_tree)


# ---------------------------------------------------------------------------
# What the entries record
# ---------------------------------------------------------------------------


def test_stage4_entries_record_stage4_compute(staged_tree, stage4_cpu):
    outcome = run_stage4_over(staged_tree)
    compute = run_entry(stores_of(staged_tree)[0])["embeddings"][OBJECTS]["compute"]
    assert compute["stage"] == "stage4" and compute["physical_batch"] >= 1
    assert compute["device"] == "cpu"
    ((backbone, queues),) = outcome.physical_batches.items()
    assert backbone.size == "base" and set(queues.values()) == {compute["physical_batch"]}


def test_two_crop_px_share_one_load_with_an_m_each(tmp_path, tiny_inputs, stage4_cpu):
    """Step 3c: one ``BackboneKey``, two geometries, two queues, each its own ``m``."""
    pipeline = _stage4_pipeline_file(tmp_path, "stage4_two_crops", {
        "small_crops": MeasureDinoEmbedding(crop_px=112, batch_size=4, name="c112"),
        "large_crops": MeasureDinoEmbedding(crop_px=224, batch_size=8, name="c224"),
    })
    tree = _built_staged_tree(tmp_path, tiny_inputs, pipeline)
    outcome = run_stage4_over(tree)
    assert len(stage4_cpu.load_calls) == 1
    ((_, queues),) = outcome.physical_batches.items()
    assert queues == {("crops", 112, 112, "crop_cls"): 4, ("crops", 224, 224, "crop_cls"): 8}
    embeddings = run_entry(stores_of(tree)[0])["embeddings"]
    assert embeddings["dinov2-vitb14/rgb/objects-c112"]["compute"]["physical_batch"] == 4
    assert embeddings["dinov2-vitb14/rgb/objects-c224"]["compute"]["physical_batch"] == 8


# ---------------------------------------------------------------------------
# Fences and orphans
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fence", ["active_check", "commit_guard"])
def test_a_fenced_epoch_commits_nothing(staged_tree, stage4_cpu, fence):
    kwargs = fence_after_first_signal(staged_tree, fence)          # either seam raises SlurmGenerationInactiveError
    with pytest.raises(SlurmGenerationInactiveError):
        run_stage4_over(staged_tree, **kwargs)
    assert any(stage4_signal_dir(staged_tree).rglob("*.parquet"))   # it got as far as one signal
    assert _nothing_committed(staged_tree)
    assert not any(valid_record(staged_tree, item) for item in staged_tree.items)


def test_orphan_signals_of_a_done_image_are_swept(staged_tree, stage4_cpu):
    run_stage4_over(staged_tree)
    done, other = staged_tree.items
    plant_stage4_signals(staged_tree.out, done)                     # a crash between record and cleanup
    temp = signal_files(staged_tree.out, done)[0].with_name(f".{done.stem}.parquet.abc123.tmp")
    temp.write_bytes(b"partial")                                    # a crash mid-write
    assert sweep_orphan_stage4_signals(staged_tree.out, *inventory_and_work_ids(staged_tree)) == 1
    assert not signal_files(staged_tree.out, done) and not temp.exists()


def test_the_sweep_keeps_a_pending_images_signals(staged_tree, stage4_cpu):
    """The control: an image without a record keeps its signals, its temp files too."""
    pending = staged_tree.items[0]
    plant_stage4_signals(staged_tree.out, pending)
    assert sweep_orphan_stage4_signals(staged_tree.out, *inventory_and_work_ids(staged_tree)) == 0
    assert signal_files(staged_tree.out, pending)
