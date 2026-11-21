"""Local Stage 4 end to end: resume, retry, crash injection, inline equality (#282 P4 T4). [torch]

Every test drives the CLI over :func:`tiny_inputs` (two plates) with an
``--embed-stage gpu`` pipeline on the CPU (``stage4_cpu``: the parametric stub
backbone, ``PHENOTYPIC_STAGE4_DEVICE=cpu``, CUDA hidden). A continuation is
the same command again; there is no ``--resume`` flag.

The crash tests run the CLI in a real subprocess (``run_cli_subprocess``) that
calls ``os._exit(137)`` right after the named durable write returns: a real
crash between two writes, never an exception the run could catch (plan review
I5). The state after the crash is read from the disk (``crashed_image``), and a
plain re-run must finish from it without re-embedding or duplicating anything.

Per-image Stage-4 faults are planted on disk (``plant_stage4_fault``), never by
monkeypatch. Run in the foundation venv (``TPT``); in the shared env the module
skips.
"""

from __future__ import annotations

import pyarrow.parquet as pq
import pytest

pytest.importorskip("torch")

from phenotypic._cli._cli_staged_orchestration import staged_completion_path  # noqa: E402
from phenotypic.sdk_ import resolve_run_state  # noqa: E402
from tests._tiny_run import (  # noqa: E402
    ONE_ROW_BATCH,
    STAGE4_DURABLE_WRITES,
    clear_stage4_fault,
    count_calls,
    crashed_image,
    deliverable_row_cosines,
    embedding_deliverables,
    invalidate_store,
    manifest_without_compute,
    new_events,
    no_stage4_signals,
    all_records_valid,
    nth_stem,
    other_item,
    plant_signals,
    plant_stage4_fault,
    record_is_valid,
    run_cli,
    run_cli_subprocess,
    run_cli_subprocess_spying,
    run_items,
    signal_files,
    spy_restart_clear,
    stage4_stubbed_out,
    stages_of,
    store_has_embeddings,
    store_root_bytes,
    tail_events,
    with_stage4_dino_objects,
)

GPU = ["--embed-stage", "gpu"]


# ---------------------------------------------------------------------------
# The deliverable: byte-equal to an inline run at the same physical pass shape
# ---------------------------------------------------------------------------

#: The per-row cosine floor between a Stage-4 and an inline deliverable at the
#: default ``batch_size`` (64), where the pass shapes differ: inline runs one
#: unpadded pass per image and key, Stage 4 one pass of both images' rows
#: padded to ``m``. Measured on ``i29`` (Broadwell, torch 2.11, 2026-11-20):
#: the two runs differ in 0 float16 values, min cosine 1 - 2.2e-16. Moving
#: every stored value of these rows one float16 ULP gives at most
#: 1 - cos = 3.9e-7 (computed from the rows), so the floor admits a one-ULP
#: drift in every value, which another CPU family or BLAS kernel can cause.
DEFAULT_BATCH_COSINE_FLOOR = 1 - 1e-6


@pytest.mark.parametrize("fake_gpu", [True, False], ids=["staged", "single-pass"])
def test_local_stage4_matches_inline_deliverables(tmp_path, tiny_inputs, stage4_cpu, fake_gpu):
    """Byte-equal at the same physical pass shape: ``batch_size=1`` on both sides (review I1).

    Inline then runs ``(1, 3, 224, 224)`` passes and Stage 4 runs ``m = 1``,
    unpadded, one row per pass. "One pass each side" is not the same shape:
    Stage 4 pads to ``m`` and batches across images, and CPU GEMM bits depend
    on the row count (spec §5).
    """
    pipe = with_stage4_dino_objects(tmp_path, fake_gpu=fake_gpu, grid=True, batch_size=ONE_ROW_BATCH)
    inline, gpu = tmp_path / "inline", tmp_path / "gpu"
    first = run_cli(tiny_inputs, inline, pipeline=pipe)
    assert first.exit_code == 0, first.output
    second = run_cli(tiny_inputs, gpu, pipeline=pipe, extra=GPU)
    assert second.exit_code == 0, second.output
    relative = embedding_deliverables(inline)
    assert relative == embedding_deliverables(gpu)
    for rel in relative:
        assert pq.read_table(inline / rel).num_rows > 0, rel
        assert (gpu / rel).read_bytes() == (inline / rel).read_bytes(), rel
    assert manifest_without_compute(gpu) == manifest_without_compute(inline)
    assert resolve_run_state(gpu, depth="deep").diagnostics.failed == 0
    # The control: Stage 4 really ran (it loaded the backbone itself), at m = 1.
    assert stage4_cpu.load_calls, "Stage 4 never loaded the backbone"
    assert _physical_batches(gpu) == {1}


def _physical_batches(out) -> set:
    """Every ``compute.physical_batch`` the run's store entries record, over every key."""
    from tests._tiny_run import run_entry

    stores = sorted(out.glob("results/*/zarr/*.ome.zarr"))
    assert stores
    return {
        entry["compute"]["physical_batch"]
        for store in stores
        for entry in run_entry(store)["embeddings"].values()
    }


@pytest.mark.parametrize("fake_gpu", [True, False], ids=["staged", "single-pass"])
def test_local_stage4_agrees_with_inline_at_the_default_batch(tmp_path, tiny_inputs, stage4_cpu, fake_gpu):
    """Always on: at the default ``batch_size`` every row's cosine to inline is above the floor.

    The pass shapes differ here, so bytes may not match; the cosine is the
    guard (review I1), and the byte test above holds only where shapes match.
    """
    pipe = with_stage4_dino_objects(tmp_path, fake_gpu=fake_gpu, grid=True)
    inline, gpu = tmp_path / "inline", tmp_path / "gpu"
    assert run_cli(tiny_inputs, inline, pipeline=pipe).exit_code == 0
    second = run_cli(tiny_inputs, gpu, pipeline=pipe, extra=GPU)
    assert second.exit_code == 0, second.output
    assert stage4_cpu.load_calls, "Stage 4 never loaded the backbone"
    cosines = deliverable_row_cosines(inline, gpu)
    for rel, values in cosines.items():
        assert values and min(values) >= DEFAULT_BATCH_COSINE_FLOOR, (rel, min(values))


# ---------------------------------------------------------------------------
# Crash injection at every durable boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "crash_after, grid, resumed_by",
    [
        ("write_stage4_signal", False, ["replace_run_embeddings", "publish_image_success", "delete_stage4_signals"]),
        # Two keys per image: the crash falls between them, so the re-run
        # embeds only the key without a signal (plan review C4, M6).
        ("write_stage4_signal", True,
         ["write_stage4_signal", "replace_run_embeddings", "publish_image_success", "delete_stage4_signals"]),
        ("replace_run_embeddings", False, ["publish_image_success", "delete_stage4_signals"]),
        ("publish_image_success", False, ["delete_stage4_signals"]),
    ],
    ids=["after-signal", "between-two-keys", "after-commit", "after-record"],
)
def test_a_real_crash_resumes_from_disk(tmp_path, tiny_inputs, stage4_cpu, crash_after, grid, resumed_by):
    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path, fake_gpu=True, grid=grid)
    first = run_cli_subprocess(tiny_inputs, out, pipeline=pipe, extra=GPU, crash_after=crash_after)
    assert first.returncode == 137, first.stdout + first.stderr          # the process really died
    crashed = crashed_image(out, crash_after)                            # read from disk, not the test
    assert not record_is_valid(out, crashed) or crash_after == "publish_image_success"
    if grid:
        assert len(signal_files(out, crashed)) == 1                       # one of its two keys
    rerun = run_cli_subprocess_spying(tiny_inputs, out, pipeline=pipe, extra=GPU, stem=crashed.stem)
    assert rerun.returncode == 0, rerun.stdout + rerun.stderr
    assert rerun.sequence == resumed_by                                  # never re-embedded
    assert no_stage4_signals(out) and all_records_valid(out)
    assert terminal_failures_of(out) == []
    assert staged_completion_path(out).is_file()


def test_a_crash_inside_the_store_transaction_leaves_a_part_and_resumes(tmp_path, tiny_inputs, stage4_cpu):
    """Review M6: a crash after ``replace_run_embeddings`` built its part, before the promote.

    The live store is untouched (no embeddings, its root unchanged), a
    ``.part`` is left beside it, the signals survive, and a plain re-run
    commits them without re-embedding.
    """
    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path, fake_gpu=True)
    first = run_cli_subprocess(tiny_inputs, out, pipeline=pipe, extra=GPU, crash_inside="replace_run_embeddings")
    assert first.returncode == 137, first.stdout + first.stderr
    parts = list(out.glob("results/*/zarr/.*.ome.zarr.*.part"))
    assert len(parts) == 1, sorted(p.name for p in out.glob("results/*/zarr/*"))
    (crashed,) = [item for item in run_items(out) if parts[0].name.startswith(f".{item.stem}.ome.zarr.")]
    assert signal_files(out, crashed)                                   # its vectors survive
    assert not store_has_embeddings(out, crashed) and not record_is_valid(out, crashed)
    rerun = run_cli_subprocess_spying(tiny_inputs, out, pipeline=pipe, extra=GPU, stem=crashed.stem)
    assert rerun.returncode == 0, rerun.stdout + rerun.stderr
    assert rerun.sequence == ["replace_run_embeddings", "publish_image_success", "delete_stage4_signals"]
    assert no_stage4_signals(out) and all_records_valid(out) and terminal_failures_of(out) == []


def terminal_failures_of(out):
    from phenotypic._cli._cli_failure_tracker import read_terminal_failures

    return read_terminal_failures(out)


def test_the_spy_sees_a_full_embed_when_nothing_crashed(tmp_path, tiny_inputs, stage4_cpu):
    """The control: the spy records all four writes when an image is embedded from scratch."""
    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path, fake_gpu=True)
    run = run_cli_subprocess_spying(tiny_inputs, out, pipeline=pipe, extra=GPU, stem=nth_stem(tiny_inputs, 0))
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.sequence == list(STAGE4_DURABLE_WRITES)


# ---------------------------------------------------------------------------
# A finished run, the orphan sweep
# ---------------------------------------------------------------------------


def test_a_finished_run_rerun_embeds_nothing(tmp_path, tiny_inputs, stage4_cpu):
    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path, fake_gpu=True)
    assert run_cli(tiny_inputs, out, pipeline=pipe, extra=GPU).exit_code == 0
    roots, loads = store_root_bytes(out), len(stage4_cpu.load_calls)
    assert roots and loads
    result = run_cli(tiny_inputs, out, pipeline=pipe, extra=GPU)
    assert result.exit_code == 0, result.output
    assert store_root_bytes(out) == roots and len(stage4_cpu.load_calls) == loads


@pytest.mark.parametrize("fake_gpu", [True, False], ids=["staged", "single-pass"])
def test_a_finished_rerun_sweeps_orphans_before_it_exits(tmp_path, tiny_inputs, stage4_cpu, fake_gpu):
    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path, fake_gpu=fake_gpu)
    assert run_cli(tiny_inputs, out, pipeline=pipe, extra=GPU).exit_code == 0
    plant_signals(out, image=0)                                 # a crash between record and cleanup
    assert not no_stage4_signals(out)
    result = run_cli(tiny_inputs, out, pipeline=pipe, extra=GPU)
    assert result.exit_code == 0, result.output
    assert "All images already processed" in result.output and no_stage4_signals(out)


# ---------------------------------------------------------------------------
# The no-detector continuation, a Stage-1 re-run, --restart
# ---------------------------------------------------------------------------


def test_a_no_detector_continuation_skips_the_core(tmp_path, tiny_inputs, stage4_cpu, monkeypatch):
    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path, fake_gpu=False)
    with stage4_stubbed_out(monkeypatch):
        run_cli(tiny_inputs, out, pipeline=pipe, extra=GPU)
    assert not all_records_valid(out)                       # every image waits on Stage 4
    calls = count_calls(monkeypatch, ["process_single_image_core"])
    result = run_cli(tiny_inputs, out, pipeline=pipe, extra=GPU)
    assert result.exit_code == 0, result.output
    assert calls == {"process_single_image_core": 0} and all_records_valid(out)
    assert all(store_has_embeddings(out, item) for item in run_items(out))


def test_a_stage1_rerun_deletes_the_images_signals(stage4_pending_with_signals, monkeypatch):
    tree = stage4_pending_with_signals
    out, item = tree
    invalidate_store(out, item)                             # its store is no longer this work id's
    calls = count_calls(monkeypatch, ["process_single_image_core"])
    with stage4_stubbed_out(monkeypatch):                   # look at the disk before Stage 4 runs
        result = tree.rerun()
    assert "Traceback" not in result.output, result.output
    assert calls == {"process_single_image_core": 1}        # only the invalidated image re-ran
    assert not any(signal_files(out, item))
    assert any(signal_files(out, other_item(tree)))         # only that image's


def test_restart_clears_stage4_signals(stage4_pending_with_signals, monkeypatch):
    tree = stage4_pending_with_signals
    assert any(signal_files(tree.out, item) for item in tree.items)
    with stage4_stubbed_out(monkeypatch), spy_restart_clear(monkeypatch) as cleared:
        tree.rerun("--restart")
    assert cleared.saw_no_stage4_payload is True            # gone right after the --restart clear


# ---------------------------------------------------------------------------
# Failures: exit 3, skipped on a plain re-run, retried with --retry-failures
# ---------------------------------------------------------------------------


def test_a_stage4_failure_exits_3_and_is_not_staged_complete(tmp_path, tiny_inputs, stage4_cpu, stage4_faults):
    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path, fake_gpu=True, faulty=True)
    plant_stage4_fault(nth_stem(tiny_inputs, 0))
    result = run_cli(tiny_inputs, out, pipeline=pipe, extra=GPU)
    assert result.exit_code == 3, result.output                          # plan review C2, not 1
    assert not staged_completion_path(out).is_file()
    bad, good = run_items(out)
    assert not record_is_valid(out, bad) and record_is_valid(out, good)
    assert [r.failed_stage for r in terminal_failures_of(out)] == ["stage4"]


@pytest.mark.parametrize("fake_gpu", [True, False], ids=["staged", "single-pass"])
def test_retry_failures_reruns_only_stage4(tmp_path, tiny_inputs, stage4_cpu, stage4_faults, fake_gpu):
    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path, fake_gpu=fake_gpu, faulty=True)
    bad = nth_stem(tiny_inputs, 0)
    plant_stage4_fault(bad)
    assert run_cli(tiny_inputs, out, pipeline=pipe, extra=GPU).exit_code == 3
    clear_stage4_fault(bad)
    loads = len(stage4_cpu.load_calls)
    skipped = run_cli(tiny_inputs, out, pipeline=pipe, extra=GPU)
    assert skipped.exit_code == 3, skipped.output                       # skipped, not retried
    assert len(stage4_cpu.load_calls) == loads
    mark = tail_events(out)
    retried = run_cli(tiny_inputs, out, pipeline=pipe, extra=[*GPU, "--retry-failures"])
    assert retried.exit_code == 0, retried.output
    assert stages_of(new_events(out, after=mark), image=f"{bad}.tif") == ["stage4"]
    assert all_records_valid(out) and staged_completion_path(out).is_file()


# ---------------------------------------------------------------------------
# One identity per image per invocation (review I4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("continuation", [False, True], ids=["first-run", "continuation"])
@pytest.mark.parametrize("fake_gpu", [True, False], ids=["staged", "single-pass"])
def test_each_input_is_hashed_once_from_the_strategy_through_stage4(
    tmp_path, tiny_inputs, stage4_cpu, monkeypatch, fake_gpu, continuation
):
    """Review I4: the strategy, the split and Stage 4 hash each input once, together.

    ``file_sha256`` of an input is what a work id costs (a whole-plate read on
    GPFS). From the strategy's ``execute`` through Stage 4, each input is
    hashed exactly once: Stage 1's identity (staged) or the strategy's
    (single-pass) is handed to the core and to Stage 4, never recomputed.
    The CLI's own startup hashing, before the strategy, is not counted.
    """
    from collections import Counter
    from pathlib import Path

    from phenotypic._cli import _cli_execution_strategies, _cli_failure_tracker, _cli_staged_strategy

    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path, fake_gpu=fake_gpu)
    if continuation:
        with stage4_stubbed_out(monkeypatch):                   # Stages 1-3 done, Stage 4 pending
            run_cli(tiny_inputs, out, pipeline=pipe, extra=GPU)
        assert not all_records_valid(out)
    inputs = {path.resolve() for path in tiny_inputs.rglob("*.tif")}
    assert len(inputs) == 2
    hashed: Counter = Counter()
    inside = [False]
    real_hash = _cli_failure_tracker.file_sha256

    def _counting_hash(path):
        if inside[0] and Path(path).resolve() in inputs:
            hashed[Path(path).resolve()] += 1
        return real_hash(path)

    def _counted(execute):
        def _execute(self, *args, **kwargs):
            inside[0] = True
            try:
                return execute(self, *args, **kwargs)
            finally:
                inside[0] = False
        return _execute

    monkeypatch.setattr(_cli_failure_tracker, "file_sha256", _counting_hash)
    for cls in (_cli_execution_strategies.LocalParallelStrategy, _cli_staged_strategy.StagedGpuStrategy):
        monkeypatch.setattr(cls, "execute", _counted(cls.execute))
    result = run_cli(tiny_inputs, out, pipeline=pipe, extra=GPU)
    assert result.exit_code == 0, result.output
    assert all_records_valid(out) and all(store_has_embeddings(out, item) for item in run_items(out))
    assert hashed == Counter({path: 1 for path in inputs}), hashed


# ---------------------------------------------------------------------------
# Node state: one line, exit 1, nothing journalled, a plain re-run finishes
# ---------------------------------------------------------------------------


def _fail_once(monkeypatch, owner, name, exc):
    """Make ``owner.name`` raise *exc* on its first call only, in this process."""
    real = getattr(owner, name)
    calls = {"n": 0}

    def _once(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise exc
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, name, _once)


@pytest.mark.parametrize("failure, njobs", [
    ("loader-oom", 1), ("loader-oom", 2), ("loader-death", 2), ("torch-oom", 1), ("encoder-oom", 1),
], ids=["loader-oom-in-process", "loader-oom-spawned", "loader-worker-death", "torch-oom", "encoder-oom"])
def test_node_state_exits_1_in_one_line_and_a_rerun_finishes(
    tmp_path, tiny_inputs, stage4_cpu, stage4_faults, monkeypatch, failure, njobs
):
    """Review M3/M6: running out of memory, or a dead loader worker, is the node's, never an image's.

    - ``loader-oom``: the loader's ``prepare_regions`` raises ``MemoryError``
      (a ``FailedImage(node_state=True)``), in this process and in a spawned
      worker; planted on disk, once.
    - ``loader-worker-death``: the spawned worker exits (``os._exit``), as a
      host OOM kill ends it: "DataLoader worker ... exited unexpectedly".
    - ``torch-oom``: ``vectors_from`` raises torch's ``OutOfMemoryError``, a
      ``RuntimeError``, not a ``MemoryError``.
    - ``encoder-oom``: the encoder raises ``Stage4OutOfMemory``.

    Each is one ``Error:`` line, exit 1, no traceback, no journal row, every
    image still pending; the same command again finishes.
    """
    import torch

    from phenotypic._cli._cli_staged_resume import stage4_status
    from phenotypic.detect.nn._helper._batch_encoder import BatchEncoder, Stage4OutOfMemory
    from phenotypic.measure import MeasureDinoEmbedding

    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path, fake_gpu=True, faulty=True)
    extra = [*GPU, "--njobs", str(njobs)]
    with monkeypatch.context() as patched:
        if failure == "loader-oom":
            plant_stage4_fault(nth_stem(tiny_inputs, 0), "oom-once")
        elif failure == "loader-death":
            plant_stage4_fault(nth_stem(tiny_inputs, 0), "die")
        elif failure == "torch-oom":
            _fail_once(patched, MeasureDinoEmbedding, "vectors_from", torch.OutOfMemoryError("CUDA out of memory"))
        else:
            assert failure == "encoder-oom", failure
            _fail_once(patched, BatchEncoder, "submit", Stage4OutOfMemory("a queue cannot halve again"))
        result = run_cli(tiny_inputs, out, pipeline=pipe, extra=extra)
    assert result.exit_code == 1, result.output
    errors = [line for line in result.output.splitlines() if line.startswith("Error:")]
    assert len(errors) == 1 and errors[0].startswith("Error: Stage 4 stopped:"), result.output
    assert "No image was recorded as failed" in errors[0]
    if failure == "loader-death":
        assert "the Stage-4 loader failed" in errors[0], errors[0]
    assert "Traceback" not in result.output and "Unexpected error" not in result.output, result.output
    assert terminal_failures_of(out) == []
    assert {stage4_status(out, i.dataset, i.stem, i.work_id) for i in run_items(out)} == {"pending"}
    clear_stage4_fault(nth_stem(tiny_inputs, 0))
    rerun = run_cli(tiny_inputs, out, pipeline=pipe, extra=extra)
    assert rerun.exit_code == 0, rerun.output
    assert all_records_valid(out) and terminal_failures_of(out) == []


# ---------------------------------------------------------------------------
# Counts and the completion predicate (review M5)
# ---------------------------------------------------------------------------


def test_the_core_counts_only_images_stage4_will_take(tmp_path, tiny_inputs, stage4_cpu, monkeypatch):
    """Review M5: a skipped image is a core success only when Stage 4 takes it.

    The local strategy is handed a done image (its record valid) and a
    pending one; Stage 4 is stubbed out. Only the pending image counts as
    completed: the done one is not this invocation's work.
    """
    from phenotypic._cli import _cli_execution_strategies
    from phenotypic._cli._cli_types import Dataset
    from tests._tiny_run import Stage4Waiting

    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path, fake_gpu=False)
    with stage4_stubbed_out(monkeypatch):
        run_cli(tiny_inputs, out, pipeline=pipe, extra=GPU)
    tree = Stage4Waiting(tiny_inputs, out, pipe)
    done, waiting = tree.items
    assert run_stage4_over_one(tree, 0).published                        # image 0: done
    assert record_is_valid(out, done) and not record_is_valid(out, waiting)
    seen = {}
    real_execute = _cli_execution_strategies.LocalParallelStrategy.execute

    def _with_both_images(self, datasets, output_dir):
        (dataset,) = datasets
        everything = sorted((tiny_inputs / dataset.name).glob("*.tif"))
        seen["result"] = real_execute(
            self, [Dataset(dataset.name, everything, dataset.input_dir, dataset.output_dir)], output_dir
        )
        return seen["result"]

    monkeypatch.setattr(_cli_execution_strategies.LocalParallelStrategy, "execute", _with_both_images)
    with stage4_stubbed_out(monkeypatch):
        run_cli(tiny_inputs, out, pipeline=pipe, extra=GPU)
    assert seen["result"].total_images == 2
    assert seen["result"].total_completed == 1 and seen["result"].total_failed == 0


def run_stage4_over_one(tree, index):
    """``run_stage4`` over *tree*'s image *index* only."""
    from phenotypic._cli._cli_output_manager import OutputManager
    from phenotypic._cli._cli_stage4_worker import run_stage4
    from tests._tiny_run import stage4_epoch, stage4_image_type

    return run_stage4(
        tree.pipeline, tree.out, [tree.manifest()[index]], image_type=stage4_image_type(tree),
        epoch=stage4_epoch(tree), num_workers=0,
        output_manager=OutputManager.from_config(base_dir=tree.out, ext=".tiff"),
    )


def test_the_completion_predicate_resolves_the_run_once(tmp_path, tiny_inputs, stage4_cpu, stage4_faults, monkeypatch):
    """Review M5: under deferral one deep ``resolve_run_state`` serves both completion questions.

    A Stage-4 failure leaves one image unpublished, so the CLI asks both
    ``_unpublished_images`` and ``_all_current_terminal_failures``; neither
    resolves the run itself.
    """
    import inspect

    import phenotypic.sdk_ as sdk

    real = sdk.resolve_run_state
    callers: list[str] = []

    def _counting(*args, **kwargs):
        if kwargs.get("depth") == "deep":
            callers.append(inspect.stack()[1].function)
        return real(*args, **kwargs)

    monkeypatch.setattr(sdk, "resolve_run_state", _counting)
    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path, fake_gpu=True, faulty=True)
    plant_stage4_fault(nth_stem(tiny_inputs, 0))
    result = run_cli(tiny_inputs, out, pipeline=pipe, extra=GPU)
    assert result.exit_code == 3, result.output                          # both helpers were asked
    assert "_unpublished_images" not in callers and "_all_current_terminal_failures" not in callers, callers
