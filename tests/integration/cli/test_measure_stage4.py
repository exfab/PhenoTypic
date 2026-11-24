"""``--mode measure --embed-stage gpu`` through a republish Stage 4, locally (#282 P6 T1). [torch]

A forward run over :func:`tiny_inputs` (two plates) embeds in Stage 4; a
measure over it then re-measures every store, **carries** its comparable
embeddings forward (U8), defers ``embed`` and the record's republish, and a
republish-mode Stage 4 re-embeds and republishes each record. A Stage-4
failure keeps the old embeddings and a record that verifies, is journalled
as ``MEASURE_FAILURE_STAGE``, and the run exits ``3``.

Everything runs on the CPU (``stage4_cpu``); faults are planted on disk
(``plant_stage4_fault``); the crash test is a real subprocess that calls
``os._exit(137)``. Run in the foundation venv (``TPT``); in the shared env
the module skips.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from phenotypic._cli._cli_failure_tracker import MEASURE_FAILURE_STAGE  # noqa: E402
from phenotypic.sdk_ import resolve_run_state  # noqa: E402
from tests._tiny_run import (  # noqa: E402
    DINO_OBJECTS_KEY,
    all_records_valid,
    certified_run_id,
    embedding_bytes,
    measure_terminal_failures,
    new_events,
    owned_run,
    pipeline_sha,
    plant_stage4_fault,
    record_certifies_root,
    record_digest,
    record_identity,
    run_cli,
    run_cli_subprocess,
    run_entry,
    stages_of,
    stem_of,
    store_of,
    stores_of,
    tail_events,
    with_measure_pipeline,
    with_stage4_dino_objects,
)

GPU = ["--embed-stage", "gpu"]
NAMED_GRID_KEY = "dinov2-vitb14/rgb/grid-wells"


def _forward(tmp_path, tiny_inputs, **pipeline):
    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path, fake_gpu=True, **pipeline)
    result = run_cli(tiny_inputs, out, pipeline=pipe, extra=GPU)
    assert result.exit_code == 0, result.output
    return out, pipe


def _measure(out, pipe, *, extra=GPU):
    return run_cli(None, out, mode="measure", pipeline=pipe, extra=extra)


def test_measure_over_a_staged_run_reembeds_and_republishes(tmp_path, tiny_inputs, stage4_cpu):
    out, pipe = _forward(tmp_path, tiny_inputs)
    before = {s: record_digest(s) for s in stores_of(out)}
    loads = len(stage4_cpu.load_calls)
    result = _measure(out, pipe)
    assert result.exit_code == 0, result.output
    assert len(stage4_cpu.load_calls) == loads + 1                     # re-embedded: one backbone, one load
    for store in stores_of(out):
        entry = run_entry(store)["embeddings"][DINO_OBJECTS_KEY]
        assert entry["compute"]["stage"] == "stage4" and entry["stale"] is False
        assert record_digest(store) != before[store]
        assert certified_run_id(out, store) == owned_run(store, pipeline_sha(pipe)).run_id
    state = resolve_run_state(out, depth="deep")
    assert state.diagnostics.failed == 0 and all(v.verdict == "verified" for v in state.images.values())


def test_a_same_pipeline_measure_reembeds_every_store(tmp_path, tiny_inputs, stage4_cpu):
    """C17 finding, the local half: the measure core leaves a byte-identical root, re-embedded anyway.

    The forward run, then an inline measure, embed inline; the inline
    measure's table rewrite leaves each root as a rewrite serialises it. A
    ``--embed-stage gpu`` measure with the same pipeline file then carries
    those entries (compute.stage "inline"), and its core leaves every root
    byte-identical, so each record still verifies. The republish Stage 4
    must still re-embed every store: a backbone loads, and every entry's
    compute block is this run's (stage4), in a rewritten file.
    """
    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path, fake_gpu=True)
    forward = run_cli(tiny_inputs, out, pipeline=pipe)                  # inline: auto, locally
    assert forward.exit_code == 0, forward.output
    inline = _measure(out, pipe, extra=())
    assert inline.exit_code == 0, inline.output
    stores = stores_of(out)
    assert all(run_entry(s)["embeddings"][DINO_OBJECTS_KEY]["compute"]["stage"] == "inline" for s in stores)
    inodes = {s: owned_run(s, pipeline_sha(pipe)).embedding_path(DINO_OBJECTS_KEY).stat().st_ino for s in stores}
    loads = len(stage4_cpu.load_calls)
    result = _measure(out, pipe)
    assert result.exit_code == 0, result.output
    assert len(stage4_cpu.load_calls) > loads
    for store in stores:
        run = owned_run(store, pipeline_sha(pipe))
        entry = run.descriptor["embeddings"][DINO_OBJECTS_KEY]
        assert entry["compute"]["stage"] == "stage4" and entry["stale"] is False
        assert run.embedding_path(DINO_OBJECTS_KEY).stat().st_ino != inodes[store]
        assert record_certifies_root(out, store)


def test_republish_never_takes_the_publish_only_shortcut(tmp_path, tiny_inputs, stage4_cpu):
    """Carried embeddings would satisfy the forward shortcut; republish re-embeds anyway."""
    out, pipe = _forward(tmp_path, tiny_inputs)
    old = {s: owned_run(s, pipeline_sha(pipe)).embedding_path(DINO_OBJECTS_KEY).stat().st_ino
           for s in stores_of(out)}
    mark = tail_events(out)
    loads = len(stage4_cpu.load_calls)
    assert _measure(out, pipe).exit_code == 0
    assert len(stage4_cpu.load_calls) == loads + 1
    events = new_events(out, after=mark)
    for store in stores_of(out):
        assert "stage4" in stages_of(events, image=store.name)
        # Rewritten by Stage 4's store transaction, not the carried hard link.
        assert owned_run(store, pipeline_sha(pipe)).embedding_path(DINO_OBJECTS_KEY).stat().st_ino != old[store]


def test_a_stage4_failure_keeps_the_old_embeddings_and_a_valid_record(
    tmp_path, tiny_inputs, stage4_cpu, stage4_faults
):
    out, pipe = _forward(tmp_path, tiny_inputs, faulty=True)
    victim = store_of(out, 0)
    old = embedding_bytes(victim, DINO_OBJECTS_KEY)
    old_record = record_digest(victim)
    old_identity = record_identity(out, victim)
    plant_stage4_fault(stem_of(victim))                                         # on disk, not a monkeypatch
    result = _measure(out, pipe)
    assert result.exit_code == 3, result.output                                 # U8
    assert embedding_bytes(victim, DINO_OBJECTS_KEY) == old                     # carried forward
    assert record_certifies_root(out, victim)                                   # a valid record
    assert record_identity(out, victim) == old_identity                         # the failure republish keeps it (M9)
    # U8: the failure still REPUBLISHES the record. After a forward Stage-4
    # run the measure core leaves the root byte-identical (the C17 finding),
    # so a valid record alone cannot tell; the rewritten record can (M10).
    assert record_digest(victim) != old_record
    assert [r.failed_stage for r in measure_terminal_failures(out)] == [MEASURE_FAILURE_STAGE]
    other = store_of(out, 1)
    assert record_certifies_root(out, other)
    assert run_entry(other)["embeddings"][DINO_OBJECTS_KEY]["compute"]["stage"] == "stage4"


def test_a_differing_measure_pipeline_carries_matching_entries(
    tmp_path, tiny_inputs, stage4_cpu, stage4_faults
):
    out, _forward_pipe = _forward(tmp_path, tiny_inputs, faulty=True, grid=True)
    victim = store_of(out, 0)
    old = embedding_bytes(victim, DINO_OBJECTS_KEY)
    measure = with_measure_pipeline(tmp_path, keep="objects", add_grid_named="wells")   # another sha: a new run folder
    plant_stage4_fault(stem_of(victim))
    result = _measure(out, measure)
    assert result.exit_code == 3, result.output
    new_run = owned_run(victim, pipeline_sha(measure))
    assert embedding_bytes(victim, DINO_OBJECTS_KEY, run=new_run) == old        # rule 2: carried from the certified run
    assert NAMED_GRID_KEY not in new_run.descriptor.get("embeddings", {})       # rule 3: nothing comparable
    assert "dinov2-vitb14/rgb/grid" not in new_run.descriptor.get("embeddings", {})  # not in the measure pipeline
    assert record_certifies_root(out, victim)
    assert certified_run_id(out, victim) == new_run.run_id                     # republished over the new folder
    assert [r.failed_stage for r in measure_terminal_failures(out)] == [MEASURE_FAILURE_STAGE]
    # The other plate re-embedded both of the measure pipeline's keys in Stage 4.
    other = owned_run(store_of(out, 1), pipeline_sha(measure))
    assert set(other.descriptor["embeddings"]) == {DINO_OBJECTS_KEY, NAMED_GRID_KEY}


def test_republish_keeps_the_records_identity(tmp_path, tiny_inputs, stage4_cpu):
    out, pipe = _forward(tmp_path, tiny_inputs)
    before = {s: record_identity(out, s) for s in stores_of(out)}
    assert _measure(out, pipe).exit_code == 0
    assert {s: record_identity(out, s) for s in stores_of(out)} == before
    # work_id, attempt_id, relative_image_path, mode, provenance, source_provenance


def test_a_local_measure_defaults_to_inline(tmp_path, tiny_inputs, stage4_cpu):
    out, pipe = _forward(tmp_path, tiny_inputs)
    events = tail_events(out)
    result = _measure(out, pipe, extra=())                                     # no --embed-stage
    assert result.exit_code == 0, result.output
    assert run_entry(store_of(out))["embeddings"][DINO_OBJECTS_KEY]["compute"]["stage"] == "inline"
    fresh = new_events(out, after=events)
    assert all("stage4" not in stages_of(fresh, image=s.name) for s in stores_of(out))


def test_measure_stage4_survives_a_real_crash(tmp_path, tiny_inputs, stage4_cpu):
    out, pipe = _forward(tmp_path, tiny_inputs)
    inodes = {s: owned_run(s, pipeline_sha(pipe)).embedding_path(DINO_OBJECTS_KEY).stat().st_ino
              for s in stores_of(out)}
    first = run_cli_subprocess(None, out, pipeline=pipe, extra=["--mode", "measure", *GPU],
                               crash_after="replace_run_embeddings")
    assert first.returncode == 137, first.stdout + first.stderr
    # Killed right after one image's Stage-4 store transaction, before its record.
    rewritten = [s for s in stores_of(out)
                 if owned_run(s, pipeline_sha(pipe)).embedding_path(DINO_OBJECTS_KEY).stat().st_ino != inodes[s]]
    assert len(rewritten) == 1
    again = run_cli_subprocess(None, out, pipeline=pipe, extra=["--mode", "measure", *GPU])
    assert again.returncode == 0, again.stdout + again.stderr
    assert all_records_valid(out) and all(record_certifies_root(out, s) for s in stores_of(out))


# ---------------------------------------------------------------------------
# P6 deep review: U16 (local), U17, I2, M3, M9
# ---------------------------------------------------------------------------


def _errors(result) -> list[str]:
    return [line for line in result.output.splitlines() if line.startswith("Error:")]


def test_a_node_state_failure_in_republish_journals_and_republishes_nothing(
    tmp_path, tiny_inputs, stage4_cpu, stage4_faults
):
    """M9: a host out-of-memory in the republish Stage 4 is the node's: no journal row, no republish, exit 1."""
    out, pipe = _forward(tmp_path, tiny_inputs, faulty=True)
    victim = store_of(out, 0)
    before = record_digest(victim)
    plant_stage4_fault(stem_of(victim), "oom-once")
    result = _measure(out, pipe, extra=[*GPU, "--njobs", "1"])
    assert result.exit_code == 1, result.output
    errors = _errors(result)
    assert len(errors) == 1 and errors[0].startswith("Error: Stage 4 stopped:"), result.output
    assert measure_terminal_failures(out) == []
    assert record_digest(victim) == before                                     # never republished
    rerun = _measure(out, pipe, extra=[*GPU, "--njobs", "1"])
    assert rerun.exit_code == 0, rerun.output


def test_a_measure_loads_the_runs_recorded_commit_not_the_hubs(tmp_path, tiny_inputs, stage4_cpu, monkeypatch):
    """U17: an unpinned revision in --mode measure reuses the forward run's recorded commit."""
    from phenotypic._cli import _cli_embed_stage as embed_stage_module

    monkeypatch.setattr(embed_stage_module, "_hub_commit", lambda hf_id: "c" * 40)
    out, pipe = _forward(tmp_path, tiny_inputs)
    assert {c["revision"] for c in stage4_cpu.load_calls} == {"c" * 40}
    monkeypatch.setattr(embed_stage_module, "_hub_commit", lambda hf_id: "d" * 40)     # the hub moved
    loads = len(stage4_cpu.load_calls)
    result = _measure(out, pipe)
    assert result.exit_code == 0, result.output
    assert {c["revision"] for c in stage4_cpu.load_calls[loads:]} == {"c" * 40}


def test_an_explicit_revision_moves_a_measure_off_the_recorded_commit(
    tmp_path, tiny_inputs, stage4_cpu, monkeypatch
):
    """U17: an explicit ``revision=`` is the way to move; the recorded commit never overrides it."""
    from phenotypic import ImagePipeline
    from phenotypic._cli import _cli_embed_stage as embed_stage_module
    from phenotypic.detect import OtsuDetector
    from phenotypic.measure import MeasureDinoEmbedding, MeasureSize
    from tests._tiny_run import _write_pipeline_file

    monkeypatch.setattr(embed_stage_module, "_hub_commit", lambda hf_id: "c" * 40)
    out, _pipe = _forward(tmp_path, tiny_inputs)
    pinned = _write_pipeline_file(tmp_path, "pinned_dino", ImagePipeline(
        name="pinned", ops=[OtsuDetector()],
        meas={"size": MeasureSize(), "emb": MeasureDinoEmbedding(revision="e" * 40)},
    ))
    monkeypatch.setattr(embed_stage_module, "_hub_commit", lambda hf_id: pytest.fail("asked the hub"))
    loads = len(stage4_cpu.load_calls)
    result = _measure(out, pinned)
    assert result.exit_code == 0, result.output
    assert {c["revision"] for c in stage4_cpu.load_calls[loads:]} == {"e" * 40}


def test_a_missing_embedding_folder_is_re_embedded(tmp_path, tiny_inputs, stage4_cpu):
    """I2, the E5 remedy: a deleted embedding folder carries nothing, and Stage 4 re-embeds it; exit 0."""
    import shutil

    out, pipe = _forward(tmp_path, tiny_inputs)
    victim = store_of(out, 0)
    run = owned_run(victim, pipeline_sha(pipe))
    shutil.rmtree(run.embedding_path(DINO_OBJECTS_KEY).parent)
    result = _measure(out, pipe)
    assert result.exit_code == 0, result.output
    entry = owned_run(victim, pipeline_sha(pipe)).descriptor["embeddings"][DINO_OBJECTS_KEY]
    assert entry["compute"]["stage"] == "stage4" and entry["stale"] is False
    assert record_certifies_root(out, victim) and measure_terminal_failures(out) == []


def _core_once(monkeypatch, victim_name: str, behaviour: str) -> None:
    """Make the measure core skip (``"skip"``) or fail (``"fail"``) for one store, in this process."""
    from phenotypic._cli import _cli_process_single

    real = _cli_process_single.process_single_store_measure_core

    def _core(*args, **kwargs):
        if Path(kwargs["store_path"]).name == victim_name:
            if behaviour == "skip":
                return True                      # "succeeded" without writing a run folder
            raise ValueError("planted measure-core failure")
        return real(*args, **kwargs)

    monkeypatch.setattr(_cli_process_single, "process_single_store_measure_core", _core)


def test_a_measured_image_stage4_cannot_take_ends_the_local_pass_unfinished(
    tmp_path, tiny_inputs, stage4_cpu, stage4_faults, monkeypatch
):
    """U16, local: an image measured but left with no run folder for Stage 4 is exit 1, named."""
    out, _pipe = _forward(tmp_path, tiny_inputs, faulty=True, grid=True)
    measure = with_measure_pipeline(tmp_path, keep="objects", add_grid_named="wells")
    victim = store_of(out, 0)
    _core_once(monkeypatch, victim.name, "skip")
    result = _measure(out, measure, extra=[*GPU, "--njobs", "1"])
    assert result.exit_code == 1, result.output
    errors = _errors(result)
    assert len(errors) == 1 and "1 measured image(s) were neither re-embedded" in errors[0], result.output
    assert victim.name in errors[0]
    other = store_of(out, 1)
    assert certified_run_id(out, other) == owned_run(other, pipeline_sha(measure)).run_id


def test_a_core_failure_and_a_stage4_failure_count_two(tmp_path, tiny_inputs, stage4_cpu, stage4_faults, monkeypatch):
    """M9 and M3: one invocation, two failed images, two journal rows (another pipeline), exit 3."""
    from tests._tiny_run import metrics

    out, _pipe = _forward(tmp_path, tiny_inputs, faulty=True, grid=True)
    measure = with_measure_pipeline(tmp_path, keep="objects", add_grid_named="wells")
    core_victim, stage4_victim = store_of(out, 0), store_of(out, 1)
    _core_once(monkeypatch, core_victim.name, "fail")
    plant_stage4_fault(stem_of(stage4_victim))
    result = _measure(out, measure, extra=[*GPU, "--njobs", "1"])
    assert result.exit_code == 3, result.output
    assert metrics(out)["failed"] == 2
    # M3: the measure-step failure is journalled too, under a pipeline unlike the forward one.
    assert sorted(r.failed_stage for r in measure_terminal_failures(out)) == [MEASURE_FAILURE_STAGE] * 2
