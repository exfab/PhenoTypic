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
    plant_stage4_fault(stem_of(victim))                                         # on disk, not a monkeypatch
    result = _measure(out, pipe)
    assert result.exit_code == 3, result.output                                 # U8
    assert embedding_bytes(victim, DINO_OBJECTS_KEY) == old                     # carried forward
    assert record_certifies_root(out, victim)                                   # a valid record
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
    assert certified_run_id(out, victim) == new_run.run_id
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
