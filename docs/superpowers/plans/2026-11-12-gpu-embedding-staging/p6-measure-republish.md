# P6 · `--mode measure` through a republish Stage 4

**Spec:** U4, U8, U10, §4.1 (`--mode measure` row), §4.6 "Measure mode", §5
"Measure mode over a staged run"

---

**Goal:** `--mode measure --embed-stage gpu` re-embeds an already-certified run
on GPUs. The measure core rewrites the run folder's measurement table, **carries
the image's existing embeddings forward** (U8), and defers both `embed` and the
record republish. A separate Stage-4 pass, flagged *republish* in the manifest,
replaces the embeddings and then republishes each image's record. A Stage-4
failure leaves the previous embeddings and a valid record, and is journalled as
`MEASURE_FAILURE_STAGE`, so the run exits `3`.

**Depends on:** P5.

### Task 1: The measure core carries and defers; local republish Stage 4; manifest v5

**Files:**
- Modify: `src/phenotypic/sdk_/_measurement_tables.py:1460-1645`.
  `replace_image_tables` gains
  `carry_embeddings: Sequence[tuple[str, str]] = ()`, pairs of
  `(source_run_id, key)`. With `embeddings=None`, each named entry and its
  files are carried into the chosen run (hard links,
  `_clone_file_without_pixel_rewrite`, `:1265`) instead of dropped, under the
  same key, and the entry keeps its fields as they were. Its `stale` flag stays
  valid, because a measure write writes no objmap and no array (`:1498-1501`).
  The plan refuses a pair whose run or key the store does not declare, before
  any part exists.
- **What the measure core carries** (U8, the user's ruling on a measure pipeline
  that differs from the forward one, re-review item 3). For each embedder of the
  measure pipeline, with `key = payload_key()` and `params = _params_sha256()`:
  1. the measure pipeline's own run folder, when the store has one and its
     entry under `key` has `producer.params_sha256 == params`;
  2. otherwise **the run the record last certified**
     (`certified_table_run`, `_cli_recompile_recovery.py:339`), when its entry
     under `key` has the same `params_sha256`. That entry is as valid as one of
     its own, because measure writes no objmap;
  3. otherwise nothing: there is no comparable embedding to keep. If Stage 4
     then fails for that image, the image is journalled as
     `MEASURE_FAILURE_STAGE` and the run exits `3`, as for any other failure;
     its new run folder simply has no table under that key.
  The core builds the pairs with
  `measure_embedding_carry(store, record_path, embedders) -> list[tuple[str, str]]`
  (new, in `_cli_process_single.py`).
- Modify: `src/phenotypic/_cli/_cli_output_manager.py:2084`.
  `replace_image_store_measurements` passes `carry_embeddings` through.
- Modify: `src/phenotypic/_cli/_cli_process_single.py:472-655`.
  `process_single_store_measure_core` gains `defer_embeddings: bool = False`.
  When set: `pipeline.embed` (`:522`) is skipped, the replace (`:577-585`) gets
  `embeddings=None, carry_embeddings=measure_embedding_carry(...)` (U8; plan
  review C3), and the
  `_republish_table_marker` block (`:629-655`) is skipped.
- Modify: `src/phenotypic/_cli/_cli_execution_strategies.py:787-870`.
  `_process_single_local_measure` passes `defer_embeddings`; after the measure
  loop, `LocalParallelStrategy.execute` calls
  `run_local_stage4(..., republish=True)` over the stores it measured, and its
  Stage-4 failures join `results.total_failed`, which becomes the CLI's
  `measure_failed` (`phenotypicCLI.py:4283`).
- Modify: `src/phenotypic/_cli/_cli_stage4_worker.py`. `run_stage4(..., republish=True)`:
  - the run is `owned_table_run_id(block, pipeline_sha256=<the measure pipeline's sha256>)`
    (keyword-only, `sdk_/_table_runs.py:829-848`; plan review I10), the folder
    the measure core just wrote. The sha is
    `pipeline_source_identity(pipeline_path)["sha256"]`, as the measure core
    computes it (`_cli_process_single.py:550`), so the two agree by construction;
  - **the publish-only shortcut is off**: republish always re-embeds, as an
    inline `--mode measure` always does (`_cli_process_single.py:522`), and
    carried embeddings must not satisfy it;
  - the record is republished through `_republish_table_marker`
    (`_cli_recompile_tables.py:52`), with `measurements=run.measurements_path()`
    and the lifecycle-epoch rule of `_cli_process_single.py:640-655` (the SLURM
    generation on SLURM, the record's own locally);
  - **a per-image Stage-4 failure in republish mode** (a `FailedImage`, a
    non-finite vector, or a payload bound to another image) still republishes
    the record, now certifying the new measurements with whatever was carried
    (nothing, under rule 3), and journals the failure under the measure work id
    with stage `MEASURE_FAILURE_STAGE` (`_cli_failure_tracker.py`), not
    `"stage4"` (U8). Node state (OOM, no CUDA) and a changed root republish
    nothing and journal nothing.
  - `run_local_stage4` gains `republish: bool = False`.
- Modify: `src/phenotypic/_cli/_cli_staged_resume.py`.
  `stage4_status(..., republish=True, pipeline_sha256=<the measure pipeline's>)`
  (a new keyword, `None` outside republish): `pending` = that pipeline's run
  folder declares a measurement table and the record does not certify the
  current root; `done` = it does. A record with no `artifacts` (never
  forward-published) is `not_ready`.
- Modify: `src/phenotypic/_cli/_cli_staged_orchestration.py:54-275`.
  `_MANIFEST_VERSION = 5`; `_READABLE_MANIFEST_VERSIONS = (2, 3, 4, 5)`;
  `write_staged_manifest(path, entries, *, republish: bool = False)` writes
  top-level `"republish"`; add `staged_manifest_republishes(path) -> bool`
  (`False` below version 5). **This flag is republish's one carrier** (plan
  review Cuts): the controller and the shard both read it from the manifest.
- Test: `tests/integration/cli/test_measure_stage4.py` (new) **[torch]**;
  `tests/unit/sdk_/test_replace_image_tables_carry.py` (new, shared env);
  `tests/unit/cli/test_staged_manifest_version.py` (extend, shared env)

**Interfaces:**
- Consumes: P4 T3's `run_stage4` and `run_local_stage4`; P4 T1's resolver
  (measure: only `--gpu-slurm` counts as a GPU profile, and a local measure
  resolves `inline` unless `--embed-stage gpu`).
- Produces: `replace_image_tables(..., carry_embeddings=...)`;
  `process_single_store_measure_core(..., defer_embeddings=...)`;
  `run_stage4(..., republish=True)`; `stage4_status(..., republish=True)`;
  manifest v5 with `staged_manifest_republishes`. Task 2 consumes all of them.

- [ ] **Step 1: Write the failing tests.**

```python
def test_measure_over_a_staged_run_reembeds_and_republishes(tmp_path, tiny_inputs, stage4_cpu):
    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path, fake_gpu=True)
    assert run_cli(tiny_inputs, out, pipeline=pipe, extra=["--embed-stage", "gpu"]).exit_code == 0
    before = {s: record_digest(s) for s in stores_of(out)}
    loads = len(stage4_cpu.load_calls)
    result = run_cli(None, out, mode="measure", pipeline=pipe, extra=["--embed-stage", "gpu"])
    assert result.exit_code == 0
    assert len(stage4_cpu.load_calls) == loads + 1                     # re-embedded: one backbone, one load
    for store in stores_of(out):
        entry = run_entry(store)["embeddings"]["dinov2-vitb14/rgb/objects"]
        assert entry["compute"]["stage"] == "stage4" and entry["stale"] is False
        assert record_digest(store) != before[store]
    state = resolve_run_state(out, depth="deep")
    assert state.diagnostics.failed == 0 and all(v.verdict == "verified" for v in state.images.values())

def test_a_stage4_failure_keeps_the_old_embeddings_and_a_valid_record(tmp_path, tiny_inputs, stage4_cpu, stage4_faults):
    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path, fake_gpu=True, faulty=True)
    run_cli(tiny_inputs, out, pipeline=pipe, extra=["--embed-stage", "gpu"])
    victim = store_of(out, 0)
    old = embedding_bytes(victim, "dinov2-vitb14/rgb/objects")
    plant_stage4_fault(stem_of(victim))                                         # on disk, not a monkeypatch
    result = run_cli(None, out, mode="measure", pipeline=pipe, extra=["--embed-stage", "gpu"])
    assert result.exit_code == 3                                                # U8
    assert embedding_bytes(victim, "dinov2-vitb14/rgb/objects") == old         # carried forward
    assert record_certifies_root(out, victim)                                   # a valid record
    assert [r.failed_stage for r in read_terminal_failures(out)] == [MEASURE_FAILURE_STAGE]

def test_a_differing_measure_pipeline_carries_matching_entries(tmp_path, tiny_inputs, stage4_cpu, stage4_faults):
    out = tmp_path / "out"
    forward = with_stage4_dino_objects(tmp_path, fake_gpu=True, faulty=True, grid=True)
    run_cli(tiny_inputs, out, pipeline=forward, extra=["--embed-stage", "gpu"])
    victim = store_of(out, 0)
    old = embedding_bytes(victim, "dinov2-vitb14/rgb/objects")
    measure = with_measure_pipeline(tmp_path, keep="objects", add_grid_named="wells")   # another sha: a new run folder
    plant_stage4_fault(stem_of(victim))
    result = run_cli(None, out, mode="measure", pipeline=measure, extra=["--embed-stage", "gpu"])
    assert result.exit_code == 3
    new_run = owned_run(victim, pipeline_sha(measure))
    assert embedding_bytes(victim, "dinov2-vitb14/rgb/objects", run=new_run) == old   # rule 2: carried from the certified run
    assert "dinov2-vitb14/rgb/grid-wells" not in new_run.descriptor.get("embeddings", {})  # rule 3: nothing comparable
    assert record_certifies_root(out, victim)
    assert [r.failed_stage for r in read_terminal_failures(out)] == [MEASURE_FAILURE_STAGE]

def test_republish_keeps_the_records_identity(tmp_path, tiny_inputs, stage4_cpu):
    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path, fake_gpu=True)
    run_cli(tiny_inputs, out, pipeline=pipe, extra=["--embed-stage", "gpu"])
    before = {s: record_identity(out, s) for s in stores_of(out)}
    run_cli(None, out, mode="measure", pipeline=pipe, extra=["--embed-stage", "gpu"])
    assert {s: record_identity(out, s) for s in stores_of(out)} == before
    # work_id, attempt_id, relative_image_path, mode, provenance, source_provenance

def test_a_local_measure_defaults_to_inline(tmp_path, tiny_inputs, stage4_cpu):
    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path, fake_gpu=True)
    run_cli(tiny_inputs, out, pipeline=pipe, extra=["--embed-stage", "gpu"])
    events = tail_events(out)
    run_cli(None, out, mode="measure", pipeline=pipe)                           # no --embed-stage
    assert run_entry(store_of(out))["embeddings"]["dinov2-vitb14/rgb/objects"]["compute"]["stage"] == "inline"
    assert "stage4" not in stages_of(new_events(out, after=events))

def test_measure_stage4_survives_a_real_crash(tmp_path, tiny_inputs, stage4_cpu):
    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path, fake_gpu=True)
    run_cli(tiny_inputs, out, pipeline=pipe, extra=["--embed-stage", "gpu"])
    first = run_cli_subprocess(None, out, pipeline=pipe, extra=["--mode", "measure", "--embed-stage", "gpu"],
                               crash_after="replace_run_embeddings")
    assert first.returncode == 137
    again = run_cli_subprocess(None, out, pipeline=pipe, extra=["--mode", "measure", "--embed-stage", "gpu"])
    assert again.returncode == 0 and all_records_valid(out)
```

  `record_digest`, `record_identity`, `embedding_bytes(store, key, *, run=None)`,
  `with_measure_pipeline(tmp_path, *, keep, add_grid_named)` (a measure pipeline
  whose objects embedder has the forward one's parameters, plus a new named
  grid embedder, so its file and sha differ), `owned_run`, `pipeline_sha`,
  `record_certifies_root`
  and `tail_events` / `new_events` / `stages_of` live in `tests/_tiny_run.py`
  (some from P4). `run_cli_subprocess(None, …)` omits `--input`, as `run_cli`
  does. In `test_replace_image_tables_carry.py`:

```python
def test_carry_keeps_the_runs_embedding_files_and_block(store_with_crop_entry):
    store = store_with_crop_entry
    run = latest_measurement_run(store)
    old = dict(run.descriptor["embeddings"])
    inode = embedding_file(store, run).stat().st_ino
    replace_image_tables(store, tables_for_store(store), pipeline_sha256=run.pipeline_sha256,
                         carry_embeddings=[(run.run_id, key) for key in old])
    run = latest_measurement_run(store)
    assert run.descriptor["embeddings"] == old and embedding_file(store, run).stat().st_ino == inode

def test_carry_from_another_run_into_a_new_folder(store_with_crop_entry):
    store = store_with_crop_entry
    source = latest_measurement_run(store)
    (key,) = source.descriptor["embeddings"]
    run = replace_image_tables(store, tables_for_store(store), pipeline_sha256="b" * 64,
                               carry_embeddings=[(source.run_id, key)])
    assert run.run_id != source.run_id
    assert run.descriptor["embeddings"][key] == {**source.descriptor["embeddings"][key],
                                                 "files": relocated(source, run, key)}

def test_carry_refuses_an_undeclared_pair(store_with_crop_entry):
    with pytest.raises(ValueError, match="9-unknown"):
        replace_image_tables(store_with_crop_entry, tables_for_store(store_with_crop_entry),
                             pipeline_sha256=None, carry_embeddings=[("9-unknown", "x/rgb/objects")])

def test_without_carry_the_embeddings_go(store_with_crop_entry):
    store = store_with_crop_entry
    run = latest_measurement_run(store)
    replace_image_tables(store, tables_for_store(store), pipeline_sha256=run.pipeline_sha256)
    assert "embeddings" not in latest_measurement_run(store).descriptor          # #281 behaviour, unchanged
```

  `store_with_crop_entry` and `tables_for_store` exist (`tests/_tiny_run.py:747,850`);
  `embedding_file(store, run)` and `relocated(source, run, key)` (the entry's
  `files` with paths moved to the new run folder, sha256 unchanged) are
  module-local. In `test_staged_manifest_version.py`:

```python
def test_v5_records_republish(tmp_path):
    path = write_staged_manifest(tmp_path / "m.json", [entry()], republish=True)
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 5
    assert staged_manifest_republishes(path) and len(load_staged_manifest(path)) == 1

def test_a_v4_manifest_reads_as_forward(tmp_path):
    path = write_v4_manifest(tmp_path / "m.json", [entry()])
    assert not staged_manifest_republishes(path)
```

- [ ] **Step 2: Run them and confirm they fail.**

  ```bash
  eval $TPT tests/integration/cli/test_measure_stage4.py
  eval $PT tests/unit/sdk_/test_replace_image_tables_carry.py tests/unit/cli/test_staged_manifest_version.py
  ```

- [ ] **Step 3: Implement.** Republish is a separate invocation from the forward
  run, and within the measure pass nothing is rewritten after the record that
  certifies it: the measure core writes the table once, Stage 4 writes the
  embeddings once, and the record is republished once, last.
- [ ] **Step 4: Run the tests and the measure suites.**

  ```bash
  eval $TPT tests/integration/cli/test_measure_stage4.py tests/integration/cli/test_embedding_deliverables.py
  eval $PT tests/unit/sdk_/test_replace_image_tables_carry.py tests/unit/cli/test_staged_manifest_version.py tests/unit/cli/test_embedded_measurement_replacement.py tests/integration/cli/test_cli_store_output.py
  ```

- [ ] **Step 5: Commit.**

  ```bash
  git add src/phenotypic/sdk_/_measurement_tables.py src/phenotypic/_cli/_cli_output_manager.py src/phenotypic/_cli/_cli_process_single.py src/phenotypic/_cli/_cli_execution_strategies.py src/phenotypic/_cli/_cli_stage4_worker.py src/phenotypic/_cli/_cli_staged_resume.py src/phenotypic/_cli/_cli_staged_orchestration.py tests/_tiny_run.py tests/conftest.py tests/integration/cli/test_measure_stage4.py tests/unit/sdk_/test_replace_image_tables_carry.py tests/unit/cli/test_staged_manifest_version.py
  git commit -m "feat(cli): --mode measure carries embeddings and re-embeds through a republish Stage 4 (#282 P6)"
  ```

### Task 2: SLURM measure mode on the controller, with a measure-mode finalizer (U10)

**Files:**
- Modify: `src/phenotypic/_cli/_cli_execution_strategies.py`.
  `uses_staged_controller` (P5 T2) is true for `measure_only` with
  `embed_stage == "gpu"`.
- Modify: `src/phenotypic/_cli/_cli_staged_slurm.py`.
  `StagedSlurmStrategy.execute` builds a measure manifest (one entry per store,
  `work_id` the measure work id `work_id_for_image` derives for it),
  `write_staged_manifest(..., republish=True)`, `phases=["stage3", "stage4"]`,
  `stage3_role="measure"`, `pipeline_sha256` the measure pipeline's, and
  `_write_staged_job_metadata` (`:485-562`) records
  `JobMetadataKey.RUN_MODE: "measure"` and the measure call's initiation keys
  the SLURM measure worker reads today (`sdk_/_io_constants.py:3030-3038`).
- Modify: `src/phenotypic/_cli/_cli_staged_slurm_worker.py`: add
  `run_measure_step`; `--stage3-role measure` dispatches to it.
- Modify: `src/phenotypic/_cli/_cli_staged_controller.py`.
  `_classify_stage4` calls `stage4_status(..., republish=staged_manifest_republishes(config["manifest_path"]), pipeline_sha256=config["pipeline_sha256"])`.
- Modify: `src/phenotypic/_cli/_cli_checkpoint_handler.py:374-406`, **the
  measure-mode finalizer** (plan review I9). When `RUN_MODE == "measure"`, the
  epoch branch does **not** ask `_all_accepted_images_succeeded` or call
  `publish_run_completion_evidence` (there is no forward run proof to write
  over a measure resubmission); it closes the fence with
  `closed_by=FENCE_CLOSED_BY_FINALIZER` and marks the orchestration complete.
  `finalize_run` is passed `execution_epoch=epoch`, so its existing journal
  count (`_cli_finalize_run.py:987-998`) gives `measure_failed`.
- Test: `tests/unit/cli/test_staged_measure_stage4.py` (new, shared env);
  `tests/unit/cli/test_stage4_slurm_worker.py` (extend) **[torch]**;
  `tests/unit/cli/test_checkpoint_measure_finalizer.py` (new, shared env)

**Interfaces:**
- Produces:
  - `run_measure_step(pipeline_path: Path, output_dir: Path, image_type: ImageTypeName, manifest: Manifest, index: int, *, epoch: str, durable_writes: bool | None = None) -> None`:
    `process_single_store_measure_core(..., defer_embeddings=True)` inside
    `measure_scientific_boundary()` (`_cli_process_single.py:440`); a failure is
    journalled under the measure work id with stage `MEASURE_FAILURE_STAGE`, as
    the SLURM measure worker does today.
  - The measure-mode finalizer branch above.

- [ ] **Step 1: Write the failing tests.**

```python
def test_a_slurm_measure_with_gpu_slurm_uses_the_controller(tmp_path, finished_tiny_run, submissions, monkeypatch):
    monkeypatch.setattr(embed_stage_module, "_hub_commit", lambda hf_id: "c" * 40)
    result = run_cli(None, finished_tiny_run, mode="measure", slurm=True,
                     pipeline=with_stage4_dino_objects(tmp_path),
                     extra=["--gpu-slurm", "slurm_partition=exfab", "--gpu-slurm", "slurm_account=exfab"])
    assert result.exit_code == 4
    config = json.loads(controller_config_path(finished_tiny_run).read_text(encoding="utf-8"))
    assert config["phases"] == ["stage3", "stage4"] and config["stage3_role"] == "measure"
    assert "republish" not in config and staged_manifest_republishes(Path(config["manifest_path"]))
    assert read_job_metadata(finished_tiny_run)["run_mode"] == "measure"

def test_a_slurm_measure_without_gpu_slurm_stays_inline(tmp_path, finished_tiny_run, submissions):
    result = run_cli(None, finished_tiny_run, mode="measure", slurm=True,
                     pipeline=with_stage4_dino_objects(tmp_path))
    assert result.exit_code == 4 and not controller_config_path(finished_tiny_run).exists()

def test_classify_stage4_in_republish_mode(controller_v2_measure):
    retryable, _ = _classify_stage4(controller_v2_measure.config, controller_v2_measure.entries)
    assert len(retryable) == controller_v2_measure.measured_without_republish
```

  In `test_checkpoint_measure_finalizer.py`:

```python
def test_the_measure_finalizer_publishes_no_run_proof_and_counts_measure_failures(measure_epoch_tree, monkeypatch):
    proof = spy(monkeypatch, "publish_run_completion_evidence")
    journal_measure_failure(measure_epoch_tree, image=0)
    _run_finalize(measure_epoch_tree.out, progress_dir(measure_epoch_tree.out), epoch=measure_epoch_tree.epoch)
    assert proof.calls == []
    assert metrics(measure_epoch_tree.out)["failed"] == 1                       # measure_failed from the journal
    assert fence_of(measure_epoch_tree.out)["closed_by"] == FENCE_CLOSED_BY_FINALIZER
```

  In `test_stage4_slurm_worker.py` **[torch]**:

```python
def test_the_measure_role_carries_embeddings_and_defers_the_republish(measured_slurm_tree, stage4_cpu):
    store = measured_slurm_tree.store(0)
    old = embedding_bytes(store, "dinov2-vitb14/rgb/objects")
    run_measure_step(**measured_slurm_tree.measure_kwargs(index=0))
    assert embedding_bytes(store, "dinov2-vitb14/rgb/objects") == old         # U8: carried
    assert not record_certifies_root(measured_slurm_tree.out, store)          # republish deferred
    run_stage4_shard(**measured_slurm_tree.shard_kwargs(shard_index=0))
    assert record_certifies_root(measured_slurm_tree.out, store)
    assert first_entry(measured_slurm_tree.out)["compute"]["stage"] == "stage4"
```

  Create (helpers): fixtures `controller_v2_measure` (in
  `tests/unit/cli/conftest.py`), `measured_slurm_tree` and `measure_epoch_tree`
  (in `tests/_tiny_run.py`), plus `controller_config_path(out)`, `metrics(out)`,
  `fence_of(out)`, `journal_measure_failure`, `spy`.
- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the P6 phase gate** as a Slurm job, both lanes, over
  importers of `_measurement_tables`, `_cli_output_manager`,
  `_cli_process_single`, `_cli_execution_strategies`, `_cli_stage4_worker`,
  `_cli_staged_slurm`, `_cli_staged_slurm_worker`, `_cli_staged_controller`,
  `_cli_staged_orchestration`, `_cli_checkpoint_handler`, plus
  `tests/integration/cli` and the GUI trees once. Expected reds: the 13
  expected-red GUI tests only.
- [ ] **Step 5: Optional live check, with the user's go-ahead** (stop point 2):
  a `--mode measure --embed-stage gpu` resubmission over P5 T4's smoke output,
  through `phenotypic-wait`. Expected: exit `0`, every record verified, every
  entry `compute.stage == "stage4"`. Append the result to `live-smoke.md`.
- [ ] **Step 6: Commit.**

  ```bash
  git add src/phenotypic/_cli/_cli_execution_strategies.py src/phenotypic/_cli/_cli_staged_slurm.py src/phenotypic/_cli/_cli_staged_slurm_worker.py src/phenotypic/_cli/_cli_staged_controller.py src/phenotypic/_cli/_cli_checkpoint_handler.py tests/unit/cli/conftest.py tests/_tiny_run.py tests/conftest.py tests/unit/cli/test_staged_measure_stage4.py tests/unit/cli/test_stage4_slurm_worker.py tests/unit/cli/test_checkpoint_measure_finalizer.py
  git commit -m "feat(slurm): SLURM measure mode on the controller, with a measure-mode finalizer (#282 P6)"
  ```

**Deep review P6** (implementation-test-reviewer, Opus):
`reports/2026-11-12-gpu-embedding-staging/review-p6.md`. It checks that a
republish never changes a record's identity; that a Stage-4 failure keeps the
old embeddings and a record that verifies (U8); and that `--mode measure`
without `--embed-stage gpu` writes what #281 wrote, apart from the entry fields
P2 adds (`compute`, `compute_dtype`; plan review, Minor).
