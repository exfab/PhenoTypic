# P6 · `--mode measure` through a republish Stage 4

**Spec:** U4, §4.1 (`--mode measure` row), §4.6 "Measure mode (U4)", §5
"Measure mode over a staged run"

---

**Goal:** `--mode measure --embed-stage gpu` re-embeds an already-certified run
on GPUs. The measure core rewrites the run folder's measurement table and
defers both `embed` and the record republish; a separate Stage-4 pass, flagged
*republish*, writes the embeddings and then republishes each image's record.

**Depends on:** P5.

### Task 1: The measure core defers; local republish Stage 4; manifest v5

**Files:**
- Modify: `src/phenotypic/_cli/_cli_process_single.py:472-655`.
  `process_single_store_measure_core` gains `defer_embeddings: bool = False`.
  When set: `pipeline.embed` (`:522`) is skipped, so
  `replace_image_store_measurements` (`:577-585`) gets `embeddings=None` and the
  run folder's old embeddings go with this write, as any same-pipeline rewrite
  drops what it is not given (`sdk_/_measurement_tables.py:1488-1495`); and
  the `_republish_table_marker` block (`:629-655`) is skipped.
- Modify: `src/phenotypic/_cli/_cli_execution_strategies.py:787-870`.
  `_process_single_local_measure` passes `defer_embeddings`; after the measure
  loop, `LocalParallelStrategy.execute` calls
  `run_local_stage4(..., republish=True)` over the stores it measured.
- Modify: `src/phenotypic/_cli/_cli_stage4_worker.py`.
  `run_stage4(..., republish=True)` publishes through `_republish_table_marker`
  (`_cli_recompile_tables.py:52`) with
  `measurements=run.measurements_path()` and the lifecycle-epoch rule of
  `_cli_process_single.py:640-655` (the SLURM generation on SLURM, the record's
  own locally). The run id is `owned_table_run_id(block, <the measure pipeline's sha256>)`,
  the folder the measure core just wrote. `run_local_stage4` gains
  `republish: bool = False`.
- Modify: `src/phenotypic/_cli/_cli_staged_resume.py`.
  `stage4_status(..., republish=True)`: `pending` = the measure pipeline's run
  folder declares a measurement table and the record does not certify the
  current root; `done` = the record certifies it. A record with no
  `artifacts` (never forward-published) is `not_ready`.
- Modify: `src/phenotypic/_cli/_cli_staged_orchestration.py:54-275`.
  `_MANIFEST_VERSION = 5`; `_READABLE_MANIFEST_VERSIONS = (2, 3, 4, 5)`;
  `write_staged_manifest(path, entries, *, republish: bool = False)` writes
  top-level `"republish"`; add `staged_manifest_republishes(path) -> bool`
  (`False` for versions below 5).
- Test: `tests/integration/cli/test_measure_stage4.py` (new) **[torch]**;
  `tests/unit/cli/test_staged_manifest_version.py` (extend, shared env)

**Interfaces:**
- Consumes: P4 T3's `run_stage4` and `run_local_stage4`; P4 T1's resolver
  (measure: only `--gpu-slurm` counts as a GPU profile, and a local measure
  resolves `inline` unless `--embed-stage gpu`).
- Produces: `process_single_store_measure_core(..., defer_embeddings=...)`;
  `run_stage4(..., republish=True)`; `stage4_status(..., republish=True)`;
  manifest v5 with `staged_manifest_republishes`. P6 T2 consumes all four.

- [ ] **Step 1: Write the failing tests.**

```python
def test_measure_over_a_staged_run_reembeds_and_republishes(tmp_path, tiny_inputs, stage4_cpu):
    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path, fake_gpu=True)
    assert run_cli(tiny_inputs, out, pipeline=pipe, extra=["--embed-stage", "gpu"]).exit_code == 0
    before = {s: record_digest(s) for s in stores_of(out)}
    loads = len(stage4_cpu.load_calls)
    result = run_cli(None, out, mode="measure", pipeline=pipe, extra=["--embed-stage", "gpu"])
    assert result.exit_code == 0
    assert len(stage4_cpu.load_calls) == loads + 1                     # one backbone, one load
    for store in stores_of(out):
        entry = run_entry(store)["embeddings"]["dinov2-vitb14/rgb/objects"]
        assert entry["compute"]["stage"] == "stage4" and entry["stale"] is False
        assert record_digest(store) != before[store]
    assert resolve_run_state(out, depth="deep").diagnostics.failed == 0
    assert all(v.verdict == "verified" for v in resolve_run_state(out, depth="deep").images.values())

def test_republish_keeps_the_records_identity(tmp_path, tiny_inputs, stage4_cpu):
    ...  # work_id, attempt_id, provenance and source_provenance of every record are unchanged

def test_a_local_measure_defaults_to_inline(tmp_path, tiny_inputs, stage4_cpu):
    ...  # no --embed-stage: compute.stage == "inline" after measure, no Stage-4 event in the log

def test_a_measure_image_that_fails_keeps_its_old_record(tmp_path, tiny_inputs, stage4_cpu):
    ...  # break_store_for_measure (tests/_tiny_run.py:248) on one store: exit 3, its record still verifies

def test_measure_stage4_is_interrupt_safe(tmp_path, tiny_inputs, stage4_cpu, monkeypatch):
    ...  # crash after replace_run_embeddings; re-run the same measure command: publish only, exit 0
```

  `record_digest(store) -> str` is module-local: sha256 of the image record
  file. In `test_staged_manifest_version.py`:

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
  eval $PT tests/unit/cli/test_staged_manifest_version.py
  ```

- [ ] **Step 3: Implement.** Republish is a separate invocation from the forward
  run, so "one writer per artifact, per pass" holds (spec §4.6): within the
  measure pass the measure core writes the table once, Stage 4 writes the
  embeddings once, and the record is republished once.
- [ ] **Step 4: Run the tests and the measure suites.**

  ```bash
  eval $TPT tests/integration/cli/test_measure_stage4.py tests/integration/cli/test_embedding_deliverables.py
  eval $PT tests/unit/cli/test_staged_manifest_version.py tests/unit/cli/test_embedded_measurement_replacement.py tests/integration/cli/test_cli_store_output.py
  ```

- [ ] **Step 5: Commit.**

  ```bash
  git add src/phenotypic/_cli/_cli_process_single.py src/phenotypic/_cli/_cli_execution_strategies.py src/phenotypic/_cli/_cli_stage4_worker.py src/phenotypic/_cli/_cli_staged_resume.py src/phenotypic/_cli/_cli_staged_orchestration.py tests/integration/cli/test_measure_stage4.py tests/unit/cli/test_staged_manifest_version.py
  git commit -m "feat(cli): --mode measure re-embeds through a republish Stage 4 (#282 P6)"
  ```

### Task 2: Measure mode on SLURM through the controller **(pending O7)**

**Files:**
- Modify: `src/phenotypic/_cli/_cli_execution_strategies.py`.
  `uses_staged_controller` (P5 T2) is true for `measure_only` with
  `embed_stage == "gpu"`.
- Modify: `src/phenotypic/_cli/_cli_staged_slurm.py`.
  `StagedSlurmStrategy.execute` builds a measure manifest (one entry per store,
  `work_id` the measure work id `work_id_for_image` derives for it),
  `write_staged_manifest(..., republish=True)`, `phases=["stage3", "stage4"]`,
  `stage3_role="measure"`, `pipeline_sha256` the measure pipeline's.
- Modify: `src/phenotypic/_cli/_cli_staged_slurm_worker.py`: add
  `run_measure_step`; `--stage3-role measure` dispatches to it; `--republish`
  reaches `run_stage4_shard`.
- Modify: `src/phenotypic/_cli/_cli_staged_controller.py`.
  `_classify_stage4` calls `stage4_status(..., republish=config["republish"])`.
- Test: `tests/unit/cli/test_staged_measure_stage4.py` (new, shared env);
  `tests/unit/cli/test_stage4_slurm_worker.py` (extend) **[torch]**

**Interfaces:**
- Produces: `run_measure_step(pipeline_path: Path, output_dir: Path, image_type: ImageTypeName, manifest: Manifest, index: int, *, epoch: str, durable_writes: bool | None = None) -> None`:
  `process_single_store_measure_core(..., defer_embeddings=True)` inside
  `measure_scientific_boundary()` (`_cli_process_single.py:440`), a failure
  journalled under the measure work id with stage `MEASURE_FAILURE_STAGE`, as
  the SLURM measure worker does today.

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
    assert config["republish"] is True
    assert staged_manifest_republishes(Path(config["manifest_path"]))

def test_a_slurm_measure_without_gpu_slurm_stays_inline(tmp_path, finished_tiny_run, submissions):
    result = run_cli(None, finished_tiny_run, mode="measure", slurm=True,
                     pipeline=with_stage4_dino_objects(tmp_path))
    assert result.exit_code == 4 and not controller_config_path(finished_tiny_run).exists()

def test_classify_stage4_in_republish_mode(controller_v2_measure):
    retryable, _ = _classify_stage4(controller_v2_measure.config, controller_v2_measure.entries)
    assert len(retryable) == controller_v2_measure.measured_without_republish
```

  In `test_stage4_slurm_worker.py` **[torch]**:

```python
def test_the_measure_role_defers_embed_and_the_republish(tmp_path, measured_slurm_tree, stage4_cpu):
    run_measure_step(**measured_slurm_tree.measure_kwargs(index=0))
    store = measured_slurm_tree.store(0)
    assert "embeddings" not in run_entry(store)                       # dropped, deferred
    assert not record_certifies_root(measured_slurm_tree.out, store)  # republish deferred
    run_stage4_shard(**measured_slurm_tree.shard_kwargs(shard_index=0, republish=True))
    assert record_certifies_root(measured_slurm_tree.out, store)
```

  Create (helpers): fixtures `controller_v2_measure` (in
  `tests/unit/cli/conftest.py`) and `measured_slurm_tree` (in
  `tests/_tiny_run.py`), plus `controller_config_path(out)` and
  `record_certifies_root(out, store)`.
- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the P6 phase gate** as a Slurm job, both lanes, over
  importers of `_cli_process_single`, `_cli_execution_strategies`,
  `_cli_stage4_worker`, `_cli_staged_slurm`, `_cli_staged_slurm_worker`,
  `_cli_staged_controller`, `_cli_staged_orchestration`, plus
  `tests/integration/cli` and the GUI trees once. Expected reds: the 13
  expected-red GUI tests only.
- [ ] **Step 5: Optional live check, with the user's go-ahead** (stop point 2):
  a `--mode measure --embed-stage gpu` resubmission over P5 T4's smoke output,
  through `phenotypic-wait`. Expected: exit `0`, every record verified, every
  entry `compute.stage == "stage4"`. Append the result to `live-smoke.md`.
- [ ] **Step 6: Commit.**

  ```bash
  git add src/phenotypic/_cli/_cli_execution_strategies.py src/phenotypic/_cli/_cli_staged_slurm.py src/phenotypic/_cli/_cli_staged_slurm_worker.py src/phenotypic/_cli/_cli_staged_controller.py tests/unit/cli/conftest.py tests/_tiny_run.py tests/conftest.py tests/unit/cli/test_staged_measure_stage4.py tests/unit/cli/test_stage4_slurm_worker.py
  git commit -m "feat(slurm): SLURM measure mode re-embeds through the controller's republish Stage 4 (#282 P6)"
  ```

**Deep review P6** (implementation-test-reviewer, Opus):
`reports/2026-11-12-gpu-embedding-staging/review-p6.md`. It checks that a
republish never changes a record's identity, that a failed measure image keeps a
record that still verifies, and that `--mode measure` without `--embed-stage gpu`
is byte-identical to #281 (inline).
