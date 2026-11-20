# P4 · Local Stage 4: resolver, refusals, deferral, gated recovery, resume

**Spec:** U2, U3, H5, H9, H10, §4.1, §4.2, §4.4 (plan, loop, DataLoader),
§4.6, §4.7 (R1–R3), §4.8, §5 "Local end-to-end with a stub"

---

**Goal:** `--embed-stage gpu` works on one machine. Stage 3 (or the single-pass
core) defers embeddings and the record; a local Stage 4 embeds every pending
image backbone by backbone, commits each store, then publishes its record; and
an interrupted run resumes from what is on disk.

**Depends on:** P2 (and P2 T4's measurement, stop rule 3), P3.

**Test fixtures used throughout** (create in Task 1, in `tests/_tiny_run.py`):
- `with_stage4_dino_objects(directory: Path, *, fake_gpu: bool = False, grid: bool = False) -> Path`:
  like `with_dino_objects` (`:1296`) but `device="auto"`, so R2 does not refuse
  it. With `fake_gpu`, the ops gain `FakeGpuDetector`
  (`tests/_fakes/fake_gpu_detector.py`; import `tests._fakes.register_fake_gpu`
  first and set `PHENOTYPIC_PRELOAD_MODULES`, per the `phenotypic-regression-baseline`
  note on its registration). With `grid`, `meas["grid"]` is a 2 × 2
  `ManualGridFinder` embedder on the same backbone.
- fixture `stage4_cpu(monkeypatch, stub_backbone)`: sets
  `PHENOTYPIC_STAGE4_DEVICE=cpu` and yields `stub_backbone`.

### Task 1: The resolver, R1–R3, and recording

**Files:**
- Create: `src/phenotypic/_cli/_cli_embed_stage.py`
- Modify: `src/phenotypic/_cli/_cli_types.py:271-276`: `ExecutionConfig` gains
  `embed_stage_request: EmbedStageRequest = "auto"` and
  `embed_stage: EmbedStage = "inline"` (the resolved value).
- Modify: `src/phenotypic/phenotypicCLI.py`
  - `--embed-stage` option beside `--gpu-slurm` (`:2366-2373`), `click.Choice(["auto", "gpu", "inline"])`, default `auto`;
  - `phenotypic_cli` (`:2512`) takes it; after the `ExecutionConfig` is built
    (`:3140-3158`), resolve and set `config.embed_stage`;
  - R1 and R2 beside `_refuse_process_meas_it_would_lose` (`:3181`);
  - R3 right after `load_processing_state` (`:3224`), before
    `validate_resume_compatibility`;
  - O5's note line when `gpu` was asked for and there is no embedder.
- Modify: `src/phenotypic/_cli/_cli_state_management.py:362-385`: config gains
  `"embed_stage": config.embed_stage`.
- Modify: `src/phenotypic/sdk_/_io_constants.py:2990`: `JobMetadataKey` gains
  `EMBED_STAGE = "embed_stage"` and `EMBED_REVISIONS = "embed_revisions"`.
- Modify: `_write_staged_job_metadata` (`_cli_staged_slurm.py:485-562`) and the
  `AutonomousSLURMStrategy` metadata writer (`_cli_execution_strategies.py:1131`)
  record `EMBED_STAGE`.
- Test: `tests/unit/cli/test_cli_embed_stage.py` (new, shared env);
  `tests/unit/cli/test_cli_preflight_ordering.py` (extend);
  `tests/integration/cli/test_embed_stage_refusals.py` (new) **[torch]**

**Interfaces:**
- Produces, in `_cli_embed_stage.py`:
  - `EmbedStage = Literal["gpu", "inline"]`, `EmbedStageRequest = Literal["auto", "gpu", "inline"]`
  - `pipeline_embedders(pipeline: ImagePipeline) -> list[tuple[str, MeasureEmbedding]]`:
    root `meas` embedders, in `meas` order (`_image_pipeline_core.py:1413-1417`)
  - `resolve_embed_stage(requested: EmbedStageRequest, *, mode: RunMode, is_slurm: bool, has_gpu_detector: bool, gpu_slurm_given: bool, has_embedders: bool) -> EmbedStage`,
    the README's resolution table. `mode="process"` always gives `"inline"`.
  - `refuse_embed_stage_for_process(requested: EmbedStageRequest, mode: RunMode) -> None` (R1)
  - `refuse_unstageable_embedders(pipeline: ImagePipeline, resolved: EmbedStage) -> None` (R2):
    when `resolved == "gpu"`, an embedder with `device="cpu"`, or a
    `MeasureEmbedding` that is not a `MeasureDinoEmbedding` **(pending O3)**
  - `refuse_embed_stage_switch(recorded: str | None, resolved: EmbedStage) -> None` (R3);
    `None` reads `"inline"`
  - `defers_embeddings(config: ExecutionConfig) -> bool`: `config.embed_stage == "gpu"`
  - Each refusal raises `click.UsageError` with one line. The exact copy:
    - R1: `--embed-stage gpu does not apply to --mode process: process mode embeds inline on the CPU`
    - R2: `--embed-stage resolves to gpu, but meas '<key>' (<Class>) runs on device='cpu'; set device='auto' or 'cuda', or pass --embed-stage inline`
      (O3's variant: `... (<Class>) has no GPU Stage-4 encoder; ...`)
    - R3: `this run was started with --embed-stage <recorded>; continuing it as <resolved> would mix CPU and GPU embeddings. Re-run with --embed-stage <recorded>, or re-embed the finished run with --mode measure --embed-stage gpu`

- [ ] **Step 1: Write the failing tests.**

```python
@pytest.mark.parametrize("requested, mode, slurm, detector, gpu_slurm, embedders, expected", [
    ("auto", "full", True, True, False, True, "gpu"),
    ("auto", "full", True, False, True, True, "gpu"),
    ("auto", "full", True, False, False, True, "inline"),
    ("auto", "full", False, True, True, True, "inline"),
    ("gpu", "full", False, False, False, True, "gpu"),
    ("auto", "measure", True, True, False, True, "inline"),   # only --gpu-slurm counts
    ("auto", "measure", True, False, True, True, "gpu"),
    ("auto", "process", True, True, True, True, "inline"),
    ("gpu", "full", True, True, True, False, "inline"),       # O5
    ("inline", "full", True, True, True, True, "inline"),
])
def test_resolution_table(requested, mode, slurm, detector, gpu_slurm, embedders, expected):
    assert resolve_embed_stage(requested, mode=mode, is_slurm=slurm, has_gpu_detector=detector,
                               gpu_slurm_given=gpu_slurm, has_embedders=embedders) == expected

def test_r1_refuses_gpu_in_process_mode():
    with pytest.raises(click.UsageError, match="does not apply to --mode process"):
        refuse_embed_stage_for_process("gpu", "process")

def test_r2_refuses_a_cpu_pinned_embedder():
    pipe = ImagePipeline(meas={"emb": MeasureDinoEmbedding(device="cpu")})
    with pytest.raises(click.UsageError, match=r"meas 'emb' \(MeasureDinoEmbedding\) runs on device='cpu'"):
        refuse_unstageable_embedders(pipe, "gpu")
    refuse_unstageable_embedders(pipe, "inline")              # inline: accepted

def test_r3_names_the_recorded_flag_and_the_remedy():
    with pytest.raises(click.UsageError, match=r"started with --embed-stage inline.*--mode measure --embed-stage gpu"):
        refuse_embed_stage_switch("inline", "gpu")
    refuse_embed_stage_switch(None, "inline")                 # a pre-#282 state

def test_embed_stage_is_not_in_the_work_id(make_exec_config, tiny_pipeline_file, tmp_path):
    a = make_exec_config(pipeline_json=tiny_pipeline_file, input_path=tmp_path)
    b = dataclasses.replace(a, embed_stage="gpu", embed_stage_request="gpu")
    assert processing_configuration_digest(a) == processing_configuration_digest(b)
```

  In `test_cli_preflight_ordering.py`, add R1, R2 and R3 to the cases that must
  refuse before `--dry-run` and before `--overwrite` deletes anything, in the
  file's own style.

  In `test_embed_stage_refusals.py` **[torch]**:

```python
def test_r3_refuses_a_switched_continuation(tmp_path, tiny_inputs, stage4_cpu):
    out = tmp_path / "out"
    pipe = with_stage4_dino_objects(tmp_path)
    assert run_cli(tiny_inputs, out, pipeline=pipe).exit_code == 0          # inline, recorded
    result = run_cli(tiny_inputs, out, pipeline=pipe, extra=["--embed-stage", "gpu"])
    assert result.exit_code == 2 and "started with --embed-stage inline" in result.output

def test_the_resolved_value_is_recorded(tmp_path, tiny_inputs, stage4_cpu):
    out = tmp_path / "out"
    run_cli(tiny_inputs, out, pipeline=with_stage4_dino_objects(tmp_path), extra=["--embed-stage", "gpu"])
    assert load_processing_state(out).config["embed_stage"] == "gpu"
```

  The second test passes only once Tasks 2–3 exist; mark it with
  `pytest.mark.xfail(strict=True, reason="P4 T3")` here and remove the mark in
  Task 3 Step 4, so it cannot be forgotten.
- [ ] **Step 2: Run them and confirm they fail.**

  ```bash
  eval $PT tests/unit/cli/test_cli_embed_stage.py tests/unit/cli/test_cli_preflight_ordering.py
  eval $TPT tests/integration/cli/test_embed_stage_refusals.py
  ```

- [ ] **Step 3: Implement.** The resolver needs `has_gpu_detector` and
  `has_embedders` before the mutating half: reuse the pipeline load the
  staged-GPU refusal probe already does (`uses_staged_gpu_strategy`,
  `phenotypicCLI.py:3165`), and a load failure resolves `inline` and is left to
  the pipeline validation below it, as that probe does.
- [ ] **Step 4: Run the tests.** Expected: all pass, the `xfail` reads `XFAIL`.
- [ ] **Step 5: Commit.**

  ```bash
  git add src/phenotypic/_cli/_cli_embed_stage.py src/phenotypic/_cli/_cli_types.py src/phenotypic/phenotypicCLI.py src/phenotypic/_cli/_cli_state_management.py src/phenotypic/sdk_/_io_constants.py src/phenotypic/_cli/_cli_staged_slurm.py src/phenotypic/_cli/_cli_execution_strategies.py tests/_tiny_run.py tests/conftest.py tests/unit/cli/test_cli_embed_stage.py tests/unit/cli/test_cli_preflight_ordering.py tests/integration/cli/test_embed_stage_refusals.py
  git commit -m "feat(cli): --embed-stage, its resolution, and refusals R1-R3 (#282 P4)"
  ```

### Task 2: Deferral, gated recovery shortcuts, and the `stage4` tag

**Files:**
- Modify: `src/phenotypic/_cli/_cli_staged_workers.py:575-773`.
  `stage3_merge_measure_core` gains `defer_embeddings: bool = False`; when set,
  `replay_pipeline.embed` (`:658`) is not called and `save_image_store` gets
  `embeddings=None`.
- Modify: `src/phenotypic/_cli/_cli_process_single.py:202-437`.
  `process_single_image_core` gains `defer_embeddings: bool = False`; when set,
  `pipeline.embed` (`:368`) is skipped, and after the final save it writes
  `write_stage3_completion_marker` (`_cli_staged_resume.py:162`), so the record
  holds `stages.stage3` and nothing else.
- Modify: `src/phenotypic/_cli/_cli_staged_strategy.py`
  - `_terminal_output_exists` (`:122-198`): its publish branch (`:136-157`)
    returns `False` without publishing when `defers_embeddings(cfg)`;
  - `_stage3` (`:315-388`): passes `defer_embeddings`, and skips
    `_publish_local_image_success` (`~:345`) when deferring. The Stage-3 marker
    and the Stage-2 token/raw consumption stay.
- Modify: `src/phenotypic/_cli/_cli_staged_slurm_worker.py:432-642`.
  `run_stage3_step`'s publish-only branch (`:479-529`) and its publish
  (`:572-603`) are skipped when the controller config's `phases` contains
  `"stage4"`. Until P5 writes `phases`, the worker reads it from
  `job_metadata.json` `EMBED_STAGE`.
- Modify: `src/phenotypic/_cli/_cli_execution_strategies.py:525-650`.
  `_process_single_local` passes `defer_embeddings`, skips
  `_publish_local_image_success` (`:574`) when deferring, and tags its events
  `stage="stage3"`.
- Modify: `src/phenotypic/_cli/_stages.py`: `StageTag` gains `"stage4"`;
  `STAGE_EMBED: StageTag = "stage4"`; `VALID_STAGE_TAGS` includes it; add
  `run_terminal_stage(state_config: Mapping[str, object] | None) -> StageTag`:
  `"stage4"` when `state_config["embed_stage"] == "gpu"`, else
  `STAGED_TERMINAL_STAGE`.
- Modify: `src/phenotypic/_cli/_cli_update_state.py:266-420`.
  `aggregate_state_from_events` and `..._with_diagnostics` gain
  `terminal_stage: StageTag = STAGED_TERMINAL_STAGE`, used at `:372`. Callers
  pass `run_terminal_stage(state.config)`:
  `_cli_state_management.py:269`, `_dashboard/_manifest_builder.py:337,526`.
- Test: `tests/unit/cli/test_stage4_deferral.py` (new) **[torch]**;
  `tests/unit/cli/test_cli_update_state_stage.py` (extend, shared env)

**Interfaces:**
- Consumes: `defers_embeddings` (Task 1).
- Produces: the `defer_embeddings` keyword on both cores (P5 T2 and P6 call
  them); `STAGE_EMBED`; `run_terminal_stage`; `aggregate_state_from_events(..., terminal_stage=...)`.

- [ ] **Step 1: Write the failing tests.**

```python
def test_a_deferring_stage3_writes_no_record_and_no_embeddings(tmp_path, tiny_inputs, stage4_cpu, monkeypatch):
    monkeypatch.setattr(stage4_worker_module, "run_local_stage4", lambda *a, **k: {})  # stop before Stage 4
    out = tmp_path / "out"
    run_cli(tiny_inputs, out, pipeline=with_stage4_dino_objects(tmp_path, fake_gpu=True),
            extra=["--embed-stage", "gpu"])
    store = one_store(out)
    record = read_image_record(out, *dataset_and_stem(store))
    assert set(record["stages"]) == {"stage3"} and not record.get("artifacts")
    assert "embeddings" not in run_entry(store)
    assert journal_measurement_table(store) is not None

def test_the_single_pass_core_defers_the_same_way(tmp_path, tiny_inputs, stage4_cpu, monkeypatch):
    ...  # the same assertions, with fake_gpu=False (LocalParallelStrategy)

def test_a_stage3_complete_image_is_not_published_by_the_shortcut(tmp_path, tiny_inputs, stage4_cpu, monkeypatch):
    ...  # run once with Stage 4 stubbed out, run again: still no record, the shortcut returned False

def test_process_mode_ignores_embed_stage(tmp_path, tiny_inputs, stage4_cpu):
    out = tmp_path / "out"
    result = run_cli(tiny_inputs, out, mode="process", pipeline=with_stage4_dino_objects(tmp_path),
                     extra=["--layer", "rgb", "--process-format", "zarr"])
    assert result.exit_code == 0
    entry = run_entry(one_store(out))["embeddings"]["dinov2-vitb14/rgb/objects"]
    assert entry["compute"]["stage"] == "inline"
```

  `stage4_worker_module` is `phenotypic._cli._cli_stage4_worker`, created
  empty-but-importable with a `run_local_stage4` stub in this task so the
  monkeypatch target exists; Task 3 fills it. In `test_cli_update_state.py`:

```python
def test_stage3_completion_is_not_completion_when_the_run_embeds_on_gpu(tmp_path):
    log = event_log_with(tmp_path, [("ds", "a", "completed", "stage3")])
    assert "a" not in aggregate_state_from_events(log, terminal_stage="stage4")["ds"].completed
    assert "a" in aggregate_state_from_events(log)["ds"].completed

def test_run_terminal_stage():
    assert run_terminal_stage({"embed_stage": "gpu"}) == "stage4"
    assert run_terminal_stage({}) == run_terminal_stage(None) == "stage3"
```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.** Every gate reads one predicate,
  `defers_embeddings(config)`, or on the SLURM worker the recorded value; no
  site re-derives it from flags.
- [ ] **Step 4: Run the tests, and the staged and single-pass suites they touch.**

  ```bash
  eval $TPT tests/unit/cli/test_stage4_deferral.py tests/unit/cli/test_staged_resume.py tests/unit/cli/test_staged_resume_parity.py tests/unit/cli/test_staged_nested_equivalence.py
  eval $PT tests/unit/cli/test_cli_update_state_stage.py tests/unit/cli/test_staged_routing.py
  ```

  Expected: all pass. Inline runs are unchanged: nothing defers unless
  `embed_stage == "gpu"`.
- [ ] **Step 5: Commit.**

  ```bash
  git add src/phenotypic/_cli/_cli_staged_workers.py src/phenotypic/_cli/_cli_process_single.py src/phenotypic/_cli/_cli_staged_strategy.py src/phenotypic/_cli/_cli_staged_slurm_worker.py src/phenotypic/_cli/_cli_execution_strategies.py src/phenotypic/_cli/_stages.py src/phenotypic/_cli/_cli_update_state.py src/phenotypic/_cli/_cli_state_management.py src/phenotypic/_cli/_dashboard/_manifest_builder.py src/phenotypic/_cli/_cli_stage4_worker.py tests/unit/cli/test_stage4_deferral.py tests/unit/cli/test_cli_update_state_stage.py
  git commit -m "feat(cli): Stage 3 and the single-pass core defer embeddings and the record (#282 P4)"
  ```

### Task 3: The Stage-4 worker

**Files:**
- Modify: `src/phenotypic/_cli/_cli_stage4_worker.py` (created in Task 2)
- Modify: `src/phenotypic/_cli/_cli_staged_strategy.py`: after the Stage-3
  `Parallel` loop (`:389-395`), `run_local_stage4` when deferring; the
  dataset results count an image completed only when Stage 4 published it.
- Modify: `src/phenotypic/_cli/_cli_execution_strategies.py`:
  `LocalParallelStrategy.execute` (`:369`) does the same after its loop.
- Modify: `src/phenotypic/_cli/_cli_staged_resume.py`: add `stage4_status`
  (below); Task 4 and P5 T1 consume it.
- Test: `tests/unit/cli/test_stage4_worker.py` (new) **[torch]**

**Interfaces:**
- Consumes: `BatchEncoder`, `BackboneKey`, `physical_batch_size`,
  `configure_determinism` (P2); `prepare_regions`, `vectors_from`,
  `payload_from`, `entry_descriptor_for`, `compute_block` (P1, P2);
  `write_stage4_signal`, `read_stage4_signal`, `stage4_signals_complete`,
  `delete_stage4_signals` (P3 T2); `replace_run_embeddings` (P3 T1);
  `StagedManifestEntry` and `staged_manifest_entry` (`_cli_staged_slurm.py:565`)
  as the work item; `owned_table_run_id` (`sdk_/_table_runs.py:829`);
  `publish_image_success` (`_cli_completion.py:217`), `image_data_artifact`,
  `image_measurement_artifact` (`:127,176`); `append_terminal_failure`
  (`_cli_failure_tracker.py:562`); `stage_event` (`_cli_staged_workers.py:204`);
  `require_license_acceptance` (`_checkpoint_manager.py:791`).
- Produces:
  - `class Stage4DeviceError(RuntimeError)`
  - `stage4_device() -> str`: `"cuda"`, or `"cpu"` when
    `PHENOTYPIC_STAGE4_DEVICE=cpu`. With `"cuda"`, `check_cuda_usable()` first:
    CUDA available, and `f"sm_{major}{minor}"` of device 0 in
    `torch.cuda.get_arch_list()`; else `Stage4DeviceError` naming the device and
    the arch list (spec §4.8, the P100 case).
  - `plan_stage4(pipeline: ImagePipeline, *, revisions: Mapping[str, str | None] | None = None) -> list[tuple[BackboneKey, list[tuple[str, MeasureDinoEmbedding]]]]`:
    sorted by `BackboneKey.sort_key()`; embedders sharing a key share one load (H10).
    `revisions` maps `hf_id` to a commit and fills `BackboneKey.revision` for an
    embedder whose `revision` is `None` (U5, P5 T2); the embedder is not copied.
  - `stage4_status(output_dir: Path, dataset: str, stem: str, work_id: str, *, republish: bool = False) -> Literal["done", "pending", "not_ready"]`,
    in `_cli_staged_resume.py` (spec §4.6): `done` = the record is valid
    (`valid_image_success`, `_cli_completion.py:342`); `pending` = the record
    holds `stages.stage3`, the store matches the work id
    (`staged_store_matches_work_id`, `:117`), a measurement table exists
    (`journal_measurement_table`), and there is no valid record; else
    `not_ready`. `republish` is P6 T1's.
  - `@dataclass class Stage4Outcome`: `published: list[str]`,
    `failed: dict[str, str]` (identity → error type), `pending: list[str]`,
    `physical_batch: dict[BackboneKey, int]`
  - `run_stage4(pipeline_path: Path, output_dir: Path, items: Sequence[StagedManifestEntry], *, image_type: ImageTypeName, run_pipeline_sha256: str | None, epoch: str | None, republish: bool = False, revisions: Mapping[str, str | None] | None = None, num_workers: int, output_manager: OutputManager, active_check: ActiveCheck | None = None, commit_guard: CommitGuard | None = None) -> Stage4Outcome`
  - `run_local_stage4(config: ExecutionConfig, output_manager: OutputManager, output_dir: Path, datasets: Sequence[Dataset]) -> dict[str, tuple[int, int]]`:
    `{dataset: (published, failed)}`. It raises `click.ClickException` (exit
    `1`) on `Stage4DeviceError`, `Stage4OutOfMemory` or a host `MemoryError`,
    after the commit pool drains, journalling no image.

**The loop** (spec §4.4; every step is an `active_check()` point):
1. Licence: for every v3 `BackboneKey`, `require_license_acceptance(..., interactive=False)`
   before any load. Then `stage4_device()` and `configure_determinism()`.
2. Per item, in order: the run id is `owned_table_run_id(block, run_pipeline_sha256)`;
   then
   - record already valid with signals left over → `delete_stage4_signals` (orphan);
   - store holds every key fresh, `producer.params_sha256` and `compute_dtype`
     matching → **publish only**;
   - `stage4_signals_complete` → **commit only**;
   - otherwise → to the loader.
3. Per backbone, in plan order: load once through the measurer module's
   `load_dino_backbone` (so the test stub applies; never `_cached_backbone`),
   with `dtype=key.compute_dtype`; compute `m`; stream the loader's images
   through one `BatchEncoder`; as each `(image, key)` completes,
   `vectors_from` → `payload_from(prepared, vectors, entry_descriptor_for(model, compute_block(..., stage="stage4", physical_batch=m)))`
   → `write_stage4_signal(..., root_sha256=<the root the loader read>)`. Then
   `del` the model and `torch.cuda.empty_cache()`.
4. A `ThreadPoolExecutor(max_workers=2)` commits each image whose signals are
   complete, under `commit_guard`: `replace_run_embeddings(store, run_id, payloads, expected_root_sha256=signal.root_sha256, objmap_sha256=..., array_sha256=...)`
   with the digests taken from the payloads' `binds` (equal to the store's,
   because the root matched); then the record (`publish_image_success`,
   `mode="full"`, the artifacts Stage 3 used to publish, `_cli_staged_slurm_worker.py:572-603`);
   then `delete_stage4_signals`. `republish=True` is P6 T1.
5. **DataLoader:** `torch.utils.data.DataLoader(Stage4ImageDataset(...), batch_size=None, num_workers=num_workers, persistent_workers=num_workers > 0, prefetch_factor=2 if num_workers else None)`,
   order-preserving (the default). An item is one image: the worker opens the
   store read-only, reads the root sha, and calls each embedder's
   `prepare_regions`. **A per-image exception returns a picklable
   `FailedImage(identity, error_type, message, traceback)`**; it never raises
   through the loader.
6. Failures: a `FailedImage`, a non-finite float16, or a
   `StoreRootChangedError` on its second occurrence for the image → that
   image's terminal `stage4` failure through `append_terminal_failure`
   (`failed_stage="stage4"`); the first `StoreRootChangedError` leaves it
   pending (spec §4.8, a plain retry next round).

- [ ] **Step 1: Write the failing tests.**

```python
def test_commit_then_record_then_signals(tmp_path, staged_tree, stage4_cpu, monkeypatch):
    order = spy_calls(monkeypatch, ["replace_run_embeddings", "publish_image_success", "delete_stage4_signals"])
    outcome = run_stage4_over(staged_tree)
    assert order == ["replace_run_embeddings", "publish_image_success", "delete_stage4_signals"] * len(outcome.published)

def test_backbone_major_one_load_per_backbone(staged_tree_two_backbones, stage4_cpu):
    run_stage4_over(staged_tree_two_backbones)
    assert [(c["dino_size"], c["dtype"]) for c in stage4_cpu.load_calls] == [("base", "float32"), ("small", "float32")]

def test_objects_and_grid_share_one_load(staged_tree_objects_and_grid, stage4_cpu):
    run_stage4_over(staged_tree_objects_and_grid)
    assert len(stage4_cpu.load_calls) == 1

def test_an_image_with_no_colonies_commits_an_empty_table(staged_tree_one_empty_plate, stage4_cpu):
    outcome = run_stage4_over(staged_tree_one_empty_plate)
    empty = store_of(staged_tree_one_empty_plate, "empty")
    assert identity_of(empty) in outcome.published
    assert pq.read_table(embedding_path(empty, "dinov2-vitb14/rgb/objects")).num_rows == 0

def test_a_loader_worker_error_fails_only_its_image(staged_tree, stage4_cpu):
    bad = corrupt_one_store(staged_tree)                  # deletes its objmap chunks
    outcome = run_stage4_over(staged_tree, num_workers=2)
    assert set(outcome.failed) == {identity_of(bad)}
    assert len(outcome.published) == n_images(staged_tree) - 1
    assert failure_stage(staged_tree, bad) == "stage4"

def test_no_cuda_fails_the_stage_without_journalling_images(tmp_path, tiny_inputs, stub_backbone, monkeypatch):
    monkeypatch.delenv("PHENOTYPIC_STAGE4_DEVICE", raising=False)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    out = tmp_path / "out"
    result = run_cli(tiny_inputs, out, pipeline=with_stage4_dino_objects(tmp_path), extra=["--embed-stage", "gpu"])
    assert result.exit_code == 1 and "CUDA" in result.output
    assert read_terminal_failures(out) == []
    assert all(stage4_status(out, *ds_stem_wid(s)) == "pending" for s in stores_of(out))

def test_dinov3_licence_is_refused_before_any_load(staged_tree_dinov3, stage4_cpu, monkeypatch):
    monkeypatch.delenv("PHENOTYPIC_ACCEPT_MODEL_LICENSE", raising=False)
    with pytest.raises(RuntimeError, match="licen"):
        run_stage4_over(staged_tree_dinov3)
    assert stage4_cpu.load_calls == []

def test_a_non_finite_vector_fails_the_image(staged_tree, stage4_cpu, monkeypatch):
    monkeypatch.setattr(MeasureDinoEmbedding, "vectors_from",
                        lambda self, p, o: np.full((len(p.keys), 8), 1e6, np.float32))
    outcome = run_stage4_over(staged_tree)
    assert len(outcome.failed) == n_images(staged_tree)

def test_stage4_entries_record_stage4_compute(staged_tree, stage4_cpu):
    run_stage4_over(staged_tree)
    compute = run_entry(store_of(staged_tree))["embeddings"]["dinov2-vitb14/rgb/objects"]["compute"]
    assert compute["stage"] == "stage4" and compute["physical_batch"] >= 1

def test_a_fenced_epoch_commits_nothing(staged_tree, stage4_cpu):
    with pytest.raises(SlurmGenerationInactiveError):
        run_stage4_over(staged_tree, active_check=fence_after_first_signal())
    assert all(not run_entry(s).get("embeddings") for s in stores_of(staged_tree))
```

  Create (helpers), in `tests/_tiny_run.py`: fixture `staged_tree` (a local
  `--embed-stage gpu` run of `with_stage4_dino_objects(fake_gpu=True)` with
  `run_local_stage4` patched out, so every image is Stage-4 pending) and its
  variants `staged_tree_two_backbones`, `staged_tree_objects_and_grid`,
  `staged_tree_one_empty_plate`, `staged_tree_dinov3`; `run_stage4_over(tree, **kw) -> Stage4Outcome`,
  which builds the items with `staged_manifest_entry` and calls `run_stage4`;
  `stores_of(out)`, `store_of(out, stem=None)`, `identity_of(store)`,
  `corrupt_one_store(out) -> Path`, `failure_stage(out, store) -> str`,
  `spy_calls`, `fence_after_first_signal`, `ds_stem_wid(store)`,
  `embedding_path(store, key)`.
- [ ] **Step 2: Run them and confirm they fail.**

  ```bash
  eval $TPT tests/unit/cli/test_stage4_worker.py
  ```

- [ ] **Step 3: Implement** the Interfaces and the loop above.
- [ ] **Step 4: Remove Task 1's `xfail`, and run the worker tests, the refusal
  tests, and the startup-import guards.**

  ```bash
  eval $TPT tests/unit/cli/test_stage4_worker.py tests/integration/cli/test_embed_stage_refusals.py
  eval $PT tests/unit/ci/test_startup_imports.py tests/unit/ci/test_deferred_imports.py
  ```

- [ ] **Step 5: Commit.**

  ```bash
  git add src/phenotypic/_cli/_cli_stage4_worker.py src/phenotypic/_cli/_cli_staged_strategy.py src/phenotypic/_cli/_cli_execution_strategies.py src/phenotypic/_cli/_cli_staged_resume.py tests/_tiny_run.py tests/conftest.py tests/unit/cli/test_stage4_worker.py tests/integration/cli/test_embed_stage_refusals.py
  git commit -m "feat(cli): the Stage-4 worker: backbone-major, batched, commit then record (#282 P4)"
  ```

### Task 4: Resume, retry, and crash injection

**Files:**
- Modify: `src/phenotypic/_cli/_cli_staged_resume.py`
  - `ResumeStage` (`:36`) gains `"stage4"`; `_STAGE_ORDER` (`:37-42`) puts it
    between `"stage3"` and `"complete"`; `StagedResumePlan.counts` (`:63-72`);
  - `classify_staged_image` (`:226`) gains `defers_embeddings: bool = False`: its
    "stage3 marker plus table" branch (`:276-283`) returns `"stage4"` when set;
  - `build_staged_resume_plan` (`:322`) passes it through;
  - `clear_downstream_artifacts_for_stage1` (`:424`) also calls
    `delete_stage4_signals`.
- Modify: `src/phenotypic/phenotypicCLI.py:3813-3900`: the continuation passes
  `defers_embeddings(config)`; an all-`stage4` plan sets
  `config.staged_resume_phase = "stage4"`.
- Modify: `src/phenotypic/_cli/_cli_stage4_worker.py`: add
  `split_stage4_pending(datasets: Sequence[Dataset], output_dir: Path, config: ExecutionConfig) -> tuple[list[Dataset], list[StagedManifestEntry]]`
  for a `LocalParallelStrategy` continuation: images `stage4_status == "pending"`
  skip the single-pass core and go straight to Stage 4.
- Test: `tests/integration/cli/test_local_stage4.py` (new) **[torch]**;
  `tests/unit/cli/test_staged_resume.py` (extend, shared env)

**Interfaces:**
- Consumes: Task 3.
- Produces: `ResumeStage` with `"stage4"`,
  `classify_staged_image(..., defers_embeddings=...)`, `split_stage4_pending`.
  P5 T1's `_classify_stage4` calls Task 3's `stage4_status`.

- [ ] **Step 1: Write the failing tests.**

```python
@pytest.mark.parametrize("fake_gpu", [True, False], ids=["staged", "single-pass"])
def test_local_stage4_matches_inline_deliverables(tmp_path, tiny_inputs, stage4_cpu, fake_gpu):
    pipe = with_stage4_dino_objects(tmp_path, fake_gpu=fake_gpu, grid=True)
    inline, gpu = tmp_path / "inline", tmp_path / "gpu"
    assert run_cli(tiny_inputs, inline, pipeline=pipe).exit_code == 0
    assert run_cli(tiny_inputs, gpu, pipeline=pipe, extra=["--embed-stage", "gpu"]).exit_code == 0
    for rel in embedding_deliverables(inline):
        assert (gpu / rel).read_bytes() == (inline / rel).read_bytes(), rel
    assert manifest_without_compute(gpu) == manifest_without_compute(inline)
    assert resolve_run_state(gpu, depth="deep").diagnostics.failed == 0

@pytest.mark.parametrize("crash_after, resumed_by", [
    ("write_stage4_signal", ["replace_run_embeddings", "publish_image_success", "delete_stage4_signals"]),
    ("replace_run_embeddings", ["publish_image_success", "delete_stage4_signals"]),
    ("publish_image_success", ["delete_stage4_signals"])])
def test_a_crash_resumes_from_disk(tmp_path, tiny_inputs, stage4_cpu, monkeypatch, crash_after, resumed_by):
    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path, fake_gpu=True)
    with crash_once_after(monkeypatch, crash_after) as crashed:          # yields the crashed image's stem
        run_cli(tiny_inputs, out, pipeline=pipe, extra=["--embed-stage", "gpu"])
    calls = spy_calls_for_stem(monkeypatch, crashed.stem,
        ["write_stage4_signal", "replace_run_embeddings", "publish_image_success", "delete_stage4_signals"])
    assert run_cli(tiny_inputs, out, pipeline=pipe, extra=["--embed-stage", "gpu"]).exit_code == 0
    assert calls == resumed_by                                           # never re-embedded
    assert no_stage4_signals(out) and all_records_valid(out)

def test_a_finished_run_rerun_embeds_nothing(tmp_path, tiny_inputs, stage4_cpu):
    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path, fake_gpu=True)
    run_cli(tiny_inputs, out, pipeline=pipe, extra=["--embed-stage", "gpu"])
    roots, loads = store_root_bytes(out), len(stage4_cpu.load_calls)
    assert run_cli(tiny_inputs, out, pipeline=pipe, extra=["--embed-stage", "gpu"]).exit_code == 0
    assert store_root_bytes(out) == roots and len(stage4_cpu.load_calls) == loads

def test_a_stage1_rerun_deletes_the_images_signals(tmp_path, tiny_inputs, stage4_cpu, monkeypatch):
    ...  # leave signals (crash after write_stage4_signal), invalidate one store, re-run: its signals are gone before Stage 4

def test_retry_failures_reruns_only_stage4(tmp_path, tiny_inputs, stage4_cpu, monkeypatch):
    ...  # fail one image in Stage 4; plain re-run skips it (exit 3); --retry-failures: no Stage-1/3 event for it, one stage4 event, exit 0

def test_restart_clears_stage4_signals(tmp_path, tiny_inputs, stage4_cpu, monkeypatch):
    ...  # signals left; --restart removes progress/stage4_payload with progress/
```

  `crash_once_after(monkeypatch, name)` raises a private `BaseException`
  subclass once, right after the named function first returns, so the run dies
  between two durable writes, and yields a namespace whose `stem` names that
  image. `spy_calls_for_stem` records, in order, which of the named functions
  the re-run calls for that stem. The three rows are spec §4.6's interrupted
  states: signals complete → commit only; promoted without record → publish
  only; record with signals → orphan cleanup.
  In `test_staged_resume.py`:

```python
def test_stage3_complete_without_a_record_classifies_stage4(tmp_path, staged_store_stage3_only):
    out, image, work_id = staged_store_stage3_only
    assert classify_staged_image(..., expected_work_id=work_id, defers_embeddings=True) == "stage4"
    assert classify_staged_image(..., expected_work_id=work_id) == "stage3"           # inline: unchanged
```

- [ ] **Step 2: Run them and confirm they fail.**

  ```bash
  eval $TPT tests/integration/cli/test_local_stage4.py
  eval $PT tests/unit/cli/test_staged_resume.py
  ```

- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the P4 phase gate**, as a Slurm job (detached worktree at
  the task's commit; see README gate shape), over importers of
  `_cli_staged_resume`, `_cli_staged_workers`, `_cli_staged_strategy`,
  `_cli_process_single`, `_cli_execution_strategies`, `_cli_update_state`,
  `_stages`, `phenotypicCLI`, plus `tests/integration/cli`, `tests/unit/gui` and
  `tests/gui` once, in both lanes. Expected: the only reds are the 13
  expected-red GUI tests.
- [ ] **Step 5: Commit.**

  ```bash
  git add src/phenotypic/_cli/_cli_staged_resume.py src/phenotypic/phenotypicCLI.py src/phenotypic/_cli/_cli_stage4_worker.py tests/_tiny_run.py tests/conftest.py tests/integration/cli/test_local_stage4.py tests/unit/cli/test_staged_resume.py
  git commit -m "feat(cli): Stage-4 resume is content-defined; --retry-failures re-runs only Stage 4 (#282 P4)"
  ```

**Deep review P3–P4** (implementation-test-reviewer, Opus) after the gate:
`reports/2026-11-12-gpu-embedding-staging/review-p3-p4.md`. It checks in
particular that every publishing site in the Files lists above is gated on one
predicate, and that no inline run's bytes moved (P0 T2's golden, and
`tests/integration/cli/test_embedding_deliverables.py`).
