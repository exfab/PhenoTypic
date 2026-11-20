# P4 · Local Stage 4: resolver, refusals, deferral, gated recovery, resume

**Spec:** U2, U3, U9, H5, H10, §4.1, §4.2, §4.4 (plan, loop, DataLoader),
§4.6, §4.7 (R1–R3), §4.8, §5 "Local end-to-end with a stub"

---

**Goal:** `--embed-stage gpu` works on one machine. Stage 3 (or the single-pass
core) defers embeddings and the record; a local Stage 4 embeds every pending
image backbone by backbone, commits each store, then publishes its record; and
an interrupted run resumes from what is on disk, without re-running Stages 1–3
for an image that only needs Stage 4.

**Depends on:** P2 (and P2 T4's measurement, stop rule 3), P3.

**Test fixtures used throughout** (create in Task 1, in `tests/_tiny_run.py`):
- `with_stage4_dino_objects(directory: Path, *, fake_gpu: bool = False, grid: bool = False) -> Path`:
  like `with_dino_objects` (`:1296`) but `device="auto"`, which U9 lets resolve
  to Stage 4. With `fake_gpu`, the ops gain `FakeGpuDetector`
  (`tests/_fakes/fake_gpu_detector.py`; import `tests._fakes.register_fake_gpu`
  first and set `PHENOTYPIC_PRELOAD_MODULES`). With `grid`, `meas["grid"]` is a
  2 × 2 `ManualGridFinder` embedder on the same backbone. With `faulty`, the
  objects embedder is `FaultyDinoEmbedding` (below), so a test can fail one
  image on disk.
- `FaultyDinoEmbedding`, in `tests/_fakes/faulty_dino_embedder.py` with a
  `register_faulty_dino_embedder.py` for `PHENOTYPIC_PRELOAD_MODULES`: a
  `MeasureDinoEmbedding` whose `prepare_regions` raises `ValueError` when the
  file `$PHENOTYPIC_TEST_FAULT_DIR/<image stem>` exists. **Per-image faults are
  injected through that on-disk trigger, never a monkeypatch** (re-review item
  2): a spawned DataLoader worker imports the class afresh and inherits the
  environment variable, but sees no monkeypatch. `plant_stage4_fault(stem: str) -> None`
  writes the trigger and `clear_stage4_fault(stem)` removes it; `nth_stem(inputs, i)`
  names an input's stem; the fixture `stage4_faults(monkeypatch, tmp_path)` sets
  the variable.
- fixture `stage4_cpu(monkeypatch, param_stub_backbone)`: sets
  `PHENOTYPIC_STAGE4_DEVICE=cpu` and `CUDA_VISIBLE_DEVICES=""`, **and** patches
  `torch.cuda.is_available` to `False`, because `CUDA_VISIBLE_DEVICES` does
  nothing once the process has initialised CUDA (re-review item 5). The inline
  comparison run's `device="auto"` then resolves to the CPU on a GPU node too
  (plan review I13). It yields `param_stub_backbone`.
- `run_cli_subprocess(inputs: Path, out: Path, *, pipeline: Path, extra: Sequence[str] = (), crash_after: str | None = None) -> subprocess.CompletedProcess`:
  one run in a fresh interpreter, built on the `_SUBPROCESS_RUN` pattern of
  `tests/integration/cli/test_embedding_deliverables.py:344-375` (parametric
  stub installed, `PYTHONPATH` set, `PHENOTYPIC_STAGE4_DEVICE=cpu`,
  `CUDA_VISIBLE_DEVICES=""`, `torch.cuda.is_available` patched to `False` before
  anything touches CUDA). With `crash_after`, the script wraps that function
  of `phenotypic._cli._cli_stage4_worker` so that right after it first returns
  the process calls `os._exit(137)`: a real crash between two durable writes,
  not an exception a caller can catch (plan review I5).

### Task 1: The resolver, R1–R3, CUDA and determinism at startup, and recording

**Files:**
- Create: `src/phenotypic/_cli/_cli_embed_stage.py`
- Modify: `src/phenotypic/_cli/_cli_types.py:271-276`: `ExecutionConfig` gains
  `embed_stage_request: EmbedStageRequest = "auto"` and
  `embed_stage: EmbedStage = "inline"` (the resolved value).
- Modify: `src/phenotypic/phenotypicCLI.py`
  - `--embed-stage` option beside `--gpu-slurm` (`:2366-2373`),
    `click.Choice(["auto", "gpu", "inline"])`, default `auto`;
  - `phenotypic_cli` (`:2512`) takes it; after the `ExecutionConfig` is built
    (`:3140-3158`), resolve and set `config.embed_stage`, and print U9's and
    O5's note lines;
  - R1 and R2 beside `_refuse_process_meas_it_would_lose` (`:3181`);
  - R3 right after `load_processing_state` (`:3224`), before
    `validate_resume_compatibility`;
  - when the resolved value is `gpu`, `prepare_stage4_environment(config)`
    right after resolution, before any strategy can initialise CUDA (plan
    review I4, I16).
- Modify: `src/phenotypic/_cli/_cli_state_management.py:362-385`: config gains
  `"embed_stage": config.embed_stage`.
- Modify: `src/phenotypic/sdk_/_io_constants.py:2990`: `JobMetadataKey` gains
  `EMBED_STAGE = "embed_stage"`, `EMBED_REVISIONS = "embed_revisions"` and
  `RUN_MODE = "run_mode"` (P6 T2 reads it).
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
  - `resolve_embed_stage(requested: EmbedStageRequest, *, mode: RunMode, is_slurm: bool, has_gpu_detector: bool, gpu_slurm_given: bool, has_embedders: bool, cpu_pinned: bool) -> EmbedStage`:
    the README's resolution table. `mode="process"` gives `"inline"`. Under
    `auto`, `cpu_pinned` (an in-scope embedder with `device="cpu"`) gives
    `"inline"` (U9). No embedder gives `"inline"` (O5).
  - `refuse_embed_stage_for_process(requested: EmbedStageRequest, mode: RunMode) -> None` (R1)
  - `refuse_unstageable_embedders(pipeline: ImagePipeline, requested: EmbedStageRequest, resolved: EmbedStage) -> None` (R2):
    `requested == "gpu"` with an embedder pinned to `device="cpu"` (U9); or
    `resolved == "gpu"` with a `MeasureEmbedding` that is not a
    `MeasureDinoEmbedding` (O3).
  - `refuse_embed_stage_switch(recorded: str | None, resolved: EmbedStage, *, is_slurm: bool) -> None` (R3);
    `None` reads `"inline"`.
  - `defers_embeddings(config: ExecutionConfig) -> bool`: `config.embed_stage == "gpu"`
  - `prepare_stage4_environment(config: ExecutionConfig) -> None`: sets
    `os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")` (it must be in
    the environment before the process's first cuBLAS handle; local Stage 4 runs
    in the CLI process, which Stage 2 may already have put on the GPU), then,
    **for a local run only**, `stage4_device()` (Task 3). A local run with no
    usable CUDA stops here with **exit `1`**, an environment error per the exit
    table (`RunExitCode.ERROR`, `_cli/CLAUDE.md` "Exit codes"), as one line,
    before Stage 1 writes anything (plan review I16). It raises
    `click.ClickException`, never a `UsageError`: the options are fine, the
    machine is not. The torch import this costs happens only for an explicit
    local `--embed-stage gpu` (a local `auto` never resolves `gpu`), and it
    happens in the read-only half, so a `--dry-run` with that flag pays it too;
    the tutorial says so (P7 T1).
  - The exact copy:
    - R1: `--embed-stage gpu does not apply to --mode process: process mode embeds inline on the CPU`
    - R2 (U9): `--embed-stage gpu was given, but meas '<key>' (<Class>) is pinned to device='cpu'; set device='auto' or 'cuda', or drop --embed-stage gpu to embed inline`
    - R2 (O3): `--embed-stage resolves to gpu, but meas '<key>' (<Class>) has no GPU Stage-4 encoder; pass --embed-stage inline`
    - R3: `this run was started with --embed-stage <recorded>; continuing it as <resolved> would mix CPU and GPU embeddings. Re-run with --embed-stage <recorded>, or re-embed the finished run with --mode measure --embed-stage gpu`,
      plus `(on SLURM, also --gpu-slurm)` when `is_slurm` (plan review, Minor)
    - U9 note: `meas '<key>' is pinned to device='cpu', so --embed-stage auto resolves to inline`
    - O5 note: `--embed-stage gpu: the pipeline has no embedder, so nothing runs in Stage 4`

- [ ] **Step 1: Write the failing tests.**

```python
@pytest.mark.parametrize("requested, mode, slurm, detector, gpu_slurm, embedders, cpu_pinned, expected", [
    ("auto", "full", True, True, False, True, False, "gpu"),
    ("auto", "full", True, False, True, True, False, "gpu"),
    ("auto", "full", True, False, False, True, False, "inline"),
    ("auto", "full", False, True, True, True, False, "inline"),
    ("gpu", "full", False, False, False, True, False, "gpu"),
    ("auto", "measure", True, True, False, True, False, "inline"),   # only --gpu-slurm counts
    ("auto", "measure", True, False, True, True, False, "gpu"),
    ("auto", "process", True, True, True, True, False, "inline"),
    ("gpu", "full", True, True, True, False, False, "inline"),       # O5
    ("auto", "full", True, True, True, True, True, "inline"),        # U9
    ("inline", "full", True, True, True, True, False, "inline"),
])
def test_resolution_table(requested, mode, slurm, detector, gpu_slurm, embedders, cpu_pinned, expected):
    assert resolve_embed_stage(requested, mode=mode, is_slurm=slurm, has_gpu_detector=detector,
                               gpu_slurm_given=gpu_slurm, has_embedders=embedders,
                               cpu_pinned=cpu_pinned) == expected

def test_r1_refuses_gpu_in_process_mode():
    with pytest.raises(click.UsageError, match="does not apply to --mode process"):
        refuse_embed_stage_for_process("gpu", "process")

def test_r2_refuses_a_cpu_pin_only_with_an_explicit_gpu():
    pipe = ImagePipeline(meas={"emb": MeasureDinoEmbedding(device="cpu")})
    with pytest.raises(click.UsageError, match=r"meas 'emb' \(MeasureDinoEmbedding\) is pinned to device='cpu'"):
        refuse_unstageable_embedders(pipe, "gpu", "gpu")
    refuse_unstageable_embedders(pipe, "auto", "inline")       # U9: auto resolved inline, accepted

def test_r2_refuses_a_non_dino_embedder_under_gpu(toy_embedder):
    with pytest.raises(click.UsageError, match="has no GPU Stage-4 encoder"):
        refuse_unstageable_embedders(ImagePipeline(meas={"emb": toy_embedder}), "auto", "gpu")

def test_r3_names_the_recorded_flag_and_the_remedy():
    with pytest.raises(click.UsageError, match=r"started with --embed-stage inline.*--mode measure --embed-stage gpu"):
        refuse_embed_stage_switch("inline", "gpu", is_slurm=False)
    with pytest.raises(click.UsageError, match="also --gpu-slurm"):
        refuse_embed_stage_switch("inline", "gpu", is_slurm=True)
    refuse_embed_stage_switch(None, "inline", is_slurm=False)   # a pre-#282 state

def test_embed_stage_is_not_in_the_work_id(make_exec_config, tiny_pipeline_file, tmp_path):
    a = make_exec_config(pipeline_json=tiny_pipeline_file, input_path=tmp_path)
    b = dataclasses.replace(a, embed_stage="gpu", embed_stage_request="gpu")
    assert processing_configuration_digest(a) == processing_configuration_digest(b)

def test_the_cublas_variable_is_set_before_any_strategy(monkeypatch, make_exec_config, tiny_pipeline_file, tmp_path):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setenv("PHENOTYPIC_STAGE4_DEVICE", "cpu")
    prepare_stage4_environment(dataclasses.replace(
        make_exec_config(pipeline_json=tiny_pipeline_file, input_path=tmp_path), embed_stage="gpu"))
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
```

  In `test_cli_preflight_ordering.py`, add R1, R2 and R3 to the cases that must
  refuse before `--dry-run` and before `--overwrite` deletes anything, in the
  file's own style. In `test_embed_stage_refusals.py` **[torch]**:

```python
def test_r3_refuses_a_switched_continuation(tmp_path, tiny_inputs, stage4_cpu):
    out = tmp_path / "out"
    pipe = with_stage4_dino_objects(tmp_path)
    assert run_cli(tiny_inputs, out, pipeline=pipe).exit_code == 0          # local auto: inline, recorded
    result = run_cli(tiny_inputs, out, pipeline=pipe, extra=["--embed-stage", "gpu"])
    assert result.exit_code == 2 and "started with --embed-stage inline" in result.output

def test_no_cuda_stops_before_stage1(tmp_path, tiny_inputs, param_stub_backbone, monkeypatch):
    monkeypatch.delenv("PHENOTYPIC_STAGE4_DEVICE", raising=False)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    out = tmp_path / "out"
    result = run_cli(tiny_inputs, out, pipeline=with_stage4_dino_objects(tmp_path), extra=["--embed-stage", "gpu"])
    assert result.exit_code == 1 and "CUDA" in result.output
    assert not list(out.rglob("*.ome.zarr")) and read_terminal_failures(out) == []
```

  `test_no_cuda_stops_before_stage1` passes only once Task 3 provides
  `stage4_device`; mark it `pytest.mark.xfail(strict=True, reason="P4 T3")` and
  remove the mark in Task 3 Step 4.
- [ ] **Step 2: Run them and confirm they fail.**

  ```bash
  eval $PT tests/unit/cli/test_cli_embed_stage.py tests/unit/cli/test_cli_preflight_ordering.py
  eval $TPT tests/integration/cli/test_embed_stage_refusals.py
  ```

- [ ] **Step 3: Implement.** The resolver needs `has_gpu_detector`,
  `has_embedders` and `cpu_pinned` before the mutating half: reuse the pipeline
  load the staged-GPU refusal probe already does (`uses_staged_gpu_strategy`,
  `phenotypicCLI.py:3165`). A load failure resolves `inline` and is left to the
  pipeline validation below it, as that probe does.
- [ ] **Step 4: Run the tests.** Expected: all pass; the `xfail` reads `XFAIL`.
- [ ] **Step 5: Commit.**

  ```bash
  git add src/phenotypic/_cli/_cli_embed_stage.py src/phenotypic/_cli/_cli_types.py src/phenotypic/phenotypicCLI.py src/phenotypic/_cli/_cli_state_management.py src/phenotypic/sdk_/_io_constants.py src/phenotypic/_cli/_cli_staged_slurm.py src/phenotypic/_cli/_cli_execution_strategies.py tests/_tiny_run.py tests/conftest.py tests/unit/cli/test_cli_embed_stage.py tests/unit/cli/test_cli_preflight_ordering.py tests/integration/cli/test_embed_stage_refusals.py
  git commit -m "feat(cli): --embed-stage, its resolution, refusals R1-R3, and the Stage-4 environment (#282 P4)"
  ```

### Task 2: Deferral, gated recovery shortcuts, the completion predicate, and the `stage4` tag

**One predicate, one rule** (plan review C1). Under deferral, an image whose
record holds `stages.stage3`, whose store matches its work id and which has a
measurement table has **finished Stages 1–3**. Every shortcut that used to
publish it now returns early and publishes nothing; every stage loop skips it;
Stage 4 owns it from there. Every re-run of a core over an image deletes that
image's Stage-4 signals first.

**Files:**
- Modify: `src/phenotypic/_cli/_cli_staged_resume.py`: add
  `stages_1_to_3_done(output_dir: Path, dataset: str, stem: str, work_id: str) -> bool`:
  `stage3_completion_exists` (`:136`) and `staged_store_matches_work_id`
  (`:117`) and `journal_measurement_table(store) is not None`.
- Modify: `src/phenotypic/_cli/_cli_staged_workers.py:575-773`.
  `stage3_merge_measure_core` gains `defer_embeddings: bool = False`; when set,
  `replay_pipeline.embed` (`:658`) is not called and `save_image_store` gets
  `embeddings=None`.
- Modify: `src/phenotypic/_cli/_cli_process_single.py:202-437`.
  `process_single_image_core` gains `defer_embeddings: bool = False`; when set,
  it first drops the image's old `stages.stage3` (`remove_stage3_completion_marker`,
  `_cli_staged_resume.py:203`) and its Stage-4 signals (`delete_stage4_signals`),
  so a crash mid-core can never leave a "done" image behind; then it skips
  `pipeline.embed` (`:368`), and
  after the final save writes `write_stage3_completion_marker`
  (`_cli_staged_resume.py:162`), so the record holds `stages.stage3` and no
  artifacts.
- Modify: `src/phenotypic/_cli/_cli_staged_strategy.py`
  - `_terminal_output_exists` (`:122-198`): under deferral it returns
    `valid_image_success(...) or stages_1_to_3_done(...)` and never publishes.
    That one function also gates **Stage 2** (`:250-259`), so an image waiting
    on Stage 4, whose token and raw are already consumed, is not re-detected;
  - `_stage3` (`:315-388`): passes `defer_embeddings`, skips
    `_publish_local_image_success` (`~:345`) when deferring, and keeps the
    Stage-3 marker (`:354`) and the Stage-2 consumption.
- Modify: `src/phenotypic/_cli/_cli_staged_slurm_worker.py:432-642`. Under
  deferral, `run_stage3_step` returns early when `stages_1_to_3_done` (before
  the Stage-2 prereq check at `:536-552`, which would otherwise `SystemExit(1)`
  on the consumed signal); its publish (`:572-603`) is skipped. The worker reads
  deferral from `job_metadata.json` `EMBED_STAGE` (P5 adds `phases`, which
  carries the same fact).
- Modify: `src/phenotypic/_cli/_cli_execution_strategies.py:525-650`.
  `_process_single_local` passes `defer_embeddings`, skips
  `_publish_local_image_success` (`:574`) when deferring, and, **only when
  deferring**, tags its events `stage="stage3"`. An inline run's events stay
  untagged, as today.
- Modify: `src/phenotypic/phenotypicCLI.py`
  - **the local completion predicate** (plan review C2): under deferral,
    `_stage3_less_images` (`:1535-1553`) at its call (`:4329`) is replaced by
    `_unpublished_images(output_dir, inventory)`, the images with no valid
    record; `staged_complete` (`:4332-4336`) and `staged_terminal_incomplete`
    (`:4342-4346`) are computed from it, so a Stage-4 failure exits `3`, and a
    pending image is never marked staged-complete;
  - `local_staged_publication` (`:4198-4201`) is also true for a local
    no-detector run with `defers_embeddings(config)`;
  - so the finalization's `reconcile_stage3_publications(..., staged_detector_slot(config.pipeline_json), ...)`
    (`:4264-4271`) is now reached by a pipeline with no detector, which
    `staged_detector_slot` refuses (`_cli_stage2_token.py:98`). Guard that call
    on the pipeline holding a `GpuDetector` (`uses_staged_gpu_strategy(config)`),
    and correct its comment (`:4265-4266`), which says the guard implies one
    (re-review item 1).
- Modify: `src/phenotypic/_cli/_stages.py`: `StageTag` gains `"stage4"`;
  `STAGE_EMBED: StageTag = "stage4"`; `VALID_STAGE_TAGS` includes it; add
  `run_terminal_stage(state_config: Mapping[str, object] | None) -> StageTag`.
- Modify: `src/phenotypic/_cli/_cli_update_state.py:266-420`.
  `aggregate_state_from_events` and `..._with_diagnostics` gain
  `terminal_stage: StageTag = STAGED_TERMINAL_STAGE`, used at `:372`. **Every**
  caller passes `run_terminal_stage(state.config)` (plan review I8):
  `_cli_state_management.py:269` and `:427`, `_cli_checkpoint_handler.py:544`,
  `_dashboard/_manifest_builder.py:337` and `:526`, `sdk_/generate_report.py:83`
  (which already imports `_cli._cli_update_state`).
- Test: `tests/unit/cli/test_stage4_deferral.py` (new) **[torch]**;
  `tests/unit/cli/test_cli_update_state_stage.py` (extend, shared env)

**Interfaces:**
- Consumes: `defers_embeddings` (Task 1).
- Produces: `stages_1_to_3_done`; the `defer_embeddings` keyword on both cores
  (P5 T2 and P6 call them); `_unpublished_images`; `STAGE_EMBED`;
  `run_terminal_stage`; `aggregate_state_from_events(..., terminal_stage=...)`.

- [ ] **Step 1: Write the failing tests.**

```python
@pytest.fixture
def no_stage4(monkeypatch):
    monkeypatch.setattr(stage4_worker_module, "run_local_stage4", lambda *a, **k: {})

@pytest.mark.parametrize("fake_gpu", [True, False], ids=["staged", "single-pass"])
def test_deferral_writes_no_record_and_no_embeddings(tmp_path, tiny_inputs, stage4_cpu, no_stage4, fake_gpu):
    out = tmp_path / "out"
    run_cli(tiny_inputs, out, pipeline=with_stage4_dino_objects(tmp_path, fake_gpu=fake_gpu),
            extra=["--embed-stage", "gpu"])
    store = one_store(out)
    record = read_image_record(out, *dataset_and_stem(store))
    assert set(record["stages"]) == {"stage3"} and not record.get("artifacts")
    assert "embeddings" not in run_entry(store)
    assert journal_measurement_table(store) is not None

def test_a_waiting_image_is_neither_redetected_nor_remeasured(tmp_path, tiny_inputs, stage4_cpu, no_stage4, monkeypatch):
    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path, fake_gpu=True)
    run_cli(tiny_inputs, out, pipeline=pipe, extra=["--embed-stage", "gpu"])
    roots = store_root_bytes(out)
    calls = count_calls(monkeypatch, ["stage2_detect_core", "stage3_merge_measure_core", "publish_image_success"])
    assert run_cli(tiny_inputs, out, pipeline=pipe, extra=["--embed-stage", "gpu"]).exit_code != 2
    assert calls == {"stage2_detect_core": 0, "stage3_merge_measure_core": 0, "publish_image_success": 0}
    assert store_root_bytes(out) == roots

def test_the_slurm_stage3_step_returns_early_for_a_waiting_image(tmp_path, stage4_slurm_waiting):
    item = stage4_slurm_waiting.items[0]
    run_stage3_step(**stage4_slurm_waiting.stage3_kwargs(0))                 # no SystemExit
    assert not valid_image_success(stage4_slurm_waiting.out, dataset=item.dataset,
                                   image_stem=item.stem, work_id=item.work_id)

def test_a_core_rerun_deletes_the_images_signals(tmp_path, stage4_pending_with_signals):
    out, item = stage4_pending_with_signals
    invalidate_store(out, item)                                           # forces the core to re-run it
    run_cli(*stage4_pending_with_signals.cli_args, extra=["--embed-stage", "gpu"])
    assert not any(signal_files(out, item))

def test_a_no_detector_local_run_finalizes(tmp_path, tiny_inputs, stage4_cpu):
    out = tmp_path / "out"
    result = run_cli(tiny_inputs, out, pipeline=with_stage4_dino_objects(tmp_path, fake_gpu=False),
                     extra=["--embed-stage", "gpu"])
    assert result.exit_code == 0, result.output                 # no staged_detector_slot refusal
    assert (out / "deliverables" / "embeddings" / "manifest.json").is_file()

def test_a_core_rerun_drops_the_old_stage3_first(tmp_path, stage4_pending_with_signals, monkeypatch):
    out, item = stage4_pending_with_signals
    invalidate_store(out, item)
    crash_inside_the_core(monkeypatch, item)                    # raises after the drop, before the save
    run_cli(*stage4_pending_with_signals.cli_args, extra=["--embed-stage", "gpu"])
    assert not stages_1_to_3_done(out, item.dataset, item.stem, item.work_id)

def test_inline_events_stay_untagged(tmp_path, tiny_inputs, stage4_cpu):
    out = tmp_path / "out"
    run_cli(tiny_inputs, out, pipeline=with_stage4_dino_objects(tmp_path))       # local auto: inline
    assert {e.stage for e in read_events(out)} == {None}

def test_process_mode_ignores_embed_stage(tmp_path, tiny_inputs, stage4_cpu):
    out = tmp_path / "out"
    result = run_cli(tiny_inputs, out, mode="process", pipeline=with_stage4_dino_objects(tmp_path),
                     extra=["--layer", "rgb", "--process-format", "zarr"])
    assert result.exit_code == 0
    entry = run_entry(one_store(out))["embeddings"]["dinov2-vitb14/rgb/objects"]
    assert entry["compute"]["stage"] == "inline"
```

  `crash_inside_the_core` makes the core raise right after its first write;
  `read_events(out)` parses the event log. `stage4_worker_module` is
  `phenotypic._cli._cli_stage4_worker`, created in
  this task importable, with a `run_local_stage4` placeholder that raises
  `NotImplementedError`, so the monkeypatch target exists; Task 3 fills it.
  Create (helpers): `count_calls(monkeypatch, names) -> dict[str, int]`,
  `store_root_bytes(out) -> dict[Path, bytes]`, `dataset_and_stem(store)`,
  `invalidate_store(out, item)`, `signal_files(out, item)`, fixtures
  `stage4_slurm_waiting` and `stage4_pending_with_signals`. In
  `test_cli_update_state_stage.py`:

```python
def test_stage3_completion_is_not_completion_when_the_run_embeds_on_gpu(tmp_path):
    log = event_log_with(tmp_path, [("ds", "a", "completed", "stage3")])
    assert "a" not in aggregate_state_from_events(log, terminal_stage="stage4")["ds"].completed
    assert "a" in aggregate_state_from_events(log)["ds"].completed

def test_run_terminal_stage():
    assert run_terminal_stage({"embed_stage": "gpu"}) == "stage4"
    assert run_terminal_stage({}) == run_terminal_stage(None) == "stage3"
```

  The completion predicate's own test (a Stage-4 failure exits `3`; a pending
  image is never staged-complete) is in Task 4, once Stage 4 can fail.
- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.** Every gate reads `defers_embeddings(config)`, or on
  the SLURM worker the recorded value; no site re-derives it from flags.
- [ ] **Step 4: Run the tests, and the staged and single-pass suites they touch.**

  ```bash
  eval $TPT tests/unit/cli/test_stage4_deferral.py tests/unit/cli/test_staged_resume.py tests/unit/cli/test_staged_resume_parity.py tests/unit/cli/test_staged_nested_equivalence.py
  eval $PT tests/unit/cli/test_cli_update_state_stage.py tests/unit/cli/test_staged_routing.py tests/unit/cli/test_cli_progress_dashboard.py
  ```

  Expected: all pass. Inline runs are unchanged: nothing defers unless
  `embed_stage == "gpu"`.
- [ ] **Step 5: Commit.**

  ```bash
  git add src/phenotypic/_cli/_cli_staged_resume.py src/phenotypic/_cli/_cli_staged_workers.py src/phenotypic/_cli/_cli_process_single.py src/phenotypic/_cli/_cli_staged_strategy.py src/phenotypic/_cli/_cli_staged_slurm_worker.py src/phenotypic/_cli/_cli_execution_strategies.py src/phenotypic/phenotypicCLI.py src/phenotypic/_cli/_stages.py src/phenotypic/_cli/_cli_update_state.py src/phenotypic/_cli/_cli_state_management.py src/phenotypic/_cli/_cli_checkpoint_handler.py src/phenotypic/_cli/_dashboard/_manifest_builder.py src/phenotypic/sdk_/generate_report.py src/phenotypic/_cli/_cli_stage4_worker.py tests/_tiny_run.py tests/conftest.py tests/unit/cli/test_stage4_deferral.py tests/unit/cli/test_cli_update_state_stage.py
  git commit -m "feat(cli): Stages 1-3 defer embeddings and the record; one done-predicate gates every shortcut (#282 P4)"
  ```

### Task 3: The Stage-4 worker

**Files:**
- Modify: `src/phenotypic/_cli/_cli_stage4_worker.py` (created in Task 2)
- Modify: `src/phenotypic/_cli/_cli_staged_resume.py`: add `stage4_status`.
- Modify: `src/phenotypic/_cli/_cli_staged_strategy.py`: after the Stage-3
  `Parallel` loop (`:389-395`), `run_local_stage4` when deferring; an image
  counts completed only when Stage 4 published it.
- Modify: `src/phenotypic/_cli/_cli_execution_strategies.py`:
  `LocalParallelStrategy.execute` (`:369`) does the same after its loop.
- Test: `tests/unit/cli/test_stage4_worker.py` (new) **[torch]**

**Interfaces:**
- Consumes: `BatchEncoder`, `BackboneKey`, `physical_batch_size`,
  `configure_determinism` (P2); `prepare_regions`, `vectors_from`,
  `payload_from`, `entry_descriptor_for`, `compute_block`,
  `load_dino_processor` (P1, P2); `write_stage4_signal`, `read_stage4_signal`,
  `classify_stage4_signals`, `delete_stage4_signal_keys`,
  `delete_stage4_signals` (P3 T2); `replace_run_embeddings` (P3 T1);
  `StagedManifestEntry` and `staged_manifest_entry` (`_cli_staged_slurm.py:565`)
  as the work item; `journal_table_run` (`sdk_/_table_runs.py:861`);
  `objmap_digest`, `array_digest` (`sdk_/_measurement_tables.py:223,245`);
  `publish_image_success` (`_cli_completion.py:217`), `image_data_artifact`,
  `image_measurement_artifact` (`:127,176`); `append_terminal_failure`
  (`_cli_failure_tracker.py:562`); `stage_event` (`_cli_staged_workers.py:204`);
  `require_license_acceptance` (`_checkpoint_manager.py:791`);
  `resolve_local_worker_count` (`_cli_utils.py:88`).
- Produces:
  - `class Stage4DeviceError(RuntimeError)`
  - `stage4_device() -> str`: `"cuda"`, or `"cpu"` when
    `PHENOTYPIC_STAGE4_DEVICE=cpu`. With `"cuda"`, `check_cuda_usable()` first:
    CUDA available, and `f"sm_{major}{minor}"` of device 0 in
    `torch.cuda.get_arch_list()`; else `Stage4DeviceError` naming the device and
    the arch list (spec §4.8, the P100 case).
  - `stage4_status(output_dir: Path, dataset: str, stem: str, work_id: str, *, republish: bool = False) -> Literal["done", "pending", "not_ready"]`
    in `_cli_staged_resume.py` (spec §4.6): `done` = the record is valid
    (`valid_image_success`, `_cli_completion.py:342`); `pending` =
    `stages_1_to_3_done` and no valid record; else `not_ready`. `republish` is
    P6 T1's.
  - `plan_stage4(pipeline: ImagePipeline, *, revisions: Mapping[str, str | None] | None = None) -> list[tuple[BackboneKey, list[tuple[str, MeasureDinoEmbedding]]]]`:
    sorted by `BackboneKey.sort_key()`; embedders sharing a key share one load
    (H10). `revisions` maps `hf_id` to a commit and fills `BackboneKey.revision`
    for an embedder whose `revision` is `None` (U5, P5 T2); the embedder itself
    is not copied, so `producer.params_sha256` stays the inline one.
  - `@dataclass class Stage4Outcome`: `published: list[str]`,
    `failed: dict[str, str]` (identity → error type), `pending: list[str]`,
    `signals_written: int`, `physical_batch: dict[BackboneKey, int]`
  - `sweep_orphan_stage4_signals(output_dir: Path, inventory: Mapping[str, Sequence[str]], work_ids: Mapping[str, Mapping[str, str]], *, commit_guard: CommitGuard | None = None) -> int`:
    every image of the **whole inventory** whose record is valid loses its
    leftover signals (plan review I7; a done image never reaches the loop).
  - `run_stage4(pipeline_path: Path, output_dir: Path, items: Sequence[StagedManifestEntry], *, image_type: ImageTypeName, epoch: str | None, republish: bool = False, revisions: Mapping[str, str | None] | None = None, num_workers: int, output_manager: OutputManager, active_check: ActiveCheck | None = None, commit_guard: CommitGuard | None = None) -> Stage4Outcome`
  - `run_local_stage4(config: ExecutionConfig, output_manager: OutputManager, output_dir: Path, datasets: Sequence[Dataset]) -> dict[str, tuple[int, int]]`:
    `{dataset: (published, failed)}`. It sweeps orphans first, uses
    `num_workers = resolve_local_worker_count(config.n_jobs, n_items) - 1`, so
    `--njobs` is honoured and `--njobs 1` runs the loader **in-process**, with
    no worker at all (re-review item 2, plan review I15). It raises
    `click.ClickException` (exit `1`) after the commit pool drains, journalling
    no image, on `Stage4DeviceError`, `Stage4OutOfMemory`, a host
    `MemoryError`, **or any image left pending** (a store root that changed at
    commit): `"<n> image(s) still need Stage 4 because their store changed
    during the run; run the same command again"` (re-review, Minor). Every other
    image it is handed ends published or journalled (plan review C2).

**The loop** (spec §4.4). Every durable step is preceded by `active_check()`,
and every durable write runs under `commit_guard`.
1. Licence: for every v3 `BackboneKey`, `require_license_acceptance(..., interactive=False)`
   before any load. Then `stage4_device()` and `configure_determinism()`.
2. Per item, in order, with `run = journal_table_run(store)` (the forward
   run's folder, plan review I10; P6 uses the measure pipeline's) and the live
   root sha:
   - `classify_stage4_signals(..., live_root_sha256=<live>)`; delete the
     `stale` keys (plan review C5);
   - every key valid → **commit only**;
   - the store already holds every key fresh, its files verify
     (`read_embedding_entry`, `sdk_/_embedding_tables.py:417`), and
     `producer.params_sha256` and `compute_dtype` match → **publish only**
     (forward mode only; P6's republish always re-embeds);
   - otherwise → to the loader, for **only the keys without a valid signal**
     (plan review C4).
3. Per backbone, in plan order: load once through the measurer module's
   `load_dino_backbone` (so the test stub applies; never `_cached_backbone`),
   with `dtype=key.compute_dtype`; compute `m`; stream the loader's images
   through one `BatchEncoder`; as each `(image, key)` drains, `vectors_from` →
   `payload_from(prepared, vectors, entry_descriptor_for(model, compute_block(..., stage="stage4", physical_batch=m)))`
   → `write_stage4_signal(..., root_sha256=<the root the loader read>)`. Then
   `del` the model and `torch.cuda.empty_cache()`. A key's signal is written
   when that key drains, so a walltime loses at most the batches in flight.
4. A **one-thread** commit pool (`ThreadPoolExecutor(max_workers=1)`, plan
   review Cuts) commits each image whose signals are all valid, overlapping GPU
   work:
   1. read the live root; if it is not the signals' root, delete them and leave
      the image pending (spec §4.8: a plain retry, never terminal; C5);
   2. compute `objmap_sha256` and `array_sha256` **from the live store**
      (plan review I11). A payload whose `binds` do not match them would be
      written `stale: true`; instead the image fails here as a per-image
      `stage4` failure, its signals are deleted, and nothing is committed or
      published (re-review, Minor: never a published-then-dropped entry);
   3. `replace_run_embeddings(store, run.run_id, payloads, expected_root_sha256=<signals' root>, objmap_sha256=..., array_sha256=...)`;
      a `StoreRootChangedError` deletes the signals and leaves the image
      pending;
   4. the record: `publish_image_success`, `mode="full"`, the artifacts Stage 3
      used to publish (`_cli_staged_slurm_worker.py:572-603`); P6 T1 adds the
      republish form;
   5. `delete_stage4_signals`.
5. **DataLoader** (plan review I3): one `torch.utils.data.DataLoader` per
   backbone over `Stage4ImageDataset(items, keys_for_backbone, processors, ...)`, with
   `batch_size=None`, `collate_fn=_identity_collate` (module level, so it
   pickles; ndarrays stay ndarrays), `num_workers=num_workers`,
   `multiprocessing_context="spawn"` (CUDA is live and the commit thread exists
   when workers start), `persistent_workers=False` (no per-backbone state
   survives), `prefetch_factor=2` when `num_workers > 0`, and the default
   in-order iteration. `processors` maps each `BackboneKey` with a
   `masked_dense` key to its processor, loaded once **in the main process** with
   `load_dino_processor` and pickled into the dataset; a worker loads nothing
   (re-review item 2). An item is one image: the worker opens the store read
   only, reads the root sha, and calls each pending key's
   `prepare_regions(image, processor=...)`. **A per-image exception returns a picklable
   `FailedImage(identity, error_type, message, traceback)`**; it never raises
   through the loader.
6. Failures that are the image's: a `FailedImage`, a non-finite float16 → a
   terminal `stage4` failure through `append_terminal_failure`
   (`failed_stage="stage4"`). Node state (OOM, no CUDA) and a changed root are
   never terminal (spec §4.8, U7).

- [ ] **Step 1: Write the failing tests.**

```python
def test_commit_then_record_then_signals_per_image(staged_tree, stage4_cpu, monkeypatch):
    calls = spy_calls_by_image(monkeypatch, ["replace_run_embeddings", "publish_image_success", "delete_stage4_signals"])
    outcome = run_stage4_over(staged_tree)
    assert set(calls) == set(outcome.published)
    assert all(seq == ["replace_run_embeddings", "publish_image_success", "delete_stage4_signals"]
               for seq in calls.values())

def test_backbone_major_one_load_per_backbone(staged_tree_two_backbones, stage4_cpu):
    run_stage4_over(staged_tree_two_backbones)
    assert [(c["dino_size"], c["dtype"]) for c in stage4_cpu.load_calls] == [("base", "float32"), ("small", "float32")]

def test_objects_and_grid_share_one_load(staged_tree_objects_and_grid, stage4_cpu):
    run_stage4_over(staged_tree_objects_and_grid)
    assert len(stage4_cpu.load_calls) == 1

def test_a_second_round_embeds_only_the_keys_it_lacks(staged_tree_two_backbones, stage4_cpu, monkeypatch):
    write_valid_signals(staged_tree_two_backbones, keys=["dinov2-vitb14/rgb/objects"])  # backbone 1 done
    run_stage4_over(staged_tree_two_backbones)
    assert [c["dino_size"] for c in stage4_cpu.load_calls] == ["small"]                 # plan review C4

def test_a_signal_from_an_old_root_is_deleted_and_reembedded(staged_tree, stage4_cpu):
    write_valid_signals(staged_tree, root_sha256="0" * 64)                              # before a reprocess
    outcome = run_stage4_over(staged_tree)
    assert len(stage4_cpu.load_calls) == 1 and len(outcome.published) == n_images(staged_tree)

def test_a_root_change_at_commit_leaves_the_image_pending(staged_tree, stage4_cpu, monkeypatch):
    touch_root_before_commit(monkeypatch, staged_tree, image=0)
    outcome = run_stage4_over(staged_tree)
    assert identity_of(store_of(staged_tree, 0)) in outcome.pending
    assert read_terminal_failures(staged_tree) == []                                   # never terminal

def test_a_local_run_left_pending_exits_1(tmp_path, tiny_inputs, stage4_cpu, monkeypatch):
    touch_root_before_commit(monkeypatch, None, image=0)                               # inside the CLI run
    result = run_cli(tiny_inputs, tmp_path / "out", pipeline=with_stage4_dino_objects(tmp_path, fake_gpu=True),
                     extra=["--embed-stage", "gpu"])
    assert result.exit_code == 1 and "still need Stage 4" in result.output

def test_a_payload_bound_to_another_image_fails_the_image(staged_tree, stage4_cpu):
    write_valid_signals(staged_tree, binds_from_another_objmap=True)                   # checked against the live store
    outcome = run_stage4_over(staged_tree)
    assert len(outcome.failed) == n_images(staged_tree)
    assert all(not run_entry(s).get("embeddings") for s in stores_of(staged_tree))   # never published-then-dropped
    assert {failure_stage(staged_tree, s) for s in stores_of(staged_tree)} == {"stage4"}

def test_an_image_with_no_colonies_commits_an_empty_table(staged_tree_one_empty_plate, stage4_cpu):
    outcome = run_stage4_over(staged_tree_one_empty_plate)
    empty = store_of(staged_tree_one_empty_plate, "empty")
    assert identity_of(empty) in outcome.published
    assert pq.read_table(embedding_path(empty, "dinov2-vitb14/rgb/objects")).num_rows == 0

@pytest.mark.parametrize("num_workers", [0, 2])
def test_a_loader_worker_error_fails_only_its_image(staged_tree_faulty, stage4_cpu, stage4_faults, num_workers):
    bad = store_of(staged_tree_faulty, 0)
    plant_stage4_fault(stem_of(bad))                             # on disk: a spawned worker sees it
    outcome = run_stage4_over(staged_tree_faulty, num_workers=num_workers)
    assert set(outcome.failed) == {identity_of(bad)}
    assert len(outcome.published) == n_images(staged_tree_faulty) - 1
    assert failure_stage(staged_tree_faulty, bad) == "stage4"

def test_a_corrupt_store_fails_only_its_image(staged_tree, stage4_cpu):
    bad = corrupt_one_store(staged_tree)                         # deletes its objmap chunks
    outcome = run_stage4_over(staged_tree, num_workers=2)
    assert set(outcome.failed) == {identity_of(bad)}

def test_njobs_1_runs_the_loader_in_process(staged_tree, stage4_cpu, monkeypatch):
    seen = capture_dataloader_kwargs(monkeypatch)
    run_local_stage4_over(staged_tree, n_jobs=1)
    assert seen["num_workers"] == 0

def test_the_dataset_pickles_with_its_processor(staged_tree_dense, stage4_cpu):
    dataset = stage4_dataset_for(staged_tree_dense)
    again = pickle.loads(pickle.dumps(dataset))                  # what spawn does
    assert again.processors.keys() == dataset.processors.keys()

def test_loader_items_arrive_as_ndarrays(staged_tree, stage4_cpu):
    item = next(iter(stage4_loader_for(staged_tree, num_workers=1)))
    assert isinstance(item.prepared["dinov2-vitb14/rgb/objects"].inputs.arrays[0], np.ndarray)

def test_no_cuda_fails_the_stage_without_journalling_images(staged_tree, param_stub_backbone, monkeypatch):
    monkeypatch.delenv("PHENOTYPIC_STAGE4_DEVICE", raising=False)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(Stage4DeviceError, match="CUDA"):
        run_stage4_over(staged_tree)
    assert read_terminal_failures(staged_tree) == []
    assert all(stage4_status(staged_tree, *ds_stem_wid(s)) == "pending" for s in stores_of(staged_tree))

def test_dinov3_licence_is_refused_before_any_load(staged_tree_dinov3, stage4_cpu, monkeypatch):
    monkeypatch.delenv("PHENOTYPIC_ACCEPT_MODEL_LICENSE", raising=False)
    with pytest.raises(RuntimeError, match="licen"):
        run_stage4_over(staged_tree_dinov3)
    assert stage4_cpu.load_calls == []

def test_a_non_finite_vector_fails_the_image(staged_tree, stage4_cpu, monkeypatch):
    monkeypatch.setattr(MeasureDinoEmbedding, "vectors_from",
                        lambda self, p, o: np.full((len(p.keys), STUB_HIDDEN_SIZE), 1e6, np.float32))
    outcome = run_stage4_over(staged_tree)
    assert len(outcome.failed) == n_images(staged_tree)

def test_stage4_entries_record_stage4_compute(staged_tree, stage4_cpu):
    run_stage4_over(staged_tree)
    compute = run_entry(store_of(staged_tree))["embeddings"]["dinov2-vitb14/rgb/objects"]["compute"]
    assert compute["stage"] == "stage4" and compute["physical_batch"] >= 1

@pytest.mark.parametrize("fence", ["active_check", "commit_guard"])
def test_a_fenced_epoch_commits_nothing(staged_tree, stage4_cpu, fence):
    kwargs = fence_after_first_signal(fence)               # either seam raises SlurmGenerationInactiveError
    with pytest.raises(SlurmGenerationInactiveError):
        run_stage4_over(staged_tree, **kwargs)
    assert all(not run_entry(s).get("embeddings") for s in stores_of(staged_tree))
    assert not any(valid_record(staged_tree, s) for s in stores_of(staged_tree))

def test_orphan_signals_of_a_done_image_are_swept(staged_tree, stage4_cpu):
    run_stage4_over(staged_tree)
    plant_signals(staged_tree, image=0)                     # a crash between record and cleanup
    assert sweep_orphan_stage4_signals(staged_tree, *inventory_and_work_ids(staged_tree)) == 1
```

  Create (helpers), in `tests/_tiny_run.py`: fixture `staged_tree` (a local
  `--embed-stage gpu` run of `with_stage4_dino_objects(fake_gpu=True)` with
  `run_local_stage4` patched out, so every image is Stage-4 pending) and its
  variants `staged_tree_two_backbones`, `staged_tree_objects_and_grid`,
  `staged_tree_one_empty_plate`, `staged_tree_dinov3`, `staged_tree_faulty`
  (`faulty=True`), `staged_tree_dense` (a `masked_dense` embedder); `stem_of(store)`;
  `run_local_stage4_over(tree, *, n_jobs)`, `capture_dataloader_kwargs`,
  `stage4_dataset_for`; `run_stage4_over(tree, **kw) -> Stage4Outcome`
  (items from `staged_manifest_entry`, then `run_stage4`); `stage4_loader_for`;
  `spy_calls_by_image(monkeypatch, names) -> dict[str, list[str]]`, keyed by
  image identity, so the one-thread pool's order is compared per image (plan
  review I5); `write_valid_signals`, `plant_signals`, `touch_root_before_commit`,
  `write_valid_signals(..., binds_from_another_objmap=False)`,
  `fence_after_first_signal(seam) -> dict`
  (an `active_check` or a `commit_guard` that raises after the first signal),
  `stores_of`, `store_of`, `identity_of`, `corrupt_one_store`, `failure_stage`,
  `ds_stem_wid`, `embedding_path`, `valid_record`, `inventory_and_work_ids`.
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
  git add src/phenotypic/_cli/_cli_stage4_worker.py src/phenotypic/_cli/_cli_staged_resume.py src/phenotypic/_cli/_cli_staged_strategy.py src/phenotypic/_cli/_cli_execution_strategies.py tests/_tiny_run.py tests/conftest.py tests/unit/cli/test_stage4_worker.py tests/integration/cli/test_embed_stage_refusals.py
  git commit -m "feat(cli): the Stage-4 worker: backbone-major, per-key signals at the live root, commit then record (#282 P4)"
  ```

### Task 4: Resume, retry, the no-detector continuation, and crash injection

**Files:**
- Modify: `src/phenotypic/_cli/_cli_staged_resume.py`
  - `ResumeStage` (`:36`) gains `"stage4"`; `_STAGE_ORDER` (`:37-42`) puts it
    between `"stage3"` and `"complete"`; `StagedResumePlan.counts` (`:63-72`);
  - `classify_staged_image` (`:226`) gains `defers_embeddings: bool = False`: its
    "Stage-3 marker plus table" branch (`:276-283`) returns `"stage4"` when set;
  - `build_staged_resume_plan` (`:322`) passes it through;
  - `clear_downstream_artifacts_for_stage1` (`:424`) also calls
    `delete_stage4_signals`.
- Modify: `src/phenotypic/phenotypicCLI.py`
  - the staged continuation (`:3792-3900`) passes `defers_embeddings(config)`;
    an all-`stage4` plan sets `config.staged_resume_phase = "stage4"`;
  - **a no-detector continuation** (plan review C1): that block runs only under
    `staged_gpu_resume` (`:3792`). For a run with `defers_embeddings(config)` and
    no `GpuDetector`, add a sibling branch that calls `split_stage4_pending`,
    so images that finished Stages 1–3 skip the single-pass core;
  - under deferral, `sweep_orphan_stage4_signals` runs **before** the
    "✓ All images already processed!" exit (`:3889-3897`), which a finished run
    takes before Stage 4 is ever reached (re-review, Minor).
- Modify: `src/phenotypic/_cli/_cli_stage4_worker.py`: add
  `split_stage4_pending(datasets: Sequence[Dataset], output_dir: Path, config: ExecutionConfig) -> tuple[list[Dataset], list[StagedManifestEntry]]`:
  the datasets still needing the core, and the Stage-4 items.
- Test: `tests/integration/cli/test_local_stage4.py` (new) **[torch]**;
  `tests/unit/cli/test_staged_resume.py` (extend, shared env)

**Interfaces:**
- Consumes: Tasks 2 and 3.
- Produces: `ResumeStage` with `"stage4"`,
  `classify_staged_image(..., defers_embeddings=...)`, `split_stage4_pending`.
  P5 T2 reuses `split_stage4_pending` for the SLURM no-detector continuation.

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
def test_a_real_crash_resumes_from_disk(tmp_path, tiny_inputs, stage4_cpu, crash_after, resumed_by):
    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path, fake_gpu=True)
    first = run_cli_subprocess(tiny_inputs, out, pipeline=pipe, extra=["--embed-stage", "gpu"],
                               crash_after=crash_after)
    assert first.returncode == 137                                       # the process really died
    crashed = crashed_image(out, crash_after)                            # read from disk, not the test
    calls = run_cli_subprocess_spying(tiny_inputs, out, pipeline=pipe, extra=["--embed-stage", "gpu"],
                                      stem=crashed)
    assert calls.returncode == 0 and calls.sequence == resumed_by       # never re-embedded
    assert no_stage4_signals(out) and all_records_valid(out)

def test_a_finished_run_rerun_embeds_nothing(tmp_path, tiny_inputs, stage4_cpu):
    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path, fake_gpu=True)
    run_cli(tiny_inputs, out, pipeline=pipe, extra=["--embed-stage", "gpu"])
    roots, loads = store_root_bytes(out), len(stage4_cpu.load_calls)
    assert run_cli(tiny_inputs, out, pipeline=pipe, extra=["--embed-stage", "gpu"]).exit_code == 0
    assert store_root_bytes(out) == roots and len(stage4_cpu.load_calls) == loads

def test_a_finished_rerun_sweeps_orphans_before_it_exits(tmp_path, tiny_inputs, stage4_cpu):
    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path, fake_gpu=True)
    run_cli(tiny_inputs, out, pipeline=pipe, extra=["--embed-stage", "gpu"])
    plant_signals(out, image=0)                                 # a crash between record and cleanup
    result = run_cli(tiny_inputs, out, pipeline=pipe, extra=["--embed-stage", "gpu"])
    assert "All images already processed" in result.output and no_stage4_signals(out)

def test_a_no_detector_continuation_skips_the_core(tmp_path, tiny_inputs, stage4_cpu, monkeypatch):
    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path, fake_gpu=False)
    with stage4_stubbed_out(monkeypatch):
        run_cli(tiny_inputs, out, pipeline=pipe, extra=["--embed-stage", "gpu"])
    calls = count_calls(monkeypatch, ["process_single_image_core"])
    assert run_cli(tiny_inputs, out, pipeline=pipe, extra=["--embed-stage", "gpu"]).exit_code == 0
    assert calls == {"process_single_image_core": 0} and all_records_valid(out)

def test_a_stage4_failure_exits_3_and_is_not_staged_complete(tmp_path, tiny_inputs, stage4_cpu, stage4_faults):
    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path, fake_gpu=True, faulty=True)
    plant_stage4_fault(nth_stem(tiny_inputs, 0))
    result = run_cli(tiny_inputs, out, pipeline=pipe, extra=["--embed-stage", "gpu"])
    assert result.exit_code == 3                                          # plan review C2, not 1
    assert not staged_completion_path(out).is_file()

def test_a_stage1_rerun_deletes_the_images_signals(tmp_path, stage4_pending_with_signals, monkeypatch):
    out, item = stage4_pending_with_signals
    invalidate_stage1_store(out, item)
    with stage4_stubbed_out(monkeypatch):                  # look at the disk before Stage 4 runs
        run_cli(*stage4_pending_with_signals.cli_args, extra=["--embed-stage", "gpu"])
    assert not any(signal_files(out, item))
    assert any(signal_files(out, other_item(stage4_pending_with_signals)))   # only that image's

def test_retry_failures_reruns_only_stage4(tmp_path, tiny_inputs, stage4_cpu, stage4_faults):
    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path, fake_gpu=True, faulty=True)
    plant_stage4_fault(nth_stem(tiny_inputs, 0))
    assert run_cli(tiny_inputs, out, pipeline=pipe, extra=["--embed-stage", "gpu"]).exit_code == 3
    clear_stage4_fault(nth_stem(tiny_inputs, 0))
    assert run_cli(tiny_inputs, out, pipeline=pipe, extra=["--embed-stage", "gpu"]).exit_code == 3   # skipped
    events = tail_events(out)
    assert run_cli(tiny_inputs, out, pipeline=pipe,
                   extra=["--embed-stage", "gpu", "--retry-failures"]).exit_code == 0
    assert stages_of(new_events(out, after=events), image=0) == ["stage4"]

def test_restart_clears_stage4_signals(tmp_path, stage4_pending_with_signals, monkeypatch):
    out, _ = stage4_pending_with_signals
    with stage4_stubbed_out(monkeypatch), spy_restart_clear(monkeypatch) as cleared:
        run_cli(*stage4_pending_with_signals.cli_args, extra=["--embed-stage", "gpu", "--restart"])
    assert cleared.saw_no_stage4_payload                   # gone right after the --restart clear
```

  `stage4_stubbed_out(monkeypatch)` patches `run_local_stage4` to return `{}`;
  `spy_restart_clear` wraps the `--restart` clear and records whether
  `progress/stage4_payload` exists right after it; `other_item` names a second
  pending image of the fixture. `run_cli_subprocess_spying(...)` is
  `run_cli_subprocess` with a preload that
  records, for one stem, the order of the four Stage-4 functions into a JSON
  file the test reads back. `crashed_image(out, name)` names the one image whose
  state on disk shows the crash (signals without a commit; a commit without a
  record; a record with signals). In `test_staged_resume.py`:

```python
def test_stage3_complete_without_a_record_classifies_stage4(staged_store_stage3_only):
    out, image, work_id = staged_store_stage3_only
    assert classify_staged_image(**classify_kwargs(out, image, work_id), defers_embeddings=True) == "stage4"
    assert classify_staged_image(**classify_kwargs(out, image, work_id)) == "stage3"     # inline: unchanged
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
  git commit -m "feat(cli): Stage-4 resume is content-defined, with and without a detector (#282 P4)"
  ```

**Deep review P3–P4** (implementation-test-reviewer, Opus) after the gate:
`reports/2026-11-12-gpu-embedding-staging/review-p3-p4.md`. It checks in
particular that every site in Task 2's Files list reads one predicate
(`stages_1_to_3_done` under `defers_embeddings`), that no inline run's bytes
moved (P0 T2's golden, and `tests/integration/cli/test_embedding_deliverables.py`),
and that each crash test dies in a real subprocess.
