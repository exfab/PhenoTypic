# P5 · SLURM: controller v2, rounds, worklist, the no-detector plan, preflight

**Spec:** U2, U5, U6, H6, §4.1 (SLURM rows), §4.3, §4.7 (the two warnings),
§4.8 (walltime, fence, cold load), §5 "Controller unit tests", "Live SLURM smoke"

---

**Goal:** A SLURM run with Stage 4 on goes through the recoverable controller:
after the last Stage-3 chunk the controller classifies Stage 4, submits GPU
rounds over a per-round worklist until nothing is retryable or two rounds make
no progress, then submits the finalizer. A run with no `GpuDetector` and
`--gpu-slurm` gets the same controller with phases `["stage3", "stage4"]`.

**Depends on:** P4.

### Task 1: Controller v2, Stage-4 rounds and the worklist

**Files:**
- Modify: `src/phenotypic/_cli/_cli_staged_controller.py`
  - `_load_controller_config` (`:52-56`) accepts version 1 and 2; a v1 config
    reads as `phases = ["stage1", "stage2", "stage3"]`;
  - add `_classify_stage4`, `_submit_stage4_round`, `_journal_stage4_no_progress`;
  - `_submit_next_stage3_or_finalizer` (`:204-253`): when the Stage-3 chunks are
    exhausted and `"stage4"` is in `phases`, classify Stage 4 instead of
    submitting the finalizer;
  - `run_staged_controller` (`:292-392`): phase `"stage4"` dispatches to
    `_classify_stage4` → `_submit_stage4_round` or the finalizer; a config
    without `"stage1"` starts at `"stage3"`.
- Modify: `src/phenotypic/_cli/_cli_staged_orchestration.py`
  - `initialize_orchestration` (`:138-169`) state gains `stage4_round: 0`,
    `stage4_last_retryable_digest: None`, `stage4_zero_progress_rounds: 0`;
  - add `stage4_worklist_path(output_dir) -> Path`
    (`.phenotypic/slurm_scripts/stage4_worklist.json`),
    `write_stage4_worklist(path: Path, *, round_index: int, retryable: Sequence[StagedManifestEntry], n_shards: int) -> Path`,
    `load_stage4_worklist(path: Path, shard_index: int) -> tuple[int, list[StagedManifestEntry]]`
    (the round and that shard's entries).
- Test: `tests/unit/cli/test_staged_controller_stage4.py` (new, shared env).
  Its fake scheduler is the existing pattern of
  `tests/unit/cli/test_staged_controller.py:108-129`, monkeypatching
  `submit_with_intent`, `scheduler_job_is_active` and `update_job_dependency` in
  `phenotypic._cli._cli_staged_controller`, packaged as a fixture.

**Interfaces:**
- Consumes: `stage4_status` (P4 T3); `retryable_digest` (`_cli_staged_orchestration.py:276`);
  `submit_with_intent` (`:511`); `append_terminal_failure` (`_cli_failure_tracker.py:562`);
  `read_terminal_failures`; `reference_digest_for` (`_cli_reference.py`).
- Produces:
  - Controller config v2: v1's keys plus `"phases": list[str]`,
    `"stage4_script": str`, `"stage4_worklist": str`, `"n_shards": int`,
    `"stage3_role": "staged" | "single_pass" | "measure"`,
    `"pipeline_sha256": str | None`, `"republish": bool`. `"detector_slot"` and
    the Stage-1/2 scripts are present only when `"stage1"` is in `phases`.
  - `_classify_stage4(config: dict[str, Any], entries: Sequence[StagedManifestEntry]) -> tuple[list[StagedManifestEntry], list[StagedManifestEntry]]`
    (retryable, terminal), spec §4.3: done when `stage4_status == "done"`;
    terminal on a recorded failure under this epoch, `not_ready` (no
    `stages.stage3`, or a store that does not match the work id), or a changed
    reference digest (`_classify_stage2`'s rule, `:107-116`); retryable
    otherwise.
  - `_submit_stage4_round(config, state, successor_id, current_job_id, retryable) -> None`:
    Stage 2's zero-progress rule on its **own** counters (`stage4_round`,
    `stage4_last_retryable_digest`, `stage4_zero_progress_rounds`). Below two
    unchanged rounds it writes the worklist, then submits
    `submit_with_intent(..., token=f"stage4-round-{n}", role="stage4", round_index=n, script_path=config["stage4_script"])`
    and retargets the successor. At two, it calls
    `_journal_stage4_no_progress` and submits the finalizer.
  - `_journal_stage4_no_progress(config, entries) -> int`: one terminal failure
    per entry, `failed_stage="stage4"`, error type `Stage4NoProgress`, message
    `"no Stage-4 progress in two GPU rounds"`, under the epoch **(pending O8)**.
  - The worklist: `{"version": 1, "round": n, "shards": [[entry, …], …]}`, the
    retryable set sorted by `identity` and split by `partition_shards`
    (`_cli_staged_slurm.py:64`), the rule P0 T1's `worklist_partition.py`
    re-derives.

- [ ] **Step 1: Write the failing tests.**

```python
def test_after_the_last_stage3_chunk_stage4_is_classified(controller_v2, fake_scheduler):
    finish_stage3(controller_v2, stage4_pending=3)
    run_staged_controller(controller_v2.config_path)
    assert fake_scheduler.submitted_roles()[-1] == ("stage4", "stage4-round-1")
    assert load_stage4_worklist(controller_v2.worklist, 0)[0] == 1

def test_all_done_goes_straight_to_the_finalizer(controller_v2, fake_scheduler):
    finish_stage3(controller_v2, stage4_pending=0)
    run_staged_controller(controller_v2.config_path)
    assert [r for r, _ in fake_scheduler.submitted_roles()] [-1] == "finalizer"
    assert "stage4" not in [r for r, _ in fake_scheduler.submitted_roles()]

def test_rounds_have_their_own_counters(controller_v2, fake_scheduler):
    state = load_orchestration_state(controller_v2.out)
    state.update(zero_progress_rounds=1, last_retryable_digest="x")     # Stage 2's
    save_orchestration_state(controller_v2.out, state)
    finish_stage3(controller_v2, stage4_pending=2)
    run_staged_controller(controller_v2.config_path)
    state = load_orchestration_state(controller_v2.out)
    assert state["stage4_zero_progress_rounds"] == 0 and state["zero_progress_rounds"] == 1

def test_two_unchanged_rounds_journal_and_finalize(controller_v2, fake_scheduler):
    finish_stage3(controller_v2, stage4_pending=2)
    for _ in range(3):
        run_round_without_progress(controller_v2, fake_scheduler)
    failures = [r for r in read_terminal_failures(controller_v2.out) if r.failed_stage == "stage4"]
    assert len(failures) == 2
    assert [r for r, _ in fake_scheduler.submitted_roles()][-1] == "finalizer"

def test_the_worklist_partitions_only_the_retryable_set(controller_v2, fake_scheduler):
    finish_stage3(controller_v2, stage4_pending=5, done=3, n_shards=2)
    run_staged_controller(controller_v2.config_path)
    shards = [load_stage4_worklist(controller_v2.worklist, i)[1] for i in (0, 1)]
    assert sorted(len(s) for s in shards) == [2, 3]
    assert not ({e.identity for s in shards for e in s} & controller_v2.done_identities)

def test_more_shards_than_work_leaves_empty_shards_harmless(controller_v2, fake_scheduler):
    finish_stage3(controller_v2, stage4_pending=1, n_shards=4)
    run_staged_controller(controller_v2.config_path)
    assert [len(load_stage4_worklist(controller_v2.worklist, i)[1]) for i in range(4)] == [1, 0, 0, 0]

def test_stage4_rounds_are_ledgered_as_stage4(controller_v2, fake_scheduler):
    finish_stage3(controller_v2, stage4_pending=1)
    run_staged_controller(controller_v2.config_path)
    assert ("stage4", "stage4-round-1") in fake_scheduler.submitted_roles()

def test_a_stage4_ledger_row_is_live(tmp_path, monkeypatch):
    append_job_ledger(tmp_path, epoch="e", token="stage4-round-1", role="stage4", job_id="77", status="submitted")
    monkeypatch.setattr(orchestration_module, "scheduler_job_is_active", lambda job_id: job_id == "77")
    assert "77" in live_slurm_job_ids(tmp_path).live

def test_no_sidecar_job(controller_v2, fake_scheduler):
    finish_stage3(controller_v2, stage4_pending=2)
    run_staged_controller(controller_v2.config_path)
    active = fake_scheduler.active_roles()
    assert sorted(active) == ["controller", "stage4"]        # one work array plus its recovery controller

def test_classify_stage4_terminal_cases(controller_v2):
    retryable, terminal = _classify_stage4(controller_v2.config, controller_v2.entries_with(
        failed=1, no_stage3=1, wrong_work_id=1, reference_changed=1, pending=2))
    assert len(terminal) == 4 and len(retryable) == 2

def test_a_v1_config_still_runs_three_phases(v1_controller_config, fake_scheduler):
    finish_stage3(v1_controller_config, stage4_pending=0)
    run_staged_controller(v1_controller_config.config_path)
    assert [r for r, _ in fake_scheduler.submitted_roles()][-1] == "finalizer"

def test_no_detector_phases_start_at_stage3(controller_v2_no_detector, fake_scheduler):
    run_staged_controller(controller_v2_no_detector.config_path)
    assert fake_scheduler.submitted_roles()[-1][0] == "stage3"
```

  Create (helpers), in `tests/unit/cli/conftest.py`: fixtures `controller_v2`,
  `controller_v2_no_detector`, `v1_controller_config` (built on
  `_controller_fixture`, writing a v2 config and the Stage-4 script stubs), and
  `finish_stage3(fixture, *, stage4_pending, done=0, n_shards=1)`, which writes
  stores, records and markers so each entry classifies as asked.
  `run_round_without_progress` marks the active Stage-4 job terminal and runs
  one controller transition. Fixture `fake_scheduler` records
  `(role, token)` per `submit_with_intent` call (`submitted_roles()`) and which
  submitted jobs it reports active (`active_roles()`, `set_active(job_id)`).
  Match `append_job_ledger`'s real keyword names (`_cli_staged_orchestration.py:383`)
  in the ledger test; the call above shows intent, not its exact signature.
- [ ] **Step 2: Run them and confirm they fail.**

  ```bash
  eval $PT tests/unit/cli/test_staged_controller_stage4.py
  ```

- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them and the existing controller tests unchanged.**

  ```bash
  eval $PT tests/unit/cli/test_staged_controller_stage4.py tests/unit/cli/test_staged_controller.py tests/unit/cli/test_staged_manifest_version.py
  ```

- [ ] **Step 5: Commit.**

  ```bash
  git add src/phenotypic/_cli/_cli_staged_controller.py src/phenotypic/_cli/_cli_staged_orchestration.py tests/unit/cli/conftest.py tests/unit/cli/test_staged_controller_stage4.py
  git commit -m "feat(slurm): controller v2, Stage-4 rounds on their own counters, the per-round worklist (#282 P5)"
  ```

### Task 2: Scripts, the worker's `--stage 4`, the no-detector plan, and the revision pin

**Files:**
- Modify: `src/phenotypic/_cli/_cli_staged_slurm.py`
  - `_write_stage_script` (`:219-252`) gains `prelude: str = SLURM_THREAD_PIN_BASH`;
  - `_stage_worker_body` (`:137-216`): stage 4 emits `--worklist <path>` and,
    when set, `--republish`; stage 3 emits `--stage3-role <role>` when the role
    is not `"staged"`;
  - `generate_staged_scripts` (`:307-471`) gains
    `phases: Sequence[str]`, `stage3_role: Literal["staged", "single_pass", "measure"] = "staged"`,
    `pipeline_sha256: str | None`, `republish: bool = False`; `detector_slot`
    becomes `str | None`. It writes `stage4.sh`, an array of `n_shards` tasks on
    `resolve_stage_slurm_args(gpu_slurm_args, cpu_slurm_args)` with prelude
    `SLURM_THREAD_PIN_BASH + "export CUBLAS_WORKSPACE_CONFIG=:4096:8\n"`, never
    chunked; Stage-1/2 scripts only when `"stage1"` is in `phases`; the v2
    controller config (Task 1);
  - `_ordered_staged_scripts` (`:474`) includes `stage4`;
  - `StagedSlurmStrategy.execute` (`:601-740`): chooses `phases` and
    `stage3_role` from the config, calls `staged_detector_slot` only when the
    pipeline has a `GpuDetector`, resolves revisions (U5) and records them.
- Modify: `src/phenotypic/_cli/_cli_staged_slurm_worker.py`
  - `main` (`:645-746`): `--stage` choices `(1, 2, 3, 4)`, `--stage3-role`,
    `--worklist`, `--republish`;
  - add `run_stage4_shard` and `run_single_pass_step` (Interfaces).
- Modify: `src/phenotypic/_cli/_cli_execution_strategies.py:1351-1444`: add
  `uses_staged_controller(config) -> bool`; `create_execution_strategy`
  (`:1405`) routes a SLURM run with `config.embed_stage == "gpu"` to
  `StagedSlurmStrategy`.
- Modify: `src/phenotypic/_cli/_cli_embed_stage.py`: add `resolve_embed_revisions`.
- Test: `tests/unit/cli/test_staged_slurm_scripts_stage4.py` (new, shared env);
  `tests/unit/cli/test_stage4_slurm_worker.py` (new) **[torch]**

**Interfaces:**
- Consumes: `run_stage4` (P4 T3), `process_single_image_core(..., defer_embeddings=True)`
  (P4 T2), `write_stage3_completion_marker`, `_recorded_image_read`
  (`_cli_staged_slurm_worker.py:182`, reads `NROWS`/`NCOLS`/`BIT_DEPTH`/`DETECT_MODE`
  from job metadata as Stage 1 does), `load_stage4_worklist` (Task 1).
- Produces:
  - `uses_staged_controller(config: ExecutionConfig) -> bool`:
    `uses_staged_gpu_strategy(config) or (config.is_slurm_mode() and config.embed_stage == "gpu")`
  - `run_stage4_shard(pipeline_path: Path, output_dir: Path, image_type: ImageTypeName, worklist_path: Path, shard_index: int, *, epoch: str, durable_writes: bool | None = None, republish: bool = False) -> int`:
    exits `0` on an empty slice; reads `EMBED_REVISIONS` and the run's pipeline
    sha from `job_metadata.json`; calls `run_stage4` with the epoch's
    `_active_check` and `_commit_guard` (`:143-166`) and
    `num_workers = int(os.environ["SLURM_CPUS_PER_TASK"]) - 1`; returns non-zero
    on `Stage4DeviceError`, `Stage4OutOfMemory` or `MemoryError`, journalling no
    image.
  - `run_single_pass_step(pipeline_path, output_dir, image_type, manifest, index, ext, *, epoch, resume, overlay_alpha, durable_writes, drop_originals, pipeline_identity) -> None`:
    the "stage3" role of a no-detector Stage-4 run. It runs
    `process_single_image_core(..., work_id=item.work_id, defer_embeddings=True)`
    inside `stage_event(..., STAGE_MEASURE)`, and records a scientific failure
    as `run_stage3_step` does (`:631-642`). It publishes no record.
  - `resolve_embed_revisions(pipeline: ImagePipeline) -> dict[str, str | None]`:
    `{hf_id: commit}` for every `MeasureDinoEmbedding` with `revision=None`,
    from `huggingface_hub.HfApi().model_info(hf_id).sha`, called once in the
    submitting process. With `HF_HUB_OFFLINE=1` or any hub error, the value is
    `None` and one warning line names the model and says each shard resolves on
    its own, the existing `PF-EMBED-REVISION-UNPINNED` case (spec U5)
    **(pending O9: Stage-4 runs only)**.

- [ ] **Step 1: Write the failing tests.**

```python
def test_stage4_script_is_one_unchunked_gpu_array(tmp_path, staged_script_kwargs):
    scripts = generate_staged_scripts(**staged_script_kwargs(tmp_path, phases=STAGED_V2, n_shards=3,
                                      gpu_slurm_args={"slurm_partition": "exfab", "slurm_account": "exfab"}))
    text = scripts["stage4"].read_text(encoding="utf-8")
    assert "#SBATCH --array=0-2" in text and "--gpus-per-node=1" in text
    assert "export CUBLAS_WORKSPACE_CONFIG=:4096:8" in text
    assert "--stage 4" in text and "--worklist" in text

def test_the_no_detector_plan_has_no_stage1_or_stage2(tmp_path, staged_script_kwargs):
    scripts = generate_staged_scripts(**staged_script_kwargs(tmp_path, phases=["stage3", "stage4"],
                                      stage3_role="single_pass", detector_slot=None))
    config = json.loads(scripts["controller_config"].read_text(encoding="utf-8"))
    assert config["version"] == 2 and config["phases"] == ["stage3", "stage4"]
    assert "stage1_scripts" not in config and "detector_slot" not in config
    assert "--stage3-role single_pass" in scripts["stage3"][0].read_text(encoding="utf-8")

@pytest.mark.parametrize("embed_stage, strategy", [("inline", AutonomousSLURMStrategy), ("gpu", StagedSlurmStrategy)])
def test_slurm_dispatch_follows_the_resolved_embed_stage(make_exec_config, tmp_path, embed_stage, strategy):
    config = dataclasses.replace(
        make_exec_config(pipeline_json=dino_pipeline_path(tmp_path), input_path=tmp_path,
                         slurm_args={"slurm_partition": "intel"}, force_local=False),
        embed_stage=embed_stage)
    assert isinstance(create_execution_strategy(config, output_manager_for(config)), strategy)

def test_revisions_are_resolved_once_and_recorded(tmp_path, monkeypatch, submissions):
    calls = []
    monkeypatch.setattr(embed_stage_module, "_hub_commit", lambda hf_id: calls.append(hf_id) or "c" * 40)
    submit_staged_no_detector(tmp_path, n_shards=3)
    assert calls == ["facebook/dinov2-base"]
    assert read_job_metadata(tmp_path / "out")["embed_revisions"] == {"facebook/dinov2-base": "c" * 40}

def test_offline_revision_falls_back_with_one_warning(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert resolve_embed_revisions(pipeline_with_dino()) == {"facebook/dinov2-base": None}
    assert capsys.readouterr().err.count("resolves on its own") == 1
```

  `_hub_commit(hf_id) -> str` is the one private seam around
  `HfApi().model_info`. In `test_stage4_slurm_worker.py` **[torch]**:

```python
def test_an_empty_shard_exits_zero(tmp_path, stage4_slurm_tree, stage4_cpu):
    assert run_stage4_shard(**stage4_slurm_tree.shard_kwargs(shard_index=3)) == 0

def test_every_shard_loads_the_recorded_revision(tmp_path, stage4_slurm_tree, stage4_cpu):
    for i in range(2):
        run_stage4_shard(**stage4_slurm_tree.shard_kwargs(shard_index=i))
    assert {c["revision"] for c in stage4_cpu.load_calls} == {"c" * 40}

def test_the_single_pass_role_defers_and_marks_stage3(tmp_path, stage4_slurm_tree, stage4_cpu):
    run_single_pass_step(**stage4_slurm_tree.single_pass_kwargs(index=0))
    record = record_of(stage4_slurm_tree.out, 0)
    assert set(record["stages"]) == {"stage3"} and not record.get("artifacts")

def test_the_params_digest_is_the_inline_one(tmp_path, stage4_slurm_tree, stage4_cpu):
    run_stage4_shard(**stage4_slurm_tree.shard_kwargs(shard_index=0))
    entry = first_entry(stage4_slurm_tree.out)
    assert entry["producer"]["params_sha256"] == MeasureDinoEmbedding()._params_sha256()
```

  Create (helpers), in `tests/_tiny_run.py`: `staged_script_kwargs(tmp_path, **overrides) -> dict`,
  `STAGED_V2 = ["stage1", "stage2", "stage3", "stage4"]`, fixture
  `stage4_slurm_tree` (stores Stage-3 complete under an epoch, a worklist with
  2 shards of work and 2 empty, `EMBED_REVISIONS = {"facebook/dinov2-base": "c" * 40}`
  in job metadata; methods `shard_kwargs(shard_index)` and
  `single_pass_kwargs(index)`), `submit_staged_no_detector`, `read_job_metadata`,
  `record_of`, `first_entry`, `pipeline_with_dino`, `dino_pipeline_path(tmp_path)` (writes it), `output_manager_for`.
  `submissions` is the SLURM stub fixture (`tests/unit/cli/conftest.py`).
- [ ] **Step 2: Run them and confirm they fail.**

  ```bash
  eval $PT tests/unit/cli/test_staged_slurm_scripts_stage4.py
  eval $TPT tests/unit/cli/test_stage4_slurm_worker.py
  ```

- [ ] **Step 3: Implement.** Mixed GPU models across shards are allowed (U6):
  no script pins a GPU type; `compute` records what ran.
- [ ] **Step 4: Run the tests plus the staged-script, routing and preflight-order suites.**

  ```bash
  eval $PT tests/unit/cli/test_staged_slurm_scripts_stage4.py tests/unit/cli/test_staged_slurm_scripts.py tests/unit/cli/test_staged_routing.py tests/unit/cli/test_cli_preflight_ordering.py
  eval $TPT tests/unit/cli/test_stage4_slurm_worker.py
  ```

- [ ] **Step 5: Commit.**

  ```bash
  git add src/phenotypic/_cli/_cli_staged_slurm.py src/phenotypic/_cli/_cli_staged_slurm_worker.py src/phenotypic/_cli/_cli_execution_strategies.py src/phenotypic/_cli/_cli_embed_stage.py tests/_tiny_run.py tests/conftest.py tests/unit/cli/test_staged_slurm_scripts_stage4.py tests/unit/cli/test_stage4_slurm_worker.py
  git commit -m "feat(slurm): the Stage-4 array, the no-detector plan, and one revision per run (#282 P5)"
  ```

### Task 3: Preflight codes

**Files:**
- Modify: `src/phenotypic/_cli/_cli_preflight.py`
  - `FindingCode` (`:48-89`) gains `"PF-EMBED-SHARD-AMORTIZE"` and `"PF-EMBED-RESOURCES"`;
  - `HINTS` (`:91`) gains one remedy each (below);
  - add `_stage4_slurm_run(context) -> bool` (a SLURM run whose config resolves
    `embed_stage == "gpu"`), and make `_staged_slurm_run` (`:731-740`) also
    true for it, so `PF-SLURM-LIMIT` (`:1016`) and `PF-GPU-PARTITION`
    (`:1035`) check the Stage-4 GPU profile of a no-detector run;
  - add `check_embed_shard_amortization` and `check_embed_loader_memory`, both
    warnings, registered in `CHECKS` (`:1690-1715`) after
    `check_staged_slurm_limits`.
- Modify: `docs/source/tutorials/pages/cli_batch_processing.md`, the preflight
  code table (`:108` onward): one row per new code.
- Test: `tests/unit/cli/test_preflight_embed_stage.py` (new, shared env);
  `tests/unit/test_docs_preflight_codes.py` (unchanged; must pass)

**Interfaces:**
- Consumes: P0 T1's `AMORTIZE_MIN_IMAGES_PER_SHARD`, `LOADER_BASE_BYTES`,
  `LOADER_PER_WORKER`, copied into `_cli_preflight.py` as private constants
  whose docstring names `shard_amortization.py`; the input-header shapes the
  existing header checks read (`check_input_headers_readable`).
- Produces:
  - `check_embed_shard_amortization(context) -> list[PreflightFinding]`: when
    `n_images / max(1, gpu_shards) < AMORTIZE_MIN_IMAGES_PER_SHARD(size)` for
    the largest in-scope backbone. Message:
    `"<n> images over <k> GPU shards is about <n/k> per shard, below the <t> that pays off a cold model load of about <s> s; use fewer --gpu-shards, or --embed-stage inline"`.
    HINT: `"Use fewer --gpu-shards, pre-stage the weights and set HF_HUB_OFFLINE=1, or pass --embed-stage inline"`.
  - `check_embed_loader_memory(context) -> list[PreflightFinding]`: when the
    Stage-4 profile's `--mem` is below
    `LOADER_BASE_BYTES + (cpus_per_task - 1) · LOADER_PER_WORKER(H, W)` for the
    largest header. Message names the requested and the needed GB. HINT:
    `"Raise mem_gb in --gpu-slurm, or lower slurm_cpus_per_task there (fewer loader workers)"`.
  - Neither imports torch; neither reads a pixel.

- [ ] **Step 1: Write the failing tests.**

```python
def test_too_few_images_per_shard_warns(tmp_path):
    findings = preflight_for(tmp_path, n_images=40, gpu_shards=4, embed_stage="gpu")
    (f,) = [f for f in findings if f.code == "PF-EMBED-SHARD-AMORTIZE"]
    assert f.severity == "warning" and "10 per shard" in f.message

def test_enough_images_per_shard_is_silent(tmp_path):
    assert not codes(preflight_for(tmp_path, n_images=800, gpu_shards=2, embed_stage="gpu")) & {"PF-EMBED-SHARD-AMORTIZE"}

def test_low_mem_for_the_loader_warns(tmp_path):
    findings = preflight_for(tmp_path, n_images=800, gpu_slurm={"mem_gb": 8, "slurm_cpus_per_task": 8},
                             header_hw=(4012, 6016), embed_stage="gpu")
    assert "PF-EMBED-RESOURCES" in codes(findings)

def test_inline_runs_get_neither(tmp_path):
    assert not codes(preflight_for(tmp_path, n_images=4, gpu_shards=4, embed_stage="inline")) & {
        "PF-EMBED-SHARD-AMORTIZE", "PF-EMBED-RESOURCES"}

def test_a_no_detector_gpu_embed_run_checks_the_gpu_partition(tmp_path, monkeypatch):
    findings = preflight_for(tmp_path, n_images=800, gpu_slurm={"slurm_partition": "intel"},
                             embed_stage="gpu", sinfo_gres={"intel": ""})
    assert "PF-GPU-PARTITION" in codes(findings)

def test_dinov3_7b_is_refused_at_preflight_without_the_licence(tmp_path, monkeypatch):
    monkeypatch.delenv("PHENOTYPIC_ACCEPT_MODEL_LICENSE", raising=False)
    findings = preflight_for(tmp_path, n_images=800, embed_stage="gpu",
                             embedder=MeasureDinoEmbedding(version=3, size="7b", compute_dtype="bfloat16"))
    assert "PF-LICENSE" in codes(findings)                   # the existing check, now reaching 7b (spec §4.5)

def test_the_checks_import_no_torch(tmp_path):
    assert "torch" not in modules_loaded_by(lambda: preflight_for(tmp_path, n_images=40, embed_stage="gpu"))
```

  `preflight_for`, `codes`, `modules_loaded_by` are module-local, built on
  `tests/_tiny_run.py`'s `run_preflight` (`:150`) and a stubbed `sinfo`.
- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement**, and add the two docs rows in the table's own format.
- [ ] **Step 4: Run them, the docs-code drift test and the existing preflight suite.**

  ```bash
  eval $PT tests/unit/cli/test_preflight_embed_stage.py tests/unit/test_docs_preflight_codes.py tests/unit/cli/test_cli_preflight_core.py
  ```

- [ ] **Step 5: Commit.**

  ```bash
  git add src/phenotypic/_cli/_cli_preflight.py docs/source/tutorials/pages/cli_batch_processing.md tests/unit/cli/test_preflight_embed_stage.py
  git commit -m "feat(preflight): PF-EMBED-SHARD-AMORTIZE and PF-EMBED-RESOURCES (#282 P5)"
  ```

**P5 phase gate** before Task 4: importers of `_cli_staged_controller`,
`_cli_staged_orchestration`, `_cli_staged_slurm`, `_cli_staged_slurm_worker`,
`_cli_execution_strategies`, `_cli_preflight`, plus `tests/integration/cli` and
the GUI trees once, both lanes, as a Slurm array from a detached worktree.
Expected reds: the 13 expected-red GUI tests only.

### Task 4: Live SLURM smoke (stop point 2: the user approves first)

**Files:**
- Create: `docs/superpowers/plans/2026-11-12-gpu-embedding-staging/live_smoke.sh`
- Report: `docs/superpowers/reports/2026-11-12-gpu-embedding-staging/live-smoke.md`

The orchestrator runs this, not an implementer, and only after the user says go.
It spends real GPU time. The user also names the input set (below).

- [ ] **Step 1: Write `live_smoke.sh <sha> <inputs>`.** It adds a detached
  worktree at `<sha>`, writes two pipelines (`OtsuDetector` +
  `MeasureDinoEmbedding(size="small", revision=<pinned>)` objects and a 2 × 2
  grid; and the same with `FakeGpuDetector` under `PHENOTYPIC_PRELOAD_MODULES`),
  and launches each through `phenotypic-wait --output <out> -- python -m phenotypic …`
  so the waiter's exit code is the result. Both runs use
  `--gpu-shards 2`, `--slurm` on `intel`, and a `--gpu-slurm` profile on the
  GPU partition with the earlier predicted start (README conventions; `exfab`
  needs `slurm_account=exfab`; `short_gpu` needs `slurm_constraint=gpu_latest`
  and `slurm_qos=short_gpu`).
- [ ] **Step 2: Size the walltime kill.** From P2 T4's per-image time on the
  chosen GPU, set `--gpu-slurm time=` so round 1 is killed after some images
  have committed and before all have. The user picks an input set large enough
  for that (a bounded subset of an existing raw dataset); write the arithmetic
  into the report.
- [ ] **Step 3: Run it and verify**, recording each in the report:
  - the ledger shows `stage4-round-1` ended `TIMEOUT` and a `stage4-round-2`
    that completed;
  - every image has a valid record; `compute.physical_batch` and
    `compute.device_name` are recorded per entry; images committed in round 1
    were not re-embedded in round 2 (`compute` timestamps or job ids in the
    shard logs);
  - `phenotypic-wait` exits `0` for both runs;
  - the no-detector run's scripts have no `stage1`/`stage2` and its controller
    config has `phases: ["stage3", "stage4"]`;
  - an `afterany` cleanup removed the worktree.
- [ ] **Step 4: Commit the script and the report.**

  ```bash
  git add docs/superpowers/plans/2026-11-12-gpu-embedding-staging/live_smoke.sh docs/superpowers/reports/2026-11-12-gpu-embedding-staging/live-smoke.md
  git commit -m "docs(smoke): live SLURM Stage 4, walltime recovery by round (#282 P5)"
  ```

**Deep review P5** (implementation-test-reviewer, Opus):
`reports/2026-11-12-gpu-embedding-staging/review-p5.md`, checking in particular
the sidecar rule (`_cli/CLAUDE.md`, "Array trigger routing"), the counter
separation, and that a v1 controller config still drives an in-flight #281 run.
