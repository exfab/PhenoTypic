# P2 · DVC run contract

**Spec:** Part C (C1–C14)

---

**Goal:** An orchestrator can drive a PhenoTypic run as one DVC stage and act on
the result without parsing logs. That needs:
- distinct exit codes;
- a waiter for SLURM runs;
- a byte-stable run summary and column schema;
- byte-stable deliverables;
- guards on the output layout;
- a how-to page.

**GUI:** the run console reads exit `4` as a failed submission
(`_gui/run_console/_slurm.py:826`). That is a known #283 break; the GUI SLURM
tests mock the return code and stay green.

### Task 1: Exit codes

**Files:**
- Create: `src/phenotypic/sdk_/_exit_codes.py` (`RunExitCode`), exported from
  `phenotypic.sdk_`
- Modify: `src/phenotypic/phenotypicCLI.py`, every exit site below:

  | Line | Today | Becomes |
  |---|---|---|
  | :2528 | `sys.exit(0)` after local **and** SLURM recompile | branch: SLURM → `SUBMITTED`; local → `COMPLETE` |
  | :3434 | staged "All images already processed!" → `0` | `COMPLETE`, or `COMPLETE_WITH_FAILURES` when the run holds recorded terminal failures and `--retry-failures` is off |
  | :3477 | forward run with nothing left submits a SLURM recompile → `0` | `SUBMITTED` |
  | :3497 | "All images already processed!" / "N recorded failure(s) skipped" (:3488-3494) → `0` | as :3434 |
  | :3712 | process SLURM submission → `0` | `SUBMITTED` |
  | :3729 | full/measure/staged SLURM submission → `0` | `SUBMITTED` |
  | :3759 | process-only local → `0 if failed == 0 else 1` | `COMPLETE` or `COMPLETE_WITH_FAILURES` |
  | :3894 | staged finalization failed → `1` | `ERROR`, unchanged |
  | :3916 | local full/measure → `0 if failed == 0 else 1` | `COMPLETE` or `COMPLETE_WITH_FAILURES` |
  | :3025 | dry run → `0` | unchanged |
  | :3920 | interrupt → `130` | unchanged |

- Modify: `src/phenotypic/_cli/_cli_migrate.py`. `handle_migrate_mode` returns
  `SUBMITTED` for a submitted attempt (:2576) and for a submitted SLURM
  `--dry-run`, which submits the validation chain (:2409-2410). The docstring
  says so. A local migrate keeps `0`/`1`.
- Modify: `src/phenotypic/_cli/CLAUDE.md:667`. "a submitted migration exits 0"
  becomes "exits `4`", and the exit-code table is added to the SLURM chaining
  section.
- Modify: `tests/unit/cli/conftest.py`. Move the `submissions` fixture from
  `tests/unit/cli/test_cli_slurm_submission_summary.py:67-118` here, unchanged.
- Create (helpers), in `tests/_tiny_run.py`: fixture `tiny_inputs_one_bad`,
  `tiny_inputs` plus one truncated `.tif`. Its read fails inside the per-image
  worker in every mode, so it is recorded as a terminal failure; confirm that
  in Step 2 from the run's terminal-failure journal.
- Test: `tests/unit/cli/test_cli_exit_codes.py` (new)

**Interfaces:**
- Produces: `class RunExitCode(IntEnum)` with `COMPLETE=0`, `ERROR=1`,
  `COMPLETE_WITH_FAILURES=3`, `SUBMITTED=4`, `ENDED_UNPUBLISHED=5`,
  `WAIT_TIMEOUT=6`. Code 2 stays Click's.

- [ ] **Step 1: Write the failing tests**

```python
def test_clean_local_run_exits_0(tiny_inputs, tmp_path):
    assert run_cli(tiny_inputs, tmp_path / "out").exit_code == RunExitCode.COMPLETE

def test_local_run_with_a_terminal_failure_exits_3(tiny_inputs_one_bad, tmp_path):
    result = run_cli(tiny_inputs_one_bad, tmp_path / "out")
    assert result.exit_code == RunExitCode.COMPLETE_WITH_FAILURES
    assert (tmp_path / "out" / "deliverables" / "measurements.tsv").is_file()

def test_rerunning_a_run_that_exited_3_still_exits_3(tiny_inputs_one_bad, tmp_path):
    run_cli(tiny_inputs_one_bad, tmp_path / "out")
    assert run_cli(tiny_inputs_one_bad, tmp_path / "out").exit_code == RunExitCode.COMPLETE_WITH_FAILURES

def test_process_local_run_with_a_failure_exits_3(tiny_inputs_one_bad, tmp_path):
    result = run_cli(tiny_inputs_one_bad, tmp_path / "out", mode="process",
                     extra=["--layer", "gray", "--process-format", "zarr"])
    assert result.exit_code == RunExitCode.COMPLETE_WITH_FAILURES

@pytest.mark.parametrize(("mode", "extra"), [("full", []),
                                             ("process", ["--layer", "gray", "--process-format", "zarr"])])
def test_every_slurm_submission_exits_4(tiny_inputs, tmp_path, submissions, mode, extra):
    result = run_cli(tiny_inputs, tmp_path / "out", mode=mode, slurm=True, extra=extra)
    assert result.exit_code == RunExitCode.SUBMITTED

def test_a_slurm_recompile_exits_4(finished_tiny_run, submissions):
    assert run_cli(None, finished_tiny_run, mode="recompile", slurm=True).exit_code == RunExitCode.SUBMITTED

def test_a_forward_slurm_run_with_nothing_left_exits_4(finished_tiny_run, tiny_inputs, submissions,
                                                       metadata_bytes, tmp_path):
    meta = tmp_path / "meta.csv"; meta.write_bytes(metadata_bytes[".csv"])     # a new snapshot
    result = run_cli(tiny_inputs, finished_tiny_run, slurm=True, extra=["--metadata", str(meta)])
    assert result.exit_code == RunExitCode.SUBMITTED

def test_a_usage_refusal_still_exits_2(tiny_inputs, tmp_path):     # Click's own code
    assert run_cli(tiny_inputs, tmp_path / "out", extra=["--revert"]).exit_code == 2
```

  A metadata file the run has not snapshotted is what routes a no-work forward
  run to the SLURM recompile (:3460-3477). The migrate submission is pinned by the existing
  `tests/unit/cli/test_cli_migrate_slurm.py` once Step 4 updates its asserts.
- [ ] **Step 2: Run them and confirm they fail.** Expect exits `1` and `0`
  where `3` and `4` are expected.
- [ ] **Step 3: Implement.** Replace each literal with the enum member. The
  "already processed" sites reuse the `terminal_failure_index` count that
  :3488 already computes.
- [ ] **Step 4: Update the SLURM tests.** Derive the list mechanically:
  `grep -rlE "slurm|SLURM" tests/unit/cli tests/integration | xargs grep -l "exit_code == 0"`
  (31 files today, among them `test_cli_recompile_slurm.py`,
  `test_cli_migrate_slurm.py`, `test_cli_provenance_migration_slurm.py`,
  `test_staged_stage1_reads_like_single_pass.py`, `test_cli_reference_e2e.py`,
  `test_cli_image_manifest.py`, `test_cli_store_options.py`,
  `test_process_only_e2e.py`). In each, an assert about a **SLURM submission**
  becomes `== RunExitCode.SUBMITTED`; every other `== 0` stays.
- [ ] **Step 5: Run the new tests and every file from Step 4.** They pass.
- [ ] **Step 6: Commit.** Do **not** merge to `main` before #283 (README merge
  gate).

### Task 2: `phenotypic-wait`

**Files:**
- Create: `src/phenotypic/_cli/_cli_wait.py`
- Modify: the SLURM finalizers (ordinary, staged, recompile). They stamp the
  generation they ran under into the aggregate proof as `execution_epoch`
  (`null` for a local run), threaded through `aggregate_measurements` and
  `finalize_run` to `publish_aggregate_snapshot`, and their fence close records
  `closed_by: "finalizer"` (`FENCE_CLOSED_BY_FINALIZER`). Cancels and failure
  teardowns leave `closed_by` unset.
- Modify: `pyproject.toml` `[project.scripts]`:
  `phenotypic-wait = "phenotypic._cli._cli_wait:main"`
- Modify: `tests/unit/ci/test_startup_imports.py`. Add `phenotypic-wait --help`
  to the lazy entry points: no `HEAVY_STARTUP_MODULES`.
- Test: `tests/unit/cli/test_cli_wait.py`; the finalizer stamps in
  `test_cli_checkpoint_handler.py`, `test_staged_controller.py` and
  `test_cli_recompile_slurm.py`

**Interfaces:**
- Consumes:
  - `resolve_run_state(output_dir, depth="shallow")` (`phenotypic.sdk_`)
  - `live_slurm_job_ids(output_dir) -> LiveSlurmJobs` (`_cli/_cli_staged_orchestration.py:709`)
  - `load_slurm_lifecycle(output_dir)` (`_cli/_cli_slurm_lifecycle.py:165`)
  - the run completion marker's `execution_epoch` (`_cli_completion.py:1234`)
  - `RunExitCode`
- Produces:
  - Module-level wrappers in `_cli_wait`, each importing lazily so `--help`
    stays light, and each the seam tests monkeypatch:
    - `_completion(d) -> str`: `resolve_run_state(d, depth="shallow").completion`
    - `_live_jobs(d) -> LiveSlurmJobs`: `live_slurm_job_ids(d)`
    - `_fence_generation(d) -> str | None`: the fence's `generation`, or `None`
      with no fence
    - `_published(d, generation: str | None) -> bool`
    - `_is_process_run(d) -> bool`: the run config's `process_only_layer` is set
  - `decide_wait_outcome(completion: str, live: LiveSlurmJobs, *, published: bool, process_run: bool) -> RunExitCode | None`
    (`None` means keep waiting)
  - `wait_for_run(output_dir: Path, *, poll_seconds: float = 60.0, timeout_seconds: float | None = None, sleep=time.sleep, clock=time.monotonic, echo=click.echo) -> RunExitCode`
  - `main()`: a click command taking `--output/-o` (required), `--poll-seconds`
    and `--timeout-seconds`

**Published** means this submission published. `wait_for_run` reads the
fence's generation **once, at the first poll**, and `_published(d, generation)`
is true when any of these holds:
- that generation's fence is closed by its finalizer: same `generation`,
  `active: false`, `closed_by: "finalizer"`, no `terminal_status`;
- the run completion marker's `execution_epoch` equals that generation;
- a valid aggregate proof (`valid_aggregate_snapshot`) whose `execution_epoch`
  equals that generation. A proof without the field never counts.

Matching run state is not enough: a `--mode measure` resubmission keeps no run
state, so an earlier run's aggregate proof would still match after the new
finalizer died.

With no fence at the first poll (the run before it was local), a `complete` or
`failed` run state counts as published.

The decision, applied top to bottom:

| Any job `live`/`unverified`/`unresolved`? | `completion` | `published` | process run? | Outcome |
|---|---|---|---|---|
| yes | any | any | any | keep waiting; print which group and why |
| no | `complete` | yes | any | `COMPLETE` |
| no | `failed` | yes | any | `COMPLETE_WITH_FAILURES` |
| no | `failed` | no | yes | `COMPLETE_WITH_FAILURES` (process mode publishes no aggregate proof, `_cli_completion.py:1254-1258`) |
| no | anything else | | | `ENDED_UNPUBLISHED` |

The waiter does not serve `--mode migrate`, which keeps its lifecycle under an
external control root, so `live_slurm_job_ids(output_dir)` sees none of its
jobs.

- [ ] **Step 1: Write the failing tests**

```python
NONE = LiveSlurmJobs()
LIVE = LiveSlurmJobs(live=("123",))

def test_a_live_job_keeps_waiting_even_when_complete():
    # a measure resubmission over a completed run: the old proof is still current
    assert decide_wait_outcome("complete", LIVE, published=False, process_run=False) is None

def test_unqueryable_scheduler_keeps_waiting():
    assert decide_wait_outcome("active", LiveSlurmJobs(unresolved=("phenotypic:g:t",)),
                               published=False, process_run=False) is None

def test_complete_and_published_is_0():
    assert decide_wait_outcome("complete", NONE, published=True, process_run=False) is RunExitCode.COMPLETE

def test_failed_and_published_is_3():
    assert decide_wait_outcome("failed", NONE, published=True, process_run=False) is RunExitCode.COMPLETE_WITH_FAILURES

def test_a_stale_proof_with_a_dead_finalizer_is_5():
    assert decide_wait_outcome("failed", NONE, published=False, process_run=False) is RunExitCode.ENDED_UNPUBLISHED

def test_a_process_run_with_one_failure_is_3():
    assert decide_wait_outcome("failed", NONE, published=False, process_run=True) is RunExitCode.COMPLETE_WITH_FAILURES

def test_dead_finalizer_with_active_fence_ends_unpublished():
    # Review Focus 4: the fence still says active, but no job is alive
    assert decide_wait_outcome("active", NONE, published=False, process_run=False) is RunExitCode.ENDED_UNPUBLISHED

def test_measure_resubmission_waits_then_binds_to_its_own_generation(monkeypatch, tmp_path):
    jobs = iter([LIVE, LIVE, NONE])
    monkeypatch.setattr(_cli_wait, "_completion", lambda d: "complete")
    monkeypatch.setattr(_cli_wait, "_live_jobs", lambda d: next(jobs))
    generations = iter(["g-measure", "g-later"])
    monkeypatch.setattr(_cli_wait, "_fence_generation", lambda d: next(generations))
    seen = []
    monkeypatch.setattr(_cli_wait, "_published", lambda d, g: seen.append(g) or True)
    monkeypatch.setattr(_cli_wait, "_is_process_run", lambda d: False)
    sleeps = []
    assert wait_for_run(tmp_path, poll_seconds=5, sleep=sleeps.append) is RunExitCode.COMPLETE
    assert sleeps == [5, 5] and set(seen) == {"g-measure"}   # read once, at the first poll

def test_an_intent_only_ledger_row_keeps_waiting(monkeypatch, tmp_path):
    initialize_slurm_lifecycle(tmp_path, generation="g", mode="full")
    append_lifecycle_entry(tmp_path, generation="g", token="t", role="finalizer", status="intent")
    monkeypatch.setattr(_cli_staged_orchestration, "query_scheduler_comments",
                        lambda **_: (_ for _ in ()).throw(SchedulerQueryUnavailable("down")))
    assert decide_wait_outcome("active", _cli_wait._live_jobs(tmp_path),
                               published=False, process_run=False) is None

def test_timeout_returns_6_and_cancels_nothing(monkeypatch, tmp_path):
    cancelled = []
    monkeypatch.setattr(_cli_slurm_lifecycle, "cancel_generation", lambda *a, **k: cancelled.append(a))
    clock = iter([0.0, 10.0, 20.0])
    monkeypatch.setattr(_cli_wait, "_completion", lambda d: "active")
    monkeypatch.setattr(_cli_wait, "_live_jobs", lambda d: LIVE)
    monkeypatch.setattr(_cli_wait, "_fence_generation", lambda d: "g")
    assert wait_for_run(tmp_path, poll_seconds=10, timeout_seconds=15,
                        clock=lambda: next(clock), sleep=lambda s: None) is RunExitCode.WAIT_TIMEOUT
    assert cancelled == []

def test_cli_exit_code_matches(finished_tiny_run):
    assert CliRunner().invoke(main, ["--output", str(finished_tiny_run)]).exit_code == 0

def test_measure_resubmission_with_a_dead_finalizer_ends_unpublished(...): ...   # → 5
def test_a_cancelled_resubmission_ends_unpublished(...): ...                     # → 5
def test_an_unstamped_aggregate_proof_never_counts(...): ...
```

  `append_lifecycle_entry` is `_cli_slurm_lifecycle.py:232`; the row has no
  `submitted` successor.
- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.**
  - Each poll prints one line: `completion`, the live-job groups, and the
    elapsed time.
  - The waiter never aggregates, publishes, submits or cancels. Its only write
    is the verification cache `resolve_run_state` already refreshes.
- [ ] **Step 4: Run the tests and the startup-import guard.** They pass.
- [ ] **Step 5: Commit.**

### Task 3: `metrics.json` and `schema.json`

**Files:**
- Create: `src/phenotypic/sdk_/_run_metrics.py`
- Modify: `src/phenotypic/util/_measurement_outputs.py`. Add
  `build_output_schema`, built on `_member_for_column` (:435) and
  `_producer_column_groups` (:258).
- Modify: `src/phenotypic/sdk_/_io_constants.py`. Add
  `METRICS_JSON = "metrics.json"`, `SCHEMA_JSON = "schema.json"`,
  `metrics_json_path(output_dir)` and `schema_json_path(output_dir)`, all under
  `deliverables/`.
- Modify: `src/phenotypic/_cli/_cli_finalize_run.py:407`. `finalize_run` writes
  both after the mirror and splits, **before** the aggregate proof, and declares
  them in the proof's artifact map (`_cli_completion.py:1131-1135`).
- Modify: `src/phenotypic/_cli/_cli_readme_generator.py`. Document both files.
- Create (helpers), in `tests/_tiny_run.py`: fixture `tiny_mirror`, the
  `measurements.parquet` frame of `finished_tiny_run`
- Test: `tests/unit/sdk_/test_run_metrics.py`, `tests/unit/util/test_output_schema.py` (new)

**Interfaces:**
- Produces:
  - `build_run_metrics(*, phenotypic_version: str, pipeline_sha256: str, accepted: int, succeeded: int, failed: int, measured_objects: int, metadata_phantoms: int, embeddings: Mapping[str, Mapping[str, int]] | None = None) -> dict`
    - fixed keys: `schema_version: 1`,
      `images.{accepted,succeeded,failed}`, `objects.measured`,
      `metadata.phantoms`
    - `embeddings` is present only when given (P4 T4 fills it)
  - `build_output_schema(tables: Mapping[str, pl.DataFrame]) -> dict` with shape
    `{"schema_version": 1, "tables": {name: [{"name", "dtype", "unit",
    "producer", "member", "desc"}]}}`. Columns no enum owns get
    `producer: null, member: null, desc: null`.
  - `write_canonical_json(path: Path, obj) -> None`: atomic,
    `json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n"`

- [ ] **Step 1: Write the failing tests**

```python
def test_metrics_has_no_wall_clock_and_is_canonical(tmp_path):
    m = build_run_metrics(phenotypic_version="0.21.0", pipeline_sha256="a" * 64, accepted=3,
                          succeeded=2, failed=1, measured_objects=180, metadata_phantoms=4)
    write_canonical_json(tmp_path / "m.json", m)
    text = (tmp_path / "m.json").read_text(encoding="utf-8")
    assert text.endswith("}\n") and "utc" not in text.lower()
    assert json.loads(text)["images"] == {"accepted": 3, "succeeded": 2, "failed": 1}
    assert "embeddings" not in m

def test_schema_names_producer_and_unit(tiny_mirror):
    schema = build_output_schema({"measurements": tiny_mirror})
    area = next(c for c in schema["tables"]["measurements"] if c["name"] == "Size_Area")
    assert area["producer"] == "MeasureSize" and area["unit"] == str(UNITS.PX2)

def test_finalize_writes_both_and_the_proof_covers_them(finished_tiny_run):
    d = finished_tiny_run / "deliverables"
    assert (d / "metrics.json").is_file() and (d / "schema.json").is_file()
    (d / "metrics.json").write_text("{}\n")
    assert aggregate_proof_refusal(finished_tiny_run) is not None
```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.**
  - The counts come from what finalize already holds, never from a deep
    `resolve_run_state`, which would run an O(N) verification and write the
    verification cache inside the finalizer:
    - `accepted`: the state's `work_ids`;
    - `failed`: the terminal-failure journal, restricted to those work ids;
    - `succeeded`: the images in the source set finalize aggregated;
    - `measured_objects` and `metadata_phantoms`: the master and the mirror
      frames.
  - `pipeline_sha256` is the run config's `pipeline_sha256`
    (`_cli_state_management.py:378`). P3 T4 replaces it with the sha of the run
    folders the finalizer aggregated.
  - `unit` is taken from the member, falling back to the enum's
    `default_unit()`.
- [ ] **Step 4: Run them, plus the importers of `_cli_finalize_run` and
  `_cli_completion`.** They pass.
- [ ] **Step 5: Commit.**

### Task 4: Byte-stable deliverables

**Files:**
- Test: `tests/integration/cli/test_deliverables_byte_stable.py` (new)
- Modify: whichever writer the test shows to vary. Likely suspects to inspect
  first: row order after a fan-out merge, JSON key order, and any Parquet
  key/value metadata that carries a path or time.

**Interfaces:**
- Consumes: Task 3's files.

- [ ] **Step 1: Write the test**

```python
STABLE = ("measurements.parquet", "measurements.tsv", "master_measurements.parquet",
          "metrics.json", "schema.json")
STABLE_DIRS = ("measurements_by_feature", "measurements_by_category", "embeddings")

def _digests(out):
    d = out / "deliverables"
    files = [d / n for n in STABLE if (d / n).exists()]
    files += [p for sub in STABLE_DIRS for p in sorted((d / sub).rglob("*")) if p.is_file()]
    return {str(p.relative_to(d)): file_sha256(p) for p in files}

def test_identical_runs_give_identical_deliverables(tiny_inputs, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    run_cli(tiny_inputs, a); run_cli(tiny_inputs, b)
    assert _digests(a) == _digests(b)
```

- [ ] **Step 2: Run it.** Expect a failure listing the differing files, or a
  pass. A pass needs no code change; keep the test.
- [ ] **Step 3: For each differing file, remove or pin the varying field.**
  Never exempt a file from `STABLE`; the spec lists them (C11).
- [ ] **Step 4: Run it three times.** It passes every time.
- [ ] **Step 5: Commit.**

### Task 5: Layout guards

**Files:**
- Modify: `src/phenotypic/_cli/_cli_preflight.py`. Add the check
  `_check_dvc_cache_link_type` to `CHECKS`, `FindingCode` `"PF-DVC-LINKED-CACHE"`
  (warning) and a `HINTS` entry.
- Modify: `docs/source/tutorials/pages/cli_batch_processing.md`. Add the
  finding-code row; `tests/unit/test_docs_preflight_codes.py` fails without it.
- Create (helpers), in `tests/_tiny_run.py`:
  - fixture `all_bad_inputs`: two truncated `.tif` files, as in `tiny_inputs_one_bad`
  - `make_dvc_repo(root, *, cache_type: str | None, local_type: str | None = None)`:
    writes `.dvc/config` (and `.dvc/config.local` when given) with `[cache] type`
  - `run_preflight(*, output_dir) -> list[PreflightFinding]`: the preflight
    over `tiny_pipeline_file` with that `--output`
- Test: `tests/unit/cli/test_preflight_dvc_cache.py`, `tests/integration/cli/test_dvc_layout_guards.py` (new)

**Interfaces:**
- Produces:
  - `find_dvc_cache_type(start: Path) -> str | None`. It walks up from `start`
    to the first directory holding `.dvc/`, reads `[cache] type` from
    `.dvc/config` and overlays `.dvc/config.local` on it (stdlib
    `configparser`), and returns the raw value. Global and system DVC config
    are ignored, and the warning text says so.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize(("value", "warns"), [("hardlink", True), ("symlink,copy", True),
                                              ("reflink,copy", False), (None, False)])
def test_linked_cache_warns(tmp_path, value, warns):
    make_dvc_repo(tmp_path, cache_type=value)
    findings = run_preflight(output_dir=tmp_path / "out")
    assert any(f.code == "PF-DVC-LINKED-CACHE" for f in findings) is warns

def test_config_local_overrides_config(tmp_path):
    make_dvc_repo(tmp_path, cache_type="reflink,copy", local_type="hardlink")
    assert find_dvc_cache_type(tmp_path / "out") == "hardlink"

def test_full_and_measure_write_only_the_three_outs(tiny_inputs, tmp_path):     # C1
    run_cli(tiny_inputs, tmp_path / "out"); run_cli(None, tmp_path / "out", mode="measure")
    assert {p.name for p in (tmp_path / "out").iterdir()} == {"deliverables", "results", ".phenotypic"}

def test_outs_exist_when_every_image_fails(all_bad_inputs, tmp_path):        # C14
    run_cli(all_bad_inputs, tmp_path / "out")
    assert all((tmp_path / "out" / n).is_dir() for n in ("deliverables", "results", ".phenotypic"))

def test_restored_liveness_state_fails_safe(finished_tiny_run, monkeypatch):    # C13
    out = finished_tiny_run
    initialize_slurm_lifecycle(out, generation="g-old", mode="full")         # a restored `active: true` fence
    append_lifecycle_entry(out, generation="g-old", token="t", role="finalizer",
                           status="submitted", job_id="77")
    monkeypatch.setattr(_cli_staged_orchestration, "scheduler_job_is_active", lambda job_id: False)
    write_owner_record(out, pid=dead_pid())                                    # gui_launch_owner.json, pid gone
    assert not live_slurm_job_ids(out)
    assert resolve_run_state(out, depth="deep").completion != "active"
    assert run_cli(None, out, mode="recompile").exit_code == RunExitCode.COMPLETE   # the guard lets it run
```

  `write_owner_record` writes `gui_launch_owner_path(out)` in the shape
  `RunRegistry._persist_record_locked` writes (`_gui/shell/_runs_registry.py:1481`);
  `dead_pid()` returns the pid of a child process that has exited. Both are
  module-local.
- [ ] **Step 2: Run them and confirm they fail** (the check doesn't exist yet;
  C1, C13 and C14 may already pass).
- [ ] **Step 3: Implement the check.**
  - It warns when any comma-separated value is `hardlink` or `symlink`.
  - The message names the config file, says `.phenotypic/` holds files
    appended in place, and says global and system DVC config are not read.
  - If C1 or C14 fails, make that mode create its outs at startup, below the
    `--dry-run` exit.
  - If C13 fails, fix the guard that trusted the restored file, and record the
    migration-lease result in the commit message (spec C13).
- [ ] **Step 4: Run them, plus `tests/unit/cli/test_cli_preflight_ordering.py`
  and `tests/unit/test_docs_preflight_codes.py`.** They pass.
- [ ] **Step 5: Commit.**

### Task 6: DVC how-to page

**Files:**
- Create: `docs/source/how_to/pages/dvc.md`, and add it to the how-to toctree
- Test: a render check, not pytest. Run as a Slurm job:
  `sphinx-build -j "$SLURM_CPUS_PER_TASK" -D nbsphinx_execute=never`. Read the
  generated `dvc.html`.

The page covers, each with a copy-ready block:

1. A local `dvc.yaml`:
   - `cmd`: `python -m phenotypic -i data -o out -p pipeline.pht-pipe.json --njobs 16`
   - `deps`: the input dir
   - `params`: `pipeline.pht-pipe.json:`
   - three `outs`, each `persist: true`
   - top-level `metrics: [out/deliverables/metrics.json]`
   - `plots` over `out/deliverables/measurements.tsv`
2. The SLURM variant. `cmd` is a list: submit, then `phenotypic-wait --output out`.
   - The `sbatch` line that runs `dvc repro` uses `--partition=intel
     --cpus-per-task=1 --mem=4G --time=7-00:00:00`, and the page says why: head
     nodes kill long processes.
   - The waiter refreshes `.phenotypic/verification_cache.json` as it polls;
     that is its only write, and it is harmless.
   - The waiter does not serve `--mode migrate`.
3. The process-mode variant. The out is the whole `<out>` root, because process
   output mirrors the input tree and has no `deliverables/`.
4. An in-place process run (`--output` equal to `--input`, P5 T3) is **not** a
   DVC stage: its input tree would also be its out.
5. `cache.type`: keep `reflink,copy`. On GPFS that means copy, and the page says
   what it costs. `hardlink`/`symlink` are unsupported (C6). Only the
   project's `.dvc/config` and `.dvc/config.local` are checked.
6. Params: `dvc exp run -S 'pipeline.pht-pipe.json:pipe_cfgs.<OpKey>.params.<field>=<v>'`,
   with the caveats that an op key containing `.` can't be addressed and that
   DVC likely rewrites the file as one line.
7. After GUI curation, `dvc status` shows `deliverables` modified; `dvc commit`
   records it.
8. The exit codes table (README).
9. `dvc gc` for the cache growth tracking `.phenotypic/` causes.
10. Whether `results/` should be `cache: true` or `false`, from Step 1's
    measurement (C8).

- [ ] **Step 1: Measure DVC on a bounded subset (C8).** A full copy-cache
  `dvc add` of the migrated maresca tree would duplicate 1,354 GB, HDF
  hardlinks included, so measure a subset and extrapolate. Run it as a Slurm
  job (`slurm-job` skill): `intel`, 4 CPU, 16 GB, `--time=12:00:00`, one
  configuration per fresh node (`benchmark-on-cold-nodes` memory).
  - The scratch DVC repo sits on `/bigdata`, the same filesystem as the source.
  - Its `out/results` holds **one dataset** of `2026-08-11-ome-zarr`, copied
    with `cp -al`, `hdf/` excluded. Record its file count and bytes.
  - Time `uvx dvc add out/results` with `cache: true` and the default copy
    cache, then `dvc status` cold and warm, then `dvc commit` after touching
    one store.
  - Extrapolate each time and the cache size to the full tree by file count
    and bytes, and record measured and extrapolated numbers in the page.
  - Recommend `cache: false` for `results/` if the extrapolated `dvc add`
    exceeds an hour.
  - Delete the scratch repo and its cache afterwards.
- [ ] **Step 2: Write the page.**
- [ ] **Step 3: Build the docs as a Slurm job** (`slurm-job` skill), with
  notebook execution off.
- [ ] **Step 4: Read the rendered page.** Every block renders, and the toctree
  links to it.
- [ ] **Step 5: Commit.**
