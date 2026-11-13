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

### Task 1: Exit codes

**Files:**
- Create: `src/phenotypic/sdk_/_exit_codes.py` (`RunExitCode`), exported from
  `phenotypic.sdk_`
- Modify: `src/phenotypic/phenotypicCLI.py`:
  - :3700-3730, the SLURM submission exits, both `sys.exit(0)` → `SUBMITTED`
  - :3742-3760, the process-only local exit
  - :3830-3920, the local full and measure exits: `0 if failed == 0 else 1`
    becomes `COMPLETE` or `COMPLETE_WITH_FAILURES`; a finalization failure stays
    `ERROR`
  - the recompile and migrate SLURM submission exits → `SUBMITTED`
- Test: `tests/unit/cli/test_cli_exit_codes.py` (new)

**Interfaces:**
- Produces: `class RunExitCode(IntEnum)` with `COMPLETE=0`, `ERROR=1`,
  `COMPLETE_WITH_FAILURES=3`, `SUBMITTED=4`, `ENDED_UNPUBLISHED=5`,
  `WAIT_TIMEOUT=6`. Code 2 stays Click's.

- [ ] **Step 1: Write the failing tests**

```python
def test_clean_local_run_exits_0(tiny_inputs, tmp_path):
    assert run_cli(tiny_inputs, tmp_path).exit_code == RunExitCode.COMPLETE

def test_local_run_with_a_terminal_failure_exits_3(tiny_inputs_one_blank, tmp_path):
    result = run_cli(tiny_inputs_one_blank, tmp_path)   # one plate fails NoObjectsError
    assert result.exit_code == RunExitCode.COMPLETE_WITH_FAILURES
    assert (tmp_path / "deliverables" / "measurements.tsv").is_file()

def test_slurm_submission_exits_4(tiny_inputs, tmp_path, fake_sbatch):
    assert run_cli(tiny_inputs, tmp_path, slurm=True).exit_code == RunExitCode.SUBMITTED

def test_refused_pipeline_still_exits_2(tmp_path):     # Click usage error, unchanged
    assert run_cli_with_unstageable_gpu(tmp_path).exit_code == 2
```

  `fake_sbatch` is the existing fake-SLURM fixture used by
  `tests/unit/cli/test_cli_v2.py` (`TestLiveSlurmRunGuard`).
- [ ] **Step 2: Run them and confirm they fail.** Expect exits 1 and 0 where 3
  and 4 are expected.
- [ ] **Step 3: Implement.** Replace each literal with the enum member.
  `sys.exit(130)` for an interrupt stays as is.
- [ ] **Step 4: Run them, plus `tests/unit/cli/test_cli_v2.py` and
  `tests/unit/cli/test_cli_slurm_submission_summary.py`.** They pass. Any test
  asserting `exit_code == 0` for a SLURM submission now asserts
  `RunExitCode.SUBMITTED`.
- [ ] **Step 5: Commit.** Do **not** merge to `main` before #283 fixes the run
  console (README merge gate).

### Task 2: `phenotypic-wait`

**Files:**
- Create: `src/phenotypic/_cli/_cli_wait.py`
- Modify: `pyproject.toml` `[project.scripts]`:
  `phenotypic-wait = "phenotypic._cli._cli_wait:main"`
- Modify: `tests/unit/ci/test_startup_imports.py`. Add `phenotypic-wait --help`
  to the lazy entry points: no `HEAVY_STARTUP_MODULES`.
- Test: `tests/unit/cli/test_cli_wait.py` (new)

**Interfaces:**
- Consumes:
  - `resolve_run_state(output_dir, depth="shallow")` (`phenotypic.sdk_`)
  - `aggregate_proof_refusal(output_dir) -> str | None` (`sdk_/_run_state.py:1285`)
  - `live_slurm_job_ids(output_dir) -> LiveSlurmJobs` (`_cli/_cli_staged_orchestration.py:709`)
  - `RunExitCode`
- Produces:
  - `decide_wait_outcome(completion: str, aggregate_refusal: str | None, live: LiveSlurmJobs) -> RunExitCode | None`
    (`None` means keep waiting)
  - `wait_for_run(output_dir: Path, *, poll_seconds: float = 60.0, timeout_seconds: float | None = None, sleep=time.sleep, clock=time.monotonic, echo=click.echo) -> RunExitCode`
  - `main()`: a click command taking `--output/-o` (required), `--poll-seconds`
    and `--timeout-seconds`

The decision, applied top to bottom:

| `completion` | any job `live`/`unverified`/`unresolved`? | aggregate proof current? | Outcome |
|---|---|---|---|
| `complete` | any | any | `COMPLETE` |
| any other | yes | any | keep waiting; print which group and why |
| `failed` | no | yes | `COMPLETE_WITH_FAILURES` |
| `failed`, `incomplete` or `active` | no | no | `ENDED_UNPUBLISHED` |

- [ ] **Step 1: Write the failing tests**

```python
NONE = LiveSlurmJobs(live=(), unverified=(), unresolved=())

def test_complete_wins(): assert decide_wait_outcome("complete", None, NONE) is RunExitCode.COMPLETE
def test_live_job_keeps_waiting():
    assert decide_wait_outcome("active", "x", LiveSlurmJobs(("123",), (), ())) is None
def test_unqueryable_scheduler_keeps_waiting():
    assert decide_wait_outcome("active", "x", LiveSlurmJobs((), (), ("g:t",))) is None
def test_failed_but_published_is_3(): assert decide_wait_outcome("failed", None, NONE) is RunExitCode.COMPLETE_WITH_FAILURES

def test_dead_finalizer_with_active_fence_ends_unpublished():
    # Review Focus 4: the fence still says active, but no job is alive
    assert decide_wait_outcome("active", "no aggregate proof", NONE) is RunExitCode.ENDED_UNPUBLISHED

def test_wait_for_run_polls_then_returns(monkeypatch, tmp_path):
    states = iter(["active", "active", "complete"])
    monkeypatch.setattr(_cli_wait, "_completion", lambda d: next(states))
    monkeypatch.setattr(_cli_wait, "live_slurm_job_ids", lambda d: LiveSlurmJobs(("1",), (), ()))
    sleeps = []
    assert wait_for_run(tmp_path, poll_seconds=5, sleep=sleeps.append) is RunExitCode.COMPLETE
    assert sleeps == [5, 5]

def test_timeout_returns_6_and_cancels_nothing(monkeypatch, tmp_path, fake_scancel):
    clock = iter([0.0, 10.0, 20.0])
    monkeypatch.setattr(_cli_wait, "_completion", lambda d: "active")
    monkeypatch.setattr(_cli_wait, "live_slurm_job_ids", lambda d: LiveSlurmJobs(("1",), (), ()))
    assert wait_for_run(tmp_path, poll_seconds=10, timeout_seconds=15,
                        clock=lambda: next(clock), sleep=lambda s: None) is RunExitCode.WAIT_TIMEOUT
    assert fake_scancel.calls == []

def test_cli_exit_code_matches(tmp_path, finished_tiny_run):
    assert CliRunner().invoke(main, ["--output", str(finished_tiny_run)]).exit_code == 0
```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.**
  - `_completion(output_dir)` returns `resolve_run_state(..., depth="shallow").completion`.
  - Import `resolve_run_state` and `live_slurm_job_ids` inside the functions, so
    `--help` stays light.
  - Each poll prints one line: `completion`, the live-job groups, and the
    elapsed time.
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
  them in the proof's artifact map (`_cli_completion.py:1133` neighbourhood).
- Modify: `src/phenotypic/_cli/_cli_readme_generator.py`. Document both files.
- Test: `tests/unit/sdk_/test_run_metrics.py`, `tests/unit/util/test_output_schema.py` (new)

**Interfaces:**
- Produces:
  - `build_run_metrics(*, phenotypic_version: str, pipeline_sha256: str, accepted: int, succeeded: int, failed: int, measured_objects: int, metadata_phantoms: int, embeddings: Mapping[str, Mapping[str, int]] | None = None) -> dict`
    - fixed keys: `schema_version: 1`,
      `images.{accepted,succeeded,failed}`, `objects.measured`,
      `metadata.phantoms`
    - `embeddings` is present only when given (P4 fills it)
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
    text = (tmp_path / "m.json").read_text()
    assert text.endswith("}\n") and "utc" not in text.lower()
    assert json.loads(text)["images"] == {"accepted": 3, "succeeded": 2, "failed": 1}
    assert "embeddings" not in m

def test_schema_names_producer_and_unit(tiny_mirror):
    schema = build_output_schema({"measurements": tiny_mirror})
    area = next(c for c in schema["tables"]["measurements"] if c["name"] == "Size_Area")
    assert area["producer"] == "MeasureSize" and area["unit"] == str(UNITS.PX)

def test_finalize_writes_both_and_the_proof_covers_them(finished_tiny_run):
    d = finished_tiny_run / "deliverables"
    assert (d / "metrics.json").is_file() and (d / "schema.json").is_file()
    (d / "metrics.json").write_text("{}\n")
    assert aggregate_proof_refusal(finished_tiny_run) is not None
```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.**
  - The counts come from `resolve_run_state(output_dir, depth="shallow").diagnostics`
    and the frames finalize already holds. No new tracked value.
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
    return {str(p.relative_to(d)): sha256_file(p) for p in files}

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
- Test: `tests/unit/cli/test_preflight_dvc_cache.py`, `tests/integration/cli/test_dvc_layout_guards.py` (new)

**Interfaces:**
- Produces:
  - `find_dvc_cache_type(start: Path) -> str | None`. It walks up from `start`
    to the first directory holding `.dvc/`, reads `[cache] type` from
    `.dvc/config.local` and then `.dvc/config` (stdlib `configparser`), and
    returns the raw value.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize(("value", "warns"), [("hardlink", True), ("symlink,copy", True),
                                              ("reflink,copy", False), (None, False)])
def test_linked_cache_warns(tmp_path, value, warns):
    make_dvc_repo(tmp_path, cache_type=value)
    findings = run_preflight(output_dir=tmp_path / "out")
    assert any(f.code == "PF-DVC-LINKED-CACHE" for f in findings) is warns

def test_full_and_measure_write_only_the_three_outs(tiny_inputs, tmp_path):     # C1
    run_cli(tiny_inputs, tmp_path); run_cli_measure(tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {"deliverables", "results", ".phenotypic"}

def test_outs_exist_when_every_image_fails(all_blank_inputs, tmp_path):        # C14
    run_cli(all_blank_inputs, tmp_path)
    assert all((tmp_path / n).is_dir() for n in ("deliverables", "results", ".phenotypic"))

def test_restored_older_state_fails_safe(two_revision_run):                       # C13
    out, old_state = two_revision_run      # .phenotypic/ from revision 1, stores from revision 2
    shutil.rmtree(out / ".phenotypic"); shutil.copytree(old_state, out / ".phenotypic")
    state = resolve_run_state(out, depth="deep")
    assert state.completion != "complete"
    assert not live_slurm_job_ids(out)    # a restored fence never reads as live work
```

- [ ] **Step 2: Run them and confirm they fail** (the check doesn't exist yet;
  C1, C13 and C14 may already pass).
- [ ] **Step 3: Implement the check.**
  - It warns when any comma-separated value is `hardlink` or `symlink`.
  - The message names the config file, and says `.phenotypic/` holds files
    appended in place.
  - If C1 or C14 fails, make that mode create its outs at startup, below the
    `--dry-run` exit.
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
   The `sbatch` line that runs `dvc repro` uses `--partition=intel --cpus-per-task=1
   --mem=4G --time=7-00:00:00`, and the page says why: head nodes kill long
   processes.
3. The process-mode variant. The out is the whole `<out>` root, because process
   output mirrors the input tree and has no `deliverables/`.
4. `cache.type`: keep `reflink,copy`. On GPFS that means copy, and the page says
   what it costs. `hardlink`/`symlink` are unsupported (C6).
5. Params: `dvc exp run -S 'pipeline.pht-pipe.json:pipe_cfgs.<OpKey>.params.<field>=<v>'`,
   with the caveats that an op key containing `.` can't be addressed and that
   DVC likely rewrites the file as one line.
6. After GUI curation, `dvc status` shows `deliverables` modified; `dvc commit`
   records it.
7. The exit codes table (README).
8. `dvc gc` for the cache growth tracking `.phenotypic/` causes.

9. Whether `results/` should be `cache: true` or `false`, from Step 1's
   measurement (C8).

- [ ] **Step 1: Measure DVC on a real tree (C8).** Run this as a Slurm job:
  `slurm-job` skill, `intel`, 4 CPU, 16 GB, one configuration per fresh node
  (`benchmark-on-cold-nodes` memory).
  - Work on a scratch DVC repo whose `out/` is a hardlinked copy
    (`cp -al`) of the migrated maresca tree `2026-08-11-ome-zarr`.
  - Time `uvx dvc add out/results` with `cache: true` and the default copy
    cache.
  - Time `dvc status` cold and warm.
  - Record file count, wall time and cache size in the page.
  - Recommend `cache: false` for `results/` if `dvc add` exceeds an hour.
- [ ] **Step 2: Write the page.**
- [ ] **Step 3: Build the docs as a Slurm job** (`slurm-job` skill), with
  notebook execution off.
- [ ] **Step 4: Read the rendered page.** Every block renders, and the toctree
  links to it.
- [ ] **Step 5: Commit.**
