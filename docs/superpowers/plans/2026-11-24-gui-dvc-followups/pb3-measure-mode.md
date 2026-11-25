# PB3 · `--mode measure` in the run console (D6)

**Spec:** PB5, §3.4; D10, D13; review I3, C3, Minors 5, 18

---

**Goal:** A Full/Measure selector launches `--mode measure` with exactly the flags the CLI
accepts; the registry records the CLI mode; the observer shows a measure pass's outcome,
terminal in every case.

**Depends on:** PB2.

### Task 1: the selector, the form rules, Validate (C19)

**Files:**
- Modify: `src/phenotypic/_gui/run_console/_state.py` (`cli_mode` field,
  `run_state_from_json`/`run_state_to_json` (`:121, 151`), `to_argv`); `_ids.py` (`RC_RADIO_CLI_MODE = "rc-radio-cli-mode"`); `_form.py` (selector
  beside `rc-radio-mode`, `_ids.py:105`).
- Modify: `src/phenotypic/_gui/run_console/_callbacks.py`: slot visibility and disabling;
  `update_run_disabled` (`:2839-2855`) and the Validate button; the launch seam's validate arm
  (`:2063-2066`); `_launch_refusal`'s measure hook (PB2).
- Modify: `src/phenotypic/_cli/_cli_reference.py` and `src/phenotypic/phenotypicCLI.py`: move
  `_reference_operation_paths_for` (`phenotypicCLI.py:739`) and
  `_refuse_measuring_reference_pipeline` (`:761-773`) into `_cli_reference.py`, which the CLI
  then imports; add `measure_reference_refusal` there. A run-console callback then never
  imports the 4,800-line CLI module (Minor 13). Before the move, `grep -rn
  '_refuse_measuring_reference_pipeline\|_reference_operation_paths_for' tests/` on the rebased
  #282 tree: today there is no monkeypatch target; leave a re-export in `phenotypicCLI.py` only
  if one has appeared (R2-M6).
- Test: `tests/unit/gui/run_console/test_state.py`, `tests/integration/gui/test_run_console_callbacks.py`,
  `tests/integration/cli/test_gui_measure_argv.py` (new).

**Interfaces:**
- Consumes: `_output_holds_a_run` (`shell/_runs_registry.py:187-200`);
  `measure_reference_refusal(pipeline_json: Path) -> str | None` in `_cli/_cli_reference.py`
  (above), the refusal's message or `None`;
  `resolve_metadata_snapshot_path`.
- Produces:
  - `RunConsoleState.cli_mode: Literal["full", "measure"] = "full"`, in the command digest.
  - `to_argv(state)` for measure: `["--mode", "measure", "--pipeline", p, "--output", o]` plus
    `--metadata`, `--nrows`, `--ncols`, `--image-type`, `--njobs`, `--embed-stage` when set;
    never `--input`, `--dry-run`, `--retry-failures`, `--sample`, `--restart`, `--overwrite`.
    `to_argv` no longer requires `input_dir` in measure mode.
  - The input slot is hidden in measure mode but **kept in state** (Review Focus 5).

- [ ] **Step 1: Write the failing tests.**

```python
REFUSED = {"--input", "--dry-run", "--retry-failures", "--sample", "--restart", "--overwrite"}

def test_the_measure_argv_carries_only_accepted_flags():
    argv = to_argv(full_state(cli_mode="measure", retry_failures=True, dry_run=True, advanced_args={"sample": 2}))
    assert argv[:2] == ["--mode", "measure"] and not REFUSED & set(argv)

def test_switching_cli_mode_keeps_the_input_slot():                         # Review Focus 5
    s = full_state(input_dir="/p/in"); s2 = flip(flip(s, "measure"), "full")
    assert s2.input_dir == "/p/in" and "--input" in to_argv(s2)

def test_an_old_preset_loads_as_full(): ...
def test_validate_is_disabled_in_measure_mode(): ...
def test_the_seam_refuses_a_stale_validate_in_measure_mode(): ...
def test_an_output_without_a_run_disables_run_with_the_message(tmp_path):
    assert "Measure re-measures an existing output" in run_disabled_reason(cli_mode="measure", output_dir=tmp_path / "new")
def test_an_empty_output_disables_run_with_the_message(tmp_path):              # Minor 14
    (tmp_path / "empty").mkdir()
    assert "Measure re-measures an existing output" in run_disabled_reason(cli_mode="measure", output_dir=tmp_path / "empty")
def test_both_metadata_snapshots_are_refused_in_measure_mode(tmp_path): ...    # the resolver's message (Minor 14)
def test_a_reference_metadata_pipeline_is_refused_in_measure_mode(): ...
def test_the_reference_requirement_alert_stays_closed_in_measure_mode(): ...
def test_the_gpu_section_shows_for_dino_only_and_not_for_a_detector_alone_in_measure(): ...
```

And the argv acceptance test (Minor 18):
```python
@pytest.mark.parametrize("form", MEASURE_FORMS)            # every flag combination the form can build
def test_every_measure_argv_parses(form, monkeypatch, measured_output):
    # measured_output holds a finished forward run and NO lifecycle fence: _refuse_live_slurm_run
    # runs for measure too (phenotypicCLI.py:3343) and would query the scheduler (Minor 14)
    class Reached(Exception): ...
    monkeypatch.setattr("phenotypic.phenotypicCLI.scan_store_outputs", lambda *_: (_ for _ in ()).throw(Reached()))
    result = CliRunner().invoke(phenotypic_cli, to_argv(form.with_output(measured_output)))   # phenotypicCLI.py:2655
    assert isinstance(result.exception, Reached) and result.exit_code != 2
```
(`scan_store_outputs` is called at `phenotypicCLI.py:3518`; every usage check sits above it.)
- [ ] **Step 2: Run; they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the three files; all pass.**
- [ ] **Step 5: Commit.**

### Task 2: the registry records a measure run (C19)

**Files:**
- Modify: `src/phenotypic/_gui/shell/_runs_registry.py`: `RunRecord.cli_mode`, serialisation;
  `observe_local_exit`'s measure arm. Recent Runs and the banner show `measure` beside the
  execution mode, in their own cell (not through `status_label`).
- Test: `tests/unit/gui/shell/test_runs_registry.py`.

**Interfaces:**
- Consumes: PA1's `complete_with_failures`, `failed_images`; `_generation_failures`
  (`_cli_wait.py:304-322` at `0993c8e06`) for SLURM N (Task 3).
- Produces:
  - `RunRecord.cli_mode: Literal["full", "measure"] = "full"`, read tolerantly.
  - For `cli_mode == "measure"`, `observe_local_exit` maps `0` → `complete`, `3` →
    `complete_with_failures`, `1` → `failed` with the CLI's last output line, and never calls
    `_local_completion_evidence_conflict` (Minor 5's reason). `observe_local_exit` gains
    `detail: str | None = None`; the run console's exit callback (`_callbacks.py:2192-2197`)
    passes the last non-empty line of `handle.stdout_log_path`, which carries stderr too (the
    runner merges it, `_runner.py:226`) (Minor 14).
  - `measure_failure_detail(n_pass: int | None, n_forward: int) -> tuple[int | None, str]`:
    spec PB5's two wordings (I3).

- [ ] **Step 1: Locate a local pass's own failure count on disk** (spec [call], plan finding 2):
  read the terminal-failure journal rows a local `--mode measure` writes
  (`MEASURE_FAILURE_STAGE`, measure work ids) and the epoch they carry. If no per-pass count is
  derivable, the cluster stops and reports; the spec's fallback wording is then used for every
  local exit `3`.
- [ ] **Step 2: Write the failing tests.**

```python
def test_a_measure_record_round_trips_cli_mode(): ...
@pytest.mark.parametrize("rc,status", [(0, "complete"), (3, "complete_with_failures"), (1, "failed")])
def test_measure_local_exits_map_without_the_forward_check(rc, status, monkeypatch, tmp_path):
    spy = spy_on(monkeypatch, "...RunRegistry._local_completion_evidence_conflict")
    assert observe(tmp_path, cli_mode="measure", rc=rc).status == status and spy.call_count == 0

def test_exit_3_with_no_pass_failure_uses_the_forward_wording():           # I3
    assert measure_failure_detail(0, 2) == (2, "measure pass complete; the run still has 2 failed images, which a measure pass does not retry")
def test_exit_3_with_pass_failures_uses_the_pass_wording():
    assert measure_failure_detail(1, 2)[1].startswith("1 images failed to re-measure")
```
- [ ] **Step 3: Run; they fail. Step 4: Implement. Step 5: Run the file and `tests/unit/gui/run_console/`; all pass. Step 6: Commit.**

### Task 3: the observer's measure outcomes (C19, D13)

**Files:**
- Modify: `src/phenotypic/_gui/run_console/_slurm_observer.py` (PA1 T5's hook): read the
  fence's `terminal_status`/`terminal_error`; map measure outcomes.
- Test: `tests/unit/gui/run_console/test_slurm_observer.py`.

**Interfaces:**
- Consumes: `finished_run_outcome` (PA1 T2, with #282's override after PB0);
  `load_slurm_lifecycle`; Task 2's `measure_failure_detail`.
- Produces: spec PB5's four rows. U16 (`closed_by: finalizer` + `terminal_status: "failed"`) →
  `failed`, **terminal**, detail = `terminal_error`. Nothing persists `incomplete`.

- [ ] **Step 1: Write the failing tests.**

```python
def test_a_clean_measure_pass_is_complete(tmp_path): ...
def test_a_measure_pass_with_failures_names_them(tmp_path): ...
def test_a_clean_pass_over_a_forward_run_with_failures_uses_the_forward_wording(tmp_path):   # I3
    assert "0 images failed" not in detail_of(observe_measure(tmp_path, pass_failures=0, forward_failed=2))

def test_a_u16_leftover_is_terminal_failed_and_releases_the_output(tmp_path):            # C3, D13
    reg, rec = u16_measure_record(tmp_path)          # mark_generation_failed(closed_by=finalizer)
    obs = _bound_observer(reg, ...); obs.observe_once()
    got = reg.get(rec.run_id)
    assert got.status == "failed" and "re-run the same command" in got.status_detail
    assert obs.tracked_generation_counts() == (0, 0)   # polling stopped (_slurm_observer.py:409; Minor 12)
    assert reg.allocate(mode="slurm", output_dir=rec.output_dir, rel_path=rec.rel_path,
                        command_digest="x", status="submitting")   # not refused as nonterminal

def test_a_cancelled_measure_fence_is_cancelled(tmp_path): ...
```
- [ ] **Step 2: Run; they fail. Step 3: Implement. Step 4: Run the file and `tests/unit/gui/shell/`; all pass. Step 5: Commit.**

### PB1–PB3 gate

- Per phase: `tests/unit/gui/run_console/`, `tests/unit/gui/shell/`, `tests/integration/gui/`,
  `tests/integration/cli/test_gui_measure_argv.py`, `tests/unit/cli/` (embed-stage) on Slurm.
- **Deep review** (`implementation-test-reviewer`, Opus) over PB1–PB3. Report: `pb1-pb3-review.md`.
