# PB2 · Stage-4 refusals in `rc-staged-gpu-refusal`

**Spec:** PB3

---

**Goal:** Every refusal the CLI would raise for the form's argv, that the GUI can know before
launch, shows the CLI's own message in the existing alert, with Run disabled.

**Depends on:** PB1 (the resolver, the control, `normalised_gpu_tokens`).

### Task 1: `_launch_refusal` (C18)

**Files:**
- Modify: `src/phenotypic/_gui/run_console/_callbacks.py`: replace `_staged_gpu_capability`'s
  refusal half (`:255-277`) with `_launch_refusal`; `show_staged_gpu_controls`
  (`:1840-1865`) gains the inputs; the launch seam (`:2039-2041`) calls it.
- Test: `tests/integration/gui/test_run_console_callbacks.py` (`:831-1143` region holds today's
  refusal tests), `tests/unit/gui/run_console/test_form.py`.

**Interfaces:**
- Consumes: `refuse_unstageable_embedders(pipeline, requested, resolved)` (`:190-226`),
  `refuse_embed_stage_switch` (`:229-281`), `adopt_recorded_embed_stage` (`:284-347`),
  `UnstageableGpuDetectorError`, `pipeline_requires_gpu` (`_cli_validation.py:442`); PB1's
  resolver and `normalised_gpu_tokens`.
- Produces:
  `_launch_refusal(pipeline_path: str | None, *, cli_mode: Literal["full", "measure"], exec_mode: Literal["local", "slurm"], embed_stage: str, gpu_slurm_tokens: tuple[str, ...], output_dir: Path | None) -> str | None`.
  Order: the unstageable-`GpuDetector` check (full mode only); R2; R3/U12 (full mode, output
  holds a processing state); PB3's measure refusals (measure mode; the hook is a no-op until
  PB3 T1). Catches `click.UsageError` and returns `exc.message`. An unreadable pipeline returns
  `None` (unchanged rule, `:262-277`).

- [ ] **Step 1: Write the failing tests.**

```python
def test_r2_gpu_with_a_cpu_pinned_embedder_is_shown(): assert "pinned to device='cpu'" in alert(dino(device="cpu"), embed_stage="gpu")
def test_r2_gpu_with_a_stageless_embedder_is_shown(): assert "has no GPU Stage-4 encoder" in alert(...)
def test_r3_an_explicit_switch_on_a_continuation_is_shown(tmp_path):
    out = continuation_with_recorded_stage(tmp_path, "gpu")
    assert "this run was started with --embed-stage gpu" in alert(dino(), embed_stage="inline", output_dir=out)
def test_u12_auto_with_recorded_gpu_and_no_profile_is_shown(tmp_path):
    out = continuation_with_recorded_stage(tmp_path, "gpu")
    assert "re-run with the run's --gpu-slurm options" in alert(dino(), exec_mode="slurm", gpu_slurm_tokens=(), output_dir=out)
def test_r3_is_never_asked_in_measure_mode(tmp_path, monkeypatch):
    spy = spy_on(monkeypatch, "...refuse_embed_stage_switch")
    alert(dino(), cli_mode="measure", embed_stage="inline", output_dir=continuation_with_recorded_stage(tmp_path, "gpu"))
    assert spy.call_count == 0
def test_the_gpu_detector_refusal_is_full_mode_only(): ...
def test_run_is_disabled_while_the_alert_is_open(): ...   # update_run_disabled, :2839-2855
def test_the_seam_refuses_a_racing_click_before_allocating(): ...
def test_an_unreadable_pipeline_stays_silent(): ...       # unchanged
```
- [ ] **Step 2: Run; they fail.**
- [ ] **Step 3: Implement.** `processing_state.json` is read with `load_processing_state`,
  memoised on `(path, mtime_ns)` (spec R7).
- [ ] **Step 4: Run the two files and `tests/unit/gui/run_console/`; all pass.**
- [ ] **Step 5: Commit.**
