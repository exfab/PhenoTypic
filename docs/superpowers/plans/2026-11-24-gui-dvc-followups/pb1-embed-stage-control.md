# PB1 · One resolver; the `--embed-stage` control and the GPU section

**Spec:** PB1, PB2; review I9, I10

Anchors at `0993c8e06`; PB0 T1 Step 4 re-finds them.

---

**Goal:** The form's GPU section, Auto hint and sent flags come from the CLI's own resolver.

**Depends on:** PB0.

### Task 1: `resolve_embed_stage_for_pipeline` (C17)

**Files:**
- Modify: `src/phenotypic/_cli/_cli_embed_stage.py`: factor the pipeline half of
  `resolve_embed_stage_for_config` (`:390-468`) into a public function;
  `resolve_embed_stage_for_config` calls it.
- Test: `tests/unit/cli/test_cli_embed_stage.py`, `tests/unit/cli/test_preflight_embed_stage.py`.

**Interfaces:**
- Consumes: `resolve_embed_stage` (`:132-174`), `pipeline_embedders` (`:88`), and the private
  `_cpu_pinned_key`, `_unstageable_embedder`, `_has_gpu_detector` (`:109, 124, 471`).
- Produces:
  `resolve_embed_stage_for_pipeline(pipeline: ImagePipeline, requested: EmbedStageRequest, *, mode: "_cli_preflight.RunMode", is_slurm: bool, gpu_slurm_given: bool) -> tuple[EmbedStage, list[str]]`:
  `RunMode` here is `_cli_preflight.RunMode` (`0993c8e06 _cli_embed_stage.py:74`), not the
  registry's `RunMode` (`shell/_runs_registry.py:88`); the notes are a `list[str]`, as today
  (`:390-468`) (Minor 11).
  the stage and the notes `resolve_embed_stage_for_config` prints today (U9, O3, O5 texts,
  verbatim).

- [ ] **Step 1: Write the failing tests.**

```python
@pytest.mark.parametrize("pipe,req,mode,slurm,gpu,stage", RESOLUTION_TABLE)   # README table of #282
def test_the_pipeline_resolver_matches_the_config_resolver(pipe, req, mode, slurm, gpu, stage):
    assert resolve_embed_stage_for_pipeline(pipe, req, mode=mode, is_slurm=slurm, gpu_slurm_given=gpu)[0] == stage
    assert resolve_embed_stage_for_pipeline(...) == resolve_embed_stage_for_config(make_exec_config(...), pipe)

def test_a_cpu_pin_note_is_verbatim(): assert "is pinned to device='cpu', so --embed-stage auto resolves to inline" in notes
def test_measure_mode_ignores_a_gpu_detector_alone(): assert stage == "inline"     # §4.1
```
- [ ] **Step 2: Run; they fail.**
- [ ] **Step 3: Implement the factoring;** `resolve_embed_stage_for_config`'s output is
  unchanged.
- [ ] **Step 4: Run the file, `tests/unit/cli/test_cli_preflight_ordering.py` and every test
  importing `_cli_embed_stage`; all pass.**
- [ ] **Step 5: Commit.**

### Task 2: the control, visibility, hint and no hidden flags (C17)

**Files:**
- Modify: `src/phenotypic/_gui/run_console/_state.py:67-195, 515-603` (`embed_stage` field,
  `run_state_from_json`/`run_state_to_json` (`0993c8e06 _state.py:121, 151`), `to_argv`); `_ids.py:136-145` (`RC_RADIO_EMBED_STAGE = "rc-radio-embed-stage"`,
  `RC_EMBED_STAGE_HINT = "rc-embed-stage-hint"`, comment); `_form.py:509-567` (header "GPU
  stages", shards help); the always-visible options block beside `--retry-failures`.
- Modify: `src/phenotypic/_gui/run_console/_callbacks.py`: `_staged_gpu_capability`
  (`:255-277`) → `_gpu_section_visible(...)`; `show_staged_gpu_controls` (`:1840-1865`); the
  positional controls (`:411, 515-516, 552-553`); the textarea normalisation (`:1964-1968`).
- Modify: `src/phenotypic/_gui/run_console/_slurm.py:197-210` (`_build_subprocess_argv`).
- Modify: `docs/source/tutorials/gui/05_run_slurm.md:22, 44` (wording).
- Test: `tests/unit/gui/run_console/test_state.py`, `test_form.py`, `test_slurm.py`,
  `tests/integration/gui/test_run_console_callbacks.py`.

**Interfaces:**
- Consumes: Task 1; `adopt_recorded_embed_stage` (`_cli_embed_stage.py:284-347`);
  `load_processing_state` (`_cli_state_management.py:249`); `EMBED_STAGE_CONFIG_KEY` (`:83`).
- Produces:
  - `RunConsoleState.embed_stage: Literal["auto", "gpu", "inline"] = "auto"`; `to_argv` emits
    `--embed-stage <v>` only when `v != "auto"`.
  - `RunConsoleState.gpu_section_shown: bool = False`, **set server-side** at the launch seam
    (`_state_from_action_controls`) from `_gpu_section_visible(...)`, never read from the
    browser and never persisted in presets. `_build_subprocess_argv` emits
    `--gpu-slurm`/`--gpu-shards` only when it is `True`.
  - `normalised_gpu_tokens(text: str | None) -> tuple[str, ...]` (the stripped non-empty lines);
    the one producer of both the sent tokens and `gpu_slurm_given`.
  - `_gpu_section_visible(pipeline: ImagePipeline | None, *, exec_mode: str, cli_mode: str, embed_stage: str) -> bool`:
    `exec_mode == "slurm"` and (`cli_mode == "full"` and a `GpuDetector`) or
    `resolve_embed_stage_for_pipeline(..., gpu_slurm_given=True)[0] == "gpu"` (I10).
  - `embed_stage_hint(pipeline, *, exec_mode, cli_mode, embed_stage, gpu_tokens, output_dir) -> str`:
    the resolver, then on a full-mode continuation `adopt_recorded_embed_stage` (I9); the text
    is the final stage plus the CLI's note.

- [ ] **Step 1: Write the failing tests.**

```python
def test_auto_emits_no_flag(): assert "--embed-stage" not in to_argv(state(embed_stage="auto"))
def test_gpu_and_inline_emit_the_flag(): assert to_argv(state(embed_stage="gpu"))[-2:] == ["--embed-stage", "gpu"]
def test_an_old_preset_loads_as_auto(): assert run_state_from_json(OLD_PRESET).embed_stage == "auto"   # Minor 12

def test_a_dino_only_pipeline_shows_the_section_on_slurm():
    assert _gpu_section_visible(dino_pipe(), exec_mode="slurm", cli_mode="full", embed_stage="auto")
def test_a_cpu_pinned_dino_pipeline_hides_it():
    assert not _gpu_section_visible(dino_pipe(device="cpu"), exec_mode="slurm", cli_mode="full", embed_stage="auto")
def test_local_mode_never_shows_it(): ...
def test_inline_hides_it_for_a_dino_only_pipeline(): ...

def test_hidden_tokens_are_never_sent():                                        # I10
    argv = _build_subprocess_argv(state(gpu_slurm_args=("partition=exfab",), gpu_section_shown=False))
    assert "--gpu-slurm" not in argv

def test_the_hint_shows_adoption_on_a_pre_282_continuation(tmp_path):           # I9
    out = continuation_with_recorded_stage(tmp_path, None)                       # read as inline
    hint = embed_stage_hint(dino_pipe(), exec_mode="slurm", cli_mode="full", embed_stage="auto",
                            gpu_tokens=("partition=exfab",), output_dir=out)
    assert hint.startswith("Auto → inline")

def test_the_embed_control_is_disabled_without_an_embedder(...): ...
def test_the_section_header_reads_gpu_stages(): assert "GPU stages" in render_form()
```
- [ ] **Step 2: Run; they fail.**
- [ ] **Step 3: Implement.** The pipeline is loaded once per callback with
  `load_pipeline_or_none` (`_cli_embed_stage.py:483`), memoised on `(path, mtime_ns)` like
  the classifier.
- [ ] **Step 4: Run the four test files; all pass.**
- [ ] **Step 5: Commit.**
