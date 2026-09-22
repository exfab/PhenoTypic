# C4 + C5 gate review — figure-backend-routing

Range: `0e073654..2aa87a4f` (926fa3fc, e3ea2bfd, 2aa87a4f). `HEAD` (a90d74b4) differs
only in `plan.md`, so every `file:line` below is valid at both. Probe and mutation
output was run by the orchestrator against its own hash snapshot and relayed verbatim;
nothing here is quoted from a commit message.

## Verdict

**FAIL — one blocking defect (F1), then pass with fixes.** All three plan-required
checks PASS. The blocker: guard and fence rejections are now swallowed and *written
into the durable failure record*, in a run or bundle the writer no longer owns
(measured, Probes A and B). Seven concrete mutations survive the gate suite (all 166
tests green under each).

## Required checks

### 1. `"qc dependency"` → `"qc"` fixed at the call site — PASS

`_coordinator.py:327-329` passes `lifecycle="qc", log_label="dependent QC"` from
`emit_dependent_qc`; `_emit_aggregate`'s handler (`:353-356`) only forwards
`lifecycle`. The own-handler path at `:330-333` also passes `"qc"`. Pinned by
`test_every_emit_point_records_one_failure_under_a_closed_lifecycle[emit_dependent_qc-via-aggregate]`
and `[emit_dependent_qc-own-handler]` (`tests/unit/plotting/test_coordinator.py`);
orchestrator reports the call-site mutation killed.

### 2. Every record call passes the exception object — PASS

All `record_plot_failure` / `_record_failure*` calls pass a raised exception:
`_coordinator.py:113-119` (`exc`), `:187` (`exc`), `:274` / `:278-283` (`error=exc`),
`:331-333` (`exc`), `:354-356` (`exc`), `:484-492` (`error=error`, taken straight from
`_render_page`'s `list[BaseException]`), forwarded unchanged at `:412-420`;
`_writer.py:317-323` (`error=exc`). No string and no re-wrap reaches a recorder. The
`RuntimeError` built at `_coordinator.py:497-501` is *raised*, not recorded directly;
its message uses `_format_error(errors[0])`, and
`test_a_flat_image_render_failure_records_the_real_exception_class` pins both the class
(`TypeError`) and the single prefix. Caveat, see F3: that `RuntimeError` has no
`from`, which loses the cause chain.

### 3. One-bundle guard — PASS for the image paths, gap on the aggregate path

Every caller passes `plots_base=self._plots_base`: `_coordinator.py:429`
(aggregate), `:455` (multi-page image), `:475` (flat image → `_render_page`).
`test_a_multi_page_plotly_image_plot_writes_exactly_one_bundle` counts across the
whole tree, and the orchestrator reports that dropping `plots_base` on the multi-page
publish was killed. `test_a_single_figure_plotly_image_plot_publishes_html` does the
same for the flat path. **Gap:** dropping it at `:429` (aggregate) survives (M4
below). No coordinator-level test publishes a Plotly aggregate and checks where the
bundle lands.

## Findings, by severity

### F1 (HIGH, blocking): guard and fence rejections are swallowed and recorded as plot failures

**(a) What a rejected guard raises.**
- `publication_guard` (GUI only): `PlotPublicationBlocked` (`_writer.py:42`), from
  `_require_plot_publication` (`_writer.py:398-405`) or `_coordinator.py:503-511`.
- `commit_guard` (CLI/SLURM): the staged guard is `_cli_staged_slurm_worker.py:117-127`,
  which is `generation_publication_guard` (lock with a 300 s timeout,
  `_cli_slurm_lifecycle.py:227` → `ArtifactLockTimeout`) followed by
  `assert_active_epoch` (`_cli_staged_orchestration.py:200-214` →
  `SlurmGenerationInactiveError`). It **never** raises `PlotPublicationBlocked`.
- Inside `_render_page`, the commit guard is entered at `_writer.py:79` (inside
  `_atomic_write`). Only `PlotPublicationBlocked` is re-raised (`:159`, `:176`); a
  fence error falls into the generic `except Exception` at `:161` / `:182` and becomes
  an ordinary render error in `errors`.
- Aggregate and multi-page paths: page commits are swallowed the same way (and
  recorded with lifecycle `"page"`, `_writer.py:316-323`). The manifest commit at
  `_writer.py:385-387` is outside any `try`, so the fence error escapes
  `publish_plot_output`.

**(b) Does `emit_image` swallow it? Yes, at all three sites.** None of
`_cli_process_single.py:350`, `:452` or `_cli_staged_workers.py:583` passes `strict`
any more, so `_coordinator.py:110-119` catches and records everything. Even
`strict=True` would not help on the flat path: `_render_page` has already converted
the fence error into an entry in `errors`, and `:497` raises a *new* `RuntimeError`
with no `from`. Probe B, measured:

```
non-strict returned normally
{"binding_id": "img", ..., "error": "SlurmGenerationInactiveError: epoch superseded", "lifecycle": "image", ...}
{"binding_id": "img", ..., "error": "RuntimeError: plot 'img' produced no file for ds/p1: SlurmGenerationInactiveError: epoch superseded", ...}
strict raised RuntimeError | cause chain finds fence: None
```

Before 2aa87a4f, the flat path's `with publication_commit(self._commit_guard)` sat
outside any handler in `_publish_image_value`, so the fence error left `emit_image`
directly, and `strict=True` let it reach Stage 3's `except SlurmGenerationInactiveError: raise`.

The four aggregate handlers (`:186`, `:272`, `:330`, `:353`) swallow
`PlotPublicationBlocked` too. Before this change that was a log line; now it is a
file write. Probe A (GUI guard returns False), measured:

```
['plots', 'plots/.failures.jsonl', 'plots/.failures.lock']
{"binding_id": "m", "error": "PlotPublicationBlocked: Plot publication blocked because its output snapshot changed.", "lifecycle": "measurements", ...}
```

**(c) Consequence.**
- *Fenced Stage-3 worker.* It does not go on to publish the store:
  `_check_active(active_check)` at `_cli_staged_workers.py:589` runs the same
  `assert_active_epoch` and raises into `except SlurmGenerationInactiveError: raise`.
  But before it gets there, it has appended 2 records (flat path; more for multi-page)
  to `deliverables/plots/.failures.jsonl` of a run whose epoch it no longer owns. The
  append takes no guard (`_failures.py:105-118`), and entries carry no epoch or job id,
  so they cannot be told apart from real plot failures.
- *Lock timeout (`ArtifactLockTimeout`).* The active check does not catch it again.
  The worker finishes the image without the plot, recorded as a plot failure. That is
  consistent with the F3 best-effort policy (spec §3), but it is a lifecycle event
  mislabelled as a figure defect.
- *Full path* (`_cli_process_single.py:350`): same shape. `_check_active` follows at
  `:354`.
- *GUI.* A refresh refused by `OutputMutationGuard` (e.g. a CLI run is in progress,
  `_gui/results_viewer/_mutation_guard.py:121-125`) now writes `.failures.jsonl` and
  `.failures.lock` into the output tree it was just told not to touch. It also
  appends a non-plot-failure line to the log the in-progress CLI run is writing.
  `test_mutation_guard.py`'s tree-unchanged invariant (`:595-601`) is asserted only
  for a direct `publish_plot_output` call, never through the coordinator, which is why
  this is green.

**(d) Minimal fix.**
1. Writer: turn a failure *to enter* the commit guard into
   `PlotPublicationBlocked(...) from exc`. Wrap only the guard's entry, never
   `os.replace`, which is a real write error. Do this in `_atomic_write`
   (`_writer.py:79`) and at the manifest commit (`_writer.py:385`), for example with a
   small `_enter_commit(commit_guard)` helper. `_render_page`'s existing re-raise
   (`:159`, `:176`) then covers fences as well, the plotting layer never imports the
   `_cli` exception, and `slurm_generation_inactive_cause` (which walks `__cause__`)
   still finds the fence from Stage 3's generic handler
   (`_cli_staged_workers.py`, the `except Exception` after `:633`).
2. Coordinator: add `except PlotPublicationBlocked: raise` ahead of each of the five
   `except Exception` handlers (`:110`, `:186`, `:272`, `:330`, `:353`). A blocked
   publication voids the whole refresh, so stopping the loop is correct. The GUI
   callback (`_gui/results_viewer/_callbacks.py:153-161`) already logs and returns
   `no_update`.
3. Also add `from errors[0] if errors else None` at `_coordinator.py:497`, so a strict
   caller keeps the renderer's cause.
4. Tests: a coordinator-level variant of Probe A (`publication_guard=lambda: False` →
   `PlotPublicationBlocked` propagates, tree unchanged), and Probe B (a fencing
   `commit_guard` → `slurm_generation_inactive_cause(exc)` is not None, no
   `.failures.jsonl`).

### F2 (MEDIUM): the flat path leaves a stale sibling from an earlier generation

`_coordinator.py:467-501` writes `<stem>-<hash>.html` and `.png` independently, with
no manifest. When a rerun cannot produce one of them (Chrome absent on this node; or
the plot changed backend), the old file survives next to the new one, and nothing on
disk says which generation each belongs to. Probe C, measured (first run with Chrome,
second without):

```
p1-e27b6fad6926.html 1790036669139266038
p1-e27b6fad6926.png 1790036669137266040   <- first run's PNG, left in place
```

Before C5 the flat path only ever wrote the PNG and always replaced it, so this could
not happen. Realistic trigger: `--mode measure` re-emits image plots
(`_cli_process_single.py:450-456`) on a different node from the original run.
**Fix:** after `_render_page`, remove the absent sibling (`.png` when
`backend == "plotly" and "png" not in files`; `.html` when `backend == "mpl"`) under
`publication_commit(self._commit_guard)`.

### F3 (MEDIUM): preflight classifies by *every* visible figure, but `inspect()` renders only the primary

`_backends.py:196-203` and `_declared_backends` (`:222-241`) union the backends of all
visible `@figure` methods. `PhtPlot.inspect` renders only `_primary_spec()`
(`abc_/plotting/_pht_plot.py:444-461`). Probe D (primary `mpl`, secondary `plotly`),
measured:

```
inspect returns: matplotlib.figure
['Chrome is not available; 1 Plotly plots will publish HTML only, without PNG: Mixed. ...']
```

That announcement is false. The plot publishes a PNG. The reverse case also exists:
a class that overrides `inspect()` while keeping `@figure` methods (the escape hatch
that the spec's §4 plan for `custom_plotter.md` documents) is classified by decorators
it never renders. A class with only `mpl` declarations whose `inspect` returns Plotly
suppresses the probe entirely (`:205`), so its flat-path PNG goes missing with no
announcement and, by the §2 decision, no record. No production class hits this today:
all 27 declarations are `plotly` and no QC class declares a figure. **Fix:** classify
by `_primary_spec().backend` when `inspect` is not overridden (`type(plot).inspect is
PhtPlot.inspect`), and put overridden-`inspect` classes in the undeclared group.

### F4 (LOW): failure records carry no run identity

`_failures.py:93-103` writes `ts, binding_id, plot_class, lifecycle, error` and
nothing else. The file is append-only and never reset, so records from reruns,
fenced generations (F1) and GUI refreshes accumulate side by side with no way to tell
them apart. Consider adding `SLURM_JOB_ID` and a lifecycle epoch when available. A
record lost to a 30 s lock timeout (`sdk_/_file_locking.py:25`) is logged only at
DEBUG (`_failures.py:119-122`). The coordinator's WARNING has already fired, so the
failure itself is not silent.

### F5 (LOW): `strict=True` removal — no test or caller depended on it

`grep strict=` across `src/` finds no remaining `emit_image(..., strict=True)`. The
only test use is `test_image_plot_strict_mode_propagates_publication_failure`
(`test_coordinator.py:115`), which raises from `inspect` and never reaches the flat
path's wrapper. No Stage-3 test exercises a plot failure. Apart from F1, the
semantic change is the intended F3 policy.

### F6 (LOW): flat-path comment overstates why `publish_plot_output` is avoided

`_coordinator.py:460-466` says routing through `publish_plot_output` would lock
`base` for every image and collide on `default.*`. That holds only for
`directory=base`. The multi-page branch right above (`:450-458`) uses
`base / output_stem`, which has per-image lock and no collision. The real costs of
that route are a directory plus `manifest.json` per image, and a changed layout
(`<stem>-<hash>/default.png` instead of `<stem>-<hash>.png`), which breaks the
documented layout and `test_image_plot_output_name_is_stable_for_reruns`. It should
be reworded to say that.

### F7 (LOW): validation message wording

A `PlotBackendUnavailable` surfaces as `"Failed to load pipeline: PlotBackendUnavailable: ..."`
(`_cli_validation.py:76-77`). The pipeline loaded fine; the message misleads. Suggest
catching it before the generic handler and returning its own message.

### Noted only (pre-existing)

- A user `PlotAnalysis` class whose name starts with `_` fails analysis-id validation
  at `registry.get(type(binding.plot).__name__)` (`_coordinator.py:176`). This is now
  *recorded* as an analysis failure rather than only logged.
  `test_coordinator.py`'s `RaisingAnalysisPlot` comment documents the trap.
- `emit_qc` skips a QC reference whose module did not succeed
  (`_coordinator.py:250-251`, `continue`) with no record. The QC runner presumably owns
  that failure; it is out of scope here.

## Surviving mutations (measured: 166 passed under each)

Suite: `tests/unit/plotting/ tests/unit/gui/test_plot_refresh.py tests/gui/results_viewer/test_mutation_guard.py`.

| # | Mutation | Why it survives | Test that would kill it |
|---|---|---|---|
| M1 | `_coordinator.py:477-478` `publication_guard=None`, `commit_guard=None` on the flat `_render_page` call | No test drives `emit_image` with either guard | `emit_image` on a `PlotCoordinator(commit_guard=<recording cm>)`: assert the guard was entered once per written file; plus the F1(d)4 fence test |
| M2 | `:483` `for error in (errors if not files else []):` | Only the everything-failed case (unsupported figure) is tested; a partial failure is not | Plotly flat figure, `chrome_available → True`, `FigureAdapter.save_png` raising `OSError`: expect `.html` present, exactly one record `"OSError: ..."`, no `RuntimeError` record |
| M3 | `:493` `if not files:` → `if errors:` | Same gap | Same test: a partial failure must not raise, and must not add a second record |
| M4 | `:429` delete `plots_base=self._plots_base` in `_publish_aggregate` | No coordinator-level Plotly aggregate checks the bundle location (`test_emit_qc_prelude_failure_names_the_right_binding` publishes one but asserts only `is_dir`) | `emit_measurements` with a Plotly `PlotMeas`: `sorted(tmp_path.rglob("plotly.min.js")) == [plots_dir(tmp_path) / "plotly.min.js"]` |
| M5 | `:481` `FigureAdapter.close(figure)` → `pass` | Nothing counts open matplotlib figures | Flat mpl image plot: `plt.get_fignums()` unchanged across `emit_image` |
| M6 | `_backends.py:236` `for klass in ():` | See below | QC-recipe binding whose `cls` (a `PlotQc` subclass in the test) *inherits* a `@figure(backend="mpl")`: `preflight_plot_backends` must not probe Chrome (`_must_not_probe` as at `test_backends.py:231`); and with `matplotlib` hidden it must raise `PlotBackendUnavailable` |
| M7 | `_backends.py:238` `if name in shadowed:` → `if False:` | No test has an undecorated override shadowing an inherited `@figure` on a QC class | QC `cls` whose parent declares `@figure(backend="plotly")` and which overrides the method undecorated: expect it in the *undeclared* wording, not the "Plotly plots" wording |

**Why `test_preflight_names_qc_recipe_bindings` cannot catch M6.** The orchestrator's
guess is right, and the cause is broader than the mutant. The test uses
`GridOccupancy`, which declares no `@figure` anywhere in its MRO: it overrides
`inspect` (`analysis/qc/_grid_occupancy.py:241`, parent
`analysis/qc/_expected_vs_detected.py:632`). The **unmutated** walk therefore already
returns `set()`, so the binding lands in the undeclared group, and that group also
names it in the line. The test asserts only `"grid-output" in line`, which both the
mutant and the real code satisfy. No QC class in `src/` declares a figure (`grep`),
so the `__mro__` branch (`_backends.py:231-241`) has never run against a declaration
in any test. It needs a synthetic `PlotQc` subclass that declares one, and an
assertion on *which* sentence names the binding.

Also vacuous: the D2 block at the end of
`test_a_multi_page_plotly_image_plot_writes_exactly_one_bundle` asserts
`.failures.jsonl` is absent in a run where nothing fails. It cannot tell a hoisted
record from a per-directory one. To test hoisting, make one page fail and assert the
single record sits at `deliverables/plots/.failures.jsonl`.
