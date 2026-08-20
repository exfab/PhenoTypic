# C6 — Tasks 15, 16, 18: implementation report

**Branch:** `feat/mcp-server` · **Base:** `b753df3c6` · **Tree:** `/bigdata/iwheeldonlab/anguy344/PhenoTypic`

| Commit | Task |
|---|---|
| `2228e9b17` | 15 — refuse `--screen` with `--slurm` |
| `042c2b185` | 16 — `--slurm key=value` on the tune CLI |
| `5ce78d3af` | 16 follow-up — the GUI drift gate this change tripped |
| `e86420fbf` | 18 — re-runnable distributed finalize |
| `3b1e5257a` | 18 follow-up — idempotence pinned to every published byte |

This report is about **what would have to be true for the tests to be lying**. The
diff describes what was built; the gate output describes that it is green. Neither
shows whether the tests can fail.

---

## 0. Two environment facts that shaped the work

**`optuna` was not installed.** Every `--slurm` test in `tests/unit/tune` was
skipping. The first Task 15 run reported *"1 passed, 3 skipped"* — a green that
proved nothing. Installed additively with `uv pip install optuna` (does not touch
the lock); a later `uv sync` removes it again. **Any C6 verification run without
optuna present is meaningless**, because the three tests that matter are exactly
the ones that skip.

**`tests/unit/sdk_` aborts on a head node.** `test_label_editor_widget.py` needs a
display: `Fatal Python error: Aborted`, `qt.qpa.xcb: could not connect to display`.
`QT_QPA_PLATFORM=offscreen` fixes it. This is environmental and unrelated — but it
**silently truncated my first full sweep**, and the `test_command.py` regression
described in §2 nearly escaped because of it. Any sweep that includes `tests/unit/sdk_`
needs that env var, or the run dies before reporting.

**Process note, for the next agent in a shared tree.** I ran `git commit --amend`
after HEAD had moved under me and rewrote another agent's commit (`86d9138ad`,
C8-gate2's). Caught immediately: the amend only *added* a file and *appended* to the
message; `git reset 86d9138ad` restored it and `git diff 86d9138ad HEAD` was empty.
My change went into its own commit `5ce78d3af` instead. I also used `git stash` once
for a mypy baseline before realising the same hazard applies. **Neither `--amend` nor
`stash` is safe in a tree another agent is committing to.**

---

## 1. Task 15 — `--screen` + `--slurm`

Guard placed in `_validate_slurm_request(..., screen: bool = False)` per correction B9,
not at the `if slurm:` early return where the task body put it.

### Mutations run

| # | Mutation | Result |
|---|---|---|
| 1 | Guard deleted entirely | `test_screen_plus_slurm_is_refused` **FAIL**, `test_the_refusal_writes_no_run_artifacts` **FAIL** |
| 2 | Guard moved to the task's original position (immediately above `if slurm:`) | **only** `test_the_refusal_writes_no_run_artifacts` **FAIL** — the other three still pass |
| 3 | Local `if screen:` branch in `run_tuning` disabled | `test_screen_alone_still_works` **FAIL** |

**Mutation 2 is the point of the task.** The misplaced guard still raises, so a test
that only checks `pytest.raises(ValueError)` passes against it. What it does not do is
raise *before* the `deliverables/` mkdir, the `tuning_spec.json` echo and the
`run.json` marker — so a refused run leaves an output directory the GUI shell
classifier reports as a live tune output. `test_the_refusal_writes_no_run_artifacts`
exists solely to catch that, and it is the only test that does.

### Where the plan was wrong about its own tests

All three of the task's tests were unrunnable or wrong, as B9 said:

- All three omitted the **required positional `images`**, so they would have raised
  `TypeError` before reaching the assertion under test.
- `test_screen_alone_still_works` ran a **real local tuning study**. Rewritten to stub
  `ScreeningController` and assert the *routing* (screen → the two-round freeze, not
  the plain engine), with a `_FakeEngine` that raises if the plain path is taken —
  which is the anti-vacuity guard mutation 3 trips.

---

## 2. Task 16 — `--slurm key=value`

### Did this touch the coverage gate? No.

**I did not modify `tests/unit/services/test_argv_coverage.py`, the `_DENIED` list, or
the `(32, 17, 15)` count lock.** Nothing was loosened.

The gate is structurally out of scope for this change, and that is by its own design,
not by luck:

- `_EMITTING_FUNCTIONS = ("to_argv", "slurm_argv_extension", "to_subprocess_argv")` —
  the AST walk covers only those three. I edited `tune_run_tail` / `tune_run_argv`,
  which live in the same module and are deliberately excluded (the gate's own comment
  says so: *"Restricting the walk to these three is what keeps `phenotypic.tune`'s
  flags … out of the comparison"*).
- `_cli_option_flags()` enumerates `phenotypic_cli.params` — the **forward** CLI. Every
  flag I changed is on `phenotypic.tune`'s argparse parser, which that function never
  reads.

So the gate passing is not evidence about my change either way. It is unchanged and
green (24 passed) because my change is outside its subject.

### A different gate did fire — correctly

`tests/unit/gui/tune/test_command.py::test_render_parses_through_the_real_cli_parser`
parses the rendered command through the **real** argparse parser rather than a
hand-copied flag list. Its `assert namespace.slurm is True` broke, because the value is
now `[None]`.

Fixed in `5ce78d3af` by asserting through `_resolve_slurm_request` — the function the
CLI itself uses to turn that namespace into `(submit, slurm_args)` — rather than
re-encoding the new list shape. That is a stronger claim than the boolean identity it
replaces, and it keeps holding if the parser shape changes again. Mutation C below
covers it, since it exercises the same function.

**This is the one that nearly escaped**: the truncated sweep (§0) hid it, and it only
surfaced on a re-run with `QT_QPA_PLATFORM=offscreen`.

### Bare `--slurm` still means "submit" — what pins it

Two tests, at two levels:

1. `test_a_bare_slurm_still_means_submit` (`tests/unit/tune/test_tune_slurm_kv.py`) —
   drives `main([... "--slurm"])` end to end and asserts `_submit_slurm_fleet` was
   reached (`captured["output_dir"] == out`) with `slurm_args == {}`. It asserts the
   fleet path was **taken**, not merely that parsing succeeded.
2. `test_slurm_parses_as_a_repeatable_optional_value` — pins the parser shape itself:
   bare → `[None]`, two pairs → `["a=1", "b=2"]`, absent → `None`.

And the negative side, which matters just as much: `test_no_slurm_flag_runs_locally`
asserts that **absence** of the flag is not read as an empty profile — a local run must
not submit.

**Mutation C** (below) changes `_resolve_slurm_request` so an empty pair list returns
`False` instead of `True` — i.e. bare `--slurm` stops meaning submit. Both
`test_a_bare_slurm_still_means_submit` and `test_legacy_flags_still_work` fail.

### Mutations run

| # | Mutation | Result |
|---|---|---|
| A | Merge precedence inverted (sugar flags overwrite explicit `key=value`) | `test_explicit_kv_wins_over_the_sugar_flag`, `test_the_unprefixed_spelling_also_wins_over_the_sugar_flag` **FAIL** |
| B | `_canonical_slurm_key` made identity (no folding) | `test_the_unprefixed_spelling_also_wins_over_the_sugar_flag` **FAIL**; `test_merge_leaves_non_sugar_keys_unprefixed` correctly still passes |
| C | Bare `--slurm` no longer implies submit | `test_a_bare_slurm_still_means_submit`, `test_legacy_flags_still_work` **FAIL** |
| D | `click.BadParameter` not wrapped into `parser.error` | `test_a_malformed_pair_is_a_usage_error_not_a_traceback` **FAIL** |
| E | `_services/argv.py`: pairs emitted regardless of `slurm` | `test_a_profile_never_turns_a_local_run_into_a_cluster_job` **FAIL** |
| F | `_services/argv.py`: bare flag emitted *alongside* the pairs | 4 tests **FAIL** |
| G | `_services/argv.py`: empty key/value skip removed | `test_empty_keys_and_values_are_skipped` **FAIL** |

### A bug the plan did not cover

`format_sbatch_directives` (`sdk_/slurm/_sbatch.py:135`) strips a leading `slurm_` and
turns remaining underscores into dashes. So `partition=` and `slurm_partition=` render
the **same** `#SBATCH --partition`. Left as two dict keys, a run carrying
`--slurm-partition batch --slurm partition=epyc` emits that directive **twice with
contradictory values**.

This is not hypothetical: `partition=` is exactly what the forward CLI's own emitter
`slurm_argv_extension` produces, so it is the spelling a copied command line arrives
in. `merge_slurm_args` folds the four sugar names onto the `slurm_`-prefixed key.
Folding is deliberately limited to those four — `mem_gb` is a separate special case in
`format_sbatch_directives` (`→ --mem=<N>G`), and folding it would render `--mem-gb=8`.
Mutation B is what proves the folding is load-bearing; `test_merge_leaves_non_sugar_keys_unprefixed`
is what stops someone "fixing" it by folding everything.

---

## 3. Task 18 — the distributed finalize

### I4: fixed the test, did not implement around it

The task's `assert order.index("_finalize_best_params") == 2` passes even if step 4 is
silently dropped. I did not take that on faith — **I demonstrated it**:

> With `_finalize_generalization_from_disk` short-circuited to `return False`
> (step 4 silently dropped):
> - the plan's original assertion, `order.index("_finalize_best_params") == 2` → **PASSED**
> - my replacement, `len(order) == 4` → **FAILED**

My version asserts `len(order) == 4` **first**, then the full four-element list in
order. The `len()` check comes first deliberately: on a three-element list the
index assertion is still satisfiable, so ordering the assertions the other way
around would report a confusing failure instead of the real one.

### Idempotence — the weak claim and the real one

The lead's question was the right one. My first version of `test_finalize_is_rerunnable`
asserted only that a second call does not raise, plus a byte-comparison of
`best_params.json`. **That is the weak claim**, and it would have passed against a
finalize that appended to `trials.parquet`, that stamped a fresh timestamp into the
completion marker, or that stopped writing one of the other files entirely.

Strengthened in `3b1e5257a`. The test now snapshots **every published artifact by
bytes** — all of `deliverables/` plus `trials.parquet` — and compares both the contents
and the **file set** (a value-only comparison cannot see a file that stopped being
written). `.pht-tune-cache/` is excluded because the study DB and its SQLite WAL are
engine state, and re-reading a study legitimately touches them. The snapshot is guarded
against being empty, which is the shape a test takes when it cannot fail.

Verified against two plausible bugs — **neither of which the previous version caught**:

| # | Mutation | Result |
|---|---|---|
| N1 | `best_params.json` payload gains `"finalized_at": datetime.now(...)` | `test_finalize_is_rerunnable` **FAIL** |
| N2 | `_export_trials_parquet` seeds the mirror from the existing parquet (the natural "resume" mistake — duplicates every trial) | `test_finalize_is_rerunnable` **FAIL** |

All published bytes, `trials.parquet` included, are stable across runs.

### Mutations run

| # | Mutation | Result |
|---|---|---|
| M1 | Step-4 rescan short-circuited (`return False`) | `test_the_generalization_report_is_written_from_a_rescan`, `test_a_missing_image_directory_is_reported_not_swallowed`, `test_best_params_is_written_last` **FAIL** |
| M2 | `_finalize_best_params` moved to first | `test_best_params_is_written_last` **FAIL** |
| M3 | Terminal gate removed | `test_refuses_a_running_study` **FAIL** |
| M4 | Sentinel cleared in a `finally` (interruption hidden) | `test_interrupted_finalize_leaves_a_marker` **FAIL** |
| M5 | Winnerless warning dropped | `test_a_winnerless_study_reports_instead_of_leaving_a_silent_hole` **FAIL** |
| M6 | Headline winner = `trials[0]` instead of `_headline_winner(store)` | `test_best_params_names_the_actual_winner` **FAIL** |
| M7 | Step 1 dropped entirely | `test_finalize_writes_what_the_slurm_branch_skipped`, `test_best_params_is_written_last` **FAIL** |
| M8 | Run marker stops recording `nrows`/`ncols` | `test_run_marker_records_the_fixed_grid` **FAIL** |
| N1, N2 | see idempotence above | `test_finalize_is_rerunnable` **FAIL** |

**M6 is the anti-"a file exists" mutation.** Fixture trial costs *descend* with trial
number, so the last recorded trial is the winner and a test can tell a real winner from
"whatever came first". Without that, `best_params.json` existing would have been the
whole assertion.

### What is NOT stubbed

The fixtures build a **real** Optuna SQLite study with real `Trial` rows, a real
resolved `tuning_spec.json`, a real `run.json` marker, and **real plate PNGs on disk**
under the `images_dir` the marker records. The generalization test genuinely re-scans
them from disk. Only the ordering and interruption tests monkeypatch anything, and only
the specific step each is about.

### Known limitation, stated rather than hidden

`finalize_distributed_study` has **no CLI or GUI caller**. The task specifies only the
library entry point and Phase 2 wires it. Flagging it explicitly because *"wiring that
could be deleted with the suite green"* is on the known-false-green list: here there is
no wiring to delete, by design — but nothing user-facing reaches this code yet, and no
test would notice if Phase 2 never wired it.

---

## 4. Where the plan disagreed with the code

| Plan claim | Reality |
|---|---|
| Task 16 line numbers into `_services/argv.py` (`tune_run_tail` at `:417`, `tune_run_argv` at `:513`) | Stale — C8's restructure moved both (`:556` and `:654` on `b753df3c6`). Located by symbol, not by line. |
| Task 18: *"`_finalize_pareto_outputs` … has no call sites outside `run_tuning`"* | Wrong, and the corrections already flagged it. It has one in `tests/unit/tune/test_run_tuning_pareto.py`. The four steps are therefore **imported** into `_finalize.py`, not moved — one definition run by both paths is the only thing keeping a local and a distributed output identical. |
| Task 18 / B7: *"the run marker records what is needed to re-scan"* | **Incomplete — this is a correctness gap, not a nitpick.** The marker recorded `images_dir` but not `nrows`/`ncols`. Re-loading without the run's fixed grid assigns a different `nrows × ncols` than the search used, and a `Grid_RowNum`/`Grid_ColNum`-grouped scorer then scores the held-out pass against a grid that never existed — a wrong generalization verdict with no error. Marker bumped to **v2**; no consumer branches on `version` and every reader uses `.get()`, so v1 markers still load. Mutation M8 pins it. |
| Task 18: `_run.py:744` docstring refers to a `--recompile` finalize | That flag has never existed. Corrected to name `finalize_distributed_study`. |
| Task 15/16 test bodies | Task 15's three tests all omitted the required positional `images`; Task 16's used `CliRunner().invoke(cli, ...)` against a CLI that is argparse with no `cli` object. Both rewritten per the corrections. |
| Task 16 Step 3: merge inside `_submit_slurm_fleet` | Superseded by correction B4 — merged in `run_tuning`; `_submit_slurm_fleet` lost its four `slurm_*` params for one `slurm_args` dict. Three call sites in `tests/unit/tune/test_slurm_passthrough.py` updated. |

One thing added that no plan step asked for: `FINALIZE_IN_PROGRESS_MARKER` +
`tune_finalize_marker_path` in `sdk_/_io_constants.py`, so the sentinel path is
resolved through an `sdk_` helper per the never-hand-join rule rather than being
joined inline in `_finalize.py`.
