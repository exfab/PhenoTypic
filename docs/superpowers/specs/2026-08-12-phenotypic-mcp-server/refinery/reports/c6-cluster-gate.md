# C6 cluster gate — Tasks 15, 16, 18

**Verdict: NOT sound as committed.** Task 15 (`--screen` + `--slurm` refusal) and
Task 16 (`--slurm key=value`) are correct, well-tested, and their tests genuinely
discriminate — every mutation I constructed against them failed loudly. Task 18
(`finalize_distributed_study`) has **two blockers that compose into one
default-path failure**: a fleet killed by a Slurm timeout leaves orphaned
`RUNNING` Optuna trials; the terminal gate miscounts them as progress and opens;
the winner selector then picks one, because an un-told trial reads back as
`score=0.0` — the *best possible* cost under this module's minimize convention.
`best_params.json` is published with `params={}` and a perfect score in place of
the real winner. No `force`, no error, no warning. Proven end-to-end below.

Scope: `git diff 49b1dc5ac..HEAD -- src/ tests/` at `0744abb4`, branch
`feat/mcp-server`.

**Recommended action:** land Tasks 15 and 16 as they stand. Hold Task 18 for one
fix — filter non-terminal Optuna trials out of both the terminal gate and the
winner selection (`_finalize.py:150` and `:153`), and add an orphaned-`RUNNING`
trial to the test fixture. Nothing user-facing calls this code today, so the
blocker is not shipping right now; it must not be wired in Phase 2 as-is.

## Environment

Both traps the lead flagged were checked, and a third surfaced.

- **`optuna` is present (4.9.0) and the `--slurm` tests really ran.** Baseline
  with `-rs` reported **81 passed, 0 skipped** across the eight touched test
  files. No `skipif(not _OPTUNA)` lane was silently green.
- **`QT_QPA_PLATFORM=offscreen`** was set for every run; no Qt abort.
- **NEW: the working tree was dirty when this gate started.** Another agent was
  concurrently editing `src/phenotypic/tune/_tune_cli/_run.py`,
  `sdk_/_io_constants.py`, `sdk_/__init__.py`, `tune/_study/_optuna_store.py`
  and adding `tune/_study/_storage.py` (a `journal://` storage backend for
  `--slurm`). My first baseline therefore ran against un-reviewed code. **Every
  result below was re-derived in an isolated `git worktree` detached at
  `0744abb4`**, driven with `PYTHONPATH=<worktree>/src` so the editable install
  in `.venv` could not shadow it. The pristine baseline is also 81 passed.
  Note for whoever gates that concurrent work: it changes the default
  `--slurm` storage URL, which is an input to `finalize_distributed_study`.

---

## Blockers

### B1 — finalize publishes a phantom winner from an orphaned RUNNING trial

`src/phenotypic/tune/_tune_cli/_finalize.py:153` (`_headline_winner(store)`),
via `src/phenotypic/tune/_tune_cli/_run.py:822` and
`src/phenotypic/tune/_study/_optuna_store.py:292` (`best()`).

An Optuna trial that was never told — the state a worker leaves behind when
Slurm kills it at the walltime — stays `RUNNING` in the store forever.
`_to_trial` (`_optuna_store.py:264-267`) maps `frozen.value is None` to
`score = 0.0`, and under this module's minimize-cost convention **`0.0` is the
best possible cost**. `best()` filters only `t.failed`, so the orphan outranks
every real trial.

`finalize_distributed_study(out, force=True)` is documented for exactly this
case — "the operator who knows the fleet is gone" (`_finalize.py:119-122`).
Reproduced end-to-end on a study with three real trials (costs 0.50/0.45/0.40)
plus one orphaned `RUNNING` trial:

```
FinalizeResult.winner_trial_number = 3
best_params.json = {
  "trial_number": 3, "score": 0.0, "objectives": {},
  "params": {}, "selection": "single_best"
}
# the real winner is trial 2, score 0.40, params {'0.ignore_zeros': False}
```

**And it does not need `force`.** Composed with B2 below, the default path
reaches it: 5 completed trials + 1 orphaned `RUNNING` = 6 total = budget 6, so
the terminal gate opens, and the orphan is then selected:

```
# finalize_distributed_study(out)   <-- NO force
FinalizeResult.winner_trial_number = 5
best_params.json = {"trial_number": 5, "score": 0.0, "params": {}, ...}
# the real winner is trial 4, score 0.30
```

`params={}` means `prepare_best_from_run` exports the **untuned base pipeline**
as the tuned optimum, reporting a perfect cost. There is no error and no
warning. `FinalizeResult.best_params_written` is `True`.

The local path is unaffected: an in-process engine tells every trial. This
surfaces only on the distributed path C6 built.

**Fix:** exclude non-terminal trials from winner selection in the finalize path
— filter `store.trials` to `COMPLETE`/`PRUNED` before `_headline_winner`, or
give `StudyStore` a `terminal_trials()` and use it in `_finalize.py:153`. A
`score=0.0`-with-empty-`params` trial should never be selectable. Add the
orphaned-`RUNNING` trial to the `_build_study` fixture — no current test can
produce one, which is why this passed.

### B2 — the terminal gate counts a different quantity than the budget it compares against

`src/phenotypic/tune/_tune_cli/_finalize.py:149-150, 276`.

`n_seen = len(list(store.trials))` counts **every** trial: `COMPLETE`, `PRUNED`,
`FAIL` **and** `RUNNING` (`_optuna_store.py:286` calls `get_trials()` with no
state filter). The budget it is compared against is the strategy's `n_trials`,
which `OptunaStrategy.is_exhausted` (`strategy/_optuna.py:441-456`) measures as
`COMPLETE + PRUNED` only — its own docstring says "failed trials do not consume
the budget".

So the gate opens early by `(#failed + #in-flight)` trials. Measured on a study
with 3 complete, 1 failed, 1 running against budget 6: the fleet's real progress
is 3/6, but the gate sees 5/6 and opens at the next trial. On an 8-worker fleet
with a few failures the gate greenlights a study with a dozen trials still to
run — the precise concurrency hazard `_require_terminal_study` exists to
prevent (`_finalize.py:248-256`). Its docstring's claim that "once the fleet has
drained it, no worker asks for more" does not hold in the unit it measures.

**Fix:** compare against terminal, non-failed trials — the same
`COMPLETE + PRUNED` count `is_exhausted` uses — not `len(store.trials)`. Note
`store.completed_count()` is *not* the right substitute either: it is
"non-failed", which still counts `RUNNING`.

Both blockers share one root cause (`store.trials` conflates terminal and
non-terminal states) and one fix closes both.

---

## C6's own claims — replicated

Every load-bearing claim the lead asked me to check was reproduced in the
isolated worktree. C6's report is accurate.

### Task 15, mutation 2 — exactly one test catches the misplaced guard

I reproduced the plan's original guard position *faithfully* (inside the
submission `if slurm:` branch, so `--screen` alone is untouched):

```
M1c guard inside the submission `if slurm:` (plan's original position)
FAILED tests/unit/tune/test_screen_slurm_guard.py::test_the_refusal_writes_no_run_artifacts
1 failed, 3 passed
```

`test_screen_plus_slurm_is_refused` **passes** against the misplaced guard, as
C6 said. `test_the_refusal_writes_no_run_artifacts`
(`tests/unit/tune/test_screen_slurm_guard.py:99`) is the only test that catches
it, and it genuinely does — it asserts the absence of `tuning_spec.json`,
`run.json` **and** `deliverables/`, all of which `run_tuning` writes between the
validator and the submission branch. Deleting the guard entirely fails both
tests, so neither is vacuous.

### Task 16 — the `slurm_`-prefix folding bug, its fix, and the four-name limit

The bug is real. `format_sbatch_directives`
(`src/phenotypic/sdk_/slurm/_sbatch.py:135`) does
`key.replace("slurm_", "")`, so `partition` and `slurm_partition` render the
same directive name. Rendered without the fix:

```
#SBATCH --partition=batch
#SBATCH --partition=epyc
```

Both mutations discriminate, and in opposite directions — the fold is pinned
from both sides:

| Mutation | Result |
|---|---|
| `_canonical_slurm_key` folds nothing | `test_the_unprefixed_spelling_also_wins_over_the_sugar_flag` **FAIL** |
| `_canonical_slurm_key` folds every key | `test_merge_leaves_non_sugar_keys_unprefixed` **FAIL** |

The four-name limit test can fail. Confirmed.

**And Task 16 delivers its actual payoff.** Rendering the merged UCR HPCC
profile end-to-end through `format_sbatch_directives`:

```
#SBATCH --partition=exfab
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --account=exfab
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
```

`--account` reaches a tune fleet, so `exfab` and `preempt` are now reachable.
This was previously impossible.

### Bare `--slurm` still means submit

Pinned by `test_a_bare_slurm_still_means_submit`
(`tests/unit/tune/test_tune_slurm_kv.py:142`, asserting the fleet path was
taken via `captured["output_dir"]`) and
`test_slurm_parses_as_a_repeatable_optional_value` (the parser shape:
`bare.slurm == [None]`, `absent.slurm is None`). The GUI-side assertion was
correctly rewritten from `namespace.slurm is True` — which the `nargs="?"` +
`append` change makes meaningless — to the meaning, through
`_resolve_slurm_request`.

### C8's argv coverage gate — the blindness claim is TRUE

`tests/unit/services/test_argv_coverage.py` passes unchanged (24 tests). The
structural blindness C6 claims is real and verifiable in two lines of the gate:

- `_cli_option_flags()` (`:57`) enumerates `phenotypic_cli.params` — the Click
  command for `python -m phenotypic` **only**. `phenotypic.tune` is argparse and
  has no Click command, so none of its flags can ever enter the comparison.
- `_EMITTING_FUNCTIONS = ("to_argv", "slurm_argv_extension", "to_subprocess_argv")`
  (`:31`) deliberately excludes `tune_run_tail` / `tune_run_argv`, with a comment
  saying so.

So the `(32, 17, 15)` lock is untouchable by anything C6 did to `tune_run_tail`,
and it cannot regress. **But the gap is real** — diffing the tune parser against
what `tune_run_tail` can emit:

```
tune run flags NOT emittable: --nrows, --ncols, --slurm-constraint
```

(`--input`/`--output` are emitted as `-i`/`-o`; `--help`/`--no-screen` are
trivial.) `--nrows`/`--ncols` are the sharp ones: C6's own marker v2 exists so
finalize can reproduce the run's fixed grid, but no service-tier caller can ask
for a fixed grid in the first place, so every MCP- or GUI-launched tune run
records `nrows: null, ncols: null`. Worth a sibling gate keyed on the tune
parser.

### Task 18 idempotence — the strengthened version genuinely bites

Both mutations C6 reported (N1/N2) replicate, and four more:

| Mutation | Result |
|---|---|
| `best_params.json` gains `"finalized_at": datetime.now()` | `test_finalize_is_rerunnable` **FAIL** |
| `_export_trials_parquet` seeds the mirror from the existing `trials.parquet` | `test_finalize_is_rerunnable` **FAIL** |
| step 4 (generalization) dropped | 4 tests **FAIL**, incl. `test_best_params_is_written_last` |
| sentinel never written | `test_interrupted_finalize_leaves_a_marker` **FAIL** |
| sentinel cleaned up on exception ("helpful" try/except) | `test_interrupted_finalize_leaves_a_marker` **FAIL** |
| bare `--slurm` stops meaning submit | 3 tests **FAIL** across two files |

The "does not raise" weak version C6 replaced would have survived N1 and N2.
`_published_bytes` (`tests/unit/tune/test_distributed_finalize.py:312`) snapshots
`deliverables/**` plus `trials.parquet` by bytes, compares the **file set**
first, and carries its own anti-vacuity assertions (`len(before) >= 4` and a
`best_params.json` key check) — so an empty snapshot cannot make it pass
trivially. `assert len(order) == 4` in `test_best_params_is_written_last`
(`:242`) is present as the plan's correction required, and the step-4-dropped
mutation confirms it fires.

---

## False greens

### F1 — marker v2's *read* side could be deleted with the suite green

`src/phenotypic/tune/_tune_cli/_finalize.py:321-323`.

The whole justification for bumping the run marker to v2 is that finalize must
re-scan the plates onto the run's fixed grid. The **write** side is pinned
(`test_run_marker_records_the_fixed_grid` fails when the marker stops recording
them). The **read** side is not: `_build_study`
(`tests/unit/tune/test_distributed_finalize.py:120-121`) hardcodes
`"nrows": None, "ncols": None`, and no test anywhere puts a non-null `nrows`
into a run marker. So `_load_images(images_dir, nrows=..., ncols=...)` is only
ever called with `None`. Confirmed by mutation — replacing that call with a bare
`_load_images(images_dir)`:

```
G1 finalize ignores the marker's nrows/ncols (marker v2 read side)
18 passed, 25 warnings in 40.99s
exit=0
```

The whole read side can be deleted with the suite green.

This is the half of the fix that actually prevents the wrong generalization
verdict C6 correctly identified. **Fix:** parametrize the fixture over
`(None, None)` and `(8, 12)`, and assert the re-loaded plates carry the fixed
grid.

### Hypothesis tested and refuted — the `slurm_args` wiring is NOT deletable

Every one of C6's `--slurm` tests stubs `SlurmExecutor` and asserts the **dict**
handed to it, never a rendered `#SBATCH` line. That is the shape of "wiring that
could be deleted with the suite green", so I mutated
`_execution/_slurm.py:184` to pass `slurm_args={}` into the generated worker
array script — which would silently stop `--account=exfab` reaching the cluster,
reproducing the exact bug Task 16 exists to fix.

**It is caught.** `tests/unit/tune/test_slurm_executor.py:82`
(`test_worker_array_script_carries_sbatch_directives`) renders the real script
and asserts `--partition=short` and `--mem=8G` appear in its text. Run over
`tests/unit/tune/`, `tests/unit/services/` and `tests/unit/gui/tune/`:

```
FAILED tests/unit/tune/test_slurm_executor.py::test_worker_array_script_carries_sbatch_directives
2 failed, 1158 passed, 2 skipped in 392.97s
```

So the chain `merge_slurm_args → _submit_slurm_fleet → SlurmExecutor →
format_sbatch_directives` is pinned end-to-end. It is pinned only through the
legacy keys (`partition`, `mem`), not the new ones (`account`, `qos`), but they
share the same code path, and I confirmed the new keys render correctly by
direct call. No action needed.

That 1158-test run is also the broad regression signal for this change: the only
failures were the injected mutation and one residue from a harness collision I
caused (see Method).


---

## Non-blocking findings

### N1 — the GUI renders a `--screen --slurm` plan the CLI now refuses

`src/phenotypic/_services/tune_spec.py:554` (`build_tune_command`).

The GUI exposes the SLURM/Local radio (`gui/tune/_layout.py:422`) and the
two-round-screening checkbox (`:429`) independently, and `build_tune_command`
adds no issue for the combination. Verified:

```
issues: ()
argv: (..., '-m', 'phenotypic.tune', 'run', ..., '--slurm', '--screen')
```

Before Task 15 that command ran (silently unscreened). Now it raises
`ValueError` from `_validate_slurm_request`. Refusing is the correct decision —
but the refusal lands as a runtime crash after launch instead of a preflight
issue in the form. **Fix:** one line in `build_tune_command` —
`if slurm and screen: issues.append("Two-round screening is not supported on SLURM.")`
— plus a test. Cheap, and it makes Task 15's decision visible where the user
makes it.

### N2 — `mem_gb` + `--slurm-mem` still emits `#SBATCH --mem` twice

`src/phenotypic/tune/_tune_cli/_run.py:753` (`_canonical_slurm_key`).

The fold covers the four sugar names. `mem_gb` is deliberately excluded — and
correctly so, since folding it would render `--mem-gb`. But `mem_gb` and
`slurm_mem` *also* collapse to the same directive in
`format_sbatch_directives`, so `--slurm-mem 8G --slurm mem_gb=16` produces:

```
#SBATCH --mem=8G
#SBATCH --mem=16G
```

Slurm takes the last, so the pair happens to win — matching the intended
precedence by luck, not design. The `_canonical_slurm_key` docstring implies
this case is handled ("every other key passes through verbatim so the special
cases … keep their meaning"); it keeps its meaning but not its uniqueness.
Requires two contradictory user flags, so low severity. The same docstring is
also wrong about `time`: `time` **is** in `_SLURM_SUGAR_KEYS` and does get
folded to `slurm_time` (harmlessly — `format_sbatch_directives` treats the two
spellings identically).

### N3 — `finalize_distributed_study` has no caller, and the bug it fixes is still shipped

Confirmed: zero production callers, not exported from `tune/_tune_cli/__init__.py`
or `tune/__init__.py`, imported only by its own test file. C6 flagged this
honestly and the plan (Task 18) does specify only the library entry point.

**Does it need a guard now?** A guard on unused code is the wrong instrument —
the thing to notice is that the *user-visible* bug the module docstring opens
with is untouched: `prepare_best_from_run` still raises `FileNotFoundError` on
every `--slurm` study, exactly as `_finalize.py:7-9` describes. Task 18 ships
the machinery without the fix. A ~15-line `finalize` subcommand on the tune
argparse CLI would close it now and give the module a caller that a test can
exercise end-to-end. Recommend that over a guard. If Phase 2 is genuinely
imminent, leaving it is defensible — but say so deliberately rather than by
omission.

### N4 — the `--slurm` token-swallowing hazard is benign

`nargs="?"` means `--slurm` swallows a following non-option token, which the
help text warns about (`tune/__main__.py:100-103`). The only positional on the
`run` subcommand is `spec`, so `run -i imgs --slurm spec.json` fails with
argparse's "the following arguments are required: spec" (exit 2), not a silent
misroute. Malformed pairs are also a clean usage error —
`test_a_malformed_pair_is_a_usage_error_not_a_traceback` pins exit code 2 and
the `KEY=VALUE` message, and `click.BadParameter` is correctly converted through
`parser.error`. No action needed.

---

## What is sound

Stated plainly, because most of this change is good work:

- **Task 15 is correct and correctly placed.** The guard lives in
  `_validate_slurm_request` (`_run.py:730-741`), above the `deliverables/`
  mkdir, the spec echo and the run marker, and the one test that can tell the
  difference does tell the difference.
- **Task 16 is correct and delivers real capability.** `--account`, `--qos`,
  `--cpus-per-task`, `--gpus-per-node` now render into a tune worker script;
  `exfab` and `preempt` were previously unreachable from a tune fleet. The
  `slurm_`-prefix folding bug C6 found unprompted is real, the fix is right, and
  it is pinned from both directions.
- **`merge_slurm_args` precedence is right and single-sourced.** Merging moved
  up into `run_tuning` so `_submit_slurm_fleet` has one SLURM input and no
  precedence rules of its own — a genuine simplification, not just a move.
- **Marker v2 (`nrows`/`ncols`) is a real correctness fix**, not scope creep:
  re-scanning the plates on a different grid than the search used would change a
  `Grid_RowNum`-grouped scorer's held-out verdict with no error.
- **The Task 18 fixtures are honest.** A real Optuna SQLite study, real `Trial`
  rows, a real resolved spec, a real marker, real plate PNGs on disk. Only the
  ordering and interruption tests monkeypatch, and only the step each is about.
- **The idempotence test earns its claim.** Every published byte plus the file
  set, with anti-vacuity guards. It catches both plausible bugs C6 named.
- **C6's report is accurate.** Every claim I spot-checked reproduced. The
  self-reported limitation (no caller) is real and was not hidden.

## Method

All results were produced in a `git worktree` detached at `0744abb4`, isolated
from the concurrently-dirty main tree, with `PYTHONPATH` pointed at the
worktree's `src` so the editable install could not shadow it. Every mutation was
applied to a file, run, and reverted by a harness that restores from an
in-memory backup in a `finally`. Runs used `QT_QPA_PLATFORM=offscreen` and
`-p no:randomly`.

One correction to my own process: I launched the Task 18 mutation batch twice
concurrently, so two harness instances interleaved backup/restore on the same
file and produced inconsistent counts (and left one mutation applied, which is
the second unexplained failure in the W1 run above). **I discarded both runs and
re-ran that batch alone**; only the clean numbers are reported. The worktree was
`git checkout`-restored to `0744abb4` and verified byte-clean before and after.
