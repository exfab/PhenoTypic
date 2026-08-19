---
name: run-phenotypic-test
description: How to actually run PhenoTypic's pytest suite on the UCR HPCC without getting a wrong or misleading answer. Use before running any non-trivial pytest invocation here — especially the full tests/unit suite, anything on a compute node, anything headless, or when establishing a baseline to compare a change against.
---

# Running the test suite here

Four traps in this repo turn a test run into a **wrong answer** rather than a
slow one. Each below was paid for once; none is discoverable from the failure
message alone.

## 1. Headless runs MUST set `QT_QPA_PLATFORM=offscreen`

```bash
export QT_QPA_PLATFORM=offscreen
```

Without it, the first test reaching `pytest-qt`'s `qapp` fixture tries to open a
real display, Qt calls `abort()`, and **the whole pytest process dies** with:

```
Fatal Python error: Aborted
  File ".../pytestqt/plugin.py", line 76 in qapp
```

Exit code **134** (SIGABRT), Slurm `ExitCode 6:0`. You get no summary line, no
pass/fail counts, and a several-hundred-line C-extension dump — none of which
names the cause. It is not an OOM and not a test failure.

`tests/unit` reaches the Qt tests around **79%**, so a run that dies there looks
like it "mostly worked". It didn't; it produced no result at all.

This is not new knowledge — all three CI workflows already set it
(`run-pytest.yml:119`, `run-pytest-full.yml:55`, `gui-checks.yml:166`). **Read
the workflow env blocks before writing any new runner**; they are the record of
what this suite needs.

## 2. Never `pytest -n auto` on the cluster

`-n auto` asks the *node* how many cores it has. Inside a Slurm allocation that
is the wrong number — often by 50×.

```bash
python3 -c "import os; print(len(os.sched_getaffinity(0)))"   # your real cores
nproc                                                          # the node's
```

Oversubscribing manufactures **timeout failures that look like real bugs**. The
usual casualty is
`tests/unit/cli/test_cli_terminal_failures.py::test_concurrent_process_appends_do_not_lose_records`,
which spawns 8 processes and joins each with a 20 s timeout. Starved, it fails
with `assert process.exitcode == 0` where the value is `None` — the child had
not exited, not crashed. On a quiet allocation with ≥8 real cores it passes.

Pass an explicit `-n <cores>` or omit `-n`.

## 3. Clear `addopts` for any run whose output goes to a file

```toml
addopts = "--verbose --capture=no -m 'not slow'"    # pyproject.toml:221
```

`--capture=no` streams every line of test output to wherever stdout points. To a
terminal or a discarded pipe that is free; to a log on **shared** `/bigdata` or
`/rhome` it dominates the runtime. Measured: 38 minutes to reach a third of the
suite on an *idle* 64-core node, versus roughly half that with capture restored.

```bash
uv run pytest tests/unit -q --no-header -p no:randomly -o addopts= -m "not slow"
```

`-o addopts=` clears the whole string, which **also drops `-m 'not slow'`** — so
re-add it explicitly, or you silently pull in the slow-marker sweep and change
what you are measuring.

## 4. `-x` does not produce a baseline

`-x` stops at the first failure. `tests/unit/cli/` sorts early, so a run that
looks like a clean sweep may have covered a third of the suite. A summary line
reading `946 passed, 1 failed, 32 skipped, 588 deselected` from an `-x` run is
**not** the suite's result and must never be recorded as a baseline — the real
totals are larger and unknown.

Use `-x` to iterate on a known failure. Drop it to measure anything.

## Timings — set expectations before you set a timeout

| What | How long |
|---|---|
| `tests/unit`, full, capture on, quiet node | **~65 min** |
| `tests/unit`, `--capture=no` to a shared-FS log | ~2 h (dominated by I/O) |
| A single test file | seconds |

The suite is **not** a two-minute job. Anything over a couple of minutes is a
Slurm job — see the **`slurm-job`** skill.

## The canonical batch invocation

Committed and reusable:
`docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch`

```bash
#SBATCH --partition=short          # 2 h cap; the suite fits
#SBATCH --cpus-per-task=16         # ≥8 so the 8-process test isn't starved
#SBATCH --mem=32G                  # peak RSS observed ~3.3 GB; DefMemPerCPU is 1 GB
#SBATCH --time=02:00:00
#SBATCH --output=/bigdata/.../slurm_logs/%j.log     # SHARED storage, must exist

export QT_QPA_PLATFORM=offscreen
status=0
uv run pytest tests/unit -q --no-header -p no:randomly -o addopts= -m "not slow" || status=$?
```

Two details that bite:

- **`--output` must be shared storage and the directory must already exist.**
  Pointing it at `/scratch/<user>/<jobid>` — including a Claude scratchpad —
  gives `FAILED`, `ExitCode 0:53`, and **no log at all**, intermittently,
  because that path is node-local to a different job.
- **`|| status=$?`, never a bare call**, under `set -e`. Otherwise the script
  aborts before printing its summary on the one outcome that matters.

## Environment facts

- `uv` only. Never bare `python` or `pip`. Full env:
  `uv sync --group dev --group test-qt --group docs --extra gui --extra napari`
- **There is no `test` dependency group** — only `dev`, `test-qt`, `docs`.
- `testpaths` covers `tests/unit`, `tests/smoke`, `tests/integration`,
  `tests/gui`. Naming one path narrows the run; a phase gate that runs only
  `tests/unit/cli` can be green while the default lane is red.
- Qt/napari widget tests need the `test-qt` group **and** the `napari` extra.

## Interpreting a result honestly

- **A crash is not a failure count.** Exit 134/139 with a `Fatal Python error`
  dump means the process died; there is no result to report.
- **Say which invocation produced a number.** `-x`, `-m`, `-n`, and a narrowed
  path all change what a count means, and a count without its command is not
  reproducible.
- **A gate that is already red at the baseline is not a gate.** `uv run mypy
  src/phenotypic` reports 417 errors in 124 files and `ruff check src/phenotypic`
  reports 25, both pre-existing. Compare against those counts; do not state them
  as "passes".
