#!/bin/bash
#SBATCH --job-name=omezarr-unit
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/bigdata/exfab/anguy344/slurm_logs/%j.log
set -euo pipefail

# Regression check for the OME-Zarr worktree after Phase 0 landed.
#
# Run as a batch job because the interactive node (r23) is at load ~467 and the
# suite -- 150 s uncontended -- could not finish inside 900 s there. A dedicated
# allocation can also land on a different node entirely.
#
# Baseline to compare against, at the Phase 2 gate:
#     8,646 passed, 6 failed
#
# NOT the "946 passed" figure this script used to quote. That number came from
# `-x` runs, which truncate at the first failure in tests/unit/cli -- it was a
# third of the suite mislabelled as the whole of it. The first genuinely
# complete wide run happened at the Phase 2 gate.
#
# Of the 6: three are FilFinderDetector smoke tests failing on an absent
# `topology` extra (invisible to every earlier run, because none of them
# included tests/smoke -- a gate that has never run is not a gate); one is
# test_inspect_remeasures_when_explicit_image_changes; and one is
# test_concurrent_process_appends_do_not_lose_records, which spawns 8 processes
# and joins each with a 20 s timeout, so it fails by starvation on a contended
# allocation and passes alone. This job requests 16 CPUs so those children get
# real cores.

WORKTREE=/bigdata/exfab/anguy344/PhenoTypic/.worktrees/worktree-ome-zarr-image-store

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# MANDATORY on a compute node. Without it the Qt tests reach
# `pytestqt`'s `qapp` fixture, try to open a real display, and Qt calls abort()
# -- SIGABRT, exit 134, killing the whole pytest process rather than failing one
# test. Job 27561771 died exactly that way at 79%.
#
# Not a discovery: all three of the project's own CI workflows already set this
# (run-pytest.yml:119, run-pytest-full.yml:55, gui-checks.yml:166). Any headless
# runner needs it.
export QT_QPA_PLATFORM=offscreen

echo "Date:  $(date -Is)"
echo "Node:  $(hostname -s)"
echo "Job:   ${SLURM_JOB_ID}"
echo "CPUs:  ${SLURM_CPUS_PER_TASK}"
echo "Load:  $(uptime)"
echo

cd "$WORKTREE"
echo "Commit under test: $(git rev-parse --short HEAD) $(git log -1 --format=%s)"
echo

# Deliberately NOT `-n auto`: pytest-xdist reads the node's core count, not the
# allocation's, so `auto` would oversubscribe and manufacture exactly the
# starvation failure this job exists to rule out. Serial also matches how the
# 946/1/32 baseline was measured -- apples to apples.
# `-p no:randomly` for the same reason: fixed order, comparable counts.
# `-o addopts=` clears the project default, which is
#     addopts = "--verbose --capture=no -m 'not slow'"
# `--capture=no` streams every line of test output straight to this log on
# SHARED /bigdata storage. A first attempt (job 27561428) took 38 minutes to
# reach a third of the suite on an idle 64-core node -- roughly 15x slower than
# the same suite run interactively, where the output was piped to `tail` and
# discarded. The I/O was the cost, not the tests.
#
# `-m "not slow"` is re-added explicitly, because clearing addopts drops it too
# and the slow-marker sweep is deliberately excluded from the default lane.
#
# `|| status=$?` and not a bare call: `set -e` would abort the script the moment
# pytest returned non-zero, so the summary lines below would never print on the
# one outcome they matter most for -- a failure.
status=0
# Phase 2 reached the GUI builder's preview cache and tile renderer, so the
# wide run must cover tests/gui and tests/integration as well as tests/unit --
# a unit-only run would be green while the default lane is red.
#
# Two tests/migration FILES are listed explicitly, not the directory. That
# directory is not in `testpaths` (pyproject.toml:218), so nothing has run it,
# and Phase 3 makes part of it load-bearing: C11 changed recompile dataset
# discovery, and an unconverted legacy bundle is discoverable only through the
# DIR_HDF branch that the very recompile run authorizing its metadata migration
# depends on. test_metadata_schema_migration.py + test_curation_imagefile_rename.py
# cover that and pass (52 passed, 23 s).
#
# test_equivalence.py is deliberately EXCLUDED, and adding the bare directory
# would have made this gate permanently red for reasons unrelated to it:
# 57 of its 341 tests fail on STALE GOLDENS, not on drift. The failures are
# structural, not numeric -- the goldens expect `LogGrowthModel_r` while the
# code emits `LogGrowthModel_Area_r`. Analysis headers became metric-qualified
# in 67cfa259, an intentional and documented convention change; the goldens
# were last captured at e2a91078/f3fa28b3, before it, and were never re-taken.
# The suite is a real pydantic-v2 regression detector that has silently rotted
# because it is not in `testpaths`. Re-capturing it is its own piece of work,
# and it belongs to whoever owns that convention -- not to this change, which
# touches none of the failing modules (denoisers, refiners, analysis, measure).
#
# Adding these two files changes the total, so the count below is not
# comparable to the 8,646 baseline -- compare failures, not totals.
#
# tests/e2e is the other orphan, and is deliberately NOT added: it drives
# Playwright against a real browser, and it is GUI surface that Phase 4 owns.
# It needs its own gate, not a line in this one.
#
# Use the repository's sole supported runner so the batch lane resolves the
# same locked environment as every local verification command.
uv run python -m pytest tests/unit tests/gui tests/integration tests/smoke \
    tests/migration/test_metadata_schema_migration.py \
    tests/migration/test_curation_imagefile_rename.py \
    -q --no-header -p no:randomly -o addopts= -m "not slow" || status=$?

echo
echo "pytest exit status: ${status}"
echo "Finished: $(date -Is)"
exit "${status}"
