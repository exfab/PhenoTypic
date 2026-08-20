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
# Baseline to compare against, measured twice at commit f3564336:
#     946 passed, 1 failed, 32 skipped
# The one failure is test_concurrent_process_appends_do_not_lose_records, which
# spawns 8 processes and joins each with a 20 s timeout; it fails by starvation
# on a contended 4-core allocation, and was observed PASSING under lighter load.
# This job requests 16 CPUs specifically so those 8 children get real cores --
# so the expected result here is 947 passed, 0 failed.

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
uv run pytest tests/unit tests/gui tests/integration tests/smoke \
    -q --no-header -p no:randomly -o addopts= -m "not slow" || status=$?

echo
echo "pytest exit status: ${status}"
echo "Finished: $(date -Is)"
exit "${status}"
