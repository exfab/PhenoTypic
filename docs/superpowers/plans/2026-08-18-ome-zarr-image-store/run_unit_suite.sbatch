#!/bin/bash
#SBATCH --job-name=omezarr-unit
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:45:00
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
# `|| status=$?` and not a bare call: `set -e` would abort the script the moment
# pytest returned non-zero, so the summary lines below would never print on the
# one outcome they matter most for -- a failure.
status=0
uv run pytest tests/unit -q --no-header -p no:randomly || status=$?

echo
echo "pytest exit status: ${status}"
echo "Finished: $(date -Is)"
exit "${status}"
