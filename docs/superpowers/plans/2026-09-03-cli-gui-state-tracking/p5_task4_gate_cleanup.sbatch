#!/bin/bash
#SBATCH --job-name=p5-gate-cleanup
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=/bigdata/exfab/anguy344/slurm_logs/p5_gate_cleanup_%j.log
set -euo pipefail

# Terminal finalizer for the P5 Task 4 fan-out gate. Submitted by the gate
# itself with --dependency=afterany, BEFORE any work, never as a line at the
# end of that script.
#
# afterany, not afterok: the gate holds while an inner SLURM array runs, so
# hitting the 2 h wall is its likely failure rather than an exotic one -- and a
# job killed at the wall never reaches its own last line. afterany fires on
# success, failure, cancellation and timeout alike.
#
# This is a terminal finalizer, not a parallel sidecar: it holds no allocation
# beside the gate and cannot contend with it for submission bounds.

: "${WORKTREE:?WORKTREE not exported}"
: "${REPO:?REPO not exported}"
: "${GATE_WORK:?GATE_WORK not exported}"

echo "Date: $(date -Is)  Node: $(hostname -s)  Job: ${SLURM_JOB_ID}"

# --force, because unforced `remove` REFUSES a dirty worktree -- which turns
# guaranteed cleanup into a silent no-op precisely when something went wrong
# inside it. The path is spelled exactly, never globbed: other sessions keep
# live work in sibling directories.
if [[ -d "$WORKTREE" ]]; then
    git -C "$REPO" worktree remove --force "$WORKTREE" \
        && echo "removed worktree: ${WORKTREE}" \
        || { echo "worktree remove failed; falling back to rm -rf" >&2
             rm -rf "$WORKTREE"; }
else
    echo "worktree already gone: ${WORKTREE}"
fi
git -C "$REPO" worktree prune
echo "pruned administrative entries"

# The two 5.6 GB tree copies and the 1.7 GB of inputs -- ~13 GB per run.
# Removed only under the job-scoped directory the gate created, never a glob.
if [[ "$GATE_WORK" == /bigdata/exfab/anguy344/slurm_logs/p5_gate_* ]]; then
    rm -rf "$GATE_WORK" && echo "removed gate scratch: ${GATE_WORK}"
else
    echo "REFUSING to remove unexpected path: ${GATE_WORK}" >&2
    exit 1
fi

echo "Finished: $(date -Is)"
