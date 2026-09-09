#!/bin/bash
#SBATCH --job-name=cgst-regression-cleanup
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=/bigdata/exfab/anguy344/slurm_logs/%j.log
set -euo pipefail

# Terminal finalizer for the regression array. Submitted with
# --dependency=afterany, NOT afterok.
#
# afterany because a task that is OOM-killed or hits the wall never reaches its
# own cleanup line -- so cleanup written into the task body leaks in exactly the
# cases that produce leaks. afterany fires on success, failure, cancellation and
# timeout alike.
#
# This is a terminal finalizer, not a parallel sidecar: it holds no allocation
# beside the array and cannot contend with it for submission bounds.

: "${WORKTREE:?WORKTREE not exported}"
: "${REPO:?REPO not exported}"
: "${ARRAY_ID:?ARRAY_ID not exported}"

echo "Date: $(date -Is)  Node: $(hostname -s)  Job: ${SLURM_JOB_ID}"
echo "Cleaning gate worktree: ${WORKTREE}"
echo

# Report the result BEFORE removing the tree, so the summary survives even if
# removal fails.
echo "=============== array ${ARRAY_ID} per-task exit states ==============="
sacct -j "${ARRAY_ID}" --format=JobID%20,State%15,Elapsed,MaxRSS,ExitCode -n -P \
    | grep -vE '\.(batch|extern)\|' || true
echo
echo "=============== non-zero shards ==============="
sacct -j "${ARRAY_ID}" --format=JobID%20,State,ExitCode -n -P \
    | grep -vE '\.(batch|extern)\|' \
    | grep -vE '\|COMPLETED\|0:0$' || echo "  none -- every shard exited 0"
echo

# --force because unforced `remove` REFUSES a dirty worktree, which turns
# guaranteed cleanup into a silent no-op precisely when something went wrong
# inside it. The path is spelled exactly, never globbed: other sessions keep
# live work in sibling directories.
if [[ -d "$WORKTREE" ]]; then
    git -C "$REPO" worktree remove --force "$WORKTREE" \
        && echo "removed: ${WORKTREE}" \
        || { echo "worktree remove failed; falling back to rm -rf" >&2
             rm -rf "$WORKTREE"; }
else
    echo "already gone: ${WORKTREE}"
fi

git -C "$REPO" worktree prune
echo "pruned administrative entries"
echo "Finished: $(date -Is)"
