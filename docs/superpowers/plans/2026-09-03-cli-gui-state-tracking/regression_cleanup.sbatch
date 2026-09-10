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
LOG_DIR=/bigdata/exfab/anguy344/slurm_logs

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

# Triage the reds the HARNESS caused, before anyone spends a traceback on them.
#
# A test that errors because its optional dependency was never installed is
# indistinguishable, in the summary line, from a test that found a real defect
# -- and the default reading is that the change did it. Array 28236077 lost 25
# of its 26 failing names to two missing extras, and the second one surfaced
# only because someone re-read a log after fixing the first.
#
# TWO signatures, because neither alone is complete. The package raises
# deliberately and names its own extra, which is the actionable form -- but
# only where the import is routed through a guarded path. A test that imports
# the dependency directly raises a bare ModuleNotFoundError naming no extra,
# and 20 of that array's 26 names were that shape.
echo "=============== harness-caused failures (missing dependencies) ==============="
_HITS=0
if grep -hoE "Install (PhenoTypic with the .[a-z_-]+. extra|the '[a-z_-]+' extra)" \
        "${LOG_DIR}"/"${ARRAY_ID}"_*.log 2>/dev/null | sort | uniq -c | sort -rn | grep .; then
    _HITS=1
fi
if grep -hoE "ModuleNotFoundError: No module named '[a-zA-Z_][a-zA-Z0-9_]*'" \
        "${LOG_DIR}"/"${ARRAY_ID}"_*.log 2>/dev/null | sort | uniq -c | sort -rn | grep .; then
    _HITS=1
fi
if ((_HITS)); then
    echo
    echo "  ^ these are the HARNESS, not the tree under test. Add the extra to"
    echo "    submit_regression_gate.sh's uv sync line and re-gate; do not"
    echo "    attribute these names to the change."
else
    echo "  none -- no shard reported a missing optional dependency"
fi
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
