#!/bin/bash
# Submit the figures-in-ome-zarr full regression (plan Task 9) as a 24-way
# sharded array against a detached worktree at one commit.
#
#   bash submit_regression_gate.sh <SHA> <LABEL>
#
# Run it twice -- once at the branch head, once at its merge-base with
# origin/main -- and compare the two failing-name sets with
# ../2026-09-03-cli-gui-state-tracking/collect_results.py --baseline.
#
# This is a thin wrapper over the cli-gui-state-tracking gate harness
# (regression_shard.sbatch + regression_cleanup.sbatch), which already solves
# the three things that void a sharded result: a tree edited inside the array's
# spread (detached worktree), several worktrees importing one venv (per-shard
# provenance assertion), and missing optional extras reading as failures (the
# uv sync line and the finalizer's triage). See that script's header for the
# measured incidents behind each.
set -euo pipefail

SHA_IN=${1:?usage: submit_regression_gate.sh <SHA> <LABEL>}
LABEL=${2:?usage: submit_regression_gate.sh <SHA> <LABEL>}
REPO=/bigdata/exfab/anguy344/PhenoTypic
SRC_WORKTREE="${REPO}/.claude/worktrees/figures-ome-zarr-storage"
HARNESS="${SRC_WORKTREE}/docs/superpowers/plans/2026-09-03-cli-gui-state-tracking"
GATE_ROOT=/bigdata/exfab/anguy344/gate-worktrees
LOG_DIR=/bigdata/exfab/anguy344/slurm_logs

SHA=$(git -C "$SRC_WORKTREE" rev-parse "$SHA_IN")
SHORT=$(git -C "$SRC_WORKTREE" rev-parse --short "$SHA")
WORKTREE="${GATE_ROOT}/fioz-${LABEL}-${SHORT}"
mkdir -p "$GATE_ROOT" "$LOG_DIR"

echo "=== ${LABEL}: ${SHA} -- $(git -C "$SRC_WORKTREE" log -1 --format=%s "$SHA")"
echo "    worktree: ${WORKTREE}"

if [[ -d "$WORKTREE" ]]; then
    echo "reusing existing gate worktree"
else
    git -C "$REPO" worktree add --detach "$WORKTREE" "$SHA"
fi

(cd "$WORKTREE" && uv sync --group dev --group test-qt \
    --extra gui --extra napari --extra tune --extra topology)

RESOLVED=$(cd "$WORKTREE" && uv run python -c "import phenotypic; print(phenotypic.__file__)")
case "$RESOLVED" in
    "$WORKTREE"/*) echo "provenance OK: ${RESOLVED}" ;;
    *) echo "ABORT: phenotypic resolves outside ${WORKTREE}: ${RESOLVED}" >&2; exit 3 ;;
esac

# 8 CPUs, not the harness's 12: two arrays (branch + baseline) at 24 x 8 = 384
# fit the iwheeldonlab cap together instead of one queueing behind the other.
# 8 is the floor -- one test spawns 8 processes against a 20 s join.
ARRAY_ID=$(sbatch --parsable --cpus-per-task=8 --job-name="fioz-${LABEL}" \
    --export=ALL,WORKTREE="$WORKTREE" "${HARNESS}/regression_shard.sbatch" 2>&1)
[[ "$ARRAY_ID" =~ ^[0-9]+$ ]] || { echo "SUBMIT FAILED: ${ARRAY_ID}" >&2; exit 1; }

CLEANUP_ID=$(sbatch --parsable --dependency=afterany:"$ARRAY_ID" \
    --export=ALL,WORKTREE="$WORKTREE",REPO="$REPO",ARRAY_ID="$ARRAY_ID" \
    "${HARNESS}/regression_cleanup.sbatch" 2>&1)
[[ "$CLEANUP_ID" =~ ^[0-9]+$ ]] || { echo "SUBMIT FAILED (cleanup): ${CLEANUP_ID}" >&2; exit 1; }

echo "array: ${ARRAY_ID}   finalizer: ${CLEANUP_ID}"
echo "junit: ${LOG_DIR}/junit_${ARRAY_ID}_*.xml"
scontrol show job "$ARRAY_ID" | grep -oE 'JobState=[^ ]+|Reason=[^ ]+' | sort -u
