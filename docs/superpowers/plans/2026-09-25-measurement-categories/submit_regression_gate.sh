#!/bin/bash
# Submit a measurement-categories regression gate as a 24-way sharded array
# against a worktree detached at one commit.
#
#   bash submit_regression_gate.sh <SHA> <LABEL>
#
# Run it at the gate commit and at the branch base (a8b6e17c), then compare the
# two failing-name sets with
# ../2026-09-03-cli-gui-state-tracking/collect_results.py --baseline.
#
# A thin wrapper over the cli-gui-state-tracking gate harness
# (regression_shard.sbatch + regression_cleanup.sbatch), which already guards
# the three things that void a sharded result: a tree edited inside the array's
# spread (detached worktree), several worktrees importing one venv (per-shard
# provenance assertion), and missing optional extras reading as failures.
# Adapted from ../2026-09-22-figures-in-ome-zarr/submit_regression_gate.sh.
set -euo pipefail

SHA_IN=${1:?usage: submit_regression_gate.sh <SHA> <LABEL>}
LABEL=${2:?usage: submit_regression_gate.sh <SHA> <LABEL>}
REPO=/bigdata/exfab/anguy344/PhenoTypic
SRC_WORKTREE="${REPO}/.claude/worktrees/measurement-tags"
HARNESS="${SRC_WORKTREE}/docs/superpowers/plans/2026-09-03-cli-gui-state-tracking"
GATE_ROOT=/bigdata/exfab/anguy344/gate-worktrees
LOG_DIR=/bigdata/exfab/anguy344/slurm_logs

SHA=$(git -C "$SRC_WORKTREE" rev-parse "$SHA_IN")
SHORT=$(git -C "$SRC_WORKTREE" rev-parse --short "$SHA")
WORKTREE="${GATE_ROOT}/mcat-${LABEL}-${SHORT}"
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

# 8 CPUs: two arrays (gate + base) at 24 x 8 = 384 fit the iwheeldonlab cap
# together. 8 is the floor -- one test spawns 8 processes against a 20 s join.
ARRAY_ID=$(sbatch --parsable --cpus-per-task=8 --job-name="mcat-${LABEL}" \
    --export=ALL,WORKTREE="$WORKTREE" "${HARNESS}/regression_shard.sbatch" 2>&1)
[[ "$ARRAY_ID" =~ ^[0-9]+$ ]] || { echo "SUBMIT FAILED: ${ARRAY_ID}" >&2; exit 1; }

CLEANUP_ID=$(sbatch --parsable --dependency=afterany:"$ARRAY_ID" \
    --export=ALL,WORKTREE="$WORKTREE",REPO="$REPO",ARRAY_ID="$ARRAY_ID" \
    "${HARNESS}/regression_cleanup.sbatch" 2>&1)
[[ "$CLEANUP_ID" =~ ^[0-9]+$ ]] || { echo "SUBMIT FAILED (cleanup): ${CLEANUP_ID}" >&2; exit 1; }

echo "array: ${ARRAY_ID}   finalizer: ${CLEANUP_ID}"
echo "junit: ${LOG_DIR}/junit_${ARRAY_ID}_*.xml"
scontrol show job "$ARRAY_ID" | grep -oE 'JobState=[^ ]+|Reason=[^ ]+' | sort -u
