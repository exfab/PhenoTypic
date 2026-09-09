#!/bin/bash
# Submit the cli-gui-state-tracking full regression as a 24-way sharded array
# against a detached worktree at one commit.
#
#   bash submit_regression_gate.sh [<SHA>]
#
# Defaults to HEAD of the cli-gui-state-tracking worktree.
#
# Why this exists: the full scope is 725 test files and ~90+ minutes serially,
# which is a Slurm job, not a foreground command. Sharding it is the easy part.
# Making the answer MEAN something is the part that goes wrong, and the two
# things that void it are handled here rather than by asking people to be
# careful:
#
#   1. Shards do not start together. A file edited inside the array's spread
#      produces a union across two trees that no single tree ever produced.
#      Fixed structurally: the array reads a detached worktree that physically
#      cannot see the main checkout's edits, so work continues there meanwhile.
#   2. Several worktrees sharing one prebuilt venv all import the same tree, so
#      24 results become 24 copies of one, with nothing failing to say so.
#      Fixed by `uv sync` per worktree plus a provenance assertion, checked here
#      AND again inside every shard.
set -euo pipefail

REPO=/bigdata/exfab/anguy344/PhenoTypic
SRC_WORKTREE="${REPO}/.worktrees/cli-gui-state-tracking"
# Outside the repo on shared storage: not /scratch/<user>/<jobid> (node-local
# AND per-job, so other nodes would see an empty directory), and not the repo's
# own .worktrees/, where other sessions keep live work.
GATE_ROOT=/bigdata/exfab/anguy344/gate-worktrees
LOG_DIR=/bigdata/exfab/anguy344/slurm_logs
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SHA=$(git -C "$SRC_WORKTREE" rev-parse "${1:-HEAD}")
SHORT=$(git -C "$SRC_WORKTREE" rev-parse --short "$SHA")
WORKTREE="${GATE_ROOT}/regr-${SHORT}"

mkdir -p "$GATE_ROOT" "$LOG_DIR"

echo "=== gate scope ==="
echo "  commit:   ${SHA}"
echo "  subject:  $(git -C "$SRC_WORKTREE" log -1 --format=%s "$SHA")"
echo "  worktree: ${WORKTREE}"
echo

# Uncommitted work is INVISIBLE to a new worktree, so a dirty source tree means
# the gate silently measures something other than what you are looking at. Make
# a WIP commit -- never `git stash`, whose stack is shared with every other
# worktree of this repo and with other live sessions.
DIRTY=$(git -C "$SRC_WORKTREE" status --porcelain | wc -l)
if ((DIRTY > 0)); then
    echo "REFUSING: ${SRC_WORKTREE} has ${DIRTY} uncommitted file(s)." >&2
    echo "The gate would measure ${SHORT}, which does not contain them." >&2
    git -C "$SRC_WORKTREE" status --short >&2
    exit 1
fi

# Serialize worktree creation: `git worktree add` mutates the shared
# .git/worktrees/ directory, so concurrent adds contend for no benefit.
if [[ -d "$WORKTREE" ]]; then
    echo "reusing existing gate worktree at ${WORKTREE}"
else
    git -C "$REPO" worktree add --detach "$WORKTREE" "$SHA"
fi

echo
echo "=== syncing the gate venv (own venv, shared uv cache) ==="
(cd "$WORKTREE" && uv sync --group dev --group test-qt --extra gui --extra napari)

echo
echo "=== provenance check ==="
RESOLVED=$(cd "$WORKTREE" && uv run python -c "import phenotypic; print(phenotypic.__file__)")
echo "  phenotypic resolves to: ${RESOLVED}"
case "$RESOLVED" in
    "$WORKTREE"/*) echo "  OK -- inside the gate worktree" ;;
    *)  echo "ABORT: resolves OUTSIDE the gate worktree; all 24 shards would" >&2
        echo "       measure the same wrong tree and agree with each other." >&2
        exit 3 ;;
esac

echo
echo "=== submitting ==="
# --parsable returns an EMPTY id on rejection while printing the error, so a
# driver that skips this check "runs" for hours having submitted nothing.
ARRAY_ID=$(sbatch --parsable \
    --export=ALL,WORKTREE="$WORKTREE" \
    "${HERE}/regression_shard.sbatch" 2>&1)
if [[ ! "$ARRAY_ID" =~ ^[0-9]+$ ]]; then
    echo "SUBMIT FAILED: ${ARRAY_ID}" >&2
    exit 1
fi
echo "  array:     ${ARRAY_ID}"

CLEANUP_ID=$(sbatch --parsable \
    --dependency=afterany:"$ARRAY_ID" \
    --export=ALL,WORKTREE="$WORKTREE",REPO="$REPO",ARRAY_ID="$ARRAY_ID" \
    "${HERE}/regression_cleanup.sbatch" 2>&1)
if [[ ! "$CLEANUP_ID" =~ ^[0-9]+$ ]]; then
    echo "SUBMIT FAILED (cleanup): ${CLEANUP_ID}" >&2
    echo "The array is running; remove ${WORKTREE} by hand when it finishes." >&2
    exit 1
fi
echo "  finalizer: ${CLEANUP_ID} (afterany)"

echo
echo "=== will it start? (submission is not execution) ==="
scontrol show job "$ARRAY_ID" | grep -oE 'JobState=[^ ]+|StartTime=[^ ]+|Reason=[^ ]+' | sort -u

cat <<EOF

Logs:    ${LOG_DIR}/${ARRAY_ID}_*.log
Watch:   squeue -j ${ARRAY_ID} -o "%.12i %.8T %.10M %R"
Result:  sacct -j ${ARRAY_ID} --format=JobID%20,State,Elapsed,ExitCode

Report the SHA and the shard spread alongside the result. Compare failing
NAMES, never counts -- counts move with node load and with how the shards
happened to pack; the failing set does not.
EOF
