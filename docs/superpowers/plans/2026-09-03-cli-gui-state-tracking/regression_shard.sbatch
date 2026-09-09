#!/bin/bash
#SBATCH --job-name=cgst-regression
#SBATCH --partition=short
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-23%24
#SBATCH --output=/bigdata/exfab/anguy344/slurm_logs/%A_%a.log
set -euo pipefail

# One shard of the cli-gui-state-tracking full regression.
#
# WORKTREE is passed by submit_regression_gate.sh via --export. It is a DETACHED
# worktree at a fixed SHA, living outside the repo, so no shard can observe an
# edit made in the main checkout while the array is spread over several minutes.
# A file edited inside that spread does not yield a stale result -- it yields a
# union across two trees that no single tree ever produced. Void, not old.
#
# Sizing: 24 x 12 CPU = 288, against the iwheeldonlab cap of 384. Note that cap
# is per ACCOUNT, and `preempt` and `exfab` draw on separate pools -- summing
# `squeue -t RUNNING` across all accounts over-counts what this array competes
# with, and would talk you out of a job that fits.

: "${WORKTREE:?WORKTREE not exported -- submit via submit_regression_gate.sh}"
PYTEST_GROUP_COUNT=24
PYTEST_GROUP_ID=${SLURM_ARRAY_TASK_ID}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
# Without this the Qt tests abort the interpreter at ~79% with no summary --
# which reads as a hang, not a failure.
export QT_QPA_PLATFORM=offscreen

echo "Date:  $(date -Is)"
echo "Node:  $(hostname -s)"
echo "Job:   ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Group: ${PYTEST_GROUP_ID}/${PYTEST_GROUP_COUNT}"
echo "CPUs:  ${SLURM_CPUS_PER_TASK}"
echo "Tree:  ${WORKTREE}"
echo

cd "$WORKTREE"
echo "Commit under test: $(git rev-parse HEAD)"
echo "Working tree dirty: $(git status --porcelain | wc -l) file(s)"
echo

# The venv trap: several worktrees pointed at one prebuilt venv all resolve an
# editable install through a finder pinned to a single path, so every shard
# silently imports the SAME tree and all 24 results become copies of one, with
# nothing failing to say so. This is the check that catches it, and it runs in
# every shard rather than once at submit -- a per-shard check also catches a
# single task that landed on a node with a stale cache.
RESOLVED=$(uv run python -c "import phenotypic; print(phenotypic.__file__)")
echo "phenotypic resolves to: ${RESOLVED}"
case "$RESOLVED" in
    "$WORKTREE"/*) ;;
    *)  echo "ABORT: phenotypic resolves OUTSIDE the gate worktree." >&2
        echo "       Every shard would measure the same wrong tree." >&2
        exit 3 ;;
esac
echo

mapfile -t ALL_TEST_FILES < <(
    {
        find tests/unit tests/gui tests/integration tests/smoke \
            -type f -name 'test_*.py' -print
        printf '%s\n' \
            tests/migration/test_metadata_schema_migration.py \
            tests/migration/test_curation_imagefile_rename.py
    } | LC_ALL=C sort
)

declare -a TEST_TARGETS=()
for file_index in "${!ALL_TEST_FILES[@]}"; do
    if ((file_index % PYTEST_GROUP_COUNT == PYTEST_GROUP_ID)); then
        TEST_TARGETS+=("${ALL_TEST_FILES[file_index]}")
    fi
done

# A shard with no files is a sharding bug, not an empty success. Exiting 0 here
# would let a broken split report green from tasks that ran nothing.
if ((${#TEST_TARGETS[@]} == 0)); then
    echo "ABORT: group ${PYTEST_GROUP_ID} has no test files" >&2
    exit 2
fi

echo "Discovered files: ${#ALL_TEST_FILES[@]}"
echo "Group files:      ${#TEST_TARGETS[@]}"
printf '  %s\n' "${TEST_TARGETS[@]}"
echo

# Tests model ordinary CLI jobs and intentionally mock SLURM_JOB_ID. Do not leak
# this runner's array identity into their subject.
unset SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID SLURM_ARRAY_TASK_COUNT
unset SLURM_ARRAY_TASK_MIN SLURM_ARRAY_TASK_MAX SLURM_ARRAY_TASK_STEP

# `-o addopts=` clears the project default, whose `--capture=no` streams
# uncaptured output and can triple runtime when stdout is a file on GPFS; the
# `not slow` selection it also carried is restored explicitly. `-n auto` is safe
# here because tests/conftest.py:96 resolves xdist workers from
# sched_getaffinity rather than the node's core count -- but 12 CPUs per task is
# already the parallelism, so shards stay serial and their timings comparable.
pytest_status=0
uv run python -m pytest "${TEST_TARGETS[@]}" -q --no-header \
    -p no:randomly -o addopts= -m "not slow" -rf || pytest_status=$?

echo
echo "pytest group ${PYTEST_GROUP_ID} exit status: ${pytest_status}"
echo "Finished: $(date -Is)"
exit "${pytest_status}"
