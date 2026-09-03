#!/bin/bash
#SBATCH --job-name=phenotypic-import-laziness
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-47%40
#SBATCH --output=/bigdata/exfab/anguy344/slurm_logs/%A_%a.log
set -euo pipefail

# Gate runner for the import-laziness branch. Derived from
# docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch,
# widened from 24 to 48 shards so a stage gate finishes in a coffee break.
#
# Sizing against the account caps (iwheeldonlab: 384 CPU / 1 TB, summed across
# every running job): 40 concurrent tasks * 8 CPU = 320 CPU, * 16 GB = 640 GB.
# Exceeding a CPU cap does not fail at submit -- it queues with
# Reason=AssocGrpCpuLimit -- so the headroom is deliberate, not incidental.
WORKTREE=/bigdata/exfab/anguy344/PhenoTypic/.worktrees/import-laziness
PYTEST_GROUP_COUNT=48

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
# Without this the interpreter aborts partway through the Qt lane with no summary.
export QT_QPA_PLATFORM=offscreen
# Without this the browser lane silently skips rather than failing.
export PLAYWRIGHT=1

echo "Date:  $(date -Is)"
echo "Node:  $(hostname -s)"
echo "Job:   ${SLURM_JOB_ID}"
echo "Group: ${SLURM_ARRAY_TASK_ID}/${PYTEST_GROUP_COUNT}"
echo "CPUs:  ${SLURM_CPUS_PER_TASK}"
echo "Load:  $(uptime)"
echo

cd "$WORKTREE"
echo "Commit under test: $(git rev-parse --short HEAD) $(git log -1 --format=%s)"
echo

PYTEST_GROUP_ID=${SLURM_ARRAY_TASK_ID}
mapfile -t ALL_TEST_FILES < <(
    {
        find tests/unit tests/gui tests/integration tests/smoke \
            -type f -name 'test_*.py' -print
        # tests/migration is not in testpaths and must be named explicitly. The
        # equivalence suite is the repo's pipeline round-trip golden and is a
        # first-class gate for this branch, so the whole directory is included
        # rather than the two files the ome-zarr runner named.
        find tests/migration -type f -name 'test_*.py' -print
    } | LC_ALL=C sort
)

declare -a TEST_TARGETS=()
for file_index in "${!ALL_TEST_FILES[@]}"; do
    if ((file_index % PYTEST_GROUP_COUNT == PYTEST_GROUP_ID)); then
        TEST_TARGETS+=("${ALL_TEST_FILES[file_index]}")
    fi
done

if ((${#TEST_TARGETS[@]} == 0)); then
    echo "Pytest group ${PYTEST_GROUP_ID} has no test files" >&2
    exit 2
fi

echo "Discovered files: ${#ALL_TEST_FILES[@]}"
echo "Group files:      ${#TEST_TARGETS[@]}"
printf '  %s\n' "${TEST_TARGETS[@]}"
echo

# Tests model ordinary jobs and intentionally mock SLURM_JOB_ID. Do not leak
# this verification runner's array identity into their subject.
unset SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID SLURM_ARRAY_TASK_COUNT
unset SLURM_ARRAY_TASK_MIN SLURM_ARRAY_TASK_MAX SLURM_ARRAY_TASK_STEP

# Do not use `-n auto`: xdist would see the node's core count rather than this
# allocation's, and manufacture timeout failures. Clear repository addopts to
# avoid uncaptured shared-filesystem log traffic, then restore the default
# `not slow` selection explicitly.
pytest_status=0
uv run python -m pytest "${TEST_TARGETS[@]}" -q --no-header \
    -p no:randomly -o addopts= -m "not slow" || pytest_status=$?

echo
echo "pytest group ${PYTEST_GROUP_ID} exit status: ${pytest_status}"
echo "Finished: $(date -Is)"
exit "${pytest_status}"
