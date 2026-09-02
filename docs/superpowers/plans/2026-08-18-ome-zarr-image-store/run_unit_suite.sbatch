#!/bin/bash
#SBATCH --job-name=phenotypic-tests
#SBATCH --partition=short
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-23%24
#SBATCH --output=/bigdata/exfab/anguy344/slurm_logs/%A_%a.log
set -euo pipefail

# The full non-slow verification scope is deterministically file-sharded across
# 24 array tasks. Twelve CPUs leave enough headroom for tests that spawn eight
# child processes while keeping the fully active array below account caps.
WORKTREE=/bigdata/exfab/anguy344/PhenoTypic/.worktrees/worktree-ome-zarr-image-store
PYTEST_GROUP_COUNT=24

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export QT_QPA_PLATFORM=offscreen

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

# Do not use `-n auto`: xdist would see the node rather than this allocation.
# Clear repository addopts to avoid uncaptured shared-filesystem log traffic,
# then restore the default `not slow` selection explicitly.
pytest_status=0
uv run python -m pytest "${TEST_TARGETS[@]}" -q --no-header \
    -p no:randomly -o addopts= -m "not slow" || pytest_status=$?

echo
echo "pytest group ${PYTEST_GROUP_ID} exit status: ${pytest_status}"
echo "Finished: $(date -Is)"
exit "${pytest_status}"
