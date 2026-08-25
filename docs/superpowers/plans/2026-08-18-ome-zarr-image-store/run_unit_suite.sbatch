#!/bin/bash
#SBATCH --job-name=omezarr-tests
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/bigdata/exfab/anguy344/slurm_logs/%A_%a.log
set -euo pipefail

# Six disjoint pytest groups run in parallel. Keep the CLI group at 16 CPUs:
# its concurrency regression launches eight child processes and is otherwise
# prone to false timeout failures under scheduler contention.
WORKTREE=/bigdata/exfab/anguy344/PhenoTypic/.worktrees/worktree-ome-zarr-image-store

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export QT_QPA_PLATFORM=offscreen

echo "Date:  $(date -Is)"
echo "Node:  $(hostname -s)"
echo "Job:   ${SLURM_JOB_ID}"
echo "Group: ${SLURM_ARRAY_TASK_ID}"
echo "CPUs:  ${SLURM_CPUS_PER_TASK}"
echo "Load:  $(uptime)"
echo

cd "$WORKTREE"
echo "Commit under test: $(git rev-parse --short HEAD) $(git log -1 --format=%s)"
echo

PYTEST_GROUP_ID=${SLURM_ARRAY_TASK_ID}
declare -a TEST_TARGETS
case "${PYTEST_GROUP_ID}" in
    0) TEST_TARGETS=(tests/unit/core tests/unit/sdk_) ;;
    1) TEST_TARGETS=(tests/unit/cli tests/unit/analysis tests/unit/schema) ;;
    2) TEST_TARGETS=(tests/unit/gui) ;;
    3) TEST_TARGETS=(tests/unit/tune) ;;
    4) TEST_TARGETS=(
        tests/unit/abc_ tests/unit/assets tests/unit/ci tests/unit/correction
        tests/unit/detect tests/unit/docs tests/unit/enhance tests/unit/grid
        tests/unit/measure tests/unit/plotting tests/unit/post tests/unit/prefab
        tests/unit/qc tests/unit/refine tests/unit/resources tests/unit/util
        tests/unit/viz tests/unit/test_*.py
    ) ;;
    5) TEST_TARGETS=(
        tests/gui tests/integration tests/smoke
        tests/migration/test_metadata_schema_migration.py
        tests/migration/test_curation_imagefile_rename.py
    ) ;;
    *) echo "Unknown pytest group: ${PYTEST_GROUP_ID}" >&2; exit 2 ;;
esac

echo "Targets: ${TEST_TARGETS[*]}"
echo

# Controller tests model ordinary jobs and intentionally mock SLURM_JOB_ID.
# Do not leak this verification runner's array identity into their subject.
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
