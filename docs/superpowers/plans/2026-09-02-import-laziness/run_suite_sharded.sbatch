#!/bin/bash
#SBATCH --job-name=phenotypic-import-laziness
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-47%8
#SBATCH --output=/bigdata/exfab/anguy344/slurm_logs/%A_%a.log
set -euo pipefail

# Gate runner for the import-laziness branch. Derived from
# docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch,
# widened from 24 to 48 shards so a stage gate finishes in a coffee break.
#
# Sizing against the account caps (iwheeldonlab: 384 CPU / 1 TB, summed across
# every running job, INCLUDING a production run the same account is hosting):
# 8 concurrent tasks * 8 CPU = 64 CPU, * 8 GB = 64 GB. Deliberately small.
# Exceeding a cap does not fail at submit -- it queues with
# Reason=AssocGrpCpuLimit or AssocGrpMemLimit -- so a wide array simply sits
# there, and a wide array that *does* start starves whatever else the account
# is running. Measured peak RSS per shard is 1.3 GB, so 8 GB is a 6x margin;
# the 16 GB inherited from the ome-zarr runner was over-provisioned by 12x and
# was what made this collide with a concurrent 256-task migration array.
# Keep --cpus-per-task at 8 or more: test_concurrent_process_appends_do_not_lose_records
# spawns eight children and joins each with a 20 s timeout.
# Raise --array's %N only after checking `squeue -h -u $USER -t RUNNING`.
#
# GATE_WORKTREE names the tree under test and MUST NOT be edited while the array
# runs. A shard that starts after a source file changes reports failures that
# belong to no commit: pytest reads the new file for a traceback whose line
# numbers came from the old one, and prints `>   ???` where the statement should
# be. Point it at a detached worktree pinned to the commit being gated:
#
#   git worktree add --detach ../gates/<sha> <sha>
#   sbatch --export=ALL,GATE_WORKTREE=/abs/path/to/gates/<sha> run_suite_sharded.sbatch
#
# The virtualenv is deliberately NOT taken from the gate tree. A pinned checkout
# has no `.venv`, and syncing one per gate is minutes of wasted work; instead the
# development environment's interpreter is used with PYTHONPATH pointing at the
# gate tree's `src`, which shadows the editable install. Tests, fixtures and
# goldens then all come from the pinned tree because pytest runs inside it.
DEV_WORKTREE=/bigdata/exfab/anguy344/PhenoTypic/.worktrees/import-laziness
GATE_WORKTREE=${GATE_WORKTREE:-$DEV_WORKTREE}
VENV_PYTHON="$DEV_WORKTREE/.venv/bin/python"
PYTEST_GROUP_COUNT=48

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
# Without this the interpreter aborts partway through the Qt lane with no summary.
export QT_QPA_PLATFORM=offscreen
# Without this the browser lane silently skips rather than failing.
export PLAYWRIGHT=1
# Shadow the editable install so the pinned tree's source is what runs.
export PYTHONPATH="$GATE_WORKTREE/src${PYTHONPATH:+:$PYTHONPATH}"

echo "Date:  $(date -Is)"
echo "Node:  $(hostname -s)"
echo "Job:   ${SLURM_JOB_ID}"
echo "Group: ${SLURM_ARRAY_TASK_ID}/${PYTEST_GROUP_COUNT}"
echo "CPUs:  ${SLURM_CPUS_PER_TASK}"
echo "Load:  $(uptime)"
echo

cd "$GATE_WORKTREE"
echo "Gate tree:         $GATE_WORKTREE"
echo "Commit under test: $(git rev-parse --short HEAD) $(git log -1 --format=%s)"
echo "Dirty files:       $(git status --porcelain | wc -l)"
echo

PYTEST_GROUP_ID=${SLURM_ARRAY_TASK_ID}
mapfile -t ALL_TEST_FILES < <(
    {
        find tests/unit tests/gui tests/integration tests/smoke \
            -type f -name 'test_*.py' -print
        # tests/migration is not in testpaths and is named file by file. Its
        # equivalence suite (test_equivalence.py) is excluded on purpose: its
        # goldens are keyed on pre-rename class names (GaussianBlur, GrayOpening,
        # BM3DDenoiser ...) and it fails with FileNotFoundError on this branch
        # before any change of ours. Including it would bury a real regression
        # under ~60 pre-existing failures.
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

# Do not use `-n auto`: xdist would see the node's core count rather than this
# allocation's, and manufacture timeout failures. Clear repository addopts to
# avoid uncaptured shared-filesystem log traffic, then restore the default
# `not slow` selection explicitly.
pytest_status=0
"$VENV_PYTHON" -m pytest "${TEST_TARGETS[@]}" -q --no-header \
    -p no:randomly -o addopts= -m "not slow" || pytest_status=$?

echo
echo "pytest group ${PYTEST_GROUP_ID} exit status: ${pytest_status}"
echo "Finished: $(date -Is)"
exit "${pytest_status}"
