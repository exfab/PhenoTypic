#!/bin/bash
#SBATCH --job-name=scatter-tests
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-15%16
#SBATCH --output=/bigdata/exfab/anguy344/slurm_logs/%A_%a.log
set -euo pipefail

# Sharded pytest for the Scatter tab branch. Adapted from the proven
# run_unit_suite.sbatch, resized for this worktree and this account's headroom.
#
# SIZING: the iwheeldonlab cap is 384 CPU aggregate across all running jobs.
# 16 tasks x 8 CPUs = 128 peak, which leaves room for an interactive session
# to hold ~170 without pushing the array into AssocGrpCpuLimit. Raising either
# number without checking `squeue -u $USER -t RUNNING -h -o %C | paste -sd+ | bc`
# first is how an array ends up queued behind its own submitter.
#
# SCOPE is chosen by the SCOPE env var, defaulting to the GUI tree:
#   sbatch --export=ALL,SCOPE=gui  run_scatter_tests.sbatch   # per-cluster gate
#   sbatch --export=ALL,SCOPE=full run_scatter_tests.sbatch   # Task 15
WORKTREE=/bigdata/exfab/anguy344/PhenoTypic/.worktrees/results-scatter-gui
PYTEST_GROUP_COUNT=16
SCOPE=${SCOPE:-gui}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
# Without this the interpreter aborts partway through with no summary, which
# reads as a hang rather than a missing display.
export QT_QPA_PLATFORM=offscreen
# Point kaleido at the browser Playwright vendors for the e2e suite. Without
# it, every Chrome-guarded export test SKIPS and the array silently verifies
# less than a local run does -- which is how a gate reports green over tests
# that never executed. No Chrome is installed on these nodes; this is the
# only browser available, and it renders correctly (measured on epyc and
# intel alike).
_PW_CHROME=$(ls -d "$HOME"/.cache/ms-playwright/chromium-*/chrome-linux64/chrome 2>/dev/null | tail -1)
if [[ -n "${_PW_CHROME}" ]]; then
    export BROWSER_PATH="${_PW_CHROME}"
    echo "BROWSER_PATH: ${BROWSER_PATH}"
else
    echo "WARNING: no Playwright chromium found; export tests will SKIP" >&2
fi

echo "Date:  $(date -Is)"
echo "Node:  $(hostname -s)"
echo "Job:   ${SLURM_JOB_ID}"
echo "Group: ${SLURM_ARRAY_TASK_ID}/${PYTEST_GROUP_COUNT}   Scope: ${SCOPE}"
echo

cd "$WORKTREE"
echo "Commit under test: $(git rev-parse --short HEAD) $(git log -1 --format=%s)"
echo

case "$SCOPE" in
  gui)  SEARCH_DIRS=(tests/unit/gui tests/gui) ;;
  full) SEARCH_DIRS=(tests/unit tests/gui tests/integration tests/smoke) ;;
  *)    echo "unknown SCOPE=${SCOPE} (want gui|full)" >&2; exit 2 ;;
esac

PYTEST_GROUP_ID=${SLURM_ARRAY_TASK_ID}
mapfile -t ALL_TEST_FILES < <(
    find "${SEARCH_DIRS[@]}" -type f -name 'test_*.py' -print | LC_ALL=C sort
)

declare -a TEST_TARGETS=()
for file_index in "${!ALL_TEST_FILES[@]}"; do
    if ((file_index % PYTEST_GROUP_COUNT == PYTEST_GROUP_ID)); then
        TEST_TARGETS+=("${ALL_TEST_FILES[file_index]}")
    fi
done

if ((${#TEST_TARGETS[@]} == 0)); then
    echo "Pytest group ${PYTEST_GROUP_ID} has no test files (harmless on a narrow scope)"
    exit 0
fi

echo "Discovered files: ${#ALL_TEST_FILES[@]}"
echo "Group files:      ${#TEST_TARGETS[@]}"
printf '  %s\n' "${TEST_TARGETS[@]}"
echo

# Tests model ordinary jobs and mock SLURM_JOB_ID; do not leak this runner's
# array identity into their subject.
unset SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID SLURM_ARRAY_TASK_COUNT
unset SLURM_ARRAY_TASK_MIN SLURM_ARRAY_TASK_MAX SLURM_ARRAY_TASK_STEP

# Never `-n auto`: xdist reads the node's core count, not this allocation, and
# manufactures timeout failures. `-o addopts=` clears the repo's
# `--capture=no`, which triples runtime when stdout is a file on shared storage;
# `-m "not slow"` then restores the default selection explicitly.
pytest_status=0
uv run python -m pytest "${TEST_TARGETS[@]}" -q --no-header \
    -p no:randomly -o addopts= -m "not slow" || pytest_status=$?

echo
echo "pytest group ${PYTEST_GROUP_ID} exit status: ${pytest_status}"
echo "Finished: $(date -Is)"
exit "${pytest_status}"
