#!/bin/bash
#SBATCH --job-name=phenotypic-phase-gate
#SBATCH --partition=short
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/bigdata/exfab/anguy344/slurm_logs/%A_%a.log
set -uo pipefail

# Parameterised phase gate for the nested-GpuDetector plans. Derived from the
# committed full-suite runner at
# docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch
#
# Usage:
#   PHENO_GATE_PATHS="tests/unit/cli tests/integration/cli" \
#   sbatch --array=0-3%4 run_phase_gate.sbatch
#
#   PHENO_GATE_PATHS="tests/unit/tune" sbatch --array=0-0 run_phase_gate.sbatch
#
# --output MUST be shared storage and the directory MUST already exist.
# Pointing it at /scratch/<user>/<jobid> -- including a Claude scratchpad --
# gives FAILED, ExitCode 0:53 and NO LOG AT ALL, intermittently, because that
# path is node-local to a different job.

WORKTREE=/bigdata/exfab/anguy344/PhenoTypic/.claude/worktrees/nested-gpu-staging
: "${PHENO_GATE_PATHS:?set PHENO_GATE_PATHS to the test paths this gate covers}"
GROUP_COUNT=${SLURM_ARRAY_TASK_COUNT:-1}
GROUP_ID=${SLURM_ARRAY_TASK_ID:-0}

# Qt aborts the whole interpreter without this -- exit 134, no summary, no
# counts, several hundred lines of C dump naming nothing. tests/unit reaches the
# Qt tests around 79%, so a run that dies there looks like it "mostly worked".
export QT_QPA_PLATFORM=offscreen
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

cd "$WORKTREE"
echo "Date:   $(date -Is)"
echo "Node:   $(hostname -s)"
echo "Job:    ${SLURM_JOB_ID}"
echo "Group:  ${GROUP_ID}/${GROUP_COUNT}"
echo "CPUs:   ${SLURM_CPUS_PER_TASK}"
echo "Commit: $(git rev-parse --short HEAD) $(git log -1 --format=%s)"
echo "Paths:  ${PHENO_GATE_PATHS}"
echo

# Deterministic file sharding, so every task covers a disjoint slice.
mapfile -t ALL < <(find ${PHENO_GATE_PATHS} -type f -name 'test_*.py' | sort)
MINE=()
for i in "${!ALL[@]}"; do
    if (( i % GROUP_COUNT == GROUP_ID )); then MINE+=("${ALL[$i]}"); fi
done
echo "Files:  ${#MINE[@]} of ${#ALL[@]}"
if (( ${#MINE[@]} == 0 )); then echo "nothing to do"; exit 0; fi

# -o addopts= clears --capture=no (which streams every line to a shared-FS log
# and dominates the runtime) but ALSO drops -m 'not slow', so re-add it.
# Explicit -n: `-n auto` reads the NODE's core count, not the allocation's, and
# manufactures timeout failures that look like real bugs.
# No -x anywhere: it stops at the first failure, and tests/unit/cli sorts early,
# so an -x run that looks like a clean sweep may have covered a third of the set.
status=0
uv run pytest "${MINE[@]}" \
    -q --no-header -p no:randomly \
    -o addopts= -m "not slow" \
    -n "${SLURM_CPUS_PER_TASK:-1}" || status=$?

echo
echo "pytest exit: ${status}"
exit "${status}"
