#!/bin/bash
#SBATCH --job-name=sps-affected
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/bigdata/exfab/anguy344/slurm_logs/%j.log
set -euo pipefail
WORKTREE=/bigdata/exfab/anguy344/PhenoTypic/.worktrees/sps-gate-2bf26ea
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export QT_QPA_PLATFORM=offscreen
cd "$WORKTREE"
echo "Node: $(hostname -s) Job: ${SLURM_JOB_ID} CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Commit under test: $(git rev-parse --short HEAD)"
pytest_status=0
uv run python -m pytest tests/unit/enhance tests/unit/abc_ tests/unit/tune tests/unit/ci tests/gui/builder \
    -n "${SLURM_CPUS_PER_TASK}" -q --no-header -p no:randomly -o addopts= -m "not slow" || pytest_status=$?
echo "pytest exit status: ${pytest_status}"
exit "${pytest_status}"
