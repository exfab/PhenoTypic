#!/bin/bash
#SBATCH --job-name=c6-binv
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
# #282 P2 Task 4: one batch-invariance measurement on one GPU.
# Usage (submit_batch_invariance.sh passes partition, account, GRES, output):
#   sbatch ... batch_invariance.sbatch <detached tree> <out.json> [batch_invariance.py args]
# Runs the tree's own batch_invariance.py in the foundation venv, offline,
# with the deterministic cuBLAS workspace set before the process touches CUDA.
set -uo pipefail
TREE=${1:?detached worktree}
OUT=${2:?output json}
shift 2
PY=/bigdata/exfab/anguy344/gate-results/c18/venv-foundation/bin/python
export PYTHONPATH="$TREE/src:$TREE" QT_QPA_PLATFORM=offscreen
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
cd "$TREE" || exit 3
echo "commit $(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l) node $(hostname -s) partition ${SLURM_JOB_PARTITION:-} start $(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
"$PY" -c 'import phenotypic, torch; print("phenotypic from", phenotypic.__file__, "torch", torch.__version__, torch.version.cuda)'
"$PY" "$TREE/docs/superpowers/plans/2026-11-12-gpu-embedding-staging/batch_invariance.py" "$OUT" "$@"
status=$?
echo "exit $status end $(date -Is) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
exit $status
