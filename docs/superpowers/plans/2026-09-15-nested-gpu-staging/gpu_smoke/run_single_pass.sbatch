#!/bin/bash
#SBATCH --job-name=pht-smoke-single-pass
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --time=01:30:00
#SBATCH --output=/bigdata/exfab/anguy344/slurm_logs/gpusmoke-single-%j.log
set -uo pipefail

# Arm B. Pass the partition on the command line (and --account=exfab for
# exfab; --constraint=gpu_latest for short_gpu, which also holds P100s):
#
#   SMOKE_TREE=... SMOKE_INPUT=... SMOKE_REFERENCE=<fresh dir> \
#   sbatch -p exfab --account=exfab run_single_pass.sbatch
: "${SMOKE_TREE:?}" "${SMOKE_INPUT:?}" "${SMOKE_REFERENCE:?}"
PROJECT=/bigdata/exfab/anguy344/projects/ucr_033_e_d_Linzer_Ganoderma
PIPE=$PROJECT/config/F1gfd5.json.pht-pipe
HERE=$SMOKE_TREE/docs/superpowers/plans/2026-09-15-nested-gpu-staging/gpu_smoke

[[ -e $SMOKE_REFERENCE ]] && { echo "REFUSING: $SMOKE_REFERENCE exists"; exit 1; }
cd "$SMOKE_TREE" || exit 1
unset VIRTUAL_ENV
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "Node:   $(hostname -s)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Tree:   $SMOKE_TREE @ $(git rev-parse --short HEAD), dirty=$(git status --porcelain --untracked-files=no | wc -l)"
echo

status=0
.venv/bin/python "$HERE/single_pass_reference.py" "$PIPE" "$SMOKE_INPUT" "$SMOKE_REFERENCE" || status=$?
echo "exit: $status"
exit "$status"
