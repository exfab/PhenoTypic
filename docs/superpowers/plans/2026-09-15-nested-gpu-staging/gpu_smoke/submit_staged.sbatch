#!/bin/bash
#SBATCH --job-name=pht-smoke-staged-submit
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/bigdata/exfab/anguy344/slurm_logs/gpusmoke-staged-submit-%j.log
set -uo pipefail

# Arm A: the production path. Submits the staged engine through its own
# controller (Stage 1 CPU array -> Stage 2 GPU -> Stage 3 CPU array ->
# finalizer) and returns; this job does not wait for the stages.
#
#   SMOKE_TREE=<frozen checkout> SMOKE_INPUT=<root> SMOKE_OUTPUT=<fresh dir> \
#   SMOKE_GPU_PARTITION=exfab|short_gpu  sbatch submit_staged.sbatch
#
# Workers run `sys.executable`, i.e. SMOKE_TREE's venv, so every stage runs the
# code under test. The check after submission proves that from the generated
# scripts rather than assuming it.
: "${SMOKE_TREE:?}" "${SMOKE_INPUT:?}" "${SMOKE_OUTPUT:?}" "${SMOKE_GPU_PARTITION:?}"
PROJECT=/bigdata/exfab/anguy344/projects/ucr_033_e_d_Linzer_Ganoderma
PIPE=$PROJECT/config/F1gfd5.json.pht-pipe

[[ -e $SMOKE_OUTPUT ]] && { echo "REFUSING: $SMOKE_OUTPUT exists"; exit 1; }

gpu=(--gpu-slurm "slurm_partition=$SMOKE_GPU_PARTITION"
     --gpu-slurm "slurm_constraint=gpu_latest"
     --gpu-slurm "mem_gb=48"
     --gpu-slurm "slurm_cpus_per_task=8"
     --gpu-slurm "slurm_gpus_per_node=1"
     --gpu-slurm "slurm_time=90")
[[ $SMOKE_GPU_PARTITION == exfab ]] && gpu+=(--gpu-slurm "slurm_account=exfab")

cd "$SMOKE_TREE" || exit 1
unset VIRTUAL_ENV
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "Tree:   $SMOKE_TREE @ $(git rev-parse --short HEAD), dirty=$(git status --porcelain --untracked-files=no | wc -l)"
echo "Python: $(.venv/bin/python -c 'import phenotypic,sys; print(sys.executable, phenotypic.__file__)')"
echo "Routes: $(.venv/bin/python - "$PIPE" <<'PY'
import sys
from pathlib import Path
from phenotypic._cli._cli_validation import find_gpu_detectors
from phenotypic import ImagePipeline
print([p for p, _ in find_gpu_detectors(ImagePipeline.from_json(Path(sys.argv[1])))])
PY
)"
echo

status=0
.venv/bin/python -m phenotypic \
    -i "$SMOKE_INPUT" -p "$PIPE" -o "$SMOKE_OUTPUT" \
    --image-type GridImage --nrows 8 --ncols 12 --bit-depth 16 \
    --slurm "slurm_partition=short" \
    --slurm "mem_gb=16" \
    --slurm "slurm_cpus_per_task=4" \
    --slurm "slurm_time=60" \
    "${gpu[@]}" \
    --gpu-shards 1 || status=$?
echo "cli exit: $status"

echo
echo "--- interpreters named by the generated scripts (must be $SMOKE_TREE/.venv):"
grep -rhoE '/[^ "]*/bin/python[0-9.]*' "$SMOKE_OUTPUT/.phenotypic/slurm_scripts" 2>/dev/null | sort | uniq -c
exit "$status"
