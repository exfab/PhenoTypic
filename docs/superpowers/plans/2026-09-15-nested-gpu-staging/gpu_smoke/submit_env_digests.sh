#!/bin/bash
# Fan the reproducibility digests out across node types and processes.
#   submit_env_digests.sh <frozen-tree> <fixed-input.ome.zarr> <out-dir>
set -euo pipefail
TREE=${1:?}; FIXED=${2:?}; OUT=${3:?}
PROJECT=/bigdata/exfab/anguy344/projects/ucr_033_e_d_Linzer_Ganoderma
PIPE=$PROJECT/config/F1gfd5.json.pht-pipe
TIFF=$PROJECT/data/pipeline_dev/inputs/nested_staging_smoke_2026-09-16/Full_experiment_2026-05-22/d000415_300_028_2026-05-24_14-08-31.tiff
SCRIPT=$(dirname "$(readlink -f "$0")")/diag_env_digests.py
LOGS=/bigdata/exfab/anguy344/slurm_logs
mkdir -p "$OUT"

submit() {  # name, then sbatch args..., then -- command
    local name=$1; shift
    local id
    id=$(sbatch --parsable --job-name="digest-$name" --chdir="$TREE" \
        --output="$LOGS/gpusmoke-digest-$name-%j.log" "$@")
    [[ $id =~ ^[0-9]+$ ]] || { echo "submit failed for $name: $id" >&2; exit 1; }
    echo "$name $id"
}

for spec in cascade:4 cascade:8 abu_dhabi:4 abu_dhabi:8 rome:4; do
    feature=${spec%%:*}; cpus=${spec##*:}
    submit "cpu-$feature-$cpus" -p short --constraint="$feature" \
        --cpus-per-task="$cpus" --mem=32G --time=1:00:00 \
        --wrap="OMP_NUM_THREADS=$cpus QT_QPA_PLATFORM=offscreen .venv/bin/python $SCRIPT cpu $PIPE $TIFF"
done
for run in 1 2; do
    submit "gpu-run$run" -p exfab --account=exfab --gpus-per-node=1 \
        --cpus-per-task=8 --mem=48G --time=0:45:00 \
        --wrap="HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 QT_QPA_PLATFORM=offscreen .venv/bin/python $SCRIPT gpu $PIPE $FIXED $OUT/raw_run$run.npy"
done
