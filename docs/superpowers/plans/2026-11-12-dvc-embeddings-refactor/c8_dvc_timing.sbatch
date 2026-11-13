#!/bin/bash
#SBATCH --job-name=c8-dvc-timing
#SBATCH --partition=intel
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/bigdata/exfab/anguy344/slurm_logs/%x-%j.log
#
# P2 Task 6 Step 1 / spec C8: time `dvc add`, `dvc status` (cold, warm) and
# `dvc commit` on a bounded subset of the migrated maresca copy, in a scratch
# DVC repo on /bigdata (same filesystem as the source, so `cp -al` works).
#
# The migrated copy holds ONE dataset (7-24-26_redo_full, 6,657 stores, ~1.3 TB
# with hdf/ excluded), so "bounded subset" = every STEP-th store of that
# dataset's zarr/ directory (default ~300 stores). Extrapolate by store count.
# One config on one node: a cold-node number for the single job, not a
# two-node pair; repeat by resubmitting if the sign matters.
#
# SAFETY: the source is never modified. Scratch files are hard links to the
# source inodes, so the only file ever written is first replaced by a private
# copy (rename within the scratch dir) and asserted to have link count 1.
set -euo pipefail

SRC_RUN=/bigdata/exfab/anguy344/projects/ucr_029_e_d_Maresca/data/results/2026-08-11-ome-zarr/results
DATASET=7-24-26_redo_full
STEP=${STEP:-22}                       # every 22nd store -> ~303 of 6,657
SCRATCH_ROOT=/bigdata/exfab/anguy344/scratch
WORK="$SCRATCH_ROOT/c8-dvc-timing-${SLURM_JOB_ID:-manual}"
DVC=(uvx --from 'dvc>=3,<4' dvc)

case "$WORK" in "$SCRATCH_ROOT"/c8-dvc-timing-*) ;; *) echo "bad WORK=$WORK" >&2; exit 2;; esac

cleanup() {
    # Only ever removes the scratch repo (hard links: unlinks scratch names only).
    case "$WORK" in "$SCRATCH_ROOT"/c8-dvc-timing-*) chmod -R u+w "$WORK" 2>/dev/null || true; rm -rf -- "$WORK";; esac
}
trap cleanup EXIT

declare -A T
FAILED=""
timed() {   # timed <label> <cmd...>: records wall seconds in T[label]; never aborts
    local label=$1; shift
    local t0 t1 rc=0
    t0=$(date +%s.%N)
    "$@" || rc=$?
    t1=$(date +%s.%N)
    T[$label]=$(echo "$t1 - $t0" | bc -l)
    echo "[time] $label = ${T[$label]} s (rc=$rc)"
    if [[ $rc -ne 0 ]]; then FAILED="$FAILED $label"; fi
    return 0
}

echo "host=$(hostname -s) job=${SLURM_JOB_ID:-none} start=$(date -Is)"
mkdir -p "$WORK/out/results/$DATASET/zarr"
cd "$WORK"
git init -q .
"${DVC[@]}" --version
"${DVC[@]}" init -q

# Hard-linked subset of the dataset (hdf/ never included).
mapfile -t STORES < <(ls "$SRC_RUN/$DATASET/zarr" | awk -v s="$STEP" 'NR%s==1')
for s in "${STORES[@]}"; do
    cp -al "$SRC_RUN/$DATASET/zarr/$s" "out/results/$DATASET/zarr/$s"
done
FILES=$(find out/results -type f | wc -l)
BYTES=$(du -sb out/results | cut -f1)
TOTAL_STORES=$(ls "$SRC_RUN/$DATASET/zarr" | wc -l)
echo "subset: stores=${#STORES[@]} of $TOTAL_STORES files=$FILES bytes=$BYTES"

timed add "${DVC[@]}" add out/results
timed status_cold "${DVC[@]}" status
timed status_warm "${DVC[@]}" status

# Modify one small file in one store, only after proving it is private.
MODIFIED=no; SKIP_REASON=""
target=$(find "out/results/$DATASET/zarr/${STORES[0]}/tables" -type f -size -4k | head -1 || true)
if [[ -z "$target" ]]; then
    SKIP_REASON="no small file found in first store's tables/"
else
    cp -- "$target" "$target.c8tmp" && mv -f -- "$target.c8tmp" "$target"   # private copy; source inode untouched
    links=$(stat -c %h "$target")
    if [[ "$links" == "1" ]]; then
        printf 'c8' >> "$target"
        MODIFIED=yes
    else
        SKIP_REASON="link count $links != 1 for $target"
    fi
fi
echo "modify: $MODIFIED ${SKIP_REASON:+($SKIP_REASON)}"
if [[ $MODIFIED == yes ]]; then
    timed status_after_modify "${DVC[@]}" status
    timed commit "${DVC[@]}" commit --force out/results.dvc
fi

CACHE_BYTES=$(du -sb .dvc/cache | cut -f1)
echo "=== C8 SUMMARY ==="
echo "dataset=$DATASET"
echo "stores=${#STORES[@]}"
echo "total_stores=$TOTAL_STORES"
echo "files=$FILES"
echo "bytes=$BYTES"
for k in add status_cold status_warm status_after_modify commit; do
    echo "${k}_s=${T[$k]:-NA}"
done
echo "modified=$MODIFIED"
echo "skip_reason=$SKIP_REASON"
echo "cache_bytes=$CACHE_BYTES"
echo "failed_steps=${FAILED:-none}"
echo "=== END C8 SUMMARY ==="
[[ -z "$FAILED" ]]
