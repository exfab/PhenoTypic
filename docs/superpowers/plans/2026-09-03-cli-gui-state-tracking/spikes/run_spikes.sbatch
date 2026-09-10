#!/bin/bash
#SBATCH --job-name=phenotypic-spikes
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:45:00
#SBATCH --output=/bigdata/exfab/anguy344/slurm_logs/spikes_%j.out
#SBATCH --error=/bigdata/exfab/anguy344/slurm_logs/spikes_%j.err
#
# Phase 0 spike gate -- S-2 (shard sizing) and S-3 (merge cost).
# S-1 and S-4 are cut; S-5 moved to P1's phase gate. Neither is submitted here.
#
# Deliberate omissions, per the user's global CLAUDE.md:
#   * NO --account flag. `short` takes the default account, and an empty
#     --account= value is an invalid account that makes sbatch reject the whole
#     submission.
#   * --mem and --time are always set. DefMemPerCPU is 1 GB/CPU, which is the
#     usual cause of a silent OOM kill; S-3 needs >= 64 GB, so 96 GB is the
#     floor plus headroom to measure a true peak rather than get killed at one.
#   * --output lives on /bigdata. NEVER /scratch/<user>/<jobid> -- that is
#     node-local and per-job, and a job landing on another node fails with
#     ExitCode 0:53 and no log file at all.
#
# ORDER IS LOAD-BEARING: S-2 runs first, on a cold page cache.
# The two scripts read the same ~6.5k parquet tables, and a compute node's page
# cache holds all 527 MB of them, so whichever runs second runs warm. S-2's
# verdict is a seconds-per-image figure used to SIZE shards, and a real shard
# worker starts cold -- so S-2 gets the cold cache. S-3's verdict is peak RSS,
# which the page cache does not affect; its wall-clock numbers are warm-cache
# and RESULTS.md says so.

set -uo pipefail

WORKTREE=/bigdata/exfab/anguy344/PhenoTypic/.worktrees/cli-gui-state-tracking
SPIKES="${WORKTREE}/docs/superpowers/plans/2026-09-03-cli-gui-state-tracking/spikes"
FIXTURE=/bigdata/exfab/anguy344/projects/ucr_029_e_d_Maresca/data/results/2026-08-11

# Shard parquets go on GPFS, not node-local scratch: in production a later
# finalizer task reads them, possibly from another node, so GPFS is where they
# will really live and what the merge should really be timed against.
SCRATCH=/bigdata/exfab/anguy344/spike-scratch/s3_${SLURM_JOB_ID}

cd "${WORKTREE}" || exit 1
mkdir -p "${SCRATCH}"

echo "host=$(hostname -s) job=${SLURM_JOB_ID} start=$(date -Is)"
echo "fixture=${FIXTURE}"
df -T "${FIXTURE}" | tail -1
echo

echo "################ S-2 shard sizing (COLD cache) ################"
uv run python "${SPIKES}/s2_shard_sizing.py" "${FIXTURE}"
s2=$?
echo "s2_exit=${s2}"
echo

echo "################ S-3 merge cost (warm cache; RSS unaffected) ################"
uv run python "${SPIKES}/s3_merge_cost.py" "${FIXTURE}" "${SCRATCH}" 16
s3=$?
echo "s3_exit=${s3}"
echo

echo "shard scratch contents:"
ls -la "${SCRATCH}" | head -25
du -sh "${SCRATCH}"
rm -rf "${SCRATCH}"

echo "end=$(date -Is) s2_exit=${s2} s3_exit=${s3}"
exit $(( s2 != 0 || s3 != 0 ))
