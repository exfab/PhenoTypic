#!/bin/bash
#SBATCH --job-name=pheno-coldstart
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:40:00
#SBATCH --output=/bigdata/exfab/anguy344/slurm_logs/coldstart_%j.out
#SBATCH --error=/bigdata/exfab/anguy344/slurm_logs/coldstart_%j.err
set -uo pipefail

# Which cold start is cheapest: drop the overlay from the proof, or ship the
# on-disk cache tier?
#
# WHY A SLURM JOB. The question is about a COLD GPFS client cache, and the
# interactive node's cache is now fully warm -- every earlier measurement in
# this session touched the whole tree, so a repeat there reads 0.04 ms/stat and
# answers a different question. The client cache is PER NODE, so a fresh
# compute node is cold by construction. That is the only instrument available:
# the caches cannot be dropped without root, and P0 established there is no
# second fixture tree of this size.
#
# Each lane gets a DISJOINT seeded half so neither warms the other, which is the
# fix P0's S-2 spike needed for the same reason.

WT=/bigdata/exfab/anguy344/PhenoTypic/.worktrees/cli-gui-state-tracking
S="$WT/docs/superpowers/plans/2026-09-03-cli-gui-state-tracking/spikes/cold_start_options.py"

echo "node=$(hostname -s) job=${SLURM_JOB_ID} start=$(date -Is)"
cd "$WT" || exit 1

# C first on its half, then B on the other half. Order does not matter here
# because the halves are disjoint, but record it anyway.
uv run python "$S" C first
uv run python "$S" B second

echo "end=$(date -Is)"
