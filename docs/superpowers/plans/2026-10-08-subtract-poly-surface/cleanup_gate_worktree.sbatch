#!/bin/bash
#SBATCH --job-name=sps-cleanup
#SBATCH --partition=short
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=/bigdata/exfab/anguy344/slurm_logs/%j.log
# afterany finalizer: remove the detached gate worktree once every gate job ended.
git -C /bigdata/exfab/anguy344/PhenoTypic/.claude/worktrees/subtract-poly-surface worktree remove --force /bigdata/exfab/anguy344/PhenoTypic/.worktrees/sps-gate-2bf26ea
