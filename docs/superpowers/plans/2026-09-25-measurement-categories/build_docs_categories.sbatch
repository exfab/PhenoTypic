#!/bin/bash
#SBATCH --job-name=cat-docs
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/bigdata/exfab/anguy344/slurm_logs/cat-docs_%j.log
# Build the Sphinx docs for the measurement-categories change (plan Task 9).
# Notebooks are NOT executed (-D nbsphinx_execute=never). Pass WORKTREE=<tree> via --export;
# output goes to <tree>/docs/_build/categories (gitignored). Read the HTML afterwards: an exit
# code of 0 is not evidence that the Categories page, badges or navbar entry rendered.
set -euo pipefail
WORKTREE=${WORKTREE:?set WORKTREE}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
echo "Date: $(date -Is)  Node: $(hostname -s)  Job: ${SLURM_JOB_ID}  Tree: ${WORKTREE}"
cd "$WORKTREE"
echo "Commit: $(git rev-parse --short HEAD)"
status=0
PHENOTYPIC_DOCS_BUILD=1 uv run --group docs sphinx-build -b html \
    -j "$SLURM_CPUS_PER_TASK" -D nbsphinx_execute=never \
    docs/source docs/_build/categories || status=$?
echo "sphinx-build exit: ${status}"
exit "$status"
