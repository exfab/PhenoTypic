#!/bin/bash
#SBATCH --job-name=size-final-checks
#SBATCH --partition=epyc
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/bigdata/exfab/anguy344/slurm_logs/size-final-checks_%j.log
# End-of-implementation checks not covered by the sharded suite (plan Task 10, amendment A9):
#   1. the swept e2e Playwright files (tests/e2e is outside testpaths and needs PLAYWRIGHT=1);
#   2. full-package mypy, whose error count is compared with main's by the orchestrator.
# Pass WORKTREE=<tree at the commit under test> via --export.
set -uo pipefail
WORKTREE=${WORKTREE:?set WORKTREE}
export QT_QPA_PLATFORM=offscreen PLAYWRIGHT=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd "$WORKTREE"
echo "Date: $(date -Is)  Node: $(hostname -s)  Commit: $(git rev-parse --short HEAD)"
echo "=== e2e"
uv run pytest tests/e2e/gui/test_scatter_tab.py tests/e2e/gui/test_analysis_app.py \
    -o addopts= -m "not slow" -q -p no:randomly -rfE
echo "e2e exit: $?"
echo "=== mypy"
uv run mypy src/phenotypic | tail -3
echo "mypy exit: ${PIPESTATUS[0]}"
