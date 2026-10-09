#!/bin/bash
#SBATCH --job-name=sps-triage
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=/bigdata/exfab/anguy344/slurm_logs/%A_%a.log
# Task 0: merge-base 8159b8819, gate env (no tune/topology extras).
# Task 1: branch 2bf26eaf3, env synced with --extra tune --extra topology.
set -uo pipefail
export QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
W=/bigdata/exfab/anguy344/PhenoTypic/.worktrees
if [[ $SLURM_ARRAY_TASK_ID == 0 ]]; then T=$W/sps-base-8159b8819; else T=$W/sps-triage-2bf26ea; fi
cd "$T"
echo "tree: $T  commit: $(git rev-parse --short HEAD)"
unset SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID
uv run python -m pytest tests/unit/tune/test_distributed_finalize_task2.py tests/unit/tune/test_distributed_lifecycle_task2.py tests/unit/tune/test_engine.py tests/unit/tune/test_journal_backend_task1.py -q --no-header -p no:randomly -o addopts= -m "not slow" -n 4 2>&1 | tail -40
uv run python -m pytest tests/smoke/test_operation.py -k FilFinderDetector -q --no-header -p no:randomly -o addopts= -m "not slow" 2>&1 | tail -15
