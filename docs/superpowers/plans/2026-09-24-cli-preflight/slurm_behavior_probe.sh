#!/bin/bash
# Plan Task 13 Step 1: confirm the Slurm behaviors the run preflight relies on.
# Run on a login node of the target cluster; submits NO job (--test-only only).
# Usage: bash slurm_behavior_probe.sh <cpu-partition> [<gpu-partition>] [<drained-node>]
# Record the output in docs/superpowers/reports/2026-09-24-cli-preflight/slurm-behavior.md.
set -u
CPU_PARTITION=${1:?cpu partition}
GPU_PARTITION=${2:-}
DRAINED_NODE=${3:-}

echo "== EnforcePartLimits"
scontrol show config | grep -i EnforcePartLimits

script() {  # $@ = extra #SBATCH lines
    printf '#!/bin/bash\n'
    for line in "$@"; do printf '#SBATCH %s\n' "$line"; done
    printf 'true\n'
}

echo "== 1. valid request (expect exit 0 and 'to start at')"
script "--partition=$CPU_PARTITION" "--time=00:10:00" | sbatch --test-only; echo "exit=$?"

echo "== 2. time far above MaxTime (is it rejected, or accepted to pend?)"
script "--partition=$CPU_PARTITION" "--time=99-00:00:00" | sbatch --test-only; echo "exit=$?"

echo "== 3. unknown partition"
script "--partition=does-not-exist" "--time=00:10:00" | sbatch --test-only; echo "exit=$?"

echo "== 4. GPU request on a CPU partition"
script "--partition=$CPU_PARTITION" "--gpus-per-node=1" "--time=00:10:00" | sbatch --test-only; echo "exit=$?"

echo "== 5. misspelled option"
script "--partition=$CPU_PARTITION" "--tiem=00:10:00" | sbatch --test-only; echo "exit=$?"

echo "== 6. sinfo on an unknown partition (exit status and stdout length; review E6)"
out=$(sinfo -p does-not-exist -h -o %G 2>/dev/null); echo "exit=$? stdout_length=${#out} stdout=[$out]"

echo "== 7. partition MaxTime and Default"
scontrol show partition "$CPU_PARTITION" | tr ' ' '\n' | grep -E '^(MaxTime|Default)='
if [ -n "$GPU_PARTITION" ]; then
    echo "== 8. GPU partition gres, untruncated (-o %G) vs --Format=gres (20 chars; review E5)"
    sinfo -p "$GPU_PARTITION" -h -o %G; echo "exit=$?"
    sinfo -p "$GPU_PARTITION" --Format=gres --noheader; echo "exit=$?"
fi
if [ -n "$DRAINED_NODE" ]; then
    echo "== 9. a job only a DRAINED node can run (review E1: expect --test-only to fail"
    echo "      with 'Requested node configuration is not available')"
    script "--partition=$CPU_PARTITION" "--nodelist=$DRAINED_NODE" "--time=00:10:00" | sbatch --test-only; echo "exit=$?"
fi
