#!/bin/bash
# #282 P2 Task 4: submit the GPU batch-invariance jobs from a detached worktree.
#
#   submit_batch_invariance.sh <commit> [gpu-type ...]     (default: ada6000 a100 blackwell6000)
#   submit_batch_invariance.sh <commit> --smoke <gpu-type>  (one image, every row, no repeat)
#
# Per GPU type: a measurement job, then a separate-process repeat of its
# reference (`--repeat-of`, `afterany`: it fails on its own when the first
# left no result, and an `afterok` dependency would pend forever here, since
# the cluster does not set kill_invalid_depend). Then one `afterany` job that
# writes the cross-GPU table and removes the worktree. Each job's partition is
# the one with the earlier predicted start (`sbatch --test-only`) among those
# with that GPU: ada6000 is on `exfab` and `short_gpu`; a100 and blackwell6000
# on `short_gpu` only. Never `preempt`. Every captured id is checked.
set -euo pipefail
REPO=/bigdata/exfab/anguy344/PhenoTypic
SHA=$(git -C "$REPO" rev-parse --verify "${1:?commit}^{commit}")
shift
SHORT=${SHA:0:9}
D=/bigdata/exfab/anguy344/gate-results/c6-282
TREE=/bigdata/exfab/anguy344/gate-worktrees/c6-$SHORT
PLAN=docs/superpowers/plans/2026-11-12-gpu-embedding-staging
PY=/bigdata/exfab/anguy344/gate-results/c18/venv-foundation/bin/python
SMOKE=0
if [[ ${1:-} == --smoke ]]; then SMOKE=1; shift; fi
GPUS=("$@")
(( ${#GPUS[@]} )) || GPUS=(ada6000 a100 blackwell6000)

for size in base large giant; do
  compgen -G "$HOME/.cache/huggingface/hub/models--facebook--dinov2-$size/snapshots/*/model.safetensors" >/dev/null \
    || { echo "dinov2-$size weights are not cached; pre-stage them first"; exit 1; }
done
mkdir -p "$D"
[ -d "$TREE" ] || git -C "$REPO" worktree add --detach "$TREE" "$SHA" >/dev/null
[[ $(git -C "$TREE" rev-parse HEAD) == "$SHA" ]] || { echo "$TREE is not at $SHA"; exit 1; }

# flags_for <partition> <gpu-type>: the sbatch flags for one GPU on that partition.
flags_for() {
  local flags=(--partition="$1" --gres="gpu:$2:1")
  [[ $1 == exfab ]] && flags+=(--account=exfab)
  [[ $1 == short_gpu ]] && flags+=(--constraint=gpu_latest)
  echo "${flags[@]}"
}

# pick_partition <gpu-type> <time>: the partition with the earlier predicted start.
pick_partition() {
  local gpu=$1 time=$2 best="" best_start="" p start
  local parts=(short_gpu)
  [[ $gpu == ada6000 ]] && parts=(exfab short_gpu)
  for p in "${parts[@]}"; do
    # The controller intermittently answers "Requested node configuration is
    # not available" or "Zero Bytes were transmitted" to a request it accepts
    # a moment later, so ask up to five times, ten seconds apart.
    start=""
    for attempt in 1 2 3 4 5; do
      (( attempt > 1 )) && sleep 10
      # shellcheck disable=SC2046
      start=$(sbatch --test-only $(flags_for "$p" "$gpu") --cpus-per-task=8 --mem=64G \
                --time="$time" --wrap=true 2>&1 | grep -o 'to start at [0-9T:-]*' | awk '{print $4}') || true
      [[ -n $start ]] && break
    done
    echo "  $gpu on $p: predicted start ${start:-none}" >&2
    if [[ -n $start && ( -z $best_start || $start < $best_start ) ]]; then best=$p; best_start=$start; fi
  done
  [[ -n $best ]] || { echo "no partition can run $gpu" >&2; return 1; }
  echo "$best"
}

# submit <name> <partition> <gpu> <time> <dependency or ""> <args...>: prints the job id.
submit() {
  local name=$1 part=$2 gpu=$3 time=$4 dep=$5 out id
  shift 5
  local depflag=()
  [[ -n $dep ]] && depflag=(--dependency="$dep")
  # shellcheck disable=SC2046
  out=$(sbatch --parsable $(flags_for "$part" "$gpu") --time="$time" "${depflag[@]}" \
          --job-name="$name" --output="$D/%x-%j.log" \
          "$TREE/$PLAN/batch_invariance.sbatch" "$TREE" "$@" 2>&1) || true
  id=${out%%;*}
  [[ $id =~ ^[0-9]+$ ]] || { echo "submit $name failed: $out" >&2; return 1; }
  echo "$id"
}

ids=()
runs=()
for gpu in "${GPUS[@]}"; do
  if (( SMOKE )); then
    part=$(pick_partition "$gpu" 00:45:00)
    id=$(submit "c6-smoke-$gpu" "$part" "$gpu" 00:45:00 "" "$D/smoke-$gpu.json" --images 1)
    echo "smoke $gpu: job $id on $part"
    ids+=("$id")
    continue
  fi
  part=$(pick_partition "$gpu" 01:30:00)
  first=$(submit "c6-$gpu" "$part" "$gpu" 01:30:00 "" "$D/$gpu.json")
  part2=$(pick_partition "$gpu" 00:45:00)
  repeat=$(submit "c6-$gpu-repeat" "$part2" "$gpu" 00:45:00 "afterany:$first" \
             "$D/$gpu-repeat.json" --repeat-of "$D/$gpu.json")
  echo "$gpu: measure $first on $part, repeat $repeat on $part2"
  ids+=("$first" "$repeat")
  runs+=("$D/$gpu.json")
done

dep=$(IFS=:; echo "afterany:${ids[*]}")
compare_cmd=true
(( SMOKE )) || compare_cmd="$PY $TREE/$PLAN/batch_invariance.py --compare $D/cross_gpu.json ${runs[*]}"
out=$(sbatch --parsable --partition=short --cpus-per-task=2 --mem=8G --time=00:20:00 \
        --dependency="$dep" --job-name=c6-finalize --output="$D/%x-%j.log" \
        --wrap "$compare_cmd; echo compare exit \$?; git -C $REPO worktree remove --force $TREE; git -C $REPO worktree prune; echo cleanup exit \$?" 2>&1) || true
clean=${out%%;*}
[[ $clean =~ ^[0-9]+$ ]] || { echo "submit finalize failed: $out" >&2; exit 1; }
echo "finalize (compare + worktree cleanup): $clean"
echo "tree $TREE at $SHORT; logs $D"
