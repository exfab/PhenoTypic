#!/bin/bash
# Submit one sharded test gate for the size-measures-consolidation branch.
#
#   submit_phase_gate.sh <SHA-or-ref> <label> "<space-separated test roots>" [base]
#
# Runs the committed harness docs/superpowers/plans/2026-09-03-cli-gui-state-tracking/
# run_suite.sbatch (16 shards x 8 CPU on `short`) against ONE tree:
#   * head (default): a detached worktree at <SHA> under gate-worktrees/, created here,
#     uv-synced, provenance-checked, and removed by an afterany finalizer;
#   * base: the long-lived main worktree gate-worktrees/size-main (never removed here).
# Results: /bigdata/exfab/anguy344/slurm_logs/size-gate_<label>/shard_<i>.xml; compare
# with run_suite's sibling collect_results.py (--baseline <other results dir>).
set -euo pipefail

REPO=/bigdata/exfab/anguy344/PhenoTypic/.claude/worktrees/size-measures-consolidation
HARNESS=$REPO/docs/superpowers/plans/2026-09-03-cli-gui-state-tracking/run_suite.sbatch
GATES=/bigdata/exfab/anguy344/gate-worktrees
REF=${1:?SHA or ref}; LABEL=${2:?label}; SCOPE=${3:?test roots}; KIND=${4:-head}
RESULTS=/bigdata/exfab/anguy344/slurm_logs/size-gate_${LABEL}
SHARDS=16

[[ -e $RESULTS ]] && { echo "results dir exists, refusing to mix runs: $RESULTS" >&2; exit 2; }

if [[ $KIND == base ]]; then
    TREE=$GATES/size-main
else
    SHA=$(git -C "$REPO" rev-parse --verify "${REF}^{commit}")
    TREE=$GATES/size-${LABEL}-${SHA:0:8}
    [[ -e $TREE ]] && { echo "worktree exists: $TREE" >&2; exit 2; }
    git -C "$REPO" worktree add --detach "$TREE" "$SHA"
    (cd "$TREE" && uv sync --group dev --group test-qt --extra gui --extra napari -q)
fi

# Provenance: the package must import from THIS tree, or every shard measures another one.
where=$(cd "$TREE" && uv run python -c "import phenotypic, pathlib; print(pathlib.Path(phenotypic.__file__).resolve())")
[[ $where == "$TREE"/* ]] || { echo "PROVENANCE FAIL: phenotypic imports from $where, not $TREE" >&2; exit 3; }
echo "tree $TREE @ $(git -C "$TREE" rev-parse --short HEAD); phenotypic from $where"

jid=$(sbatch --parsable --array=0-$((SHARDS - 1))%${SHARDS} --job-name="size-gate-${LABEL}" \
      --export=ALL,WORKTREE="$TREE",SCOPE="$SCOPE",SHARDS=$SHARDS,RESULTS_DIR="$RESULTS" \
      "$HARNESS" 2>&1)
[[ $jid =~ ^[0-9]+$ ]] || { echo "SUBMIT FAILED: $jid" >&2; exit 1; }
echo "array $jid -> $RESULTS"

deps=$jid
if [[ ${DOCS:-0} == 1 ]]; then
    djid=$(sbatch --parsable --export=ALL,WORKTREE="$TREE" \
           "$REPO/docs/superpowers/plans/2026-09-24-size-measures-consolidation/build_docs_size_note.sbatch" 2>&1)
    [[ $djid =~ ^[0-9]+$ ]] || { echo "DOCS SUBMIT FAILED: $djid" >&2; exit 1; }
    echo "docs build $djid -> $TREE/docs/_build/size-note"
    deps="$jid:$djid"
fi

if [[ $KIND != base ]]; then
    # Keep the tree when the docs were built in it, so the HTML can be read afterwards.
    [[ ${DOCS:-0} == 1 ]] && { echo "tree kept for docs inspection; remove it by hand: git worktree remove --force $TREE"; exit 0; }
    fin=$(sbatch --parsable --dependency=afterany:"$deps" --partition=short --time=00:10:00 \
          --mem=2G --job-name="size-gate-${LABEL}-cleanup" \
          --output=/bigdata/exfab/anguy344/slurm_logs/size-gate_${LABEL}_cleanup_%j.log \
          --wrap="git -C '$REPO' worktree remove --force '$TREE'; git -C '$REPO' worktree prune" 2>&1)
    [[ $fin =~ ^[0-9]+$ ]] || { echo "FINALIZER SUBMIT FAILED: $fin" >&2; exit 1; }
    echo "cleanup finalizer $fin (afterany:$jid)"
fi
