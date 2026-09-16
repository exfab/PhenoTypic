#!/bin/bash
# Build a frozen checkout for a phase gate, and prove it is frozen.
#
#   ./make_gate_tree.sh <sha-ish> [purpose]     # defaults: HEAD, "gate"
#
# Prints the tree path on success. Submit the array against it with:
#
#   PHENO_GATE_TREE=<path> PHENO_GATE_PATHS="tests/unit/cli" \
#     sbatch --array=0-3%4 run_phase_gate.sbatch
#
# ONE TREE PER MEASUREMENT, and that is what `purpose` is for. A gate array and
# a mutation harness pointed at the SAME frozen tree collide exactly as badly as
# either colliding with the live worktree: the harness rewrites a source file,
# the array reads it mid-mutation, and the array reports a failure that belongs
# to the harness. That happened here -- a 16-shard gate reported one novel
# failure which was the harness's M1 mutant, and the whole array had to be
# cancelled.
#
# "Frozen" is a property of a tree WITH RESPECT TO ITS READERS, not an intrinsic
# one. A detached checkout that something is actively writing to is not frozen;
# it is just not the live worktree. Passing a distinct `purpose` gives each
# measurement its own directory, which makes the collision impossible rather
# than merely forbidden -- the same reason the checkout is detached in the first
# place.
#
#   TREE=$(./make_gate_tree.sh HEAD gate)        # for the array
#   TREE=$(./make_gate_tree.sh HEAD mutation)    # for a mutation harness
#
# WHY THIS EXISTS. A parallel gate measures ONE tree. If a file changes while
# an array's shards are spread across nodes, the result is a union across two
# trees that no single tree ever produced -- void, not stale, and nothing in
# the output says so. Running the array against a checkout detached at a commit
# makes that impossible rather than merely forbidden, and lets everyone keep
# working in the live worktree meanwhile.
#
# Three things this script checks that a human reliably forgets:
#   1. the tree is clean (a dirty gate tree is not a commit)
#   2. `import phenotypic` resolves to THIS tree, not the live worktree -- the
#      editable install is a bare .pth path entry, so a mis-synced tree silently
#      imports someone else's source and every number is attributed wrongly
#   3. the venv exists at all, before 24 array tasks discover it in parallel
set -uo pipefail

SRC=/bigdata/exfab/anguy344/PhenoTypic/.claude/worktrees/nested-gpu-staging
BASE=/bigdata/exfab/anguy344/gate-trees
REF=${1:-HEAD}
PURPOSE=${2:-gate}

SHA=$(git -C "$SRC" rev-parse --short "$REF") || exit 1
TREE="$BASE/$SHA-$PURPOSE"

if [[ ! -d $TREE ]]; then
    git -C "$SRC" worktree add --detach "$TREE" "$SHA" >/dev/null 2>&1 || exit 1
fi

cd "$TREE" || exit 1

DIRTY=$(git status --porcelain | wc -l)
if (( DIRTY != 0 )); then
    echo "REFUSING: gate tree $TREE has $DIRTY modified file(s); it is not a commit" >&2
    exit 1
fi

# `--extra napari` is NOT optional for a gate. Without it, 14 tests across
# tests/unit/core/test_napari_pipeline_viewer.py and
# tests/unit/sdk_/test_label_editor_widget.py fail at import, and a gate that
# reports 14 reds it cannot explain trains its reader to skim red shards.
# Measured: with the extra installed those same 39 tests pass in 4.2s.
#
# It also happens to be the ONLY coverage for `_operation_tree`'s decision to
# key on ImagePipelineCore rather than ImagePipeline -- `NapariPipelineViewer`
# is the second concrete subclass and the entire reason that keying exists.
# Leaving the extra out made the one deviation with nothing else behind it
# invisible to the gate as well.
if [[ ! -x .venv/bin/python ]]; then
    uv sync --group dev --group test-qt --extra gui --extra napari \
        >/dev/null 2>&1 || exit 1
fi

RESOLVED=$(uv run python -c 'import phenotypic; print(phenotypic.__file__)' 2>/dev/null | tail -1)
case "$RESOLVED" in
    "$TREE"/src/phenotypic/__init__.py) ;;
    *)
        echo "REFUSING: $TREE imports phenotypic from $RESOLVED, not from itself" >&2
        exit 1
        ;;
esac

echo "$TREE"
